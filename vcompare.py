#!/usr/bin/env python3
"""
vcompare.py — 三主流抽帧算法对比（同一时间范围）

算法：
  uniform     — 均匀间隔采样      (llm-frames / john-ver 思路)
  scene       — ffmpeg scene 检测 (claude-video / bradautomates 思路)
  scene-dedup — scene + dHash 去重 (peepshow 思路)

输出格式和 videoframe 一致：
  · 4 列 contact_sheet
  · 帧右下角水印 `{t}s [{algo}] #{idx}`

用法：
  python vcompare.py --in=kards.mp4 --out=cmp/ --range=0:30 --max-frames=15
  python vcompare.py --in=kards.mp4 --out=cmp/ --range=0:30 --algo=scene
"""
import sys, os, re, json, subprocess, time, shutil
from pathlib import Path

import numpy as np
from PIL import Image

import videoframe as VF




def _display_size(info):
    """返回显示尺寸 (DW, DH)。rotation 90/270 时交换宽高。"""
    rot = info.get('rotation', 0) % 360
    if rot in (90, 270):
        return info['h'], info['w'], rot
    return info['w'], info['h'], rot

def _extract_and_watermark(video_path, t, idx, out_dir, algo, info):
    """抽一帧 + 加水印，返回 {'t', 'reason', 'path'} 或 None。"""
    p = out_dir / f'f_{idx:03d}_{t:08.2f}.jpg'
    DW, DH, rot = _display_size(info)
    if not VF.extract_frame_at(video_path, t, p,
                                src_w=DW, src_h=DH,
                                max_w=1280, max_h=760,
                                rotation=rot):
        return None
    VF.add_watermark(str(p), f"{t:.2f}s  [{algo}]  #{idx:03d}",
                     style='bar')
    return {'t': round(t, 3), 'reason': algo, 'path': str(p)}


# ══════════════════════════════════════════════════════
# uniform — 均匀间隔（llm-frames 思路）
# ══════════════════════════════════════════════════════
def algo_uniform(video_path, out_dir, info, t_start, t_end, max_frames):
    t0 = time.time()
    dur = t_end - t_start
    # N 帧 → 间隔 dur/N（避免首尾偏移）
    step = dur / max_frames
    ts = [t_start + (i + 0.5) * step for i in range(max_frames)]
    ts = [t for t in ts if t < t_end - 1e-3]

    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, t in enumerate(ts):
        c = _extract_and_watermark(video_path, t, i + 1, frames_dir,
                                    'uniform', info)
        if c: frames.append(c)

    return {'frames': frames, 'elapsed': time.time() - t0}


# ══════════════════════════════════════════════════════
# scene — ffmpeg scene detection（claude-video 思路）
# ══════════════════════════════════════════════════════
def algo_scene(video_path, out_dir, info, t_start, t_end,
                max_frames, threshold=0.3):
    t0 = time.time()
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ffmpeg scene detection
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'info',
           '-ss', f'{t_start:.3f}',
           '-t', f'{t_end - t_start:.3f}',
           '-i', str(video_path),
           '-vf', f"select='gt(scene,{threshold})',showinfo",
           '-f', 'null', '-']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    ts = []
    for line in r.stderr.splitlines():
        m = re.search(r'pts_time:([0-9.]+)', line)
        if m:
            try:
                # -ss 后的 pts 是相对时间，加 t_start 变全局
                ts.append(t_start + float(m.group(1)))
            except ValueError:
                pass
    ts = sorted(set(round(t, 2) for t in ts))
    raw_n = len(ts)

    # 限制 max_frames（均匀取）
    if len(ts) > max_frames:
        step = len(ts) / max_frames
        ts = [ts[int(i * step)] for i in range(max_frames)]

    frames = []
    for i, t in enumerate(ts):
        c = _extract_and_watermark(video_path, t, i + 1, frames_dir,
                                    'scene', info)
        if c: frames.append(c)

    return {'frames': frames, 'elapsed': time.time() - t0,
            'raw_scene_count': raw_n}


# ══════════════════════════════════════════════════════
# scene-dedup — scene + dHash 去重（peepshow 思路）
# ══════════════════════════════════════════════════════
def _dhash(img_path, hash_size=16):
    img = Image.open(img_path).convert('L')
    img = img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.int16)
    return (arr[:, 1:] > arr[:, :-1]).flatten()


def algo_scene_dedup(video_path, out_dir, info, t_start, t_end,
                      max_frames, threshold=0.3, hamming=5):
    t0 = time.time()
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 先跑 scene（子目录，不入水印）
    sub_dir = out_dir / '_raw'
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_frames_dir = sub_dir / 'frames'
    sub_frames_dir.mkdir(parents=True, exist_ok=True)

    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'info',
           '-ss', f'{t_start:.3f}',
           '-t', f'{t_end - t_start:.3f}',
           '-i', str(video_path),
           '-vf', f"select='gt(scene,{threshold})',showinfo",
           '-f', 'null', '-']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    ts = []
    for line in r.stderr.splitlines():
        m = re.search(r'pts_time:([0-9.]+)', line)
        if m:
            try:
                ts.append(t_start + float(m.group(1)))
            except ValueError:
                pass
    ts = sorted(set(round(t, 2) for t in ts))
    raw_n = len(ts)

    # 抽全部，然后 dHash 去重
    DW, DH, rot = _display_size(info)
    all_frames = []
    for i, t in enumerate(ts):
        p = sub_frames_dir / f'f_{i:04d}.jpg'
        if VF.extract_frame_at(video_path, t, p,
                                src_w=DW, src_h=DH,
                                max_w=1280, max_h=760,
                                rotation=rot):
            all_frames.append({'t': t, 'path': str(p)})

    # dHash 去重
    kept = []
    last_h = None
    for c in all_frames:
        try:
            h = _dhash(c['path'])
            if last_h is not None and int(np.sum(h != last_h)) <= hamming:
                continue
            kept.append(c)
            last_h = h
        except Exception:
            continue

    # 限制 max_frames
    if len(kept) > max_frames:
        step = len(kept) / max_frames
        kept = [kept[int(i * step)] for i in range(max_frames)]

    # 移到正式目录 + 加水印
    frames = []
    for i, c in enumerate(kept):
        p = frames_dir / f'f_{i:03d}_{c["t"]:08.2f}.jpg'
        shutil.copy2(c['path'], p)
        VF.add_watermark(str(p), f"{c['t']:.2f}s  [scene-dedup]  #{i+1:03d}",
                         style='bar')
        frames.append({'t': c['t'], 'reason': 'scene-dedup',
                        'path': str(p)})

    shutil.rmtree(sub_dir, ignore_errors=True)
    return {'frames': frames, 'elapsed': time.time() - t0,
            'raw_scene_count': raw_n}


# ══════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════
def main():
    argv = sys.argv[1:]
    if not argv or '-h' in argv or '--help' in argv:
        print(__doc__); return 0

    opts = {
        'in': None, 'out': 'vcompare_out',
        'range': (0.0, 30.0),
        'max_frames': 15,
        'algo': 'all',
        'scene_threshold': 0.3,
        'hamming': 5,
    }
    for a in argv:
        if a.startswith('--in='): opts['in'] = a.split('=', 1)[1]
        elif a.startswith('--out='): opts['out'] = a.split('=', 1)[1]
        elif a.startswith('--range='):
            p = a.split('=', 1)[1].split(':')
            opts['range'] = (float(p[0]), float(p[1]))
        elif a.startswith('--max-frames='):
            opts['max_frames'] = max(1, int(a.split('=', 1)[1]))
        elif a.startswith('--algo='): opts['algo'] = a.split('=', 1)[1]
        elif a.startswith('--scene-threshold='):
            opts['scene_threshold'] = float(a.split('=', 1)[1])
        elif a.startswith('--hamming='):
            opts['hamming'] = int(a.split('=', 1)[1])

    if not opts['in']:
        print('✗ 需要 --in=video.mp4', file=sys.stderr); return 1

    info = VF.probe_video(opts['in'])
    if info is None:
        print('✗ 无法探测视频', file=sys.stderr); return 1

    t_start, t_end = opts['range']
    out_root = Path(opts['out'])
    out_root.mkdir(parents=True, exist_ok=True)

    print(f'🎬 vcompare — 算法对比')
    print(f'  视频 {opts["in"]}')
    print(f'  范围 [{t_start:.2f}, {t_end:.2f}]s')
    print(f'  每算法最多 {opts["max_frames"]} 帧')
    print()

    algos = (['uniform', 'scene', 'scene-dedup']
             if opts['algo'] == 'all' else [opts['algo']])

    results = {}
    for algo in algos:
        d = out_root / algo
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        print(f'  ▶ {algo}...', end=' ', flush=True)
        try:
            if algo == 'uniform':
                r = algo_uniform(opts['in'], d, info, t_start, t_end,
                                  opts['max_frames'])
            elif algo == 'scene':
                r = algo_scene(opts['in'], d, info, t_start, t_end,
                                opts['max_frames'], opts['scene_threshold'])
            elif algo == 'scene-dedup':
                r = algo_scene_dedup(opts['in'], d, info, t_start, t_end,
                                      opts['max_frames'],
                                      opts['scene_threshold'],
                                      opts['hamming'])
            else:
                print(f'未知算法: {algo}'); continue
        except Exception as e:
            print(f'✗ {e}')
            import traceback; traceback.print_exc()
            continue
        results[algo] = r
        msg = f"{len(r['frames'])} 帧  {r['elapsed']:.1f}s"
        if 'raw_scene_count' in r:
            msg += f"  (scene 原始 {r['raw_scene_count']})"
        print(msg)

        # contact sheet（和 videoframe 一致）
        if r['frames']:
            VF.make_contact_sheet(r['frames'],
                                    d / 'contact_sheet.jpg')

    # 汇总
    print()
    print('=' * 60)
    print(f"  {'算法':<15} {'帧数':>5} {'耗时':>8}")
    print('-' * 60)
    for algo, r in results.items():
        print(f"  {algo:<15} {len(r['frames']):>5} {r['elapsed']:>7.1f}s")
    print('=' * 60)

    print(f'\n✅ 完成 → {out_root}/')
    for algo in results:
        print(f'   {algo}/contact_sheet.jpg')

    print()
    print('对比用：把下列四张 contact_sheet 并排看')
    for algo in results:
        print(f'   {out_root}/{algo}/contact_sheet.jpg')
    print(f'   kards_VF/chunk_000/contact_sheet.jpg  ← 你的 focus 算法')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
