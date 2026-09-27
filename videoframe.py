#!/usr/bin/env python3
"""
videoframe.py v0.1 — 智能视频抽帧

三条独立规则（任一触发即抽）：
  1. 固定间隔  --every=2.0
  2. 内容变化  --diff-threshold=0.08  （和上一个抽出的帧的像素差）
  3. 焦点异常  --focus-threshold=0.30 （和上一个抽出的帧的焦点图局部差）

冷却 0.05s，超上限时按"分段取最高分"裁剪。

用法：
  python videoframe.py --in=video.mp4 --out=keys/
  python videoframe.py --in=video.mp4 --out=keys/ --every=1.0 --max-frames=100
  python videoframe.py --in=video.mp4 --out=keys/ --no-focus
"""
import sys, os, json, subprocess, time, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    from PIL import Image, ImageFilter, ImageDraw
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

__version__ = '0.1'


# ============================================================
# 环境探测
# ============================================================
def probe_video(path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries',
           'stream=width,height,r_frame_rate,nb_frames,duration',
           '-of', 'json', str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return None
        s = json.loads(r.stdout)['streams'][0]
        w = int(s['width']); h = int(s['height'])
        fr = s.get('r_frame_rate', '0/1')
        if '/' in fr:
            n, d = fr.split('/')
            fps = float(n) / float(d) if float(d) else 0.0
        else:
            fps = float(fr)
        nf = s.get('nb_frames')
        nf = int(nf) if nf and str(nf).isdigit() else None
        dur = float(s.get('duration', 0) or 0)

        # ── 读 rotation（手机录屏宽高写反时，ffprobe 报容器尺寸）──
        rotation = 0
        try:
            r2 = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream_side_data=rotation',
                 '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
                capture_output=True, text=True, timeout=5)
            if r2.returncode == 0 and r2.stdout.strip():
                rotation = int(float(r2.stdout.strip().split()[0]))
        except Exception:
            pass
        # 也试 stream_tags=rotate（老式写法）
        if rotation == 0:
            try:
                r3 = subprocess.run(
                    ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                     '-show_entries', 'stream_tags=rotate',
                     '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
                    capture_output=True, text=True, timeout=5)
                if r3.returncode == 0 and r3.stdout.strip():
                    rotation = int(r3.stdout.strip())
            except Exception:
                pass

        # ── 帧率合理性检查 ──
        # 手机录屏常见 VFR / 元数据错乱，可能报 90000 / 1000 这种假值
        # 合理范围 [1, 240]。超出 → 兜底推算
        if fps <= 1.0 or fps > 240.0:
            print(f"  ⚠ ffprobe 报 fps={fps:.2f}，不合理，兜底推算", file=sys.stderr)
            if dur > 0 and nf:
                fps = nf / dur
                print(f"  · 用 nb_frames/duration 推算 fps={fps:.2f}",
                      file=sys.stderr)
            elif dur > 0:
                # 连 nb_frames 都没有 → 用 ffprobe -count_frames 数一遍
                print(f"  · 无 nb_frames，尝试 count_frames（可能慢）...",
                      file=sys.stderr)
                try:
                    r2 = subprocess.run(
                        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                         '-count_frames', '-show_entries',
                         'stream=nb_read_frames', '-of', 'csv=p=0', str(path)],
                        capture_output=True, text=True, timeout=60)
                    if r2.returncode == 0 and r2.stdout.strip().isdigit():
                        nf = int(r2.stdout.strip())
                        fps = nf / dur
                        print(f"  · count_frames 得 {nf} 帧，"
                              f"推算 fps={fps:.2f}", file=sys.stderr)
                except Exception:
                    pass
            # 还是不合理 → 硬兜底 30fps
            if fps <= 1.0 or fps > 240.0:
                print(f"  ⚠ 兜底 fps=30", file=sys.stderr)
                fps = 30.0

        if dur <= 0 and nf and fps > 0:
            dur = nf / fps
        return {'w': w, 'h': h, 'fps': fps, 'duration': dur,
                'n_frames': nf, 'rotation': rotation}
    except Exception:
        return None




# ============================================================
# 预压缩：生成低分辨率临时文件用于扫描
# ============================================================
def _precompress_for_analysis(video_path, out_dir, analysis_width,
                                src_w, src_h, duration,
                                min_sec=30, min_width_mult=2):
    """生成一个临时小视频（低分辨率 + 无音频 + h264 高压缩）用于扫描。

    返回 (path, was_compressed)
      · 原视频已经够小 → (video_path, False)
      · 否则 → (temp_path, True)

    规则：
      · 时长 < min_sec 且长边 < analysis_width × 2 → 不压（开销不值）
      · 长边 <= analysis_width × 2 → 不压（已够小）
      · 否则压到 长边 = analysis_width
    """
    if not shutil.which('ffmpeg'):
        return (video_path, False)

    long_edge = max(src_w, src_h)
    if long_edge <= analysis_width * min_width_mult and duration < min_sec:
        return (video_path, False)
    if long_edge <= analysis_width * min_width_mult:
        return (video_path, False)

    # 目标尺寸：长边 = analysis_width（按显示尺寸算）
    # 这里 src_w/src_h 是"显示尺寸"（rotation 已修正）
    if src_w >= src_h:
        tw = analysis_width
        th = int(src_h * analysis_width / src_w)
    else:
        th = analysis_width
        tw = int(src_w * analysis_width / src_h)
    tw = max(2, tw // 2 * 2)
    th = max(2, th // 2 * 2)

    out_path = Path(out_dir) / '_precompressed.mp4'
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-i', str(video_path),
           '-vf', f'scale={tw}:{th}',
           '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
           '-an',   # 无音频
           '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart',
           str(out_path)]

    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=1800)
    except Exception as e:
        print(f"  ⚠ 预压缩异常: {e}", file=sys.stderr)
        return (video_path, False)

    if r.returncode != 0 or not out_path.exists():
        print(f"  ⚠ 预压缩失败: {r.stderr.decode()[-200:]}", file=sys.stderr)
        return (video_path, False)

    dt = time.time() - t0
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  · 预压缩 {src_w}x{src_h} → {tw}x{th}  "
          f"({dt:.1f}s, {size_mb:.1f} MB)", flush=True)
    return (str(out_path), True)

# ============================================================
# 帧迭代器
# ============================================================
def iter_frames(path, t_start=0.0, t_end=None, scale=1.0, rotation=0):
    """逐帧 yield (t, rgb_uint8, w, h)。

    rotation 非 0 时自动 transpose，yield 出来的画面已经是"转正"的。
    """
    cmd = ['ffmpeg', '-v', 'error']
    if t_start > 0:
        cmd += ['-ss', f'{t_start:.3f}']
    cmd += ['-i', str(path)]
    if t_end is not None and t_end > t_start:
        cmd += ['-t', f'{t_end - t_start:.3f}']

    # rotation 交给 ffmpeg 默认 autorotate 处理
    if scale != 1.0:
        cmd += ['-vf', f'scale=iw*{scale}:ih*{scale}']
    else:
        cmd += ['-vf', 'null']
    cmd += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
    info = probe_video(path)
    if info is None:
        proc.kill(); return

    # 显示尺寸（ffmpeg 已自动旋转，宽高已交换）
    src_w, src_h = info['w'], info['h']
    rot = rotation % 360
    if rot in (90, 270):
        disp_w, disp_h = src_h, src_w
    else:
        disp_w, disp_h = src_w, src_h
    w = int(disp_w * scale) // 2 * 2
    h = int(disp_h * scale) // 2 * 2
    fps = info['fps']
    frame_bytes = w * h * 3
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            yield t_start + idx / fps, arr, w, h
            idx += 1
    finally:
        try:
            proc.stdout.close()
            proc.wait(timeout=5)
        except Exception:
            pass


# ============================================================
# 图像分析
# ============================================================
def _discrete_laplacian(rgb):
    """离散拉普拉斯（原地优化版）。

    原版分 7 次临时数组分配 (padded, up+down, +left, +right,
    /4, -mean, ^2)；改后只分配 padded 和 1 个输出数组。

    rgb 必须是 float32。
    """
    H, W = rgb.shape[:2]
    # padded 数组：手动 edge fill，比 np.pad 少一次拷贝
    p = np.empty((H + 2, W + 2, 3), dtype=np.float32)
    p[1:-1, 1:-1] = rgb
    p[0, 1:-1] = rgb[0]
    p[-1, 1:-1] = rgb[-1]
    p[1:-1, 0] = rgb[:, 0]
    p[1:-1, -1] = rgb[:, -1]
    p[0, 0] = rgb[0, 0]; p[0, -1] = rgb[0, -1]
    p[-1, 0] = rgb[-1, 0]; p[-1, -1] = rgb[-1, -1]

    # out = 4*center − up − down − left − right = 4*(center − mean_4)
    # 目标 = (center − mean_4)^2 = (out/4)^2
    out = p[1:-1, 1:-1, :] * 4.0
    out -= p[:-2, 1:-1, :]
    out -= p[2:, 1:-1, :]
    out -= p[1:-1, :-2, :]
    out -= p[1:-1, 2:, :]
    out *= 0.25
    np.multiply(out, out, out=out)
    return (out[..., 0] + out[..., 1]) + out[..., 2]


def _contour_density(contrast, percentile=99.0, block_size=16):
    # np.percentile 内部排序整个数组 → 慢。
    # np.partition 是 O(n) 选择算法，只找第 k 大的值。
    flat = contrast.ravel()
    n = flat.size
    k = int(n * percentile / 100.0)
    k = min(max(k, 0), n - 1)
    work = flat.copy()
    thr = float(np.partition(work, k)[k])
    mask = contrast > thr
    H, W = mask.shape
    density = np.zeros_like(mask, dtype=np.float32)
    by, bx = H // block_size, W // block_size
    if by == 0 or bx == 0:
        return density
    trimmed = mask[:by*block_size, :bx*block_size].astype(np.float32)
    trimmed = trimmed.reshape(by, block_size, bx, block_size)
    blk = trimmed.mean(axis=(1, 3))
    density[:by*block_size, :bx*block_size] = \
        np.repeat(np.repeat(blk, block_size, axis=0), block_size, axis=1)
    return density


def compute_saliency(rgb):
    """(H, W, 3) uint8 → (H, W) float32 in [0,1]，越大越抢眼。"""
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    contrast = _discrete_laplacian(rgb)
    density = _contour_density(contrast, percentile=99.0)
    d_min, d_max = density.min(), density.max()
    inv_d = 1.0 / (d_max - d_min) if d_max - d_min > 1e-9 else 0.0
    density_norm = (density - d_min) * inv_d

    # ── color_norm：einsum 通道平方和 ──
    mean_color = rgb.reshape(-1, 3).mean(axis=0)
    d = rgb - mean_color
    s = np.einsum('hwc,hwc->hw', d, d, optimize=True)
    np.sqrt(s, out=s)
    c_min = float(s.min())
    c_max = float(s.max())
    if c_max > c_min + 1e-9:
        s -= c_min
        s *= 1.0 / (c_max - c_min)
    else:
        s.fill(0.0)
    color_norm = s

    color_pil = Image.fromarray(
        (color_norm * 255).astype(np.uint8), mode='L'
    ).filter(ImageFilter.GaussianBlur(radius=8))
    color_anomaly = np.asarray(color_pil, dtype=np.float32) / 255.0

    # ── combined：原地 ──
    alpha = 4.0
    combined = color_anomaly * alpha
    combined += 1.0
    combined *= density_norm
    c_min2 = float(combined.min())
    c_max2 = float(combined.max())
    if c_max2 > c_min2 + 1e-9:
        combined -= c_min2
        combined *= 1.0 / (c_max2 - c_min2)
    else:
        combined.fill(0.0)
    base = combined

    macro_pil = Image.fromarray(
        (base * 255).astype(np.uint8), mode='L'
    ).filter(ImageFilter.GaussianBlur(radius=6))
    macro_pil = macro_pil.filter(
        ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return np.asarray(macro_pil, dtype=np.float32) / 255.0




# ============================================================
# 单 chunk 扫描（可并行调用）
# ============================================================
def _scan_one_chunk(video_path, t_start, t_end, opts, video_info,
                     chunk_idx=0, chunk_total=1, scan_rotation=None):
    """扫描 [t_start, t_end) 区间，返回 (candidates, stats)。

    每个 chunk 独立：
      · fixed 时间轴从 t_start 起算
      · cooldown 状态独立
      · first 规则只在 chunk_idx == 0 且 chunk_total > 1 时保留；
        单 chunk 模式（chunk_total == 1）走老行为
    """
    W, H = video_info['w'], video_info['h']
    if scan_rotation is None:
        rot = video_info.get('rotation', 0) % 360
    else:
        rot = scan_rotation % 360
    if rot in (90, 270):
        DW, DH = H, W
    else:
        DW, DH = W, H
    a_scale = min(1.0, opts['analysis_width'] / max(1, DW))
    _stride = max(1, opts['analysis_stride'])

    candidates = []
    n_total = 0
    last_t = -1e9
    last_rgb = None
    last_sal = None
    next_fixed_t = t_start
    first_seen = False
    n_skipped_cooldown = 0

    for t, rgb, aw, ah in iter_frames(video_path, t_start=t_start,
                                       t_end=t_end, scale=a_scale,
                                       rotation=rot):
        n_total += 1
        if (n_total - 1) % _stride != 0 and first_seen:
            continue

        # 先推进 fixed 时间轴
        fixed_due = False
        if not opts.get('no_fixed', False):
            if t >= next_fixed_t - 1e-6:
                fixed_due = True
                while next_fixed_t <= t:
                    next_fixed_t += opts['every']

        in_cooldown = (t - last_t) < opts['cooldown']
        if in_cooldown and not fixed_due and first_seen:
            n_skipped_cooldown += 1
            continue

        sal = None
        if not opts['no_focus']:
            sal = compute_saliency(rgb)

        reasons = []
        diff_val = 0.0
        focus_val = 0.0

        # first 规则：只在视频首帧（chunk 0 且未见过 first）触发
        if not first_seen:
            first_seen = True
            if chunk_idx == 0:
                reasons.append('first')

        if fixed_due:
            reasons.append('fixed')

        if not in_cooldown and last_rgb is not None:
            d = float(np.abs(rgb.astype(np.float32) - last_rgb).mean() / 255.0)
            diff_val = d
            if d >= opts['diff_threshold']:
                reasons.append('content')
            if sal is not None and last_sal is not None:
                fdiff = np.abs(sal - last_sal)
                fmax = float(np.percentile(fdiff, 99.5))
                focus_val = fmax
                if fmax >= opts['focus_threshold']:
                    reasons.append('focus')

        if not reasons:
            continue

        if 'first' in reasons:
            reason = 'first'
        elif 'focus' in reasons:
            reason = 'focus'
        elif 'content' in reasons:
            reason = 'content'
        else:
            reason = 'fixed'

        candidates.append({
            't': round(float(t), 3),
            'reason': reason,
            'diff': round(diff_val, 4),
            'focus_val': round(focus_val, 4),
        })
        last_t = t
        last_rgb = rgb.copy()
        last_sal = sal.copy() if sal is not None else None

    stats = {
        'n_total': n_total,
        'n_skipped_cooldown': n_skipped_cooldown,
        'n_candidates': len(candidates),
    }
    return candidates, stats

# ============================================================
# 抽帧（全分辨率，ffmpeg -ss 精确抽取）
# ============================================================
_FONT_CACHE = {}


def _load_watermark_font(size):
    """按优先级找字体。返回 ImageFont 对象。"""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    from PIL import ImageFont
    candidates = []

    def _try_add(p):
        p = (p or '').strip()
        if p and p not in candidates and Path(p).exists():
            candidates.append(p)

    # 1. fc-match sans-serif（fontconfig 匹配规则，最聪明）
    for query in ('sans-serif', 'sans-serif:lang=zh'):
        try:
            r = subprocess.run(
                ['fc-match', '-f', '%{file}', query],
                capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                _try_add(r.stdout)
        except Exception:
            pass

    # 2. fc-list 第一项（fc-match 挂了的备用）
    try:
        r = subprocess.run(
            ['fc-list', '-f', '%{file}\n'],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                _try_add(line)
                if candidates:
                    break
    except Exception:
        pass

    # 3. glob 扫描常见字体目录（不写死文件名，只写目录）
    search_dirs = [
        '/system/fonts',                                  # Android
        '/data/data/com.termux/files/usr/share/fonts',    # Termux
        '/usr/share/fonts',                               # Linux 系统
        '/usr/local/share/fonts',                         # Linux 本地
        Path.home() / '.fonts',                           # 用户
        Path.home() / '.local' / 'share' / 'fonts',       # XDG
        '/System/Library/Fonts',                          # macOS
        'C:/Windows/Fonts',                               # Windows
    ]
    for d in search_dirs:
        dp = Path(d)
        if not dp.exists():
            continue
        for pat in ('*.ttf', '*.ttc', '*.otf',
                    '**/*.ttf', '**/*.ttc', '**/*.otf'):
            added = False
            for f in dp.glob(pat):
                if f.is_file():
                    _try_add(str(f))
                    added = True
                    break
            if added:
                break
        if len(candidates) >= 3:
            break

    # 逐个尝试打开
    for p in candidates:
        try:
            f = ImageFont.truetype(p, size)
            _FONT_CACHE[size] = f
            return f
        except Exception:
            continue

    # 最后兜底：PIL 默认字体（连字体文件都没有的裸环境也能跑）
    f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def add_watermark(img_path, text, style='bar'):
    """在图片右下角加水印。
    style='bar'  → 半透明黑底 + 白字
    style='text' → 白字 + 黑描边（无背景）
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    except Exception:
        return False

    W, H = img.size
    # 字号按短边自动缩放
    fsize = max(14, min(W, H) // 45)
    font = _load_watermark_font(fsize)
    margin = max(8, fsize // 2)

    if style == 'text':
        draw = ImageDraw.Draw(img)
        try:
            draw.text((W - margin, H - margin), text,
                      font=font, fill=(255, 255, 255),
                      anchor='rb',
                      stroke_width=max(1, fsize // 12),
                      stroke_fill=(0, 0, 0))
        except TypeError:
            # 老 PIL 不支持 anchor/stroke
            try:
                bbox = font.getbbox(text)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                draw.text((W - margin - tw, H - margin - th), text,
                          font=font, fill=(255, 255, 255))
            except Exception:
                return False
        img.save(str(img_path), 'JPEG', quality=92, optimize=True)
        return True

    # style == 'bar'
    try:
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw, th = fsize * len(text) // 2, fsize
    pad_x = max(6, fsize // 2)
    pad_y = max(3, fsize // 3)
    bar_w = tw + pad_x * 2
    bar_h = th + pad_y * 2
    x0 = W - bar_w - margin
    y0 = H - bar_h - margin

    # 半透明黑底
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([x0, y0, x0 + bar_w, y0 + bar_h], fill=(0, 0, 0, 170))
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')

    # 白字
    draw = ImageDraw.Draw(img)
    try:
        draw.text((x0 + pad_x, y0 + pad_y - bbox[1]), text,
                  font=font, fill=(255, 255, 255))
    except Exception:
        draw.text((x0 + pad_x, y0 + pad_y), text,
                  font=font, fill=(255, 255, 255))

    img.save(str(img_path), 'JPEG', quality=92, optimize=True)
    return True


def _compute_frame_size(src_w, src_h, rotation, max_w, max_h):
    """算输出尺寸。

    返回 (target_w, target_h, need_transpose)
      need_transpose: ffmpeg 的 transpose 是否要跑（rotation 非 0）
    最终输出永远 <= (max_w, max_h)，按比例缩放。
    """
    # rotation 修正：±90/270 → 宽高对调
    rot = rotation % 360
    if rot in (90, 270):
        disp_w, disp_h = src_h, src_w  # 显示时的实际宽高
        need_transpose = True
    else:
        disp_w, disp_h = src_w, src_h
        need_transpose = False

    # 目标尺寸：在 (max_w, max_h) 内按比例缩放
    if max_w > 0 and max_h > 0:
        # 尽量填满但不变形
        k = min(max_w / disp_w, max_h / disp_h)
        # 如果已经小于上限，就不放大
        if k >= 1.0:
            tw, th = disp_w, disp_h
        else:
            tw = int(disp_w * k)
            th = int(disp_h * k)
    elif max_w > 0 and disp_w > max_w:
        tw = max_w
        th = int(disp_h * max_w / disp_w)
    elif max_h > 0 and disp_h > max_h:
        th = max_h
        tw = int(disp_w * max_h / disp_h)
    else:
        tw, th = disp_w, disp_h
    # 偶数化
    tw = max(2, tw // 2 * 2)
    th = max(2, th // 2 * 2)
    return tw, th, need_transpose


def extract_frame_at(video_path, t, out_path, src_w=None, src_h=None,
                      max_size=0, max_w=0, max_h=0, rotation=0):
    """抽一帧。

    max_size > 0 → 长边压缩（旧接口）
    max_w/max_h > 0 → 按 (max_w, max_h) 边界内等比缩放（新接口）
    rotation: 从 probe_video 读到的旋转角度，非 0 时自动 transpose
    """
    ext = Path(out_path).suffix.lower()
    vf_parts = []

    # rotation 交给 ffmpeg 默认 autorotate 处理
    # src_w/src_h 已经是"旋转后的显示尺寸"

    # 尺寸计算
    if src_w and src_h:
        if max_w > 0 or max_h > 0:
            tw, th, _ = _compute_frame_size(
                src_w, src_h, 0,
                max_w or 99999, max_h or 99999)
            if (tw, th) != (src_w, src_h):
                vf_parts.append(f'scale={tw}:{th}')
        elif max_size > 0:
            long_edge = max(src_w, src_h)
            if long_edge > max_size:
                if src_w >= src_h:
                    nw, nh = max_size, int(src_h * max_size / src_w)
                else:
                    nh, nw = max_size, int(src_w * max_size / src_h)
                nw = max(2, nw // 2 * 2)
                nh = max(2, nh // 2 * 2)
                vf_parts.append(f'scale={nw}:{nh}')

    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-ss', f'{t:.3f}', '-i', str(video_path),
           '-vframes', '1']
    if vf_parts:
        cmd += ['-vf', ','.join(vf_parts)]
    if ext in ('.jpg', '.jpeg'):
        cmd += ['-q:v', '2']
    elif ext == '.png':
        cmd += ['-compression_level', '1']
    cmd += [str(out_path)]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    return r.returncode == 0 and Path(out_path).exists()


# ============================================================
# 上限裁剪：分段取最高分
# ============================================================
def _score_of(f):
    base = {'first': 100.0, 'focus': 3.0, 'content': 2.0,
            'fixed': 1.0}.get(f['reason'], 0.5)
    return base + f.get('diff', 0.0) + f.get('focus_val', 0.0)


def select_top_n(frames, max_n):
    if len(frames) <= max_n:
        return frames
    fs = sorted(frames, key=lambda f: f['t'])
    t0 = fs[0]['t']; t1 = fs[-1]['t']
    if t1 <= t0:
        return fs[:max_n]
    seg = (t1 - t0) / max_n
    out = []
    for i in range(max_n):
        lo = t0 + i * seg
        hi = lo + seg
        chunk = [f for f in fs if lo <= f['t'] < hi]
        if not chunk:
            continue
        out.append(max(chunk, key=_score_of))
    return out


# ============================================================
# Contact sheet
# ============================================================
def make_contact_sheet(frames, out_path, cols=4, thumb_w=320):
    if not frames or not _HAS_PIL:
        return False
    _is_jpg = str(out_path).lower().endswith(('.jpg', '.jpeg'))
    thumbs = []
    for f in frames:
        try:
            img = Image.open(f['path']).convert('RGB')
            w, h = img.size
            tw = thumb_w
            th = max(1, int(h * tw / w))
            rs = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') \
                 else Image.LANCZOS
            img = img.resize((tw, th), rs)
            thumbs.append((img, f))
        except Exception:
            continue
    if not thumbs:
        return False
    rows = (len(thumbs) + cols - 1) // cols
    th_all = max(t.height for t, _ in thumbs)
    label_h = 22
    cell_h = th_all + label_h
    sheet = Image.new('RGB', (cols * thumb_w, rows * cell_h), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for i, (img, f) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = c * thumb_w
        y = r * cell_h
        sheet.paste(img, (x, y))
        label = f"{f['t']:7.2f}s  [{f['reason']}]"
        dr.text((x + 4, y + th_all + 3), label, fill=(200, 200, 200))
    if _is_jpg:
        sheet.save(str(out_path), 'JPEG', quality=90, optimize=True)
    else:
        sheet.save(str(out_path))
    return True


# ============================================================
# main
# ============================================================
def main():
    argv = sys.argv[1:]
    if not argv or '-h' in argv or '--help' in argv:
        print(__doc__); return 0
    if not (_HAS_NUMPY and _HAS_PIL):
        print("✗ 需要 numpy + pillow", file=sys.stderr); return 1

    opts = {
        'in': None, 'out': 'keys',
        'every': 2.0,
        'diff_threshold': 0.08,
        'focus_threshold': 0.30,
        'cooldown': 0.05,
        'max_frames': 300,
        'analysis_width': 640,
        'no_focus': False,
        'no_fixed': False,
        'json_out': None,
        't_start': None, 't_end': None,
        'ext': 'jpg',
        'no_watermark': False,
        'watermark_style': 'bar',
        'analysis_stride': 2,
        'frame_max': 0,      # 旧语义：长边限制（0 = 不用）
        'frame_w': 1280,     # 新语义：显示尺寸 ≤ frame_w
        'frame_h': 760,      # 新语义：显示尺寸 ≤ frame_h
        'chunk_sec': 0,      # 分块时长（秒），0 = 不分块
        'chunk_parallel': 2, # 并行 chunk 数
        'precompress': True,      # 扫描前用 ffmpeg 预压缩
        'precompress_min_sec': 30,  # 小于此时长不压缩
    }
    for a in argv:
        if a.startswith('--in='): opts['in'] = a.split('=', 1)[1]
        elif a.startswith('--out='): opts['out'] = a.split('=', 1)[1]
        elif a.startswith('--every='):
            try: opts['every'] = max(0.05, float(a.split('=', 1)[1]))
            except ValueError: pass
        elif a.startswith('--diff-threshold='):
            try: opts['diff_threshold'] = float(a.split('=', 1)[1])
            except ValueError: pass
        elif a.startswith('--focus-threshold='):
            try: opts['focus_threshold'] = float(a.split('=', 1)[1])
            except ValueError: pass
        elif a.startswith('--cooldown='):
            try: opts['cooldown'] = max(0.0, float(a.split('=', 1)[1]))
            except ValueError: pass
        elif a.startswith('--max-frames='):
            try: opts['max_frames'] = max(1, int(a.split('=', 1)[1]))
            except ValueError: pass
        elif a.startswith('--analysis-width='):
            try: opts['analysis_width'] = max(160, int(a.split('=', 1)[1]))
            except ValueError: pass
        elif a.startswith('--analysis-stride='):
            try: opts['analysis_stride'] = max(1, int(a.split('=', 1)[1]))
            except ValueError: pass
        elif a == '--no-focus': opts['no_focus'] = True
        elif a == '--no-fixed': opts['no_fixed'] = True
        elif a.startswith('--json='): opts['json_out'] = a.split('=', 1)[1]
        elif a.startswith('--ext='):
            e = a.split('=', 1)[1].strip().lower().lstrip('.')
            if e in ('jpg', 'jpeg', 'png'):
                opts['ext'] = e
        elif a == '--no-watermark':
            opts['no_watermark'] = True
        elif a.startswith('--frame-max='):
            try:
                v = int(a.split('=', 1)[1])
                opts['frame_max'] = max(0, v)
            except ValueError:
                pass
        elif a.startswith('--frame-max-w=') or a.startswith('--frame-w='):
            try:
                opts['frame_w'] = max(0, int(a.split('=', 1)[1]))
            except ValueError:
                pass
        elif a.startswith('--frame-max-h=') or a.startswith('--frame-h='):
            try:
                opts['frame_h'] = max(0, int(a.split('=', 1)[1]))
            except ValueError:
                pass
        elif a.startswith('--chunk-sec='):
            try:
                opts['chunk_sec'] = max(0.0, float(a.split('=', 1)[1]))
            except ValueError:
                pass
        elif a.startswith('--chunk-parallel='):
            try:
                opts['chunk_parallel'] = max(1, int(a.split('=', 1)[1]))
            except ValueError:
                pass
        elif a == '--no-precompress':
            opts['precompress'] = False
        elif a.startswith('--precompress-min-sec='):
            try:
                opts['precompress_min_sec'] = max(0, int(a.split('=', 1)[1]))
            except ValueError:
                pass
        elif a.startswith('--watermark-style='):
            v = a.split('=', 1)[1].strip().lower()
            if v in ('bar', 'text', 'none'):
                if v == 'none':
                    opts['no_watermark'] = True
                else:
                    opts['watermark_style'] = v
        elif a.startswith('--range='):
            try:
                p = a.split('=', 1)[1].split(':')
                opts['t_start'] = float(p[0])
                opts['t_end'] = float(p[1])
            except Exception: pass

    if not opts['in']:
        print("✗ 需要 --in=视频.mp4", file=sys.stderr); return 1
    if not Path(opts['in']).exists():
        print(f"✗ 文件不存在: {opts['in']}", file=sys.stderr); return 1

    print(f"🎞 videoframe v{__version__}")
    info = probe_video(opts['in'])
    if info is None:
        print(f"✗ 无法探测视频", file=sys.stderr); return 1
    W, H = info['w'], info['h']
    fps = info['fps']
    dur = info['duration']
    t_s = opts['t_start'] if opts['t_start'] is not None else 0.0
    t_e = opts['t_end'] if opts['t_end'] is not None else dur

    print(f"  输入   {opts['in']}")
    print(f"  尺寸   {W}x{H} @ {fps:.2f}fps  {t_s:.2f}~{t_e:.2f}s")
    print(f"  输出   {opts['out']}/frames/*.{opts['ext']}")
    _rules = []
    if opts['no_fixed']:
        _rules.append('fixed(off)')
    else:
        _rules.append(f"every={opts['every']}s")
    _rules.append(f"diff≥{opts['diff_threshold']}")
    if opts['no_focus']:
        _rules.append('focus(off)')
    else:
        _rules.append(f"focus≥{opts['focus_threshold']}")
    print(f"  规则   {'  '.join(_rules)}")
    if opts['no_watermark']:
        print(f"  水印   关闭")
    else:
        print(f"  水印   {opts['watermark_style']}")
    print(f"  冷却   {opts['cooldown']}s  上限 {opts['max_frames']}")

    # 显示尺寸（rotation 修正）
    _rot = info.get('rotation', 0) % 360
    if _rot in (90, 270):
        DW, DH = H, W
    else:
        DW, DH = W, H

    # 目标尺寸计算
    if opts['frame_max'] > 0:
        # 旧语义：长边限制
        if max(DW, DH) > opts['frame_max']:
            if DW >= DH:
                ow = opts['frame_max']
                oh = int(DH * opts['frame_max'] / DW) // 2 * 2
            else:
                oh = opts['frame_max']
                ow = int(DW * opts['frame_max'] / DH) // 2 * 2
            print(f"  帧尺寸 {DW}x{DH} → {ow}x{oh}  (长边 {opts['frame_max']})")
        else:
            print(f"  帧尺寸 {DW}x{DH}  (未超 {opts['frame_max']}，保持原尺寸)")
    elif opts['frame_w'] > 0 and opts['frame_h'] > 0:
        # 新语义：边界框
        k = min(opts['frame_w'] / DW, opts['frame_h'] / DH)
        if k >= 1.0:
            print(f"  帧尺寸 {DW}x{DH}  (已 ≤ {opts['frame_w']}x{opts['frame_h']})")
        else:
            ow = int(DW * k) // 2 * 2
            oh = int(DH * k) // 2 * 2
            print(f"  帧尺寸 {DW}x{DH} → {ow}x{oh}  "
                  f"(装进 {opts['frame_w']}x{opts['frame_h']})")
    else:
        print(f"  帧尺寸 {DW}x{DH}  (不压缩)")
    if _rot != 0:
        print(f"  旋转   {W}x{H} → {DW}x{DH}  (rotation={_rot}°)")
    if opts['chunk_sec'] > 0 and (t_e - t_s) > opts['chunk_sec'] * 1.5:
        _n_chunks = int((t_e - t_s) / opts['chunk_sec'] + 0.999)
        print(f"  分块   {_n_chunks} 个 × {opts['chunk_sec']:.0f}s  "
              f"(并行 {opts['chunk_parallel']})")
    print()

    out_dir = Path(opts['out'])
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── 预压缩 ──
    analysis_path = opts['in']
    was_compressed = False
    if opts.get('precompress', True):
        analysis_path, was_compressed = _precompress_for_analysis(
            opts['in'], out_dir, opts['analysis_width'],
            DW, DH, t_e - t_s,
            min_sec=opts.get('precompress_min_sec', 30))
    if was_compressed:
        print(f"  扫描源 {analysis_path}  (预压缩)")
    else:
        print(f"  扫描源 {opts['in']}  (原视频)")

    # 分析分辨率（用显示宽度算，因为 rotation 后宽度可能变）
    a_scale = min(1.0, opts['analysis_width'] / max(1, DW))
    _aw = int(W * a_scale); _ah = int(H * a_scale)
    print(f"  · 分析分辨率 {_aw}x{_ah}  stride={opts['analysis_stride']}")
    if _aw > 400 and opts['analysis_width'] > 400:
        print(f"    (提示: --analysis-width=320 可再快 3~4 倍，精度影响很小)")

    # ── chunk 划分 ──
    use_chunks = (opts['chunk_sec'] > 0 and
                  (t_e - t_s) > opts['chunk_sec'] * 1.5)

    if use_chunks:
        chunk_bounds = []
        tt = t_s
        while tt < t_e - 1e-6:
            chunk_bounds.append((tt, min(t_e, tt + opts['chunk_sec'])))
            tt += opts['chunk_sec']
    else:
        chunk_bounds = [(t_s, t_e)]

    n_chunks = len(chunk_bounds)

    # ── 扫描（并行，如果 n_chunks > 1）──
    t_loop0 = time.time()

    if n_chunks == 1:
        scan_rot = 0 if was_compressed else _rot
        candidates, stats = _scan_one_chunk(
            analysis_path, t_s, t_e, opts, info, 0, 1, scan_rot)
        chunk_results = [(candidates, stats)]
    else:
        print(f"  · 并行扫描 {n_chunks} 个 chunk "
              f"(max_workers={min(opts['chunk_parallel'], n_chunks)})...",
              flush=True)
        chunk_results = [None] * n_chunks
        with ThreadPoolExecutor(
                max_workers=min(opts['chunk_parallel'], n_chunks)) as ex:
            futures = {}
            scan_rot = 0 if was_compressed else _rot
            for ci, (cs, ce) in enumerate(chunk_bounds):
                fut = ex.submit(_scan_one_chunk, analysis_path, cs, ce,
                                 opts, info, ci, n_chunks, scan_rot)
                futures[fut] = ci
            for fut in as_completed(futures):
                ci = futures[fut]
                try:
                    cand, st = fut.result()
                    chunk_results[ci] = (cand, st)
                    print(f"    chunk_{ci:03d} 扫描完成: 候选 {len(cand)} "
                          f"(帧 {st['n_total']})", flush=True)
                except Exception as e:
                    print(f"    ⚠ chunk_{ci:03d} 扫描失败: {e}",
                          file=sys.stderr)
                    chunk_results[ci] = ([], {'n_total': 0,
                                                'n_skipped_cooldown': 0,
                                                'n_candidates': 0})

    el_scan = time.time() - t_loop0

    # ── 汇总 ──
    total_n = sum(st['n_total'] for _, st in chunk_results)
    total_skip = sum(st['n_skipped_cooldown'] for _, st in chunk_results)
    total_cand = sum(len(c) for c, _ in chunk_results)
    skip_msg = f"  (冷却跳过 {total_skip})" if total_skip > 0 else ""
    print(f"  · 扫描完成 {total_n} 帧 {el_scan:.1f}s{skip_msg}")
    print(f"  · 触发 {total_cand} 帧")

    # ── 每 chunk：上限裁剪 + 抽帧 + 拼图 + 子报告 ──
    all_frames = []       # 汇总到顶层
    chunk_reports = []    # 每个 chunk 的报告
    t_extract0 = time.time()
    total_ok = 0

    for ci, (cs, ce) in enumerate(chunk_bounds):
        candidates, stats = chunk_results[ci]
        if not candidates:
            continue

        if n_chunks == 1:
            chunk_dir = out_dir
            frames_dir_c = frames_dir
        else:
            chunk_dir = out_dir / f'chunk_{ci:03d}'
            chunk_dir.mkdir(parents=True, exist_ok=True)
            frames_dir_c = chunk_dir / 'frames'
            frames_dir_c.mkdir(parents=True, exist_ok=True)

        # 上限裁剪（每 chunk）
        if len(candidates) > opts['max_frames']:
            print(f"  · chunk_{ci:03d} 超上限，分段取最高分 → "
                  f"{opts['max_frames']}")
            candidates = select_top_n(candidates, opts['max_frames'])

        # 抽帧
        if n_chunks == 1:
            print(f"\n  · 精确抽帧（全分辨率）...")
        else:
            print(f"\n  · chunk_{ci:03d} [{cs:.2f}, {ce:.2f}] 抽帧 "
                  f"({len(candidates)} 张)...")

        ok_count = 0
        for idx, c in enumerate(candidates, 1):
            name = f"f_{c['t']:08.2f}_{c['reason']}.{opts['ext']}"
            out_path = frames_dir_c / name
            if extract_frame_at(opts['in'], c['t'], out_path,
                                 src_w=DW, src_h=DH,
                                 max_size=opts['frame_max'],
                                 max_w=opts['frame_w'] if opts['frame_max'] == 0 else 0,
                                 max_h=opts['frame_h'] if opts['frame_max'] == 0 else 0,
                                 rotation=_rot):
                c['path'] = str(out_path)
                c['chunk'] = ci
                ok_count += 1
                if not opts['no_watermark'] and opts['ext'] in ('jpg', 'jpeg'):
                    wm_text = f"{c['t']:.2f}s  [{c['reason']}]  #{idx:03d}"
                    add_watermark(str(out_path), wm_text,
                                  style=opts['watermark_style'])
            else:
                print(f"    ⚠ 抽帧失败 t={c['t']}", file=sys.stderr)
                c['path'] = None
            if idx % 10 == 0 or idx == len(candidates):
                el2 = time.time() - t_extract0
                print(f"    {idx}/{len(candidates)}  {el2:.1f}s", flush=True)

        candidates = [c for c in candidates if c.get('path')]
        total_ok += len(candidates)
        all_frames.extend(candidates)

        # 每 chunk contact sheet
        if candidates:
            cs_path = chunk_dir / f'contact_sheet.{opts["ext"]}'
            if make_contact_sheet(candidates, cs_path):
                print(f"  ✓ {cs_path}")

            # 每 chunk report
            c_stats = {'total': len(candidates), 'first': 0, 'fixed': 0,
                       'content': 0, 'focus': 0}
            for c in candidates:
                c_stats[c['reason']] = c_stats.get(c['reason'], 0) + 1
            chunk_reports.append({
                'chunk': ci,
                't_start': round(cs, 3),
                't_end': round(ce, 3),
                'stats': c_stats,
                'frames': candidates,
            })

            if n_chunks > 1:
                rpath = chunk_dir / 'report.json'
                rpath.write_text(
                    json.dumps(chunk_reports[-1],
                                indent=2, ensure_ascii=False),
                    encoding='utf-8')
                print(f"  ✓ {rpath}")

    # ── 顶层汇总报告 ──
    stats = {'total': total_ok, 'first': 0, 'fixed': 0,
             'content': 0, 'focus': 0}
    for c in all_frames:
        stats[c['reason']] = stats.get(c['reason'], 0) + 1

    json_path = Path(opts['json_out']) if opts['json_out'] \
                else (out_dir / 'report.json')
    report = {
        'video': {'path': str(opts['in']), 'w': W, 'h': H,
                   'fps': fps, 'duration': dur, 'n_frames': total_n,
                   'rotation': _rot},
        'params': {'every': opts['every'],
                    'diff_threshold': opts['diff_threshold'],
                    'focus_threshold': opts['focus_threshold'],
                    'cooldown': opts['cooldown'],
                    'max_frames': opts['max_frames'],
                    'chunk_sec': opts['chunk_sec'],
                    'chunk_parallel': opts['chunk_parallel']},
        'chunks': [{'chunk': r['chunk'],
                     't_start': r['t_start'], 't_end': r['t_end'],
                     'stats': r['stats']} for r in chunk_reports],
        'frames': all_frames,
        'stats': stats,
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                          encoding='utf-8')
    print(f"  ✓ {json_path}")

    if n_chunks == 1:
        print(f"\n✅ {total_ok} 帧 → {frames_dir}/  "
              f"({stats.get('first',0)}首 + {stats.get('focus',0)}f + "
              f"{stats.get('content',0)}c + {stats.get('fixed',0)}x)")
    else:
        print(f"\n✅ {total_ok} 帧 / {len(chunk_reports)} chunks → {out_dir}/  "
              f"({stats.get('first',0)}首 + {stats.get('focus',0)}f + "
              f"{stats.get('content',0)}c + {stats.get('fixed',0)}x)")
        for r in chunk_reports:
            print(f"    chunk_{r['chunk']:03d}: {r['stats']['total']} 帧 "
                  f"[{r['t_start']:.1f}, {r['t_end']:.1f}]")

    # 清理临时压缩文件
    if was_compressed:
        try:
            Path(analysis_path).unlink()
            print(f"  · 已清理 {analysis_path}")
        except Exception:
            pass
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
