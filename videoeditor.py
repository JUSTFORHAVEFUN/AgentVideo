#!/usr/bin/env python3
"""
videoeditor.py v1.1.0


GenericAIPPTEditor · 让 AI 像写 XML 一样写视频

v1.1.0 新增：
  · 音频延迟合成：分块渲染全部静音（-an，无 AAC 转码），
    最后一步一次性合成所有音频轨。视频部分快 20-40%。
    例外：如果 XML 里显式标记 audio 需要逐 chunk 处理（未来扩展），
    则回退到旧行为。
  · 画中画：overlay/video clip 支持 <style w="320" h="180" x="..." y="..."/>，
    不再强制全屏缩放。
  · MP4 工具箱（独立子命令，无需 project.xml）：
      --mp4-info      video.mp4                       查看信息
      --mp4-snapshot  video.mp4 --at=3.5              抽帧
      --mp4-trim      video.mp4 --range=10:20         裁剪
      --mp4-volume    video.mp4 --volume=0.5          音量
      --mp4-transform video.mp4 --crop=W:H:X:Y        画布变换
                                 --scale=W:H --pos=X:Y

v1.0.3-rc.5 保留：
  · --quality=ultra|fast|special
  · --render-frame=T, --render-range=A:B
  · --aggressive=0~3, --merge-aggressive=0~3
  · --stable, --resume, --keep, --purge, --preview

用法示例：
  python videoeditor.py project.xml --aggressive=3
  python videoeditor.py project.xml --quality=ultra --aggressive=3
  python videoeditor.py project.xml --render-frame=3.5 --format=png
  python videoeditor.py project.xml --render-range=30:40
  python videoeditor.py --mp4-info video.mp4
  python videoeditor.py --mp4-snapshot video.mp4 --at=3.5 --format=png
  python videoeditor.py --mp4-trim video.mp4 --range=10:20 --out=cut.mp4
  python videoeditor.py --mp4-volume video.mp4 --volume=0.3 --out=quieter.mp4
  python videoeditor.py --mp4-transform video.mp4 --crop=1280:720:100:50 --scale=640:360
"""
__version__ = '1.0.2'

import re, os, sys, math, base64, shutil, hashlib, subprocess, time, traceback, json, copy
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

__version__ = '1.0.2'

# 视锥剔除开关（环境变量控制，无 exp 前缀）
_CULL_ENABLED = os.environ.get('XMLVE_NO_CULL') != '1'
_CULL_DEBUG   = os.environ.get('XMLVE_DEBUG_CULL') == '1'

# 动态实例化开关（exp，环境变量控制）
# 把有 anims 但结构一致的 mesh（如烟花粒子）合并成一次 draw call
_DYN_INST_ENABLED = os.environ.get('XMLVE_DYN_INST') == '1'
_DYN_INST_DEBUG   = os.environ.get('XMLVE_DEBUG_DYN_INST') == '1'

# 🆕 快速路径总开关（默认关 = 视觉正确，慢）
# XMLVE_FASTPATH=1 → 开启 auto-inst + param 批量渲染（快，但非均匀缩放下光照会失真）
# 背景：instancing 路径在非均匀缩放时法线变换有视觉折扣（"贴纸感"），
#       当前 shader 未完全修复。默认关，让用户自己权衡。
_FASTPATH_ENABLED = os.environ.get('XMLVE_FASTPATH') == '1'

# 🆕 GPU 2D 光栅化（可选）
_HAS_GPU_2D = False
_GPU_2D_STATS = {'gpu': 0, 'cairo': 0}
try:
    from scene2d import svg_to_raster2d as _svg_to_raster2d
    from raster2d import Raster2D as _Raster2D
    _HAS_GPU_2D = True
except ImportError:
    _svg_to_raster2d = None
    _Raster2D = None

# 帧中间格式：png / bmp（--exp-bmp 时切 bmp）
def _normalize_frame_ext(e):
    e = (e or '.png').strip().lower()
    if not e.startswith('.'):
        e = '.' + e
    return e if e in ('.png', '.bmp') else '.png'

_FRAME_EXT = _normalize_frame_ext(os.environ.get('XMLVE_FRAME_EXT', '.png'))

def _set_frame_ext(e):
    global _FRAME_EXT
    _FRAME_EXT = _normalize_frame_ext(e)
    os.environ['XMLVE_FRAME_EXT'] = _FRAME_EXT


# 字体 weight 探测：(family, weight) -> bool
_FC_WEIGHT_CACHE = {}


def _check_weight_ok(svg_bytes):
    """返回 True 表示 GPU 路径可以处理这个 SVG 的字重。

    raster2d 已经实现 SYNTHETIC 加粗（无真 Bold face 时用 stroke 模拟），
    所以 GPU 路径**总是能**渲染任何 font-weight。保留此函数仅为向后兼容，
    永远返回 True。
    """
    return True


# GPU 不支持的 SVG 标签
_GPU_BLOCKED_TAGS = frozenset([
    'path', 'mask', 'clipPath', 'linearGradient', 'radialGradient',
    'filter', 'pattern', 'use', 'symbol', 'polygon', 'polyline',
    'line', 'defs', 'marker', 'textPath', 'tref', 'switch',
])




# 渲染精度预设

QUALITY_PRESETS = {
    'ultra':   {'desc': '原分辨率 · fast CRF18 · -c copy',
                'w': None, 'h': None, 'fps': None,
                'chunk_preset': 'fast', 'chunk_crf': '18', 'chunk_ab': '256k',
                'merge_reencode': False,
                'merge_preset': 'medium', 'merge_crf': '23', 'merge_ab': '192k'},
    'fast':    {'desc': '原分辨率 · ultrafast CRF16 → medium CRF23',
                'w': None, 'h': None, 'fps': None,
                'chunk_preset': 'ultrafast', 'chunk_crf': '16', 'chunk_ab': '256k',
                'merge_reencode': True,
                'merge_preset': 'medium', 'merge_crf': '23', 'merge_ab': '192k'},
    'special': {'desc': '640x360 @ 30fps · ultrafast CRF16 · -c copy',
                'w': 640, 'h': 360, 'fps': 30,
                'chunk_preset': 'ultrafast', 'chunk_crf': '16', 'chunk_ab': '128k',
                'merge_reencode': False,
                'merge_preset': 'medium', 'merge_crf': '23', 'merge_ab': '128k'},
    'qsv':     {'desc': '原分辨率 · h264_qsv · -c copy',
                'w': None, 'h': None, 'fps': None,
                'chunk_preset': None, 'chunk_crf': None, 'chunk_ab': '192k',
                'chunk_codec': ['-c:v', 'h264_qsv', '-global_quality', '22'],
                'merge_reencode': False,
                'merge_preset': 'medium', 'merge_crf': '23', 'merge_ab': '192k'},
    'nvenc':   {'desc': '原分辨率 · h264_nvenc · -c copy',
                'w': None, 'h': None, 'fps': None,
                'chunk_preset': None, 'chunk_crf': None, 'chunk_ab': '192k',
                'chunk_codec': ['-c:v', 'h264_nvenc', '-cq', '23'],
                'merge_reencode': False,
                'merge_preset': 'medium', 'merge_crf': '23', 'merge_ab': '192k'},
}



# 环境探测

class Env:
    ffmpeg = ffprobe = latex = dvisvgm = None
    has_x264 = has_aac = False
    has_stillimage = False
    can_link = can_symlink = can_replace = False
    cairosvg = pillow = False
    cpu = 1
    fs_type = '?'
    font_sans = None        # 最佳无衬线中文字体名
    font_serif = None       # 最佳衬线中文字体名
    font_mono = None        # 最佳等宽字体名
    fonts_zh = []           # 所有探测到的中文字体
    # 硬件编码
    has_qsv = False         # Intel Quick Sync Video 可用
    has_nvenc = False       # NVIDIA NVENC 可用
    nvenc_gpu_mem = 0       # NVIDIA 显存总量 (MB)
    hw_accel_active = None  # 当前实际使用的硬件编码器: 'qsv' / 'nvenc' / None

    @classmethod
    def probe(cls, workdir):
        cls.cpu = os.cpu_count() or 1
        cls.ffmpeg  = shutil.which('ffmpeg')
        cls.ffprobe = shutil.which('ffprobe')
        cls.latex   = shutil.which('latex')
        cls.dvisvgm = shutil.which('dvisvgm')
        try:
            import cairosvg; cls.cairosvg = True
        except Exception:
            pass
        try:
            import PIL; cls.pillow = True
        except Exception:
            pass
        if cls.ffmpeg:
            cls.has_x264 = cls._has_encoder('libx264')
            cls.has_aac  = cls._has_encoder('aac')
            cls.has_stillimage = cls._has_tune('stillimage')
        cls._probe_fonts()
        cls._probe_hw_encoders()
        wt = Path(workdir)
        try:
            wt.mkdir(parents=True, exist_ok=True)
            cls.fs_type = cls._fs_type(wt)
            cls._probe_atomic(wt)
        except Exception:
            pass
        return cls

    @classmethod
    def _probe_atomic(cls, wt):
        src, link = wt / '.p_src', wt / '.p_link'
        sym, tmp, dst = wt / '.p_sym', wt / '.p_tmp', wt / '.p_dst'
        try:
            src.write_bytes(b'x')
        except Exception:
            return
        if hasattr(os, 'link'):
            try: os.link(str(src), str(link)); cls.can_link = True
            except Exception: pass
        if hasattr(os, 'symlink'):
            try: os.symlink(src.name, str(sym)); cls.can_symlink = True
            except Exception: pass
        try:
            tmp.write_bytes(b'y')
            os.replace(str(tmp), str(dst))
            cls.can_replace = True
        except Exception:
            pass
        for p in (src, link, sym, tmp, dst):
            try:
                if p.exists() or p.is_symlink(): p.unlink()
            except Exception: pass

    @classmethod
    def _probe_fonts(cls):
        """探测系统可用的中文字体。
        优先级：显式指定 > 常见优质字体 > 第一个可用
        """
        # sans-serif 优先级（从最稳到最次）
        SANS_PRIORITY = [
            'Noto Sans CJK SC',
            'Source Han Sans CN',
            'Noto Sans SC',
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Droid Sans Fallback',
            'iQOO type',
        ]
        SERIF_PRIORITY = [
            'Noto Serif CJK SC',
            'Source Han Serif CN',
            'Noto Serif SC',
        ]
        MONO_PRIORITY = [
            'Noto Sans Mono CJK SC',
            'Source Han Mono SC',
        ]

        # 从 fc-list 拿到所有中文字体
        families = []
        try:
            rc, out, _ = safe_run(['fc-list', ':lang=zh', 'family'],
                                   capture_output=True, text=True, timeout=5)
            if rc == 0:
                for line in out.splitlines():
                    for name in line.split(','):
                        name = name.strip()
                        if name and name not in families:
                            families.append(name)
        except Exception:
            pass

        # 也从 fc-list 拿所有字体（含非中文的 serif / mono）
        all_families = []
        try:
            rc, out, _ = safe_run(['fc-list', ':', 'family'],
                                   capture_output=True, text=True, timeout=5)
            if rc == 0:
                for line in out.splitlines():
                    for name in line.split(','):
                        name = name.strip()
                        if name and name not in all_families:
                            all_families.append(name)
        except Exception:
            pass

        cls.fonts_zh = families

        def pick(priority, pool, fallback=None):
            for name in priority:
                if name in pool:
                    return name
            return fallback if pool else None

        cls.font_sans = pick(SANS_PRIORITY, families,
                              families[0] if families else None)
        cls.font_serif = pick(SERIF_PRIORITY, families + all_families,
                               cls.font_sans)
        cls.font_mono = pick(MONO_PRIORITY, all_families,
                              cls.font_sans)

    @classmethod
    def _probe_hw_encoders(cls):
        """探测硬件编码器可用性。
        QSV: 检查 ffmpeg 是否支持 h264_qsv，并测试能否初始化。
        NVENC: 检查 ffmpeg 是否支持 h264_nvenc/hevc_nvenc，
               并通过 nvidia-smi 确认显存 >= 512MB。
        """
        if not cls.ffmpeg:
            return

        # ---- QSV ----
        try:
            rc, out, _ = safe_run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                timeout=8)
            if rc == 0 and 'h264_qsv' in out:
                # 尝试初始化 QSV（有些机器编码器存在但设备不可用）
                rc2, _, err2 = safe_run(
                    ['ffmpeg', '-hide_banner', '-loglevel', 'error',
                     '-init_hw_device', 'qsv=hw', '-filter_hw_device', 'hw',
                     '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=0.1',
                     '-c:v', 'h264_qsv', '-f', 'null', '-'],
                    timeout=15)
                if rc2 == 0:
                    cls.has_qsv = True
        except Exception:
            pass

        # ---- NVENC ----
        try:
            if 'h264_nvenc' in out or 'hevc_nvenc' in out:
                # 查显存
                rc3, out3, _ = safe_run(
                    ['nvidia-smi', '--query-gpu=memory.total',
                     '--format=csv,noheader,nounits'],
                    timeout=8)
                if rc3 == 0 and out3.strip():
                    try:
                        mem = int(out3.strip().split('\n')[0].strip())
                        cls.nvenc_gpu_mem = mem
                        if mem >= 512:
                            # 尝试初始化 NVENC
                            rc4, _, _ = safe_run(
                                ['ffmpeg', '-hide_banner', '-loglevel', 'error',
                                 '-f', 'lavfi', '-i',
                                 'color=c=black:s=64x64:d=0.1',
                                 '-c:v', 'h264_nvenc', '-f', 'null', '-'],
                                timeout=15)
                            if rc4 == 0:
                                cls.has_nvenc = True
                    except (ValueError, IndexError):
                        pass
        except FileNotFoundError:
            pass  # nvidia-smi 不存在
        except Exception:
            pass

    @staticmethod
    def _has_encoder(enc):
        try:
            r = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                               capture_output=True, text=True, timeout=8)
            return enc in r.stdout
        except Exception: return False

    @staticmethod
    def _has_tune(tune):
        try:
            r = subprocess.run(['ffmpeg', '-hide_banner', '-h', 'encoder=libx264'],
                               capture_output=True, text=True, timeout=8)
            return tune in r.stdout
        except Exception: return False

    @staticmethod
    def _fs_type(p):
        for cmd in (['df', '-T', str(p)], ['stat', '-f', '-c', '%T', str(p)]):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
                if r.returncode != 0: continue
                out = r.stdout.strip()
                if cmd[0] == 'df':
                    lines = out.split('\n')
                    if len(lines) >= 2:
                        parts = lines[1].split()
                        if len(parts) >= 2: return parts[1]
                else:
                    return out or '?'
            except Exception: continue
        return '?'

    @classmethod
    def report(cls):
        m = lambda v: '✓' if v else '✗'
        return (f"环境能力\n"
                f"  CPU       {cls.cpu} 核\n"
                f"  ffmpeg    {m(cls.ffmpeg)} {cls.ffmpeg or ''}\n"
                f"  ffprobe   {m(cls.ffprobe)}\n"
                f"  libx264   {m(cls.has_x264)}\n"
                f"  aac       {m(cls.has_aac)}\n"
                f"  tune=stillimage {m(cls.has_stillimage)}\n"
                f"  latex     {m(cls.latex)}\n"
                f"  dvisvgm   {m(cls.dvisvgm)}\n"
                f"  cairosvg  {m(cls.cairosvg)}\n"
                f"  pillow    {m(cls.pillow)}\n"
                f"  numpy     {m(_HAS_NUMPY)}\n"
                f"  fs        {cls.fs_type}\n"
                f"  link      {m(cls.can_link)}   symlink {m(cls.can_symlink)}"
                f"   replace {m(cls.can_replace)}")

    @classmethod
    def require(cls, latex_needed=False, ffmpeg_needed=True):
        if ffmpeg_needed and not cls.ffmpeg:
            raise SystemExit("✗ 缺少 ffmpeg")
        if not cls.cairosvg:
            raise SystemExit("✗ 缺少 cairosvg")
        if latex_needed:
            miss = [t for t, v in (('latex', cls.latex),
                                   ('dvisvgm', cls.dvisvgm)) if not v]
            if miss:
                raise SystemExit(f"✗ 缺少 {', '.join(miss)}")



# 安全原语

def safe_link_or_copy(src, dst):
    s, d = str(src), str(dst)
    if hasattr(os, 'link'):
        try: os.link(s, d); return 'link'
        except OSError: pass
    if hasattr(os, 'symlink'):
        try:
            os.symlink(os.path.relpath(s, os.path.dirname(d)), d)
            return 'symlink'
        except OSError: pass
    try: shutil.copy2(s, d); return 'copy'
    except Exception: pass
    try:
        with open(s, 'rb') as fi: data = fi.read()
        with open(d, 'wb') as fo: fo.write(data)
        return 'rawrw'
    except Exception as e:
        print(f"  ⚠ link/copy/rawrw 全失败 {s} → {d}: {e}", file=sys.stderr)
        return 'fail'


def safe_replace(tmp, final):
    t, f = str(tmp), str(final)
    try: os.replace(t, f); return 'replace'
    except OSError: pass
    try: os.rename(t, f); return 'rename'
    except OSError: pass
    try: shutil.copy2(t, f); os.unlink(t); return 'copy'
    except Exception as e:
        print(f"  ⚠ replace 失败: {e}", file=sys.stderr)
        return 'fail'


def safe_rmtree(p, label=''):
    p = str(p)
    if not os.path.exists(p): return True
    try:
        shutil.rmtree(p, ignore_errors=True)
        return not os.path.exists(p)
    except Exception as e:
        print(f"  ⚠ 删除失败 {p}{' ('+label+')' if label else ''}: {e}",
              file=sys.stderr)
        return False


def _os_err(path, msg, label=''):
    tag = f" [{label}]" if label else ""
    print(f"  ✗ {msg}: {path}{tag}", file=sys.stderr)


def _safe_read_text(path, label=''):
    """读文本。失败打印友好信息并返回 None（不抛 Traceback）。"""
    try:
        return Path(path).read_text(encoding='utf-8')
    except FileNotFoundError:
        _os_err(path, '文件不存在', label)
    except IsADirectoryError:
        _os_err(path, '路径是目录，不是文件', label)
    except NotADirectoryError:
        _os_err(path, '父路径不是目录', label)
    except PermissionError:
        _os_err(path, '无权限读取', label)
    except OSError as e:
        _os_err(path, f'读取失败（{e.strerror or e}）', label)
    return None


def _safe_read_bytes(path, label=''):
    """读二进制。失败打印友好信息并返回 None。"""
    try:
        return Path(path).read_bytes()
    except FileNotFoundError:
        _os_err(path, '文件不存在', label)
    except IsADirectoryError:
        _os_err(path, '路径是目录，不是文件', label)
    except NotADirectoryError:
        _os_err(path, '父路径不是目录', label)
    except PermissionError:
        _os_err(path, '无权限读取', label)
    except OSError as e:
        _os_err(path, f'读取失败（{e.strerror or e}）', label)
    return None


def _safe_write_text(path, text, label=''):
    """写文本。失败打印友好信息并返回 False。"""
    try:
        Path(path).write_text(text, encoding='utf-8')
        return True
    except FileNotFoundError:
        _os_err(Path(path).parent, '目录不存在，无法写入', label)
    except IsADirectoryError:
        _os_err(path, '路径是目录，不是文件', label)
    except NotADirectoryError:
        _os_err(path, '父路径不是目录', label)
    except PermissionError:
        _os_err(path, '无权限写入', label)
    except OSError as e:
        _os_err(path, f'写入失败（{e.strerror or e}）', label)
    return False


def safe_run(cmd, **kw):
    kw.setdefault('capture_output', True)
    kw.setdefault('text', True)
    kw.setdefault('timeout', 600)
    try:
        r = subprocess.run(cmd, **kw)
        return r.returncode, r.stdout or '', r.stderr or ''
    except subprocess.TimeoutExpired: return 124, '', 'timeout'
    except FileNotFoundError as e: return 127, '', str(e)
    except Exception as e: return 1, '', f'{type(e).__name__}: {e}'



# Easing

def ease_apply(name, t):
    if not name or name in ('linear', 'none'): return t
    if name in ('ease', 'ease-in-out', 'smoothstep'): return t*t*(3-2*t)
    if name in ('ease-in', 'ease-in-quad'): return t*t
    if name in ('ease-out', 'ease-out-quad'): return 1-(1-t)**2
    if name == 'ease-in-cubic': return t**3
    if name == 'ease-out-cubic': return 1-(1-t)**3
    if name == 'ease-in-out-cubic':
        return 4*t**3 if t < 0.5 else 1-(-2*t+2)**3/2
    if name == 'ease-out-expo':
        return 1 if t >= 1 else 1-2**(-10*t)
    if name == 'back-out':
        c1, c3 = 1.70158, 2.70158
        return 1+c3*(t-1)**3+c1*(t-1)**2
    if name == 'back-in':
        c1, c3 = 1.70158, 2.70158
        return c3*t**3-c1*t**2
    return t



# XML 解析

def parse_xml(path, base_dir='.'):
    root = ET.parse(path).getroot()
    proj = {
        'fps': int(root.get('fps', 30)),
        'w': int(root.get('width', 1280)),
        'h': int(root.get('height', 720)),
        'duration': float(root.get('duration', 5)),
        'assets': {}, 'tracks': [], 'relations': [],
        'spaces': {},
        'base_dir': Path(base_dir),
        '_exp_encode': False, '_exp_merge_svg': False,
        '_exp_merge_static': False, '_merge_aggressive': 2,
        '_quality': 'fast',
        '_out_w': None, '_out_h': None,
    }

    # 🆕 解析顶层 <macros>
    macros = {}
    macros_el = root.find('macros')
    if macros_el is not None:
        for mac in macros_el:
            if mac.tag == 'macro':
                mid = mac.get('id')
                if mid:
                    macros[mid] = mac

    # 🆕 解析 <spaces>（支持 inherit="parent_id"）
    spaces_el = root.find('spaces')
    if spaces_el is not None:
        raw = {}
        for sp in spaces_el:
            if sp.tag == 'space3d':
                sid = sp.get('id')
                if sid:
                    raw[sid] = sp

        resolved = {}

        def _resolve(sid, chain=()):
            if sid in resolved:
                return resolved[sid]
            if sid not in raw:
                print(f"  ⚠ space3d 继承目标不存在: {sid}", file=sys.stderr)
                return None
            if sid in chain:
                print(f"  ⚠ space3d 循环继承: {' -> '.join(chain)} -> {sid}",
                      file=sys.stderr)
                return None
            sp = raw[sid]
            pid = sp.get('inherit')
            if pid:
                parent = _resolve(pid, chain + (sid,))
                if parent is not None:
                    sp = _merge_space_elem(parent, sp)
            resolved[sid] = sp
            return sp

        for sid in raw:
            sp = _resolve(sid)
            if sp is None:
                continue
            # 🆕 展开 space 里的 <use macro="..."/>
            _expand_macros_in_space(sp, macros)
            overlays = []
            for ov in sp.findall('overlay'):
                overlays.append({
                    'id': f"_ov_{sid}_{len(overlays)}",
                    'asset': ov.get('asset'),
                    'x': ov.get('x'), 'y': ov.get('y'),
                    'w': ov.get('w'), 'h': ov.get('h'),
                    'opacity': ov.get('opacity', '1.0'),
                    'rotate': ov.get('rotate', '0'),
                    'scale': ov.get('scale', '1.0'),
                    't0': ov.get('t0'), 't1': ov.get('t1'),
                    'fade-in': ov.get('fade-in'),
                    'fade-out': ov.get('fade-out'),
                })
            _ex = ET.tostring(sp, encoding='unicode')
            proj['spaces'][sid] = {
                'id': sid,
                'duration': float(sp.get('duration', 6)),
                'bg': sp.get('bg', '#0a0e1a'),
                'overlays': overlays,
                '_elem_xml': _ex,
                'mesh_count': _ex.count('<mesh '),
            }

    assets_el = root.find('assets')
    if assets_el is not None:
        for a in assets_el:
            ad = dict(a.attrib)
            if a.text and a.text.strip(): ad['content'] = a.text.strip()
            if ad.get('type') == 'svg' and 'content' not in ad and 'src' in ad:
                p = Path(base_dir) / ad['src']
                txt = _safe_read_text(p, label=f'asset {ad["id"]}')
                if txt is not None:
                    ad['content'] = txt
            if 'src' in ad and ad['type'] in ('audio', 'video', 'image'):
                ad['src'] = str((Path(base_dir) / ad['src']).resolve())
            # 【3D】保留 asset 的完整 XML（含子节点 camera/light/mesh）
            if ad.get('type') == '3d':
                ad['_elem_xml'] = ET.tostring(a, encoding='unicode')
            proj['assets'][ad['id']] = ad

    def parse_container(elem):
        c = dict(elem.attrib); c['_style'] = {}; c['_anims'] = []
        for child in elem:
            if child.tag == 'style': c['_style'].update(dict(child.attrib))
            elif child.tag == 'animate': c['_anims'].append(child)
            elif child.tag == 'content' and child.text: c['content'] = child.text.strip()
        return c

    tracks_el = root.find('tracks')
    if tracks_el is not None:
        for t in tracks_el:
            tr = {'id': t.get('id'), 'kind': t.get('kind', 'video'), 'clips': []}
            for c in t:
                clip = parse_container(c)
                clip['_elements'] = []; clip['_mask'] = None
                for child in c:
                    if child.tag == 'element':
                        clip['_elements'].append(parse_container(child))
                    elif child.tag == 'mask':
                        clip['_mask'] = parse_container(child)
                if clip.get('space'):
                    clip['_space'] = clip['space']
                    clip['asset'] = None
                tr['clips'].append(clip)
            proj['tracks'].append(tr)

    rel = root.find('relations')
    if rel is not None:
        for r in rel: proj['relations'].append(dict(r.attrib, kind=r.tag))
    return proj


def preflight(proj):
    missing = []
    for aid, a in proj['assets'].items():
        if a.get('type') in ('audio', 'video', 'image') and 'src' in a:
            if not Path(a['src']).exists():
                missing.append({'asset': aid, 'src': a['src'], 'kind': a['type']})
    return missing


def drop_missing_clips(proj, missing_assets):
    keep = {c['id'] for t in proj['tracks'] for c in t['clips']
            if c.get('asset') not in missing_assets}
    for tr in proj['tracks']:
        tr['clips'] = [c for c in tr['clips'] if c['id'] in keep]
    for rel in proj['relations']:
        for k in ('clip', 'from', 'to'):
            if k in rel and rel[k] not in keep: rel['_skip'] = True
    return proj


def clip_duration(proj, c):
    return float(c['duration']) if 'duration' in c else proj['duration']


def resolve(proj):
    clips = {c['id']: c for t in proj['tracks'] for c in t['clips']}
    transitions = {}
    for r in proj['relations']:
        if r.get('_skip'): continue
        if r['kind'] == 'transition':
            transitions[(r['from'], r['to'])] = {
                'duration': float(r['duration']),
                'effect': r.get('effect', 'fade')}
    proj['_transitions'] = transitions

    for r in proj['relations']:
        if r.get('_skip'): continue
        k = r['kind']
        try:
            if k == 'start':
                clips[r['clip']]['_start'] = float(r['at'])
            elif k == 'seq':
                a, b = clips[r['from']], clips[r['to']]
                gap = float(r.get('gap', 0))
                trans = transitions.get((r['from'], r['to']))
                extra = trans['duration'] if trans else 0
                b['_start'] = a['_start'] + clip_duration(proj, a) + gap - extra
            elif k == 'sync':
                clips[r['clip']]['_start'] = 60.0 / float(r['bpm']) * float(r['beat'])
            elif k == 'transition': pass
            else: print(f"  ⚠ 未知关系: {k}", file=sys.stderr)
        except Exception as e:
            print(f"  ⚠ 关系 {k} 失败: {e}", file=sys.stderr)

    for c in clips.values():
        c.setdefault('_start', 0.0)
        c['_end'] = c['_start'] + clip_duration(proj, c)

    for (fid, tid), trans in transitions.items():
        if fid not in clips or tid not in clips: continue
        a, b = clips[fid], clips[tid]
        if '_start' not in b or b['_start'] >= a['_end']:
            b['_start'] = a['_end'] - trans['duration']
            b['_end'] = b['_start'] + clip_duration(proj, b)
    return proj



# 关键帧 / expr

def interp(stops, prop, t):
    series = [(tt, float(d[prop]), e) for tt, d, e in stops if prop in d]
    if not series: return None
    series.sort(key=lambda s: s[0])
    if t <= series[0][0]: return series[0][1]
    if t >= series[-1][0]: return series[-1][1]
    for i in range(len(series) - 1):
        t0, v0, e0 = series[i]
        t1, v1, _ = series[i + 1]
        if t0 <= t <= t1:
            a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            a = ease_apply(e0, a)
            return v0 + (v1 - v0) * a
    return series[-1][1]


_SAFE_NS = {
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'asin': math.asin, 'acos': math.acos, 'atan': math.atan, 'atan2': math.atan2,
    'abs': abs, 'min': min, 'max': max, 'pow': pow, 'round': round,
    'pi': math.pi, 'tau': math.tau, 'e': math.e,
    'sqrt': math.sqrt, 'exp': math.exp, 'log': math.log,
    'floor': math.floor, 'ceil': math.ceil,
}


def _eval_expr(expr, t, props):
    ns = dict(_SAFE_NS); ns['t'] = t
    for k, v in props.items():
        try: ns[k] = float(v)
        except (ValueError, TypeError): pass
    return eval(expr, {'__builtins__': {}}, ns)


def _parse_anim(anim_elem):
    stops, exprs = [], []
    for child in anim_elem:
        if child.tag == 'keyframe':
            kt = float(child.get('t', 0))
            kp = {k: v for k, v in child.attrib.items() if k not in ('t', 'ease')}
            stops.append((kt, kp, child.get('ease', 'linear')))
        elif child.tag == 'expr':
            p, v = child.get('prop'), child.get('value')
            if p and v: exprs.append((p, v))
    return stops, exprs


def props_at_t(container, t):
    props = dict(container.get('_style', {}))
    for anim in container.get('_anims', []):
        stops, exprs = _parse_anim(anim)
        all_props = set()
        for _, p, _ in stops: all_props |= set(p.keys())
        for prop in all_props:
            v = interp(stops, prop, t)
            if v is not None: props[prop] = v
        for prop, expr in exprs:
            try: props[prop] = _eval_expr(expr, t, props)
            except Exception as e:
                print(f"  ⚠ expr {prop}={expr!r} 失败: {e}", file=sys.stderr)
    return props



# 静态段

def _props_equal(a, b, keys):
    for k in keys:
        va, vb = a.get(k, '__M__'), b.get(k, '__M__')
        sa = str(va).strip() if va is not None else '__M__'
        sb = str(vb).strip() if vb is not None else '__M__'
        if sa != sb: return False
    return True


def segment_clip(proj, clip):
    total = clip_duration(proj, clip)
    kfs = []; has_expr = [False]

    def collect(container):
        for anim in container.get('_anims', []):
            stops, exprs = _parse_anim(anim)
            if exprs: has_expr[0] = True
            for kt, kp, _ in stops:
                kfs.append({'t': kt, 'props': kp})

    collect(clip)
    for el in clip.get('_elements', []): collect(el)
    if clip.get('_mask'): collect(clip['_mask'])

    if has_expr[0]: return [(0.0, total, False)]

    kfs.sort(key=lambda x: x['t'])
    if not kfs: return [(0.0, total, True)]

    all_keys = set()
    for k in kfs: all_keys |= set(k['props'].keys())
    for prop in all_keys:
        last = None
        for k in kfs:
            if prop in k['props']: last = k['props'][prop]
            elif last is not None: k['props'][prop] = last

    segs = []
    if kfs[0]['t'] > 1e-6:
        init = clip.get('_style', {})
        keys = set(init) | set(kfs[0]['props'])
        segs.append((0.0, kfs[0]['t'], _props_equal(init, kfs[0]['props'], keys)))

    for i in range(len(kfs) - 1):
        t0, t1 = kfs[i]['t'], kfs[i + 1]['t']
        if t1 - t0 < 1e-6: continue
        keys = set(kfs[i]['props']) | set(kfs[i + 1]['props'])
        segs.append((t0, t1, _props_equal(kfs[i]['props'], kfs[i + 1]['props'], keys)))

    if kfs[-1]['t'] < total - 1e-6:
        segs.append((kfs[-1]['t'], total, True))

    if not segs: return [(0.0, total, True)]

    merged = []
    for s in segs:
        if merged and merged[-1][2] and s[2]:
            merged[-1] = (merged[-1][0], s[1], True)
        else: merged.append(s)
    return merged


def is_pure_static(proj, clip):
    return all(s[2] for s in segment_clip(proj, clip))



# LaTeX

_LATEX_CACHE = {}


def latex_png(latex_str, fontsize=48, workdir=None):
    key = (latex_str, fontsize)
    if key in _LATEX_CACHE: return _LATEX_CACHE[key]
    workdir = Path(workdir or './tmp/gappt_work')
    d = workdir / 'latex'
    d.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(f"{latex_str}|{fontsize}".encode()).hexdigest()[:12]
    tex, dvi, svg = d / f"{h}.tex", d / f"{h}.dvi", d / f"{h}.svg"
    pt = max(12, int(fontsize * 0.75))
    tex.write_text(
        "\\documentclass[preview,border=2pt]{standalone}\n"
        "\\usepackage{amsmath,amssymb}\n\\begin{document}\n"
        f"{{\\fontsize{{{pt}}}{{{int(pt*1.2)}}}\\selectfont ${latex_str}$}}\n"
        "\\end{document}\n")
    rc, out, _ = safe_run(['latex', '-interaction=nonstopmode', '-halt-on-error',
                           '-output-directory', str(d), str(tex)],
                          cwd=str(d), timeout=60)
    if rc != 0: raise RuntimeError(f"latex 失败: {out[-800:]}")
    rc, _, err = safe_run(['dvisvgm', '--no-fonts', '--exact-bbox',
                           str(dvi), '-o', str(svg)], timeout=60)
    if rc != 0: raise RuntimeError(f"dvisvgm 失败: {err[-400:]}")
    import cairosvg
    from PIL import Image
    png_bytes = cairosvg.svg2png(url=str(svg))
    img = Image.open(BytesIO(png_bytes))
    result = (base64.b64encode(png_bytes).decode(), img.size)
    _LATEX_CACHE[key] = result
    return result



# SVG 生成

def inject_draw(svg_inner, draw):
    if draw is None: return svg_inner
    try: d = float(draw)
    except (ValueError, TypeError): return svg_inner
    if d >= 1: return svg_inner
    offset = 1 - d
    def repl(m):
        name, attrs, tail = m.group(1), m.group(2), m.group(3)
        if re.search(r'\bstroke-dashoffset\s*=', attrs):
            attrs = re.sub(r'\bstroke-dashoffset\s*=\s*"[^"]*"',
                           f'stroke-dashoffset="{offset}"', attrs)
        else:
            attrs = f'{attrs} stroke-dasharray="1" stroke-dashoffset="{offset}"'
        return f'<{name}{attrs}{tail}'
    return re.sub(r'<(path)(\s+[^>]*?)(/?>)', repl, svg_inner, flags=re.I)


_CAIRO_CJK_CACHE = {'done': False, 'family': None}


def _probe_cairo_cjk_family():
    """探测 cairo 真正能渲染中文的 font-family 名。

    termux 环境下 fc-match 命令行能匹配到 Noto CJK，但 cairosvg 内部
    走的是不同的 fontconfig 视图，可能匹配不到 → 中文变方块。
    这里直接让 cairosvg 渲染一个"中"字，用 ink/bbox 比例判断是
    真字形（笔画分散，比例 ~0.35-0.55）还是实心方块（比例 > 0.85）。
    """
    if _CAIRO_CJK_CACHE['done']:
        return _CAIRO_CJK_CACHE['family']
    _CAIRO_CJK_CACHE['done'] = True

    if not Env.cairosvg or not _HAS_NUMPY:
        return None

    try:
        import cairosvg
        from io import BytesIO as _B
        from PIL import Image as _I
        import numpy as _np
    except Exception:
        return None

    # 优先测 Env.font_sans（fc-list 认定的中文首选），再测常见名
    cands = []
    if Env.font_sans:
        cands.append(Env.font_sans)
    for c in ('Noto Sans CJK SC', 'Source Han Sans CN', 'Noto Sans SC',
              'Droid Sans Fallback', 'WenQuanYi Micro Hei',
              'WenQuanYi Zen Hei', 'Noto Sans CJK JP', 'Noto Sans CJK TC',
              'Noto Sans CJK HK', 'Noto Sans Mono CJK SC', 'Noto Sans'):
        if c not in cands:
            cands.append(c)
    for f in (Env.fonts_zh or []):
        if f not in cands:
            cands.append(f)

    for fam in cands:
        try:
            esc = fam.replace('&', '&amp;').replace('"', '&quot;')
            svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
                   f'width="120" height="120">'
                   f'<text x="60" y="90" font-family="{esc}" '
                   f'font-size="80" text-anchor="middle" fill="black">'
                   f'中</text></svg>')
            png = cairosvg.svg2png(bytestring=svg.encode(),
                                    output_width=120, output_height=120,
                                    background_color='white')
            arr = _np.array(_I.open(_B(png)).convert('L'))
            m = arr < 128
            n = int(m.sum())
            if n < 300:
                continue
            ys, xs = _np.where(m)
            bb = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
            ratio = n / bb if bb > 0 else 0
            # "中" 正常 ~0.35-0.55；方块 ~0.85-1.0
            if 0.15 < ratio < 0.75:
                _CAIRO_CJK_CACHE['family'] = fam
                print(f"  · cairo CJK 探测: 使用 '{fam}'", file=sys.stderr)
                return fam
        except Exception:
            continue

    print("  ⚠ cairo CJK 探测: 所有候选都渲染不出中文"
          "（cairo 回退路径可能显示方块）", file=sys.stderr)
    return None


def _resolve_font_family(declared):
    """把泛型字体声明映射到系统实际可用的字体名。
    declared 可能是 'system-ui, sans-serif' 或 'Noto Sans CJK SC'。

    优先使用 _probe_cairo_cjk_family() 探测到的 cairo 能识别字体
    —— termux 等环境下 fc-list 与 cairo 的字体视图可能不一致。
    """
    safe = _probe_cairo_cjk_family()

    if not declared:
        return safe or Env.font_sans or 'sans-serif'
    d = str(declared).strip()
    lower = d.lower()
    if ('system-ui' in lower or d == 'sans-serif'
            or 'system' == lower or d.startswith('system-ui')):
        return safe or Env.font_sans or 'sans-serif'
    if d == 'serif' or 'georgia' in lower or 'times' in lower:
        return safe or Env.font_serif or Env.font_sans or 'serif'
    if 'monospace' in lower or 'courier' in lower or 'consol' in lower:
        return safe or Env.font_mono or Env.font_sans or 'monospace'
    # 用户显式指定字体：如果探测出 cairo 能用别的字体渲染中文，
    # 就用探测结果（否则 cairo 路径会显示方块）
    if safe and safe != d:
        return safe
    return d


def _render_3d_content(proj, container, t, asset):
    """渲染 type='3d' 的 asset 为 SVG <image> 元素"""
    try:
        from scene3d_embed import render_3d_frame
    except ImportError as e:
        print(f"  ⚠ [3d] 未加载渲染器: {e}", file=sys.stderr)
        return ''

    W, H = proj['w'], proj['h']
    props = props_at_t(container, t)

    try:
        x = float(props.get('x', W/2))
        y = float(props.get('y', H/2))
        opacity = float(props.get('opacity', 1))
        rotate = float(props.get('rotate', 0))
        scale = float(props.get('scale', 1))
    except (ValueError, TypeError):
        x, y, opacity, rotate, scale = W/2, H/2, 1, 0, 1

    # 3D 渲染尺寸
    try:
        rw = int(float(props.get('w', W)))
    except (ValueError, TypeError):
        rw = W
    try:
        rh = int(float(props.get('h', H)))
    except (ValueError, TypeError):
        rh = H

    # 从 _elem_xml 恢复 Element
    elem_xml = asset.get('_elem_xml')
    if not elem_xml:
        return ''
    try:
        elem = ET.fromstring(elem_xml)
    except Exception as e:
        print(f"  ⚠ [3d] XML 解析失败: {e}", file=sys.stderr)
        return ''

    base_dir = proj.get('base_dir', Path('.'))
    try:
        png_bytes = render_3d_frame(elem, t, rw, rh,
                                     base_dir=str(base_dir),
                                     duration=clip_duration(proj, container))
    except Exception as e:
        print(f"  ⚠ [3d] 渲染失败: {e}", file=sys.stderr)
        return f'<g transform="translate({x} {y})" opacity="0"></g>'

    b64 = base64.b64encode(png_bytes).decode()
    transform = f"translate({x} {y}) rotate({rotate}) scale({scale})"
    return (f'<g transform="{transform}" opacity="{opacity}">'
            f'<image xlink:href="data:image/png;base64,{b64}" '
            f'href="data:image/png;base64,{b64}" '
            f'x="{-rw/2}" y="{-rh/2}" width="{rw}" height="{rh}"/>'
            f'</g>')


def _render_content(proj, container, t, asset=None):
    W, H = proj['w'], proj['h']
    props = props_at_t(container, t)
    try:
        x = float(props.get('x', W/2))
        y = float(props.get('y', H/2))
        opacity = float(props.get('opacity', 1))
        rotate = float(props.get('rotate', 0))
        scale = float(props.get('scale', 1))
    except (ValueError, TypeError):
        x, y, opacity, rotate, scale = W/2, H/2, 1, 0, 1
    draw = props.get('draw')
    transform = f"translate({x} {y}) rotate({rotate}) scale({scale})"
    if not asset: return f'<g transform="{transform}" opacity="{opacity}"></g>'
    t_type = asset.get('type', 'svg')

    if t_type == '3d':
        return _render_3d_content(proj, container, t, asset)

    if t_type == 'text':
        content = escape(asset.get('content', ''))
        size = float(props.get('font-size', 96))
        color = props.get('color', 'white')
        family = _resolve_font_family(props.get('font-family', 'sans-serif'))
        weight = props.get('font-weight', 'normal')
        align = props.get('align', 'center')
        anchor = {'left':'start','center':'middle','right':'end'}.get(align, 'middle')
        return (f'<text x="0" y="0" font-family="{family}" font-size="{size}" '
                f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" '
                f'dominant-baseline="middle" transform="{transform}" '
                f'opacity="{opacity}">{content}</text>')

    if t_type == 'latex':
        try:
            fontsize = float(props.get('font-size', 40))
            b64, (iw, ih) = latex_png(asset.get('content', ''), fontsize,
                                       proj.get('workdir'))
            return (f'<g transform="{transform}" opacity="{opacity}">'
                    f'<image xlink:href="data:image/png;base64,{b64}" '
                    f'href="data:image/png;base64,{b64}" '
                    f'x="{-iw/2}" y="{-ih/2}" width="{iw}" height="{ih}"/></g>')
        except Exception as e:
            print(f"  ⚠ LaTeX 失败: {e}", file=sys.stderr)
            return f'<g transform="{transform}" opacity="{opacity}"></g>'

    if t_type == 'svg':
        inner = asset.get('content', '')
        m = re.search(r'<svg[^>]*>(.*)</svg>', inner, re.S)
        if m: inner = m.group(1)
        try: inner = inject_draw(inner, draw)
        except Exception as e: print(f"  ⚠ draw 注入失败: {e}", file=sys.stderr)
        return f'<g transform="{transform}" opacity="{opacity}">{inner}</g>'

    if t_type == 'image':
        data = _safe_read_bytes(asset.get('src', ''),
                                 label=f"asset {asset.get('id', '?')}")
        if data is None:
            return ''
        ext = Path(asset['src']).suffix.lstrip('.').lower()
        mime = {'png':'image/png','jpg':'image/jpeg','jpeg':'image/jpeg',
                'gif':'image/gif','webp':'image/webp'}.get(ext, 'image/png')
        b64 = base64.b64encode(data).decode()
        iw = float(props.get('w', 640))
        ih = float(props.get('h', 360))
        return (f'<g transform="{transform}" opacity="{opacity}">'
                f'<image xlink:href="data:{mime};base64,{b64}" '
                f'href="data:{mime};base64,{b64}" '
                f'x="{-iw/2}" y="{-ih/2}" width="{iw}" height="{ih}"/></g>')
    return ''


def frame_svg_inner(proj, clip, asset, t, mask_prefix=''):
    W, H = proj['w'], proj['h']
    elements = clip.get('_elements', [])
    if elements:
        parts = []
        for el in elements:
            el_asset = None
            if el.get('asset'): el_asset = proj['assets'].get(el['asset'])
            if el_asset is None and 'content' in el:
                el_asset = {'type': 'svg', 'content': el['content']}
            if el_asset is None: continue
            parts.append(_render_content(proj, el, t, el_asset))
        props = props_at_t(clip, t)
        try:
            px = float(props.get('x', W/2))
            py = float(props.get('y', H/2))
            protate = float(props.get('rotate', 0))
            pscale = float(props.get('scale', 1))
            pop = float(props.get('opacity', 1))
        except (ValueError, TypeError):
            px, py, protate, pscale, pop = W/2, H/2, 0, 1, 1
        body = (f'<g transform="translate({px} {py}) '
                f'rotate({protate}) scale({pscale})" '
                f'opacity="{pop}">{"".join(parts)}</g>')
    else:
        body = _render_content(proj, clip, t, asset)

    mask = clip.get('_mask')
    if mask:
        mprops = props_at_t(mask, t)
        mcontent = mask.get('content', '')
        for k, v in mprops.items():
            mcontent = mcontent.replace('{' + k + '}', str(v))
        mid = f"{mask_prefix}mask_{clip['id']}"
        body = (f'<defs><mask id="{mid}" maskUnits="userSpaceOnUse" '
                f'x="0" y="0" width="{W}" height="{H}">'
                f'<rect x="0" y="0" width="{W}" height="{H}" fill="black"/>'
                f'{mcontent}</mask></defs>'
                f'<g mask="url(#{mid})">{body}</g>')
    return body


def frame_svg(proj, clip, asset, t):
    W, H = proj['w'], proj['h']
    body = frame_svg_inner(proj, clip, asset, t)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">{body}</svg>')


def frame_svg_merged(proj, clips_assets, t_abs):
    W, H = proj['w'], proj['h']
    parts = []
    for i, (clip, asset) in enumerate(clips_assets):
        t_clip = t_abs - clip['_start']
        try:
            inner = frame_svg_inner(proj, clip, asset, t_clip,
                                     mask_prefix=f"m{i}_")
            parts.append(inner)
        except Exception as e:
            print(f"  ⚠ merged inner {clip['id']} 失败: {e}", file=sys.stderr)
    body = "".join(parts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">{body}</svg>')



def _inline_overlay_to_clip(ov, space_seg, proj):
    """
    把 <space3d> 里的 <overlay asset="xxx" ...> 转成标准 clip 结构。
    时间窗口 = space_seg 的时间窗口（相对全局）。
    """
    W, H = proj.get('_out_w') or proj['w'], proj.get('_out_h') or proj['h']
    style = {}
    if ov.get('x') is not None: style['x'] = ov['x']
    else: style['x'] = str(W / 2)
    if ov.get('y') is not None: style['y'] = ov['y']
    else: style['y'] = str(H / 2)
    if ov.get('w') is not None: style['w'] = ov['w']
    if ov.get('h') is not None: style['h'] = ov['h']
    style['opacity'] = ov.get('opacity', '1.0')
    if ov.get('rotate'): style['rotate'] = ov['rotate']
    if ov.get('scale'): style['scale'] = ov['scale']

    # t0/t1 是相对 space 本地时间，转成全局时间
    space_start = space_seg['start']
    space_dur = space_seg['end'] - space_seg['start']
    t0 = float(ov['t0']) if ov.get('t0') is not None else 0.0
    t1 = float(ov['t1']) if ov.get('t1') is not None else space_dur
    # 严格裁剪到 space 内
    t0 = max(0.0, min(t0, space_dur))
    t1 = max(t0, min(t1, space_dur))

    return {
        'id': ov['id'],
        'asset': ov['asset'],
        '_style': style,
        '_anims': [],
        '_elements': [],
        '_mask': None,
        '_start': space_start + t0,
        '_end': space_start + t1,
        '_is_inline_overlay': True,
    }


def _merge_space_elem(parent, child):
    """合并 parent space3d 和 child space3d（child 覆盖 parent）。

    规则：
      · 属性：child 有则覆盖（inherit 属性会被清除）
      · camera：child 有则整体替换
      · light/mesh/overlay：按 id 覆盖；child 独有的追加
      · <remove id="X"/>：从合并结果里移除 id=X 的元素
    """
    import copy
    merged = copy.deepcopy(parent)
    for k, v in child.attrib.items():
        merged.set(k, v)
    merged.attrib.pop('inherit', None)

    remove_ids = {r.get('id') for r in child.findall('remove') if r.get('id')}
    if remove_ids:
        for c in list(merged):
            if c.get('id') in remove_ids:
                merged.remove(c)

    child_cam = child.find('camera')
    if child_cam is not None:
        for c in list(merged):
            if c.tag == 'camera':
                merged.remove(c)
        merged.append(copy.deepcopy(child_cam))

    for tag in ('light', 'mesh', 'overlay'):
        child_items = {e.get('id'): e for e in child.findall(tag) if e.get('id')}
        if not child_items:
            continue
        for c in list(merged):
            if c.tag == tag and c.get('id') in child_items:
                merged.remove(c)
        for e in child.findall(tag):
            merged.append(copy.deepcopy(e))

    # 🆕 <use>：同 id 覆盖，id 不同的追加
    # （展开成 mesh 前它们是"引用"，语义上和 mesh 一致：同名替换，新增追加）
    child_uses = {e.get('id'): e for e in child.findall('use') if e.get('id')}
    for c in list(merged):
        if c.tag == 'use' and c.get('id') in child_uses:
            merged.remove(c)
    for e in child.findall('use'):
        merged.append(copy.deepcopy(e))

    return merged


def _build_inline_overlays_for_chunk(proj, chunk):
    """
    给定一个 3D chunk，收集它包含的 space 的所有内联 overlay，
    转成 clip 列表（时间已转成全局时间）。
    """
    space_id = chunk.get('space_id')
    if not space_id:
        return []
    sp = proj['spaces'].get(space_id)
    if not sp:
        return []
    # 用 chunk 的时间窗口作为 space_seg
    space_seg = {'start': chunk['start'], 'end': chunk['end']}
    clips = []
    for ov in sp.get('overlays', []):
        if not ov.get('asset'):
            continue
        clips.append(_inline_overlay_to_clip(ov, space_seg, proj))
    return clips



# 单帧渲染

def _resolve_video_bg(proj, t_abs):
    for tr in proj['tracks']:
        if tr['kind'] != 'video': continue
        for c in tr['clips']:
            if c['_start'] <= t_abs < c['_end']:
                a = proj['assets'].get(c.get('asset')) if c.get('asset') else None
                if a and a.get('type') == 'color': return ('color', a.get('value', '#000'), 0)
                if a and a.get('type') == 'video' and 'src' in a:
                    return ('video', a['src'], t_abs - c['_start'])
                if a and a.get('type') == 'image' and 'src' in a:
                    return ('image', a['src'], 0)
    return ('color', '#000', 0)


def render_single_frame(proj, t_abs, out_base, fmt='both'):
    W, H = proj['w'], proj['h']
    parts = []
    bg = _resolve_video_bg(proj, t_abs)
    if bg[0] == 'color':
        parts.append(f'<rect width="{W}" height="{H}" fill="{bg[1]}"/>')
    elif bg[0] == 'image':
        data = _safe_read_bytes(bg[1], label='background image')
        if data is None:
            parts.append(f'<rect width="{W}" height="{H}" fill="#000"/>')
        else:
            ext = Path(bg[1]).suffix.lstrip('.').lower()
            mime = {'png':'image/png','jpg':'image/jpeg','jpeg':'image/jpeg',
                    'gif':'image/gif','webp':'image/webp'}.get(ext, 'image/png')
            b64 = base64.b64encode(data).decode()
            parts.append(f'<image xlink:href="data:{mime};base64,{b64}" '
                         f'href="data:{mime};base64,{b64}" '
                         f'width="{W}" height="{H}"/>')
    elif bg[0] == 'video':
        try:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            tmp.close()
            subprocess.run(['ffmpeg', '-y', '-ss', str(bg[2]), '-i', bg[1],
                            '-vframes', '1', '-f', 'image2', tmp.name],
                           capture_output=True, timeout=20)
            data = Path(tmp.name).read_bytes(); os.unlink(tmp.name)
            b64 = base64.b64encode(data).decode()
            parts.append(f'<image xlink:href="data:image/png;base64,{b64}" '
                         f'href="data:image/png;base64,{b64}" '
                         f'width="{W}" height="{H}"/>')
        except Exception:
            parts.append(f'<rect width="{W}" height="{H}" fill="#000"/>')

    for tr in proj['tracks']:
        if tr['kind'] != 'overlay': continue
        for c in tr['clips']:
            if not (c['_start'] <= t_abs < c['_end']): continue
            t_clip = t_abs - c['_start']
            a = proj['assets'].get(c.get('asset')) if c.get('asset') else None
            try:
                inner = frame_svg_inner(proj, c, a, t_clip,
                                         mask_prefix=f"f_{c['id']}_")
                parts.append(inner)
            except Exception as e:
                print(f"  ⚠ overlay {c['id']} 失败: {e}", file=sys.stderr)

    body = "".join(parts)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">{body}</svg>')

    out_base = Path(out_base); written = []
    if fmt in ('svg', 'both'):
        p = out_base.with_suffix('.svg')
        p.write_text(svg, encoding='utf-8')
        print(f"  ✓ {p}"); written.append(str(p))
    if fmt in ('png', 'both'):
        try:
            p = out_base.with_suffix('.png')
            ow = proj.get('_out_w') or W
            oh = proj.get('_out_h') or H
            # 走 _render_png：GPU 优先，失败才回退 cairo
            # （单帧模式原本硬编码 cairosvg，在 termux 下会因 fontconfig
            #   fallback 到无中文字体而渲染成方块）
            _render_png(svg.encode('utf-8'), str(p), ow, oh)
            print(f"  ✓ {p}"); written.append(str(p))
        except Exception as e:
            print(f"  ⚠ PNG 输出失败: {e}", file=sys.stderr)
    return written



# 原子渲染

def _svg_gpu_level(svg_bytes, force=False):
    """返回 0=cairo / 1=basic / 2=full"""
    if not _HAS_GPU_2D:
        return 0
    env = os.environ.get('XMLVE_2D', '').lower()
    if env == 'cairo':
        return 0
    if env == 'force':
        force = True
    # 字体字重探测：系统缺 face 时回退 cairo
    if not force and not _check_weight_ok(svg_bytes):
        return 0
    try:
        from scene2d import svg_gpu_level
        lv = svg_gpu_level(svg_bytes)
    except Exception:
        lv = 0
    if lv == 0 and force:
        lv = 1  # 强制尝试
    return lv


def _svg_gpu_capable(svg_bytes):
    """向后兼容"""
    return _svg_gpu_level(svg_bytes) > 0


# 全局计时累加（跨帧）
_GPU_TIMES = {'decode': 0.0, 'render': 0.0, 'encode': 0.0, 'write': 0.0, 'n': 0}

# space3d 纹理预处理缓存：asset_id → data URI 字符串。
# 1200 座建筑引用同一 asset → cairosvg 只跑一次，且拿到相同的字符串对象，
# 让 scene3d 侧 upload_texture_by_key 自然去重。
_TEX_URI_CACHE = {}


def _expand_macros_in_space(sp_elem, macros):
    """就地展开 space3d 里的 <use macro="..."/> 元素。

    规则：
      · 每个 <use> 独立展开 → mesh id = f"{use_id}_{macro_mesh_id}"
      · pos 叠加：macro mesh 的 pos + use 的 pos
      · deepcopy 保留所有子节点（含 <animate>），相同动画自然套到不同元素
      · 展开后 <use> 从树中移除，不影响其他元素
    """
    if not macros:
        return
    uses = [c for c in list(sp_elem) if c.tag == 'use']
    if not uses:
        return
    auto_n = 0
    for use in uses:
        macro_id = use.get('macro')
        macro_elem = macros.get(macro_id)
        if macro_elem is None:
            print(f"  ⚠ 宏不存在: {macro_id}", file=sys.stderr)
            sp_elem.remove(use)
            continue
        use_id = use.get('id')
        if not use_id:
            use_id = f"_auto{auto_n}"
            auto_n += 1
        try:
            ux, uy, uz = [float(x) for x in
                          (use.get('pos', '0 0 0')).split()[:3]]
        except (ValueError, AttributeError):
            ux = uy = uz = 0.0
        for m in macro_elem.findall('mesh'):
            new_m = copy.deepcopy(m)
            old_id = new_m.get('id', 'm')
            new_m.set('id', f"{use_id}_{old_id}")
            try:
                mx, my, mz = [float(x) for x in
                              (new_m.get('pos', '0 0 0')).split()[:3]]
            except (ValueError, AttributeError):
                mx = my = mz = 0.0
            new_m.set('pos', f"{mx + ux:g} {my + uy:g} {mz + uz:g}")
            sp_elem.append(new_m)
        sp_elem.remove(use)


def _render_png_gpu(svg_bytes, final_path, W, H):
    """GPU 路径：SVG → Raster2D → PNG (fast encode)"""
    import time as _t
    tmp = final_path + '.tmp'
    try:
        t0 = _t.perf_counter()
        svg_str = svg_bytes.decode('utf-8')
        t1 = _t.perf_counter()
        arr = _svg_to_raster2d(svg_str, W, H)
        t2 = _t.perf_counter()
        from PIL import Image as _PILImage
        # compress_level=1：几乎不压缩，快 5~8 倍
        _PILImage.fromarray(arr, 'RGBA').save(
            tmp, format='PNG', optimize=False, compress_level=1)
        t3 = _t.perf_counter()
        safe_replace(tmp, final_path)
        t4 = _t.perf_counter()
        # 累加统计
        _GPU_TIMES['decode'] += t1 - t0
        _GPU_TIMES['render'] += t2 - t1
        _GPU_TIMES['encode'] += t3 - t2
        _GPU_TIMES['write']  += t4 - t3
        _GPU_TIMES['n'] += 1
        return True
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass
        print(f"  ⚠ [2d-gpu] {e}", file=sys.stderr)
        return False


def _render_png(svg_bytes, final_path, W, H):
    """优先 GPU，失败回退 cairo。输出格式由 final_path 后缀决定 (.png/.bmp)"""
    ext = Path(final_path).suffix.lower() or _FRAME_EXT
    if ext not in ('.png', '.bmp'):
        ext = '.png'

    # 1. GPU 路径
    if _svg_gpu_capable(svg_bytes):
        try:
            from PIL import Image as _PILImage
            arr = _svg_to_raster2d(svg_bytes.decode('utf-8'), W, H)
            if ext == '.bmp':
                _PILImage.fromarray(arr, 'RGBA').save(final_path, 'BMP')
            else:
                _PILImage.fromarray(arr, 'RGBA').save(
                    final_path, 'PNG', optimize=False, compress_level=1)
            _GPU_2D_STATS['gpu'] += 1
            return
        except Exception as e:
            print(f"  ⚠ [2d-gpu] {e}", file=sys.stderr)
            # 落到 cairo

    # 2. cairo 回退
    import cairosvg
    if ext == '.bmp':
        # cairo 只输出 PNG，先到临时，再转 BMP
        tmp_png = final_path + '.tmp.png'
        cairosvg.svg2png(bytestring=svg_bytes, write_to=tmp_png,
                         output_width=W, output_height=H)
        from PIL import Image as _PILImage
        with _PILImage.open(tmp_png) as im:
            im.convert('RGBA').save(final_path, 'BMP')
        try:
            os.unlink(tmp_png)
        except Exception:
            pass
    else:
        cairosvg.svg2png(bytestring=svg_bytes, write_to=final_path,
                         output_width=W, output_height=H)
    _GPU_2D_STATS['cairo'] += 1




def _render_png_cairo_only(svg_bytes, final_path, W, H):
    """强制走 cairo，不尝试 GPU 路径。

    用于 --exp-autocairo 模式：长静止段的首帧用 cairo 渲染，
    让画面跟 SVG 规范 100% 一致（GPU 路径 PIL/freetype hinting 有
    ±1px 级别差异，cairo 的 freetype hinting 更接近原生）。
    """
    import cairosvg
    cairosvg.svg2png(bytestring=svg_bytes, write_to=final_path,
                     output_width=W, output_height=H)


def _render_task(args):
    svg_bytes, path, W, H = args
    try:
        _render_png(svg_bytes, path, W, H)
        return True
    except Exception as e:
        print(f"  ⚠ 帧渲染失败 {path}: {e}", file=sys.stderr)
        return False



# 分块

def _estimate_chunk_weight(proj, t0, t1):
    """估算 [t0, t1) 时间段的渲染复杂度：mesh 数 + overlay element 数。"""
    w = 0
    for tr in proj['tracks']:
        kind = tr['kind']
        if kind == 'video':
            for c in tr['clips']:
                if c['_end'] <= t0 or c['_start'] >= t1: continue
                if c.get('_space'):
                    sp = proj['spaces'].get(c['_space'])
                    w += (sp.get('mesh_count', 0) if sp else 0)
                else:
                    w += 1
        elif kind == 'overlay':
            for c in tr['clips']:
                if c['_end'] <= t0 or c['_start'] >= t1: continue
                w += len(c.get('_elements', [])) or 1
    return w


def _split_oversized(proj, chunks, max_weight):
    """weight 超上限的 video chunk 二次切分。"""
    result = []
    for ch in chunks:
        if ch['type'] != 'video':
            result.append(ch); continue
        w = _estimate_chunk_weight(proj, ch['start'], ch['end'])
        if w <= max_weight:
            result.append(ch); continue
        n = int(w / max_weight) + 1
        dur = ch['end'] - ch['start']
        seg = dur / n
        for k in range(n):
            nc = {**ch, 'video_ids': list(ch['video_ids'])}
            nc['start'] = ch['start'] + k * seg
            nc['end'] = ch['start'] + (k + 1) * seg
            result.append(nc)
    return result


def _merge_small(proj, chunks, max_weight, max_sec):
    """相邻小 video chunk 合并，减少碎片。"""
    result = []
    for ch in chunks:
        if (result and ch['type'] == 'video'
                and result[-1]['type'] == 'video'
                and ch['end'] - result[-1]['start'] <= max_sec):
            w = _estimate_chunk_weight(proj, result[-1]['start'], ch['end'])
            if w <= max_weight:
                result[-1]['end'] = ch['end']
                for vid in ch['video_ids']:
                    if vid not in result[-1]['video_ids']:
                        result[-1]['video_ids'].append(vid)
                continue
        result.append({**ch, 'video_ids': list(ch['video_ids'])})
    return result


def build_chunks(proj, max_chunk_sec=12.0, time_range=None,
                 auto_split_sec=None,
                 max_weight=400, merge_sec=18.0):
    """切 chunk 规则（v1.4+）：
        · 3D space clip 每个单独成 chunk（type='space3d'）
        · 2D clip 按 max_chunk_sec 切
        · xfade 不再阻止切块（时间轴已在 resolve 里联合计算）
        · 相邻 chunk 之间有 transition 的，在合并阶段用 xfade 拼
    """
    vclips = [c for t in proj['tracks'] if t['kind'] == 'video'
              for c in t['clips']]
    vclips.sort(key=lambda c: c['_start'])
    if not vclips:
        return [{'idx': 0, 'start': 0.0, 'end': proj['duration'],
                 'video_ids': [], 'transitions': [], 'type': 'video'}]

    # 记录每个 clip 与下一个 clip 之间是否有 transition
    def _xfade_between(a, b):
        return (a['id'], b['id']) in proj['_transitions']

    chunks = []
    cur = None

    for i, c in enumerate(vclips):
        is_space = bool(c.get('_space'))
        prev = vclips[i - 1] if i > 0 else None

        if is_space:
            # 3D space：切断当前 chunk，自己独立成 chunk
            if cur is not None:
                chunks.append(cur)
                cur = None
            chunks.append({
                'start': c['_start'],
                'end': c['_end'],
                'video_ids': [],
                'transitions': [],
                'type': 'space3d',
                'space_id': c['_space'],
                'space_clip_id': c['id'],
            })
            continue

        clip_start = c['_start']
        clip_end = c['_end']
        clip_span = clip_end - clip_start

        # auto_split：单个 clip 超长 → 直接切 N 段独立 chunk
        if auto_split_sec is not None and clip_span > auto_split_sec * 1.5:
            if cur is not None:
                chunks.append(cur)
                cur = None
            n_seg = int(clip_span / auto_split_sec + 0.999)
            seg_len = clip_span / n_seg
            for k in range(n_seg):
                chunks.append({
                    'start': clip_start + k * seg_len,
                    'end': min(clip_end, clip_start + (k + 1) * seg_len),
                    'video_ids': [c['id']],
                    'transitions': [],
                    'type': 'video',
                })
            continue

        # 2D clip
        if cur is None:
            cur = {
                'start': c['_start'],
                'end': c['_end'],
                'video_ids': [c['id']],
                'transitions': [],
                'type': 'video',
            }
        else:
            has_xfade = prev and _xfade_between(prev, c)
            span = c['_end'] - cur['start']
            if span > max_chunk_sec and not has_xfade:
                chunks.append(cur)
                cur = {
                    'start': c['_start'],
                    'end': c['_end'],
                    'video_ids': [c['id']],
                    'transitions': [],
                    'type': 'video',
                }
            else:
                cur['video_ids'].append(c['id'])
                cur['end'] = c['_end']

    if cur is not None:
        chunks.append(cur)

    if time_range:
        rs, re_ = time_range
        new_chunks = []
        for ch in chunks:
            ns = max(ch['start'], rs); ne = min(ch['end'], re_)
            if ne - ns < 1e-6: continue
            ch2 = dict(ch); ch2['start'] = ns; ch2['end'] = ne
            new_chunks.append(ch2)
        chunks = new_chunks

    chunks = _split_oversized(proj, chunks, max_weight)
    chunks = _merge_small(proj, chunks, max_weight, merge_sec)

    for i, ch in enumerate(chunks):
        ch['idx'] = i
    return chunks



# 合并检测

def _is_dynamic_clip(proj, c):
    a = proj['assets'].get(c.get('asset')) if c.get('asset') else None
    if a is None: return True
    if a.get('type') in ('text', 'latex', 'svg', '3d'): return True
    if c.get('_elements'): return True
    return False


def _time_windows_similar(a_s, a_e, b_s, b_e, threshold=0.5):
    inter = max(0.0, min(a_e, b_e) - max(a_s, b_s))
    union = max(a_e, b_e) - min(a_s, b_s)
    if union <= 1e-9: return False
    return (inter / union) >= threshold


def _group_by_exact_window(candidates, min_size):
    groups = {}
    for cand in candidates:
        key = (round(cand['start'], 3), round(cand['end'], 3))
        groups.setdefault(key, []).append(cand)
    return [g for g in groups.values() if len(g) >= min_size]


def _group_by_similar_window(candidates, min_size, threshold=0.5):
    sorted_c = sorted(candidates, key=lambda c: c['start'])
    used = set(); groups = []
    for i, a in enumerate(sorted_c):
        if i in used: continue
        group = [a]
        for j in range(i + 1, len(sorted_c)):
            if j in used: continue
            b = sorted_c[j]
            if _time_windows_similar(a['start'], a['end'],
                                      b['start'], b['end'], threshold):
                group.append(b); used.add(j)
        if len(group) >= min_size:
            used.add(i)
            min_s = min(g['start'] for g in group)
            max_e = max(g['end'] for g in group)
            for g in group:
                g['_group_start'] = min_s; g['_group_end'] = max_e
            groups.append(group)
    return groups


def find_mergeable_dynamic(proj, overlay_ids, clips_by_id, chunk_window, aggressive=2):
    if aggressive == 0: return []
    c0, c1 = chunk_window
    cands = []
    for cid in overlay_ids:
        c = clips_by_id[cid]
        if not _is_dynamic_clip(proj, c): continue
        if is_pure_static(proj, c): continue
        s, e = max(c['_start'], c0), min(c['_end'], c1)
        if e - s < 1e-6: continue
        a = proj['assets'].get(c.get('asset')) if c.get('asset') else None
        cands.append({'id': cid, 'clip': c, 'asset': a, 'start': s, 'end': e})
    if aggressive == 1: return _group_by_exact_window(cands, 3)
    if aggressive == 2: return _group_by_exact_window(cands, 2)
    return _group_by_similar_window(cands, 2, 0.5)


def find_mergeable_static(proj, overlay_ids, clips_by_id, chunk_window, aggressive=2):
    if aggressive == 0: return []
    c0, c1 = chunk_window
    cands = []
    for cid in overlay_ids:
        c = clips_by_id[cid]
        if not is_pure_static(proj, c): continue
        s, e = max(c['_start'], c0), min(c['_end'], c1)
        if e - s < 1e-6: continue
        a = proj['assets'].get(c.get('asset')) if c.get('asset') else None
        cands.append({'id': cid, 'clip': c, 'asset': a, 'start': s, 'end': e})
    if aggressive == 1: return _group_by_exact_window(cands, 3)
    if aggressive == 2: return _group_by_exact_window(cands, 2)
    return _group_by_similar_window(cands, 2, 0.3)



# 合并渲染

def _frame_hash(svg_str):
    if _HAS_NUMPY:
        arr = np.frombuffer(svg_str.encode('utf-8'), dtype=np.uint8)
        return hashlib.md5(arr.tobytes()).hexdigest()
    return hashlib.md5(svg_str.encode('utf-8')).hexdigest()


# ============================================================
# GPU 直绘：合并组不走 SVG 字符串
# ============================================================
def _hex_to_rgba(s, opacity=1.0):
    """'#rrggbb' / '#rrggbbaa' / 'white' → (r,g,b,a) in 0..1"""
    if not s:
        return (1, 1, 1, opacity)
    s = str(s).strip().lower()
    named = {'white': (1,1,1), 'black': (0,0,0), 'red': (1,0,0),
             'green': (0,0.5,0), 'blue': (0,0,1), 'yellow': (1,1,0),
             'cyan': (0,1,1), 'magenta': (1,0,1), 'orange': (1,0.65,0),
             'gray': (0.5,0.5,0.5), 'grey': (0.5,0.5,0.5)}
    if s in named:
        r, g, b = named[s]
        return (r, g, b, opacity)
    if s.startswith('#'):
        h = s[1:]
        if len(h) == 3:
            h = ''.join(c * 2 for c in h)
        if len(h) >= 6:
            r = int(h[0:2], 16) / 255
            g = int(h[2:4], 16) / 255
            b = int(h[4:6], 16) / 255
            a = int(h[6:8], 16) / 255 if len(h) >= 8 else 1.0
            return (r, g, b, a * opacity)
    return (1, 1, 1, opacity)


def _draw_element_to_r2d(r2d, proj, container, asset, t, base_mat,
                          parent_opacity=1.0):
    """把一个 asset 直接画到 r2d（无 SVG 中间态）"""
    from raster2d import Mat2D
    import math
    if not asset:
        return
    props = props_at_t(container, t)
    try:
        opacity = float(props.get('opacity', 1)) * parent_opacity
    except (ValueError, TypeError):
        opacity = parent_opacity

    t_type = asset.get('type', 'svg')

    # 元素的局部偏移（element 相对 clip 的 x/y）
    try:
        ex = float(props.get('x', 0) or 0)
        ey = float(props.get('y', 0) or 0)
        erot = float(props.get('rotate', 0) or 0)
        esc = float(props.get('scale', 1) or 1)
    except (ValueError, TypeError):
        ex = ey = 0; erot = 0; esc = 1

    local = Mat2D.translate(ex, ey)
    if erot: local = local @ Mat2D.rotate(math.radians(erot))
    if esc != 1: local = local @ Mat2D.scale(esc)
    mat = base_mat @ local

    r2d.set_matrix(mat)

    if t_type == 'text':
        content = asset.get('content', '')
        if not content:
            return
        try:
            size = float(props.get('font-size', 96))
        except (ValueError, TypeError):
            size = 96
        color = props.get('color', 'white')
        rgba = _hex_to_rgba(color, opacity)
        family = _resolve_font_family(props.get('font-family', 'sans-serif'))
        align = props.get('align', 'center')
        anchor = {'left':'start','center':'middle','right':'end'}.get(align, 'middle')
        r2d.draw_text(content, 0, 0, size, rgba,
                       font=family, anchor=anchor, baseline='middle')

    elif t_type == 'image':
        try:
            data = Path(asset['src']).read_bytes()
            iw = float(props.get('w', 640))
            ih = float(props.get('h', 360))
            r2d.draw_image(data, -iw/2, -ih/2, iw, ih, opacity=opacity)
        except Exception as e:
            print(f"  ⚠ [2d-gpu] image: {e}", file=sys.stderr)

    elif t_type == 'svg':
        inner = asset.get('content', '')
        m = re.search(r'<svg[^>]*>(.*)</svg>', inner, re.S)
        if m: inner = m.group(1)
        try:
            from scene2d import draw_svg_inner_to_r2d
            draw_svg_inner_to_r2d(r2d, inner, mat, opacity)
        except Exception as e:
            print(f"  ⚠ [2d-gpu] svg: {e}", file=sys.stderr)

    elif t_type == 'latex':
        try:
            fontsize = float(props.get('font-size', 40))
            b64, (iw, ih) = latex_png(asset.get('content', ''), fontsize,
                                       proj.get('workdir'))
            data = base64.b64decode(b64)
            r2d.draw_image(data, -iw/2, -ih/2, iw, ih, opacity=opacity)
        except Exception as e:
            print(f"  ⚠ [2d-gpu] latex: {e}", file=sys.stderr)


def _draw_clip_to_r2d(r2d, proj, clip, asset, t_clip, W, H):
    """把一个 clip（含 elements）直接画到 r2d"""
    from raster2d import Mat2D
    import math
    props = props_at_t(clip, t_clip)
    try:
        x = float(props.get('x', W/2))
        y = float(props.get('y', H/2))
        opacity = float(props.get('opacity', 1))
        rotate = float(props.get('rotate', 0))
        scale = float(props.get('scale', 1))
    except (ValueError, TypeError):
        x, y, opacity, rotate, scale = W/2, H/2, 1, 0, 1

    mat = Mat2D.translate(x, y)
    if rotate: mat = mat @ Mat2D.rotate(math.radians(rotate))
    if scale != 1: mat = mat @ Mat2D.scale(scale)

    elements = clip.get('_elements', [])
    if elements:
        for el in elements:
            el_asset = proj['assets'].get(el.get('asset')) if el.get('asset') else None
            if el_asset is None and 'content' in el:
                el_asset = {'type': 'svg', 'content': el['content']}
            if el_asset is None:
                continue
            _draw_element_to_r2d(r2d, proj, el, el_asset, t_clip, mat, opacity)
    else:
        # 无 elements：asset 就是 clip 本身。clip 的 x/y/rotate/scale/opacity
        # 由 _draw_element_to_r2d 从 clip.props 里读一次即可。
        # base_mat 必须 identity、parent_opacity 必须 1.0，
        # 否则会与上面已算出的 mat/opacity 叠加 → 位置/透明度双重偏移。
        _draw_element_to_r2d(r2d, proj, clip, asset, t_clip,
                              Mat2D.identity(), 1.0)


def _group_frame_hash(proj, group, t_abs):
    """对合并组的每一帧做特征 hash（用于去重）"""
    parts = []
    for g in group:
        clip = g['clip']
        t_clip = t_abs - clip['_start']
        p = props_at_t(clip, t_clip)
        for k in sorted(p.keys()):
            parts.append(f"{k}={p[k]}")
    s = '|'.join(parts)
    return hashlib.md5(s.encode('utf-8')).hexdigest()[:16]


def render_group_merged(proj, group, workers, inputs, filters,
                         chunk_idx=0, static_only=False):
    t_start = time.time()
    c0 = group[0].get('_group_start', group[0]['start'])
    c1 = group[0].get('_group_end', group[0]['end'])
    dur = c1 - c0; fps = proj['fps']
    n = max(1, int(round(dur * fps)))
    merged_id = f"mrg_c{chunk_idx}_" + "_".join(g['id'][:8] for g in group[:3])
    out_base = proj['workdir'] / f"frames_{merged_id}"
    safe_rmtree(out_base, label=f'merged {merged_id}')
    out_base.mkdir(parents=True, exist_ok=True)
    clips_assets = [(g['clip'], g['asset']) for g in group]

    if static_only:
        fp = out_base / ("static" + _FRAME_EXT)
        ok = False
        # 优先 GPU
        if _HAS_GPU_2D and _svg_gpu_level(b'', force=False) >= 0:
            try:
                from raster2d import get_raster2d
                W = proj.get('_out_w') or proj['w']
                H = proj.get('_out_h') or proj['h']
                r2d = get_raster2d(W, H)
                r2d.begin(clear=(0, 0, 0, 0))
                for g in group:
                    _draw_clip_to_r2d(r2d, proj, g['clip'], g['asset'],
                                       c0 - g['clip']['_start'], W, H)
                arr = r2d.end()
                from PIL import Image as _I
                img = _I.fromarray(arr, 'RGBA')
                if _FRAME_EXT == '.bmp':
                    img.save(str(fp), 'BMP')
                else:
                    img.save(str(fp), 'PNG', compress_level=1)
                ok = True
            except Exception as e:
                print(f"  ⚠ [merged-static GPU] {e}", file=sys.stderr)
        if not ok:
            # cairo 回退
            try:
                svg = frame_svg_merged(proj, clips_assets, c0)
                _render_png(svg.encode(), str(fp),
                            proj.get('_out_w') or proj['w'],
                            proj.get('_out_h') or proj['h'])
                ok = True
            except Exception as e:
                print(f"  ⚠ 静态合并 {merged_id} 失败: {e}", file=sys.stderr)
                return None
        idx = len(inputs)
        inputs.append(['-loop', '1', '-framerate', str(fps), '-i', str(fp)])
        lbl = f"cm_{chunk_idx}_{merged_id}"
        filters.append(f"[{idx}:v]trim=end_frame={n},"
                       f"setpts=PTS-STARTPTS,format=rgba[{lbl}]")
        print(f"  · [merged-static] {'+'.join(g['id'] for g in group)}: 1 帧",
              flush=True)
        print(f"[TIME] {merged_id} 1f {time.time()-t_start:.2f}s")
        return lbl

    # 🆕 GPU 快速路径
    if _HAS_GPU_2D:
        try:
            return _render_group_merged_gpu(
                proj, group, workers, inputs, filters,
                chunk_idx, merged_id, out_base, n, c0, c1, fps, clips_assets,
                t_start)
        except Exception as e:
            import traceback
            print(f"  ⚠ [merged-gpu] 失败，回退: {e}", file=sys.stderr)
            traceback.print_exc()

    svg_strings = []
    for i in range(n):
        t = c0 + i / fps
        try: svg_strings.append(frame_svg_merged(proj, clips_assets, t))
        except Exception as e:
            print(f"  ⚠ merged {merged_id} 帧 {i} SVG 失败: {e}", file=sys.stderr)
            svg_strings.append(None)

    tasks = []; reuse_map = {}
    last_hash = None; last_idx = None
    for i, svg in enumerate(svg_strings):
        if svg is None: continue
        h = _frame_hash(svg)
        if h == last_hash and last_idx is not None:
            reuse_map[i] = last_idx
        else:
            last_hash = h; last_idx = i
            tasks.append((svg.encode(), str(out_base / f"{i:05d}{_FRAME_EXT}"),
                          proj.get('_out_w') or proj['w'],
                          proj.get('_out_h') or proj['h']))

    if not tasks and not reuse_map:
        print(f"  ⚠ merged {merged_id} 无可用帧", file=sys.stderr)
        return None

    we = max(1, min(workers, Env.cpu))
    if we == 1:
        for tk in tasks: _render_task(tk)
    else:
        try:
            with ThreadPoolExecutor(max_workers=we) as ex:
                list(ex.map(_render_task, tasks))
        except Exception as e:
            print(f"  ⚠ merged 并行失败，回退串行: {e}", file=sys.stderr)
            for tk in tasks: _render_task(tk)

    reuse_count = 0; reuse_failed = []
    for dst_idx, src_idx in reuse_map.items():
        src = out_base / f"{src_idx:05d}{_FRAME_EXT}"
        dst = out_base / f"{dst_idx:05d}{_FRAME_EXT}"
        if not src.exists(): reuse_failed.append(dst_idx); continue
        if safe_link_or_copy(src, dst) == 'fail': reuse_failed.append(dst_idx)
        else: reuse_count += 1
    if reuse_failed:
        retry = []
        for idx in reuse_failed:
            if idx < len(svg_strings) and svg_strings[idx] is not None:
                retry.append((svg_strings[idx].encode(),
                              str(out_base / f"{idx:05d}{_FRAME_EXT}"),
                              proj['w'], proj['h']))
        if retry:
            print(f"  ⚠ {len(retry)} 帧复用失败，重新渲染", file=sys.stderr)
            we2 = max(1, min(workers, Env.cpu))
            if we2 == 1:
                for t in retry: _render_task(t)
            else:
                with ThreadPoolExecutor(max_workers=we2) as ex:
                    list(ex.map(_render_task, retry))
    missing = [i for i in range(n) if not (out_base / f"{i:05d}{_FRAME_EXT}").exists()]
    if missing:
        print(f"  ⚠ 缺失 {len(missing)} 帧，补渲", file=sys.stderr)
        for i in missing:
            svg = svg_strings[i] if i < len(svg_strings) else None
            if svg is None:
                for j in range(i-1, -1, -1):
                    if j < len(svg_strings) and svg_strings[j]:
                        svg = svg_strings[j]; break
            if svg is None:
                for j in range(i+1, n):
                    if j < len(svg_strings) and svg_strings[j]:
                        svg = svg_strings[j]; break
            if svg:
                _render_task((svg.encode(), str(out_base / f"{i:05d}{_FRAME_EXT}"),
                              proj['w'], proj['h']))

    rendered_n = len(tasks)
    idx = len(inputs)
    inputs.append(['-framerate', str(fps), '-i', str(out_base / ('%05d' + _FRAME_EXT))])
    lbl = f"cm_{chunk_idx}_{merged_id}"
    filters.append(f"[{idx}:v]format=rgba[{lbl}]")
    parts = []
    if reuse_count > 0: parts.append(f"复用 {reuse_count}")
    if reuse_failed: parts.append(f"重渲 {len(reuse_failed)}")
    dedup_msg = f", {', '.join(parts)}" if parts else ""
    print(f"  · [merged] {'+'.join(g['id'] for g in group)}: "
          f"{n} 帧 (渲染 {rendered_n}{dedup_msg})", flush=True)
    print(f"[TIME] {merged_id} {n}f {time.time()-t_start:.2f}s")
    return lbl



# 渲染动态 clip

def _render_group_merged_gpu(proj, group, workers, inputs, filters,
                              chunk_idx, merged_id, out_base, n, c0, c1, fps,
                              clips_assets, t_start):
    """GPU 快速路径：直接遍历 clips 画 r2d，不拼 SVG 字符串。

    --exp-autocairo 模式下：
      · 先用逐帧 hash 分析，找出「连续 hash 相同且长度 >= autocairo_min」的静止窗口
      · 窗口首帧用 cairo 渲染（质量优先）
      · 窗口其余帧复用首帧（硬链接/拷贝）
      · 动态帧照常走 GPU 直绘
    """
    from raster2d import get_raster2d
    from PIL import Image as _PILImage
    W = proj.get('_out_w') or proj['w']
    H = proj.get('_out_h') or proj['h']

    r2d = get_raster2d(W, H)

    total_elems = sum(len(g['clip'].get('_elements', [])) or 1 for g in group)

    # ─── PassPNG：逐帧写 raw RGBA，无 PNG/BMP 中间体 ───
    if proj.get('_exp_pass_png'):
        raw_path = out_base / (merged_id + ".raw")
        prev_bytes = None
        last_hash_ = None
        written = 0
        t_render = 0.0
        with open(str(raw_path), 'wb') as fh:
            for i in range(n):
                t_abs = c0 + i / fps
                h_ = _group_frame_hash(proj, group, t_abs)
                if h_ == last_hash_ and prev_bytes is not None:
                    fh.write(prev_bytes)
                    continue
                last_hash_ = h_
                ts = time.perf_counter()
                r2d.begin(clear=(0, 0, 0, 0))
                for g in group:
                    _draw_clip_to_r2d(r2d, proj, g['clip'], g['asset'],
                                       t_abs - g['clip']['_start'], W, H)
                arr = r2d.end()
                t_render += time.perf_counter() - ts
                b = arr.tobytes()
                fh.write(b)
                prev_bytes = b
                written += 1
        idx = len(inputs)
        inputs.append(['-f', 'rawvideo', '-pixel_format', 'rgba',
                        '-video_size', f'{W}x{H}', '-framerate', str(fps),
                        '-i', str(raw_path)])
        lbl = f"cm_{chunk_idx}_{merged_id}"
        filters.append(f"[{idx}:v]format=rgba[{lbl}]")
        proj.setdefault('_raw_cleanup', []).append(str(raw_path))
        avg_render = t_render / written * 1000 if written else 0
        print(f"  · [merged-raw] {'+'.join(g['id'] for g in group)}: "
              f"{n} 帧, 渲染 {written} 复用 {n - written}  "
              f"render {avg_render:.1f}ms/帧  (passPNG)", flush=True)
        print(f"[TIME] {merged_id} {n}f {time.time()-t_start:.2f}s")
        return lbl

    # ---- Autocairo：预扫描找长静止窗口 ----
    autocairo_enabled = bool(proj.get('_exp_autocairo'))
    autocairo_min = int(proj.get('_autocairo_min', 30))
    static_windows = []
    autocairo_set = {}

    if autocairo_enabled and autocairo_min > 1:
        hashes = []
        for i in range(n):
            t_abs = c0 + i / fps
            hashes.append(_group_frame_hash(proj, group, t_abs))
        i = 0
        while i < n:
            j = i
            while j + 1 < n and hashes[j + 1] == hashes[i]:
                j += 1
            if (j - i + 1) >= autocairo_min:
                static_windows.append((i, j))
            i = j + 1
        for s, e in static_windows:
            for k in range(s, e + 1):
                autocairo_set[k] = s

    extra = ""
    if static_windows:
        frames_saved = sum(e - s for s, e in static_windows)
        extra = f", {len(static_windows)} 个静止段(≥{autocairo_min}帧→CPU, {frames_saved} 帧复用)"

    print(f"  · [merged-gpu] {'+'.join(g['id'] for g in group)}: "
          f"{n} 帧, {len(group)} clips, ~{total_elems} 元素{extra}", flush=True)

    reuse_map = {}
    last_hash = None; last_idx = None
    render_times = []
    encode_times = []
    cairo_count = 0

    for i in range(n):
        t_abs = c0 + i / fps

        # --- 静止窗口内 ---
        if i in autocairo_set:
            s = autocairo_set[i]
            fp = out_base / f"{i:05d}{_FRAME_EXT}"
            if i == s:
                # 窗口首帧：cairo 渲染
                ts = time.perf_counter()
                ok = False
                try:
                    svg = frame_svg_merged(proj, clips_assets, t_abs)
                    _render_png_cairo_only(svg.encode(), str(fp), W, H)
                    ok = True
                    cairo_count += 1
                except Exception as e:
                    print(f"  ⚠ [autocairo] frame {i} 失败, 回退 GPU: {e}",
                          file=sys.stderr)
                if not ok:
                    r2d.begin(clear=(0, 0, 0, 0))
                    for g in group:
                        _draw_clip_to_r2d(r2d, proj, g['clip'], g['asset'],
                                           t_abs - g['clip']['_start'], W, H)
                    arr = r2d.end()
                    _PILImage.fromarray(arr, 'RGBA').save(
                        str(fp), 'PNG', compress_level=1)
                t_render = time.perf_counter() - ts
                render_times.append(t_render)
                encode_times.append(0.0)
                # 让后续动态帧的 hash 状态继续跟踪
                last_hash = None
                last_idx = None
            else:
                reuse_map[i] = s
            continue

        # --- 动态帧：GPU 渲染 ---
        h = _group_frame_hash(proj, group, t_abs)
        if h == last_hash and last_idx is not None:
            reuse_map[i] = last_idx
            continue
        last_hash = h; last_idx = i

        ts = time.perf_counter()
        r2d.begin(clear=(0, 0, 0, 0))
        for g in group:
            clip = g['clip']
            asset = g['asset']
            _draw_clip_to_r2d(r2d, proj, clip, asset,
                               t_abs - clip['_start'], W, H)
        arr = r2d.end()
        t_render = time.perf_counter() - ts

        te = time.perf_counter()
        img = _PILImage.fromarray(arr, 'RGBA')
        fp = out_base / f"{i:05d}{_FRAME_EXT}"
        if _FRAME_EXT == '.bmp':
            img.save(str(fp), 'BMP')
        else:
            img.save(str(fp), 'PNG', compress_level=1)
        t_encode = time.perf_counter() - te

        render_times.append(t_render)
        encode_times.append(t_encode)

    # ---- 复用：硬链接 / 拷贝 ----
    reuse_count = 0
    for dst_idx, src_idx in reuse_map.items():
        src = out_base / f"{src_idx:05d}{_FRAME_EXT}"
        dst = out_base / f"{dst_idx:05d}{_FRAME_EXT}"
        if not src.exists():
            continue
        if safe_link_or_copy(src, dst) != 'fail':
            reuse_count += 1

    # ---- 补渲（如果有缺） ----
    missing = [i for i in range(n)
               if not (out_base / f"{i:05d}{_FRAME_EXT}").exists()]
    if missing:
        print(f"  ⚠ [merged-gpu] 缺 {len(missing)} 帧，补渲", file=sys.stderr)
        for i in missing:
            t_abs = c0 + i / fps
            r2d.begin(clear=(0, 0, 0, 0))
            for g in group:
                clip = g['clip']
                _draw_clip_to_r2d(r2d, proj, clip, g['asset'],
                                   t_abs - clip['_start'], W, H)
            arr = r2d.end()
            img = _PILImage.fromarray(arr, 'RGBA')
            fp = out_base / f"{i:05d}{_FRAME_EXT}"
            if _FRAME_EXT == '.bmp':
                img.save(str(fp), 'BMP')
            else:
                img.save(str(fp), 'PNG', compress_level=1)

    rendered = len(render_times)
    avg_render = sum(render_times) / rendered * 1000 if rendered else 0
    avg_encode = sum(encode_times) / rendered * 1000 if rendered else 0

    idx = len(inputs)
    inputs.append(['-framerate', str(fps), '-i',
                    str(out_base / ('%05d' + _FRAME_EXT))])
    lbl = f"cm_{chunk_idx}_{merged_id}"
    filters.append(f"[{idx}:v]format=rgba[{lbl}]")

    autocairo_msg = f"  cairo {cairo_count}" if cairo_count else ""
    print(f"  · [merged-gpu] 渲染 {rendered} 复用 {reuse_count}{autocairo_msg}  "
          f"render {avg_render:.1f}ms/帧  encode {avg_encode:.1f}ms/帧",
          flush=True)
    print(f"[TIME] {merged_id} {n}f {time.time()-t_start:.2f}s")
    return lbl


def render_clip_segments(proj, clip, asset, workers, inputs, filters,
                          chunk_idx=0, chunk_window=None):
    t_clip = time.time()
    cs, ce = clip['_start'], clip['_end']
    if chunk_window:
        cs, ce = max(cs, chunk_window[0]), min(ce, chunk_window[1])
    in_start, in_end = cs - clip['_start'], ce - clip['_start']
    if in_end - in_start < 1e-6: return None

    out_base = proj['workdir'] / f"frames_c{chunk_idx}_{clip['id']}"
    safe_rmtree(out_base, label=f'clip {clip["id"]}')
    out_base.mkdir(parents=True, exist_ok=True)

    fps = proj['fps']
    segs = []
    for s0, s1, st in segment_clip(proj, clip):
        cs2, ce2 = max(s0, in_start), min(s1, in_end)
        if ce2 - cs2 > 1e-6: segs.append((cs2, ce2, st))

    seg_labels, rendered_cnt = [], 0
    for si, (t0, t1, static) in enumerate(segs):
        n = max(1, int(round((t1 - t0) * fps)))
        if static:
            try:
                svg = frame_svg(proj, clip, asset, t0)
                fp = out_base / f"seg{si:02d}_static{_FRAME_EXT}"
                _render_png(svg.encode(), str(fp),
                            proj.get('_out_w') or proj['w'],
                            proj.get('_out_h') or proj['h'])
                rendered_cnt += 1
            except Exception as e:
                print(f"  ⚠ {clip['id']} 段 {si} 失败: {e}", file=sys.stderr)
                continue
            idx = len(inputs)
            inputs.append(['-loop', '1', '-framerate', str(fps), '-i', str(fp)])
            lbl = f"cs_{clip['id']}_{chunk_idx}_{si}"
            filters.append(f"[{idx}:v]trim=end_frame={n},"
                           f"setpts=PTS-STARTPTS,format=rgba[{lbl}]")
            seg_labels.append(lbl)
        else:
            sd = out_base / f"seg{si:02d}"; sd.mkdir(exist_ok=True)
            # 🆕 先全量生成 SVG，然后按帧 hash 去重
            # （相同 hash 的帧只渲染一次，其余硬链接/拷贝复用）
            svgs = []
            for i in range(n):
                t = t0 + i / fps
                try:
                    svgs.append(frame_svg(proj, clip, asset, t))
                except Exception as e:
                    print(f"  ⚠ {clip['id']} 帧 {i} 失败: {e}", file=sys.stderr)
                    svgs.append(None)

            # 建 hash → 最近一次出现的帧索引
            # （用 last_seen 而非 first_seen：复用源永远相邻，OS 缓存热）
            last_seen = {}
            reuse_map = {}
            tasks = []
            for i, svg in enumerate(svgs):
                if svg is None:
                    continue
                h = _frame_hash(svg)
                if h in last_seen:
                    reuse_map[i] = last_seen[h]
                else:
                    tasks.append((svg.encode(), str(sd / f"{i:05d}{_FRAME_EXT}"),
                                  proj.get('_out_w') or proj['w'],
                                  proj.get('_out_h') or proj['h']))
                last_seen[h] = i

            if not tasks and not reuse_map:
                continue

            we = max(1, min(workers, Env.cpu))
            if we == 1:
                for tk in tasks: _render_task(tk)
            else:
                try:
                    with ThreadPoolExecutor(max_workers=we) as ex:
                        list(ex.map(_render_task, tasks))
                except Exception as e:
                    print(f"  ⚠ 并行失败: {e}", file=sys.stderr)
                    for tk in tasks: _render_task(tk)

            # 复用：硬链接/拷贝
            reuse_count = 0
            for dst_idx, src_idx in reuse_map.items():
                src = sd / f"{src_idx:05d}{_FRAME_EXT}"
                dst = sd / f"{dst_idx:05d}{_FRAME_EXT}"
                if src.exists() and safe_link_or_copy(src, dst) != 'fail':
                    reuse_count += 1

            # 补渲（极端情况）
            missing = [i for i in range(n)
                       if not (sd / f"{i:05d}{_FRAME_EXT}").exists()]
            if missing:
                for i in missing:
                    if i < len(svgs) and svgs[i]:
                        _render_task((svgs[i].encode(),
                                      str(sd / f"{i:05d}{_FRAME_EXT}"),
                                      proj.get('_out_w') or proj['w'],
                                      proj.get('_out_h') or proj['h']))

            if reuse_count > 0:
                print(f"  · {clip['id']} 段 {si}: 渲染 {len(tasks)} 帧, "
                      f"复用 {reuse_count} 帧 ({n} 总)",
                      file=sys.stderr)
            rendered_cnt += n
            idx = len(inputs)
            inputs.append(['-framerate', str(fps), '-i', str(sd / ('%05d' + _FRAME_EXT))])
            lbl = f"cd_{clip['id']}_{chunk_idx}_{si}"
            filters.append(f"[{idx}:v]format=rgba[{lbl}]")
            seg_labels.append(lbl)

    if not seg_labels:
        lbl = f"cb_{clip['id']}_{chunk_idx}"
        filters.append(f"color=c=black@0.0:s={proj['w']}x{proj['h']}:r={fps}:"
                       f"d={in_end - in_start},format=rgba[{lbl}]")
        return lbl

    print(f"  · {clip['id']}: {rendered_cnt} 帧, {len(seg_labels)} 段", flush=True)
    print(f"[TIME] clip_{clip['id']}_{chunk_idx} "
          f"{rendered_cnt}f {time.time()-t_clip:.2f}s")
    if len(seg_labels) == 1: return seg_labels[0]
    lbl = f"c_{clip['id']}_{chunk_idx}"
    ins = "".join(f"[{x}]" for x in seg_labels)
    filters.append(f"{ins}concat=n={len(seg_labels)}:v=1:a=0[{lbl}]")
    return lbl



# 视频链

def build_video_chain(filters, vlist, transitions, chunk_idx):
    if not vlist: return None
    segments = [[vlist[0]]]
    for i in range(1, len(vlist)):
        if (vlist[i-1]['id'], vlist[i]['id']) in transitions:
            segments.append([vlist[i]])
        else: segments[-1].append(vlist[i])

    seg_labels = []
    for si, seg in enumerate(segments):
        if len(seg) == 1: seg_labels.append(seg[0]['label'])
        else:
            ins = "".join(f"[{c['label']}]" for c in seg)
            lbl = f"seg_{chunk_idx}_{si}"
            filters.append(f"{ins}concat=n={len(seg)}:v=1:a=0[{lbl}]")
            seg_labels.append(lbl)

    if len(seg_labels) == 1: return seg_labels[0]

    prev_label = seg_labels[0]
    prev_dur = sum(c['dur'] for c in segments[0])
    for i in range(1, len(seg_labels)):
        fid = segments[i-1][-1]['id']; tid = segments[i][0]['id']
        tr = transitions[(fid, tid)]
        d, eff = tr['duration'], tr['effect']
        offset = prev_dur - d
        lbl = f"xfd_{chunk_idx}_{i}"
        filters.append(f"[{prev_label}][{seg_labels[i]}]"
                       f"xfade=transition={eff}:duration={d}:offset={offset}[{lbl}]")
        prev_label = lbl
        prev_dur = prev_dur + sum(c['dur'] for c in segments[i]) - d
    return prev_label



# 编码参数

def _chunk_encode_args(proj):
    q = proj.get('_quality', 'fast')
    p = QUALITY_PRESETS.get(q, QUALITY_PRESETS['fast'])
    # 硬件编码器：用 chunk_codec 字段
    if p.get('chunk_codec'):
        return (list(p['chunk_codec']),
                ['-c:a', 'aac', '-b:a', p['chunk_ab'],
                 '-ar', '44100', '-ac', '2'])
    # 软件编码器
    return (['-c:v', 'libx264', '-preset', p['chunk_preset'],
             '-crf', p['chunk_crf'], '-pix_fmt', 'yuv420p'],
            ['-c:a', 'aac', '-b:a', p['chunk_ab'], '-ar', '44100', '-ac', '2'])


def _merge_encode_args(proj_or_quality, aggressive=0):
    if isinstance(proj_or_quality, dict):
        q = proj_or_quality.get('_quality', 'fast')
    else:
        q = proj_or_quality or 'fast'
    p = QUALITY_PRESETS.get(q, QUALITY_PRESETS['fast'])
    return (['-c:v', 'libx264', '-preset', p['merge_preset'],
             '-crf', p['merge_crf'], '-pix_fmt', 'yuv420p'],
            ['-c:a', 'aac', '-b:a', p['merge_ab'], '-ar', '44100', '-ac', '2'])



# 【新】音频轨收集 + 全局合成

def collect_audio_specs(proj, time_range=None):
    """收集所有 audio clip 的全局时间信息。
    返回 [{src, start, end, volume, fade_in, fade_out, loop}, ...]
    time_range 存在时，所有 spec 的时间会裁剪到 [rs, re] 并偏移到从 0 起。
    """
    rs, re_ = (time_range if time_range else (0.0, proj['duration']))
    specs = []
    for tr in proj['tracks']:
        if tr['kind'] != 'audio': continue
        for c in tr['clips']:
            a = proj['assets'].get(c.get('asset'))
            if not a or 'src' not in a: continue
            if not Path(a['src']).exists(): continue
            s, e = max(c['_start'], rs), min(c['_end'], re_)
            if e - s < 1e-6: continue
            st = c.get('_style', {})
            # 裁剪后的时间窗，转成从 0 起
            specs.append({
                'src': a['src'],
                'start': s - rs,
                'end': e - rs,
                'in_clip_offset': s - c['_start'],
                'volume': float(st.get('volume', 1)),
                'fade_in': float(st.get('fade-in', 0)),
                'fade_out': float(st.get('fade-out', 0)),
                'loop': c.get('loop', 'false').lower() == 'true',
                'clip_total': clip_duration(proj, c),
            })
    return specs


def mix_global_audio(proj, video_path, out_path, time_range=None):
    """在已渲染好的静音视频上一次性叠加所有音频轨。
    这是音频延迟合成的核心 —— 视频部分不参与 AAC 转码。
    """
    specs = collect_audio_specs(proj, time_range)
    if not specs:
        print("  · 无音频轨，跳过音频合成")
        if str(video_path) != str(out_path):
            try: safe_replace(str(video_path), str(out_path)); return True
            except Exception:
                try: shutil.copy2(str(video_path), str(out_path)); return True
                except Exception as e:
                    print(f"  ✗ 复制失败: {e}", file=sys.stderr); return False
        return True

    inputs = ['-i', str(video_path)]
    filter_parts = []; audio_labels = []

    for i, spec in enumerate(specs):
        ai = i + 1
        if spec['loop']:
            inputs += ['-stream_loop', '-1', '-i', spec['src']]
        else:
            inputs += ['-i', spec['src']]

        seg_dur = spec['end'] - spec['start']
        in_off = spec['in_clip_offset']
        chain = [f"atrim={in_off}:{in_off + seg_dur}", "asetpts=PTS-STARTPTS"]
        if spec['volume'] != 1.0:
            chain.append(f"volume={spec['volume']}")

        # fade-in：clip 内相对时间 < fade_in 时应用
        if spec['fade_in'] > 0 and in_off < spec['fade_in']:
            remain = spec['fade_in'] - in_off
            chain.append(f"afade=t=in:st=0:d={remain}")
        # fade-out：clip 尾部
        if spec['fade_out'] > 0:
            clip_end_rel = spec['clip_total'] - spec['fade_out']
            if spec['end'] > spec['start'] + clip_end_rel:
                st_fade = max(0, clip_end_rel - in_off)
                chain.append(f"afade=t=out:st={st_fade}:d={spec['fade_out']}")

        delay_ms = int(spec['start'] * 1000)
        if delay_ms > 0:
            chain.append(f"adelay={delay_ms}|{delay_ms}")

        lbl = f"aud{i}"
        filter_parts.append(f"[{ai}:a]" + ",".join(chain) + f"[{lbl}]")
        audio_labels.append(lbl)

    if len(audio_labels) == 1:
        filter_parts.append(f"[{audio_labels[0]}]anull[aout]")
    else:
        mix_in = "".join(f"[{l}]" for l in audio_labels)
        filter_parts.append(f"{mix_in}amix=inputs={len(audio_labels)}:"
                            f"duration=longest:normalize=0[aout]")

    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
    cmd += inputs
    cmd += ['-filter_complex', ';'.join(filter_parts),
            '-map', '0:v', '-map', '[aout]',
            '-c:v', 'copy',
            '-c:a', 'aac', '-b:a', '192k', '-ar', '44100', '-ac', '2',
            '-shortest', str(out_path)]
    rc, _, err = safe_run(cmd, timeout=3600)
    if rc != 0:
        print(f"  ✗ 全局音频合成失败: {err[-500:]}", file=sys.stderr)
        return False
    return True



# 渲染单个 chunk（改：silent 模式）

# ============================================================
# GL3D 进程/线程本地单例
# ============================================================
_GL3D_CACHE = {}


def _get_gl3d(W, H):
    """每 (W, H, thread) 一个 GL3D 单例（EGL context 是线程本地的）"""
    import threading
    key = (W, H, threading.get_ident())
    if key not in _GL3D_CACHE:
        try:
            import scene3d as S
        except ImportError:
            return None
        _GL3D_CACHE[key] = S.GL3D(W, H, verbose=False)
    return _GL3D_CACHE[key]


def _release_gl3d():
    """释放当前线程的 GL3D"""
    import threading
    key = (None, None, threading.get_ident())
    for k in list(_GL3D_CACHE.keys()):
        if k[2] == threading.get_ident():
            try:
                _GL3D_CACHE[k].close()
            except Exception:
                pass
            del _GL3D_CACHE[k]



def _upload_scene_geometry(gl3d, scene, verbose=False):
    """上传 scene 里所有 mesh 的几何体 + 显式 <instances> 数据。"""
    S = sys.modules['scene3d']
    inst_count = 0
    for m in scene['meshes']:
        parts = m['size'].split()
        sz = float(parts[0]) if parts else 1.0
        color = m['color']
        typ = m['type']
        if typ == 'cube':
            fc = [color, tuple(c * 0.55 for c in color), color,
                  tuple(c * 0.55 for c in color),
                  tuple(c * 0.8 for c in color),
                  tuple(c * 0.45 for c in color)]
            sizes = [float(x) for x in parts]
            if len(sizes) == 1: sizes = sizes * 3
            gl3d.upload_mesh(m['id'], S.make_cube_vertices(tuple(sizes[:3]), fc))
        elif typ == 'plane':
            w = float(parts[0]) if len(parts) > 0 else 8
            h = float(parts[1]) if len(parts) > 1 else w
            gl3d.upload_mesh(m['id'], S.make_plane_vertices(w, h,
                             float(m['pos'][1]), color))
        elif typ == 'prism':
            hgt = m.get('_height', sz * 2.0)
            fc = {'top': color, 'bottom': tuple(c * 0.5 for c in color),
                  's1': tuple(min(1.0, c * 1.15) for c in color), 's2': color,
                  's3': tuple(c * 0.7 for c in color)}
            gl3d.upload_mesh(m['id'], S.make_prism_vertices(sz, hgt, fc))
        elif typ == 'sphere':
            gl3d.upload_mesh(m['id'], S.make_sphere_vertices(sz, 32, 16, color))
        elif typ == 'cylinder':
            hgt = m.get('_height', sz * 2.0)
            gl3d.upload_mesh(m['id'], S.make_cylinder_vertices(sz, hgt, 32, color))
        elif typ == 'cone':
            hgt = m.get('_height', sz * 2.0)
            gl3d.upload_mesh(m['id'], S.make_cone_vertices(sz, hgt, 32, color))
        elif typ == 'paper':
            w = float(parts[0]) if len(parts) > 0 else 2
            h = float(parts[1]) if len(parts) > 1 else w
            gl3d.upload_mesh(m['id'], S.make_plane_subdiv_vertices(w, h, 2, 2, color))
        elif typ == 'wave':
            w = float(parts[0]) if len(parts) > 0 else 3
            h = float(parts[1]) if len(parts) > 1 else w
            seg = m.get('_seg', 48)
            gl3d.upload_mesh(m['id'],
                             S.make_plane_subdiv_vertices(w, h, seg, seg, color))
        elif typ in ('flag', 'curl'):
            w = float(parts[0]) if len(parts) > 0 else 3
            h = float(parts[1]) if len(parts) > 1 else w
            seg = m.get('_seg', 48)
            gl3d.upload_mesh(m['id'],
                             S.make_plane_subdiv_vertices_v(w, h, seg, seg, color))

        if m.get('_instances'):
            try:
                mats = S._instances_to_matrices(m['_instances'])
                n = gl3d.upload_instance_transforms(m['id'], mats)
                inst_count += n
                if verbose:
                    print(f"    · [inst] {m['id']} ← {n} 实例")
            except Exception as e:
                print(f"  ⚠ [inst] {m['id']}: {e}", file=sys.stderr)
                m.pop('_instances', None)
    return inst_count


def _upload_scene_textures(gl3d, scene, base_dir, verbose=False):
    """上传 scene 里所有 mesh 的纹理（按 key 去重）。
    副作用：往 m 里写 _tex_names。返回 (tex_count, tex_refs)。"""
    S = sys.modules['scene3d']
    tex_count = 0
    tex_refs = {}
    for m in scene['meshes']:
        spec = m.get('texture')
        if not spec: continue
        try:
            resolved = S._resolve_texture_specs(spec, base_dir)
            tex_names = []
            for key, s in resolved:
                name = gl3d.upload_texture_by_key(key, s)
                tex_names.append(name)
                tex_refs[key] = tex_refs.get(key, 0) + 1
            m['_tex_names'] = tex_names
            tex_count += len(resolved)
        except Exception as e:
            print(f"  ⚠ [space3d] texture {m['id']}: {e}", file=sys.stderr)
    if verbose and tex_refs:
        uniq = len(tex_refs)
        total = sum(tex_refs.values())
        if uniq < total:
            print(f"    · [tex] {total} 次引用 → {uniq} 份唯一纹理")
    return tex_count, tex_refs


def _setup_instancing_groups(gl3d, scene, verbose=False):
    """建立静态 + 动态自动实例化分组。

    返回 (static_groups, static_member_set,
           dynamic_groups, dynamic_member_set)

    XMLVE_FASTPATH=1 时才启用（默认关，因为非均匀缩放下光照会失真）
    """
    S = sys.modules['scene3d']
    static_groups = []
    static_member_set = set()
    dynamic_groups = []
    dynamic_member_set = set()

    _fp = scene.get('fastpath', _FASTPATH_ENABLED)
    if not _fp:
        if verbose:
            print("    · [auto-inst] 已跳过（fastpath=0）")
        return (static_groups, static_member_set,
                dynamic_groups, dynamic_member_set)

    # 静态分组
    try:
        static_groups = S.group_instancing_candidates(scene['meshes'])
        for ref, members in static_groups:
            ref_m = scene['meshes'][ref]
            mats = S.build_relative_instance_matrices(
                scene['meshes'], members, ref)
            gl3d.upload_instance_transforms(ref_m['id'], mats)
            for i in members:
                static_member_set.add(i)
            if verbose:
                print(f"    · [auto-inst] {ref_m['id']} ← {len(members)} 个同类 mesh")
    except Exception as e:
        print(f"  ⚠ [auto-inst] 失败: {e}", file=sys.stderr)
        static_groups = []
        static_member_set = set()

    # 动态分组（exp）
    if _DYN_INST_ENABLED:
        try:
            dynamic_groups = S.find_dynamic_groups(scene['meshes'])
            for ref, members in dynamic_groups:
                ref_m = scene['meshes'][ref]
                # 用 t=0 先占位；主循环里每帧 update
                mats0 = S.build_dynamic_instance_matrices(
                    scene['meshes'], members, 0.0)
                gl3d.upload_instance_transforms(ref_m['id'], mats0)
                cols = S.build_dynamic_instance_colors(scene['meshes'], members)
                gl3d.upload_instance_colors(ref_m['id'], cols)
                for i in members:
                    dynamic_member_set.add(i)
                if verbose or _DYN_INST_DEBUG:
                    print(f"    · [dyn-inst] {ref_m['id']} ← {len(members)} 个动态 mesh")
        except Exception as e:
            print(f"  ⚠ [dyn-inst] 失败: {e}", file=sys.stderr)
            dynamic_groups = []
            dynamic_member_set = set()

    return (static_groups, static_member_set,
            dynamic_groups, dynamic_member_set)


def _setup_param_instances(gl3d, scene, verbose=False):
    """为参数化 mesh 上传几何体 + 占位 instance matrix VBO。

    返回 [(mesh_id, mesh, local_half), ...]

    param 是**功能**（用户用 $var 显式声明的渲染方式），
    不受 fastpath 控制。fastpath 只管 auto-inst。
    """
    S = sys.modules['scene3d']
    param_meshes = [m for m in scene['meshes'] if m.get('_param_instances')]
    if not param_meshes:
        return []

    groups = []
    for m in param_meshes:
        inst = m['_param_instances']
        count = inst['count']
        try:
            typ = m.get('type', 'cube')
            parts = (m.get('size') or '1').split()
            def _f(i, d=1.0):
                return float(parts[i]) if i < len(parts) else d
            color = m.get('color', (1, 1, 1))
            if typ == 'cube':
                sizes = [_f(i) for i in range(3)]
                fc = [color, tuple(c * 0.55 for c in color), color,
                      tuple(c * 0.55 for c in color),
                      tuple(c * 0.8 for c in color),
                      tuple(c * 0.45 for c in color)]
                gl3d.upload_mesh(m['id'], S.make_cube_vertices(tuple(sizes), fc))
            elif typ == 'sphere':
                gl3d.upload_mesh(m['id'], S.make_sphere_vertices(
                    _f(0), 16, 8, color))
            elif typ == 'cylinder':
                gl3d.upload_mesh(m['id'], S.make_cylinder_vertices(
                    _f(0), m.get('_height', _f(0) * 2.0), 24, color))
            elif typ == 'cone':
                gl3d.upload_mesh(m['id'], S.make_cone_vertices(
                    _f(0), m.get('_height', _f(0) * 2.0), 24, color))
            else:
                print(f"  ⚠ [param] mesh {m['id']} type={typ} 不支持批量",
                      file=sys.stderr)
                continue

            placeholder = np.tile(
                np.eye(4, dtype=np.float32), (count, 1, 1))
            gl3d.upload_instance_transforms(m['id'], placeholder)
            # 🆕 per-instance color（如果有 palette param）
            if inst.get('colors') is not None:
                gl3d.upload_instance_colors(m['id'], inst['colors'])
            lh = S._local_halfsize(m).astype(np.float32)
            groups.append((m['id'], m, lh))
            if verbose:
                print(f"    · [param] {m['id']} ← {count} 个参数化实例")
        except Exception as e:
            print(f"  ⚠ [param] {m['id']}: {e}", file=sys.stderr)
    return groups

def _prepare_cull_arrays(scene, grouped_indices):
    """预计算视锥剔除用的 numpy 数组。

    grouped_indices 里的 mesh 已被实例化分组覆盖 → 不参与单 mesh cull。

    返回 dict：
      n_mesh, static_idx, static_centers, static_halfs, dynamic_idx
    """
    S = sys.modules['scene3d']
    n_mesh = len(scene['meshes'])
    result = {
        'n_mesh': n_mesh,
        'static_idx': np.zeros(0, dtype=np.intp),
        'static_centers': np.zeros((0, 3), dtype=np.float32),
        'static_halfs': np.zeros((0, 3), dtype=np.float32),
        'dynamic_idx': np.zeros(0, dtype=np.intp),
    }
    if not _CULL_ENABLED or n_mesh == 0:
        return result
    s_idx, d_idx = [], []
    s_c, s_h = [], []
    for j, m in enumerate(scene['meshes']):
        if j in grouped_indices:
            continue
        if not m.get('anims'):
            if m.get('_instances'):
                c, h = S.instanced_mesh_world_aabb(m)
            else:
                c, h = S.mesh_world_aabb(m, 0.0)
            s_idx.append(j); s_c.append(c); s_h.append(h)
        else:
            d_idx.append(j)
    result['static_idx'] = np.array(s_idx, dtype=np.intp)
    result['dynamic_idx'] = np.array(d_idx, dtype=np.intp)
    if s_c:
        result['static_centers'] = np.array(s_c, dtype=np.float32)
        result['static_halfs'] = np.array(s_h, dtype=np.float32)
    return result


def _prepare_group_abbs(scene, groups, label='group'):
    """为自动实例化组计算整体 world AABB（用于组级剔除）。"""
    S = sys.modules['scene3d']
    if not groups:
        return np.zeros((0, 3), dtype=np.float32), \
               np.zeros((0, 3), dtype=np.float32)
    gc, gh = [], []
    for ref, members in groups:
        cs, hs = [], []
        for i in members:
            c, h = S.mesh_world_aabb(scene['meshes'][i], 0.0)
            cs.append(c); hs.append(h)
        cs = np.array(cs); hs = np.array(hs)
        cmin = (cs - hs).min(axis=0)
        cmax = (cs + hs).max(axis=0)
        gc.append((cmin + cmax) * 0.5)
        gh.append((cmax - cmin) * 0.5)
    return np.array(gc, dtype=np.float32), np.array(gh, dtype=np.float32)


def _render_space3d_to_mp4(proj, space_id, duration, out_path, W, H, fps,
                             verbose=True, t_offset=0.0):
    """
    用 GL3D 渲染 space3d 的所有帧 → 静音 MP4。
    处理 space3d 里 texture="asset:xxx" 的引用，把 2D 资产预渲染成 PNG 再上传。
    """
    t0 = time.time()
    sp = proj['spaces'].get(space_id)
    if not sp:
        print(f"  ⚠ [space3d] {space_id} 不存在", file=sys.stderr)
        return None

    if verbose:
        print(f"  · [3D] space={space_id}  {W}x{H}@{fps}  {duration:.2f}s")

    # 1. 预处理 space XML（把 asset:xxx 换成 data URI）
    xml_str = _prepare_space3d_xml(proj, space_id, t_preview=0.0,
                                    verbose=verbose)
    if not xml_str:
        return None

    try:
        import scene3d as S
    except ImportError as e:
        print(f"  ⚠ [space3d] 缺少 scene3d 模块: {e}", file=sys.stderr)
        return None

    scene = S.parse_scene_xml(xml_str, base_dir=str(proj['base_dir']))
    scene['width'] = W
    scene['height'] = H

    gl3d = _get_gl3d(W, H)
    if gl3d is None:
        print(f"  ⚠ [space3d] GL3D 初始化失败", file=sys.stderr)
        return None

    # ─── setup 1：几何 + 显式 instances ───
    inst_count = _upload_scene_geometry(gl3d, scene, verbose=verbose)

    # ─── setup 2：纹理（按 key 去重）───
    tex_count, _tex_refs = _upload_scene_textures(
        gl3d, scene, proj['base_dir'], verbose=verbose)

    # ─── setup 3：自动实例化分组（静态 + 动态）───
    (_inst_groups, _inst_member_set,
     _dyn_groups, _dyn_member_set) = _setup_instancing_groups(
        gl3d, scene, verbose=verbose)

    # ─── setup 3.5：参数化实例（新）───
    _param_groups = _setup_param_instances(gl3d, scene, verbose=verbose)
    _param_member_set = set()
    for _pid, _pm, _plh in _param_groups:
        for _j, _m in enumerate(scene['meshes']):
            if _m['id'] == _pid:
                _param_member_set.add(_j)
                break

    if verbose:
        inst_msg = f"  instances={inst_count}" if inst_count else ""
        ai_msg = f"  auto-groups={len(_inst_groups)}" if _inst_groups else ""
        di_msg = f"  dyn-groups={len(_dyn_groups)}" if _dyn_groups else ""
        pa_msg = f"  param-groups={len(_param_groups)}" if _param_groups else ""
        print(f"    · meshes={len(scene['meshes'])}  "
              f"lights={len(scene['lights'])}  textures={tex_count}"
              f"{inst_msg}{ai_msg}{di_msg}{pa_msg}")

    # ─── setup 4：视锥剔除数组 ───
    _all_grouped = _inst_member_set | _dyn_member_set | _param_member_set
    _cull = _prepare_cull_arrays(scene, _all_grouped)
    _n_mesh = _cull['n_mesh']
    _static_idx_arr  = _cull['static_idx']
    _dynamic_idx_arr = _cull['dynamic_idx']
    _static_centers  = _cull['static_centers']
    _static_halfs    = _cull['static_halfs']
    if verbose or _CULL_DEBUG:
        print(f"    · [cull] static={len(_static_idx_arr)}  "
              f"dynamic={len(_dynamic_idx_arr)}  grouped={len(_all_grouped)}")

    # ─── setup 5：实例化组的整体 AABB（组级剔除）───
    _group_abbs_c = np.zeros((0, 3), dtype=np.float32)
    _group_abbs_h = np.zeros((0, 3), dtype=np.float32)
    if _CULL_ENABLED and _inst_groups:
        _group_abbs_c, _group_abbs_h = _prepare_group_abbs(scene, _inst_groups)

    lights = [{'type': lt['type'], 'pos': lt['pos'], 'dir': lt['dir'],
               'color': lt['color'], 'intensity': lt['intensity'],
               'range': lt['range'], 'spot_cos': lt.get('spot_cos', 0.9)}
              for lt in scene['lights']]

    # ffmpeg：rawvideo stdin → 静音 MP4
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-f', 'rawvideo', '-pixel_format', 'rgba',
           '-video_size', f'{W}x{H}', '-framerate', str(fps), '-i', '-',
           '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
           '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart',
           str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    n_frames = max(1, int(round(duration * fps)))
    t_render = 0.0

    # 🆕 编码并行：writer 线程从 queue 拿 bytes 写 proc.stdin
    # 主线程只负责渲染 + glReadPixels → 立即塞 queue → 继续下一帧
    import queue as _queue
    import threading as _threading
    _frame_q = _queue.Queue(maxsize=2)
    _writer_err = [None]
    _writer_done = _threading.Event()

    def _writer_loop():
        try:
            while True:
                b = _frame_q.get()
                if b is None:
                    break
                proc.stdin.write(b)
        except Exception as e:
            _writer_err[0] = e
        finally:
            _writer_done.set()

    _writer_t = _threading.Thread(target=_writer_loop, daemon=True)
    _writer_t.start()

    try:
        for i in range(n_frames):
            t = i / fps
            t_global = t_offset + t
            ts = time.perf_counter()
            cam_pos, cam_look = S._camera_at(scene['camera'], t)
            proj_m = S.m_perspective(scene['camera']['fov'], W / H, 0.1, 100.0)
            view = S.m_lookat(cam_pos, cam_look, [0, 1, 0])

            # 🆕 动态实例组：每帧重建 per-instance matrix 并更新 VBO
            for _ref, _members in _dyn_groups:
                _mats = S.build_dynamic_instance_matrices(
                    scene['meshes'], _members, t)
                gl3d.update_instance_transforms(
                    scene['meshes'][_ref]['id'], _mats)

            # 🆕 参数化 mesh：批量算 pos/rot/scale → matrix → AABB → 剔除
            _param_visible = {}
            if _param_groups:
                _planes_param = S.frustum_planes(proj_m @ view)
                for _pid, _pm, _plh in _param_groups:
                    try:
                        _bat = S.batch_param_at(_pm, t)
                        if _bat is None:
                            continue
                        _pos = _bat.get('pos')
                        # scene3d 里 prop 名是 'rotate'，这里兼容 'rot' 和 'rotate'
                        _rot = _bat.get('rot')
                        if _rot is None:
                            _rot = _bat.get('rotate')
                        _scl = _bat.get('scale')
                        if _pos is None:
                            continue
                        _N = len(_pos)
                        if _rot is None: _rot = np.zeros((_N, 3), np.float32)
                        if _scl is None: _scl = np.ones((_N, 3), np.float32)
                        _mats = S.batch_model_matrices(_pos, _rot, _scl)
                        _c, _h = S.batch_param_aabb(_pos, _rot, _scl, _plh)
                        _vis = S.batch_aabb_visible(_c, _h, _planes_param)
                        # projectile 模式有 alive mask：落地的不画
                        if 'alive' in _bat:
                            _vis = _vis & _bat['alive']
                        _vis_n = int(_vis.sum())
                        if _vis_n > 0:
                            gl3d.update_instance_transforms(_pid, _mats[_vis])
                        _param_visible[_pid] = _vis_n
                    except Exception as _e:
                        print(f"  ⚠ [param] {_pid} 帧 {i}: {_e}",
                              file=sys.stderr)
                        _param_visible[_pid] = 0

            # 🆕 视锥剔除
            vis = None
            if _CULL_ENABLED and _n_mesh > 0:
                _planes = S.frustum_planes(proj_m @ view)
                vis = np.zeros(_n_mesh, dtype=bool)
                if len(_static_idx_arr):
                    vis[_static_idx_arr] = S.batch_aabb_visible(
                        _static_centers, _static_halfs, _planes)
                if len(_dynamic_idx_arr):
                    _nc = len(_dynamic_idx_arr)
                    _cd = np.empty((_nc, 3), dtype=np.float32)
                    _hd = np.empty((_nc, 3), dtype=np.float32)
                    for _k, _jj in enumerate(_dynamic_idx_arr):
                        _c, _h = S.mesh_world_aabb(scene['meshes'][_jj], t)
                        _cd[_k] = _c; _hd[_k] = _h
                    vis[_dynamic_idx_arr] = S.batch_aabb_visible(_cd, _hd, _planes)
                if _CULL_DEBUG and (i % 30 == 0 or i == n_frames - 1):
                    print(f"    · [cull] frame {i}: {int(vis.sum())}/{_n_mesh} visible")

            # 🆕 auto-inst 组整体可见性
            _group_vis = None
            if vis is not None and _group_abbs_c.shape[0] > 0:
                _group_vis = S.batch_aabb_visible(
                    _group_abbs_c, _group_abbs_h, _planes)

            gl3d.begin(clear_color=scene['bg'])
            gl3d.set_lights(lights)
            gl3d.set_shading(scene.get('shading', 'lit') == 'flat')

            # 🆕 先画 auto-inst 组（每组合并成一次 draw_instanced）
            for _gi, (_ref, _members) in enumerate(_inst_groups):
                if _group_vis is not None and not _group_vis[_gi]:
                    continue
                _ref_m = scene['meshes'][_ref]
                gl3d.draw_instanced(_ref_m['id'], proj_m @ view,
                                     textures=_ref_m.get('_tex_names', []))

            # 🆕 再画动态实例组（每帧更新过的 VBO）
            for _ref, _members in _dyn_groups:
                _ref_m = scene['meshes'][_ref]
                gl3d.draw_instanced(_ref_m['id'], proj_m @ view,
                                     textures=_ref_m.get('_tex_names', []))

            # 🆕 画参数化 mesh
            for _pid, _pm, _plh in _param_groups:
                _n_vis = _param_visible.get(_pid, 0)
                if _n_vis > 0:
                    gl3d.draw_instanced(_pid, proj_m @ view,
                                         count=_n_vis,
                                         textures=_pm.get('_tex_names', []))

            for mi, m in enumerate(scene['meshes']):
                if mi in _all_grouped:
                    continue
                if vis is not None and not vis[mi]:
                    continue
                # 🆕 instancing 分支
                if m.get('_instances'):
                    gl3d.draw_instanced(m['id'], proj_m @ view,
                                         textures=m.get('_tex_names', []))
                    continue
                pos, rot, scl = S._mesh_at(m, t)
                model = (S.m_translate(*pos)
                         @ S.m_rot_y(float(rot[1]))
                         @ S.m_rot_x(float(rot[0]))
                         @ S.m_rot_z(float(rot[2]))
                         @ S.m_scale(*scl))
                mvp = proj_m @ view @ model
                tex_names = m.get('_tex_names', [])
                deform = m.get('_deform')
                if deform:
                    w = float(m['size'].split()[0]) if m['size'] else 1.0
                    gl3d.set_deform(deform, t=t_global, width=w)
                else:
                    gl3d.set_deform(None)
                gl3d.draw(m['id'], mvp, model, textures=tex_names)

            arr = gl3d.end()
            t_render += time.perf_counter() - ts
            # 🆕 塞 queue，writer 线程异步写 ffmpeg
            _frame_q.put(arr.tobytes())

            if verbose and (i + 1) % 30 == 0:
                print(f"    · [3D] {i+1}/{n_frames} 帧  {t_render:.2f}s")
    finally:
        # 🆕 收尾：等 writer 线程吃完 queue，再关 stdin
        try:
            _frame_q.put(None)
            _writer_done.wait(timeout=300)
        except Exception:
            pass
        try:
            proc.stdin.close()
            proc.wait()
        except Exception:
            pass

    if _writer_err[0] is not None:
        print(f"    ⚠ writer 线程异常: {_writer_err[0]}", file=sys.stderr)

    total = time.time() - t0
    size_kb = out_path.stat().st_size / 1024 if out_path.exists() else 0
    print(f"    · [3D] 完成 {n_frames} 帧  GPU {t_render:.2f}s  "
          f"wall {total:.2f}s  {size_kb:.0f}KB")
    return out_path


def _parse_svg_vb(content, fallback_w, fallback_h):
    """提取 SVG 的 viewBox 宽高。返回 (w, h)。"""
    m = re.search(r'<svg[^>]*\bviewBox="([^"]+)"', content, re.I)
    if m:
        parts = re.split(r'[\s,]+', m.group(1).strip())
        if len(parts) == 4:
            try:
                return float(parts[2]), float(parts[3])
            except ValueError:
                pass
    m = re.search(r'<svg[^>]*\bwidth="([^"]+)"[^>]*\bheight="([^"]+)"',
                  content, re.I)
    if m:
        try:
            return float(re.sub(r'[^0-9.].*$', '', m.group(1))), \
                   float(re.sub(r'[^0-9.].*$', '', m.group(2)))
        except ValueError:
            pass
    return fallback_w, fallback_h


def _asset_to_svg_inner(proj, asset, tw, th):
    """把 asset 转成 (inner_svg_str, vb_w, vb_h)。

    返回的 vb_w/vb_h 是 inner 内容实际使用的坐标系尺寸。
    外层包装 SVG 时应该用它作为 viewBox，而不是 tw/th —— 否则内容
    会被错误缩放（例如 1200x800 的国旗塞进 768x512 的 viewBox）。
    """
    atype = asset.get('type', 'svg')

    if atype == 'text':
        content = escape(asset.get('content', ''))
        size = float(asset.get('font-size', 96))
        color = asset.get('color', 'white')
        family = _resolve_font_family(asset.get('font-family', 'sans-serif'))
        weight = asset.get('font-weight', 'normal')
        inner = (f'<text x="50%" y="50%" font-family="{family}" '
                 f'font-size="{size}" font-weight="{weight}" '
                 f'fill="{color}" text-anchor="middle" '
                 f'dominant-baseline="middle">{content}</text>')
        return inner, tw, th

    if atype == 'svg':
        content = asset.get('content', '')
        vb_w, vb_h = _parse_svg_vb(content, tw, th)
        m = re.search(r'<svg[^>]*>(.*)</svg>', content, re.S)
        inner = m.group(1) if m else content
        return inner, vb_w, vb_h

    if atype == 'image':
        data = _safe_read_bytes(asset.get('src', ''),
                                 label=f"asset {asset.get('id', '?')}")
        if data is None:
            return '', tw, th
        ext = Path(asset['src']).suffix.lstrip('.').lower()
        mime = {'png':'image/png','jpg':'image/jpeg','jpeg':'image/jpeg',
                'gif':'image/gif','webp':'image/webp'}.get(ext, 'image/png')
        b64 = base64.b64encode(data).decode()
        inner = (f'<image xlink:href="data:{mime};base64,{b64}" '
                 f'href="data:{mime};base64,{b64}" '
                 f'width="{tw}" height="{th}"/>')
        return inner, tw, th

    if atype == 'latex':
        try:
            b64, (iw, ih) = latex_png(asset.get('content', ''), 60,
                                       proj.get('workdir'))
            inner = (f'<image xlink:href="data:image/png;base64,{b64}" '
                     f'href="data:image/png;base64,{b64}" '
                     f'x="{(tw-iw)/2}" y="{(th-ih)/2}" '
                     f'width="{iw}" height="{ih}"/>')
            return inner, tw, th
        except Exception as e:
            print(f"  ⚠ [space3d] latex 资产 {asset.get('id')}: {e}",
                  file=sys.stderr)
            return '', tw, th

    return '', tw, th


def _prepare_space3d_xml(proj, space_id, t_preview=0.0, verbose=False):
    """把 space3d 里的 texture="asset:xxx" 预处理成 data URI"""
    sp = proj['spaces'].get(space_id)
    if not sp:
        return None
    root = ET.fromstring(sp['_elem_xml'])

    for mesh in root.findall('mesh'):
        tex = mesh.get('texture')
        if not tex or not tex.startswith('asset:'):
            continue
        asset_id = tex.split(':', 1)[1]
        asset = proj['assets'].get(asset_id)
        if not asset:
            print(f"  ⚠ [space3d] 纹理资产不存在: {asset_id}", file=sys.stderr)
            mesh.attrib.pop('texture', None)
            continue
        try:
            tw = int(mesh.get('tex-width', '512'))
            th = int(mesh.get('tex-height', '512'))
            cache_key = (asset_id, tw, th)
            uri = _TEX_URI_CACHE.get(cache_key)
            if uri is None:
                inner, vb_w, vb_h = _asset_to_svg_inner(proj, asset, tw, th)
                if not inner:
                    mesh.attrib.pop('texture', None)
                    continue
                svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
                       f'xmlns:xlink="http://www.w3.org/1999/xlink" '
                       f'width="{tw}" height="{th}" '
                       f'viewBox="0 0 {vb_w:g} {vb_h:g}">'
                       f'{inner}</svg>')
                import cairosvg
                png_bytes = cairosvg.svg2png(bytestring=svg.encode('utf-8'),
                                              output_width=tw, output_height=th)
                if verbose:
                    try:
                        from PIL import Image as _I
                        from io import BytesIO as _B
                        arr = np.array(_I.open(_B(png_bytes)).convert('RGBA'))
                        rgb_mean = arr[:, :, :3].mean()
                        a_mean = arr[:, :, 3].mean()
                        print(f"    · tex {asset_id:12s} {tw}x{th}  "
                              f"rgb_mean={rgb_mean:5.1f}  alpha_mean={a_mean:5.1f}")
                    except Exception:
                        pass
                b64 = base64.b64encode(png_bytes).decode()
                uri = f'data:image/png;base64,{b64}'
                _TEX_URI_CACHE[cache_key] = uri
            mesh.set('texture', uri)
            mesh.attrib.pop('tex-width', None)
            mesh.attrib.pop('tex-height', None)
        except Exception as e:
            print(f"  ⚠ [space3d] 纹理 {asset_id}: {e}", file=sys.stderr)
            mesh.attrib.pop('texture', None)

    return ET.tostring(root, encoding='unicode')



def render_chunk(proj, chunk, workers, chunk_dir, silent_audio=True,
                 video_bg_override=None, extra_overlay_clips=None):
    ci = chunk['idx']
    t_chunk = time.time()
    c0, c1 = chunk['start'], chunk['end']
    cdur = c1 - c0
    out_path = chunk_dir / f"chunk_{ci:03d}.mp4"
    tag = ' [3D+overlay]' if video_bg_override else ''
    print(f"\n▶ Chunk {ci}: [{c0:.2f}, {c1:.2f}] ({cdur:.2f}s){tag}")

    clips_by_id = {c['id']: c for t in proj['tracks'] for c in t['clips']}
    # 把内联 overlay 加入 clips_by_id（供后续统一处理）
    if extra_overlay_clips:
        for ec in extra_overlay_clips:
            clips_by_id[ec['id']] = ec

    video_ids = set(chunk['video_ids'])
    overlay_ids, audio_ids = set(), set()
    for tr in proj['tracks']:
        if tr['kind'] == 'overlay':
            for c in tr['clips']:
                if min(c['_end'], c1) > max(c['_start'], c0) + 1e-6:
                    overlay_ids.add(c['id'])
        elif tr['kind'] == 'audio' and not silent_audio:
            for c in tr['clips']:
                if min(c['_end'], c1) > max(c['_start'], c0) + 1e-6:
                    audio_ids.add(c['id'])

    # 🆕 加入内联 overlay
    if extra_overlay_clips:
        for ec in extra_overlay_clips:
            if min(ec['_end'], c1) > max(ec['_start'], c0) + 1e-6:
                overlay_ids.add(ec['id'])

    inputs, aidx = [], {}
    for aid, a in proj['assets'].items():
        t = a.get('type')
        try:
            if t == 'color':
                aidx[aid] = len(inputs)
                inputs.append(['-f', 'lavfi', '-i',
                               f"color=c={a.get('value','black')}:"
                               f"s={proj['w']}x{proj['h']}:r={proj['fps']}"])
            elif t == 'video':
                if not Path(a.get('src', '')).exists(): continue
                aidx[aid] = len(inputs); inputs.append(['-i', a['src']])
            elif t == 'image':
                if not Path(a.get('src', '')).exists(): continue
                aidx[aid] = len(inputs)
                inputs.append(['-loop', '1', '-i', a['src']])
        except Exception as e:
            print(f"  ⚠ 素材 {aid} 失败: {e}", file=sys.stderr)

    filters, streams = [], {}

    # ---- video clip ----
    for cid in chunk['video_ids']:
        c = clips_by_id[cid]
        a = proj['assets'].get(c.get('asset')) if c.get('asset') else None
        _s = max(c['_start'], c0)
        _e = min(c['_end'], c1)
        dur = _e - _s
        clip_in_off = _s - c['_start']
        dynamic = _is_dynamic_clip(proj, c)
        try:
            if dynamic:
                label = render_clip_segments(proj, c, a, workers, inputs,
                                             filters, chunk_idx=ci,
                                             chunk_window=(c0, c1))
                if label is None: continue
            else:
                label = f"c_{cid}_{ci}"
                if c.get('asset') not in aidx: continue
                idx = aidx[c['asset']]
                ow = proj.get('_out_w') or proj['w']
                oh = proj.get('_out_h') or proj['h']
                filters.append(
                    f"[{idx}:v]trim=start={clip_in_off}:duration={dur},"
                    f"setpts=PTS-STARTPTS,"
                    f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,"
                    f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2,"
                    f"setsar=1,fps={proj['fps']},format=rgba[{label}]")
            streams[cid] = {'id': cid, 'label': label, 'clip': c,
                            'kind': 'video', 'dur': dur}
        except Exception as e:
            print(f"  ⚠ video {cid} 失败: {e}", file=sys.stderr)

    if video_bg_override is not None:
        # 🆕 3D chunk：直接用渲染好的 space3d MP4 当底图
        idx = len(inputs)
        inputs.append(['-i', str(video_bg_override)])
        ow = proj.get('_out_w') or proj['w']
        oh = proj.get('_out_h') or proj['h']
        filters.append(f"[{idx}:v]trim=duration={cdur},setpts=PTS-STARTPTS,"
                       f"scale={ow}:{oh},setsar=1,fps={proj['fps']},"
                       f"format=rgba[base]")
        last = 'base'
    else:
        vlist = sorted([s for s in streams.values() if s['kind'] == 'video'],
                       key=lambda s: s['clip']['_start'])
        if not vlist:
            ow = proj.get('_out_w') or proj['w']
            oh = proj.get('_out_h') or proj['h']
            filters.append(f"color=c=black:s={ow}x{oh}:"
                           f"r={proj['fps']}:d={cdur},format=rgba[base]")
        else:
            try:
                bl = build_video_chain(filters, vlist, proj['_transitions'], ci)
                filters.append(f"[{bl}]null[base]")
            except Exception as e:
                print(f"  ⚠ 视频链失败: {e}", file=sys.stderr)
                filters.append(f"[{vlist[0]['label']}]null[base]")
        last = 'base'

    # ---- 合并检测 ----
    merge_groups = []; cid_to_group = {}
    merge_agg = proj.get('_merge_aggressive', 2)
    if proj.get('_exp_merge_svg'):
        try:
            for group in find_mergeable_dynamic(proj, overlay_ids, clips_by_id,
                                                 (c0, c1), merge_agg):
                gi = len(merge_groups); merge_groups.append((group, False))
                for g in group: cid_to_group[g['id']] = gi
        except Exception as e:
            print(f"  ⚠ 动态合并检测失败: {e}", file=sys.stderr)
    if proj.get('_exp_merge_static'):
        try:
            for group in find_mergeable_static(proj, overlay_ids, clips_by_id,
                                                (c0, c1), merge_agg):
                gi = len(merge_groups); merge_groups.append((group, True))
                for g in group: cid_to_group[g['id']] = gi
        except Exception as e:
            print(f"  ⚠ 静态合并检测失败: {e}", file=sys.stderr)
    if merge_groups:
        dyn_n = sum(1 for _, st in merge_groups if not st)
        stat_n = sum(1 for _, st in merge_groups if st)
        print(f"  🧪 合并: {dyn_n} 动态组 + {stat_n} 静态组，"
              f"覆盖 {len(cid_to_group)} 个 clip (激进={merge_agg})")

    # ---- overlay ----
    overlay_order = sorted(overlay_ids, key=lambda x: clips_by_id[x]['_start'])
    processed_groups = set()

    for i, cid in enumerate(overlay_order):
        c = clips_by_id[cid]
        a = proj['assets'].get(c.get('asset')) if c.get('asset') else None
        s = max(c['_start'], c0) - c0
        e = min(c['_end'], c1) - c0

        try:
            if cid in cid_to_group:
                gi = cid_to_group[cid]
                if gi in processed_groups: continue
                group, static_only = merge_groups[gi]
                merged_lbl = render_group_merged(proj, group, workers, inputs,
                                                  filters, chunk_idx=ci,
                                                  static_only=static_only)
                processed_groups.add(gi)
                if merged_lbl is None:
                    print(f"  ⚠ 合并组 {gi} 失败，回退独立", file=sys.stderr)
                    for g in group:
                        gc, ga = g['clip'], g['asset']
                        gs = max(gc['_start'], c0) - c0
                        ge = min(gc['_end'], c1) - c0
                        try:
                            glbl = render_clip_segments(proj, gc, ga, workers,
                                                        inputs, filters,
                                                        chunk_idx=ci,
                                                        chunk_window=(c0, c1))
                            if glbl is None: continue
                            filters.append(f"[{glbl}]setpts=PTS+{gs}/TB[ov{i}]")
                            filters.append(f"[{last}][ov{i}]overlay=0:0:"
                                           f"enable='between(t,{gs},{ge})'[mix{i}]")
                            last = f'mix{i}'
                        except Exception as e2:
                            print(f"  ⚠ 回退 {g['id']} 失败: {e2}",
                                  file=sys.stderr)
                    continue
                gs = group[0].get('_group_start', group[0]['start']) - c0
                ge = group[0].get('_group_end', group[0]['end']) - c0
                filters.append(f"[{merged_lbl}]setpts=PTS+{gs}/TB[ov{i}]")
                filters.append(f"[{last}][ov{i}]overlay=0:0:"
                               f"enable='between(t,{gs},{ge})'[mix{i}]")
                last = f'mix{i}'
                continue

            dynamic = _is_dynamic_clip(proj, c)
            if dynamic:
                lbl = render_clip_segments(proj, c, a, workers, inputs,
                                            filters, chunk_idx=ci,
                                            chunk_window=(c0, c1))
                if lbl is None: continue
            else:
                if c.get('asset') not in aidx: continue
                idx = aidx[c['asset']]
                lbl = f"ov_static_{cid}_{ci}"
                dur = clip_duration(proj, c)
                # 【画中画】—— 如果 style 里有 w/h，用它们缩放
                st = c.get('_style', {})
                try: target_w = float(st.get('w')) if st.get('w') else None
                except (ValueError, TypeError): target_w = None
                try: target_h = float(st.get('h')) if st.get('h') else None
                except (ValueError, TypeError): target_h = None
                if target_w and target_h:
                    filters.append(
                        f"[{idx}:v]trim=duration={dur},setpts=PTS-STARTPTS,"
                        f"scale={int(target_w)}:{int(target_h)}:force_original_aspect_ratio=decrease,"
                        f"setsar=1,fps={proj['fps']},format=rgba[{lbl}]")
                else:
                    ow = proj.get('_out_w') or proj['w']
                    oh = proj.get('_out_h') or proj['h']
                    filters.append(
                        f"[{idx}:v]trim=duration={dur},setpts=PTS-STARTPTS,"
                        f"scale={ow}:{oh},setsar=1,fps={proj['fps']},"
                        f"format=rgba[{lbl}]")

            # 【画中画位置】 —— 用 x/y 而非绝对 0:0
            ov_x, ov_y = 0, 0
            if not dynamic:
                st = c.get('_style', {})
                try:
                    if st.get('x') is not None and st.get('w') is not None:
                        ov_x = int(float(st.get('x')) - float(st.get('w')) / 2)
                    if st.get('y') is not None and st.get('h') is not None:
                        ov_y = int(float(st.get('y')) - float(st.get('h')) / 2)
                except (ValueError, TypeError): pass
            filters.append(f"[{lbl}]setpts=PTS+{s}/TB[ov{i}]")
            filters.append(f"[{last}][ov{i}]overlay={ov_x}:{ov_y}:"
                           f"enable='between(t,{s},{e})'[mix{i}]")
            last = f'mix{i}'
        except Exception as e:
            print(f"  ⚠ overlay {cid} 失败: {e}", file=sys.stderr)

    # ---- audio：silent 时跳过 ----
    audio_labels = []
    if not silent_audio:
        for cid in sorted(audio_ids, key=lambda x: clips_by_id[x]['_start']):
            c = clips_by_id[cid]
            a = proj['assets'].get(c.get('asset'))
            if not a or 'src' not in a: continue
            if not Path(a['src']).exists(): continue
            try:
                st = c.get('_style', {})
                loop = c.get('loop', 'false').lower() == 'true'
                volume = float(st.get('volume', 1))
                fade_in = float(st.get('fade-in', 0))
                fade_out = float(st.get('fade-out', 0))
                s, e = max(c['_start'], c0), min(c['_end'], c1)
                ico = s - c0; iclip = s - c['_start']; sdur = e - s
                ai = len(inputs)
                if loop: inputs.append(['-stream_loop', '-1', '-i', a['src']])
                else: inputs.append(['-i', a['src']])
                lbl = f"a_{cid}_{ci}"
                chain = [f"atrim={iclip}:{iclip + sdur}", "asetpts=PTS-STARTPTS"]
                if volume != 1.0: chain.append(f"volume={volume}")
                if fade_in > 0 and iclip < fade_in:
                    chain.append(f"afade=t=in:st=0:d={fade_in - iclip}")
                if fade_out > 0:
                    cer = clip_duration(proj, c) - fade_out
                    if e > c['_start'] + cer:
                        chain.append(f"afade=t=out:st={max(0, cer - iclip)}:d={fade_out}")
                dm = int(ico * 1000)
                if dm > 0: chain.append(f"adelay={dm}|{dm}")
                filters.append(f"[{ai}:a]" + ",".join(chain) + f"[{lbl}]")
                audio_labels.append(lbl)
            except Exception as e:
                print(f"  ⚠ 音频 {cid} 失败: {e}", file=sys.stderr)

    # ---- 组装命令 ----
    v_args, a_args = _chunk_encode_args(proj)
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
    for inp in inputs: cmd += inp
    cmd += ['-filter_complex', ';'.join(filters),
            '-map', f'[{last}]', '-t', str(cdur)]
    cmd += v_args
    if Env.has_stillimage: cmd += ['-tune', 'stillimage']
    cmd += ['-r', str(proj['fps'])]

    if silent_audio:
        # 静音模式：不生成音轨
        cmd += ['-an', str(out_path)]
    else:
        # 原有音频处理
        if len(audio_labels) == 1:
            filters.append(f"[{audio_labels[0]}]anull[aout]")
        elif len(audio_labels) > 1:
            mix = "".join(f"[{l}]" for l in audio_labels)
            filters.append(f"{mix}amix=inputs={len(audio_labels)}:"
                           f"duration=longest:normalize=0[aout]")
        else:
            silent_idx = len(inputs)
            inputs.append(['-f', 'lavfi', '-t', str(cdur),
                           '-i', 'anullsrc=r=44100:cl=stereo'])
            filters.append(f"[{silent_idx}:a]anull[aout]")
        # 重新组装（因为 filters 变了）
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
        for inp in inputs: cmd += inp
        cmd += ['-filter_complex', ';'.join(filters),
                '-map', f'[{last}]', '-map', '[aout]', '-t', str(cdur)]
        cmd += v_args
        if Env.has_stillimage: cmd += ['-tune', 'stillimage']
        cmd += ['-r', str(proj['fps'])]
        cmd += a_args
        cmd += ['-shortest', str(out_path)]

    rc, _, err = safe_run(cmd, timeout=1800)

    if proj.get('_raw_cleanup'):
        for rp in proj['_raw_cleanup']:
            try: os.unlink(rp)
            except Exception: pass
        proj['_raw_cleanup'] = []

    if rc != 0:
        print(f"  ✗ Chunk {ci} 失败: {err[-500:]}", file=sys.stderr)
        print(f"[TIME] chunk_{ci:03d} FAIL {time.time()-t_chunk:.2f}s")
        return out_path, False
    print(f"  ✓ Chunk {ci} → {out_path.name}")
    print(f"[TIME] chunk_{ci:03d} OK {time.time()-t_chunk:.2f}s")
    return out_path, True



# 子进程入口

def _render_chunk_subprocess(args):
    (xml_path, base_dir, chunk_idx, workers, chunk_dir_str,
     workroot_str, max_chunk, quality,
     exp_encode, exp_merge_svg, exp_merge_static, merge_agg,
     silent_audio) = args
    workroot = Path(workroot_str); chunk_dir = Path(chunk_dir_str)
    try: Env.probe(workroot)
    except Exception as e:
        print(f"  ⚠ 子进程环境探测失败: {e}", file=sys.stderr)
    try:
        proj = parse_xml(xml_path, base_dir=base_dir)
        proj = resolve(proj)
        p = QUALITY_PRESETS.get(quality)
        if p:
            proj['_out_w'] = p.get('w') or proj['w']
            proj['_out_h'] = p.get('h') or proj['h']
            if p.get('fps'): proj['fps'] = p['fps']
        else:
            proj['_out_w'] = proj['w']
            proj['_out_h'] = proj['h']
        proj['workdir'] = workroot
        proj['_exp_encode'] = exp_encode
        proj['_exp_merge_svg'] = exp_merge_svg
        proj['_exp_merge_static'] = exp_merge_static
        proj['_merge_aggressive'] = merge_agg
        proj['_quality'] = quality
        # Autocairo：从环境变量读
        proj['_exp_autocairo'] = os.environ.get('XMLVE_AUTOCAIRO') == '1'
        _as = os.environ.get('XMLVE_AUTO_SPLIT', '').strip()
        try:
            proj['_auto_split_sec'] = float(_as) if _as else None
        except ValueError:
            proj['_auto_split_sec'] = None
        proj['_exp_pass_png'] = os.environ.get('XMLVE_PASS_PNG') == '1'
        try:
            proj['_autocairo_min'] = int(os.environ.get('XMLVE_AUTOCAIRO_MIN', '30'))
        except ValueError:
            proj['_autocairo_min'] = 30
    except Exception as e:
        print(f"  ✗ 子进程解析失败: {e}", file=sys.stderr)
        return (None, False)
    try:
        _mw = int(os.environ.get('XMLVE_MAX_CHUNK_WEIGHT', '400'))
        _ms = float(os.environ.get('XMLVE_MERGE_CHUNK_SEC', '18.0'))
        chunks = build_chunks(proj, max_chunk_sec=max_chunk,
                              auto_split_sec=proj.get('_auto_split_sec'),
                              max_weight=_mw, merge_sec=_ms)
        if chunk_idx >= len(chunks): return (None, False)
        ch = chunks[chunk_idx]

        # 🆕 3D space chunk：两阶段
        if ch.get('type') == 'space3d':
            W = proj.get('_out_w') or proj['w']
            H = proj.get('_out_h') or proj['h']
            fps = proj['fps']
            space_mp4 = chunk_dir / f"space3d_{ch['idx']:03d}.mp4"
            _render_space3d_to_mp4(proj, ch['space_id'],
                                    ch['end'] - ch['start'],
                                    space_mp4, W, H, fps,
                                    t_offset=ch['start'])
            if not space_mp4.exists():
                print(f"  ✗ [space3d] 渲染失败", file=sys.stderr)
                return (None, False)
            # 收集内联 overlay
            inline_clips = _build_inline_overlays_for_chunk(proj, ch)
            if inline_clips:
                print(f"  · [space3d] {len(inline_clips)} 个内联 overlay")
            return render_chunk(proj, ch, workers, chunk_dir,
                                 silent_audio=silent_audio,
                                 video_bg_override=space_mp4,
                                 extra_overlay_clips=inline_clips)

        # 标准 2D chunk
        return render_chunk(proj, ch, workers, chunk_dir,
                             silent_audio=silent_audio)
    except Exception as e:
        print(f"  ✗ 子进程 chunk {chunk_idx} 异常: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return (None, False)



# 验证 + 合并

def _first_clip_id_of_chunk(proj, meta):
    if meta.get('type') == 'space3d':
        return meta.get('space_clip_id')
    ids = meta.get('video_ids') or []
    return ids[0] if ids else None


def _last_clip_id_of_chunk(proj, meta):
    if meta.get('type') == 'space3d':
        return meta.get('space_clip_id')
    ids = meta.get('video_ids') or []
    return ids[-1] if ids else None


def _probe_duration(path):
    rc, out, _ = safe_run(['ffprobe', '-v', 'error',
                            '-show_entries', 'format=duration',
                            '-of', 'csv=p=0', str(path)], timeout=10)
    if rc == 0 and out.strip():
        try: return float(out.strip())
        except ValueError: pass
    return 0.0


def _concat_chunks_xfade(chunk_paths, chunk_meta, proj, out_path):
    """用 xfade 拼接多个 chunk（相邻 chunk 之间有 transition）"""
    fps = proj.get('fps', 30)
    W = proj.get('_out_w') or proj['w']
    H = proj.get('_out_h') or proj['h']

    durations = [_probe_duration(p) for p in chunk_paths]
    if any(d <= 0 for d in durations):
        print(f"  ⚠ 无法获取 chunk 时长，回退 concat", file=sys.stderr)
        listfile = chunk_paths[0].parent / 'concat.txt'
        listfile.write_text("\n".join(f"file '{p.resolve()}'" for p in chunk_paths))
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-f', 'concat', '-safe', '0', '-i', str(listfile),
               '-c', 'copy', '-movflags', '+faststart', str(out_path)]
        rc, _, err = safe_run(cmd, timeout=600)
        return rc == 0

    # 构建 ffmpeg xfade filter graph
    inputs = []
    for p in chunk_paths:
        inputs += ['-i', str(p)]

    filter_parts = []
    prev_label = '0:v'
    cumulative = durations[0]  # 已累积的总时长

    for i in range(1, len(chunk_paths)):
        ma, mb = chunk_meta[i - 1], chunk_meta[i]
        a_last = _last_clip_id_of_chunk(proj, ma)
        b_first = _first_clip_id_of_chunk(proj, mb)
        trans = proj.get('_transitions', {}).get((a_last, b_first))

        if trans:
            d = float(trans['duration'])
            eff = trans.get('effect', 'fade')
            # xfade offset = 前一段累积时长 - xfade duration
            offset = max(0.0, cumulative - d)
            new_label = f'xf{i}'
            filter_parts.append(
                f"[{prev_label}][{i}:v]xfade=transition={eff}:"
                f"duration={d}:offset={offset:.4f}[{new_label}]")
            prev_label = new_label
            cumulative = cumulative + durations[i] - d
        else:
            # 硬切
            new_label = f'cc{i}'
            filter_parts.append(
                f"[{prev_label}][{i}:v]concat=n=2:v=1:a=0[{new_label}]")
            prev_label = new_label
            cumulative += durations[i]

    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
    cmd += inputs
    cmd += ['-filter_complex', ';'.join(filter_parts),
            '-map', f'[{prev_label}]',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart',
            str(out_path)]
    rc, _, err = safe_run(cmd, timeout=1800)
    if rc != 0:
        print(f"  ✗ xfade 拼接失败: {err[-500:]}", file=sys.stderr)
        return False
    return True



def is_valid_chunk(path, need_audio=False):
    try:
        if not path.exists() or path.stat().st_size == 0: return False
    except Exception: return False
    if not Env.ffprobe: return True
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type',
           '-of', 'csv=p=0', str(path)]
    rc, out, _ = safe_run(cmd, timeout=15)
    if rc != 0: return False
    types = {s.strip() for s in out.splitlines() if s.strip()}
    if 'video' not in types: return False
    if need_audio and 'audio' not in types: return False
    return True


def _probe_stream_params(path):
    """返回 chunk 视频流的 (codec, w, h, pix_fmt, profile) tuple"""
    rc, out, _ = safe_run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=codec_name,width,height,pix_fmt,profile',
         '-of', 'csv=p=0', str(path)], timeout=10)
    if rc == 0 and out.strip():
        return out.strip()
    return None


def _chunks_same_codec(paths):
    """所有 chunk 的编码参数完全一致 → 可以 -c copy concat"""
    if len(paths) <= 1:
        return True
    params = set()
    for p in paths:
        q = _probe_stream_params(p)
        if q is None:
            return False
        params.add(q)
        if len(params) > 1:
            return False
    return True


def concat_chunks(chunk_paths, out_path, proj=None, aggressive=0,
                  chunk_meta=None):
    """合并所有 chunk 的【视频部分】（音频由 mix_global_audio 处理）。
    🆕 如果 chunk_meta 提供且相邻 chunk 之间有 transition，用 xfade 拼接。
    """
    quality = 'fast'
    if proj is not None: quality = proj.get('_quality', 'fast')
    p = QUALITY_PRESETS.get(quality, QUALITY_PRESETS['fast'])
    force_reencode = p['merge_reencode'] or aggressive >= 2

    # 🆕 智能检测：所有 chunk 编码参数一致
    # 默认：走一次 medium CRF23（小文件、画质更好）
    # XMLVE_MERGE_COPY=1：跳过重编码（快 5~10s，文件大 3~5 倍）
    if force_reencode:
        same = _chunks_same_codec(chunk_paths)
        env_copy = os.environ.get('XMLVE_MERGE_COPY', '').lower()
        if same and env_copy in ('1', 'true', 'yes'):
            print(f"  · XMLVE_MERGE_COPY=1，跳过重编码（-c copy，文件大）")
            force_reencode = False
        elif same:
            print(f"  · 所有 chunk 编码一致 → 走一次 medium CRF23（小文件）")

    valid = []
    valid_meta = []
    for i, pa in enumerate(chunk_paths):
        if not is_valid_chunk(pa, need_audio=False):
            print(f"  ⚠ 跳过无效 chunk: {pa}", file=sys.stderr)
            continue
        valid.append(pa)
        if chunk_meta and i < len(chunk_meta):
            valid_meta.append(chunk_meta[i])
        else:
            valid_meta.append(None)

    if not valid: return False

    # 🆕 检测相邻 chunk 是否有 transition
    has_any_xfade = False
    if proj and valid_meta:
        for i in range(len(valid) - 1):
            ma, mb = valid_meta[i], valid_meta[i + 1]
            if not ma or not mb: continue
            # 找 a 里最后一个 clip 和 b 里第一个 clip
            a_last = _last_clip_id_of_chunk(proj, ma)
            b_first = _first_clip_id_of_chunk(proj, mb)
            if a_last and b_first and (a_last, b_first) in proj.get('_transitions', {}):
                has_any_xfade = True
                break

    if has_any_xfade:
        print(f"  🎬 检测到跨 chunk transition，使用 xfade 拼接")
        return _concat_chunks_xfade(valid, valid_meta, proj, out_path)

    if len(valid) == 1 and not force_reencode:
        try: safe_replace(str(valid[0]), str(out_path)); return True
        except Exception:
            try: shutil.copy2(str(valid[0]), str(out_path)); return True
            except Exception as e2:
                print(f"  ✗ copy 失败: {e2}", file=sys.stderr); return False

    listfile = valid[0].parent / 'concat.txt'
    listfile.write_text("\n".join(f"file '{pa.resolve()}'" for pa in valid))

    if force_reencode:
        v_args, a_args = _merge_encode_args(proj, aggressive)
        print(f"  🧪 强制重编码 [{quality}] "
              f"(-preset {p['merge_preset']} -crf {p['merge_crf']}) ...")
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-f', 'concat', '-safe', '0', '-i', str(listfile)]
        cmd += v_args + ['-an', '-movflags', '+faststart', str(out_path)]
        t_re = time.time()
        rc, _, err = safe_run(cmd, timeout=3600)
        if rc != 0:
            print(f"  ✗ concat 重编码失败: {err[-300:]}", file=sys.stderr)
            return False
        print(f"  ✓ 重编码完成 ({time.time()-t_re:.1f}s)")
        return True

    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-f', 'concat', '-safe', '0', '-i', str(listfile),
           '-c', 'copy', '-movflags', '+faststart', str(out_path)]
    rc, _, err = safe_run(cmd, timeout=600)
    if rc == 0: return True
    print(f"  ⚠ concat copy 失败，重编码: {err[-200:]}", file=sys.stderr)
    v_args, a_args = _merge_encode_args(proj, aggressive)
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-f', 'concat', '-safe', '0', '-i', str(listfile)]
    cmd += v_args + ['-an', '-movflags', '+faststart', str(out_path)]
    rc, _, err = safe_run(cmd, timeout=3600)
    if rc != 0:
        print(f"  ✗ concat 失败: {err[-300:]}", file=sys.stderr); return False
    return True



# MP4 工具箱

def mp4_info(path):
    if not Path(path).exists():
        print(f"✗ 文件不存在: {path}", file=sys.stderr); return 1
    cmd = ['ffprobe', '-v', 'error', '-print_format', 'json',
           '-show_format', '-show_streams', str(path)]
    rc, out, err = safe_run(cmd, timeout=15)
    if rc != 0:
        print(f"✗ ffprobe 失败: {err}", file=sys.stderr); return 1
    try:
        data = json.loads(out)
    except Exception as e:
        print(f"✗ 解析失败: {e}", file=sys.stderr); return 1

    fmt = data.get('format', {})
    print(f"📹 {path}")
    print(f"  时长      {float(fmt.get('duration', 0)):.2f} s")
    print(f"  码率      {int(fmt.get('bit_rate', 0)) / 1000:.0f} kbps")
    print(f"  文件大小  {int(fmt.get('size', 0)) / 1024 / 1024:.2f} MB")

    for s in data.get('streams', []):
        codec_type = s.get('codec_type', '?')
        if codec_type == 'video':
            fps_s = s.get('r_frame_rate', '0/0')
            try:
                n, d = fps_s.split('/'); fps = float(n) / float(d) if float(d) else 0
            except Exception: fps = 0
            print(f"\n  🎞 Video")
            print(f"    编码      {s.get('codec_name','?')}")
            print(f"    分辨率    {s.get('width')}x{s.get('height')}")
            print(f"    帧率      {fps:.2f} fps")
            print(f"    像素格式  {s.get('pix_fmt','?')}")
            print(f"    总帧数    {s.get('nb_frames','?')}")
        elif codec_type == 'audio':
            print(f"\n  🔊 Audio")
            print(f"    编码      {s.get('codec_name','?')}")
            print(f"    采样率    {s.get('sample_rate','?')} Hz")
            print(f"    声道      {s.get('channels','?')}")
            print(f"    码率      {int(s.get('bit_rate', 0)) / 1000:.0f} kbps")
    return 0


def mp4_snapshot(path, at=None, frame=None, out=None, fmt='both'):
    if not Path(path).exists():
        print(f"✗ 文件不存在: {path}", file=sys.stderr); return 1
    if out is None:
        tag = f"t{at}" if at is not None else f"f{frame}"
        out = f"{Path(path).stem}_snap_{tag}"
    out_base = Path(out)
    if out_base.suffix.lower() in ('.png', '.svg'): out_base = out_base.with_suffix('')

    if frame is not None:
        # 按帧号：需要先算时间，或者用 select 滤镜
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-i', str(path), '-vf', f"select=eq(n\\,{frame})",
               '-vframes', '1', '-f', 'image2', '-']
        rc, out_bytes, err = safe_run(cmd, timeout=60)
        if rc != 0 or not out_bytes:
            print(f"✗ 抽帧失败: {err[-200:]}", file=sys.stderr); return 1
        written = []
        if fmt in ('png', 'both'):
            p = out_base.with_suffix('.png')
            try: p.write_bytes(out_bytes.encode('latin1') if isinstance(out_bytes, str) else out_bytes)
            except Exception:
                subprocess.run(['ffmpeg', '-y', '-loglevel', 'error',
                                '-i', str(path), '-vf', f"select=eq(n\\,{frame})",
                                '-vframes', '1', str(p)], timeout=60)
            print(f"  ✓ {p}"); written.append(str(p))
        return 0 if written else 1

    # 按时间
    t = at if at is not None else 0
    written = []
    if fmt in ('png', 'both'):
        p = out_base.with_suffix('.png')
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-ss', str(t), '-i', str(path), '-vframes', '1', str(p)]
        rc, _, err = safe_run(cmd, timeout=60)
        if rc != 0:
            print(f"  ✗ PNG 抽帧失败: {err[-200:]}", file=sys.stderr)
        else:
            print(f"  ✓ {p}"); written.append(str(p))
    if fmt in ('svg', 'both'):
        # 先抽 PNG，再包一层 SVG
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False); tmp.close()
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-ss', str(t), '-i', str(path), '-vframes', '1', tmp.name]
        rc, _, err = safe_run(cmd, timeout=60)
        if rc == 0:
            data = Path(tmp.name).read_bytes()
            b64 = base64.b64encode(data).decode()
            # 从 ffprobe 拿分辨率
            rc2, out2, _ = safe_run(['ffprobe', '-v', 'error',
                                      '-select_streams', 'v:0',
                                      '-show_entries', 'stream=width,height',
                                      '-of', 'csv=p=0', str(path)], timeout=10)
            try:
                w, h = [int(x) for x in out2.strip().split(',')[:2]]
            except Exception:
                w, h = 1280, 720
            svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
                   f'xmlns:xlink="http://www.w3.org/1999/xlink" '
                   f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
                   f'<image xlink:href="data:image/png;base64,{b64}" '
                   f'href="data:image/png;base64,{b64}" '
                   f'width="{w}" height="{h}"/></svg>')
            p = out_base.with_suffix('.svg')
            p.write_text(svg, encoding='utf-8')
            print(f"  ✓ {p}"); written.append(str(p))
        try: os.unlink(tmp.name)
        except Exception: pass
    return 0 if written else 1


def mp4_trim(path, range_str, out=None, precise=False):
    if not Path(path).exists():
        print(f"✗ 文件不存在: {path}", file=sys.stderr); return 1
    try:
        if ':' not in range_str:
            raise ValueError
        rs_s, re_s = range_str.split(':', 1)
        rs, re_ = float(rs_s), float(re_s)
    except Exception:
        print(f"✗ 区间格式: START:END，收到 {range_str}", file=sys.stderr); return 1
    if out is None: out = f"{Path(path).stem}_trim_{rs}_{re_}.mp4"
    dur = re_ - rs

    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-ss', str(rs), '-i', str(path), '-t', str(dur)]
    if precise:
        cmd += ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '192k']
    else:
        cmd += ['-c', 'copy']
    cmd += [str(out)]
    rc, _, err = safe_run(cmd, timeout=600)
    if rc != 0:
        print(f"✗ 裁剪失败: {err[-300:]}", file=sys.stderr); return 1
    print(f"  ✓ {out}")
    return 0


def mp4_volume(path, volume=1.0, out=None):
    if not Path(path).exists():
        print(f"✗ 文件不存在: {path}", file=sys.stderr); return 1
    if out is None: out = f"{Path(path).stem}_vol{volume}.mp4"
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-i', str(path), '-c:v', 'copy',
           '-af', f'volume={volume}',
           '-c:a', 'aac', '-b:a', '192k',
           '-movflags', '+faststart', str(out)]
    rc, _, err = safe_run(cmd, timeout=600)
    if rc != 0:
        print(f"✗ 音量调整失败: {err[-300:]}", file=sys.stderr); return 1
    print(f"  ✓ {out}")
    return 0


def mp4_transform(path, crop=None, scale=None, pos=None, out=None):
    """crop = W:H:X:Y | scale = W:H | pos = X:Y（配合 scale 用于 pad）"""
    if not Path(path).exists():
        print(f"✗ 文件不存在: {path}", file=sys.stderr); return 1
    if out is None: out = f"{Path(path).stem}_transformed.mp4"

    vf_parts = []
    if crop:
        try:
            cw, ch, cx, cy = [int(x) for x in crop.split(':')]
            vf_parts.append(f"crop={cw}:{ch}:{cx}:{cy}")
        except Exception:
            print(f"✗ crop 格式: W:H:X:Y", file=sys.stderr); return 1
    if scale:
        try:
            sw, sh = [int(x) for x in scale.split(':')]
            vf_parts.append(f"scale={sw}:{sh}")
            if pos:
                try:
                    px, py = [int(x) for x in pos.split(':')]
                    # 用 pad 实现"放大画布"效果
                    total_w = max(sw, px + sw)
                    total_h = max(sh, py + sh)
                    vf_parts.append(f"pad={total_w}:{total_h}:{px}:{py}")
                except Exception:
                    print(f"✗ pos 格式: X:Y", file=sys.stderr)
        except Exception:
            print(f"✗ scale 格式: W:H", file=sys.stderr); return 1

    if not vf_parts:
        print(f"✗ 需要 --crop 或 --scale", file=sys.stderr); return 1

    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-i', str(path), '-vf', ','.join(vf_parts),
           '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
           '-c:a', 'copy', '-movflags', '+faststart', str(out)]
    rc, _, err = safe_run(cmd, timeout=900)
    if rc != 0:
        print(f"✗ 变换失败: {err[-300:]}", file=sys.stderr); return 1
    print(f"  ✓ {out}")
    return 0



# main

def _pick_workdir(argv):
    if '--workdir' in argv:
        i = argv.index('--workdir')
        if i + 1 < len(argv): return Path(argv[i + 1]).resolve()
    env_wd = os.environ.get('XMLVIDEO_WORKDIR')
    if env_wd: return Path(env_wd).expanduser().resolve()
    return Path('./tmp/gappt_work').resolve()


def _apply_aggressive(level, opts):
    if level == 0:
        opts['exp_encode'] = False; opts['exp_merge_svg'] = False
        opts['exp_merge_static'] = False; opts['exp_pipeline'] = 1
    elif level == 1:
        opts['exp_encode'] = True; opts['exp_merge_svg'] = False
        opts['exp_merge_static'] = False; opts['exp_pipeline'] = 1
    elif level == 2:
        opts['exp_encode'] = True; opts['exp_merge_svg'] = True
        opts['exp_merge_static'] = True; opts['exp_pipeline'] = 1
    else:
        opts['exp_encode'] = True; opts['exp_merge_svg'] = True
        opts['exp_merge_static'] = True; opts['exp_pipeline'] = 0  # auto


def _print_banner(opts, level=None):
    lines = []
    q = opts.get('quality', 'fast')
    qinfo = QUALITY_PRESETS.get(q, {}).get('desc', '?')
    lines.append(f"  · 渲染精度     {q}  ({qinfo})")
    lines.append(f"  · 音频合成     延迟到最终（分块静音）")
    g2d = opts.get('exp_gpu_2d', 'auto')
    if g2d != 'auto':
        lines.append(f"  · GPU 2D       {g2d}")
    if opts.get('exp_bmp'):
        lines.append(f"  · 帧格式       BMP (更快)")
    if opts.get('exp_merge_svg'):
        lines.append(f"  · SVG合并      激进级别 {opts.get('merge_aggressive', 2)}")
    if opts.get('exp_merge_static'):
        lines.append("  · 静态合并     纯静态 overlay 合并")
    if opts.get('exp_autocairo'):
        lines.append(f"  · Autocairo    静止段(≥{opts.get('autocairo_min', 30)}帧)首帧用 cairo")
    if opts.get('exp_auto_split'):
        lines.append(f"  · AutoSplit    单 clip >{float(opts['exp_auto_split']):.0f}s 自动切分")
    if opts.get('exp_pass_png'):
        lines.append("  · PassPNG      跳过 PNG，rawvideo 直通")
    if opts.get('exp_qsv'):
        lines.append("  · QSV 硬件编码 (实验性)")
    if opts.get('exp_nvenc'):
        lines.append("  · NVENC 硬件编码 (实验性)")
    if opts['exp_pipeline'] > 1:
        lines.append(f"  · 流水线并行   {opts['exp_pipeline']} 进程")
    level_str = f" (全局级别 {level})" if level is not None else ""
    print(f"🧪 配置{level_str}：")
    for l in lines: print(l)
    print()


def _parse_time_range(s):
    if ':' not in s:
        raise ValueError(f"区间格式应为 START:END，收到: {s}")
    parts = s.split(':', 1)
    return float(parts[0]), float(parts[1])


# ---- MP4 工具箱：子命令检测 ----
def _is_mp4_mode(argv):
    keys = ('--mp4-info', '--mp4-snapshot', '--mp4-trim',
            '--mp4-volume', '--mp4-transform')
    return any(k in argv for k in keys)


def _run_mp4_mode(argv):
    """处理所有 --mp4-* 子命令。"""
    if '--mp4-info' in argv:
        i = argv.index('--mp4-info')
        p = argv[i + 1] if i + 1 < len(argv) else None
        if not p:
            print("✗ 用法: --mp4-info video.mp4", file=sys.stderr); return 1
        return mp4_info(p)

    if '--mp4-snapshot' in argv:
        i = argv.index('--mp4-snapshot')
        p = argv[i + 1] if i + 1 < len(argv) else None
        if not p:
            print("✗ 用法: --mp4-snapshot video.mp4 [--at=3.5] [--frame=100] [--format=png]", file=sys.stderr)
            return 1
        at = frame = None; out = None; fmt = 'both'
        for a in argv:
            if a.startswith('--at='): at = float(a.split('=', 1)[1])
            elif a.startswith('--frame='): frame = int(a.split('=', 1)[1])
            elif a.startswith('--format='): fmt = a.split('=', 1)[1]
            elif a.startswith('--out='): out = a.split('=', 1)[1]
        if at is None and frame is None:
            at = 0.0
        return mp4_snapshot(p, at=at, frame=frame, out=out, fmt=fmt)

    if '--mp4-trim' in argv:
        i = argv.index('--mp4-trim')
        p = argv[i + 1] if i + 1 < len(argv) else None
        if not p:
            print("✗ 用法: --mp4-trim video.mp4 --range=10:20 [--out=cut.mp4] [--precise]", file=sys.stderr)
            return 1
        rng = None; out = None; precise = '--precise' in argv
        for a in argv:
            if a.startswith('--range='): rng = a.split('=', 1)[1]
            elif a.startswith('--out='): out = a.split('=', 1)[1]
        if not rng:
            print("✗ 需要 --range=START:END", file=sys.stderr); return 1
        return mp4_trim(p, rng, out=out, precise=precise)

    if '--mp4-volume' in argv:
        i = argv.index('--mp4-volume')
        p = argv[i + 1] if i + 1 < len(argv) else None
        if not p:
            print("✗ 用法: --mp4-volume video.mp4 --volume=0.5 [--out=x.mp4]", file=sys.stderr)
            return 1
        vol = 1.0; out = None
        for a in argv:
            if a.startswith('--volume='): vol = float(a.split('=', 1)[1])
            elif a.startswith('--out='): out = a.split('=', 1)[1]
        return mp4_volume(p, volume=vol, out=out)

    if '--mp4-transform' in argv:
        i = argv.index('--mp4-transform')
        p = argv[i + 1] if i + 1 < len(argv) else None
        if not p:
            print("✗ 用法: --mp4-transform video.mp4 --crop=W:H:X:Y --scale=W:H [--pos=X:Y] [--out=x.mp4]", file=sys.stderr)
            return 1
        crop = scale = pos = out = None
        for a in argv:
            if a.startswith('--crop='): crop = a.split('=', 1)[1]
            elif a.startswith('--scale='): scale = a.split('=', 1)[1]
            elif a.startswith('--pos='): pos = a.split('=', 1)[1]
            elif a.startswith('--out='): out = a.split('=', 1)[1]
        return mp4_transform(p, crop=crop, scale=scale, pos=pos, out=out)

    print("✗ 未知 MP4 模式", file=sys.stderr); return 1


def main():
    argv = sys.argv[1:]

    # MP4 工具箱分流
    if _is_mp4_mode(argv):
        # 需要 ffprobe 探测
        try: Env.probe(Path('./tmp/gappt_work'))
        except Exception: pass
        if not Env.ffmpeg:
            print("✗ 缺少 ffmpeg", file=sys.stderr); return 1
        return _run_mp4_mode(argv)

    opts = {
        'keep': False, 'preview': False, 'resume': False, 'purge': False,
        'workers': min(4, os.cpu_count() or 1),
        'max_chunk': 12.0,
        'exp_encode': False, 'exp_pipeline': 0,  # 0 = auto
        'exp_merge_svg': False, 'exp_merge_static': False,
        'exp_qsv': False, 'exp_nvenc': False,
        'exp_gpu_2d': 'auto',   # auto|force|off
        'exp_autocairo': False,   # 长静止段用 cairo 渲染（质量优先）
        'exp_auto_split': False,  # 自动切超长 clip
        'exp_pass_png': False,    # 跳过 PNG，rawvideo 直通
        'max_chunk_weight': 400,   # 单 chunk mesh/element 上限
        'merge_chunk_sec': 18.0,   # 小于此时长的相邻 chunk 尝试合并
        'autocairo_min': 30,      # 静止段最小帧数阈值（默认 30 = 1s @30fps）
        'exp_bmp': False,       # 帧中间格式用 BMP（更快，但更大）
        'aggressive': None, 'merge_aggressive': None,
        'quality': 'fast',
        'render_frame': None, 'render_range': None,
        'format': 'both', 'out': None,
        'silent_audio': True,
    }
    skip_next = False
    for i, a in enumerate(argv):
        if skip_next: skip_next = False; continue
        if a == '--keep': opts['keep'] = True
        elif a == '--preview': opts['preview'] = True
        elif a == '--resume': opts['resume'] = True
        elif a == '--purge': opts['purge'] = True
        elif a == '--stable':
            _apply_aggressive(0, opts)
            opts['aggressive'] = 0; opts['merge_aggressive'] = 0
        elif a.startswith('--quality='):
            q = a.split('=', 1)[1].lower()
            if q not in QUALITY_PRESETS:
                print(f"  ⚠ 未知 quality: {q}，使用 fast", file=sys.stderr)
            else: opts['quality'] = q
        elif a.startswith('--aggressive='):
            try:
                lv = max(0, min(3, int(a.split('=', 1)[1])))
                opts['aggressive'] = lv; _apply_aggressive(lv, opts)
            except ValueError: print(f"  ⚠ --aggressive 非法", file=sys.stderr)
        elif a.startswith('--merge-aggressive='):
            try:
                lv = max(0, min(3, int(a.split('=', 1)[1])))
                opts['merge_aggressive'] = lv
            except ValueError: print(f"  ⚠ --merge-aggressive 非法", file=sys.stderr)
        elif a == '--exp-encode': opts['exp_encode'] = True
        elif a == '--exp-merge-svg': opts['exp_merge_svg'] = True
        elif a == '--exp-merge-static': opts['exp_merge_static'] = True
        elif a.startswith('--exp-gpu-2d='):
            v = a.split('=', 1)[1].strip().lower()
            if v in ('auto', 'force', 'off'):
                opts['exp_gpu_2d'] = v
        elif a == '--exp-gpu-2d':
            opts['exp_gpu_2d'] = 'force'
        elif a == '--exp-autocairo':
            opts['exp_autocairo'] = True
        elif a == '--exp-auto-split':
            opts['exp_auto_split'] = True
        elif a == '--exp-passPNG' or a == '--exp-pass-png':
            opts['exp_pass_png'] = True
        elif a.startswith('--exp-auto-split='):
            try: opts['exp_auto_split'] = float(a.split('=', 1)[1])
            except ValueError: pass
        elif a.startswith('--max-chunk-weight='):
            try: opts['max_chunk_weight'] = max(50, int(a.split('=', 1)[1]))
            except ValueError: pass
        elif a.startswith('--merge-chunk-sec='):
            try: opts['merge_chunk_sec'] = max(2.0, float(a.split('=', 1)[1]))
            except ValueError: pass
        elif a.startswith('--autocairo-min='):
            try: opts['autocairo_min'] = max(2, int(a.split('=', 1)[1]))
            except ValueError: pass
        elif a == '--exp-bmp':
            opts['exp_bmp'] = True
        elif a == '--exp-qsv': opts['exp_qsv'] = True
        elif a == '--exp-nvenc': opts['exp_nvenc'] = True
        elif a.startswith('--exp-pipeline='):
            v = a.split('=', 1)[1].strip().lower()
            if v in ('auto', '0'):
                opts['exp_pipeline'] = 0
            else:
                try: opts['exp_pipeline'] = max(1, int(v))
                except ValueError: pass
        elif a.startswith('--workers='):
            try: opts['workers'] = int(a.split('=', 1)[1])
            except ValueError: pass
        elif a.startswith('--chunk='):
            try: opts['max_chunk'] = float(a.split('=', 1)[1])
            except ValueError: pass
        elif a.startswith('--render-frame='):
            try: opts['render_frame'] = float(a.split('=', 1)[1])
            except ValueError: print(f"  ⚠ --render-frame 非法", file=sys.stderr)
        elif a.startswith('--render-range='):
            try: opts['render_range'] = _parse_time_range(a.split('=', 1)[1])
            except ValueError as e: print(f"  ⚠ {e}", file=sys.stderr)
        elif a.startswith('--format='):
            f = a.split('=', 1)[1].lower()
            if f in ('svg', 'png', 'both'): opts['format'] = f
        elif a.startswith('--out='): opts['out'] = a.split('=', 1)[1]
        elif a == '--no-silent-audio': opts['silent_audio'] = False
        elif a == '--workdir': skip_next = True

    if opts['merge_aggressive'] is None:
        if opts['exp_merge_svg'] or opts['exp_merge_static']:
            opts['merge_aggressive'] = opts['aggressive'] if opts['aggressive'] is not None else 2
        else: opts['merge_aggressive'] = 0
    if opts['merge_aggressive'] > 0:
        if not (opts['exp_merge_svg'] or opts['exp_merge_static']):
            opts['exp_merge_svg'] = True; opts['exp_merge_static'] = True
    if opts['merge_aggressive'] == 0:
        opts['exp_merge_svg'] = False; opts['exp_merge_static'] = False

    rest = [a for a in argv if not a.startswith('--')]
    xml = rest[0] if rest else 'project.xml'
    base = str(Path(xml).parent)
    workroot = _pick_workdir(argv)

    # exp_pipeline 默认 auto：min(chunks, cpu/2)，最小 1
    auto_pipe = (opts['exp_pipeline'] == 0)

    # 应用 --exp-gpu-2d
    if opts.get('exp_gpu_2d') == 'off':
        os.environ['XMLVE_2D'] = 'cairo'
    elif opts.get('exp_gpu_2d') == 'force':
        os.environ['XMLVE_2D'] = 'force'
    else:
        os.environ.pop('XMLVE_2D', None)

    # 应用 --exp-bmp
    if opts.get('exp_bmp'):
        _set_frame_ext('.bmp')
    else:
        _set_frame_ext('.png')

    # auto_split 提前处理（banner 需要）
    if opts.get('exp_auto_split'):
        v = opts['exp_auto_split']
        if v is True:
            v = opts['max_chunk']
        v = max(3.0, float(v))
        os.environ['XMLVE_AUTO_SPLIT'] = str(v)
        opts['exp_auto_split'] = v
    else:
        os.environ.pop('XMLVE_AUTO_SPLIT', None)
    os.environ['XMLVE_MAX_CHUNK_WEIGHT'] = str(int(opts.get('max_chunk_weight', 400)))
    os.environ['XMLVE_MERGE_CHUNK_SEC'] = str(float(opts.get('merge_chunk_sec', 18.0)))

    print(f"GenericAIPPTEditor v{__version__}")
    _print_banner(opts, opts.get('aggressive'))
    print("⏳ 探测环境...")
    render_frame_mode = opts['render_frame'] is not None
    try: Env.probe(workroot)
    except Exception as e:
        print(f"✗ 环境探测失败: {e}", file=sys.stderr); return 1
    print(Env.report())

    # 硬件编码覆盖：如果用户开了 --exp-qsv / --exp-nvenc
    # 且检测到可用，则覆盖 quality
    if opts.get('exp_qsv') and Env.has_qsv:
        print("  🎬 QSV 可用，自动切换到 h264_qsv")
        opts['quality'] = 'qsv'
        Env.hw_accel_active = 'qsv'
    elif opts.get('exp_nvenc') and Env.has_nvenc:
        print("  🎬 NVENC 可用，自动切换到 h264_nvenc")
        opts['quality'] = 'nvenc'
        Env.hw_accel_active = 'nvenc'
    elif opts.get('exp_qsv') and not Env.has_qsv:
        print("  ⚠ QSV 不可用，回退到 fast", file=sys.stderr)
    elif opts.get('exp_nvenc') and not Env.has_nvenc:
        print("  ⚠ NVENC 不可用或显存 < 512MB，回退到 fast", file=sys.stderr)

    try: proj = parse_xml(xml, base_dir=base)
    except Exception as e:
        print(f"✗ XML 解析失败: {e}"); traceback.print_exc(); return 1

    q = QUALITY_PRESETS[opts['quality']]
    # 保留逻辑坐标系（proj['w'] / proj['h'] 不变）
    proj['_out_w'] = q.get('w') or proj['w']
    proj['_out_h'] = q.get('h') or proj['h']
    if q.get('fps'): proj['fps'] = q['fps']
    proj['_quality'] = opts['quality']
    proj['_exp_encode'] = opts['exp_encode']
    proj['_exp_merge_svg'] = opts['exp_merge_svg']
    proj['_exp_merge_static'] = opts['exp_merge_static']
    proj['_merge_aggressive'] = opts['merge_aggressive']
    proj['_exp_autocairo'] = opts.get('exp_autocairo', False)
    proj['_autocairo_min'] = opts.get('autocairo_min', 30)
    proj['_auto_split_sec'] = opts.get('exp_auto_split') or None
    proj['_exp_pass_png'] = bool(opts.get('exp_pass_png'))
    if opts.get('exp_pass_png'):
        os.environ['XMLVE_PASS_PNG'] = '1'
    else:
        os.environ.pop('XMLVE_PASS_PNG', None)
    # 子进程通过环境变量读取（避免改子进程签名）
    if opts.get('exp_autocairo'):
        os.environ['XMLVE_AUTOCAIRO'] = '1'
        os.environ['XMLVE_AUTOCAIRO_MIN'] = str(opts.get('autocairo_min', 30))
    else:
        os.environ.pop('XMLVE_AUTOCAIRO', None)
        os.environ.pop('XMLVE_AUTOCAIRO_MIN', None)

    missing = preflight(proj)
    if missing:
        print(f"\n⚠ 缺失 {len(missing)} 个素材：")
        for m in missing: print(f"  {m['asset']:8s} [{m['kind']:5s}] {m['src']}")
        proj = drop_missing_clips(proj, {m['asset'] for m in missing})

    try: proj = resolve(proj)
    except Exception as e:
        print(f"✗ 时间解析失败: {e}"); traceback.print_exc(); return 1
    proj['workdir'] = workroot
    proj['_quality'] = opts['quality']
    if not proj.get('_out_w'): proj['_out_w'] = proj['w']
    if not proj.get('_out_h'): proj['_out_h'] = proj['h']

    need_latex = any(a.get('type') == 'latex' for a in proj['assets'].values())
    try: Env.require(latex_needed=need_latex, ffmpeg_needed=not render_frame_mode)
    except SystemExit as e: print(e); return 1

    # ---- 单帧模式 ----
    if render_frame_mode:
        t_abs = opts['render_frame']
        print(f"\n🎬 渲染单帧 t={t_abs}s（{proj['w']}x{proj['h']}）")
        out_base = opts['out'] or f"frame_{t_abs:g}"
        op = Path(out_base)
        if op.suffix.lower() in ('.png', '.svg'): op = op.with_suffix('')
        written = render_single_frame(proj, t_abs, op, fmt=opts['format'])
        return 0 if written else 1

    # ---- 区间模式 ----
    time_range = opts['render_range']
    if time_range:
        print(f"\n🎬 渲染区间 [{time_range[0]:.2f}, {time_range[1]:.2f}]s")
        proj['duration'] = time_range[1] - time_range[0]

    if opts['preview']:
        proj['_out_w'] = proj['_out_w'] // 2
        proj['_out_h'] = proj['_out_h'] // 2
        proj['fps'] = max(10, proj['fps'] // 2)
        print(f"🔎 preview: {proj['_out_w']}x{proj['_out_h']} @ {proj['fps']}fps")

    print('\n时间轴：')
    for tr in proj['tracks']:
        for c in tr['clips']:
            atype = proj['assets'][c['asset']]['type'] if c.get('asset') else 'group'
            print(f"  {c['id']:6s} ({c['_start']:.2f}, {c['_end']:.2f})  "
                  f"kind={tr['kind']}  type={atype}")

    chunks = build_chunks(proj, max_chunk_sec=opts['max_chunk'],
                           time_range=time_range,
                           auto_split_sec=proj.get('_auto_split_sec'),
                           max_weight=opts.get('max_chunk_weight', 400),
                           merge_sec=opts.get('merge_chunk_sec', 18.0))
    print(f'\n分块 ({len(chunks)} 块)：')
    for ch in chunks:
        print(f"  chunk {ch['idx']}: [{ch['start']:.2f}, {ch['end']:.2f}] "
              f"({ch['end'] - ch['start']:.2f}s)  clips={len(ch['video_ids'])}")

    if not chunks:
        print("✗ 时间区间内没有可渲染的 chunk", file=sys.stderr); return 1

    chunk_dir = workroot / 'chunks'; chunk_dir.mkdir(exist_ok=True)
    t0 = time.time()
    chunk_paths, failed = [], []
    # 🆕 auto pipeline
    if auto_pipe:
        cpu = os.cpu_count() or 1
        n_chunks = len(chunks) if 'chunks' in locals() else 0
        # 实测：2~3 进程最优。多了反而因为 GPU/IO 争抢而变慢。
        # 上限：min(3, cpu//2, chunk 数)
        target = max(1, min(3, cpu // 2))
        if n_chunks > 0:
            target = min(target, n_chunks)
        opts['exp_pipeline'] = target
        print(f"  · 流水线并行: auto → {target} 进程 "
              f"(CPU {cpu} 核, {n_chunks} chunk)")
    use_pipeline = (opts['exp_pipeline'] > 1 and len(chunks) >= 2)

    pending_chunks = []
    for ch in chunks:
        out_path = chunk_dir / f"chunk_{ch['idx']:03d}.mp4"
        if opts['resume'] and is_valid_chunk(out_path, need_audio=False):
            print(f"\n▶ Chunk {ch['idx']}: 复用已有")
            print(f"[TIME] chunk_{ch['idx']:03d} REUSE 0.00s")
            chunk_paths.append(out_path)
        elif opts['resume'] and out_path.exists():
            print(f"\n▶ Chunk {ch['idx']}: 已有 chunk 无效，重新渲染")
            pending_chunks.append(ch)
        else: pending_chunks.append(ch)

    workers_per_proc = opts['workers']
    if opts['exp_pipeline'] > 1 and opts['workers'] > 1:
        workers_per_proc = max(1, opts['workers'] // opts['exp_pipeline'])

    if use_pipeline and pending_chunks:
        n_procs = min(opts['exp_pipeline'], len(pending_chunks))
        print(f"\n🚀 流水线并行：{n_procs} 进程 "
              f"(每进程 {workers_per_proc} 线程，silent_audio={opts['silent_audio']})")
        tasks = [(
            str(Path(xml).resolve()), base, ch['idx'],
            workers_per_proc, str(chunk_dir), str(workroot),
            opts['max_chunk'], opts['quality'],
            opts['exp_encode'], opts['exp_merge_svg'],
            opts['exp_merge_static'], opts['merge_aggressive'],
            opts['silent_audio'],
        ) for ch in pending_chunks]

        results = {}
        try:
            with ProcessPoolExecutor(max_workers=n_procs) as ex:
                futures = {ex.submit(_render_chunk_subprocess, t): t[2] for t in tasks}
                for fut in as_completed(futures):
                    ci = futures[fut]
                    try: p, ok = fut.result(); results[ci] = (p, ok)
                    except Exception as e:
                        print(f"  ✗ Chunk {ci} 子进程异常: {e}", file=sys.stderr)
                        results[ci] = (None, False)
            for ch in pending_chunks:
                ci = ch['idx']
                p, ok = results.get(ci, (None, False))
                if ok and p: chunk_paths.append(p)
                else: failed.append(ci)
        except Exception as e:
            print(f"  ⚠ 流水线失败: {e}，回退串行", file=sys.stderr)
            existing = {int(re.search(r'chunk_(\d+)', p.name).group(1))
                        for p in chunk_paths if re.search(r'chunk_(\d+)', p.name)}
            for ch in pending_chunks:
                if ch['idx'] in existing: continue
                try:
                    p, ok = render_chunk(proj, ch, opts['workers'], chunk_dir,
                                          silent_audio=opts['silent_audio'])
                except Exception as e2:
                    print(f"  ✗ Chunk {ch['idx']} 异常: {e2}", file=sys.stderr)
                    ok, p = False, chunk_dir / f"chunk_{ch['idx']:03d}.mp4"
                if ok: chunk_paths.append(p)
                else: failed.append(ch['idx'])
    else:
        for ch in pending_chunks:
            try:
                if ch.get('type') == 'space3d':
                    # 主进程里也走两阶段
                    W = proj.get('_out_w') or proj['w']
                    H = proj.get('_out_h') or proj['h']
                    fps = proj['fps']
                    space_mp4 = chunk_dir / f"space3d_{ch['idx']:03d}.mp4"
                    _render_space3d_to_mp4(proj, ch['space_id'],
                                            ch['end'] - ch['start'],
                                            space_mp4, W, H, fps,
                                            t_offset=ch['start'])
                    inline_clips = _build_inline_overlays_for_chunk(proj, ch)
                    p, ok = render_chunk(proj, ch, opts['workers'], chunk_dir,
                                          silent_audio=opts['silent_audio'],
                                          video_bg_override=space_mp4,
                                          extra_overlay_clips=inline_clips)
                else:
                    p, ok = render_chunk(proj, ch, opts['workers'], chunk_dir,
                                          silent_audio=opts['silent_audio'])
            except Exception as e:
                print(f"  ✗ Chunk {ch['idx']} 异常: {e}", file=sys.stderr)
                traceback.print_exc()
                ok, p = False, chunk_dir / f"chunk_{ch['idx']:03d}.mp4"
            if ok: chunk_paths.append(p)
            else: failed.append(ch['idx'])

    def _chunk_idx(p):
        m = re.search(r'chunk_(\d+)', p.name)
        return int(m.group(1)) if m else 0
    chunk_paths.sort(key=_chunk_idx)

    print(f"[TIME] render_total {time.time()-t0:.2f}s")

    if failed:
        print(f"\n✗ {len(failed)} 个 chunk 失败: {failed}")
        print(f"  已成功的 chunk 保留在: {chunk_dir}")
        print("  修好后用 --resume 断点续传")
        return 2

    # ---- 合并：视频部分 ----
    silent_video = chunk_dir / '_silent_video.mp4'
    print(f"\n⏳ 合并 {len(chunk_paths)} 个 chunk 的【视频】...")
    try:
        if not concat_chunks(chunk_paths, silent_video, proj=proj,
                             aggressive=opts.get('aggressive') or 0):
            print("✗ 视频合并失败", file=sys.stderr); return 3
    except Exception as e:
        print(f"✗ 视频合并异常: {e}", file=sys.stderr)
        traceback.print_exc(); return 3

    # ---- 合并：音频一次性合成 ----
    out_final = Path(opts['out'] or 'out.mp4').resolve()
    print(f"\n🎵 全局音频合成 → {out_final.name}")
    try:
        t_audio = time.time()
        if not mix_global_audio(proj, silent_video, out_final, time_range=time_range):
            print("✗ 音频合成失败，退化为静音输出", file=sys.stderr)
            if str(silent_video) != str(out_final):
                try: shutil.copy2(str(silent_video), str(out_final))
                except Exception: pass
        else:
            print(f"  ✓ 音频合成完成 ({time.time()-t_audio:.1f}s)")
    except Exception as e:
        print(f"✗ 音频合成异常: {e}", file=sys.stderr)
        traceback.print_exc()

    size_mb = out_final.stat().st_size / 1024 / 1024 if out_final.exists() else 0
    if _HAS_GPU_2D:
        gpu_n = _GPU_2D_STATS['gpu']
        cairo_n = _GPU_2D_STATS['cairo']
        total = gpu_n + cairo_n
        if total > 0:
            print(f'   2D 光栅化: GPU {gpu_n} / cairo {cairo_n}  '
                  f'({gpu_n/total*100:.0f}% GPU)')
    if _GPU_TIMES['n'] > 0:
        n = _GPU_TIMES['n']
        print(f'   GPU 2D 时间分解 ({n} 帧):')
        print(f'     decode: {_GPU_TIMES["decode"]/n*1000:6.2f}ms/帧')
        print(f'     render: {_GPU_TIMES["render"]/n*1000:6.2f}ms/帧')
        print(f'     encode: {_GPU_TIMES["encode"]/n*1000:6.2f}ms/帧')
        print(f'     write:  {_GPU_TIMES["write"]/n*1000:6.2f}ms/帧')
        print(f'     合计:   {sum(_GPU_TIMES[k] for k in ("decode","render","encode","write"))/n*1000:6.2f}ms/帧')
    print(f'\n✅ {out_final.name}  ({time.time() - t0:.1f}s, '
          f'{len(chunks)} chunks, {size_mb:.1f} MB)')
    print(f"[TIME] wall_total {time.time()-t0:.2f}s")

    if opts['purge']:
        print(f"🧹 purge {workroot} ...")
        t_clean = time.time()
        safe_rmtree(workroot, label='purge')
        print(f"  ✓ {time.time()-t_clean:.2f}s")
    elif opts['keep']:
        print(f"📦 keep 中间文件在 {workroot}")
    else:
        frame_dirs = list(workroot.glob('frames_*'))
        if frame_dirs:
            print(f"🧹 清理 {len(frame_dirs)} 个中间帧目录 ...")
            t_clean = time.time()
            for d in frame_dirs: safe_rmtree(d, label=d.name)
            print(f"  ✓ {time.time()-t_clean:.2f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)