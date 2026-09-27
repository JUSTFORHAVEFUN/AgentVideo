#!/usr/bin/env python3
"""raster2d.py — GPU 2D 光栅化器（EGL/GLES3 + block-atlas 字体）"""
from __future__ import annotations
import ctypes, hashlib, math, os, re, sys, subprocess
from io import BytesIO
from pathlib import Path
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

c_void_p   = ctypes.c_void_p
c_int      = ctypes.c_int
c_uint     = ctypes.c_uint
c_float    = ctypes.c_float
c_char_p   = ctypes.c_char_p
c_size_t   = ctypes.c_size_t
c_ubyte    = ctypes.c_ubyte

def _detect_egl_libs():
    """自动探测 libEGL / libGLESv2 的实际路径。

    优先级：
      1. 环境变量 XMLVE_EGL_LIB / XMLVE_GLES_LIB 显式指定
      2. 常见发行版硬编码路径（Debian/Ubuntu/Arch/Fedora/RPi）
      3. ctypes.util.find_library 系统搜索
    """
    import os as _os
    _env_egl = _os.environ.get('XMLVE_EGL_LIB', '').strip()
    _env_gles = _os.environ.get('XMLVE_GLES_LIB', '').strip()
    if _env_egl and _env_gles:
        return (_env_egl, _env_gles, 'env')

    import ctypes.util as _cu
    # 按顺序探测候选对
    _candidates = [
        # ── Android ──
        ('/system/lib64/libEGL.so',    '/system/lib64/libGLESv2.so'),
        ('/vendor/lib64/libEGL.so',    '/vendor/lib64/libGLESv2.so'),
        ('/system/lib/libEGL.so',      '/system/lib/libGLESv2.so'),
        # ── Debian / Ubuntu (x86_64, aarch64, armhf) ──
        ('/usr/lib/x86_64-linux-gnu/libEGL.so.1',
         '/usr/lib/x86_64-linux-gnu/libGLESv2.so.2'),
        ('/usr/lib/aarch64-linux-gnu/libEGL.so.1',
         '/usr/lib/aarch64-linux-gnu/libGLESv2.so.2'),
        ('/usr/lib/arm-linux-gnueabihf/libEGL.so.1',
         '/usr/lib/arm-linux-gnueabihf/libGLESv2.so.2'),
        ('/usr/lib/arm-linux-gnueabi/libEGL.so.1',
         '/usr/lib/arm-linux-gnueabi/libGLESv2.so.2'),
        # ── Debian mesa-egl 子目录（老版本） ──
        ('/usr/lib/x86_64-linux-gnu/mesa-egl/libEGL.so.1',
         '/usr/lib/x86_64-linux-gnu/mesa-egl/libGLESv2.so.2'),
        # ── Arch / Fedora / CentOS (lib64) ──
        ('/usr/lib64/libEGL.so.1',    '/usr/lib64/libGLESv2.so.2'),
        ('/usr/lib/libEGL.so.1',      '/usr/lib/libGLESv2.so.2'),
        # ── NVIDIA 驱动（Fedora/openSUSE 路径） ──
        ('/usr/lib64/libEGL_nvidia.so.0',
         '/usr/lib64/libGLESv2_nvidia.so.2'),
        # ── Raspberry Pi (VideoCore) ──
        ('/opt/vc/lib/libEGL.so',     '/opt/vc/lib/libGLESv2.so'),
        ('/opt/vc/lib/libbrcmEGL.so', '/opt/vc/lib/libbrcmGLESv2.so'),
        # ── 通用 .so 名（ldconfig 路径） ──
        ('libEGL.so.1', 'libGLESv2.so.2'),
        ('libEGL.so',   'libGLESv2.so'),
        # ── Mesa 软件渲染 / LLVMpipe ──
        ('/usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0',
         '/usr/lib/x86_64-linux-gnu/libGLESv2.so.2'),
    ]

    for _ep, _gp in _candidates:
        try:
            _ = ctypes.CDLL(_ep)
            _ = ctypes.CDLL(_gp)
            return (_ep, _gp, 'probe')
        except OSError:
            continue

    # 最后尝试 ctypes.util.find_library
    try:
        _ep = _cu.find_library('EGL')
        _gp = _cu.find_library('GLESv2')
        if _ep and _gp:
            return (_ep, _gp, 'find_library')
    except Exception:
        pass

    return (None, None, 'fail')



# ============================================================
# EGL / GLES 常量
# ============================================================
EGL_DEFAULT_DISPLAY       = 0
EGL_OPENGL_ES_API         = 0x30A0
EGL_SURFACE_TYPE          = 0x3033
EGL_PBUFFER_BIT           = 0x0001
EGL_RENDERABLE_TYPE       = 0x3040
EGL_OPENGL_ES2_BIT        = 0x0004
EGL_OPENGL_ES3_BIT_KHR    = 0x0040
EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_BLUE_SIZE, EGL_ALPHA_SIZE = 0x3024, 0x3023, 0x3022, 0x3021
EGL_NONE                  = 0x3038
EGL_CONTEXT_CLIENT_VERSION = 0x3098
EGL_WIDTH, EGL_HEIGHT     = 0x3057, 0x3056

GL_VENDOR, GL_RENDERER, GL_VERSION = 0x1F00, 0x1F01, 0x1F02
GL_VERTEX_SHADER, GL_FRAGMENT_SHADER = 0x8B31, 0x8B30
GL_COMPILE_STATUS, GL_LINK_STATUS = 0x8B81, 0x8B82
GL_ARRAY_BUFFER           = 0x8892
GL_DYNAMIC_DRAW           = 0x88E8
GL_FLOAT                  = 0x1406
GL_TRIANGLES              = 0x0004
GL_COLOR_BUFFER_BIT       = 0x4000
GL_RGBA, GL_RGBA8         = 0x1908, 0x8058
GL_UNSIGNED_BYTE          = 0x1401
GL_FRAMEBUFFER            = 0x8D40
GL_COLOR_ATTACHMENT0      = 0x8CE0
GL_TEXTURE_2D             = 0x0DE1
GL_TEXTURE0               = 0x84C0
GL_LINEAR                 = 0x2601
GL_CLAMP_TO_EDGE          = 0x812F
GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER = 0x2801, 0x2800
GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T = 0x2802, 0x2803
GL_PACK_ALIGNMENT, GL_UNPACK_ALIGNMENT = 0x0D05, 0x0CF5
GL_BLEND                  = 0x0BE2
GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA = 0x0302, 0x0303
GL_MAX_TEXTURE_SIZE       = 0x0D33


# ============================================================
# 2D 仿射变换（2x3: [a c e; b d f]）
# ============================================================
class Mat2D:
    __slots__ = ('a', 'b', 'c', 'd', 'e', 'f')

    def __init__(self, a=1.0, b=0.0, c=0.0, d=1.0, e=0.0, f=0.0):
        self.a, self.b, self.c = float(a), float(b), float(c)
        self.d, self.e, self.f = float(d), float(e), float(f)

    @classmethod
    def identity(cls):    return cls()
    @classmethod
    def translate(cls, tx, ty):    return cls(1, 0, 0, 1, tx, ty)
    @classmethod
    def scale(cls, sx, sy=None):   return cls(sx, 0, 0, (sx if sy is None else sy), 0, 0)

    @classmethod
    def rotate(cls, angle_rad, cx=0.0, cy=0.0):
        ca, sa = math.cos(angle_rad), math.sin(angle_rad)
        m = cls(ca, sa, -sa, ca, 0, 0)
        if cx or cy:
            return cls.translate(cx, cy) @ m @ cls.translate(-cx, -cy)
        return m

    @classmethod
    def skew_x(cls, angle_rad):    return cls(1, 0, math.tan(angle_rad), 1, 0, 0)
    @classmethod
    def skew_y(cls, angle_rad):    return cls(1, math.tan(angle_rad), 0, 1, 0, 0)

    def __matmul__(self, o: "Mat2D") -> "Mat2D":
        """self @ o：先 o 后 self"""
        return Mat2D(
            self.a * o.a + self.c * o.b,
            self.b * o.a + self.d * o.b,
            self.a * o.c + self.c * o.d,
            self.b * o.c + self.d * o.d,
            self.a * o.e + self.c * o.f + self.e,
            self.b * o.e + self.d * o.f + self.f,
        )

    def apply(self, x, y):
        return (self.a * x + self.c * y + self.e,
                self.b * x + self.d * y + self.f)

    _PAT = re.compile(r'(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)')

    @classmethod
    def parse(cls, s: str) -> "Mat2D":
        m = cls()
        if not s:
            return m
        for mm in cls._PAT.finditer(s):
            op, raw = mm.group(1), mm.group(2)
            try:
                args = [float(x) for x in re.split(r'[\s,]+', raw.strip()) if x]
            except ValueError:
                continue
            if not args:
                continue
            if op == 'matrix' and len(args) >= 6:
                m = m @ cls(*args[:6])
            elif op == 'translate':
                m = m @ cls.translate(args[0], args[1] if len(args) > 1 else 0)
            elif op == 'scale':
                m = m @ cls.scale(args[0], args[1] if len(args) > 1 else args[0])
            elif op == 'rotate':
                ang = math.radians(args[0])
                cx = args[1] if len(args) > 1 else 0.0
                cy = args[2] if len(args) > 2 else 0.0
                m = m @ cls.rotate(ang, cx, cy)
            elif op == 'skewX':
                m = m @ cls.skew_x(math.radians(args[0]))
            elif op == 'skewY':
                m = m @ cls.skew_y(math.radians(args[0]))
        return m


# ============================================================
# 字体解析
# ============================================================
_GENERIC_SANS  = {'sans-serif', 'sans', 'system-ui', ''}
_GENERIC_SERIF = {'serif'}
_GENERIC_MONO  = {'monospace', 'mono', 'fixed'}

# fontconfig 权重表（0..215）
_FC_WEIGHT = {
    100: 0,    # thin
    200: 40,   # extralight
    300: 50,   # light
    400: 80,   # regular
    500: 100,  # medium
    600: 180,  # semibold
    700: 200,  # bold
    800: 205,  # extrabold
    900: 210,  # black
}

_WEIGHT_NUM = {
    'thin': 100, 'hairline': 100,
    'extralight': 200, 'ultralight': 200,
    'light': 300,
    'normal': 400, 'regular': 400, 'book': 400,
    'medium': 500,
    'semibold': 600, 'demibold': 600,
    'bold': 700,
    'extrabold': 800, 'ultrabold': 800,
    'black': 900, 'heavy': 900,
}


def _parse_weight(w) -> int:
    if w is None:
        return 400
    if isinstance(w, (int, float)):
        return int(w)
    s = str(w).strip().lower()
    if s in _WEIGHT_NUM:
        return _WEIGHT_NUM[s]
    try:
        return int(s)
    except ValueError:
        return 400


def _weight_code(w) -> str:
    n = _parse_weight(w)
    n = max(100, min(900, n))
    return str(_FC_WEIGHT.get(n, 80))


# ------------------------------------------------------------
# 通用 face 探测：解决 fc-match 返回的 face 实际 weight 与请求不符
# ------------------------------------------------------------
# fontconfig weight 数值 → 常见 style 命名后缀（通用命名规则，非设备特定）
_WEIGHT_SUFFIXES = {
    0:   ['Thin', 'Hairline'],
    40:  ['ExtraLight', 'UltraLight'],
    50:  ['Light', 'ExtraLight'],
    80:  ['Regular', 'Book', 'Normal'],
    100: ['Medium', 'Regular'],
    180: ['SemiBold', 'DemiBold', 'Medium'],
    200: ['Bold', 'SemiBold'],
    205: ['ExtraBold', 'UltraBold', 'Bold'],
    210: ['Black', 'Heavy', 'ExtraBold'],
}

# style 名 → fontconfig weight（顺序敏感：先长后短）
_STYLE_PATTERNS = [
    ('extralight', 40), ('extra light', 40), ('ultralight', 40),
    ('ultra light', 40), ('semilight', 60), ('semi light', 60),
    ('demibold', 180), ('demi bold', 180),
    ('semibold', 180), ('semi bold', 180),
    ('extrabold', 205), ('extra bold', 205),
    ('ultrabold', 205), ('ultra bold', 205),
    ('thin', 0), ('hairline', 0),
    ('black', 210), ('heavy', 210),
    ('light', 50),
    ('bold', 200),
    ('medium', 100),
    ('book', 80), ('regular', 80), ('normal', 80),
]

_FACE_LIST_CACHE: dict = {}    # path -> [(index, family, style), ...]
_FACE_W_CACHE: dict = {}       # (path, index) -> weight or None
_SYNTHETIC_BOLD: dict = {}     # (name, want_nominal) -> bool


def _style_to_weight(style):
    if not style:
        return 80
    s = str(style).lower()
    for pat, w in _STYLE_PATTERNS:
        if pat in s:
            return w
    return 80


def _list_font_faces(path):
    """探测字体文件里所有 face（TTC 可能含多个）。"""
    if path in _FACE_LIST_CACHE:
        return _FACE_LIST_CACHE[path]
    from PIL import ImageFont
    faces = []
    for i in range(128):
        try:
            f = ImageFont.truetype(path, 12, index=i)
            fam, style = f.getname()
            faces.append((i, fam or '', style or ''))
        except Exception:
            break
    if not faces:
        faces = [(0, '', '')]
    _FACE_LIST_CACHE[path] = faces
    return faces


def _face_actual_weight(path, index):
    """读 face 的真实 style → weight 数值。无法读取返回 None。"""
    key = (path, index)
    if key in _FACE_W_CACHE:
        return _FACE_W_CACHE[key]
    w = None
    for idx, _fam, style in _list_font_faces(path):
        if idx == index:
            w = _style_to_weight(style)
            break
    _FACE_W_CACHE[key] = w
    return w


def _find_best_face_in_file(path, want_w):
    """在同一文件里找 weight 最接近的 face。返回 (index, weight) 或 None。"""
    best = None
    for idx, _fam, style in _list_font_faces(path):
        w = _style_to_weight(style)
        d = abs(w - want_w)
        if best is None or d < best[0]:
            best = (d, idx, w)
    if best is None:
        return None
    return (best[1], best[2])


def _fc_match_raw(name, fc_weight):
    """fc_weight 是 fontconfig 数值字符串（'0'..'210'），绕过 _weight_code 转换。"""
    q = f'{name}:weight={fc_weight}'
    try:
        import subprocess as _sp
        r = _sp.run(['fc-match', '-f', '%{file}|%{index}|%{weight}', q],
                    capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split('|')
            p = parts[0] if parts else ''
            i = int(parts[1]) if len(parts) > 1 and parts[1].strip() else 0
            w = int(parts[2]) if len(parts) > 2 and parts[2].strip() else 80
            if p:
                return p, i, w
    except Exception:
        pass
    return None, 0, 80


def _try_weight_sibling(name, want_w):
    """尝试 '{name} Bold' / '{name} Light' 这类独立命名的 weight 家族。

    关键：验证返回的 family 名字是否真的对应请求的 family。
    fc-match 找不到时不会返回空，而是 fallback 到系统默认字体 ——
    那会让 'Noto Sans CJK SC Bold' 错误命中 DejaVuSans-Bold（无中文）。
    """
    suf_list = _WEIGHT_SUFFIXES.get(want_w)
    if not suf_list:
        k = min(_WEIGHT_SUFFIXES.keys(), key=lambda x: abs(x - want_w))
        suf_list = _WEIGHT_SUFFIXES[k]

    # 归一化请求的 family 名，取首个 token 做前缀匹配
    name_norm = re.sub(r'[\s_\-]+', ' ', str(name).strip().lower())
    first_token = name_norm.split(' ')[0] if name_norm else ''

    for suf in suf_list:
        if not suf:
            continue
        p, i, _ = _fc_match_raw(f'{name} {suf}', str(want_w))
        if not p:
            continue

        # 验证 face family 名：必须与请求的 family 名字有 token 重叠
        fam_name = ''
        for idx, fam, _st in _list_font_faces(p):
            if idx == i:
                fam_name = (fam or '').lower()
                break
        if not fam_name:
            continue
        fam_norm = re.sub(r'[\s_\-]+', ' ', fam_name)
        if first_token and first_token not in fam_norm:
            # family 完全不相关 → fc-match 的 fallback，丢弃
            continue

        aw = _face_actual_weight(p, i)
        if aw is not None and abs(aw - want_w) <= 20:
            return (p, i, aw)
    return None


def _weights_close(a, b, tol=20):
    if a is None or b is None:
        return False
    return abs(int(a) - int(b)) <= tol



# ============================================================
# 文本位图缓存：draw_text 用整串渲染成 bitmap 贴纹理，
# 不再逐字用字形 atlas 拼 quad —— 避免 PIL 逐字 advance 累加
# 与整串渲染推进之间的差异。
# ============================================================
_TEXT_BITMAP_CACHE: dict = {}


def _render_text_bitmap(text, font_path, font_index, size, synthetic_bold):
    """整串文本 → RGBA bitmap + 布局信息。

    返回 (rgba, off_x, off_y, adv, asc, desc)
      rgba: numpy (H, W, 4) uint8
      off_x, off_y: buffer 左上角相对 pen 原点（'la' 锚点）的偏移
      adv:  总 advance 宽度
      asc, desc: 字体 ascent/descent（相对基线）
    """
    key = (text, int(round(size)), font_path, int(font_index),
           bool(synthetic_bold))
    if key in _TEXT_BITMAP_CACHE:
        return _TEXT_BITMAP_CACHE[key]

    font = ImageFont.truetype(font_path, int(round(size)), index=font_index)
    asc, desc = font.getmetrics()

    # 合成加粗 stroke：从 size/22 降到 size/60（120px→2px, 32px→1px）
    # 这样视觉上和 cairo Regular 的粗细差异会小很多
    # 合成加粗宽度由环境变量控制：
    #   XMLVE_SYNTHETIC_BOLD=0      → 关闭（与 cairo Regular 视觉一致）★新默认
    #   XMLVE_SYNTHETIC_BOLD=1      → 弱加粗（size/90，仅大字号触发）
    #   XMLVE_SYNTHETIC_BOLD=2      → 中加粗（size/50）
    #   XMLVE_SYNTHETIC_BOLD=strong → 强加粗（size/30）
    stroke = 0
    if synthetic_bold:
        mode = os.environ.get('XMLVE_SYNTHETIC_BOLD', '0').strip().lower()
        if mode in ('0', 'false', 'off', 'no', ''):
            stroke = 0
        elif mode == 'strong':
            stroke = max(1, int(round(size / 30.0)))
        elif mode in ('2', 'medium'):
            stroke = max(1, int(round(size / 50.0)))
        else:  # '1', 'on', 'yes', 'true' 或任何其他值
            stroke = int(round(size / 90.0))   # 允许为 0，小字号不加粗

    # 相对 'la' 锚点的包围盒，包含 stroke 扩展
    bbox = font.getbbox(text)
    pad = 4 + stroke
    w = max(1, bbox[2] - bbox[0] + pad * 2)
    h = max(1, bbox[3] - bbox[1] + pad * 2)

    img = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(img)

    pen_x_buf = -bbox[0] + pad
    pen_y_buf = -bbox[1] + pad

    try:
        if stroke:
            d.text((pen_x_buf, pen_y_buf), text, font=font, fill=255,
                   stroke_width=stroke, stroke_fill=255)
        else:
            d.text((pen_x_buf, pen_y_buf), text, font=font, fill=255)
    except TypeError:
        # 老 Pillow 不支持 stroke_width
        d.text((pen_x_buf, pen_y_buf), text, font=font, fill=255)

    arr = np.array(img, dtype=np.uint8)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = arr
    rgba[:, :, 1] = arr
    rgba[:, :, 2] = arr
    rgba[:, :, 3] = arr

    adv = font.getlength(text)
    off_x = bbox[0] - pad
    off_y = bbox[1] - pad

    result = (rgba, off_x, off_y, adv, asc, desc)
    _TEXT_BITMAP_CACHE[key] = result
    return result


# 各族通用候选（按优先级）
_CJK_SANS_PRIORITY = [
    'Noto Sans CJK SC', 'Source Han Sans CN', 'Noto Sans SC',
    'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
    'Droid Sans Fallback', 'Noto Sans',
]
_CJK_SERIF_PRIORITY = [
    'Noto Serif CJK SC', 'Source Han Serif CN', 'Noto Serif SC',
    'AR PL UMing CN', 'AR PL UKai CN', 'Noto Serif',
]
_MONO_PRIORITY = [
    'Noto Sans Mono CJK SC', 'Source Han Mono SC',
    'DejaVu Sans Mono', 'Liberation Mono', 'Noto Sans Mono',
]

_FONT_PATH_CACHE: dict = {}
_FONT_LOG: set = set()
_CJK_FAMILIES = None


def _get_cjk_families():
    global _CJK_FAMILIES
    if _CJK_FAMILIES is not None:
        return _CJK_FAMILIES
    families = []
    try:
        r = subprocess.run(['fc-list', ':lang=zh', 'family'],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                for n in line.split(','):
                    n = n.strip()
                    if n and n not in families:
                        families.append(n)
    except Exception:
        pass
    _CJK_FAMILIES = families
    return families


def _fc_match_full(name, weight='normal'):
    """返回 (path, ttc_index, real_weight)。"""
    wc = _weight_code(weight)
    q = f'{name}:weight={wc}'
    try:
        r = subprocess.run(
            ['fc-match', '-f', '%{file}|%{index}|%{weight}', q],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split('|')
            path = parts[0] if parts else ''
            idx = int(parts[1]) if len(parts) > 1 and parts[1].strip() else 0
            real_w = int(parts[2]) if len(parts) > 2 and parts[2].strip() else 80
            if path:
                return path, idx, real_w
    except Exception:
        pass
    return None, 0, 80


def resolve_font_path(name, weight='normal'):
    """返回 (path, ttc_index, real_weight)。path=None 表示失败。

    不信任 fc-match 报告的 weight —— 主动读 face 的真实 style。
    不符时依次尝试：同文件内换 face → 同家族命名变体 → 合成加粗。
    """
    want = _parse_weight(weight)
    want_w = int(_FC_WEIGHT.get(max(100, min(900, want)), 80))
    key = (name, want)
    if key in _FONT_PATH_CACHE:
        return _FONT_PATH_CACHE[key]

    if isinstance(name, str) and Path(name).exists():
        r = (name, 0, 80)
        _FONT_PATH_CACHE[key] = r
        return r

    lname = (name or '').lower().strip()
    is_sans  = lname in _GENERIC_SANS
    is_serif = lname in _GENERIC_SERIF
    is_mono  = lname in _GENERIC_MONO
    is_generic = is_sans or is_serif or is_mono

    def _log(msg):
        if key not in _FONT_LOG:
            _FONT_LOG.add(key)
            print(f"[font] {name} w={want} -> {msg}", file=sys.stderr)

    def _finalize(p, i, fc_w, tag=''):
        actual = _face_actual_weight(p, i)

        if actual is None or _weights_close(actual, want_w):
            _SYNTHETIC_BOLD[key] = False
            r = (p, i, actual if actual is not None else fc_w)
            _FONT_PATH_CACHE[key] = r
            _log(f"{p} idx={i} w={r[2]}{tag}")
            return r

        # face 实际 weight 不符 → 同文件内找
        best = _find_best_face_in_file(p, want_w)
        if best is not None and best[0] != i and _weights_close(best[1], want_w):
            _SYNTHETIC_BOLD[key] = False
            r = (p, best[0], best[1])
            _FONT_PATH_CACHE[key] = r
            _log(f"{p} idx={r[1]} w={r[2]}  [in-file face]{tag}")
            return r

        # 同家族命名变体（xxx Bold / xxx Light ...）
        if not is_generic:
            alt = _try_weight_sibling(name, want_w)
            if alt is not None:
                _SYNTHETIC_BOLD[key] = False
                _FONT_PATH_CACHE[key] = alt
                _log(f"{alt[0]} idx={alt[1]} w={alt[2]}  [sibling]{tag}")
                return alt

        # 都失败 → 用当前 face + 合成加粗
        _SYNTHETIC_BOLD[key] = True
        r = (p, i, actual if actual is not None else fc_w)
        _FONT_PATH_CACHE[key] = r
        _log(f"{p} idx={i} w={r[2]}  [SYNTHETIC]{tag}")
        return r

    # 具体名
    if not is_generic:
        p, i, w = _fc_match_full(name, want)
        if p:
            return _finalize(p, i, w)

    # generic
    if is_mono:
        candidates = _MONO_PRIORITY
    elif is_serif:
        candidates = _CJK_SERIF_PRIORITY
    else:
        candidates = _CJK_SANS_PRIORITY

    for fam in candidates:
        p, i, w = _fc_match_full(fam, want)
        if p:
            return _finalize(p, i, w, tag=f'  [family={fam}]')

    families = _get_cjk_families()
    prio = _CJK_SERIF_PRIORITY if is_serif else _CJK_SANS_PRIORITY
    ordered = [f for f in prio if f in families]
    ordered += [f for f in families if f not in ordered]
    for fam in ordered:
        p, i, w = _fc_match_full(fam, want)
        if p:
            return _finalize(p, i, w, tag=f'  [family={fam}]')

    for c in (
        '/system/fonts/DroidSansFallbackZW.ttf',
        '/system/fonts/DroidSansFallback.ttf',
        '/system/fonts/NotoSansCJK-Regular.ttc',
        '/system/fonts/NotoSansCJKsc-Regular.otf',
        '/system/fonts/VivoFont.ttf',
        '/system/fonts/HYQiHei-50.ttf',
        '/system/fonts/vivotype-Regular.ttf',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    ):
        if Path(c).exists():
            r = (c, 0, 80)
            _FONT_PATH_CACHE[key] = r
            print(f"[font] {name} w={want} -> {c} (hard)", file=sys.stderr)
            return r
    return (None, 0, 80)


# ============================================================
# 字形图集：网格布局，每个 glyph 占 block×block
# ============================================================
class GlyphAtlas:
    _CACHE: dict = {}

    @classmethod
    def get(cls, font_name, size, weight='normal'):
        path, index, _ = resolve_font_path(font_name, weight)
        if not path:
            raise RuntimeError(f"找不到字体: {font_name}")
        want = _parse_weight(weight)
        synth = _SYNTHETIC_BOLD.get((font_name, want), False)
        key = (path, int(size), int(index), synth)
        if key not in cls._CACHE:
            atlas = cls(path, int(size), index, synthetic_bold=synth)
            cls._CACHE[key] = atlas
        return cls._CACHE[key]

    def __init__(self, font_path, size, index=0, synthetic_bold=False):
        self.font_path = font_path
        self.size = int(size)
        self.font_index = int(index)
        # 必须在渲染 default ASCII 之前设置好，否则 ASCII 会用错 stroke
        self.synthetic_bold = bool(synthetic_bold)
        try:
            self.font = ImageFont.truetype(font_path, self.size, index=self.font_index)
        except Exception:
            self.font = ImageFont.truetype(font_path, self.size)

        self.glyphs: dict = {}
        self.order: list = []
        self.atlas_img = None
        self.atlas_w = 0
        self.atlas_h = 0
        self.ascent, self.descent = self.font.getmetrics()
        self.cap_height = None

        # block = 字号 * 2.2，容纳绝大部分 latin/CJK
        self.block = int(self.size * 2.2)
        self.pen_x = self.block // 2
        self.pen_y = self.block // 2

        default = ''.join(chr(i) for i in range(32, 127))
        self.add_chars(default)

    def _render_char_block(self, ch):
        bs = self.block
        tmp = Image.new('L', (bs, bs), 0)
        d = ImageDraw.Draw(tmp)
        stroke = 0
        if self.synthetic_bold:
            # 合成加粗：描边宽度随字号缩放（24px→1, 96px→4）
            stroke = max(1, int(round(self.size / 22.0)))
        try:
            if stroke:
                d.text((self.pen_x, self.pen_y), ch, font=self.font, fill=255,
                       anchor='ls', stroke_width=stroke, stroke_fill=255)
            else:
                d.text((self.pen_x, self.pen_y), ch, font=self.font, fill=255,
                       anchor='ls')
        except TypeError:
            try:
                bbox = self.font.getbbox(ch)
                d.text((self.pen_x - bbox[0], self.pen_y - bbox[3]),
                       ch, font=self.font, fill=255)
            except Exception:
                return None
        except Exception:
            return None
        arr = np.array(tmp)
        return None if arr.max() == 0 else arr

    def _rebuild_atlas(self):
        n = len(self.order)
        if n == 0:
            return
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = (n + cols - 1) // cols
        bs = self.block
        W, H = cols * bs, rows * bs
        img = Image.new('L', (W, H), 0)
        new_glyphs = {}

        for idx, ch in enumerate(self.order):
            row, col = divmod(idx, cols)
            x_block, y_block = col * bs, row * bs
            adv = self.font.getlength(ch)
            arr = self._render_char_block(ch)
            if arr is None:
                new_glyphs[ch] = {
                    'u0': 0.0, 'v0': 0.0, 'u1': 0.0, 'v1': 0.0,
                    'bs': bs, 'px': self.pen_x, 'py': self.pen_y,
                    'adv': adv, 'empty': True,
                }
                continue
            img.paste(Image.fromarray(arr, 'L'), (x_block, y_block))
            new_glyphs[ch] = {
                'u0': x_block / W,
                'v0': y_block / H,
                'u1': (x_block + bs) / W,
                'v1': (y_block + bs) / H,
                'bs': bs, 'px': self.pen_x, 'py': self.pen_y,
                'adv': adv, 'empty': False,
            }

        self.atlas_img = img
        self.atlas_w, self.atlas_h = W, H
        self.glyphs = new_glyphs

    def add_chars(self, chars) -> bool:
        new = [c for c in set(chars) if c not in self.glyphs]
        if not new:
            return False
        self.order.extend(sorted(new))
        self._rebuild_atlas()
        return True

    def to_rgba_array(self):
        a = np.array(self.atlas_img, dtype=np.uint8)
        out = np.zeros((a.shape[0], a.shape[1], 4), dtype=np.uint8)
        out[:, :, 0] = a
        out[:, :, 1] = a
        out[:, :, 2] = a
        out[:, :, 3] = 255
        return out


# ============================================================
# Raster2D
# ============================================================
class Raster2D:
    VERT_STRIDE = 32  # 8 * 4

    def __init__(self, W=1280, H=720, verbose=False):
        if not _HAS_PIL:
            raise RuntimeError("PIL/Pillow 是必需的依赖")

        self.W, self.H = int(W), int(H)
        self.verbose = verbose

        self._egl = self._gles = None
        self._display = self._surface = self._context = None
        self._fbo = self._color_tex = 0
        self._prog = 0
        self._u: dict = {}
        self._a: dict = {}
        self._vbo = 0

        self._verts: list = []
        self._cur_tex = None
        self._cur_mode = 2
        self._default_tex = None
        self._tex_cache: dict = {}
        self._atlas_tex: dict = {}

        self._pixel_buf = (c_ubyte * (self.W * self.H * 4))()
        self._mat: Mat2D | None = None

        self._init_egl()
        self._init_fbo()
        self._init_shaders()
        self._create_default_tex()

        g = self._gles
        g.glEnable(GL_BLEND)
        g.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        g.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    # ---------- EGL ----------
    def _init_egl(self):
        _ep, _gp, _how = _detect_egl_libs()
        if _ep is None:
            raise RuntimeError(
                "找不到 libEGL/GLESv2。\n"
                "  可通过环境变量指定：\n"
                "    XMLVE_EGL_LIB=/path/to/libEGL.so.1 \\\n"
                "    XMLVE_GLES_LIB=/path/to/libGLESv2.so.2 python ..."
            )
        try:
            self._egl = ctypes.CDLL(_ep)
            self._gles = ctypes.CDLL(_gp)
        except OSError as _e:
            raise RuntimeError(f"加载 {_ep} / {_gp} 失败: {_e}")
        if self.verbose:
            print(f"[2D] EGL={_ep}  ({_how})", file=sys.stderr)

        e = self._egl
        e.eglGetDisplay.restype = c_void_p
        e.eglGetDisplay.argtypes = [c_void_p]
        e.eglInitialize.restype = c_uint
        e.eglInitialize.argtypes = [c_void_p, ctypes.POINTER(c_int), ctypes.POINTER(c_int)]
        e.eglBindAPI.restype = c_uint
        e.eglBindAPI.argtypes = [c_uint]
        e.eglChooseConfig.restype = c_uint
        e.eglChooseConfig.argtypes = [c_void_p, ctypes.POINTER(c_int),
                                      ctypes.POINTER(c_void_p), c_int,
                                      ctypes.POINTER(c_int)]
        e.eglCreatePbufferSurface.restype = c_void_p
        e.eglCreatePbufferSurface.argtypes = [c_void_p, c_void_p, ctypes.POINTER(c_int)]
        e.eglCreateContext.restype = c_void_p
        e.eglCreateContext.argtypes = [c_void_p, c_void_p, c_void_p, ctypes.POINTER(c_int)]
        e.eglMakeCurrent.restype = c_uint
        e.eglMakeCurrent.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p]

        self._display = e.eglGetDisplay(EGL_DEFAULT_DISPLAY)
        maj, mn = c_int(), c_int()
        if not e.eglInitialize(self._display, ctypes.byref(maj), ctypes.byref(mn)):
            raise RuntimeError("eglInitialize failed")
        e.eglBindAPI(EGL_OPENGL_ES_API)

        cfg = (c_int * 15)(
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE)
        cfgs = (c_void_p * 1)()
        n = c_int()
        if (not e.eglChooseConfig(self._display, cfg, cfgs, 1, ctypes.byref(n))
                or n.value == 0):
            cfg[3] = EGL_OPENGL_ES2_BIT
            e.eglChooseConfig(self._display, cfg, cfgs, 1, ctypes.byref(n))
        config = cfgs[0]

        pbuf = (c_int * 5)(EGL_WIDTH, self.W, EGL_HEIGHT, self.H, EGL_NONE)
        self._surface = e.eglCreatePbufferSurface(self._display, config, pbuf)

        ctx = (c_int * 3)(EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE)
        self._context = e.eglCreateContext(self._display, config, 0, ctx)
        if not self._context:
            ctx[1] = 2
            self._context = e.eglCreateContext(self._display, config, 0, ctx)
        if not self._context:
            raise RuntimeError("eglCreateContext failed")
        if not e.eglMakeCurrent(self._display, self._surface, self._surface, self._context):
            raise RuntimeError("eglMakeCurrent failed")

        g = self._gles
        g.glGetString.restype = c_char_p
        g.glGetString.argtypes = [c_uint]
        g.glGetIntegerv.argtypes = [c_uint, ctypes.POINTER(c_int)]
        if self.verbose:
            r = g.glGetString(GL_RENDERER)
            print(f"[2D] {r.decode() if r else '?'}", file=sys.stderr)
            mx = c_int()
            g.glGetIntegerv(GL_MAX_TEXTURE_SIZE, ctypes.byref(mx))
            print(f"[2D] GL_MAX_TEXTURE_SIZE = {mx.value}", file=sys.stderr)

    # ---------- FBO / 顶点 / 纹理 ----------
    def _init_fbo(self):
        g = self._gles
        sig = [
            ('glCreateShader', [c_uint]),
            ('glShaderSource', [c_uint, c_int, ctypes.POINTER(c_char_p), ctypes.POINTER(c_int)]),
            ('glCompileShader', [c_uint]),
            ('glGetShaderiv', [c_uint, c_uint, ctypes.POINTER(c_int)]),
            ('glGetShaderInfoLog', [c_uint, c_int, ctypes.POINTER(c_int), c_char_p]),
            ('glCreateProgram', []),
            ('glAttachShader', [c_uint, c_uint]),
            ('glLinkProgram', [c_uint]),
            ('glGetProgramiv', [c_uint, c_uint, ctypes.POINTER(c_int)]),
            ('glGetProgramInfoLog', [c_uint, c_int, ctypes.POINTER(c_int), c_char_p]),
            ('glUseProgram', [c_uint]),
            ('glGetUniformLocation', [c_uint, c_char_p]),
            ('glGetAttribLocation', [c_uint, c_char_p]),
            ('glGenBuffers', [c_int, ctypes.POINTER(c_uint)]),
            ('glBindBuffer', [c_uint, c_uint]),
            ('glBufferData', [c_uint, c_size_t, c_void_p, c_uint]),
            ('glEnableVertexAttribArray', [c_uint]),
            ('glVertexAttribPointer', [c_uint, c_int, c_uint, c_ubyte, c_int, c_void_p]),
            ('glUniform1i', [c_int, c_int]),
            ('glActiveTexture', [c_uint]),
            ('glEnable', [c_uint]),
            ('glBlendFunc', [c_uint, c_uint]),
            ('glViewport', [c_int, c_int, c_int, c_int]),
            ('glClearColor', [c_float, c_float, c_float, c_float]),
            ('glClear', [c_uint]),
            ('glDrawArrays', [c_uint, c_int, c_int]),
            ('glFinish', []),
            ('glGenFramebuffers', [c_int, ctypes.POINTER(c_uint)]),
            ('glBindFramebuffer', [c_uint, c_uint]),
            ('glFramebufferTexture2D', [c_uint, c_uint, c_uint, c_uint, c_int]),
            ('glGenTextures', [c_int, ctypes.POINTER(c_uint)]),
            ('glBindTexture', [c_uint, c_uint]),
            ('glTexImage2D', [c_uint, c_int, c_int, c_int, c_int, c_int,
                              c_uint, c_uint, c_void_p]),
            ('glTexParameteri', [c_uint, c_uint, c_int]),
            ('glReadPixels', [c_int, c_int, c_int, c_int, c_uint, c_uint, c_void_p]),
            ('glPixelStorei', [c_uint, c_int]),
            ('glDeleteTextures', [c_int, ctypes.POINTER(c_uint)]),
            ('glDeleteBuffers', [c_int, ctypes.POINTER(c_uint)]),
            ('glDeleteFramebuffers', [c_int, ctypes.POINTER(c_uint)]),
        ]
        for name, args in sig:
            fn = getattr(g, name, None)
            if fn is None:
                raise RuntimeError(f"GLESv2 缺少符号: {name}")
            fn.argtypes = args
        g.glCreateShader.restype = c_uint
        g.glCreateProgram.restype = c_uint
        g.glGetUniformLocation.restype = c_int
        g.glGetAttribLocation.restype = c_int

        tex = c_uint()
        g.glGenTextures(1, ctypes.byref(tex))
        self._color_tex = tex.value
        g.glBindTexture(GL_TEXTURE_2D, self._color_tex)
        g.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.W, self.H, 0,
                       GL_RGBA, GL_UNSIGNED_BYTE, None)
        for p in (GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER):
            g.glTexParameteri(GL_TEXTURE_2D, p, GL_LINEAR)
        for p in (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T):
            g.glTexParameteri(GL_TEXTURE_2D, p, GL_CLAMP_TO_EDGE)

        fbo = c_uint()
        g.glGenFramebuffers(1, ctypes.byref(fbo))
        self._fbo = fbo.value
        g.glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        g.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                 GL_TEXTURE_2D, self._color_tex, 0)
        g.glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _compile(self, src: bytes, kind):
        g = self._gles
        sh = g.glCreateShader(kind)
        sp = c_char_p(src)
        ln = c_int(len(src))
        g.glShaderSource(sh, 1, ctypes.byref(sp), ctypes.byref(ln))
        g.glCompileShader(sh)
        st = c_int()
        g.glGetShaderiv(sh, GL_COMPILE_STATUS, ctypes.byref(st))
        if not st.value:
            log = ctypes.create_string_buffer(4096)
            g.glGetShaderInfoLog(sh, 4096, None, log)
            raise RuntimeError(f"shader: {log.value.decode(errors='replace')}")
        return sh

    def _init_shaders(self):
        g = self._gles
        vs = b"""#version 300 es
in vec2 a_pos; in vec2 a_uv; in vec4 a_color;
out vec2 v_uv; out vec4 v_color;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
    v_color = a_color;
}
"""
        fs = b"""#version 300 es
precision mediump float;
in vec2 v_uv; in vec4 v_color;
uniform sampler2D u_tex; uniform int u_mode;
out vec4 frag;
void main() {
    if (u_mode == 0) {
        float a = texture(u_tex, v_uv).r;
        frag = vec4(v_color.rgb, v_color.a * a);
    } else if (u_mode == 1) {
        frag = v_color * texture(u_tex, v_uv);
    } else {
        frag = v_color;
    }
}
"""
        v = self._compile(vs, GL_VERTEX_SHADER)
        f = self._compile(fs, GL_FRAGMENT_SHADER)
        p = g.glCreateProgram()
        g.glAttachShader(p, v)
        g.glAttachShader(p, f)
        g.glLinkProgram(p)
        st = c_int()
        g.glGetProgramiv(p, GL_LINK_STATUS, ctypes.byref(st))
        if not st.value:
            log = ctypes.create_string_buffer(4096)
            g.glGetProgramInfoLog(p, 4096, None, log)
            raise RuntimeError(f"link: {log.value.decode(errors='replace')}")
        self._prog = p
        self._u['tex'] = g.glGetUniformLocation(p, b'u_tex')
        self._u['mode'] = g.glGetUniformLocation(p, b'u_mode')
        self._a['pos'] = g.glGetAttribLocation(p, b'a_pos')
        self._a['uv'] = g.glGetAttribLocation(p, b'a_uv')
        self._a['color'] = g.glGetAttribLocation(p, b'a_color')

        vbo = c_uint()
        g.glGenBuffers(1, ctypes.byref(vbo))
        self._vbo = vbo.value

    def _create_default_tex(self):
        g = self._gles
        tex = c_uint()
        g.glGenTextures(1, ctypes.byref(tex))
        g.glBindTexture(GL_TEXTURE_2D, tex.value)
        w = np.array([255, 255, 255, 255], dtype=np.uint8)
        g.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0,
                       GL_RGBA, GL_UNSIGNED_BYTE, w.ctypes.data_as(c_void_p))
        for p in (GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER):
            g.glTexParameteri(GL_TEXTURE_2D, p, GL_LINEAR)
        self._default_tex = tex.value

    def _upload_tex(self, key, rgba):
        if key in self._tex_cache:
            return self._tex_cache[key]
        g = self._gles
        arr = np.ascontiguousarray(rgba)
        h, w = arr.shape[:2]
        tex = c_uint()
        g.glGenTextures(1, ctypes.byref(tex))
        g.glBindTexture(GL_TEXTURE_2D, tex.value)
        g.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0,
                       GL_RGBA, GL_UNSIGNED_BYTE, arr.ctypes.data_as(c_void_p))
        for p in (GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER):
            g.glTexParameteri(GL_TEXTURE_2D, p, GL_LINEAR)
        for p in (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T):
            g.glTexParameteri(GL_TEXTURE_2D, p, GL_CLAMP_TO_EDGE)
        self._tex_cache[key] = tex.value
        return tex.value

    def load_font_chars(self, font_name, size, chars, weight='normal'):
        key = (font_name, int(size), str(weight))
        if key not in self._atlas_tex:
            atlas = GlyphAtlas.get(font_name, size, weight)
            self._atlas_tex[key] = (None, atlas)
        tid, atlas = self._atlas_tex[key]
        tex_key = ('atlas', key)
        changed = atlas.add_chars(chars)
        if changed or tid is None:
            # 释放旧纹理
            if tid is not None:
                old = c_uint(tid)
                self._gles.glDeleteTextures(1, ctypes.byref(old))
            self._tex_cache.pop(tex_key, None)
            tid = self._upload_tex(tex_key, atlas.to_rgba_array())
            self._atlas_tex[key] = (tid, atlas)
        return tid, atlas

    # ---------- 绘制状态 ----------
    def _flush(self):
        if not self._verts:
            return
        arr = np.asarray(self._verts, dtype=np.float32).ravel()
        g = self._gles
        g.glUseProgram(self._prog)
        g.glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        g.glBufferData(GL_ARRAY_BUFFER, arr.nbytes,
                       arr.ctypes.data_as(c_void_p), GL_DYNAMIC_DRAW)
        g.glActiveTexture(GL_TEXTURE0)
        g.glBindTexture(GL_TEXTURE_2D, self._cur_tex or self._default_tex)
        g.glUniform1i(self._u['tex'], 0)
        g.glUniform1i(self._u['mode'], self._cur_mode)

        stride = self.VERT_STRIDE
        if self._a['pos'] >= 0:
            g.glEnableVertexAttribArray(self._a['pos'])
            g.glVertexAttribPointer(self._a['pos'], 2, GL_FLOAT, 0, stride, c_void_p(0))
        if self._a['uv'] >= 0:
            g.glEnableVertexAttribArray(self._a['uv'])
            g.glVertexAttribPointer(self._a['uv'], 2, GL_FLOAT, 0, stride, c_void_p(8))
        if self._a['color'] >= 0:
            g.glEnableVertexAttribArray(self._a['color'])
            g.glVertexAttribPointer(self._a['color'], 4, GL_FLOAT, 0, stride, c_void_p(16))

        g.glDrawArrays(GL_TRIANGLES, 0, len(self._verts))
        self._verts.clear()

    def _bind(self, tex_id, mode):
        if self._cur_tex == tex_id and self._cur_mode == mode:
            return
        self._flush()
        self._cur_tex = tex_id
        self._cur_mode = mode

    def _ndc_x(self, x): return x / self.W * 2.0 - 1.0
    def _ndc_y(self, y): return 1.0 - y / self.H * 2.0

    def set_matrix(self, m: Mat2D | None):
        """设置当前仿射矩阵（None = 单位）"""
        self._mat = m

    def reset_matrix(self):
        self._mat = None

    def _apply_mat(self, x, y):
        return (x, y) if self._mat is None else self._mat.apply(x, y)

    # ---------- 基元 ----------
    def _push_quad_pts(self, p0, p1, p2, p3, uv, color):
        """p0=tl, p1=tr, p2=br, p3=bl（局部坐标）"""
        u0, v0, u1, v1 = uv
        r, g, b, a = color

        def V(p, u, vv):
            px, py = self._apply_mat(p[0], p[1])
            return (self._ndc_x(px), self._ndc_y(py), u, vv, r, g, b, a)

        v = self._verts
        v.append(V(p0, u0, v0))
        v.append(V(p3, u0, v1))
        v.append(V(p2, u1, v1))
        v.append(V(p0, u0, v0))
        v.append(V(p2, u1, v1))
        v.append(V(p1, u1, v0))

    def _push_quad(self, x0, y0, x1, y1, u0, v0, u1, v1, color):
        self._push_quad_pts((x0, y0), (x1, y0), (x1, y1), (x0, y1),
                            (u0, v0, u1, v1), color)

    def _push_triangle_2d(self, p0, p1, p2, color):
        r, g, b, a = color
        for p in (p0, p1, p2):
            px, py = self._apply_mat(p[0], p[1])
            self._verts.append((self._ndc_x(px), self._ndc_y(py),
                                0.0, 0.0, r, g, b, a))

    # ---------- 生命周期 ----------
    def begin(self, clear=(0.0, 0.0, 0.0, 1.0)):
        g = self._gles
        g.glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        g.glViewport(0, 0, self.W, self.H)
        g.glClearColor(*clear)
        g.glClear(GL_COLOR_BUFFER_BIT)
        self._verts.clear()
        self._cur_tex = None
        self._cur_mode = 2
        self._mat = None

    def end(self):
        self._flush()
        g = self._gles
        g.glFinish()
        g.glPixelStorei(GL_PACK_ALIGNMENT, 1)
        g.glReadPixels(0, 0, self.W, self.H, GL_RGBA, GL_UNSIGNED_BYTE,
                       self._pixel_buf)
        g.glBindFramebuffer(GL_FRAMEBUFFER, 0)
        arr = np.frombuffer(self._pixel_buf, dtype=np.uint8).reshape(self.H, self.W, 4)
        return arr[::-1].copy()

    def close(self):
        try:
            if self._tex_cache:
                ids = (c_uint * len(self._tex_cache))(*self._tex_cache.values())
                self._gles.glDeleteTextures(len(ids), ids)
                self._tex_cache.clear()
            if self._default_tex:
                t = c_uint(self._default_tex)
                self._gles.glDeleteTextures(1, ctypes.byref(t))
                self._default_tex = None
            if self._color_tex:
                t = c_uint(self._color_tex)
                self._gles.glDeleteTextures(1, ctypes.byref(t))
                self._color_tex = 0
            if self._vbo:
                b = c_uint(self._vbo)
                self._gles.glDeleteBuffers(1, ctypes.byref(b))
                self._vbo = 0
            if self._fbo:
                f = c_uint(self._fbo)
                self._gles.glDeleteFramebuffers(1, ctypes.byref(f))
                self._fbo = 0
        except Exception:
            pass
        try:
            if self._egl and self._display:
                self._egl.eglMakeCurrent(self._display, None, None, None)
                if self._context:
                    self._egl.eglDestroyContext(self._display, self._context)
                    self._context = None
                if self._surface:
                    self._egl.eglDestroySurface(self._display, self._surface)
                    self._surface = None
                self._egl.eglTerminate(self._display)
                self._display = None
        except Exception:
            pass

    # ---------- 图元 ----------
    def draw_rect(self, x, y, w, h, color):
        self._bind(self._default_tex, 2)
        self._push_quad(x, y, x + w, y + h, 0, 0, 1, 1, color)

    def draw_line(self, x0, y0, x1, y1, width, color, cap='butt'):
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < 1e-9:
            if cap == 'round':
                self.draw_circle(x0, y0, width * 0.5, color)
            return
        nx = -dy / L * width * 0.5
        ny =  dx / L * width * 0.5
        self._bind(self._default_tex, 2)
        p0 = (x0 + nx, y0 + ny)
        p1 = (x1 + nx, y1 + ny)
        p2 = (x1 - nx, y1 - ny)
        p3 = (x0 - nx, y0 - ny)
        self._push_quad_pts(p0, p1, p2, p3, (0, 0, 1, 1), color)

        if cap == 'round':
            self.draw_circle(x0, y0, width * 0.5, color)
            self.draw_circle(x1, y1, width * 0.5, color)
        elif cap == 'square':
            ux, uy = dx / L, dy / L
            self._push_quad_pts(
                (x0 + nx - ux * width * 0.5, y0 + ny - uy * width * 0.5),
                (x0 + nx, y0 + ny),
                (x0 - nx, y0 - ny),
                (x0 - nx - ux * width * 0.5, y0 - ny - uy * width * 0.5),
                (0, 0, 1, 1), color)
            self._push_quad_pts(
                (x1 + nx, y1 + ny),
                (x1 + nx + ux * width * 0.5, y1 + ny + uy * width * 0.5),
                (x1 - nx + ux * width * 0.5, y1 - ny + uy * width * 0.5),
                (x1 - nx, y1 - ny),
                (0, 0, 1, 1), color)

    def draw_polyline(self, points, width, color, join='round'):
        if len(points) < 2:
            return
        for i in range(len(points) - 1):
            self.draw_line(points[i][0], points[i][1],
                           points[i + 1][0], points[i + 1][1],
                           width, color)
        if join == 'round' and len(points) > 2:
            r = width * 0.5
            for i in range(1, len(points) - 1):
                self.draw_circle(points[i][0], points[i][1], r, color)

    def draw_polygon(self, points, color):
        n = len(points)
        if n < 3:
            return
        cx = sum(p[0] for p in points) / n
        cy = sum(p[1] for p in points) / n
        self._bind(self._default_tex, 2)
        for i in range(n):
            self._push_triangle_2d((cx, cy), points[i],
                                   points[(i + 1) % n], color)

    def draw_ellipse(self, cx, cy, rx, ry, color, segments=48):
        self._bind(self._default_tex, 2)
        for i in range(segments):
            a0 = 2 * math.pi * i / segments
            a1 = 2 * math.pi * (i + 1) / segments
            p0 = (cx + rx * math.cos(a0), cy + ry * math.sin(a0))
            p1 = (cx + rx * math.cos(a1), cy + ry * math.sin(a1))
            self._push_triangle_2d((cx, cy), p0, p1, color)

    def draw_circle(self, cx, cy, r, color, segments=48):
        self.draw_ellipse(cx, cy, r, r, color, segments)

    def draw_image(self, img, x, y, w, h, opacity=1.0):
        if isinstance(img, (bytes, bytearray)):
            key = ('img', hashlib.md5(img).hexdigest())
            rgba = np.array(Image.open(BytesIO(img)).convert('RGBA'))
        elif hasattr(img, 'convert'):
            rgba = np.array(img.convert('RGBA'))
            key = ('pil', img.tobytes().__hash__() if hasattr(img, 'tobytes')
                   else id(img))
        else:
            arr = np.asarray(img, dtype=np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr,
                                np.full_like(arr, 255)], axis=-1)
            elif arr.shape[2] == 3:
                arr = np.concatenate(
                    [arr, np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)],
                    axis=-1)
            key = ('arr', arr.shape, hashlib.md5(arr.tobytes()).hexdigest())
            rgba = arr
        tid = self._upload_tex(key, rgba)
        self._bind(tid, 1)
        self._push_quad(x, y, x + w, y + h, 0, 0, 1, 1, (1, 1, 1, opacity))

    # ---------- 文本 ----------
    def draw_text(self, text, x, y, size, color,
                  font='sans-serif', anchor='start', baseline='middle',
                  weight='normal'):
        """整串文本渲染成 bitmap 后贴纹理。

        位置由 PIL 的整串 layout 一手决定 —— 不再逐字拼 glyph quad。
        """
        if not text:
            return
        path, index, _ = resolve_font_path(font, weight)
        if not path:
            return
        want = _parse_weight(weight)
        synth = _SYNTHETIC_BOLD.get((font, want), False)

        rgba, off_x, off_y, adv, asc, desc = _render_text_bitmap(
            text, path, index, size, synth)

        # 水平：pen 原点（'la' 锚点）
        if anchor == 'middle':
            pen_x = x - adv / 2
        elif anchor == 'end':
            pen_x = x - adv
        else:
            pen_x = x

        # 垂直：pen 原点（'la' 锚点 = ascender 顶部）
        if baseline in ('middle', 'central'):
            # SVG 规范：em box 中线
            pen_y = y - (asc + desc) / 2
        elif baseline in ('top', 'hanging', 'text-top'):
            pen_y = y
        elif baseline in ('bottom', 'text-bottom', 'ideographic'):
            pen_y = y - (asc + desc)
        else:  # alphabetic
            pen_y = y - asc

        # buffer 在屏幕上的位置
        sx0 = pen_x + off_x
        sy0 = pen_y + off_y
        h, w = rgba.shape[:2]
        sx1 = sx0 + w
        sy1 = sy0 + h

        tex_key = ('__text_bmp__', path, int(index),
                   int(round(size)), bool(synth), text)
        tid = self._upload_tex(tex_key, rgba)
        self._bind(tid, 1)   # mode=1: v_color * texture

        r, g, b, a = color
        self._push_quad(sx0, sy0, sx1, sy1, 0, 0, 1, 1, (r, g, b, a))

# ============================================================
# 单例（按线程隔离）
# ============================================================
_SINGLETON: dict = {}


def get_raster2d(W, H, verbose=False):
    import threading
    key = (W, H, threading.get_ident())
    if key not in _SINGLETON:
        _SINGLETON[key] = Raster2D(W, H, verbose=verbose)
    return _SINGLETON[key]