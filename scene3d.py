#!/usr/bin/env python3
"""
scene3d.py v4 —— GPU 3D + 多光源 + 每面不同纹理 + 球/柱/锥
"""
from __future__ import annotations

import base64
import ctypes
import math
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path

import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

c_void_p = ctypes.c_void_p
c_int = ctypes.c_int
c_uint = ctypes.c_uint
c_float = ctypes.c_float
c_char_p = ctypes.c_char_p
c_size_t = ctypes.c_size_t
c_ubyte = ctypes.c_ubyte

# EGL
EGL_DEFAULT_DISPLAY = 0
EGL_NO_CONTEXT = 0
EGL_OPENGL_ES_API = 0x30A0
EGL_SURFACE_TYPE = 0x3033
EGL_PBUFFER_BIT = 0x0001
EGL_RENDERABLE_TYPE = 0x3040
EGL_OPENGL_ES2_BIT = 0x0004
EGL_OPENGL_ES3_BIT_KHR = 0x0040
EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_BLUE_SIZE, EGL_ALPHA_SIZE = 0x3024, 0x3023, 0x3022, 0x3021
EGL_DEPTH_SIZE = 0x3025
EGL_NONE = 0x3038
EGL_CONTEXT_CLIENT_VERSION = 0x3098
EGL_WIDTH, EGL_HEIGHT = 0x3057, 0x3056

# GL
GL_VENDOR, GL_RENDERER, GL_VERSION = 0x1F00, 0x1F01, 0x1F02
GL_VERTEX_SHADER, GL_FRAGMENT_SHADER = 0x8B31, 0x8B30
GL_COMPILE_STATUS, GL_LINK_STATUS = 0x8B81, 0x8B82
GL_ARRAY_BUFFER = 0x8892
GL_STATIC_DRAW = 0x88E4
GL_DYNAMIC_DRAW = 0x88E8
GL_FLOAT = 0x1406
GL_TRIANGLES = 0x0004
GL_DEPTH_TEST, GL_CULL_FACE = 0x0B71, 0x0B44
GL_BACK, GL_CCW = 0x0405, 0x0901
GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT = 0x4000, 0x0100
GL_RGBA, GL_RGBA8 = 0x1908, 0x8058
GL_UNSIGNED_BYTE = 0x1401
GL_FRAMEBUFFER, GL_RENDERBUFFER = 0x8D40, 0x8D41
GL_COLOR_ATTACHMENT0 = 0x8CE0
GL_DEPTH_ATTACHMENT = 0x8D00
GL_DEPTH_COMPONENT16 = 0x81A5
GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE0 = 0x84C0
GL_LINEAR = 0x2601
GL_CLAMP_TO_EDGE = 0x812F
GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER = 0x2801, 0x2800
GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T = 0x2802, 0x2803
GL_PACK_ALIGNMENT, GL_UNPACK_ALIGNMENT = 0x0D05, 0x0CF5

MAX_LIGHTS = 8
MAX_TEXTURES = 6

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
    _candidates = [
        # ── Android ──
        ('/system/lib64/libEGL.so',    '/system/lib64/libGLESv2.so'),
        ('/vendor/lib64/libEGL.so',    '/vendor/lib64/libGLESv2.so'),
        ('/system/lib/libEGL.so',      '/system/lib/libGLESv2.so'),
        # ── Debian / Ubuntu ──
        ('/usr/lib/x86_64-linux-gnu/libEGL.so.1',
         '/usr/lib/x86_64-linux-gnu/libGLESv2.so.2'),
        ('/usr/lib/aarch64-linux-gnu/libEGL.so.1',
         '/usr/lib/aarch64-linux-gnu/libGLESv2.so.2'),
        ('/usr/lib/arm-linux-gnueabihf/libEGL.so.1',
         '/usr/lib/arm-linux-gnueabihf/libGLESv2.so.2'),
        ('/usr/lib/arm-linux-gnueabi/libEGL.so.1',
         '/usr/lib/arm-linux-gnueabi/libGLESv2.so.2'),
        ('/usr/lib/x86_64-linux-gnu/mesa-egl/libEGL.so.1',
         '/usr/lib/x86_64-linux-gnu/mesa-egl/libGLESv2.so.2'),
        # ── Arch / Fedora / CentOS ──
        ('/usr/lib64/libEGL.so.1',    '/usr/lib64/libGLESv2.so.2'),
        ('/usr/lib/libEGL.so.1',      '/usr/lib/libGLESv2.so.2'),
        # ── NVIDIA ──
        ('/usr/lib64/libEGL_nvidia.so.0',
         '/usr/lib64/libGLESv2_nvidia.so.2'),
        # ── Raspberry Pi ──
        ('/opt/vc/lib/libEGL.so',     '/opt/vc/lib/libGLESv2.so'),
        ('/opt/vc/lib/libbrcmEGL.so', '/opt/vc/lib/libbrcmGLESv2.so'),
        # ── 通用 .so 名 ──
        ('libEGL.so.1', 'libGLESv2.so.2'),
        ('libEGL.so',   'libGLESv2.so'),
        # ── Mesa 软件渲染 ──
        ('/usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0',
         '/usr/lib/x86_64-linux-gnu/libGLESv2.so.2'),
    ]
    for _ep, _gp in _candidates:
        try:
            ctypes.CDLL(_ep)
            ctypes.CDLL(_gp)
            return (_ep, _gp, 'probe')
        except OSError:
            continue
    try:
        _ep = _cu.find_library('EGL')
        _gp = _cu.find_library('GLESv2')
        if _ep and _gp:
            return (_ep, _gp, 'find_library')
    except Exception:
        pass
    return (None, None, 'fail')



# 自动实例化：一组至少这么多同类型静态 mesh 才批量 draw
# 环境变量 XMLVE_INSTANCE_THRESHOLD 可调
INSTANCE_THRESHOLD = int(os.environ.get('XMLVE_INSTANCE_THRESHOLD', '16'))


# ============================================================
# 矩阵
# ============================================================
def m_perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect; m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def m_lookat(eye, center, up):
    eye = np.asarray(eye, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    f = center - eye; f = f / np.linalg.norm(f)
    s = np.cross(f, up); s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s; m[1, :3] = u; m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def m_rot_x(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32); m[1, 1] = c; m[1, 2] = -s; m[2, 1] = s; m[2, 2] = c
    return m


def m_rot_y(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32); m[0, 0] = c; m[0, 2] = s; m[2, 0] = -s; m[2, 2] = c
    return m


def m_rot_z(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32); m[0, 0] = c; m[0, 1] = -s; m[1, 0] = s; m[1, 1] = c
    return m


def m_translate(x, y, z):
    m = np.eye(4, dtype=np.float32); m[0, 3] = x; m[1, 3] = y; m[2, 3] = z
    return m


def m_scale(sx, sy, sz):
    m = np.eye(4, dtype=np.float32); m[0, 0] = sx; m[1, 1] = sy; m[2, 2] = sz
    return m


def normal_matrix(model):
    r = model[:3, :3].astype(np.float64)
    try: n = np.linalg.inv(r).T
    except np.linalg.LinAlgError: n = np.eye(3)
    return np.ascontiguousarray(n, dtype=np.float32)



# ============================================================
# 视锥剔除：frustum plane 提取 + 批量 AABB 测试
# ============================================================
def _euler_rot_matrix(rx, ry, rz):
    """Ry @ Rx @ Rz，与 m_rot_y @ m_rot_x @ m_rot_z 一致"""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float32)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float32)
    return Ry @ Rx @ Rz


def _local_halfsize(mesh):
    """mesh 在局部坐标系的 AABB 半长 (hx, hy, hz)"""
    typ = mesh.get('type', 'cube')
    parts = (mesh.get('size') or '1').split()
    def f(i, d=1.0): return float(parts[i]) if i < len(parts) else d
    if typ == 'cube':
        if len(parts) == 1:
            s = f(0); return np.array([s/2, s/2, s/2], dtype=np.float32)
        return np.array([f(0)/2, f(1)/2, f(2)/2], dtype=np.float32)
    if typ == 'sphere':
        r = f(0); return np.array([r, r, r], dtype=np.float32)
    if typ in ('cylinder', 'cone', 'prism'):
        r = f(0); h = mesh.get('_height', r * 2.0)
        return np.array([r, h/2, r], dtype=np.float32)
    if typ == 'plane':
        w = f(0); h = f(1, w)
        return np.array([w/2, 0.01, h/2], dtype=np.float32)
    if typ == 'wave':
        w = f(0); h = f(1, w)
        return np.array([w/2, 0.5, h/2], dtype=np.float32)  # deform margin
    if typ in ('flag', 'curl'):
        w = f(0); h = f(1, w)
        return np.array([w/2, h/2, 0.5], dtype=np.float32)  # deform margin
    if typ == 'paper':
        w = f(0); h = f(1, w)
        return np.array([w/2, h/2, 0.01], dtype=np.float32)
    s = f(0)
    return np.array([s, s, s], dtype=np.float32)


def mesh_world_aabb(mesh, t):
    """返回 (center(3,), halfsize(3,))，world space 的 AABB 的 AABB（保守上界）"""
    pos, rot, scl = _mesh_at(mesh, t)
    lh = _local_halfsize(mesh) * np.abs(scl)
    R = _euler_rot_matrix(float(rot[0]), float(rot[1]), float(rot[2]))
    wh = np.abs(R) @ lh
    return pos, wh


def frustum_planes(vp):
    """从 VP 矩阵提取 6 个平面 (6, 4)，法线指向 frustum 内部。
    Gribb-Hartmann 方法。"""
    m = np.asarray(vp, dtype=np.float64)
    planes = np.stack([
        m[3] + m[0],   # left
        m[3] - m[0],   # right
        m[3] + m[1],   # bottom
        m[3] - m[1],   # top
        m[3] + m[2],   # near
        m[3] - m[2],   # far
    ], axis=0)
    return planes.astype(np.float32)


def instanced_mesh_world_aabb(mesh):
    """instanced mesh 的 world AABB = 所有实例 AABB 的并集。
    实例的 pos/rot/scale 是静态的（parse 期生成），所以只算一次。"""
    insts = mesh.get('_instances') or []
    if not insts:
        return mesh_world_aabb(mesh, 0.0)
    lh = _local_halfsize(mesh)
    centers = np.empty((len(insts), 3), dtype=np.float32)
    halfs   = np.empty((len(insts), 3), dtype=np.float32)
    for i, (p, r, s) in enumerate(insts):
        lh_s = lh * np.abs(s)
        R = _euler_rot_matrix(float(r[0]), float(r[1]), float(r[2]))
        halfs[i] = np.abs(R) @ lh_s
        centers[i] = p
    cmin = (centers - halfs).min(axis=0)
    cmax = (centers + halfs).max(axis=0)
    return (cmin + cmax) * 0.5, (cmax - cmin) * 0.5

def batch_aabb_visible(centers, halfs, planes):
    """centers:(N,3) halfs:(N,3) planes:(6,4) → (N,) bool
    完全落在任一平面外侧 → False"""
    if len(centers) == 0:
        return np.zeros(0, dtype=bool)
    a = planes[:, 0]; b = planes[:, 1]; c = planes[:, 2]; d = planes[:, 3]
    sd = (centers[:, 0:1] * a[None, :]
          + centers[:, 1:2] * b[None, :]
          + centers[:, 2:3] * c[None, :]
          + d[None, :])
    rad = (halfs[:, 0:1] * np.abs(a)[None, :]
           + halfs[:, 1:2] * np.abs(b)[None, :]
           + halfs[:, 2:3] * np.abs(c)[None, :])
    outside = (sd + rad) < 0
    return ~outside.any(axis=1)


# ============================================================
# 几何：顶点格式 = pos(3)+normal(3)+color(3)+uv(2)+face_id(1) = 12 floats
# ============================================================
def _add_vert(verts, p, n, c, uv, fid):
    verts.append([p[0], p[1], p[2], n[0], n[1], n[2],
                  c[0], c[1], c[2], uv[0], uv[1], float(fid)])


def make_cube_vertices(size=1.0, face_colors=None):
    """size: float 或 (sx, sy, sz)。长方体便于建筑/柱形。"""
    if face_colors is None:
        face_colors = [
            (0.95, 0.30, 0.25), (0.55, 0.15, 0.12),
            (0.30, 0.85, 0.35), (0.12, 0.45, 0.18),
            (0.35, 0.60, 0.95), (0.15, 0.28, 0.55),
        ]
    if isinstance(size, (int, float)):
        sx = sy = sz = float(size)
    else:
        vals = [float(x) for x in size]
        if len(vals) >= 3:
            sx, sy, sz = vals[0], vals[1], vals[2]
        else:
            sx = sy = sz = vals[0]
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    faces = [
        ((0, 0, 1),  [(-hx, -hy, hz),  (hx, -hy, hz),  (hx, hy, hz),  (-hx, hy, hz)]),
        ((0, 0, -1), [(hx, -hy, -hz),  (-hx, -hy, -hz),(-hx, hy, -hz),(hx, hy, -hz)]),
        ((1, 0, 0),  [(hx, -hy, hz),   (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz)]),
        ((-1, 0, 0), [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)]),
        ((0, 1, 0),  [(-hx, hy, hz),   (hx, hy, hz),   (hx, hy, -hz), (-hx, hy, -hz)]),
        ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
    ]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    verts = []
    for fi, (n, vv) in enumerate(faces):
        c = face_colors[fi]
        for tri in [(0, 1, 2), (0, 2, 3)]:
            for vi in tri:
                _add_vert(verts, vv[vi], n, c, uvs[vi], fi)
    return np.asarray(verts, dtype=np.float32)


def make_plane_subdiv_vertices(w=4.0, h=4.0, seg_x=32, seg_z=32,
                                 color=(0.5, 0.5, 0.6), face_id=0):
    """细分平面（沿 XZ 轴，法线 +Y），用于顶点着色器变形"""
    sw, sh = w / 2, h / 2
    n = (0, 1, 0)
    verts = []
    for iz in range(seg_z):
        for ix in range(seg_x):
            x0 = -sw + w * ix / seg_x
            x1 = -sw + w * (ix + 1) / seg_x
            z0 = -sh + h * iz / seg_z
            z1 = -sh + h * (iz + 1) / seg_z
            u0 = ix / seg_x; u1 = (ix + 1) / seg_x
            v0 = iz / seg_z; v1 = (iz + 1) / seg_z
            p00 = (x0, 0, z0); p01 = (x1, 0, z0)
            p10 = (x0, 0, z1); p11 = (x1, 0, z1)
            # 三角形 1
            _add_vert(verts, p00, n, color, (u0, v0), face_id)
            _add_vert(verts, p10, n, color, (u0, v1), face_id)
            _add_vert(verts, p11, n, color, (u1, v1), face_id)
            # 三角形 2
            _add_vert(verts, p00, n, color, (u0, v0), face_id)
            _add_vert(verts, p11, n, color, (u1, v1), face_id)
            _add_vert(verts, p01, n, color, (u1, v0), face_id)
    return np.asarray(verts, dtype=np.float32)


def make_plane_subdiv_vertices_v(w=4.0, h=4.0, seg_x=32, seg_y=32,
                                    color=(0.5, 0.5, 0.6), face_id=0):
    """竖直细分平面（XY 平面，法线 +Z），用于旗帜/卷曲。

    和 make_plane_subdiv_vertices 的区别：
      · 水平版：点在 (x, 0, z)，沿 X 和 Z 展开，法线 +Y
      · 竖直版：点在 (x, y, 0)，沿 X 和 Y 展开，法线 +Z
    flag 变形的 shader 对 local.y 和 local.z 做波动 —— 在竖直平面上
    才能得到"旗面上下抖动 + 前后鼓起"的真实旗帜效果。
    """
    sw, sh = w / 2, h / 2
    n = (0, 0, 1)
    verts = []
    for iy in range(seg_y):
        for ix in range(seg_x):
            x0 = -sw + w * ix / seg_x
            x1 = -sw + w * (ix + 1) / seg_x
            y0 = -sh + h * iy / seg_y
            y1 = -sh + h * (iy + 1) / seg_y
            u0 = ix / seg_x; u1 = (ix + 1) / seg_x
            v0 = iy / seg_y; v1 = (iy + 1) / seg_y
            p00 = (x0, y0, 0); p01 = (x1, y0, 0)
            p10 = (x0, y1, 0); p11 = (x1, y1, 0)
            _add_vert(verts, p00, n, color, (u0, v0), face_id)
            _add_vert(verts, p10, n, color, (u0, v1), face_id)
            _add_vert(verts, p11, n, color, (u1, v1), face_id)
            _add_vert(verts, p00, n, color, (u0, v0), face_id)
            _add_vert(verts, p11, n, color, (u1, v1), face_id)
            _add_vert(verts, p01, n, color, (u1, v0), face_id)
    return np.asarray(verts, dtype=np.float32)


def make_plane_vertices(w=8.0, h=8.0, y=0.0, color=(0.25, 0.25, 0.30)):
    sw, sh = w / 2, h / 2
    n = (0, 1, 0); verts = []
    for tri, uvs in [
        ([(-sw, y, -sh), (sw, y, -sh), (sw, y, sh)], [(0, 0), (1, 0), (1, 1)]),
        ([(-sw, y, -sh), (sw, y, sh), (-sw, y, sh)], [(0, 0), (1, 1), (0, 1)]),
    ]:
        for i, p in enumerate(tri):
            _add_vert(verts, p, n, color, uvs[i], 0)
    return np.asarray(verts, dtype=np.float32)


def make_prism_vertices(radius=0.7, height=1.4, face_colors=None):
    if face_colors is None:
        face_colors = {
            'top': (0.95, 0.35, 0.30), 'bottom': (0.55, 0.15, 0.12),
            's1': (0.30, 0.85, 0.40), 's2': (0.18, 0.60, 0.28), 's3': (0.10, 0.40, 0.15),
        }
    r = radius; h = height / 2.0; s3 = math.sqrt(3) / 2.0
    A  = (0.0,  -h,  r); B  = ( r*s3, -h, -r/2); C  = (-r*s3, -h, -r/2)
    A2 = (0.0,   h,  r); B2 = ( r*s3,  h, -r/2); C2 = (-r*s3,  h, -r/2)
    verts = []
    tri_uv = [(0.5, 1.0), (0.0, 0.0), (1.0, 0.0)]
    quad_uv_1 = [(0, 1), (0, 0), (1, 0)]
    quad_uv_2 = [(0, 1), (1, 0), (1, 1)]
    n1 = np.array([3.0, 0.0, math.sqrt(3.0)]); n1 /= np.linalg.norm(n1); n1 = tuple(n1)
    n2 = (0.0, 0.0, -1.0)
    n3 = np.array([-3.0, 0.0, math.sqrt(3.0)]); n3 /= np.linalg.norm(n3); n3 = tuple(n3)

    # face 0 = top, 1 = bottom, 2/3/4 = sides
    for p, uv in zip((A2, B2, C2), tri_uv):
        _add_vert(verts, p, (0, 1, 0), face_colors['top'], uv, 0)
    for p, uv in zip((A, C, B), tri_uv):
        _add_vert(verts, p, (0, -1, 0), face_colors['bottom'], uv, 1)
    for p, uv in zip((A, B, B2), quad_uv_1):
        _add_vert(verts, p, n1, face_colors['s1'], uv, 2)
    for p, uv in zip((A, B2, A2), quad_uv_2):
        _add_vert(verts, p, n1, face_colors['s1'], uv, 2)
    for p, uv in zip((B, C, C2), quad_uv_1):
        _add_vert(verts, p, n2, face_colors['s2'], uv, 3)
    for p, uv in zip((B, C2, B2), quad_uv_2):
        _add_vert(verts, p, n2, face_colors['s2'], uv, 3)
    for p, uv in zip((C, A, A2), quad_uv_1):
        _add_vert(verts, p, n3, face_colors['s3'], uv, 4)
    for p, uv in zip((C, A2, C2), quad_uv_2):
        _add_vert(verts, p, n3, face_colors['s3'], uv, 4)
    return np.asarray(verts, dtype=np.float32)


def make_sphere_vertices(radius=0.8, segs=32, rings=16, color=(0.5, 0.7, 0.95)):
    verts = []
    for i in range(rings):
        lat0 = math.pi * (-0.5 + i / rings)
        lat1 = math.pi * (-0.5 + (i + 1) / rings)
        y0 = math.sin(lat0) * radius; y1 = math.sin(lat1) * radius
        r0 = math.cos(lat0) * radius; r1 = math.cos(lat1) * radius
        v0 = i / rings; v1 = (i + 1) / rings
        for j in range(segs):
            lng0 = 2 * math.pi * j / segs
            lng1 = 2 * math.pi * (j + 1) / segs
            u0 = j / segs; u1 = (j + 1) / segs
            p00 = (math.cos(lng0) * r0, y0, math.sin(lng0) * r0)
            p01 = (math.cos(lng1) * r0, y0, math.sin(lng1) * r0)
            p10 = (math.cos(lng0) * r1, y1, math.sin(lng0) * r1)
            p11 = (math.cos(lng1) * r1, y1, math.sin(lng1) * r1)
            n00 = tuple(np.array(p00) / radius)
            n01 = tuple(np.array(p01) / radius)
            n10 = tuple(np.array(p10) / radius)
            n11 = tuple(np.array(p11) / radius)
            # 两个三角形（跳过极点退化）
            if i > 0:
                _add_vert(verts, p00, n00, color, (u0, v0), 0)
                _add_vert(verts, p10, n10, color, (u0, v1), 0)
                _add_vert(verts, p11, n11, color, (u1, v1), 0)
            if i < rings - 1:
                _add_vert(verts, p00, n00, color, (u0, v0), 0)
                _add_vert(verts, p11, n11, color, (u1, v1), 0)
                _add_vert(verts, p01, n01, color, (u1, v0), 0)
    return np.asarray(verts, dtype=np.float32)


def make_cylinder_vertices(radius=0.6, height=1.4, segs=32, color=(0.5, 0.9, 0.5)):
    verts = []
    h = height / 2.0
    for j in range(segs):
        a0 = 2 * math.pi * j / segs
        a1 = 2 * math.pi * (j + 1) / segs
        x0, z0 = math.cos(a0) * radius, math.sin(a0) * radius
        x1, z1 = math.cos(a1) * radius, math.sin(a1) * radius
        n0 = (math.cos(a0), 0, math.sin(a0))
        n1 = (math.cos(a1), 0, math.sin(a1))
        u0 = j / segs; u1 = (j + 1) / segs
        p00 = (x0, -h, z0); p01 = (x1, -h, z1)
        p10 = (x0,  h, z0); p11 = (x1,  h, z1)
        # 侧面
        _add_vert(verts, p00, n0, color, (u0, 0), 2)
        _add_vert(verts, p10, n0, color, (u0, 1), 2)
        _add_vert(verts, p11, n1, color, (u1, 1), 2)
        _add_vert(verts, p00, n0, color, (u0, 0), 2)
        _add_vert(verts, p11, n1, color, (u1, 1), 2)
        _add_vert(verts, p01, n1, color, (u1, 0), 2)
        # 顶盖
        uvc = (0.5 + 0.5 * math.cos(a0), 0.5 + 0.5 * math.sin(a0))
        uvc1 = (0.5 + 0.5 * math.cos(a1), 0.5 + 0.5 * math.sin(a1))
        _add_vert(verts, (0, h, 0), (0, 1, 0), color, (0.5, 0.5), 0)
        _add_vert(verts, p11, (0, 1, 0), color, uvc1, 0)
        _add_vert(verts, p10, (0, 1, 0), color, uvc, 0)
        # 底盖
        _add_vert(verts, (0, -h, 0), (0, -1, 0), color, (0.5, 0.5), 1)
        _add_vert(verts, p00, (0, -1, 0), color, uvc, 1)
        _add_vert(verts, p01, (0, -1, 0), color, uvc1, 1)
    return np.asarray(verts, dtype=np.float32)


def make_cone_vertices(radius=0.7, height=1.5, segs=32, color=(0.95, 0.6, 0.3)):
    verts = []
    h = height / 2.0
    apex = (0.0, h, 0.0)
    for j in range(segs):
        a0 = 2 * math.pi * j / segs
        a1 = 2 * math.pi * (j + 1) / segs
        x0, z0 = math.cos(a0) * radius, math.sin(a0) * radius
        x1, z1 = math.cos(a1) * radius, math.sin(a1) * radius
        p0 = (x0, -h, z0); p1 = (x1, -h, z1)
        # 侧面（法线大致朝外上）
        n0 = np.array([x0, radius * 0.7, z0]); n0 /= np.linalg.norm(n0); n0 = tuple(n0)
        n1 = np.array([x1, radius * 0.7, z1]); n1 /= np.linalg.norm(n1); n1 = tuple(n1)
        u0 = j / segs; u1 = (j + 1) / segs
        _add_vert(verts, apex, n0, color, ((u0 + u1) / 2, 1), 2)
        _add_vert(verts, p0, n0, color, (u0, 0), 2)
        _add_vert(verts, p1, n1, color, (u1, 0), 2)
        # 底面
        uvc = (0.5 + 0.5 * math.cos(a0), 0.5 + 0.5 * math.sin(a0))
        uvc1 = (0.5 + 0.5 * math.cos(a1), 0.5 + 0.5 * math.sin(a1))
        _add_vert(verts, (0, -h, 0), (0, -1, 0), color, (0.5, 0.5), 1)
        _add_vert(verts, p0, (0, -1, 0), color, uvc, 1)
        _add_vert(verts, p1, (0, -1, 0), color, uvc1, 1)
    return np.asarray(verts, dtype=np.float32)


# ============================================================
# 纹理
# ============================================================
def _load_texture_bytes(spec):
    if isinstance(spec, str) and spec.startswith("data:"):
        header, _, b64 = spec.partition(",")
        data = base64.b64decode(b64)
        if "svg" in header:
            import cairosvg
            return cairosvg.svg2png(bytestring=data, output_width=512, output_height=512)
        return data
    if isinstance(spec, str) and spec.strip().startswith("<svg"):
        import cairosvg
        return cairosvg.svg2png(bytestring=spec.encode("utf-8"),
                                 output_width=512, output_height=512)
    p = Path(spec)
    if not p.exists():
        raise FileNotFoundError(f"纹理不存在: {spec}")
    if p.suffix.lower() == ".svg":
        import cairosvg
        return cairosvg.svg2png(url=str(p), output_width=512, output_height=512)
    return p.read_bytes()


def load_texture_rgba(spec):
    png_bytes = _load_texture_bytes(spec)
    img = Image.open(BytesIO(png_bytes)).convert("RGBA")
    return np.asarray(img, dtype=np.uint8)


# ============================================================
# GL3D
# ============================================================
class GL3D:
    def __init__(self, width=1280, height=720, verbose=True):
        self.W = width; self.H = height; self.verbose = verbose
        self._egl = self._gles = None
        self._display = self._surface = self._context = None
        self._fbo = self._color_tex = self._depth_rb = 0
        self._prog = 0
        self._u = {}       # uniform locations
        self._a = {}       # attrib locations
        self._meshes = {}
        self._textures = {}
        self._tex_by_key = {}
        self._pixel_buf = (c_ubyte * (width * height * 4))()

        self._init_egl(); self._init_fbo(); self._init_shaders()
        g = self._gles
        g.glEnable(GL_DEPTH_TEST)
        # 关闭 cull face：支持薄平面（旗帜/纸张）双面渲染
        # g.glEnable(GL_CULL_FACE)
        # g.glCullFace(GL_BACK)
        # g.glFrontFace(GL_CCW)
        g.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    def _init_egl(self):
        _ep, _gp, _how = _detect_egl_libs()
        if _ep is None:
            raise RuntimeError(
                "找不到 libEGL/GLESv2。\n"
                "  可通过环境变量指定：\n"
                "    XMLVE_EGL_LIB=/path/to/libEGL.so.1 "
                "XMLVE_GLES_LIB=/path/to/libGLESv2.so.2 python ..."
            )
        try:
            self._egl = ctypes.CDLL(_ep)
            self._gles = ctypes.CDLL(_gp)
        except OSError as _e:
            raise RuntimeError(f"加载 {_ep} / {_gp} 失败: {_e}")
        if self.verbose:
            print(f"[3D] EGL={_ep}  ({_how})", file=sys.stderr)
        self._setup_sigs()
        e = self._egl
        self._display = e.eglGetDisplay(EGL_DEFAULT_DISPLAY)
        maj, mn = c_int(), c_int()
        e.eglInitialize(self._display, ctypes.byref(maj), ctypes.byref(mn))
        e.eglBindAPI(EGL_OPENGL_ES_API)
        cfg = (c_int * 15)(
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
            EGL_DEPTH_SIZE, 24, EGL_NONE)
        cfgs = (c_void_p * 1)(); n = c_int()
        if not e.eglChooseConfig(self._display, cfg, cfgs, 1, ctypes.byref(n)) or n.value == 0:
            cfg[3] = EGL_OPENGL_ES2_BIT
            e.eglChooseConfig(self._display, cfg, cfgs, 1, ctypes.byref(n))
        config = cfgs[0]
        pbuf = (c_int * 5)(EGL_WIDTH, self.W, EGL_HEIGHT, self.H, EGL_NONE)
        self._surface = e.eglCreatePbufferSurface(self._display, config, pbuf)
        ctx = (c_int * 3)(EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE)
        self._context = e.eglCreateContext(self._display, config, EGL_NO_CONTEXT, ctx)
        if not self._context:
            ctx[1] = 2
            self._context = e.eglCreateContext(self._display, config, EGL_NO_CONTEXT, ctx)
        if not e.eglMakeCurrent(self._display, self._surface, self._surface, self._context):
            raise RuntimeError("eglMakeCurrent 失败")
        if self.verbose:
            g = self._gles
            print(f"[3D] {g.glGetString(GL_VENDOR).decode()}", file=sys.stderr)
            print(f"[3D] {g.glGetString(GL_RENDERER).decode()}", file=sys.stderr)
            print(f"[3D] {g.glGetString(GL_VERSION).decode()}", file=sys.stderr)

    def _setup_sigs(self):
        e = self._egl
        e.eglGetDisplay.restype = c_void_p; e.eglGetDisplay.argtypes = [c_void_p]
        e.eglInitialize.restype = c_uint
        e.eglInitialize.argtypes = [c_void_p, ctypes.POINTER(c_int), ctypes.POINTER(c_int)]
        e.eglBindAPI.restype = c_uint; e.eglBindAPI.argtypes = [c_uint]
        e.eglChooseConfig.restype = c_uint
        e.eglChooseConfig.argtypes = [c_void_p, ctypes.POINTER(c_int),
                                       ctypes.POINTER(c_void_p), c_int, ctypes.POINTER(c_int)]
        e.eglCreatePbufferSurface.restype = c_void_p
        e.eglCreatePbufferSurface.argtypes = [c_void_p, c_void_p, ctypes.POINTER(c_int)]
        e.eglCreateContext.restype = c_void_p
        e.eglCreateContext.argtypes = [c_void_p, c_void_p, c_void_p, ctypes.POINTER(c_int)]
        e.eglMakeCurrent.restype = c_uint
        e.eglMakeCurrent.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p]

        g = self._gles
        g.glGetString.restype = c_char_p; g.glGetString.argtypes = [c_uint]
        g.glCreateShader.restype = c_uint; g.glCreateShader.argtypes = [c_uint]
        g.glShaderSource.argtypes = [c_uint, c_int, ctypes.POINTER(c_char_p), ctypes.POINTER(c_int)]
        g.glCompileShader.argtypes = [c_uint]
        g.glGetShaderiv.argtypes = [c_uint, c_uint, ctypes.POINTER(c_int)]
        g.glGetShaderInfoLog.argtypes = [c_uint, c_int, ctypes.POINTER(c_int), c_char_p]
        g.glCreateProgram.restype = c_uint
        g.glAttachShader.argtypes = [c_uint, c_uint]
        g.glLinkProgram.argtypes = [c_uint]
        g.glGetProgramiv.argtypes = [c_uint, c_uint, ctypes.POINTER(c_int)]
        g.glGetProgramInfoLog.argtypes = [c_uint, c_int, ctypes.POINTER(c_int), c_char_p]
        g.glUseProgram.argtypes = [c_uint]
        g.glGetUniformLocation.restype = c_int
        g.glGetUniformLocation.argtypes = [c_uint, c_char_p]
        g.glGetAttribLocation.restype = c_int
        g.glGetAttribLocation.argtypes = [c_uint, c_char_p]
        g.glGenBuffers.argtypes = [c_int, ctypes.POINTER(c_uint)]
        g.glBindBuffer.argtypes = [c_uint, c_uint]
        g.glBufferData.argtypes = [c_uint, c_size_t, c_void_p, c_uint]
        g.glGenVertexArrays.argtypes = [c_int, ctypes.POINTER(c_uint)]
        g.glBindVertexArray.argtypes = [c_uint]
        g.glEnableVertexAttribArray.argtypes = [c_uint]
        g.glVertexAttribPointer.argtypes = [c_uint, c_int, c_uint, c_ubyte, c_int, c_void_p]
        g.glUniformMatrix4fv.argtypes = [c_int, c_int, c_ubyte, c_void_p]
        g.glUniformMatrix3fv.argtypes = [c_int, c_int, c_ubyte, c_void_p]
        g.glUniform3f.argtypes = [c_int, c_float, c_float, c_float]
        g.glUniform3fv.argtypes = [c_int, c_int, c_void_p]
        g.glUniform1f.argtypes = [c_int, c_float]
        g.glUniform1fv.argtypes = [c_int, c_int, c_void_p]
        g.glUniform1i.argtypes = [c_int, c_int]
        g.glUniform1iv.argtypes = [c_int, c_int, c_void_p]
        g.glActiveTexture.argtypes = [c_uint]
        g.glEnable.argtypes = [c_uint]
        g.glCullFace.argtypes = [c_uint]
        g.glFrontFace.argtypes = [c_uint]
        g.glViewport.argtypes = [c_int, c_int, c_int, c_int]
        g.glClearColor.argtypes = [c_float, c_float, c_float, c_float]
        g.glClear.argtypes = [c_uint]
        g.glDrawArrays.argtypes = [c_uint, c_int, c_int]
        g.glDrawArraysInstanced.argtypes = [c_uint, c_int, c_int, c_int]
        g.glVertexAttribDivisor.argtypes = [c_uint, c_uint]
        g.glFinish.argtypes = []
        g.glGenFramebuffers.argtypes = [c_int, ctypes.POINTER(c_uint)]
        g.glBindFramebuffer.argtypes = [c_uint, c_uint]
        g.glFramebufferTexture2D.argtypes = [c_uint, c_uint, c_uint, c_uint, c_int]
        g.glGenRenderbuffers.argtypes = [c_int, ctypes.POINTER(c_uint)]
        g.glBindRenderbuffer.argtypes = [c_uint, c_uint]
        g.glRenderbufferStorage.argtypes = [c_uint, c_uint, c_int, c_int]
        g.glFramebufferRenderbuffer.argtypes = [c_uint, c_uint, c_uint, c_uint]
        g.glGenTextures.argtypes = [c_int, ctypes.POINTER(c_uint)]
        g.glBindTexture.argtypes = [c_uint, c_uint]
        g.glTexImage2D.argtypes = [c_uint, c_int, c_int, c_int, c_int, c_int,
                                    c_uint, c_uint, c_void_p]
        g.glTexParameteri.argtypes = [c_uint, c_uint, c_int]
        g.glReadPixels.argtypes = [c_int, c_int, c_int, c_int, c_uint, c_uint, c_void_p]
        g.glPixelStorei.argtypes = [c_uint, c_int]
        g.glDeleteTextures.argtypes = [c_int, ctypes.POINTER(c_uint)]

    def _init_fbo(self):
        g = self._gles
        tex = c_uint(); g.glGenTextures(1, ctypes.byref(tex))
        self._color_tex = tex.value
        g.glBindTexture(GL_TEXTURE_2D, self._color_tex)
        g.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.W, self.H, 0,
                        GL_RGBA, GL_UNSIGNED_BYTE, None)
        for p in (GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER): g.glTexParameteri(GL_TEXTURE_2D, p, GL_LINEAR)
        for p in (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T): g.glTexParameteri(GL_TEXTURE_2D, p, GL_CLAMP_TO_EDGE)
        rb = c_uint(); g.glGenRenderbuffers(1, ctypes.byref(rb))
        self._depth_rb = rb.value
        g.glBindRenderbuffer(GL_RENDERBUFFER, self._depth_rb)
        g.glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, self.W, self.H)
        fbo = c_uint(); g.glGenFramebuffers(1, ctypes.byref(fbo))
        self._fbo = fbo.value
        g.glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        g.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._color_tex, 0)
        g.glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._depth_rb)
        g.glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _compile(self, src, kind):
        g = self._gles
        sh = g.glCreateShader(kind)
        sp = c_char_p(src); ln = c_int(len(src))
        g.glShaderSource(sh, 1, ctypes.byref(sp), ctypes.byref(ln))
        g.glCompileShader(sh)
        st = c_int(); g.glGetShaderiv(sh, GL_COMPILE_STATUS, ctypes.byref(st))
        if not st.value:
            log = ctypes.create_string_buffer(4096)
            g.glGetShaderInfoLog(sh, 4096, None, log)
            raise RuntimeError(f"shader: {log.value.decode()}")
        return sh

    def _init_shaders(self):
        g = self._gles
        # vertex shader
        vs = """#version 300 es
layout(location=0) in vec3 a_pos;
layout(location=1) in vec3 a_normal;
layout(location=2) in vec3 a_color;
layout(location=3) in vec2 a_uv;
layout(location=4) in float a_face_id;
layout(location=5) in vec4 a_inst0;
layout(location=6) in vec4 a_inst1;
layout(location=7) in vec4 a_inst2;
layout(location=8) in vec4 a_inst3;
layout(location=9) in vec4 a_inst_color;

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat4 u_vp;
uniform mat3 u_normal_mat;
uniform int  u_instanced;
uniform int  u_inst_use_color;
uniform int  u_flat_shading;

// 顶点着色器变形
uniform float u_time;
uniform int   u_deform;      // 0=none 1=wave 2=flag 3=curl
uniform float u_d_amp;       // 幅度（wave/flag）或半径（curl）
uniform float u_d_freq;      // 频率
uniform float u_d_speed;     // 时间速度
uniform float u_d_width;     // 平面宽度（flag 用）

out vec3 v_world;
out vec3 v_normal;
out vec3 v_color;
out vec2 v_uv;
out float v_face_id;

void main() {
    vec3 local = a_pos;
    vec3 nrm   = a_normal;

    if (u_deform == 1) {
        // 波浪：Y 方向正弦
        float phase = u_d_freq * a_pos.x + u_time * u_d_speed;
        local.y += u_d_amp * sin(phase);
        float dy_dx = u_d_amp * u_d_freq * cos(phase);
        nrm = normalize(vec3(-dy_dx, 1.0, 0.0));
    } else if (u_deform == 2) {
        float t_free = clamp((a_pos.x + u_d_width * 0.5) / u_d_width, 0.0, 1.0);
        float damp = smoothstep(0.0, 1.0, t_free);
        float phase = u_d_freq * a_pos.x - u_time * u_d_speed;
        local.z += u_d_amp * damp * 2.20 * sin(phase);
        local.y += u_d_amp * damp * 0.90 * cos(phase);
        local.x += u_d_amp * damp * 0.40 * sin(phase + 1.5708);
        local.z += u_d_amp * damp * 0.18 * sin(phase * 3.0);
        float dz_dx = u_d_amp * damp * u_d_freq * 2.20 * cos(phase);
        float dy_dx = -u_d_amp * damp * u_d_freq * 0.90 * sin(phase);
        float dx_dx = u_d_amp * damp * u_d_freq * 0.40 * cos(phase + 1.5708);
        nrm = normalize(vec3(-dz_dx, dy_dx, 1.0 - dx_dx * 0.3));
    } else if (u_deform == 3) {
        // 卷曲：围绕 Y 轴弯成圆柱（半径 = u_d_amp）
        float R = max(u_d_amp, 0.01);
        float theta = a_pos.x / R;
        local.x = R * sin(theta);
        local.y = R * (1.0 - cos(theta)) + a_pos.y;
        nrm = normalize(vec3(sin(theta), cos(theta), 0.0));
    }

    if (u_instanced == 1) {
        mat4 M = mat4(a_inst0, a_inst1, a_inst2, a_inst3);
        mat3 M3 = mat3(M);
        if (u_flat_shading == 1) {
            v_normal = normalize(M3 * nrm);
        } else {
            M3[0] /= max(dot(M3[0], M3[0]), 1e-12);
            M3[1] /= max(dot(M3[1], M3[1]), 1e-12);
            M3[2] /= max(dot(M3[2], M3[2]), 1e-12);
            v_normal = normalize(M3 * nrm);
        }
        v_world = (M * vec4(local, 1.0)).xyz;
        gl_Position = u_vp * M * vec4(local, 1.0);
        v_color = (u_inst_use_color == 1)
                  ? a_color * a_inst_color.rgb
                  : a_color;
    } else {
        v_world = (u_model * vec4(local, 1.0)).xyz;
        gl_Position = u_mvp * vec4(local, 1.0);
        v_normal = u_normal_mat * nrm;
        v_color = a_color;
    }
    v_uv = a_uv;
    v_face_id = a_face_id;
}
""".encode("utf-8")

        # fragment shader —— 多光源 + 多纹理
        fs = """#version 300 es
precision mediump float;

in vec3 v_world;
in vec3 v_normal;
in vec3 v_color;
in vec2 v_uv;
in float v_face_id;

// 最多 6 张纹理
uniform int u_tex_count;
uniform sampler2D u_tex0;
uniform sampler2D u_tex1;
uniform sampler2D u_tex2;
uniform sampler2D u_tex3;
uniform sampler2D u_tex4;
uniform sampler2D u_tex5;

// 最多 8 个光源
uniform int u_light_count;
uniform int u_ltype[8];        // 0=ambient 1=dir 2=point 3=spot
uniform vec3 u_lpos[8];
uniform vec3 u_ldir[8];
uniform vec3 u_lcolor[8];
uniform float u_lintensity[8];
uniform float u_lrange[8];
uniform float u_lspot_cos[8];

out vec4 frag;

vec4 sample_tex(int fid) {
    if (u_tex_count == 0) return vec4(1.0);
    if (u_tex_count == 1) return texture(u_tex0, v_uv);
    if (fid == 0) return texture(u_tex0, v_uv);
    if (fid == 1) return texture(u_tex1, v_uv);
    if (fid == 2) return texture(u_tex2, v_uv);
    if (fid == 3) return texture(u_tex3, v_uv);
    if (fid == 4) return texture(u_tex4, v_uv);
    return texture(u_tex5, v_uv);
}

void main() {
    vec3 N = normalize(v_normal);
    int fid = int(v_face_id + 0.5);
    vec4 texel = sample_tex(fid);
    vec3 albedo = v_color;
    if (u_tex_count > 0) {
        vec3 tex_rgb = texel.rgb * mix(vec3(1.0), v_color * 2.0, 0.3);
        albedo = mix(v_color, tex_rgb, texel.a);
    }

    vec3 col = vec3(0.0);
    for (int i = 0; i < 8; i++) {
        if (i >= u_light_count) break;
        int t = u_ltype[i];
        vec3 L;
        float atten = 1.0;
        if (t == 0) {   // ambient
            col += albedo * u_lcolor[i] * u_lintensity[i];
            continue;
        } else if (t == 1) {   // directional
            L = normalize(-u_ldir[i]);
        } else {   // point / spot
            vec3 tl = u_lpos[i] - v_world;
            float d = length(tl);
            L = tl / max(d, 0.0001);
            float r = u_lrange[i];
            float u = clamp(1.0 - (d*d) / (r*r), 0.0, 1.0);
            atten = u * u;
            if (t == 3) {  // spot
                float ct = dot(-L, normalize(u_ldir[i]));
                if (ct < u_lspot_cos[i]) atten = 0.0;
                else {
                    float s = (ct - u_lspot_cos[i]) / (1.0 - u_lspot_cos[i]);
                    atten *= s * s;
                }
            }
        }
        float diff = max(dot(N, L), 0.0);
        col += albedo * u_lcolor[i] * u_lintensity[i] * diff * atten;
    }
    frag = vec4(col, 1.0);
}
""".encode("utf-8")

        v = self._compile(vs, GL_VERTEX_SHADER)
        f = self._compile(fs, GL_FRAGMENT_SHADER)
        p = g.glCreateProgram()
        g.glAttachShader(p, v); g.glAttachShader(p, f)
        g.glLinkProgram(p)
        st = c_int(); g.glGetProgramiv(p, GL_LINK_STATUS, ctypes.byref(st))
        if not st.value:
            log = ctypes.create_string_buffer(4096)
            g.glGetProgramInfoLog(p, 4096, None, log)
            raise RuntimeError(f"link: {log.value.decode()}")
        self._prog = p
        for name in ('u_mvp','u_model','u_vp','u_normal_mat','u_instanced','u_inst_use_color','u_flat_shading','u_tex_count',
                     'u_tex0','u_tex1','u_tex2','u_tex3','u_tex4','u_tex5',
                     'u_light_count','u_ltype','u_lpos','u_ldir','u_lcolor',
                     'u_lintensity','u_lrange','u_lspot_cos',
                     'u_time','u_deform','u_d_amp','u_d_freq','u_d_speed','u_d_width'):
            self._u[name] = g.glGetUniformLocation(p, name.encode())
        for name in ('a_pos','a_normal','a_color','a_uv','a_face_id'):
            self._a[name] = g.glGetAttribLocation(p, name.encode())

    def upload_mesh(self, name, verts):
        g = self._gles
        v = np.ascontiguousarray(verts, dtype=np.float32)
        if v.ndim == 1: v = v.reshape(-1, 12)
        if v.shape[1] != 12:
            raise ValueError(f"需要 12 floats/顶点，收到 {v.shape[1]}")
        n_verts = v.shape[0]
        vbo = c_uint(); g.glGenBuffers(1, ctypes.byref(vbo))
        g.glBindBuffer(GL_ARRAY_BUFFER, vbo.value)
        g.glBufferData(GL_ARRAY_BUFFER, v.nbytes, v.ctypes.data_as(c_void_p), GL_STATIC_DRAW)
        vao = c_uint(); g.glGenVertexArrays(1, ctypes.byref(vao))
        g.glBindVertexArray(vao.value)
        g.glBindBuffer(GL_ARRAY_BUFFER, vbo.value)
        STRIDE = 48
        for nm, sz, off in [('a_pos',3,0),('a_normal',3,12),('a_color',3,24),('a_uv',2,36),('a_face_id',1,44)]:
            loc = self._a.get(nm, -1)
            if loc >= 0:
                g.glEnableVertexAttribArray(loc)
                g.glVertexAttribPointer(loc, sz, GL_FLOAT, 0, STRIDE, c_void_p(off))
        self._meshes[name] = {'vao': vao.value, 'count': n_verts}
        return self._meshes[name]

    def upload_texture(self, name, rgba):
        arr = np.ascontiguousarray(rgba[::-1])
        H, W = arr.shape[:2]
        g = self._gles
        tex = c_uint(); g.glGenTextures(1, ctypes.byref(tex))
        g.glBindTexture(GL_TEXTURE_2D, tex.value)
        g.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                        arr.ctypes.data_as(c_void_p))
        for p in (GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER): g.glTexParameteri(GL_TEXTURE_2D, p, GL_LINEAR)
        for p in (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T): g.glTexParameteri(GL_TEXTURE_2D, p, GL_CLAMP_TO_EDGE)
        if name in self._textures:
            old = c_uint(self._textures[name]); g.glDeleteTextures(1, ctypes.byref(old))
        self._textures[name] = tex.value
        return tex.value

    def upload_texture_spec(self, name, spec):
        return self.upload_texture(name, load_texture_rgba(spec))

    def upload_texture_by_key(self, key, spec):
        """按 key 去重上传。key 相同 → 复用同一个 GL 纹理。

        key 由调用方提供（文件绝对路径 / data URI 内容等）。
        GL 纹理名用自增计数器，无需 hash。
        返回 tex_name（可直接传给 draw/draw_instanced）。
        """
        if key in self._tex_by_key:
            return self._tex_by_key[key]
        tex_name = 'kt_%d' % len(self._tex_by_key)
        if isinstance(spec, (bytes, bytearray)):
            from io import BytesIO as _B
            rgba = np.asarray(Image.open(_B(bytes(spec))).convert('RGBA'),
                               dtype=np.uint8)
            self.upload_texture(tex_name, rgba)
        else:
            self.upload_texture_spec(tex_name, spec)
        self._tex_by_key[key] = tex_name
        return tex_name

    def begin(self, clear_color=(0.06, 0.06, 0.10)):
        g = self._gles
        g.glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        g.glViewport(0, 0, self.W, self.H)
        g.glClearColor(*clear_color, 1.0)
        g.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        g.glUseProgram(self._prog)

    def set_lights(self, lights):
        """lights: list of dict {type, pos, dir, color, intensity, range, spot_cos}"""
        g = self._gles
        n = min(len(lights), MAX_LIGHTS)
        g.glUniform1i(self._u['u_light_count'], n)
        if n == 0: return
        types = (c_int * MAX_LIGHTS)(*([0] * MAX_LIGHTS))
        pos   = np.zeros((MAX_LIGHTS, 3), dtype=np.float32)
        dirs  = np.zeros((MAX_LIGHTS, 3), dtype=np.float32)
        cols  = np.zeros((MAX_LIGHTS, 3), dtype=np.float32)
        ints  = np.zeros(MAX_LIGHTS, dtype=np.float32)
        rngs  = np.ones(MAX_LIGHTS, dtype=np.float32)
        cosv  = np.zeros(MAX_LIGHTS, dtype=np.float32)
        for i, lt in enumerate(lights[:n]):
            types[i] = lt['type']
            pos[i] = lt.get('pos', (0, 0, 0))
            dirs[i] = lt.get('dir', (0, -1, 0))
            cols[i] = lt.get('color', (1, 1, 1))
            ints[i] = lt.get('intensity', 1.0)
            rngs[i] = max(0.01, lt.get('range', 20.0))
            cosv[i] = lt.get('spot_cos', 0.9)
        g.glUniform1iv(self._u['u_ltype'], MAX_LIGHTS, types)
        g.glUniform3fv(self._u['u_lpos'], MAX_LIGHTS, pos.ctypes.data_as(c_void_p))
        g.glUniform3fv(self._u['u_ldir'], MAX_LIGHTS, dirs.ctypes.data_as(c_void_p))
        g.glUniform3fv(self._u['u_lcolor'], MAX_LIGHTS, cols.ctypes.data_as(c_void_p))
        g.glUniform1fv(self._u['u_lintensity'], MAX_LIGHTS, ints.ctypes.data_as(c_void_p))
        g.glUniform1fv(self._u['u_lrange'], MAX_LIGHTS, rngs.ctypes.data_as(c_void_p))
        g.glUniform1fv(self._u['u_lspot_cos'], MAX_LIGHTS, cosv.ctypes.data_as(c_void_p))

    def set_shading(self, flat=False):
        """flat=True → 快速路径（贴纸感）；False → 正确 3D 光照"""
        if 'u_flat_shading' in self._u:
            self._gles.glUniform1i(self._u['u_flat_shading'],
                                    1 if flat else 0)

    def set_deform(self, deform=None, t=0.0, width=1.0):
        """deform: None 或 {'type':..., 'amp':..., 'freq':..., 'speed':...}"""
        g = self._gles
        if deform is None:
            g.glUniform1i(self._u['u_deform'], 0)
            return
        tmap = {'wave': 1, 'flag': 2, 'curl': 3}
        g.glUniform1i(self._u['u_deform'], tmap.get(deform['type'], 0))
        g.glUniform1f(self._u['u_time'], float(t))
        g.glUniform1f(self._u['u_d_amp'],   float(deform.get('amp', 0.3)))
        g.glUniform1f(self._u['u_d_freq'],  float(deform.get('freq', 2.0)))
        g.glUniform1f(self._u['u_d_speed'], float(deform.get('speed', 1.5)))
        g.glUniform1f(self._u['u_d_width'], float(width))

    def _bind_textures(self, textures):
        g = self._gles
        tex_list = [t for t in (textures or []) if t and t in self._textures][:MAX_TEXTURES]
        g.glUniform1i(self._u['u_tex_count'], len(tex_list))
        for i, tname in enumerate(tex_list):
            g.glActiveTexture(GL_TEXTURE0 + i)
            g.glBindTexture(GL_TEXTURE_2D, self._textures[tname])
            g.glUniform1i(self._u[f'u_tex{i}'], i)

    def upload_instance_transforms(self, name, matrices):
        """matrices: (N, 4, 4) 或 (N, 16) numpy float32（行主序 model matrix）
        为 VAO 追加 4 个 per-instance attribute（location 5-8）
        返回实例数 N。"""
        if name not in self._meshes:
            raise RuntimeError(f"mesh 不存在: {name}")
        m = self._meshes[name]
        M = np.asarray(matrices, dtype=np.float32).reshape(-1, 4, 4)
        # GLSL mat4(v0,v1,v2,v3) 里 vi 是列 → 需要 column-major
        Mt = np.ascontiguousarray(M.transpose(0, 2, 1).reshape(-1, 16))
        g = self._gles
        vbo = c_uint()
        g.glGenBuffers(1, ctypes.byref(vbo))
        g.glBindBuffer(GL_ARRAY_BUFFER, vbo.value)
        g.glBufferData(GL_ARRAY_BUFFER, Mt.nbytes,
                        Mt.ctypes.data_as(c_void_p), GL_STATIC_DRAW)
        g.glBindVertexArray(m['vao'])
        g.glBindBuffer(GL_ARRAY_BUFFER, vbo.value)
        STRIDE_I = 64  # 16 floats
        for k in range(4):
            loc = 5 + k
            g.glEnableVertexAttribArray(loc)
            g.glVertexAttribPointer(loc, 4, GL_FLOAT, 0, STRIDE_I,
                                     c_void_p(k * 16))
            g.glVertexAttribDivisor(loc, 1)
        g.glBindVertexArray(0)
        m['inst_vbo'] = vbo.value
        m['inst_count'] = M.shape[0]
        return M.shape[0]

    def upload_instance_colors(self, name, colors):
        """colors: (N, 3) 或 (N, 4) 浮点 RGB(A) in 0-1。
        为 VAO 追加 location 9 的 per-instance color attribute。
        调用后 draw_instanced 会用 a_inst_color 调制 v_color。
        未调用 → shader 走 a_color（顶点色）路径，不受影响。"""
        if name not in self._meshes:
            raise RuntimeError(f"mesh 不存在: {name}")
        m = self._meshes[name]
        C = np.asarray(colors, dtype=np.float32)
        if C.ndim == 1:
            C = C.reshape(-1, 3)
        if C.shape[1] == 3:
            C = np.concatenate([C, np.ones((C.shape[0], 1), np.float32)], axis=1)
        C = np.ascontiguousarray(C[:, :4])
        g = self._gles
        vbo = c_uint()
        g.glGenBuffers(1, ctypes.byref(vbo))
        g.glBindBuffer(GL_ARRAY_BUFFER, vbo.value)
        g.glBufferData(GL_ARRAY_BUFFER, C.nbytes,
                        C.ctypes.data_as(c_void_p), GL_STATIC_DRAW)
        g.glBindVertexArray(m['vao'])
        g.glBindBuffer(GL_ARRAY_BUFFER, vbo.value)
        g.glEnableVertexAttribArray(9)
        g.glVertexAttribPointer(9, 4, GL_FLOAT, 0, 16, c_void_p(0))
        g.glVertexAttribDivisor(9, 1)
        g.glBindVertexArray(0)
        m['inst_color_vbo'] = vbo.value
        m['inst_use_color'] = True
        return C.shape[0]

    def update_instance_transforms(self, name, matrices):
        """原地更新已有 instance matrix VBO 的内容（不重建 VBO/VAO）。
        适用于每帧变化的动态实例组。
        要求之前调过 upload_instance_transforms。"""
        if name not in self._meshes:
            return
        m = self._meshes[name]
        if 'inst_vbo' not in m:
            return
        M = np.asarray(matrices, dtype=np.float32).reshape(-1, 4, 4)
        Mt = np.ascontiguousarray(M.transpose(0, 2, 1).reshape(-1, 16))
        g = self._gles
        g.glBindBuffer(GL_ARRAY_BUFFER, m['inst_vbo'])
        # orphan + refill：避免 GPU 等待上一帧读完
        g.glBufferData(GL_ARRAY_BUFFER, Mt.nbytes,
                        Mt.ctypes.data_as(c_void_p), GL_DYNAMIC_DRAW)
        m['inst_count'] = M.shape[0]

    def draw_instanced(self, mesh_name, vp, count=None, textures=None):
        """一次 draw call 画 N 个实例。vp = projection @ view"""
        g = self._gles
        if mesh_name not in self._meshes:
            return
        m = self._meshes[mesh_name]
        if 'inst_count' not in m:
            return
        n = count if count is not None else m['inst_count']
        if n == 0:
            return
        vp_gl = np.ascontiguousarray(vp.T, dtype=np.float32)
        g.glUniformMatrix4fv(self._u['u_vp'], 1, 0,
                             vp_gl.ctypes.data_as(c_void_p))
        g.glUniform1i(self._u['u_instanced'], 1)
        g.glUniform1i(self._u['u_inst_use_color'],
                       1 if m.get('inst_use_color') else 0)
        self._bind_textures(textures)
        g.glBindVertexArray(m['vao'])
        g.glDrawArraysInstanced(GL_TRIANGLES, 0, m['count'], n)
        g.glUniform1i(self._u['u_instanced'], 0)

    def draw(self, mesh_name, mvp, model, textures=None):
        """textures: list of tex name，最多 6 个"""
        g = self._gles
        if mesh_name not in self._meshes: return
        m = self._meshes[mesh_name]
        mvp_gl = np.ascontiguousarray(mvp.T, dtype=np.float32)
        model_gl = np.ascontiguousarray(model.T, dtype=np.float32)
        g.glUniformMatrix4fv(self._u['u_mvp'], 1, 0, mvp_gl.ctypes.data_as(c_void_p))
        g.glUniformMatrix4fv(self._u['u_model'], 1, 0, model_gl.ctypes.data_as(c_void_p))
        nm = np.ascontiguousarray(normal_matrix(model).T, dtype=np.float32)
        g.glUniformMatrix3fv(self._u['u_normal_mat'], 1, 0, nm.ctypes.data_as(c_void_p))
        g.glUniform1i(self._u['u_instanced'], 0)
        self._bind_textures(textures)
        g.glBindVertexArray(m['vao'])
        g.glDrawArrays(GL_TRIANGLES, 0, m['count'])

    def end(self):
        g = self._gles
        g.glFinish()
        g.glPixelStorei(GL_PACK_ALIGNMENT, 1)
        g.glReadPixels(0, 0, self.W, self.H, GL_RGBA, GL_UNSIGNED_BYTE, self._pixel_buf)
        g.glBindFramebuffer(GL_FRAMEBUFFER, 0)
        arr = np.frombuffer(self._pixel_buf, dtype=np.uint8).reshape(self.H, self.W, 4)
        return arr[::-1]

    def close(self):
        try:
            if self._egl and self._display:
                self._egl.eglMakeCurrent(self._display, None, None, None)
                if self._context: self._egl.eglDestroyContext(self._display, self._context)
                if self._surface: self._egl.eglDestroySurface(self._display, self._surface)
                self._egl.eglTerminate(self._display)
        except Exception: pass


# ============================================================
# XML 解析
# ============================================================
def _vec3(s, default=(0, 0, 0)):
    if not s: return np.array(default, dtype=np.float32)
    p = s.split()
    return np.array([float(x) for x in p[:3]], dtype=np.float32)


def _color(s, default=(1, 1, 1)):
    if not s: return default
    s = s.lstrip('#')
    if len(s) >= 6:
        return (int(s[0:2], 16)/255, int(s[2:4], 16)/255, int(s[4:6], 16)/255)
    return default


def _parse_bool(s, default=False):
    """解析 '1'/'0'/'true'/'false'/'yes'/'no'/'on'/'off' 为 bool"""
    if s is None:
        return default
    return str(s).strip().lower() in ('1', 'true', 'yes', 'on')


def _ease(name, t):
    if not name or name == 'linear': return t
    if name in ('ease', 'ease-in-out'): return t * t * (3 - 2 * t)
    if name == 'ease-in': return t * t
    if name == 'ease-out': return 1 - (1 - t) ** 2
    if name == 'ease-out-cubic': return 1 - (1 - t) ** 3
    return t


def _parse_anim(elem):
    prop = elem.get('prop', '')
    ease = elem.get('ease', 'linear')
    t0 = float(elem.get('t0', 0))
    t1 = elem.get('t1') or elem.get('duration') or 1.0
    t1 = float(t1)
    kfs = []
    if elem.get('from') is not None and elem.get('to') is not None:
        for t, val, e in [(t0, elem.get('from'), ease), (t1, elem.get('to'), 'linear')]:
            parts = val.split()
            if len(parts) == 1: kfs.append((t, float(parts[0]), e))
            else: kfs.append((t, np.array([float(x) for x in parts[:3]], np.float32), e))
    for kf in elem.findall('keyframe'):
        t = float(kf.get('t', 0))
        val = kf.get('value', '0').split()
        if len(val) == 1: kfs.append((t, float(val[0]), kf.get('ease', 'linear')))
        else: kfs.append((t, np.array([float(x) for x in val[:3]], np.float32), kf.get('ease', 'linear')))
    kfs.sort(key=lambda x: x[0])
    return {'prop': prop, 'keyframes': kfs}


def _eval_anim(anim, t):
    kfs = anim['keyframes']
    if not kfs: return None
    if t <= kfs[0][0]: return kfs[0][1]
    if t >= kfs[-1][0]: return kfs[-1][1]
    for i in range(len(kfs) - 1):
        t0, v0, e0 = kfs[i]; t1, v1, _ = kfs[i + 1]
        if t0 <= t <= t1:
            u = (t - t0) / (t1 - t0) if t1 > t0 else 0
            u = _ease(e0, u)
            if isinstance(v0, np.ndarray): return v0 + (v1 - v0) * u
            return v0 + (v1 - v0) * u
    return kfs[-1][1]


def _generate_instances(instances_el):
    """<instances count seed mode .../> → [(pos(3), rot(3), scale(3)), ...]"""
    count = int(instances_el.get('count', '100'))
    seed_s = instances_el.get('seed')
    seed = int(seed_s) if seed_s else None
    rng = np.random.default_rng(seed)
    mode = instances_el.get('mode', 'random-box')
    spin = float(instances_el.get('spin', '0'))
    sj = float(instances_el.get('scale-jitter', '0'))

    if mode == 'random-box':
        parts = (instances_el.get('size') or '1 1 1').split()
        w = float(parts[0]) if len(parts) > 0 else 1.0
        h = float(parts[1]) if len(parts) > 1 else 1.0
        d = float(parts[2]) if len(parts) > 2 else 1.0
        pos = np.column_stack([
            rng.uniform(-w / 2, w / 2, count),
            rng.uniform(-h / 2, h / 2, count),
            rng.uniform(-d / 2, d / 2, count),
        ])
    elif mode == 'random-sphere':
        r = float(instances_el.get('radius', '1'))
        dirs = rng.normal(size=(count, 3))
        dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        rad = (rng.uniform(0, 1, count) ** (1.0 / 3.0)) * r
        pos = dirs * rad[:, None]
    elif mode == 'random-shell':
        r = float(instances_el.get('radius', '1'))
        th = float(instances_el.get('thickness', '0.1'))
        dirs = rng.normal(size=(count, 3))
        dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        rad = rng.uniform(r - th / 2, r + th / 2, count)
        pos = dirs * np.maximum(rad, 0)[:, None]
    else:
        pos = np.zeros((count, 3), dtype=np.float32)

    out = []
    for i in range(count):
        p = pos[i].astype(np.float32)
        if spin > 0:
            r = (rng.uniform(-math.pi, math.pi, 3) * spin).astype(np.float32)
        else:
            r = np.zeros(3, dtype=np.float32)
        if sj > 0:
            s = float(1.0 + rng.uniform(-sj, sj))
            sc = np.array([s, s, s], dtype=np.float32)
        else:
            sc = np.ones(3, dtype=np.float32)
        out.append((p, r, sc))
    return out


def _instances_to_matrices(instances):
    """[(pos, rot, scale), ...] → (N, 4, 4) row-major model matrix"""
    N = len(instances)
    out = np.empty((N, 4, 4), dtype=np.float32)
    for i, (p, r, s) in enumerate(instances):
        M = (m_translate(float(p[0]), float(p[1]), float(p[2]))
             @ m_rot_y(float(r[1]))
             @ m_rot_x(float(r[0]))
             @ m_rot_z(float(r[2]))
             @ m_scale(float(s[0]), float(s[1]), float(s[2])))
        out[i] = M
    return out


# ============================================================
# 参数化实例：$var 语法 + <param> 展开
# ============================================================
_VAR_RE = re.compile(r'\$([a-zA-Z_][a-zA-Z0-9_]*)')


_PRESSURE_NS = {
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'abs': abs, 'min': min, 'max': max, 'pow': pow, 'round': round,
    'sqrt': math.sqrt, 'exp': math.exp, 'log': math.log,
    'pi': math.pi, 'tau': math.tau, 'e': math.e,
    'floor': math.floor, 'ceil': math.ceil,
}


def _eval_pressure(expr, t):
    """求 pressure(t)。出错返回 1.0（不影响发射）。"""
    if not expr:
        return 1.0
    try:
        v = eval(expr, {'__builtins__': {}}, {**_PRESSURE_NS, 't': t})
        v = float(v)
        return max(0.0, min(1.0, v))
    except Exception:
        return 1.0


def _has_vars(s):
    """字符串是否含 $var 占位符"""
    return bool(s and _VAR_RE.search(s))


def _substitute_vars(s, params, idx):
    """用 params[name][idx] 替换 s 里的 $var。
    params: {name: (N, 3) ndarray}
    idx: 实例索引
    返回字符串（与 _vec3 兼容的格式）
    """
    def repl(m):
        name = m.group(1)
        if name not in params:
            raise KeyError(f"未声明的参数: ${name}")
        v = params[name][idx]
        return f"{float(v[0]):g} {float(v[1]):g} {float(v[2]):g}"
    return _VAR_RE.sub(repl, s)


def _gen_param_array(param_el, count, seed):
    """<param name="x" mode="..." .../> → (count, 3) float32 ndarray"""
    mode = (param_el.get('mode') or 'constant').lower()
    offset = _vec3(param_el.get('offset'), (0, 0, 0)).astype(np.float32)
    if mode == 'constant':
        val = _vec3(param_el.get('value'), (0, 0, 0))
        return (np.tile(val.astype(np.float32), (count, 1)) + offset[None, :])
    rng = np.random.default_rng(seed)
    if mode == 'random-box':
        parts = (param_el.get('size') or '1 1 1').split()
        w = float(parts[0]) if len(parts) > 0 else 1.0
        h = float(parts[1]) if len(parts) > 1 else 1.0
        d = float(parts[2]) if len(parts) > 2 else 1.0
        pos = np.column_stack([
            rng.uniform(-w / 2, w / 2, count),
            rng.uniform(-h / 2, h / 2, count),
            rng.uniform(-d / 2, d / 2, count),
        ])
        return (pos + offset[None, :]).astype(np.float32)
    if mode == 'random-sphere':
        r = float(param_el.get('radius', '1'))
        dirs = rng.normal(size=(count, 3))
        dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        rad = (rng.uniform(0, 1, count) ** (1.0 / 3.0)) * r
        return (dirs * rad[:, None] + offset[None, :]).astype(np.float32)
    if mode == 'random-shell':
        r = float(param_el.get('radius', '1'))
        th = float(param_el.get('thickness', '0.1'))
        dirs = rng.normal(size=(count, 3))
        dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        rad = rng.uniform(r - th / 2, r + th / 2, count)
        return (dirs * np.maximum(rad, 0)[:, None] + offset[None, :]).astype(np.float32)
    if mode == 'fib-sphere':
        r = float(param_el.get('radius', '1'))
        offset = _vec3(param_el.get('offset'), (0, 0, 0)).astype(np.float32)
        phi = math.pi * (3 - math.sqrt(5))
        i = np.arange(count)
        y = 1 - (i / max(1, count - 1)) * 2
        rr = np.sqrt(np.maximum(0, 1 - y * y))
        theta = phi * i
        pts = np.column_stack([np.cos(theta) * rr, y, np.sin(theta) * rr])
        return (pts * r + offset).astype(np.float32)
    if mode == 'palette':
        # 颜色调色板：按顺序循环分配给每个实例
        # 返回 (N, 3) 的 RGB 值（0-1）
        colors_str = param_el.get('colors', '#ffffff')
        cs = [c.strip() for c in colors_str.split(',') if c.strip()]
        parsed = []
        for c in cs:
            c = c.lstrip('#')
            if len(c) >= 6:
                parsed.append([int(c[0:2], 16) / 255.0,
                               int(c[2:4], 16) / 255.0,
                               int(c[4:6], 16) / 255.0])
            elif len(c) == 3:
                parsed.append([int(c[0] * 2, 16) / 255.0,
                               int(c[1] * 2, 16) / 255.0,
                               int(c[2] * 2, 16) / 255.0])
            else:
                parsed.append([1.0, 1.0, 1.0])
        if not parsed:
            parsed = [[1.0, 1.0, 1.0]]
        arr = np.empty((count, 3), dtype=np.float32)
        for i in range(count):
            arr[i] = parsed[i % len(parsed)]
        return arr
    print(f"  ⚠ 未知 param mode: {mode}，按 constant 处理", file=sys.stderr)
    return np.zeros((count, 3), dtype=np.float32)


def _expand_param_instances(instances_el):
    """<instances count=N seed=S time-jitter=J>
         <param name=... mode=... />
       </instances>
    → {'count': N, 'params': {name: (N,3)}, 'jitter': (N,) or None}
    """
    count = int(instances_el.get('count', '100'))
    seed_s = instances_el.get('seed')
    base_seed = int(seed_s) if seed_s else None
    jitter_amp = float(instances_el.get('time-jitter', '0'))

    params = {}
    colors = None
    for p in instances_el.findall('param'):
        name = p.get('name')
        if not name:
            print("  ⚠ <param> 缺 name，跳过", file=sys.stderr)
            continue
        pseed = base_seed
        ps = p.get('seed')
        if ps:
            pseed = int(ps)
        if pseed is not None:
            pseed = pseed + (hash(name) & 0xFFFF)
        mode = (p.get('mode') or 'constant').lower()
        if mode == 'palette':
            colors = _gen_param_array(p, count, pseed)
            continue
        params[name] = _gen_param_array(p, count, pseed)

    jitter = None
    if jitter_amp > 0:
        rng = np.random.default_rng(base_seed)
        jitter = rng.uniform(-jitter_amp / 2, jitter_amp / 2, count).astype(np.float32)

    return {'count': count, 'params': params, 'colors': colors, 'jitter': jitter}

# ============================================================
# 参数化实例：批量关键帧插值 + 批量 model matrix
# ============================================================
def _expand_kf_value(valstr, params, count):
    """把单个 keyframe value 展开成 (count, 3) float32 数组。

    支持三种形式：
      · 纯 $var          → 直接返回 params[name]
      · 常量 "1 2 3"     → broadcast
      · 混合 "$v 0 1"    → 逐个替换（慢路径，但只在加载时跑一次）
    """
    s = valstr.strip()
    # 快速路径 1：纯 $var
    m = _VAR_RE.fullmatch(s)
    if m:
        name = m.group(1)
        if name not in params:
            raise KeyError(f"未声明参数: ${name}")
        return params[name].astype(np.float32)

    # 快速路径 2：无 $var，纯常量
    if not _has_vars(s):
        parts = s.split()
        if len(parts) >= 3:
            v = np.array([float(parts[0]), float(parts[1]), float(parts[2])],
                          dtype=np.float32)
        elif len(parts) == 1:
            v = np.array([float(parts[0])] * 3, dtype=np.float32)
        else:
            v = np.zeros(3, dtype=np.float32)
        return np.tile(v, (count, 1))

    # 慢路径：混合
    out = np.empty((count, 3), dtype=np.float32)
    for i in range(count):
        expanded = _substitute_vars(s, params, i)
        parts = expanded.split()
        out[i, 0] = float(parts[0]) if len(parts) > 0 else 0.0
        out[i, 1] = float(parts[1]) if len(parts) > 1 else 0.0
        out[i, 2] = float(parts[2]) if len(parts) > 2 else 0.0
    return out


def _expand_projectile_instances(animate_el, instances_el):
    """<animate prop="projectile" gravity="7.8" life="1.1">
         <param name="origin" mode="..." .../>
         <param name="v0"     mode="..." .../>
         <param name="color"  mode="palette" .../>
       </animate>
       <instances count="N" seed="S" time-jitter="J"/>

    返回物理参数 dict：
      {'gravity', 'life', 'origin': (N,3), 'v0': (N,3),
       'count', 'colors', 'jitter'}

    数学：
      pos(t) = origin + v0·t − ½·g·t²·ŷ
    每颗实例循环播放，周期 = life。
    """
    count = int(instances_el.get('count', '100')) if instances_el is not None else 100
    seed_s = instances_el.get('seed') if instances_el is not None else None
    base_seed = int(seed_s) if seed_s else None
    jitter_amp = float(instances_el.get('time-jitter', '0')) if instances_el is not None else 0

    g = float(animate_el.get('gravity', '9.8'))
    life = float(animate_el.get('life', '1.0'))
    pressure_expr = animate_el.get('pressure', '').strip() or None

    origin = v0 = None
    colors = None
    for p in animate_el.findall('param'):
        name = p.get('name')
        if not name:
            continue
        pseed = base_seed
        ps = p.get('seed')
        if ps:
            pseed = int(ps)
        if pseed is not None:
            pseed = pseed + (hash(name) & 0xFFFF)
        mode = (p.get('mode') or 'constant').lower()
        if mode == 'palette':
            colors = _gen_param_array(p, count, pseed)
            continue
        arr = _gen_param_array(p, count, pseed)
        if name == 'origin':
            origin = arr
        elif name == 'v0':
            v0 = arr

    if origin is None:
        origin = np.zeros((count, 3), dtype=np.float32)
    if v0 is None:
        v0 = np.zeros((count, 3), dtype=np.float32)

    jitter = None
    if jitter_amp > 0:
        rng = np.random.default_rng(base_seed)
        jitter = rng.uniform(0, jitter_amp, count).astype(np.float32)

    return {
        'gravity': g,
        'life': max(life, 1e-3),
        'pressure': pressure_expr,
        'origin': origin,
        'v0': v0,
        'count': count,
        'colors': colors,
        'jitter': jitter,
    }

def _precompute_param_kf_arrays(mesh):
    """把 mesh 的 _param_anims 展开成 numpy 数组。

    为每个 prop 存：
      mesh['_param_kf'][prop] = {
        'times':  (K,) float32
        'eases':  list of K-1 ease 名
        'values': (K, N, 3) float32
      }
    """
    inst = mesh.get('_param_instances')
    anims = mesh.get('_param_anims')
    if not inst or not anims:
        return
    count = inst['count']
    params = inst['params']

    kf_map = {}
    for anim in anims:
        prop = anim['prop']
        kfs = anim['keyframes']
        if len(kfs) < 1:
            continue
        times = np.array([k[0] for k in kfs], dtype=np.float32)
        eases = [k[1] for k in kfs[:-1]]
        values = np.empty((len(kfs), count, 3), dtype=np.float32)
        for ki, (kt, kease, valstr, _vars) in enumerate(kfs):
            values[ki] = _expand_kf_value(valstr, params, count)
        kf_map[prop] = {'times': times, 'eases': eases, 'values': values}

    mesh['_param_kf'] = kf_map


def _ease_np(name, x):
    """numpy 版 ease_apply。x 是 (N,) 数组，返回 (N,) 数组。"""
    if not name or name == 'linear' or name == 'none':
        return x
    if name in ('ease', 'ease-in-out', 'smoothstep'):
        return x * x * (3 - 2 * x)
    if name in ('ease-in', 'ease-in-quad'):
        return x * x
    if name in ('ease-out', 'ease-out-quad'):
        return 1 - (1 - x) ** 2
    if name == 'ease-in-cubic':
        return x ** 3
    if name == 'ease-out-cubic':
        return 1 - (1 - x) ** 3
    if name == 'ease-in-out-cubic':
        y = np.where(x < 0.5,
                     4 * x ** 3,
                     1 - (-2 * x + 2) ** 3 / 2)
        return y
    if name == 'ease-out-expo':
        return np.where(x >= 1, 1.0, 1 - 2 ** (-10 * x))
    if name == 'back-out':
        c1, c3 = 1.70158, 2.70158
        return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
    if name == 'back-in':
        c1, c3 = 1.70158, 2.70158
        return c3 * x ** 3 - c1 * x ** 2
    return x


def batch_param_at(mesh, t, extra_offset=None):
    """批量算出参数化 mesh 在时刻 t 的所有实例的 pos/rot/scale。

    extra_offset: (N,) 或 None —— 额外的 per-instance time offset
                  叠加在 jitter 之上

    projectile 物理模式（<animate prop="projectile">）：
      走抛物线公式，不用关键帧。
      pos(t) = origin + v0·t − ½·g·t²·ŷ
      y < floor 时缩放到 0（"落地消失"）

    返回 {'pos': (N,3), 'rot': (N,3), 'scale': (N,3)}
        projectile 模式额外带 'alive': (N,) bool
    """
    proj = mesh.get('_projectile')
    if proj is not None:
        return _batch_projectile_at(mesh, proj, t, extra_offset)

    inst = mesh.get('_param_instances')
    kf_map = mesh.get('_param_kf')
    if not inst or not kf_map:
        return None
    count = inst['count']

    # 时间轴（含 jitter + extra_offset）
    t_eff = np.full(count, t, dtype=np.float32)
    if inst.get('jitter') is not None:
        t_eff = t_eff + inst['jitter']
    if extra_offset is not None:
        t_eff = t_eff + extra_offset

    idx_all = np.arange(count)
    out = {}

    for prop, data in kf_map.items():
        times = data['times']
        eases = data['eases']
        values = data['values']  # (K, N, 3)
        K = len(times)

        if K == 1:
            out[prop] = values[0].copy()
            continue

        # searchsorted 找 t_eff 落在哪个区间
        idx = np.searchsorted(times, t_eff, side='right') - 1
        idx = np.clip(idx, 0, K - 2)

        t0 = times[idx]
        t1 = times[idx + 1]
        dt = np.maximum(t1 - t0, 1e-9)
        alpha = (t_eff - t0) / dt
        alpha = np.clip(alpha, 0.0, 1.0)

        # 每个实例的 ease 可能不同（因为落在不同段），按段分桶
        v0 = values[idx, idx_all]      # (N, 3)
        v1 = values[idx + 1, idx_all]  # (N, 3)

        # 如果所有 ease 都是 linear，直接向量化
        if all(e in (None, '', 'linear', 'none') for e in eases):
            a = alpha
        else:
            # 按段号分桶，每组用对应的 ease
            a = alpha.copy()
            for k in range(K - 1):
                mask = (idx == k)
                if not mask.any():
                    continue
                e = eases[k] if k < len(eases) else 'linear'
                a[mask] = _ease_np(e, alpha[mask])

        out[prop] = v0 + (v1 - v0) * a[:, None]

    # 补全缺省的 rot/scale（用 mesh 的静态值 broadcast）
    count_full = inst['count']
    if 'pos' not in out:
        out['pos'] = np.tile(mesh['pos'].astype(np.float32), (count_full, 1))
    if 'rot' not in out:
        out['rot'] = np.tile(mesh['rotate'].astype(np.float32), (count_full, 1))
    if 'scale' not in out:
        out['scale'] = np.tile(mesh['scale'].astype(np.float32), (count_full, 1))

    return out


def _batch_projectile_at(mesh, proj, t, extra_offset=None):
    """抛物线批量计算。

    循环周期 = life。每颗实例的相位由 jitter 决定。
    y < floor → scale=0（不可见），同时置 alive=False。
    """
    count = proj['count']
    g = proj['gravity']
    life = proj['life']
    origin = proj['origin']
    v0_base = proj['v0']

    # 时间轴（含 jitter + extra_offset）
    t_eff = np.full(count, t, dtype=np.float32)
    if proj.get('jitter') is not None:
        t_eff = t_eff + proj['jitter']
    if extra_offset is not None:
        t_eff = t_eff + extra_offset

    # 周期化
    t_mod = np.mod(t_eff, life).astype(np.float32)

    # 水压（随时间变化的发射强度）
    p = _eval_pressure(proj.get('pressure'), float(t))
    v0 = v0_base * p

    # 抛物线
    x = origin[:, 0] + v0[:, 0] * t_mod
    y = origin[:, 1] + v0[:, 1] * t_mod - 0.5 * g * t_mod * t_mod
    z = origin[:, 2] + v0[:, 2] * t_mod

    pos = np.stack([x, y, z], axis=1).astype(np.float32)

    # 落地判定（floor 用 0.05，跟桌面齐平）
    floor = 0.05
    alive = y > floor

    base_scale = np.tile(mesh['scale'].astype(np.float32), (count, 1))
    scale = base_scale * alive[:, None].astype(np.float32)

    return {
        'pos': pos,
        'rot': np.tile(mesh['rotate'].astype(np.float32), (count, 1)),
        'scale': scale,
        'alive': alive,
    }

def batch_param_aabb(pos, rot, scl, local_half):
    """批量算 (N,) AABB (centers, halfs)。

    local_half: (3,) 局部半长（mesh 自身的 AABB 半长）
    返回：centers (N, 3), halfs (N, 3)
    与 mesh_world_aabb 的语义一致：用 |R| 做保守上界
    """
    N = len(pos)
    cx, sx = np.cos(rot[:, 0]), np.sin(rot[:, 0])
    cy, sy = np.cos(rot[:, 1]), np.sin(rot[:, 1])
    cz, sz = np.cos(rot[:, 2]), np.sin(rot[:, 2])

    R_abs = np.empty((N, 3, 3), dtype=np.float32)
    R_abs[:, 0, 0] = np.abs(cy * cz + sy * sx * sz)
    R_abs[:, 0, 1] = np.abs(-cy * sz + sy * sx * cz)
    R_abs[:, 0, 2] = np.abs(sy * cx)
    R_abs[:, 1, 0] = np.abs(cx * sz)
    R_abs[:, 1, 1] = np.abs(cx * cz)
    R_abs[:, 1, 2] = np.abs(sx)
    R_abs[:, 2, 0] = np.abs(-sy * cz + cy * sx * sz)
    R_abs[:, 2, 1] = np.abs(sy * sz + cy * sx * cz)
    R_abs[:, 2, 2] = np.abs(cy * cx)

    lh = np.abs(scl) * local_half[None, :]  # (N, 3)
    wh = (R_abs @ lh[:, :, None]).squeeze(-1)  # (N, 3)
    return pos, wh

def batch_model_matrices(pos, rot, scl):
    """批量算 (N, 4, 4) model matrix。

    与现有 _mesh_at 后手动乘的一致：
      M = T(pos) @ Ry(rot[1]) @ Rx(rot[0]) @ Rz(rot[2]) @ S(scl)
    """
    N = len(pos)
    cx, sx = np.cos(rot[:, 0]), np.sin(rot[:, 0])
    cy, sy = np.cos(rot[:, 1]), np.sin(rot[:, 1])
    cz, sz = np.cos(rot[:, 2]), np.sin(rot[:, 2])

    # R = Ry @ Rx @ Rz 展开
    R = np.empty((N, 3, 3), dtype=np.float32)
    R[:, 0, 0] = cy * cz + sy * sx * sz
    R[:, 0, 1] = -cy * sz + sy * sx * cz
    R[:, 0, 2] = sy * cx
    R[:, 1, 0] = cx * sz
    R[:, 1, 1] = cx * cz
    R[:, 1, 2] = -sx
    R[:, 2, 0] = -sy * cz + cy * sx * sz
    R[:, 2, 1] = sy * sz + cy * sx * cz
    R[:, 2, 2] = cy * cx

    # 应用缩放
    RS = R * scl[:, None, :]  # 每一列乘 scale 的分量

    # 组装 4x4
    M = np.zeros((N, 4, 4), dtype=np.float32)
    M[:, :3, :3] = RS
    M[:, :3, 3] = pos
    M[:, 3, 3] = 1.0
    return M

def parse_scene_xml(xml_text, base_dir='.'):
    root = ET.fromstring(xml_text)
    scene = {
        'duration': float(root.get('duration', 5)),
        'fps': int(root.get('fps', 30)),
        'width': int(root.get('width', 1280)),
        'height': int(root.get('height', 720)),
        'bg': _color(root.get('bg', '#0f0f18'), (0.06, 0.06, 0.10)),
        'shading': root.get('shading', 'lit').lower(),
        'fastpath': _parse_bool(root.get('fastpath'), False),
        'base_dir': Path(base_dir),
        'camera': {
            'pos': np.array([0, 1.0, 5], dtype=np.float32),
            'lookAt': np.array([0, 0, 0], dtype=np.float32),
            'fov': 50.0,
            'anims': [],
        },
        'lights': [],
        'meshes': [],
    }

    cam = root.find('camera')
    if cam is not None:
        if cam.get('pos'): scene['camera']['pos'] = _vec3(cam.get('pos'))
        if cam.get('lookAt'): scene['camera']['lookAt'] = _vec3(cam.get('lookAt'))
        scene['camera']['fov'] = float(cam.get('fov', 50))
        for a in cam.findall('animate'): scene['camera']['anims'].append(_parse_anim(a))

    for lt in root.findall('light'):
        tname = lt.get('type', 'directional')
        tmap = {'directional': 1, 'ambient': 0, 'point': 2, 'spot': 3}
        entry = {
            'type': tmap.get(tname, 1),
            'dir': _vec3(lt.get('dir'), (0.5, 1.0, 0.3)),
            'pos': _vec3(lt.get('pos'), (0, 5, 0)),
            'color': _color(lt.get('color', '#ffffff')),
            'intensity': float(lt.get('intensity', 1.0)),
            'range': float(lt.get('range', 20.0)),
        }
        if tname == 'spot':
            angle = float(lt.get('angle', 25.0))  # 度
            entry['spot_cos'] = math.cos(math.radians(angle))
        scene['lights'].append(entry)

    for m in root.findall('mesh'):
        # 检测 projectile 模式（<animate prop="projectile">）
        _proj_el = None
        for _a in m.findall('animate'):
            if _a.get('prop') == 'projectile':
                _proj_el = _a
                break

        # 检测参数化模式（keyframe value 里有 $var）
        _is_param = False
        for _a in m.findall('animate'):
            for _kf in _a.findall('keyframe'):
                if _has_vars(_kf.get('value', '')):
                    _is_param = True
                    break
            if _is_param:
                break

        entry = {
            'id': m.get('id', f'mesh{len(scene["meshes"])}'),
            'type': m.get('type', 'cube'),
            'size': m.get('size', '1'),
            'pos': _vec3(m.get('pos')),
            'rotate': _vec3(m.get('rotate')),
            'scale': _vec3(m.get('scale'), (1, 1, 1)),
            'color': _color(m.get('color', '#cccccc')),
            'texture': m.get('texture'),
        }
        if _proj_el is not None:
            # projectile 物理模式
            entry['anims'] = []
            _pinst = m.find('instances')
            if _pinst is None:
                print(f"  ⚠ mesh {entry['id']}: projectile 缺 <instances>",
                      file=sys.stderr)
                entry['_param_instances'] = None
            else:
                _pdata = _expand_projectile_instances(_proj_el, _pinst)
                entry['_projectile'] = _pdata
                # 兼容：也建 _param_instances（setup 那边靠它上传几何体）
                entry['_param_instances'] = {
                    'count': _pdata['count'],
                    'params': {},
                    'colors': _pdata['colors'],
                    'jitter': _pdata['jitter'],
                }
                entry['_param_kf'] = {}   # 物理模式不用关键帧
        elif _is_param:
            # 参数化 mesh：anims 用专用解析，不填老的 anims
            entry['anims'] = []
            entry['_param_anims'] = [_parse_anim_param(a)
                                      for a in m.findall('animate')]
        else:
            entry['anims'] = [_parse_anim(a) for a in m.findall('animate')]
        if m.get('height'): entry['_height'] = float(m.get('height'))
        if m.get('seg'):    entry['_seg']    = int(m.get('seg'))

        # deform 属性
        dk = m.get('deform')
        if dk:
            entry['_deform'] = {
                'type': dk,
                'amp':   float(m.get('amp',   '0.3')),
                'freq':  float(m.get('freq',  '2.0')),
                'speed': float(m.get('speed', '1.5')),
            }
        # 单独指定 radius 作为 amp（curl 专用）
        if m.get('radius') and entry.get('_deform'):
            entry['_deform']['amp'] = float(m.get('radius'))

        # 🆕 参数化实例检测：keyframe value 里是否有 $var
        _has_param_vars = False
        for _a in m.findall('animate'):
            for _kf in _a.findall('keyframe'):
                if _has_vars(_kf.get('value', '')):
                    _has_param_vars = True
                    break
            if _has_param_vars:
                break

        instances_el = m.find('instances')
        inst_list = m.findall('instance')

        if _proj_el is not None:
            pass   # projectile 已在上面处理
        elif _has_param_vars:
            if instances_el is None:
                print(f"  ⚠ mesh {entry['id']}: 用了 $var 但缺 <instances>",
                      file=sys.stderr)
            else:
                params_el = instances_el.findall('param')
                if not params_el:
                    print(f"  ⚠ mesh {entry['id']}: <instances> 无 <param>",
                          file=sys.stderr)
                else:
                    entry['_param_instances'] = _expand_param_instances(instances_el)
                    try:
                        _precompute_param_kf_arrays(entry)
                    except Exception as _e:
                        print(f"  ⚠ mesh {entry['id']} 参数化预计算失败: {_e}",
                              file=sys.stderr)
        elif instances_el is not None:
            entry['_instances'] = _generate_instances(instances_el)
        elif inst_list:
            entry['_instances'] = [
                (_vec3(it.get('pos')),
                 _vec3(it.get('rotate')),
                 _vec3(it.get('scale'), (1, 1, 1)))
                for it in inst_list
            ]

        scene['meshes'].append(entry)

    return scene


def _parse_anim_param(anim_elem):
    """参数化版本的 animate 解析。

    返回：{'prop': 'pos', 'keyframes': [(t, ease, vars_dict), ...]}

    vars_dict: {'start': True, 'end': True} —— 只记录 keyframe 用了哪些 $var
               具体数值在渲染时按实例 index 替换。
    """
    prop = anim_elem.get('prop', '')
    kfs = []
    for kf in anim_elem.findall('keyframe'):
        t = float(kf.get('t', 0))
        ease = kf.get('ease', 'linear')
        val = kf.get('value', '')
        # 用正则找出所有 $var 名
        used_vars = set()
        for m in _VAR_RE.finditer(val):
            used_vars.add(m.group(1))
        kfs.append((t, ease, val, used_vars))
    kfs.sort(key=lambda x: x[0])
    return {'prop': prop, 'keyframes': kfs}

def _mesh_at(mesh, t):
    pos = mesh['pos'].copy(); rot = mesh['rotate'].copy(); scl = mesh['scale'].copy()
    for a in mesh['anims']:
        v = _eval_anim(a, t)
        if v is None: continue
        p = a['prop']
        if p == 'pos' and isinstance(v, np.ndarray): pos = v
        elif p == 'rotate' and isinstance(v, np.ndarray): rot = v
        elif p == 'scale' and isinstance(v, np.ndarray): scl = v
    return pos, rot, scl


def _camera_at(cam, t):
    pos = cam['pos'].copy(); look = cam['lookAt'].copy()
    for a in cam['anims']:
        v = _eval_anim(a, t)
        if v is None: continue
        p = a['prop']
        if p == 'pos' and isinstance(v, np.ndarray): pos = v
        elif p == 'lookAt' and isinstance(v, np.ndarray): look = v
    return pos, look


# ============================================================
# 渲染
# ============================================================
def _resolve_texture_specs(spec, base_dir):
    """把 texture="..." 拆解成 [(key, spec), ...]。

    key 用自然标识（不做内容 hash）：
      · 文件路径 → 绝对路径字符串
      · data URI / 内联 SVG → 整段内容本身（Python dict 按内容去重）

    videoeditor._prepare_space3d_xml 会为每个 asset 复用同一份 data URI
    字符串，所以同 asset 引用的 mesh 会拿到完全相同的 key。
    """
    spec = spec.strip()
    if not spec:
        return []
    if '|' in spec:
        parts = [s.strip() for s in spec.split('|') if s.strip()]
    else:
        parts = [spec]

    out = []
    for s in parts:
        if s.startswith('data:') or s.startswith('<svg'):
            out.append((s, s))
        else:
            p = Path(s)
            if not p.is_absolute():
                p = Path(base_dir) / s
            abs_p = str(p)
            out.append((abs_p, abs_p))
    return out


# ============================================================
# 自动实例化：分组 + 相对缩放矩阵
# ============================================================
def _parse_size_vec(size_str):
    """'0.3 0.8 0.4' → [0.3, 0.8, 0.4]；'0.5' → [0.5, 0.5, 0.5]"""
    parts = (size_str or '1').split()
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            vals.append(1.0)
    if not vals:
        vals = [1.0]
    if len(vals) == 1:
        return np.array([vals[0]] * 3, dtype=np.float32)
    while len(vals) < 3:
        vals.append(vals[-1])
    return np.array(vals[:3], dtype=np.float32)


_INSTANCABLE_TYPES = frozenset(['cube', 'sphere', 'cylinder', 'cone', 'prism'])


def group_instancing_candidates(meshes, threshold=None):
    """把大量同类型的静态 mesh 分组，为一次 draw call 做准备。

    排除条件：
      · 有 anims（动态必须独立）
      · 有 _instances（已经是 instanced）
      · 有 _deform（顶点变形需要独立 set_deform）
      · type 不在 _INSTANCABLE_TYPES

    分组键：(type, tuple(tex_names), color_tuple)
      · cylinder/cone/prism 额外带 height（Y 方向不缩放）

    返回：[(代表 idx, [成员 idx, ...]), ...]，仅返回成员数 >= threshold 的组
    """
    if threshold is None:
        threshold = INSTANCE_THRESHOLD
    if threshold <= 1:
        return []
    groups = {}
    for i, m in enumerate(meshes):
        if m.get('anims') or m.get('_instances') or m.get('_deform'):
            continue
        typ = m.get('type', 'cube')
        if typ not in _INSTANCABLE_TYPES:
            continue
        tex = tuple(m.get('_tex_names', []))
        try:
            col = tuple(float(c) for c in m.get('color', (1, 1, 1)))
        except (TypeError, ValueError):
            col = (1.0, 1.0, 1.0)
        if typ in ('cylinder', 'cone', 'prism'):
            key = (typ, tex, col, str(m.get('_height', '')))
        else:
            key = (typ, tex, col)
        groups.setdefault(key, []).append(i)

    out = []
    for idxs in groups.values():
        if len(idxs) >= threshold:
            out.append((idxs[0], idxs))
    return out


def build_relative_instance_matrices(meshes, idxs, ref_idx):
    """实例矩阵 = 自身 world model × 相对缩放（自身 size / 代表 size）。

    这样代表几何体的形状能覆盖所有成员：
      · cube: 3D size 直接参与，rel = size / ref_size
      · sphere: size 是半径，3 轴同步
      · cylinder/cone/prism: Y 方向不缩放（同组 height 已保证相同）
    """
    ref = meshes[ref_idx]
    ref_size = _parse_size_vec(ref.get('size', '1'))
    ref_size_safe = np.where(np.abs(ref_size) > 1e-6, ref_size, 1.0)
    typ = ref.get('type', 'cube')

    N = len(idxs)
    mats = np.empty((N, 4, 4), dtype=np.float32)
    for k, i in enumerate(idxs):
        m = meshes[i]
        pos, rot, scl = _mesh_at(m, 0.0)
        sz = _parse_size_vec(m.get('size', '1'))
        rel = sz / ref_size_safe
        if typ in ('cylinder', 'cone', 'prism'):
            rel[1] = 1.0
        M = (m_translate(float(pos[0]), float(pos[1]), float(pos[2]))
             @ m_rot_y(float(rot[1]))
             @ m_rot_x(float(rot[0]))
             @ m_rot_z(float(rot[2]))
             @ m_scale(float(scl[0]) * rel[0],
                       float(scl[1]) * rel[1],
                       float(scl[2]) * rel[2]))
        mats[k] = M
    return mats


def _anim_structure_key(mesh):
    """把 anims 结构编码成可 hash 的 key（不含具体数值）。

    只包含 prop / keyframe 时间戳 / ease —— 这些必须完全一致，
    同一组才值得批量计算（未来可向量化 interp，且视觉行为一致）。
    """
    parts = []
    for a in mesh.get('anims', []):
        prop = a.get('prop', '')
        kfs = tuple((round(t, 4), e) for t, _, e in a.get('keyframes', []))
        parts.append((prop, kfs))
    return tuple(parts)


def find_dynamic_groups(meshes, threshold=None):
    """找到可批量渲染的动态 mesh 组。

    入组条件（全部满足）：
      · 有 anims（静态走 group_instancing_candidates）
      · 无 _instances（已是 instanced）
      · 无 _deform（顶点变形需要每帧独立 set_deform）
      · type 在 _INSTANCABLE_TYPES
      · 同组内 anims 结构完全一致（prop / keyframe t / ease 相同）
      · 分组键不含 color：颜色通过 per-instance attribute 传

    返回：[(代表 idx, [成员 idx, ...]), ...]
    """
    if threshold is None:
        threshold = INSTANCE_THRESHOLD
    if threshold <= 1:
        return []
    groups = {}
    for i, m in enumerate(meshes):
        if not m.get('anims'): continue
        if m.get('_instances') or m.get('_deform'): continue
        typ = m.get('type', 'cube')
        if typ not in _INSTANCABLE_TYPES: continue
        struct = _anim_structure_key(m)
        tex = tuple(m.get('_tex_names', []))
        if typ in ('cylinder', 'cone', 'prism'):
            key = (typ, tex, struct, str(m.get('_height', '')))
        else:
            key = (typ, tex, struct)
        groups.setdefault(key, []).append(i)
    return [(idxs[0], idxs) for idxs in groups.values() if len(idxs) >= threshold]


def build_dynamic_instance_matrices(meshes, idxs, t):
    """为动态实例组构建 per-instance model matrix（时刻 t）"""
    N = len(idxs)
    mats = np.empty((N, 4, 4), dtype=np.float32)
    for k, i in enumerate(idxs):
        m = meshes[i]
        pos, rot, scl = _mesh_at(m, t)
        M = (m_translate(float(pos[0]), float(pos[1]), float(pos[2]))
             @ m_rot_y(float(rot[1]))
             @ m_rot_x(float(rot[0]))
             @ m_rot_z(float(rot[2]))
             @ m_scale(float(scl[0]), float(scl[1]), float(scl[2])))
        mats[k] = M
    return mats


def build_dynamic_instance_colors(meshes, idxs):
    """为动态实例组构建 per-instance color (N, 4)，只取 mesh.color"""
    N = len(idxs)
    C = np.empty((N, 4), dtype=np.float32)
    for k, i in enumerate(idxs):
        col = meshes[i].get('color', (1, 1, 1))
        try:
            C[k, 0] = float(col[0]); C[k, 1] = float(col[1]); C[k, 2] = float(col[2])
        except (TypeError, IndexError):
            C[k, 0] = C[k, 1] = C[k, 2] = 1.0
        C[k, 3] = 1.0
    return C


def render_scene(scene, out_path):
    W, H = scene['width'], scene['height']
    fps = scene['fps']; dur = scene['duration']
    n_frames = int(dur * fps)
    base_dir = scene.get('base_dir', Path('.'))

    print(f"🎬 {dur}s @ {fps}fps  {W}x{H}")
    print(f"   mesh: {len(scene['meshes'])}  light: {len(scene['lights'])}")

    g = GL3D(W, H)

    # 上传几何
    for m in scene['meshes']:
        typ = m['type']
        parts = m['size'].split()
        sz = float(parts[0]) if parts else 1.0
        color = m['color']
        if typ == 'cube':
            fc = [color, tuple(c*0.55 for c in color), color,
                  tuple(c*0.55 for c in color), tuple(c*0.8 for c in color),
                  tuple(c*0.45 for c in color)]
            sizes = [float(x) for x in parts]
            if len(sizes) == 1: sizes = sizes * 3
            g.upload_mesh(m['id'], make_cube_vertices(tuple(sizes[:3]), fc))
        elif typ == 'plane':
            w = float(parts[0]) if len(parts) > 0 else 8
            h = float(parts[1]) if len(parts) > 1 else w
            g.upload_mesh(m['id'], make_plane_vertices(w, h, float(m['pos'][1]), color))
        elif typ == 'prism':
            hgt = m.get('_height', sz * 2.0)
            fc = {'top': color, 'bottom': tuple(c*0.5 for c in color),
                  's1': tuple(min(1.0, c*1.15) for c in color), 's2': color,
                  's3': tuple(c*0.7 for c in color)}
            g.upload_mesh(m['id'], make_prism_vertices(sz, hgt, fc))
        elif typ == 'sphere':
            g.upload_mesh(m['id'], make_sphere_vertices(sz, 32, 16, color))
        elif typ == 'cylinder':
            hgt = m.get('_height', sz * 2.0)
            g.upload_mesh(m['id'], make_cylinder_vertices(sz, hgt, 32, color))
        elif typ == 'cone':
            hgt = m.get('_height', sz * 2.0)
            g.upload_mesh(m['id'], make_cone_vertices(sz, hgt, 32, color))
        elif typ == 'paper':
            # 平坦纸张，细分 2
            w = float(parts[0]) if len(parts) > 0 else 2
            h = float(parts[1]) if len(parts) > 1 else w
            g.upload_mesh(m['id'], make_plane_subdiv_vertices(w, h, 2, 2, color))
        elif typ == 'wave':
            # 水平波浪面
            w = float(parts[0]) if len(parts) > 0 else 3
            h = float(parts[1]) if len(parts) > 1 else w
            seg = m.get('_seg', 48)
            g.upload_mesh(m['id'], make_plane_subdiv_vertices(w, h, seg, seg, color))
        elif typ in ('flag', 'curl'):
            # 竖直旗帜/卷曲面（XY 平面）
            w = float(parts[0]) if len(parts) > 0 else 3
            h = float(parts[1]) if len(parts) > 1 else w
            seg = m.get('_seg', 48)
            g.upload_mesh(m['id'], make_plane_subdiv_vertices_v(w, h, seg, seg, color))

        # 🆕 instancing：上传 per-instance model matrix
        if m.get('_instances'):
            mats = _instances_to_matrices(m['_instances'])
            n = g.upload_instance_transforms(m['id'], mats)
            print(f"   [inst] {m['id']} ← {n} 实例")

    # 加载纹理（按 key 去重：同内容的 data URI / 路径只上传一次）
    _tex_refs = {}
    _tex_printed = 0
    for m in scene['meshes']:
        spec = m.get('texture')
        if not spec: continue
        resolved = _resolve_texture_specs(spec, base_dir)
        tex_names = []
        try:
            for key, s in resolved:
                name = g.upload_texture_by_key(key, s)
                tex_names.append(name)
                _tex_refs[key] = _tex_refs.get(key, 0) + 1
        except Exception as e:
            print(f"   ⚠ 纹理失败 {m['id']}: {e}", file=sys.stderr)
            continue
        m['_tex_names'] = tex_names
        if _tex_printed < 8:
            print(f"   [tex] {m['id']} ← {len(resolved)} 张")
            _tex_printed += 1
    if _tex_refs:
        total = sum(_tex_refs.values())
        uniq = len(_tex_refs)
        print(f"   [tex] 共 {total} 次引用 → {uniq} 份唯一纹理")

    # 🆕 自动分组：把大量同类型的静态 mesh 归并成一次 draw call
    _inst_groups = group_instancing_candidates(scene['meshes'])
    _inst_member_set = set()
    for _ref, _members in _inst_groups:
        for _i in _members:
            _inst_member_set.add(_i)
    for _ref, _members in _inst_groups:
        _ref_m = scene['meshes'][_ref]
        try:
            _mats = build_relative_instance_matrices(scene['meshes'],
                                                       _members, _ref)
            g.upload_instance_transforms(_ref_m['id'], _mats)
            print(f"   [auto-inst] {_ref_m['id']} ← {len(_members)} 个同类 mesh")
        except Exception as e:
            print(f"   ⚠ auto-inst {_ref_m['id']}: {e}", file=sys.stderr)
    scene['_inst_groups'] = _inst_groups
    scene['_inst_member_set'] = _inst_member_set

    # 光源（已经是 shader 格式）
    lights = [{'type': lt['type'], 'pos': lt['pos'], 'dir': lt['dir'],
               'color': lt['color'], 'intensity': lt['intensity'],
               'range': lt['range'], 'spot_cos': lt.get('spot_cos', 0.9)}
              for lt in scene['lights']]

    # ffmpeg
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-f', 'rawvideo', '-pixel_format', 'rgba',
           '-video_size', f'{W}x{H}', '-framerate', str(fps), '-i', '-',
           '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    t0 = time.perf_counter(); t_render = 0.0

    try:
        for i in range(n_frames):
            t = i / fps
            ts = time.perf_counter()

            cam_pos, cam_look = _camera_at(scene['camera'], t)
            proj = m_perspective(scene['camera']['fov'], W / H, 0.1, 100.0)
            view = m_lookat(cam_pos, cam_look, [0, 1, 0])

            g.begin(clear_color=scene['bg'])
            g.set_lights(lights)

            # 🆕 自动实例化组：整组一次 draw
            for _ref, _members in scene.get('_inst_groups', []):
                _ref_m = scene['meshes'][_ref]
                g.draw_instanced(_ref_m['id'], proj @ view,
                                  textures=_ref_m.get('_tex_names', []))

            for _mi, m in enumerate(scene['meshes']):
                # 已被自动实例化覆盖 → 跳过
                if _mi in scene.get('_inst_member_set', ()):
                    continue
                # 🆕 显式 instances 分支
                if m.get('_instances'):
                    g.draw_instanced(m['id'], proj @ view,
                                      textures=m.get('_tex_names', []))
                    continue
                pos, rot, scl = _mesh_at(m, t)
                model = (m_translate(*pos) @ m_rot_y(float(rot[1]))
                         @ m_rot_x(float(rot[0])) @ m_rot_z(float(rot[2]))
                         @ m_scale(*scl))
                mvp = proj @ view @ model
                tex_names = m.get('_tex_names', [])
                # 顶点变形
                deform = m.get('_deform')
                if deform:
                    w = float(m['size'].split()[0]) if m['size'] else 1.0
                    g.set_deform(deform, t=t, width=w)
                else:
                    g.set_deform(None)
                g.draw(m['id'], mvp, model, textures=tex_names)

            arr = g.end()
            t_render += time.perf_counter() - ts
            proc.stdin.write(arr.tobytes())
            if (i + 1) % 30 == 0 or i == n_frames - 1:
                el = time.perf_counter() - t0
                print(f"   [{i+1:>4}/{n_frames}]  render {t_render:.2f}s  "
                      f"total {el:.2f}s  ({(i+1)/el:.1f} fps)")

        proc.stdin.close(); proc.wait()
    finally:
        g.close()

    total = time.perf_counter() - t0
    print()
    print(f"✅ {n_frames} 帧, {total:.2f}s, GPU {t_render:.2f}s "
          f"({t_render/n_frames*1000:.2f}ms/帧)")


# ============================================================
# Demo：一网打尽
# ============================================================
def _svg_badge(color1, color2, label):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{color1}"/>
      <stop offset="100%" stop-color="{color2}"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" fill="url(#g)"/>
  <circle cx="256" cy="256" r="180" fill="#0f172a" opacity="0.75"/>
  <circle cx="256" cy="256" r="165" fill="none" stroke="#ffffff" stroke-width="6"/>
  <circle cx="256" cy="256" r="120" fill="none" stroke="#ffffff" stroke-width="3"
          stroke-dasharray="12 10" opacity="0.6"/>
  <text x="256" y="290" font-family="sans-serif" font-size="140"
        font-weight="800" text-anchor="middle" fill="#ffffff">{label}</text>
</svg>'''


DEMO_XML = """<?xml version="1.0"?>
<scene duration="6" fps="30" width="1280" height="720" bg="#0a0e1a">
  <camera pos="0 2.2 9" lookAt="0 0.2 0" fov="52">
    <animate prop="pos" t0="0" t1="6" from="0 2.2 9" to="0 2.6 8" ease="ease-in-out"/>
  </camera>

  <light type="directional" dir="-0.4 -1 -0.3" intensity="0.55"/>
  <light type="ambient" color="#4a5a8a" intensity="0.4"/>
  <light type="point" pos="0 3 0" color="#ffaa44" intensity="2.0" range="10">
    <animate prop="pos" t0="0" t1="6" from="4 3 0" to="-4 3 0" ease="linear"/>
  </light>
  <light type="spot" pos="0 6 0" dir="0 -1 0" color="#44ddff"
         intensity="3.0" range="14" angle="35"/>

  <!-- 立方体（6 面不同纹理） -->
  <mesh id="cube" type="cube" size="1.2" pos="-3.6 0.3 -2"
        color="#ffffff" texture="__TEX_R|__TEX_G|__TEX_B|__TEX_Y|__TEX_P|__TEX_C">
    <animate prop="rotate" t0="0" t1="6" from="0 0 0" to="0 6.283 0" ease="linear"/>
  </mesh>

  <!-- 球体 -->
  <mesh id="sphere" type="sphere" size="0.75" pos="-1.6 0.5 -2"
        color="#ffdd66" texture="__TEX_SUN"/>

  <!-- 圆柱 + 圆锥（火箭） -->
  <mesh id="cyl" type="cylinder" size="0.55" height="1.4" pos="0.2 0.4 -2"
        color="#66dd66" texture="__TEX_G"/>
  <mesh id="cone" type="cone" size="0.55" height="0.8" pos="0.2 1.5 -2"
        color="#dd6644" texture="__TEX_R"/>

  <!-- 三棱柱 -->
  <mesh id="prism" type="prism" size="0.65" height="1.3" pos="2 0.3 -2"
        color="#8866ff" texture="__TEX_P"/>

  <!-- 🆕 纸张（平坦，双面） -->
  <mesh id="paper" type="paper" size="1.4 1.0" pos="3.6 0.5 -2"
        color="#ffeecc" texture="__TEX_PAPER"/>

  <!-- 🆕 波浪平面（水平，Y 轴正弦） -->
  <mesh id="wave" type="wave" size="4 2.4" seg="48" pos="-2.2 -0.4 1.5"
        color="#66ccff" texture="__TEX_WAVE"
        amp="0.35" freq="3.5" speed="2.0"/>

  <!-- 🆕 旗帜（竖直，末端飘动） -->
  <mesh id="flag" type="flag" size="3 1.6" seg="56" pos="2.4 1.3 1.5"
        color="#ffffff" texture="__TEX_FLAG"
        amp="0.3" freq="2.5" speed="2.2"/>

  <!-- 🆕 卷曲（动态半径） -->
  <mesh id="curl" type="curl" size="4 1.4" seg="64" pos="0 -0.3 4"
        color="#dd99ff" texture="__TEX_PURPLE"
        radius="1.2">
    <animate prop="pos" t0="0" t1="6" from="0 -0.3 4" to="0 -0.3 4" ease="linear"/>
  </mesh>

  <mesh id="floor" type="plane" size="18 18" pos="0 -0.6 0" color="#1e2440"/>
</scene>
"""


if __name__ == '__main__':
    # 生成 5 张 SVG 纹理（颜色 + 文字）
    texs = {
        '__TEX_R': _svg_badge('#ff4d6d', '#ffa07a', 'R'),
        '__TEX_G': _svg_badge('#44dd66', '#2a8040', 'G'),
        '__TEX_B': _svg_badge('#44aaff', '#1a4a88', 'B'),
        '__TEX_Y': _svg_badge('#ffcc44', '#cc8800', 'Y'),
        '__TEX_P': _svg_badge('#bb66ff', '#6633aa', 'P'),
        '__TEX_C': _svg_badge('#44ffee', '#008080', 'C'),
        '__TEX_PAPER': '''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <rect width="512" height="512" fill="#f5f0e0"/>
  <g stroke="#c0b090" stroke-width="2" opacity="0.7">
    <line x1="60" y1="80" x2="452" y2="80"/>
    <line x1="60" y1="130" x2="452" y2="130"/>
    <line x1="60" y1="180" x2="452" y2="180"/>
    <line x1="60" y1="230" x2="400" y2="230"/>
    <line x1="60" y1="280" x2="452" y2="280"/>
    <line x1="60" y1="330" x2="380" y2="330"/>
    <line x1="60" y1="380" x2="452" y2="380"/>
    <line x1="60" y1="430" x2="420" y2="430"/>
  </g>
  <rect x="20" y="20" width="472" height="472" fill="none" stroke="#8b6f47" stroke-width="4"/>
</svg>''',
        '__TEX_WAVE': '''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#3a9fff"/>
      <stop offset="100%" stop-color="#0044aa"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" fill="url(#g)"/>
  <g fill="none" stroke="#aaddff" stroke-width="6" opacity="0.7">
    <path d="M0,128 Q128,64 256,128 T512,128"/>
    <path d="M0,192 Q128,128 256,192 T512,192"/>
    <path d="M0,256 Q128,192 256,256 T512,256"/>
    <path d="M0,320 Q128,256 256,320 T512,320"/>
    <path d="M0,384 Q128,320 256,384 T512,384"/>
  </g>
</svg>''',
        '__TEX_FLAG': '''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#ff2a4a"/>
      <stop offset="50%" stop-color="#ffcc22"/>
      <stop offset="100%" stop-color="#2255ff"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" fill="url(#g)"/>
  <circle cx="256" cy="256" r="120" fill="#ffffff" opacity="0.85"/>
  <text x="256" y="300" font-family="sans-serif" font-size="120"
        font-weight="800" text-anchor="middle" fill="#1a1a2e">★</text>
</svg>''',
        '__TEX_PURPLE': '''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#e0a0ff"/>
      <stop offset="100%" stop-color="#6622aa"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" fill="url(#g)"/>
  <g fill="none" stroke="#ffffff" stroke-width="3" opacity="0.6">
    <circle cx="256" cy="256" r="60"/>
    <circle cx="256" cy="256" r="120"/>
    <circle cx="256" cy="256" r="180"/>
    <line x1="0" y1="256" x2="512" y2="256"/>
    <line x1="256" y1="0" x2="256" y2="512"/>
  </g>
</svg>''',
        '__TEX_PAPER': '''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <rect width="512" height="512" fill="#f5f0e0"/>
  <g stroke="#c0b090" stroke-width="2" opacity="0.7">
    <line x1="60" y1="80" x2="452" y2="80"/>
    <line x1="60" y1="130" x2="452" y2="130"/>
    <line x1="60" y1="180" x2="452" y2="180"/>
    <line x1="60" y1="230" x2="400" y2="230"/>
    <line x1="60" y1="280" x2="452" y2="280"/>
    <line x1="60" y1="330" x2="380" y2="330"/>
    <line x1="60" y1="380" x2="452" y2="380"/>
    <line x1="60" y1="430" x2="420" y2="430"/>
  </g>
  <rect x="20" y="20" width="472" height="472" fill="none" stroke="#8b6f47" stroke-width="4"/>
</svg>''',
        '__TEX_WAVE': '''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#3a9fff"/>
      <stop offset="100%" stop-color="#0044aa"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" fill="url(#g)"/>
  <g fill="none" stroke="#aaddff" stroke-width="6" opacity="0.7">
    <path d="M0,128 Q128,64 256,128 T512,128"/>
    <path d="M0,192 Q128,128 256,192 T512,192"/>
    <path d="M0,256 Q128,192 256,256 T512,256"/>
    <path d="M0,320 Q128,256 256,320 T512,320"/>
    <path d="M0,384 Q128,320 256,384 T512,384"/>
  </g>
</svg>''',
        '__TEX_FLAG': '''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#ff2a4a"/>
      <stop offset="50%" stop-color="#ffcc22"/>
      <stop offset="100%" stop-color="#2255ff"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" fill="url(#g)"/>
  <circle cx="256" cy="256" r="120" fill="#ffffff" opacity="0.85"/>
  <text x="256" y="300" font-family="sans-serif" font-size="120"
        font-weight="800" text-anchor="middle" fill="#1a1a2e">★</text>
</svg>''',
        '__TEX_PURPLE': '''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#e0a0ff"/>
      <stop offset="100%" stop-color="#6622aa"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" fill="url(#g)"/>
  <g fill="none" stroke="#ffffff" stroke-width="3" opacity="0.6">
    <circle cx="256" cy="256" r="60"/>
    <circle cx="256" cy="256" r="120"/>
    <circle cx="256" cy="256" r="180"/>
    <line x1="0" y1="256" x2="512" y2="256"/>
    <line x1="256" y1="0" x2="256" y2="512"/>
  </g>
</svg>''',
        '__TEX_SUN': '''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <defs>
    <radialGradient id="g" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0%" stop-color="#ffffcc"/>
      <stop offset="60%" stop-color="#ffcc44"/>
      <stop offset="100%" stop-color="#ff6600"/>
    </radialGradient>
  </defs>
  <rect width="512" height="512" fill="#1a1a2e"/>
  <circle cx="256" cy="256" r="220" fill="url(#g)"/>
  <circle cx="256" cy="256" r="220" fill="none" stroke="#ffffff" stroke-width="4" opacity="0.5"/>
</svg>''',
    }

    xml_text = DEMO_XML
    for k, svg in texs.items():
        b64 = base64.b64encode(svg.encode('utf-8')).decode()
        xml_text = xml_text.replace(k, f'data:image/svg+xml;base64,{b64}')

    out = 'scene3d.mp4'
    if len(sys.argv) == 2:
        out = sys.argv[1]
    elif len(sys.argv) >= 3:
        with open(sys.argv[1], encoding='utf-8') as f:
            xml_text = f.read()
        out = sys.argv[2]

    scene = parse_scene_xml(xml_text, base_dir='.')
    render_scene(scene, out)
