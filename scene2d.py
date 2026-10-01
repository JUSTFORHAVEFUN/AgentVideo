#!/usr/bin/env python3
"""
scene2d.py v2 —— SVG 子集 → Raster2D（GPU）

支持:
  · <rect> <circle> <ellipse> <line> <polygon> <polyline>
  · <text> <image> <g> <path>（M/L/H/V/Z 命令）
  · transform: translate/scale/rotate/matrix/skewX/skewY
  · 属性: fill, opacity, stroke, stroke-width, text-anchor, font-family
"""
# scene2d.py
__version__ = '1.0.1'

from __future__ import annotations
import base64, math, re, sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from raster2d import Raster2D, Mat2D, get_raster2d


_NAMED = {
    'white': (1,1,1,1), 'black': (0,0,0,1), 'red': (1,0,0,1),
    'green': (0,0.5,0,1), 'blue': (0,0,1,1), 'yellow': (1,1,0,1),
    'orange': (1,0.65,0,1), 'cyan': (0,1,1,1), 'magenta': (1,0,1,1),
    'gray': (0.5,0.5,0.5,1), 'grey': (0.5,0.5,0.5,1),
    'transparent': (0,0,0,0), 'none': (0,0,0,0),
}


def parse_color(s, opacity=1.0):
    if not s:
        return (0, 0, 0, 0)
    s = str(s).strip().lower()
    if s in _NAMED:
        r, g, b, a = _NAMED[s]
        return (r, g, b, a * opacity)
    if s.startswith('#'):
        h = s[1:]
        if len(h) == 3:
            h = ''.join(c * 2 for c in h)
        if len(h) == 6:
            return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255,
                    int(h[4:6], 16) / 255, opacity)
        if len(h) == 8:
            return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255,
                    int(h[4:6], 16) / 255, int(h[6:8], 16) / 255 * opacity)
    m = re.match(r'rgba?\(([^)]+)\)', s)
    if m:
        parts = [p.strip() for p in m.group(1).split(',')]
        if len(parts) >= 3:
            try:
                r, g, b = [float(x) / 255 for x in parts[:3]]
                a = float(parts[3]) if len(parts) > 3 else 1.0
                return (r, g, b, a * opacity)
            except ValueError:
                pass
    return (0, 0, 0, opacity)


# ============================================================
# 元素遍历（累积变换）
# ============================================================
def _draw_ring(r2d, cx, cy, r, width, color, segments=64):
    """画圆环（描边），不填充。用 N 个梯形拼成。"""
    outer = r + width / 2.0
    inner = max(0.0, r - width / 2.0)
    for i in range(segments):
        a0 = 2 * math.pi * i / segments
        a1 = 2 * math.pi * (i + 1) / segments
        p0 = (cx + inner * math.cos(a0), cy + inner * math.sin(a0))
        p1 = (cx + outer * math.cos(a0), cy + outer * math.sin(a0))
        p2 = (cx + outer * math.cos(a1), cy + outer * math.sin(a1))
        p3 = (cx + inner * math.cos(a1), cy + inner * math.sin(a1))
        r2d._push_quad_pts(p0, p1, p2, p3, (0, 0, 1, 1), color)


def _iter_svg(elem, mat):
    tag = elem.tag
    if isinstance(tag, str) and tag.startswith('{'):
        tag = tag.split('}', 1)[1]
    local = Mat2D.parse(elem.get('transform', ''))
    cur = mat @ local
    yield tag, elem, cur
    for child in elem:
        yield from _iter_svg(child, cur)


# ============================================================
# path 命令解析（M/L/H/V/Z，C/Q/A 用直线段近似）
# ============================================================
_PATH_TOKEN = re.compile(r'([MLHVCQAZmlhvcqaz])([^MLHVCQAZmlhvcqaz]*)')


def parse_path(d):
    """返回子路径列表：[[(x,y), ...], ...]"""
    subpaths = []
    cur = []
    cx = cy = 0.0
    start_x = start_y = 0.0

    def cmd(c, args):
        nonlocal cx, cy, start_x, start_y, cur
        if c in 'M':
            if args and len(args) >= 2:
                cx, cy = args[0], args[1]
                if cur:
                    subpaths.append(cur)
                cur = [(cx, cy)]
                start_x, start_y = cx, cy
        elif c in 'L':
            for i in range(0, len(args) - 1, 2):
                cx, cy = args[i], args[i + 1]
                cur.append((cx, cy))
        elif c in 'H':
            for x in args:
                cx = x
                cur.append((cx, cy))
        elif c in 'V':
            for y in args:
                cy = y
                cur.append((cx, cy))
        elif c in 'C':
            # 三次贝塞尔 → 直线近似（端点的 1/3 采样）
            for i in range(0, len(args) - 5, 6):
                x0, y0 = cx, cy
                x1, y1 = args[i], args[i + 1]
                x2, y2 = args[i + 2], args[i + 3]
                x3, y3 = args[i + 4], args[i + 5]
                for k in range(1, 5):
                    t = k / 4
                    it = 1 - t
                    x = it ** 3 * x0 + 3 * it ** 2 * t * x1 + 3 * it * t ** 2 * x2 + t ** 3 * x3
                    y = it ** 3 * y0 + 3 * it ** 2 * t * y1 + 3 * it * t ** 2 * y2 + t ** 3 * y3
                    cur.append((x, y))
                cx, cy = x3, y3
        elif c in 'Q':
            for i in range(0, len(args) - 3, 4):
                x0, y0 = cx, cy
                x1, y1 = args[i], args[i + 1]
                x2, y2 = args[i + 2], args[i + 3]
                for k in range(1, 4):
                    t = k / 3
                    it = 1 - t
                    x = it ** 2 * x0 + 2 * it * t * x1 + t ** 2 * x2
                    y = it ** 2 * y0 + 2 * it * t * y1 + t ** 2 * y2
                    cur.append((x, y))
                cx, cy = x2, y2
        elif c in 'Z':
            if cur:
                cur.append((start_x, start_y))

    for m in _PATH_TOKEN.finditer(d):
        c = m.group(1)
        args_str = m.group(2).strip()
        try:
            args = [float(x) for x in re.split(r'[\s,]+', args_str) if x]
        except ValueError:
            continue
        upper = c.upper()
        if c != upper and c in 'mlhvcqaz':
            # 相对坐标：转成绝对
            if upper == 'M':
                base_x, base_y = cx, cy
                args = [v + (base_x if i % 2 == 0 else base_y) for i, v in enumerate(args)]
            elif upper == 'L':
                args = [v + (cx if i % 2 == 0 else cy) for i, v in enumerate(args)]
            elif upper == 'H':
                args = [v + cx for v in args]
            elif upper == 'V':
                args = [v + cy for v in args]
            elif upper == 'C':
                args = [v + (cx if i % 2 == 0 else cy) for i, v in enumerate(args)]
            elif upper == 'Q':
                args = [v + (cx if i % 2 == 0 else cy) for i, v in enumerate(args)]
            cmd(upper, args)
        else:
            cmd(upper, args)

    if cur:
        subpaths.append(cur)
    return subpaths


# ============================================================
# 主入口
# ============================================================
def svg_to_raster2d(svg_str, W, H, r2d=None, bg=None):
    if r2d is None:
        r2d = get_raster2d(W, H)
    root = ET.fromstring(svg_str)

    # viewBox 处理
    vb = root.get('viewBox')
    k = 1.0; ox = oy = 0.0
    if vb:
        parts = re.split(r'[\s,]+', vb.strip())
        if len(parts) == 4:
            try:
                vx, vy, vw, vh = [float(x) for x in parts]
                if vw > 0 and vh > 0:
                    k = min(W / vw, H / vh)
                    ox = (W - vw * k) / 2 - vx * k
                    oy = (H - vh * k) / 2 - vy * k
            except ValueError:
                pass

    r2d.begin(clear=bg if bg else (0, 0, 0, 0))
    # 全局 viewBox 缩放作为根变换
    base = Mat2D(k, 0, 0, k, ox, oy)

    for tag, elem, mat in _iter_svg(root, base):
        opacity = float(elem.get('opacity', '1') or '1')
        try:
            fill_op = float(elem.get('fill-opacity', '1') or '1')
        except (ValueError, TypeError):
            fill_op = 1.0
        try:
            stroke_op = float(elem.get('stroke-opacity', '1') or '1')
        except (ValueError, TypeError):
            stroke_op = 1.0
        fill = elem.get('fill', 'black')
        stroke = elem.get('stroke')
        stroke_w = float(elem.get('stroke-width', '0') or '0')
        fill_color = parse_color(fill, opacity * fill_op)
        stroke_color = parse_color(stroke, opacity * stroke_op) if stroke else (0, 0, 0, 0)

        # 每个元素单独 set_matrix
        r2d.set_matrix(mat)

        if tag == 'rect':
            try:
                x = float(elem.get('x', 0))
                y = float(elem.get('y', 0))
                w_raw = elem.get('width', '0')
                h_raw = elem.get('height', '0')
                w = W if w_raw.endswith('%') else float(w_raw)
                h = H if h_raw.endswith('%') else float(h_raw)
            except (ValueError, AttributeError):
                continue
            if w > 0 and h > 0:
                if fill_color[3] > 0:
                    r2d.draw_rect(x, y, w, h, fill_color)
                if stroke_color[3] > 0 and stroke_w > 0:
                    # 简单描边：4 条线
                    r2d.draw_line(x, y, x + w, y, stroke_w, stroke_color)
                    r2d.draw_line(x + w, y, x + w, y + h, stroke_w, stroke_color)
                    r2d.draw_line(x + w, y + h, x, y + h, stroke_w, stroke_color)
                    r2d.draw_line(x, y + h, x, y, stroke_w, stroke_color)

        elif tag == 'circle':
            try:
                cx = float(elem.get('cx', 0))
                cy = float(elem.get('cy', 0))
                rr = float(elem.get('r', 0))
            except ValueError:
                continue
            if rr > 0 and fill_color[3] > 0:
                r2d.draw_circle(cx, cy, rr, fill_color)
            if rr > 0 and stroke_color[3] > 0 and stroke_w > 0:
                _draw_ring(r2d, cx, cy, rr, stroke_w, stroke_color)

        elif tag == 'ellipse':
            try:
                cx = float(elem.get('cx', 0))
                cy = float(elem.get('cy', 0))
                rx = float(elem.get('rx', 0))
                ry = float(elem.get('ry', 0))
            except ValueError:
                continue
            if rx > 0 and ry > 0 and fill_color[3] > 0:
                r2d.draw_ellipse(cx, cy, rx, ry, fill_color)

        elif tag == 'line':
            try:
                x1 = float(elem.get('x1', 0))
                y1 = float(elem.get('y1', 0))
                x2 = float(elem.get('x2', 0))
                y2 = float(elem.get('y2', 0))
            except ValueError:
                continue
            w = stroke_w if stroke_w > 0 else 2.0
            c = stroke_color if stroke_color[3] > 0 else fill_color
            r2d.draw_line(x1, y1, x2, y2, w, c)

        elif tag == 'polygon':
            pts_raw = elem.get('points', '')
            try:
                nums = [float(x) for x in re.split(r'[\s,]+', pts_raw.strip()) if x]
            except ValueError:
                continue
            pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
            if len(pts) >= 3:
                if fill_color[3] > 0:
                    r2d.draw_polygon(pts, fill_color)
                if stroke_color[3] > 0 and stroke_w > 0:
                    r2d.draw_polyline(pts + [pts[0]], stroke_w, stroke_color)

        elif tag == 'polyline':
            pts_raw = elem.get('points', '')
            try:
                nums = [float(x) for x in re.split(r'[\s,]+', pts_raw.strip()) if x]
            except ValueError:
                continue
            pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
            if len(pts) >= 2:
                w = stroke_w if stroke_w > 0 else 2.0
                c = stroke_color if stroke_color[3] > 0 else fill_color
                r2d.draw_polyline(pts, w, c)

        elif tag == 'path':
            d = elem.get('d', '')
            if not d:
                continue
            subpaths = parse_path(d)
            for sp in subpaths:
                if len(sp) < 2:
                    continue
                if fill_color[3] > 0 and len(sp) >= 3:
                    r2d.draw_polygon(sp, fill_color)
                if stroke_color[3] > 0 and stroke_w > 0:
                    r2d.draw_polyline(sp, stroke_w, stroke_color)

        elif tag == 'text':
            content = (elem.text or '') + ''.join(c.tail or '' for c in elem)
            content = content.strip()
            if not content:
                continue
            try:
                x = float(elem.get('x', 0))
                y = float(elem.get('y', 0))
                size = float(elem.get('font-size', '48'))
            except ValueError:
                continue
            # 只取第一个 family（和 cairo/pango 的 fallback 语义一致）
            fam_full = elem.get('font-family', 'sans-serif') or 'sans-serif'
            family = fam_full.split(',')[0].strip().strip('"').strip("'")
            if not family:
                family = 'sans-serif'
            weight = elem.get('font-weight', 'normal')
            anchor = {'start': 'start', 'middle': 'middle',
                       'end': 'end'}.get(elem.get('text-anchor', 'start'), 'start')
            # dominant-baseline → raster2d 的 baseline 参数
            db = elem.get('dominant-baseline', 'alphabetic').strip().lower()
            baseline_map = {
                'middle': 'middle', 'central': 'central',
                'hanging': 'hanging', 'text-top': 'text-top',
                'text-bottom': 'text-bottom', 'ideographic': 'ideographic',
                'alphabetic': 'alphabetic', 'auto': 'alphabetic',
                'mathematical': 'alphabetic',
            }
            bl = baseline_map.get(db, 'alphabetic')
            r2d.draw_text(content, x, y, size, fill_color,
                           font=family, anchor=anchor, baseline=bl,
                           weight=weight)

        elif tag == 'image':
            href = elem.get('{http://www.w3.org/1999/xlink}href') or elem.get('href', '')
            if not href.startswith('data:'):
                continue
            _, _, b64 = href.partition(',')
            try:
                png = base64.b64decode(b64)
            except Exception:
                continue
            try:
                x = float(elem.get('x', 0))
                y = float(elem.get('y', 0))
                w = float(elem.get('width', '0'))
                h = float(elem.get('height', '0'))
            except ValueError:
                continue
            if w > 0 and h > 0:
                r2d.draw_image(png, x, y, w, h, opacity=opacity)

    r2d.set_matrix(None)
    return r2d.end()


# ============================================================
# 直接绘制到已 begin 的 r2d（不 end，不 ET 重复解析）
# ============================================================
def _draw_svg_root_to_r2d(r2d, root, base_mat, opacity_mult=1.0):
    """把已解析的 SVG 根节点画到 r2d（不调 begin/end）"""
    for tag, elem, mat in _iter_svg(root, base_mat):
        try:
            opacity = float(elem.get('opacity', '1') or '1') * opacity_mult
        except (ValueError, TypeError):
            opacity = opacity_mult
        try:
            fill_op = float(elem.get('fill-opacity', '1') or '1')
        except (ValueError, TypeError):
            fill_op = 1.0
        try:
            stroke_op = float(elem.get('stroke-opacity', '1') or '1')
        except (ValueError, TypeError):
            stroke_op = 1.0
        fill = elem.get('fill', 'black')
        stroke = elem.get('stroke')
        try:
            stroke_w = float(elem.get('stroke-width', '0') or '0')
        except ValueError:
            stroke_w = 0.0
        fill_color = parse_color(fill, opacity * fill_op)
        stroke_color = parse_color(stroke, opacity * stroke_op) if stroke else (0, 0, 0, 0)

        r2d.set_matrix(mat)

        if tag == 'rect':
            try:
                x = float(elem.get('x', 0)); y = float(elem.get('y', 0))
                w = float(elem.get('width', '0') or 0)
                h = float(elem.get('height', '0') or 0)
            except ValueError:
                continue
            if w > 0 and h > 0:
                if fill_color[3] > 0:
                    r2d.draw_rect(x, y, w, h, fill_color)
                if stroke_color[3] > 0 and stroke_w > 0:
                    r2d.draw_line(x, y, x+w, y, stroke_w, stroke_color)
                    r2d.draw_line(x+w, y, x+w, y+h, stroke_w, stroke_color)
                    r2d.draw_line(x+w, y+h, x, y+h, stroke_w, stroke_color)
                    r2d.draw_line(x, y+h, x, y, stroke_w, stroke_color)

        elif tag == 'circle':
            try:
                cx = float(elem.get('cx', 0)); cy = float(elem.get('cy', 0))
                rr = float(elem.get('r', 0))
            except ValueError:
                continue
            if rr > 0 and fill_color[3] > 0:
                r2d.draw_circle(cx, cy, rr, fill_color)
            if rr > 0 and stroke_color[3] > 0 and stroke_w > 0:
                _draw_ring(r2d, cx, cy, rr, stroke_w, stroke_color)

        elif tag == 'ellipse':
            try:
                cx = float(elem.get('cx', 0)); cy = float(elem.get('cy', 0))
                rx = float(elem.get('rx', 0)); ry = float(elem.get('ry', 0))
            except ValueError:
                continue
            if rx > 0 and ry > 0 and fill_color[3] > 0:
                r2d.draw_ellipse(cx, cy, rx, ry, fill_color)

        elif tag == 'line':
            try:
                x1 = float(elem.get('x1', 0)); y1 = float(elem.get('y1', 0))
                x2 = float(elem.get('x2', 0)); y2 = float(elem.get('y2', 0))
            except ValueError:
                continue
            w = stroke_w if stroke_w > 0 else 2.0
            c = stroke_color if stroke_color[3] > 0 else fill_color
            r2d.draw_line(x1, y1, x2, y2, w, c)

        elif tag in ('polygon', 'polyline'):
            pts_raw = elem.get('points', '')
            try:
                nums = [float(x) for x in re.split(r'[\s,]+', pts_raw.strip()) if x]
            except ValueError:
                continue
            pts = [(nums[i], nums[i+1]) for i in range(0, len(nums)-1, 2)]
            if tag == 'polygon' and len(pts) >= 3:
                if fill_color[3] > 0:
                    r2d.draw_polygon(pts, fill_color)
                if stroke_color[3] > 0 and stroke_w > 0:
                    r2d.draw_polyline(pts + [pts[0]], stroke_w, stroke_color)
            elif tag == 'polyline' and len(pts) >= 2:
                w = stroke_w if stroke_w > 0 else 2.0
                c = stroke_color if stroke_color[3] > 0 else fill_color
                r2d.draw_polyline(pts, w, c)

        elif tag == 'path':
            d = elem.get('d', '')
            if not d:
                continue
            for sp in parse_path(d):
                if len(sp) < 2:
                    continue
                if fill_color[3] > 0 and len(sp) >= 3:
                    r2d.draw_polygon(sp, fill_color)
                if stroke_color[3] > 0 and stroke_w > 0:
                    r2d.draw_polyline(sp, stroke_w, stroke_color)

        elif tag == 'text':
            content = (elem.text or '') + ''.join(c.tail or '' for c in elem)
            content = content.strip()
            if not content:
                continue
            try:
                x = float(elem.get('x', 0)); y = float(elem.get('y', 0))
                size = float(elem.get('font-size', '48'))
            except ValueError:
                continue
            family = elem.get('font-family', 'sans-serif')
            anchor = {'start': 'start', 'middle': 'middle',
                       'end': 'end'}.get(elem.get('text-anchor', 'start'), 'start')
            r2d.draw_text(content, x, y, size, fill_color,
                           font=family, anchor=anchor, baseline='alphabetic')

        elif tag == 'image':
            href = (elem.get('{http://www.w3.org/1999/xlink}href')
                     or elem.get('href', ''))
            if not href.startswith('data:'):
                continue
            _, _, b64 = href.partition(',')
            try:
                png = base64.b64decode(b64)
            except Exception:
                continue
            try:
                x = float(elem.get('x', 0)); y = float(elem.get('y', 0))
                w = float(elem.get('width', '0') or 0)
                h = float(elem.get('height', '0') or 0)
            except ValueError:
                continue
            if w > 0 and h > 0:
                r2d.draw_image(png, x, y, w, h, opacity=opacity)


def draw_svg_inner_to_r2d(r2d, inner_str, base_mat, opacity=1.0):
    """把 SVG inner 内容（无外层 <svg>）画到已 begin 的 r2d。
    base_mat: Mat2D（调用者的变换矩阵）"""
    if not inner_str:
        return
    wrapped = ('<svg xmlns="http://www.w3.org/2000/svg" '
               'xmlns:xlink="http://www.w3.org/1999/xlink">'
               f'{inner_str}</svg>')
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        return
    _draw_svg_root_to_r2d(r2d, root, base_mat, opacity)


def draw_svg_to_raster2d(r2d, svg_str, base_mat, opacity=1.0):
    """把完整 SVG 字符串画到已 begin 的 r2d（不调 begin/end）"""
    try:
        root = ET.fromstring(svg_str)
    except ET.ParseError:
        return
    # viewBox 处理（假设 SVG 本身的 width/height 已经是目标尺寸）
    _draw_svg_root_to_r2d(r2d, root, base_mat, opacity)


# ============================================================
# 支持度检测
# ============================================================
_LEVEL_BLOCKED_HARD = frozenset([
    'mask', 'clipPath', 'linearGradient', 'radialGradient',
    'filter', 'pattern', 'use', 'symbol', 'marker', 'textPath', 'switch',
])
_LEVEL_OK = frozenset([
    'rect', 'circle', 'ellipse', 'line', 'polygon', 'polyline',
    'path', 'text', 'image', 'g', 'svg', 'defs', 'tspan',
    'title', 'desc', 'style', 'metadata', 'stop', 'xlink',
])


def svg_gpu_level(svg_bytes):
    """返回 0=cairo, 1=gpu (basic), 2=gpu (full), 3=gpu (force 时都会尝试)"""
    try:
        text = svg_bytes.decode('utf-8', errors='ignore')
    except Exception:
        return 0
    tags = set()
    for m in re.finditer(r'<([a-zA-Z][a-zA-Z0-9]*)', text):
        tag = m.group(1)
        if tag not in _LEVEL_OK:
            tags.add(tag)
    # 有硬阻塞 → 只能 cairo
    if tags & _LEVEL_BLOCKED_HARD:
        return 0
    # 有未知标签 → basic
    if tags:
        return 1
    return 2


if __name__ == '__main__':
    import time
    SVG = '''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">
  <rect width="640" height="360" fill="#101a2a"/>
  <rect x="40" y="40" width="200" height="100" fill="#ff6644" opacity="0.9"/>
  <circle cx="420" cy="90" r="60" fill="#44ddff" opacity="0.85"/>
  <line x1="40" y1="200" x2="600" y2="200" stroke="#ffffff" stroke-width="4"/>
  <polygon points="320,220 380,300 260,300" fill="#ffcc44" opacity="0.9"/>
  <polyline points="60,320 120,280 180,320 240,280" stroke="#5dff9e" stroke-width="5" fill="none"/>
  <path d="M 450 240 L 550 240 L 550 320 L 450 320 Z" fill="#bb66ff" opacity="0.7"/>
  <text x="320" y="180" font-size="48" fill="white" text-anchor="middle"
        font-family="sans-serif">SVG → GPU 全标签</text>
</svg>'''
    W, H = 1280, 720
    t0 = time.perf_counter()
    arr = svg_to_raster2d(SVG, W, H)
    dt = time.perf_counter() - t0
    Image.fromarray(arr, 'RGBA').save('tmp/scene2d_v2.png')
    print(f"✅ 单次 {dt*1000:.1f}ms → tmp/scene2d_v2.png")

    N = 30
    t0 = time.perf_counter()
    for _ in range(N):
        svg_to_raster2d(SVG, W, H)
    dt = time.perf_counter() - t0
    print(f"✅ {N} 帧 {dt*1000:.1f}ms  单帧 {dt/N*1000:.2f}ms  {N/dt:.1f} fps")
