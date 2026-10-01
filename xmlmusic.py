#!/usr/bin/env python3
"""
xmlmusic.py v0.2 — XML → WAV/MIDI 音乐合成器

v0.2 新增：
  · 和弦语法糖 <chord pitch="Cmaj7" at="0b" dur="4b"/>
  · 鼓 pattern <pattern instrument="kick" at="0b" beat="X...X..." step="0.25b"/>
  · 琶音 <arp pitch="Cmaj7" at="0b" dur="4b" step="0.5b" order="up"/>
  · 音符级 ADSR 覆盖
  · 更多音色：bass strings organ bell tom clap crash noise
  · 立体声 + pan + delay + reverb
  · MIDI 导出
"""
__version__ = '1.0.1'

import sys, wave, re, struct
from pathlib import Path
from typing import Optional
import numpy as np
import xml.etree.ElementTree as ET

SR_DEFAULT = 44100
SEMI = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
NOTE_RE  = re.compile(r'^([A-Ga-g])([#b]?)(-?\d+)$')
CHORD_RE = re.compile(r'^([A-Ga-g])([#b]?)(-?\d+)?(.*)$')

CHORD_QUALITIES = {
    '':     [0,4,7],     'maj':  [0,4,7],      'M':    [0,4,7],
    'min':  [0,3,7],     'm':    [0,3,7],      'dim':  [0,3,6],
    'aug':  [0,4,8],     'sus2': [0,2,7],      'sus4': [0,5,7],
    '5':    [0,7],       '7':    [0,4,7,10],   'dom7': [0,4,7,10],
    'maj7': [0,4,7,11],  'M7':   [0,4,7,11],   'm7':   [0,3,7,10],
    'min7': [0,3,7,10],  'm7b5': [0,3,6,10],   'dim7': [0,3,6,9],
    '6':    [0,4,7,9],   'm6':   [0,3,7,9],    'add9': [0,4,7,14],
    '9':    [0,4,7,10,14],'maj9':[0,4,7,11,14],'m9':   [0,3,7,10,14],
}


# ═══════════════════════════════════════════════
# Pitch / Chord / Time
# ═══════════════════════════════════════════════
def parse_pitch(s):
    if s is None: return None
    s = str(s).strip()
    try: return int(s)
    except ValueError: pass
    m = NOTE_RE.match(s)
    if not m: raise ValueError(f"无法解析 pitch: {s!r}")
    letter, acc, octv = m.group(1).upper(), m.group(2), int(m.group(3))
    midi = (octv + 1) * 12 + SEMI[letter]
    if acc == '#': midi += 1
    elif acc == 'b': midi -= 1
    return midi

def note_to_freq(n):
    return 0.0 if n is None or n <= 0 else 440.0 * (2.0 ** ((n - 69) / 12.0))

def parse_chord(s, default_oct=3):
    """'Cmaj7' 或 'C4maj7' → [60, 64, 67, 71]"""
    m = CHORD_RE.match(str(s).strip())
    if not m: raise ValueError(f"无法解析 chord: {s!r}")
    letter = m.group(1).upper()
    acc = m.group(2)
    octv = int(m.group(3)) if m.group(3) else default_oct
    quality = m.group(4).strip()
    if quality not in CHORD_QUALITIES:
        raise ValueError(f"未知和弦类型: {quality!r} (from {s!r})")
    root = (octv + 1) * 12 + SEMI[letter]
    if acc == '#': root += 1
    elif acc == 'b': root -= 1
    return [root + k for k in CHORD_QUALITIES[quality]]

def parse_time(s, bpm):
    if s is None: return 0.0
    s = str(s).strip()
    if s.endswith('b'): return float(s[:-1]) * 60.0 / bpm
    if s.endswith('s'): return float(s[:-1])
    return float(s)


# ═══════════════════════════════════════════════
# 音色
# ═══════════════════════════════════════════════
def _adsr(n, a, d, s, r, sr):
    na, nd, nr = int(a*sr), int(d*sr), int(r*sr)
    ns = max(0, n - na - nd - nr)
    parts = []
    if na: parts.append(np.linspace(0, 1, na))
    if nd: parts.append(np.linspace(1, s, nd))
    if ns: parts.append(np.full(ns, s))
    if nr: parts.append(np.linspace(s, 0, nr))
    env = np.concatenate(parts) if parts else np.zeros(n)
    return np.pad(env, (0, max(0, n - len(env))))[:n]

def _osc(freq, n, kind, sr, phase=0.0):
    if freq <= 0: return np.zeros(n)
    t = np.arange(n) / sr
    ph = 2 * np.pi * freq * t + phase
    if kind == 'sine':     return np.sin(ph)
    if kind == 'saw':      return 2 * ((freq*t + phase/(2*np.pi)) % 1.0) - 1.0
    if kind == 'square':   return np.sign(np.sin(ph))
    if kind == 'triangle': return 2 * np.abs(2*((freq*t) % 1.0) - 1.0) - 1.0
    return np.sin(ph)

def _apply_adsr_override(w, n, sr, a, d, s, r):
    return w * _adsr(n, a, d, s, r, sr)

# ── 具体音色 ──
def _piano(freq, n, sr, vel):
    if freq <= 0: return np.zeros(n)
    t = np.arange(n) / sr
    w  = np.sin(2*np.pi*freq*t)   * 0.70
    w += np.sin(2*np.pi*freq*2*t) * 0.20
    w += np.sin(2*np.pi*freq*3*t) * 0.07
    w += np.sin(2*np.pi*freq*4*t) * 0.03
    env = np.exp(-t * 2.5)
    a = min(n, int(0.005 * sr))
    if a: env[:a] *= np.linspace(0, 1, a)
    return w * env * vel

def _bass(freq, n, sr, vel):
    if freq <= 0: return np.zeros(n)
    t = np.arange(n) / sr
    w  = np.sin(2*np.pi*freq*t) * 0.9
    w += np.sin(2*np.pi*freq*2*t) * 0.15
    w += _osc(freq*0.5, n, 'saw', sr) * 0.1
    env = np.exp(-t * 1.2)
    a = min(n, int(0.01 * sr))
    if a: env[:a] *= np.linspace(0, 1, a)
    return w * env * vel * 0.8

def _strings(freq, n, sr, vel):
    if freq <= 0: return np.zeros(n)
    t = np.arange(n) / sr
    # 轻微失谐叠加产生弦乐厚度
    w  = _osc(freq, n, 'saw', sr) * 0.5
    w += _osc(freq * 1.005, n, 'saw', sr) * 0.3
    w += _osc(freq * 0.997, n, 'saw', sr) * 0.3
    # 低通滤波（简单滑动平均）
    k = max(1, int(sr / freq / 4))
    w = np.convolve(w, np.ones(k)/k, mode='same')
    env = _adsr(n, 0.15, 0.1, 0.8, 0.3, sr)
    return w * env * vel * 0.6

def _organ(freq, n, sr, vel):
    if freq <= 0: return np.zeros(n)
    t = np.arange(n) / sr
    w  = np.sin(2*np.pi*freq*t)   * 0.6
    w += np.sin(2*np.pi*freq*2*t) * 0.3
    w += np.sin(2*np.pi*freq*3*t) * 0.15
    w += np.sin(2*np.pi*freq*4*t) * 0.1
    env = _adsr(n, 0.02, 0.05, 0.95, 0.1, sr)
    return w * env * vel * 0.5

def _bell(freq, n, sr, vel):
    if freq <= 0: return np.zeros(n)
    t = np.arange(n) / sr
    # 钟琴：高频泛音 + 快速初始衰减
    w  = np.sin(2*np.pi*freq*t)     * 0.5
    w += np.sin(2*np.pi*freq*2.76*t)* 0.3
    w += np.sin(2*np.pi*freq*5.4*t) * 0.15
    w += np.sin(2*np.pi*freq*8.9*t) * 0.05
    env = np.exp(-t * 3.5)
    return w * env * vel

def _noise(freq, n, sr, vel):
    rng = np.random.default_rng(int(freq) % 1000 + 100)
    t = np.arange(n) / sr
    return rng.standard_normal(n) * np.exp(-t*3) * vel * 0.4

# ── 鼓 ──
def _kick(_freq, n, sr, vel):
    t = np.arange(n) / sr
    freq = 60 * np.exp(-t * 30) + 30
    return np.sin(2*np.pi*np.cumsum(freq)/sr) * np.exp(-t*8) * vel

def _snare(_freq, n, sr, vel):
    rng = np.random.default_rng(42)
    t = np.arange(n) / sr
    return rng.standard_normal(n) * np.exp(-t*18) * vel * 0.7

def _hihat(_freq, n, sr, vel):
    rng = np.random.default_rng(43)
    t = np.arange(n) / sr
    noise = np.diff(rng.standard_normal(n), prepend=0)
    return noise * np.exp(-t*60) * vel * 0.3

def _tom(_freq, n, sr, vel):
    t = np.arange(n) / sr
    freq = 150 * np.exp(-t * 15) + 80
    return np.sin(2*np.pi*np.cumsum(freq)/sr) * np.exp(-t*10) * vel * 0.8

def _clap(_freq, n, sr, vel):
    rng = np.random.default_rng(44)
    t = np.arange(n) / sr
    env = (np.exp(-t*30) + 0.5 * np.exp(-((t-0.015)*20)**2)
                       + 0.4 * np.exp(-((t-0.030)*20)**2))
    return rng.standard_normal(n) * env * vel * 0.5

def _crash(_freq, n, sr, vel):
    rng = np.random.default_rng(45)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n)
    # 高通（一阶差分）
    noise = np.diff(noise, prepend=0)
    return noise * np.exp(-t*4) * vel * 0.4


SYNTHS = {
    'piano': _piano, 'sine': None, 'saw': None, 'square': None,
    'triangle': None, 'bass': _bass, 'strings': _strings, 'organ': _organ,
    'bell': _bell, 'noise': _noise,
    'kick': _kick, 'snare': _snare, 'hihat': _hihat,
    'tom': _tom, 'clap': _clap, 'crash': _crash,
}

def synth(type_, freq, n, sr, vel, adsr_override=None):
    t = type_.lower()
    if t in ('sine', 'saw', 'square', 'triangle'):
        w = _osc(freq, n, t, sr)
        if adsr_override:
            return _apply_adsr_override(w, n, sr, *adsr_override) * vel
        return w * _adsr(n, 0.005, 0.05, 0.7, min(0.1, n/sr/2), sr) * vel
    fn = SYNTHS.get(t)
    if fn is None:
        return _osc(freq, n, 'sine', sr) * vel
    w = fn(freq, n, sr, vel)
    if adsr_override:
        # 用 override 替换原包络 —— 直接乘回去会双重包络，所以归一化
        env_override = _adsr(n, *adsr_override, sr)
        peak = np.abs(w).max()
        if peak > 0:
            w = w / peak * env_override * vel
    return w


# ═══════════════════════════════════════════════
# 效果器
# ═══════════════════════════════════════════════
def _pan_gains(pan):
    """pan ∈ [-1, 1] → (left_gain, right_gain)"""
    p = max(-1.0, min(1.0, pan))
    a = (p + 1) * np.pi / 4  # 0..π/2
    return np.cos(a), np.sin(a)

def _apply_delay(signal, delay_samples, feedback, mix):
    """单声道 delay"""
    n = len(signal)
    out = signal.copy()
    d = int(delay_samples)
    if d <= 0 or d >= n: return out
    for i in range(d, n):
        out[i] += out[i - d] * feedback
    return (1 - mix) * signal + mix * out

def _make_reverb_ir(room, damp, sr, length=1.5):
    """简化 IR：早期反射 + 指数衰减噪声"""
    n = int(length * sr)
    ir = np.zeros(n)
    # 早期反射
    early_times = [0.007, 0.011, 0.017, 0.023, 0.031, 0.041, 0.053]
    early_gains = [0.9, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3]
    for dt, g in zip(early_times, early_gains):
        idx = int(dt * sr)
        if idx < n: ir[idx] += g * room
    # 后期扩散
    tail = int(0.05 * sr)
    if tail < n:
        rng = np.random.default_rng(7)
        decay = np.exp(-np.arange(n - tail) / sr / (0.15 + room * 1.5))
        ir[tail:] = rng.standard_normal(n - tail) * decay * room * 0.04
    # 阻尼：低通
    if damp > 0:
        k = max(1, int(damp * 60) + 1)
        ir = np.convolve(ir, np.ones(k)/k, mode='same')
    peak = np.abs(ir).max()
    if peak > 0: ir = ir / peak * 0.4
    return ir

def _fft_convolve(a, b):
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, nfft) * np.fft.rfft(b, nfft), nfft)[:n]

def _apply_reverb(signal, room, damp, mix, sr):
    if mix <= 0: return signal
    ir = _make_reverb_ir(room, damp, sr)
    wet = _fft_convolve(signal, ir)[:len(signal)]
    peak = np.abs(wet).max()
    if peak > 0: wet = wet / peak * np.abs(signal).max()
    return (1 - mix) * signal + mix * wet


# ═══════════════════════════════════════════════
# XML 解析
# ═══════════════════════════════════════════════
def _parse_adsr(ne):
    """从 note 元素读 ADSR override，没有就返回 None"""
    a = ne.get('attack'); d = ne.get('decay')
    s = ne.get('sustain'); r = ne.get('release')
    if a is None and d is None and s is None and r is None:
        return None
    return (float(a or 0.005), float(d or 0.1),
            float(s or 0.7), float(r or 0.2))

def _expand_chord(chord_el, bpm, default_dur):
    """<chord pitch="Cmaj7" at="0b" dur="4b" vel="0.4"/>
    → [note_dict, ...]"""
    at = parse_time(chord_el.get('at', '0'), bpm)
    dur = parse_time(chord_el.get('dur', str(default_dur)), bpm)
    vel = float(chord_el.get('vel', '0.5'))
    try:
        pitch_str = chord_el.get('pitch') or chord_el.get('notes')
        octv = int(chord_el.get('octave', '3'))
        pitches = parse_chord(pitch_str, default_oct=octv)
    except Exception as e:
        print(f"  ⚠ chord 解析失败: {e}", file=sys.stderr)
        return []
    adsr = _parse_adsr(chord_el)
    out = []
    for i, p in enumerate(pitches):
        out.append({'at': at, 'pitch': p, 'dur': dur, 'vel': vel,
                    'adsr': adsr})
    return out

def _expand_arp(arp_el, bpm, default_dur):
    """<arp pitch="Cmaj7" at="0b" dur="4b" step="0.5b" order="up"/>"""
    at = parse_time(arp_el.get('at', '0'), bpm)
    dur = parse_time(arp_el.get('dur', str(default_dur)), bpm)
    step = parse_time(arp_el.get('step', '0.5'), bpm)
    vel = float(arp_el.get('vel', '0.5'))
    order = arp_el.get('order', 'up').lower()
    try:
        pitch_str = arp_el.get('pitch') or arp_el.get('notes')
        octv = int(arp_el.get('octave', '4'))
        pitches = parse_chord(pitch_str, default_oct=octv)
    except Exception as e:
        print(f"  ⚠ arp 解析失败: {e}", file=sys.stderr)
        return []
    if order == 'down':
        pitches = list(reversed(pitches))
    elif order == 'updown':
        pitches = pitches + list(reversed(pitches[1:-1]))
    n_steps = max(1, int(round(dur / step)))
    adsr = _parse_adsr(arp_el)
    out = []
    for i in range(n_steps):
        p = pitches[i % len(pitches)]
        out.append({'at': at + i * step, 'pitch': p,
                    'dur': step * 0.95, 'vel': vel, 'adsr': adsr})
    return out

def _expand_pattern(pat_el, bpm):
    """<pattern instrument="kick" at="0b" beat="X...X..." step="0.25b" bars="1"/>"""
    at = parse_time(pat_el.get('at', '0'), bpm)
    step = parse_time(pat_el.get('step', '0.25'), bpm)
    beat = pat_el.get('beat', '')
    bars = int(pat_el.get('bars', '1'))
    vel_strong = float(pat_el.get('vel', '0.9'))
    vel_weak_ratio = float(pat_el.get('weak', '0.5'))
    inst = pat_el.get('instrument')
    adsr = _parse_adsr(pat_el)
    out = []
    for bar in range(bars):
        for i, ch in enumerate(beat):
            if ch == '.': continue
            vel = vel_strong if ch == 'X' else vel_strong * vel_weak_ratio
            out.append({
                'at': at + bar * (len(beat) * step) + i * step,
                'pitch': None, 'dur': step * 0.9,
                'vel': vel, 'adsr': adsr, '_inst_override': inst,
            })
    return out

def parse_xml(path):
    root = ET.parse(path).getroot()
    if root.tag != 'music':
        raise ValueError(f"root 必须是 <music>，实际 <{root.tag}>")
    cfg = {
        'bpm':         float(root.get('bpm', 60)),
        'duration':    float(root.get('duration', 8)),
        'sample_rate': int(root.get('sample_rate', SR_DEFAULT)),
        'loop':        root.get('loop', 'false').lower() == 'true',
        'loop_tail':   float(root.get('loop-tail', 0)),
        'stereo':      root.get('stereo', 'true').lower() == 'true',
        'out':         root.get('out'),
    }
    instruments = {}
    for i in root.findall('instruments/instrument'):
        if i.get('id'):
            instruments[i.get('id')] = i.get('type', 'sine')

    tracks = []
    for tr in root.findall('tracks/track'):
        tid = tr.get('id', f'track{len(tracks)}')
        inst = tr.get('instrument')
        if not inst or inst not in instruments:
            print(f"  ⚠ track {tid} instrument={inst!r} 未定义", file=sys.stderr)
            continue
        volume = float(tr.get('volume', 1.0))
        pan = float(tr.get('pan', 0.0))
        notes = []
        # 遍历所有子元素，支持 note/chord/arp/pattern
        for ne in tr:
            tag = ne.tag
            if tag == 'note':
                adsr = _parse_adsr(ne)
                notes.append({
                    'at':    parse_time(ne.get('at', '0'), cfg['bpm']),
                    'pitch': parse_pitch(ne.get('pitch')),
                    'dur':   parse_time(ne.get('dur', '1'), cfg['bpm']),
                    'vel':   float(ne.get('vel', 1.0)),
                    'adsr':  adsr,
                })
            elif tag == 'chord':
                notes.extend(_expand_chord(ne, cfg['bpm'], 4))
            elif tag == 'arp':
                notes.extend(_expand_arp(ne, cfg['bpm'], 4))
            elif tag == 'pattern':
                notes.extend(_expand_pattern(ne, cfg['bpm']))
            elif tag == 'rest':
                pass  # 显式空拍，方便对齐
        tracks.append({
            'id': tid, 'inst_type': instruments[inst],
            'volume': volume, 'pan': pan, 'notes': notes,
        })
    # 全局效果器
    fx = {'delay': None, 'reverb': None}
    fx_el = root.find('effects')
    if fx_el is not None:
        d = fx_el.find('delay')
        if d is not None:
            fx['delay'] = {
                'time': parse_time(d.get('time', '0.25'), cfg['bpm']),
                'feedback': float(d.get('feedback', '0.3')),
                'mix': float(d.get('mix', '0.2')),
            }
        r = fx_el.find('reverb')
        if r is not None:
            fx['reverb'] = {
                'room': float(r.get('room', '0.5')),
                'damp': float(r.get('damp', '0.4')),
                'mix': float(r.get('mix', '0.2')),
            }
    return cfg, tracks, fx


# ═══════════════════════════════════════════════
# 渲染
# ═══════════════════════════════════════════════
def render(cfg, tracks, fx, out_path):
    sr = cfg['sample_rate']
    tail = cfg['loop_tail'] if cfg['loop'] else 0
    n_total = int((cfg['duration'] + tail) * sr)

    # 渲染每轨到独立 stereo buffer
    all_buffers = []
    for tr in tracks:
        it = tr['inst_type']
        vol = tr['volume']
        pan = tr['pan']
        l_gain, r_gain = _pan_gains(pan)
        buf_l = np.zeros(n_total, dtype=np.float32)
        buf_r = np.zeros(n_total, dtype=np.float32)

        for note in tr['notes']:
            # pattern 可以覆盖 instrument
            inst_type = note.get('_inst_override') or it
            dur = note['dur']
            rdur = dur * (1.5 if inst_type == 'piano' else 1.0)
            if inst_type in ('kick', 'snare', 'hihat', 'tom', 'clap', 'crash'):
                rdur = min(rdur, 1.5)  # 鼓点不延长
            n = int(rdur * sr)
            freq = note_to_freq(note['pitch'])
            w = synth(inst_type, freq, n, sr, note['vel'],
                      adsr_override=note.get('adsr')) * vol
            s = int(note['at'] * sr)
            e = min(s + n, n_total)
            if e > s:
                buf_l[s:e] += w[:e - s] * l_gain
                buf_r[s:e] += w[:e - s] * r_gain
        all_buffers.append((buf_l, buf_r))

    # 混音
    mix_l = np.zeros(n_total, dtype=np.float32)
    mix_r = np.zeros(n_total, dtype=np.float32)
    for bl, br in all_buffers:
        mix_l += bl
        mix_r += br

    # 效果器
    if fx.get('delay'):
        d = fx['delay']
        ds = int(d['time'] * sr)
        mix_l = _apply_delay(mix_l, ds, d['feedback'], d['mix'])
        mix_r = _apply_delay(mix_r, ds * 1.02, d['feedback'], d['mix'])
    if fx.get('reverb'):
        r = fx['reverb']
        mix_l = _apply_reverb(mix_l, r['room'], r['damp'], r['mix'], sr)
        mix_r = _apply_reverb(mix_r, r['room'], r['damp'], r['mix'], sr)

    # 循环叠加
    if cfg['loop'] and tail > 0:
        tn = int(tail * sr)
        mix_l[:tn] += mix_l[-tn:]
        mix_r[:tn] += mix_r[-tn:]
        mix_l = mix_l[:int(cfg['duration'] * sr)]
        mix_r = mix_r[:int(cfg['duration'] * sr)]

    # 归一化
    peak = max(np.abs(mix_l).max(), np.abs(mix_r).max())
    if peak > 0:
        mix_l = mix_l / peak * 0.85
        mix_r = mix_r / peak * 0.85

    # 输出
    if cfg['stereo']:
        interleaved = np.empty(len(mix_l) * 2, dtype=np.int16)
        interleaved[0::2] = np.int16(mix_l * 32767)
        interleaved[1::2] = np.int16(mix_r * 32767)
        n_ch = 2
    else:
        mono = (mix_l + mix_r) * 0.5
        interleaved = np.int16(mono * 32767)
        n_ch = 1

    with wave.open(str(out_path), 'w') as f:
        f.setnchannels(n_ch)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(interleaved.tobytes())


# ═══════════════════════════════════════════════
# MIDI 导出（SMF Type 0）
# ═══════════════════════════════════════════════
def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))

# GM program 映射（0-127）
GM_PROGRAM = {
    'piano':    0,   # Acoustic Grand
    'bass':     32,  # Acoustic Bass
    'strings':  48,  # String Ensemble 1
    'organ':    16,  # Drawbar Organ
    'bell':     14,  # Tubular Bells
    'sine':     80,  # Lead 1 (square) → 用 80 也行
    'saw':      81,  # Lead 2 (sawtooth)
    'square':   80,
    'triangle': 82,  # Lead 3 (calliope)
    'noise':    122, # 通用
}

# GM 鼓位
GM_DRUM = {
    'kick': 36, 'snare': 38, 'hihat': 42,
    'tom': 45, 'clap': 39, 'crash': 49,
}

_DRUM_TYPES = frozenset(GM_DRUM.keys())


def export_midi(cfg, tracks, out_path):
    """把事件转成 SMF Type 0（单 track）"""
    bpm = cfg['bpm']
    tpq = 480  # ticks per quarter
    us_per_beat = int(60_000_000 / bpm)

    # 收集所有事件（含 pattern override）
    events = []  # (tick, 'on'/'off', channel, note, vel)
    for ti, tr in enumerate(tracks):
        ch = ti % 16
        for note in tr['notes']:
            inst_type = note.get('_inst_override') or tr['inst_type']
            # 鼓优先：pitch 由 GM_DRUM 决定，忽略 note['pitch']
            if inst_type in _DRUM_TYPES:
                ch_use = 9
                p = GM_DRUM[inst_type]
            else:
                p = note.get('pitch')
                if p is None or p <= 0:
                    continue
                ch_use = ch
            at_tick = int(note['at'] * bpm / 60 * tpq)
            dur_tick = max(1, int(note['dur'] * bpm / 60 * tpq))
            vel = max(1, min(127, int(note['vel'] * 127)))
            events.append((at_tick, 'on', ch_use, p, vel))
            events.append((at_tick + dur_tick, 'off', ch_use, p, 0))

    events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))

    # 写 track data
    track_data = bytearray()
    # tempo meta
    track_data += _vlq(0) + b'\xFF\x51\x03' + struct.pack('>I', us_per_beat)[1:]
    # time signature 4/4
    track_data += _vlq(0) + b'\xFF\x58\x04\x04\x02\x18\x08'
    # program change 每轨按 inst_type 选 GM program
    for ti, tr in enumerate(tracks):
        if ti == 9: continue
        inst_type = tr['inst_type']
        if inst_type in _DRUM_TYPES:
            continue  # 鼓通道不需要 program change
        prog = GM_PROGRAM.get(inst_type, 0)
        track_data += _vlq(0) + bytes([0xC0 | (ti % 16), prog & 0x7F])
    # note events
    last_tick = 0
    for tick, kind, ch, note, vel in events:
        delta = tick - last_tick
        last_tick = tick
        if kind == 'on':
            track_data += _vlq(delta) + bytes([0x90 | ch, note & 0x7F, vel & 0x7F])
        else:
            track_data += _vlq(delta) + bytes([0x80 | ch, note & 0x7F, 0])
    # end of track
    track_data += _vlq(0) + b'\xFF\x2F\x00'

    # header
    header = b'MThd' + struct.pack('>IHHH', 6, 0, 1, tpq)
    track_chunk = b'MTrk' + struct.pack('>I', len(track_data)) + bytes(track_data)
    with open(out_path, 'wb') as f:
        f.write(header)
        f.write(track_chunk)


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════
def main():
    argv = sys.argv[1:]
    xml = None
    out_override = None
    midi_out = None
    for a in argv:
        if a.startswith('--out='):   out_override = a.split('=', 1)[1]
        elif a.startswith('--midi='): midi_out = a.split('=', 1)[1]
        elif not a.startswith('--'): xml = a
    if not xml:
        print("用法: python xmlmusic.py music.xml [--out=x.wav] [--midi=x.mid]",
              file=sys.stderr)
        return 1

    print("🎵 xmlmusic v0.2")
    try:
        cfg, tracks, fx = parse_xml(xml)
    except Exception as e:
        print(f"✗ XML 解析失败: {e}", file=sys.stderr)
        return 1

    print(f"  bpm={cfg['bpm']}  duration={cfg['duration']}s  "
          f"sr={cfg['sample_rate']}  stereo={cfg['stereo']}  "
          f"loop={cfg['loop']}")
    total = sum(len(t['notes']) for t in tracks)
    print(f"  tracks={len(tracks)}  notes={total}")
    for tr in tracks:
        pan = tr.get('pan', 0.0)
        print(f"    · {tr['id']:8s} inst={tr['inst_type']:8s} "
              f"notes={len(tr['notes']):3d}  vol={tr['volume']:.2f}  "
              f"pan={pan:+.2f}")
    if fx.get('delay'):
        print(f"  fx: delay  time={fx['delay']['time']:.3f}s  "
              f"fb={fx['delay']['feedback']:.2f}  mix={fx['delay']['mix']:.2f}")
    if fx.get('reverb'):
        print(f"  fx: reverb room={fx['reverb']['room']:.2f}  "
              f"damp={fx['reverb']['damp']:.2f}  mix={fx['reverb']['mix']:.2f}")

    out = out_override or cfg['out'] or (Path(xml).stem + '.wav')
    print(f"\n⏳ 渲染 → {out}")
    render(cfg, tracks, fx, out)
    size_kb = Path(out).stat().st_size / 1024
    print(f"✅ {out}  ({size_kb:.1f} KB)")

    if midi_out:
        print(f"\n⏳ 导出 MIDI → {midi_out}")
        try:
            export_midi(cfg, tracks, midi_out)
            size_m = Path(midi_out).stat().st_size
            print(f"✅ {midi_out}  ({size_m} bytes)")
        except Exception as e:
            print(f"✗ MIDI 导出失败: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)