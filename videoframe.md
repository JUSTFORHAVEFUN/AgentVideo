# videoframe

> 给不支持视频多模态的 AI，一个"读懂视频"的入口。

输入一个视频，输出四样东西：

- `contact_sheet.jpg` —— 一张图，包含 N 张关键帧缩略图
- `frames/*.jpg` —— 关键帧原图，右下角带时间戳水印
- `report.json` —— 结构化数据，每帧带时间戳和"为什么抽"
- `summary.txt` —— 人类/AI 可读的时间轴摘要

**任何只能读图/文本的 AI，看这四样，就能复述视频内容。**

---

## 为什么需要这个

大部分 LLM 只能读文本，部分能读图。**真正能读视频的极少**——Gemini 1.5、GPT-4o 有视频接口，但输入昂贵、有长度限制。

videoframe 把视频"翻译"成图像 AI 能读的形式：

```

视频  →  40 张代表帧 + 一张 contact_sheet + JSON
↓
图像 AI 说得出视频讲了什么
纯文本 AI 读 JSON 也知道时间轴

```

---

## 与其他工具的区别

市面上已有类似工具（llm-frames、llm-video-frames、claude-video、peepshow 等），它们大多做"均匀抽帧 → 拼图 → 喂 AI"。

**videoframe 的差异**：**不是均匀抽帧，而是三个维度采样 + 标注理由。**

| 维度 | 触发条件 | 作用 |
|---|---|---|
| `first` | 第一帧 | 保证时间轴起点 |
| `fixed` | 每 N 秒 | 保证时间轴覆盖 |
| `content` | 和上一抽帧的像素差 > 阈值 | 抓"画面整体变了" |
| `focus` | 和上一抽帧的焦点图局部差 > 阈值 | 抓"出现了抢眼的东西" |

**focus 是核心差异**。它基于离散拉普拉斯（边缘密度）+ 颜色异常——不是"画面变了"，而是"画面里出现了抢眼的东西"。

举例：一个讲题视频里，老师在画面里疯狂走动，ffmpeg 的 scene detection 会抓到大量"人在动"的帧；但 videoframe 的 focus 只在**题目切换、画图、出现红色标注**时触发。因为老师走动不增加图像边缘——题目变化才增加。

**每条帧带 reason**：`{t: 16.55, reason: "focus", focus_val: 0.42}`。AI 不仅知道"第 16.55 秒有东西"，还知道"因为焦点异常触发的，异常值 0.42"。

---

## 快速开始

```bash
# 基本用法（默认装进 1280×760）
python videoframe.py --in=video.mp4 --out=keys/

# 摘要模式（推荐，40 帧上限）
python videoframe.py --in=video.mp4 --out=keys/ \
    --every=5.0 --diff-threshold=0.15 --focus-threshold=0.45 \
    --cooldown=1.5 --max-frames=40 --analysis-width=320

# 详细模式（120 帧上限）
python videoframe.py --in=video.mp4 --out=keys/ \
    --every=2.0 --diff-threshold=0.10 --focus-threshold=0.30 \
    --cooldown=0.5 --max-frames=120 --analysis-width=320

# 只抽固定间隔（关闭智能选择）
python videoframe.py --in=video.mp4 --out=keys/ --no-focus

# 不压缩（原始分辨率）
python videoframe.py --in=video.mp4 --out=keys/ \
    --frame-max=0 --frame-w=0 --frame-h=0
```

输出结构：

```
keys/
├── contact_sheet.jpg      # 所有帧缩略图拼成一张
├── report.json            # 结构化数据
├── summary.txt            # 时间轴摘要
└── frames/
    ├── f_0000.00_first.jpg
    ├── f_0004.46_focus.jpg
    ├── f_0011.57_focus.jpg
    ├── f_0016.55_focus.jpg
    ├── f_0021.59_focus.jpg
    └── ...
```

---

参数

抽帧规则

参数 默认 说明
--every=N 2.0 固定间隔秒数
--diff-threshold=X 0.08 内容变化阈值 (0~1)
--focus-threshold=X 0.30 焦点异常阈值 (0~1)
--cooldown=X 0.05 两次抽帧的最小间隔秒数
--no-focus — 关闭焦点规则

数量控制

参数 默认 说明
--max-frames=N 300 上限。超限时按"分段取最高分"裁剪，优先保留 focus

尺寸控制

参数 默认 说明
--frame-w=N 1280 输出帧宽度上限
--frame-h=N 760 输出帧高度上限
--frame-max=N 0 旧接口：长边限制。0 = 使用 frame-w/frame-h

尺寸语义：装进 (frame_w, frame_h) 的边界框，等比缩放，不变形。

· 横版 1920×1080 → 缩到 1280×720
· 竖版 1968×3150 → 缩到 474×760
· 4K 3840×2160 → 缩到 1280×720

性能

参数 默认 说明
--analysis-width=N 320 分析分辨率宽度。320 比 640 快 7 倍，精度影响很小
--analysis-stride=N 2 每 N 帧分析一次。跳帧不影响 0.1s 精度的变化检测

输出

参数 默认 说明
--ext=jpg jpg 帧格式，支持 jpg/png
--no-watermark — 关闭右下角时间戳水印
--watermark-style=bar bar bar（半透明黑底白字）/ text（白字黑描边）
--json=path out/report.json JSON 报告路径
--range=A:B — 只处理指定时间区间

---

自动处理

videoframe 自动处理以下常见问题：

情况 处理
手机录屏 rotation ffprobe 读 rotation → ffmpeg 自动旋转 → 显示尺寸自动修正
VFR / 假 fps ffprobe 报 fps > 240 → 用 nb_frames/duration 推算 → 兜底 30fps
超高清输入 长边 > 边界框 → 等比缩放到边界框内
分析 vs 输出分离 分析用低分辨率（快），输出用高分辨率（可读）

---

上限裁剪策略

当触发帧数超过 --max-frames 时：

1. 按时间均分为 N 段
2. 每段内选得分最高的帧

得分 = reason 基础分 + diff + focus_val

reason 基础分
first 100
focus 3
content 2
fixed 1

首帧永远保留。focus 优先于 content 优先于 fixed。

---

已验证场景

视频 时长 抽帧 特点
3D 喷水动画 12s 27 帧 水流连续变化，focus 密集触发
升国旗 14s 31 帧 阶段性变化，fixed + focus 混合
KARDS 对局 68s 40 帧 竖屏录制（1968×3150 rotation=90）
数学讲题 430s 40 帧 有人物走动，focus 过滤掉"动"只抓"变"

讲题视频是压力测试：7 分钟，老师在画面里"疯狂走动"。结果 40 帧全部是"题目切换、画图、标注"——走动被自动过滤。

输出帧 1216×760，题目文字清晰可读。AI 看图后能说出"初中数学讲题、三角题、圆题、讲题顺序"。

---

设计原则

1. 显式参数 —— 所有规则和阈值都是命令行参数，无"自动猜测"
2. 不做场景特判 —— 没有"如果是讲题视频..."之类的分支
3. 两层处理 —— 分析低分辨率（判断抽不抽）+ 输出高分辨率（读得清）
4. 自动兜底 —— rotation / VFR / 超大分辨率，用户不用管
5. 输出多模态友好 —— 图像（contact_sheet）+ 文本（report/summary）+ 原图（frames）三件套

---

依赖

必需：

· Python 3.10+
· numpy
· Pillow
· ffmpeg（含 ffprobe）

无 agent 依赖，无第三方 API 调用。

---

典型用法：让 AI 读视频

```bash
# 1. 抽帧
python videoframe.py --in=lecture.mp4 --out=lecture_keys/ \
    --every=5.0 --focus-threshold=0.45 --max-frames=40 --analysis-width=320

# 2. 把三样东西发给 AI
#    - lecture_keys/contact_sheet.jpg  (图像 AI)
#    - lecture_keys/report.json        (纯文本 AI)
#    - lecture_keys/summary.txt        (纯文本 AI)

# 3. 问 AI："这个视频讲了什么？"
```

AI 读 contact_sheet.jpg + report.json 后，能复述：

视频 7 分钟。0 秒开场标题"初中数学讲题达人"。9 秒切到第一道题。
23~60 秒讲解第一题。83 秒出现三角形图。134~220 秒长时间在三角形题上。
246 秒开始画圆。263~350 秒圆题讲解。421 秒"感谢大家的聆听"结束。
全程是老师在板书 + 转身写画，题目涉及三角形和圆。

---

文件

· videoframe.py —— 主程序
· videoframe.md —— 本文档

版本

v0.1
