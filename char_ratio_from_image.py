#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色正面图 → 头身比 / 头宽肩宽比
================================

从角色正面立绘自动量出 kigurumi_head_calc.py 需要的那两项：
    --ratio          头身比 N   = 全身高 / 头高（头顶到下巴）
    --head-shoulder  头宽/肩宽 r = 含发头宽 / 肩宽

原理：把角色抠成剪影，逐行统计剪影宽度得到「宽度剖面」。正面站姿的剖面
有一个很稳定的特征——从头顶往下，宽度先随头/头发涨到峰值，再收窄到脖子
（局部极小），然后被肩膀猛地拉宽。据此定位 头顶 / 下巴 / 颈 / 肩 四条线。

⚠ 这是几何启发式，不是识别模型。长发盖住脖子、抱手臂、大裙摆、动态姿势
   都会骗过它。每次都会输出一张标注图，请务必看一眼再用结果；任何一条线
   都可以用 --top-y / --chin-y / --foot-y / --head-w / --shoulder-w 手动覆盖。

依赖
----
    必需  opencv-python（人脸锚定）、Pillow
    可选  rembg        —— AI 抠图，比背景泛洪干净得多
          ultralytics  —— 开 --pose 后用 YOLO 骨架关键点量肩宽

用法
----
    python3 char_ratio_from_image.py 立绘.png
    python3 char_ratio_from_image.py 立绘.jpg --chin-y 340   # 手动改下巴线
    python3 char_ratio_from_image.py 立绘.png --pose         # 肩宽走骨架肩峰点
    python3 char_ratio_from_image.py 立绘.png --json

关于 --pose
-----------
主脚本要的 shoulder_width 是「两肩峰点直线距离」，而剪影法量的是外套外轮廓最宽
处——大衣、毛领、披风能让这两个数差出 2 倍以上，量出来的 r 是错的。--pose 用
YOLO-pose 的肩峰关键点，定义上才对得住。

但这些模型是在真人照片上训的，动漫立绘是域外输入：置信度普遍虚高（实测四张图
全是 0.9+，不同尺寸的模型却能差出 27%）。所以默认会同时跑第二个模型做一致性
交叉验证，不一致就报警告——**模型间是否一致**才是可用的可靠性信号，置信度不是。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

# 人脸锚定只是「剪影法」之上的可选增强：装了 OpenCV 才启用。
# 没装（或装的是移除了 CascadeClassifier 的 OpenCV 5.x）时退化为纯剪影法，
# 主流程会明确打印原因，绝不静默吞掉。
try:  # pragma: no cover - 取决于运行环境
    import cv2  # type: ignore
    import numpy as _np  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    _np = None  # type: ignore

# 缩放时给「较短边」留的最小工作宽度：竖长条/横长条图不至于被压成一条糊线。
MIN_WORK_SIDE = 320


# ---------------------------------------------------------------------------
# 载图（Pillow）
# ---------------------------------------------------------------------------
def load_image(path: str, max_dim: int, use_rembg: bool = True,
               rembg_model: str = "isnet-anime") -> Tuple[int, int, bytearray, float, str]:
    """返回 (工作宽, 工作高, RGBA, 缩放系数 原图/工作图, 说明)。"""
    if not os.path.exists(path):
        raise SystemExit("找不到文件：%s" % path)

    try:
        from PIL import Image  # type: ignore
    except ImportError:
        raise SystemExit("需要 Pillow 才能读图：pip3 install Pillow")

    im = Image.open(path).convert("RGBA")

    rembg_msg = ""
    if use_rembg:
        try:
            import rembg
            # rembg 默认的 u2net 是在真实照片上训的显著性模型，对动漫立绘会吃掉
            # 大片色块（尹琳那张双马尾、披风、半条手臂全被抹掉，头宽量成一半）。
            # isnet-anime 是动漫专训的，同一张图前景多出 88%。
            try:
                from rembg import new_session  # type: ignore
                im = rembg.remove(im, session=new_session(rembg_model))
                rembg_msg = " + rembg 抠图（%s）" % rembg_model
            except Exception:
                im = rembg.remove(im)
                rembg_msg = " + rembg 抠图（回退到默认模型）"
        except ImportError:
            pass

    ow, oh = im.size
    scale = max(1.0, max(ow, oh) / float(max_dim))
    # 竖长条图（如 483×2176）按「最长边→max_dim」会把这窄边压成 177 宽，细节全糊。
    # 给窄边留个下限，否则下巴/肩线会量偏。
    if scale > 1.0:
        scale = max(1.0, min(scale, min(ow, oh) / float(MIN_WORK_SIDE)))
        im = im.resize((max(1, int(ow / scale)), max(1, int(oh / scale))))
    return im.size[0], im.size[1], bytearray(im.tobytes()), scale, "Pillow" + rembg_msg


def save_rgba(path: str, w: int, h: int, rgba: bytearray) -> None:
    """把 RGBA 字节存成 PNG（标注图用）。"""
    from PIL import Image  # type: ignore
    Image.frombytes("RGBA", (w, h), bytes(rgba)).save(path)

# ---------------------------------------------------------------------------
# 抠图
# ---------------------------------------------------------------------------
def _largest_component(mask: bytearray, w: int, h: int) -> Tuple[bytearray, int]:
    """只保留面积最大的前景连通块，返回 (新 mask, 被丢掉的像素数)。

    用来干掉悬浮的特效、蝴蝶、截屏 UI、水印、设定图上散落的配件——这些东西
    会把逐行剪影宽度撑爆，肩宽和脚底线首当其冲。
    """
    n = w * h
    seen = bytearray(n)
    best: List[int] = []
    total = 0
    for i in range(n):
        if not mask[i] or seen[i]:
            continue
        q = deque([i])
        seen[i] = 1
        comp = []
        while q:
            c = q.popleft()
            comp.append(c)
            cx, cy = c % w, c // w
            for nx, ny in ((cx - 1, cy - 1), (cx, cy - 1), (cx + 1, cy - 1),
                           (cx - 1, cy),                   (cx + 1, cy),
                           (cx - 1, cy + 1), (cx, cy + 1), (cx + 1, cy + 1)):
                if 0 <= nx < w and 0 <= ny < h:
                    j = ny * w + nx
                    if mask[j] and not seen[j]:
                        seen[j] = 1
                        q.append(j)
        total += len(comp)
        if len(comp) > len(best):
            best = comp
    if not best:
        return mask, 0
    out = bytearray(n)
    for idx in best:
        out[idx] = 1
    return out, total - len(best)


def build_mask(w: int, h: int, rgba: bytearray, alpha_thr: int,
               bg_tol: int) -> Tuple[bytearray, str]:
    """返回 (mask, 用的哪种方式)。mask[i] = 1 表示角色像素。"""
    n = w * h

    # 有真正的透明区域 → 直接用 alpha，最可靠
    transparent = 0
    for i in range(3, min(len(rgba), n * 4), 4 * 97):   # 抽样看有没有透明
        if rgba[i] < 250:
            transparent += 1
            if transparent > 8:
                break
    if transparent > 8:
        mask = bytearray(n)
        for i in range(n):
            if rgba[i * 4 + 3] >= alpha_thr:
                mask[i] = 1
        # rembg 的输出经常带零星碎块（漂浮的蝴蝶/特效、设定图上散落的配件），
        # 以前只有背景泛洪那条分支做了这一步，alpha 分支漏了
        mask, dropped = _largest_component(mask, w, h)
        note = "alpha 通道"
        if dropped:
            note += "（丢掉 %d px 的悬浮碎块）" % dropped
        return mask, note

    # 不透明图 → 从四边泛洪，把与边框同色的连通区判为背景
    counts: Dict[Tuple[int, int, int], int] = {}
    for x in range(w):
        for y in (0, h - 1):
            i = (y * w + x) * 4
            k = (rgba[i] >> 4, rgba[i + 1] >> 4, rgba[i + 2] >> 4)
            counts[k] = counts.get(k, 0) + 1
    for y in range(h):
        for x in (0, w - 1):
            i = (y * w + x) * 4
            k = (rgba[i] >> 4, rgba[i + 1] >> 4, rgba[i + 2] >> 4)
            counts[k] = counts.get(k, 0) + 1
    bk = max(counts, key=lambda k: counts[k])
    br, bg_, bb = bk[0] * 16 + 8, bk[1] * 16 + 8, bk[2] * 16 + 8

    mask = bytearray(b"\x01") * n
    seen = bytearray(n)
    q: deque = deque()

    def close(i: int) -> bool:
        p = i * 4
        return (abs(rgba[p] - br) <= bg_tol and abs(rgba[p + 1] - bg_) <= bg_tol
                and abs(rgba[p + 2] - bb) <= bg_tol)

    for x in range(w):
        for y in (0, h - 1):
            i = y * w + x
            if not seen[i] and close(i):
                seen[i] = 1
                q.append(i)
    for y in range(h):
        for x in (0, w - 1):
            i = y * w + x
            if not seen[i] and close(i):
                seen[i] = 1
                q.append(i)

    while q:
        i = q.popleft()
        mask[i] = 0
        x, y = i % w, i // w
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h:
                j = ny * w + nx
                if not seen[j] and close(j):
                    seen[j] = 1
                    q.append(j)

    mask, dropped = _largest_component(mask, w, h)
    _ = dropped

    return mask, "背景泛洪 + 最大连通块提取（背景色 RGB≈%d,%d,%d）" % (br, bg_, bb)


# ---------------------------------------------------------------------------
# 人脸锚定（可选增强，需要 OpenCV）
# ---------------------------------------------------------------------------
def face_backend_note() -> str:
    """返回当前环境下人脸锚定是否可用的说明；可用时返回空串。"""
    if cv2 is None or _np is None:
        return ("未安装 opencv-python：人脸锚定不可用，将退回纯剪影法"
                "（长发/兽耳会明显拉低下巴线精度）。装：pip install \"opencv-python<5.0.0\"")
    if not hasattr(cv2, "CascadeClassifier"):
        return ("当前 OpenCV %s 已移除 cv2.CascadeClassifier（5.x 起不再提供），人脸锚定不可用，"
                "将退回纯剪影法。请改装 pip install \"opencv-python<5.0.0\"，或用 --chin-y 手动给下巴线"
                % getattr(cv2, "__version__", "?"))
    return ""


def detect_faces(w: int, h: int, rgba: bytearray) -> Tuple[Optional[List[Tuple[int, int, int, int]]], str]:
    """用动漫人脸级联找脸框。返回 (按 y 从上到下排序的框列表 或 None, 提示信息)。

    返回 None 表示没拿到脸框；提示信息解释原因（绝不会静默失败）。
    """
    note = face_backend_note()
    if note:
        return None, note

    cascade_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "lbpcascade_animeface.xml")
    if not os.path.exists(cascade_path):
        return None, "找不到 lbpcascade_animeface.xml，人脸锚定不可用，退回纯剪影法"

    try:
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return None, "lbpcascade_animeface.xml 加载失败（文件损坏？），退回纯剪影法"
        nparr = _np.frombuffer(bytes(rgba), dtype=_np.uint8).reshape((h, w, 4))
        gray = cv2.cvtColor(nparr, cv2.COLOR_RGBA2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(24, 24))
    except Exception as e:      # 关键：不静默。把真实异常带出去，别让它悄悄退化成垃圾结果。
        return None, "人脸检测出错（%s: %s），退回纯剪影法" % (type(e).__name__, e)

    if len(faces) == 0:
        return None, "人脸检测没找到任何脸框，退回纯剪影法（下巴线可能不准，必要时 --chin-y 手动给）"
    return sorted([tuple(int(v) for v in f) for f in faces], key=lambda f: f[1]), ""


# ---------------------------------------------------------------------------
# 姿态估计（可选）：YOLO-pose 给骨架关键点
# ---------------------------------------------------------------------------
# COCO-17 里我们只关心这四个点
KP_LSHO, KP_RSHO, KP_LANK, KP_RANK = 5, 6, 15, 16


def detect_pose(path: str, scale: float, models: List[str],
                min_conf: float = 0.35) -> Optional[Dict[str, Any]]:
    """跑 YOLO-pose，返回工作图坐标系下的肩/踝关键点。装不上就返回 None。

    为什么值得引入：kigurumi_head_calc.py 要的肩宽是「两肩峰点直线距离」，
    而剪影法量的是外套外轮廓最宽处——大衣、毛领、披风会让这两个数差出好几倍。
    骨架关键点给的正好是肩峰，两边定义这才对得上。

    ⚠ 这些模型是在真人照片上训的，动漫立绘属于域外输入：置信度普遍虚高
      （四张测试图全是 0.9+，但不同尺寸的模型能差出 27%）。所以这里同时跑
      两个模型做一致性交叉验证——**模型间是否一致**才是可用的可靠性信号，
      置信度不是。
    """
    try:
        from ultralytics import YOLO  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        return None

    src = Image.open(path)
    # 透明背景直接 convert("RGB") 会变成黑底，姿态模型在黑底上容易误判，先铺白。
    if src.mode in ("RGBA", "LA") or (src.mode == "P" and "transparency" in src.info):
        rgba = src.convert("RGBA")
        im = Image.new("RGB", rgba.size, (255, 255, 255))
        im.paste(rgba, mask=rgba.split()[-1])
    else:
        im = src.convert("RGB")
    # 小图会让关键点乱跳（实测 290×410 那张，放大到 3× 才收敛），先垫到 900px
    mult = 1
    while max(im.size) * mult < 900:
        mult += 1
    src_im = im.resize((im.width * mult, im.height * mult), Image.LANCZOS) if mult > 1 else im

    runs: List[Dict[str, Any]] = []
    for name in models:
        try:
            res = YOLO(name)(src_im, verbose=False)[0]
        except Exception:
            continue
        if res.keypoints is None or len(res.keypoints.xy) == 0:
            continue
        # 多个人时取框最大的那个
        idx = 0
        if res.boxes is not None and len(res.boxes.xyxy) > 1:
            areas = [(float(b[2] - b[0]) * float(b[3] - b[1]))
                     for b in res.boxes.xyxy.cpu().numpy()]
            idx = max(range(len(areas)), key=lambda i: areas[i])
        xy = res.keypoints.xy[idx].cpu().numpy() / (mult * scale)
        cf = res.keypoints.conf[idx].cpu().numpy()
        runs.append({"model": name, "xy": xy, "conf": cf})

    if not runs:
        return None

    main_run = runs[0]
    xy, cf = main_run["xy"], main_run["conf"]
    out: Dict[str, Any] = {"model": main_run["model"], "warnings": []}

    if cf[KP_LSHO] >= min_conf and cf[KP_RSHO] >= min_conf:
        lx, rx = float(xy[KP_LSHO][0]), float(xy[KP_RSHO][0])
        out["shoulder_w"] = abs(lx - rx)
        out["shoulder_span"] = (int(min(lx, rx)), int(max(lx, rx)))
        out["shoulder_y"] = int((xy[KP_LSHO][1] + xy[KP_RSHO][1]) / 2)
    if cf[KP_LANK] >= min_conf and cf[KP_RANK] >= min_conf:
        out["ankle_y"] = float((xy[KP_LANK][1] + xy[KP_RANK][1]) / 2)
        out["ankle_x"] = (float(min(xy[KP_LANK][0], xy[KP_RANK][0])),
                          float(max(xy[KP_LANK][0], xy[KP_RANK][0])))

    # 交叉验证：第二个模型的肩宽差超过 15% 就别信这个数
    if "shoulder_w" in out and len(runs) > 1:
        alt = runs[1]
        if alt["conf"][KP_LSHO] >= min_conf and alt["conf"][KP_RSHO] >= min_conf:
            aw = abs(float(alt["xy"][KP_LSHO][0]) - float(alt["xy"][KP_RSHO][0]))
            ref = max(out["shoulder_w"], aw, 1.0)
            if abs(aw - out["shoulder_w"]) / ref > 0.15:
                out["warnings"].append(
                    "两个姿态模型对肩宽意见不合（%s %.0f px vs %s %.0f px，差 %.0f%%）——"
                    "动漫立绘是这类模型的域外输入，置信度再高也不代表对，"
                    "务必看标注图，必要时 --shoulder-w 手动给"
                    % (main_run["model"], out["shoulder_w"], alt["model"], aw,
                       abs(aw - out["shoulder_w"]) / ref * 100))
    return out


# ---------------------------------------------------------------------------
# 测量
# ---------------------------------------------------------------------------
def _run_at(mask: bytearray, w: int, y: int, x: int) -> Optional[Tuple[int, int]]:
    """返回第 y 行里覆盖住 x 的那一段连续前景 (起, 止)；x 落在背景上则 None。"""
    base = y * w
    if not mask[base + x]:
        return None
    lo = x
    while lo > 0 and mask[base + lo - 1]:
        lo -= 1
    hi = x
    while hi < w - 1 and mask[base + hi + 1]:
        hi += 1
    return lo, hi


def analyse(w: int, h: int, mask: bytearray, ov: Dict[str, Optional[float]],
            shoulder_band: float, face_rects=None,
            pose: Optional[Dict[str, Any]] = None,
            head_cap: float = 1.45) -> Dict[str, Any]:
    warn: List[str] = []
    if pose:
        warn.extend(pose.get("warnings", []))

    left = [0] * h
    right = [0] * h
    width = [0] * h
    for y in range(h):
        base = y * w
        lo = -1
        hi = -1
        for x in range(w):
            if mask[base + x]:
                if lo < 0:
                    lo = x
                hi = x
        left[y], right[y] = lo, hi
        width[y] = (hi - lo + 1) if lo >= 0 else 0

    rows = [y for y in range(h) if width[y] > 0]
    if not rows:
        raise SystemExit("剪影是空的：抠图失败。试试 --bg-tol 调大，或用带透明背景的 PNG。")
    top_y, foot_y = rows[0], rows[-1]

    # 脚底线：裸剪影的最后一行经常是裙摆/披风/半透明雾，不是鞋底。有踝关键点时
    # 把搜索限制在「两踝 ± 30% 踝距」这条腿走廊里，横向摊开的布料就吃不进来。
    # 老实说这一步收益有限（实测只救回几个像素，披风垂在两腿之间时完全无效），
    # 真正有用的是下面那条警告。
    foot_src = "剪影最低行"
    if pose and "ankle_x" in pose:
        ax0, ax1 = pose["ankle_x"]
        mg = (ax1 - ax0) * 0.30 + 4
        lo, hi = max(0, int(ax0 - mg)), min(w - 1, int(ax1 + mg))
        y0 = max(0, int(pose["ankle_y"]))
        for y in range(h - 1, y0 - 1, -1):
            base = y * w
            if any(mask[base + x] for x in range(lo, hi + 1)):
                foot_y = y
                foot_src = "踝点走廊内的最低行"
                break
    
    # --- 挑一个可信的人脸框 ---
    # lbpcascade 会在盔甲、褶皱、装饰纹样上报假脸。yinlin_cropped 那张就是典型：
    # 真脸一个都没检出来，反倒在胯甲上报了一个 28×28 的框，下巴线直接钉到大腿根。
    # 两条解剖学约束足够筛掉这类误检：
    #   ① 立姿全身图里，脸绝不会出现在身体高度 40% 以下
    #   ② 有骨架时更硬：下巴一定在肩峰线上方
    face_rect = None
    if face_rects is not None and len(face_rects):
        limit = top_y + (foot_y - top_y + 1) * 0.40
        sho_y = pose.get("shoulder_y") if pose else None
        for cand in face_rects:
            cy, ch = int(cand[1]), int(cand[3])
            if cy + ch > limit:
                continue
            if sho_y is not None and cy + ch * 0.70 > sho_y + ch * 0.10:
                continue
            face_rect = cand
            break
        if face_rect is None:
            warn.append("检到 %d 个人脸框，没有一个在合理位置（脸跑到身体下半部或肩线以下）——"
                        "当成误检丢掉了，改用轮廓几何找颈，结果务必对着标注图确认"
                        % len(face_rects))

    # --- 核心：基于面部识别锚定人类头骨（无视兽耳/呆毛/帽子） ---
    chin_src = "检不出，回退到颈最窄线"
    if face_rect is not None:
        # cv2 给的是 numpy int32，落进结果字典会让 json.dumps 直接炸
        fx, fy, fw, fh = (int(v) for v in face_rect)
        # lbpcascade_animeface 给的框比「额头→下巴」大一圈：框顶落在发际线附近，
        # 框底压到锁骨/衣领上。实测标定（Amiya / Texas / SilverAsh）：
        #     头骨顶 ≈ fy - 0.25 × fh      下巴 ≈ fy + 0.70 × fh
        # 直接拿 fy+fh 当下巴，下巴线会掉到衣领上，头高被撑大、头宽还会量到肩上去。
        # 头顶用两个估计取「更靠下」的那个：
        #   ① 先验 fy - 0.25×fh —— 兽耳/呆毛不会骗它，但发量少时会顶到头骨上方
        #   ② 剪影上扫 —— 从框顶往上走，只跟着「过脸中线且宽度 ≥ 0.45×fw」的那
        #      一段连通行走，呆毛/触角太细会被甩掉；但兽耳跟头连在一起时甩不掉
        # 两者的失败方向正好相反，取 max 就等于各自补对方的漏。
        cx = fx + fw // 2
        thr = fw * 0.45
        top_scan = fy
        yy = fy
        while yy >= 0:
            run = _run_at(mask, w, yy, cx)
            if run is None or (run[1] - run[0] + 1) < thr:
                break
            top_scan = yy
            yy -= 1
        top_y = max(0, int(fy - fh * 0.25), top_scan)
        chin_y = min(h - 1, int(fy + fh * 0.70))
        chin_src = "OpenCV 面部锚定 (无视兽耳/帽子)"
        # 颈线只用于画图和兜底，取下巴下方 0.15 头高
        neck_y = min(h - 1, chin_y + max(1, int((chin_y - top_y + 1) * 0.15)))
        neck_w = width[neck_y]
    else:
        # 老的纯剪影逻辑（容易被兽耳和长发干扰）
        if ov.get("top_y") is not None:
            top_y = int(ov["top_y"])
        
        b0 = top_y + max(2, int((foot_y - top_y) * 0.06))
        b1 = top_y + int((foot_y - top_y) * 0.55)
        b1 = min(b1, h - 1)
        neck_y = min(range(b0, b1 + 1), key=lambda y: width[y] if width[y] else 10 ** 9)
        neck_w = width[neck_y]
        head_top_max = max(width[top_y:neck_y + 1]) if neck_y > top_y else width[top_y]

        if neck_w >= head_top_max * 0.80:
            warn.append("找不到明显的颈部收窄，多半是长发遮挡。")
            auto_chin = neck_y
        else:
            widest_y = max(range(top_y, neck_y + 1), key=lambda y: width[y])
            thr = neck_w * 1.08
            auto_chin = neck_y
            for y in range(widest_y, neck_y + 1):
                if width[y] <= thr:
                    auto_chin = y
                    break
        chin_y = auto_chin
        chin_src = "轮廓几何寻颈"

    # 手动覆盖
    if ov.get("top_y") is not None:
        top_y = int(ov["top_y"])
    if ov.get("foot_y") is not None:
        foot_y = int(ov["foot_y"])
    if ov.get("chin_y") is not None:
        chin_y = int(ov["chin_y"])
        chin_src = "手动指定"
        
    head_h = chin_y - top_y + 1
    total = foot_y - top_y + 1

    # --- 头宽：含发头宽（与主脚本 r 的定义一致）---
    # 下颌两侧的垂发、立领、兜帽会在接近下巴处急速外扩，所以搜索带砍掉头部最下
    # 面 12%，只在真正的颅骨段里取最大剪影宽。
    hw_hi = max(top_y, chin_y - int(head_h * 0.12))
    hw_y = max(range(top_y, hw_hi + 1), key=lambda y: width[y])
    head_w = float(width[hw_y])
    head_span = (left[hw_y], right[hw_y])
    if face_rect is not None:
        # 大毛领 / 双马尾能让剪影宽出头骨好几倍，用人脸框宽度封顶。
        # 系数太小会把宽头发（银灰那种）也一起削掉，所以默认放到 1.45，可用 --head-cap 调。
        cap = float(face_rect[2]) * head_cap
        if head_w > cap:
            head_w = cap
            cx = face_rect[0] + face_rect[2] / 2.0
            head_span = (int(cx - cap / 2), int(cx + cap / 2))
    if ov.get("head_w") is not None:
        head_w = float(ov["head_w"])
        cx = (head_span[0] + head_span[1]) / 2.0
        head_span = (int(cx - head_w / 2), int(cx + head_w / 2))

    # --- 肩宽：先找「肩线」再量，而不是在带里取中位宽度 ---
    # 下巴往下，剪影先经过脖子/衣领，再在肩膀处快速外扩。取带内宽度增长最快的那一行
    # 当肩线。比「带内中位宽」更贴近真正的肩点，也不会一路滑到外套下摆 / 披风最宽处。
    s0 = min(h - 1, chin_y + max(2, int(head_h * 0.25)))
    s1 = min(h - 1, chin_y + max(4, int(head_h * shoulder_band)))
    if s1 <= s0:
        s1 = min(h - 1, s0 + 3)
    sw = max(1, int(head_h * 0.06))              # 平滑一下，免得噪声行冒充肩线
    sm_w = []
    for y in range(s0, s1 + 1):
        lo = max(0, y - sw)
        hi = min(h - 1, y + sw)
        sm_w.append(sum(width[lo:hi + 1]) / float(hi - lo + 1))
    grad = [sm_w[i + 1] - sm_w[i] for i in range(len(sm_w) - 1)]
    sh_y = s0 + (max(range(len(grad)), key=lambda i: grad[i]) if grad else 0)
    shoulder_w = float(width[sh_y])
    shoulder_span = (left[sh_y], right[sh_y])
    shoulder_src = "剪影外轮廓·肩线（≠ 肩峰距，厚衣服会明显偏大）"
    shoulder_sil_w = shoulder_w                 # 剪影估计值，骨架不可信时的兜底

    # 骨架肩宽：定义上对（肩峰距），但动漫立绘是姿态模型的域外输入，实测常把肩点
    # 定到脖子 / 衣领上（值偏小）。用两条下限把关，判为不可信就回退剪影并告警，
    # 两个值都报出来交给用户定夺。
    pose_shoulder = float(pose["shoulder_w"]) if (pose and "shoulder_w" in pose) else None
    pose_shoulder_span = pose["shoulder_span"] if pose_shoulder is not None else None
    pose_shoulder_y = None
    used_pose = False
    if pose_shoulder is not None:
        body_h = max(1, foot_y - top_y + 1)
        span_h = pose_shoulder / body_h
        span_head = pose_shoulder / max(1.0, head_w)
        pose_shoulder_y = max(0, min(h - 1, int(pose["shoulder_y"])))
        # 两条下限都是「肩峰距」的硬约束：不能比含发头宽还窄，也不能太小于身高。
        # 动漫立绘里姿态模型最常见的失败就是把肩点定到脖子 / 衣领上（值偏小），
        # 这两条正好卡住那种情况；宽肩厚衣（值偏大）交给剪影去兜。
        plausible = (0.13 <= span_h <= 0.35) and (1.20 <= span_head <= 3.0)
        if plausible:
            shoulder_w = pose_shoulder
            shoulder_span = pose_shoulder_span
            sh_y = pose_shoulder_y
            shoulder_src = "YOLO-pose 肩峰关键点 (%s)" % pose["model"]
            used_pose = True
        else:
            warn.append(
                "骨架肩宽 %.0f px（身高的 %.0f%%、含发头宽的 %.2f 倍）不像正常肩峰距"
                "（肩应明显宽于含发头宽，且不低于身高的一成多）——动漫立绘是姿态模型的域外输入，"
                "这次多半把肩点定到了脖子 / 衣领上。已改用剪影肩线 %.0f px；"
                "要骨架值就 --shoulder-w %.0f，或对着标注图手动量"
                % (pose_shoulder, span_h * 100, span_head, shoulder_w, pose_shoulder))

    if ov.get("shoulder_w") is not None:
        shoulder_w = float(ov["shoulder_w"])
        cx = (shoulder_span[0] + shoulder_span[1]) / 2.0
        shoulder_span = (int(cx - shoulder_w / 2), int(cx + shoulder_w / 2))

    n_ratio = total / float(head_h)
    r_ratio = head_w / float(shoulder_w) if shoulder_w else float("nan")

    # 下巴线是整套方法里最不可靠的一环：它一定落在 [下巴, 颈最窄] 之间的某处。
    # 把这个区间对应的头身比范围报出来，让不确定度显式可见。
    n_alt = total / float(max(1, neck_y - top_y + 1))
    n_lo, n_hi = min(n_ratio, n_alt), max(n_ratio, n_alt)

    if not (1.8 <= n_ratio <= 9.5):
        warn.append("头身比算出来 %.2f，超出常见范围 1.8~9.5，几乎肯定是分割错了" % n_ratio)
    if not (0.35 <= r_ratio <= 1.40):
        warn.append("头宽/肩宽算出来 %.2f，超出常见范围 0.35~1.40，检查肩线" % r_ratio)
    if pose and "ankle_y" in pose and foot_y - pose["ankle_y"] > head_h:
        warn.append("脚底线在踝关键点下方 %.0f px（超过一个头高 %d px）——"
                    "多半量到了裙摆/披风/半透明雾，不是鞋底，用 --foot-y 手动压"
                    % (foot_y - pose["ankle_y"], head_h))
    # 这条只在剪影模式下有意义：剪影量的是外轮廓，头和肩是同一套定义，肩反而更窄
    # 就说明肩线被头发盖了。骨架模式下肩宽是肩峰距、头宽是含发外轮廓，本来就不是
    # 同一套尺——大头身的动漫角色 head_w > shoulder_w 是正常的（上面的 r 区间会兜底）。
    if shoulder_w <= head_w and not used_pose:
        warn.append("肩比头还窄（肩 %d px ≤ 头 %d px）——长发把肩线盖住了，"
                    "量到的多半是头发不是肩膀，建议 --shoulder-w 手动给"
                    % (shoulder_w, head_w))
    if (shoulder_w > head_w * 2.2 and ov.get("shoulder_w") is None
            and not used_pose):
        warn.append("肩宽 %.0f px 是头宽 %.0f px 的 %.1f 倍——多半量到了毛领/披风/翅膀这类"
                    "外挂物，不是肩峰距，建议 --pose 或 --shoulder-w 手动给"
                    % (shoulder_w, head_w, shoulder_w / max(1.0, head_w)))
    # 画面裁切检测：竖长条 / 截图裁出来的立绘，头或肩会被画框切掉，量到的是「画框宽」
    # 而不是角色宽。这种图自动值没意义，必须手动给或换未裁切的全身图。
    if left[hw_y] <= 0 or right[hw_y] >= w - 1:
        warn.append("头宽那一行顶到了画面左右边缘——立绘被横向裁切了，头宽量到的是画框宽、偏小；"
                    "建议换未裁切的全身图，或用 --head-w 手动给")
    if not used_pose and (left[sh_y] <= 0 or right[sh_y] >= w - 1):
        warn.append("肩宽那一行顶到了画面左右边缘——立绘被横向裁切了，肩宽量到的是画框宽，"
                    "不是角色肩宽；请用 --shoulder-w 手动给，或换未裁切的全身图")

    return {
        "top_y": top_y, "chin_y": chin_y, "neck_y": neck_y, "foot_y": foot_y,
        "head_w_y": hw_y, "shoulder_y": sh_y,
        "total_px": total, "head_h_px": head_h,
        "head_w_px": head_w, "shoulder_w_px": shoulder_w, "neck_w_px": neck_w,
        "head_span": head_span,
        "shoulder_span": shoulder_span,
        "shoulder_sil_px": shoulder_sil_w,
        "pose_shoulder_px": pose_shoulder,
        "pose_shoulder_span": pose_shoulder_span,
        "pose_shoulder_y": pose_shoulder_y,
        "used_pose": used_pose,
        "ratio": n_ratio, "head_shoulder": r_ratio,
        "ratio_lo": n_lo, "ratio_hi": n_hi,
        "width_profile": width, "left": left, "right": right,
        "warnings": warn,
        "chin_src": "手动指定" if ov.get("chin_y") is not None else chin_src,
        "foot_src": "手动指定" if ov.get("foot_y") is not None else foot_src,
        "shoulder_src": "手动指定" if ov.get("shoulder_w") is not None else shoulder_src,
    }


# ---------------------------------------------------------------------------
# 标注图
# ---------------------------------------------------------------------------
RED = (232, 62, 62)
BLUE = (58, 132, 232)
GREEN = (46, 184, 106)
ORANGE = (240, 150, 40)
GRAY = (150, 150, 150)
PURPLE = (150, 60, 200)


def annotate(w: int, h: int, rgba: bytearray, m: Dict[str, Any]) -> bytearray:
    out = bytearray(rgba)

    def px(x: int, y: int, c: Tuple[int, int, int], a: float = 1.0) -> None:
        if not (0 <= x < w and 0 <= y < h):
            return
        i = (y * w + x) * 4
        if out[i + 3] < 255:                       # 透明处先铺白，线才看得见
            out[i] = out[i + 1] = out[i + 2] = 255
            out[i + 3] = 255
        for k in range(3):
            out[i + k] = int(out[i + k] * (1 - a) + c[k] * a)

    def hline(y: int, x0: int, x1: int, c, thick: int = 2) -> None:
        for t in range(thick):
            for x in range(max(0, x0), min(w, x1 + 1)):
                px(x, y + t, c)

    def vtick(x: int, y: int, c, half: int = 6) -> None:
        for yy in range(y - half, y + half + 1):
            px(x, yy, c)
            px(x + 1, yy, c)

    # 全身范围（红=头顶/下巴，蓝=脚底）
    hline(m["top_y"], 0, w - 1, RED)
    hline(m["chin_y"], 0, w - 1, RED)
    hline(m["foot_y"], 0, w - 1, BLUE)
    hline(m["neck_y"], 0, w - 1, GRAY, 1)

    # 头宽（绿）与肩宽（橙）
    hl, hr = m["head_span"]
    hline(m["head_w_y"], hl, hr, GREEN, 3)
    vtick(hl, m["head_w_y"], GREEN)
    vtick(hr, m["head_w_y"], GREEN)

    sl, sr = m["shoulder_span"]
    hline(m["shoulder_y"], sl, sr, ORANGE, 3)
    vtick(sl, m["shoulder_y"], ORANGE)
    vtick(sr, m["shoulder_y"], ORANGE)

    # 骨架肩线（紫色）：即便最终判定不可信也画出来，方便和橙色的剪影肩线对照
    if m.get("pose_shoulder_span"):
        pl, pr = m["pose_shoulder_span"]
        py = m["pose_shoulder_y"]
        hline(py, pl, pr, PURPLE, 3)
        vtick(pl, py, PURPLE)
        vtick(pr, py, PURPLE)

    # 左边缘画宽度剖面曲线，方便看分割对不对
    prof = m["width_profile"]
    mx = max(prof) or 1
    for y in range(h):
        x = int(prof[y] / float(mx) * 70)
        for k in range(2):
            px(x + k, y, GRAY, 0.75)
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description="从角色正面立绘量出 头身比 与 头宽/肩宽",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="量完会直接给出可粘贴到 kigurumi_head_calc.py 的参数。",
    )
    p.add_argument("image", help="角色正面全身图（png/jpg/webp…，带透明背景最准）")
    p.add_argument("--max-dim", type=int, default=800,
                   help="内部工作分辨率上限，越大越慢越精细 (默认 800)")
    p.add_argument("--alpha-thr", type=int, default=128, help="alpha 二值化阈值")
    p.add_argument("--bg-tol", type=int, default=30,
                   help="无透明通道时的背景色容差，抠不干净就调大 (默认 30)")
    p.add_argument("--shoulder-band", type=float, default=0.6,
                   help="肩线搜索带的下沿 = 下巴下方 N × 头高，上沿固定 0.25 (默认 0.6)")
    p.add_argument("--head-cap", type=float, default=1.45,
                   help="头宽封顶 = 人脸框宽 × 它，防止毛领/双马尾把剪影头宽撑爆 (默认 1.45；"
                        "宽发角色可调大，窄脸可调小)")

    g = p.add_argument_group("手动覆盖（像素值按原图坐标）")
    g.add_argument("--top-y", type=float, help="头顶线 y")
    g.add_argument("--chin-y", type=float, help="下巴线 y")
    g.add_argument("--foot-y", type=float, help="脚底线 y")
    g.add_argument("--head-w", type=float, help="头宽 px")
    g.add_argument("--shoulder-w", type=float, help="肩宽 px")

    p.add_argument("--out", help="标注图输出路径 (默认 <原名>_analysis.png)")
    p.add_argument("--no-annotate", action="store_true", help="不输出标注图")
    p.add_argument("--pose", action="store_true",
                   help="用 YOLO-pose 的肩峰关键点量肩宽、用踝点约束脚线"
                        "（需 pip install ultralytics；没装会自动退回剪影法）")
    p.add_argument("--pose-model", default="yolo11s-pose.pt",
                   help="主姿态模型 (默认 yolo11s-pose.pt；实测 x 号在动漫图上不一定更准)")
    p.add_argument("--pose-check", default="yolo11n-pose.pt",
                   help="做一致性交叉验证的第二个模型，none 关闭 (默认 yolo11n-pose.pt)")
    p.add_argument("--no-rembg", action="store_true", help="禁用 rembg AI 抠图（即使系统已安装该库）")
    p.add_argument("--rembg-model", default="isnet-anime",
                   help="rembg 模型，动漫立绘务必用 isnet-anime；u2net 会吃掉"
                        "双马尾/披风这类大色块 (默认 isnet-anime)")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    a = p.parse_args(argv)

    w, h, rgba, scale, how = load_image(a.image, a.max_dim, not a.no_rembg, a.rembg_model)
    mask, mhow = build_mask(w, h, rgba, a.alpha_thr, a.bg_tol)

    # 覆盖值是原图坐标，换算到工作图坐标
    ov: Dict[str, Optional[float]] = {}
    for k in ("top_y", "chin_y", "foot_y", "head_w", "shoulder_w"):
        v = getattr(a, k)
        ov[k] = None if v is None else v / scale

    
    # --- 人脸锚定（可选增强；拿不到就明确说明，不静默退化）---
    face_rects, face_note = detect_faces(w, h, rgba)
    if face_note:
        print("⚠ " + face_note, file=sys.stderr)

    pose = None
    if a.pose:
        models = [a.pose_model]
        if a.pose_check and a.pose_check.lower() != "none":
            models.append(a.pose_check)
        pose = detect_pose(a.image, scale, models)
        if pose is None:
            print("⚠ --pose 打开了但没能跑起来：装一下 ultralytics 再试\n"
                  "  pip3 install --user ultralytics\n"
                  "  这次先按剪影法算。\n", file=sys.stderr)

    m = analyse(w, h, mask, ov, a.shoulder_band, face_rects, pose, a.head_cap)
    if face_note:
        # 人脸锚定失败是头高误差的主因，把它顶到结果警告最前面
        m["warnings"].insert(0, face_note)

    out_path = None
    if not a.no_annotate:
        out_path = a.out or (os.path.splitext(a.image)[0] + "_analysis.png")
        save_rgba(out_path, w, h, annotate(w, h, rgba, m))

    result = {
        "图片": a.image,
        "读取方式": how,
        "抠图方式": mhow,
        "工作分辨率": [w, h],
        "缩放系数_原图除以工作图": round(scale, 3),
        "关键线_工作图y": {
            "头顶": m["top_y"], "下巴": m["chin_y"],
            "颈最窄": m["neck_y"], "脚底": m["foot_y"],
        },
        "测量_工作图px": {
            "全身高": m["total_px"], "头高": m["head_h_px"],
            "头宽": round(m["head_w_px"], 1), "肩宽": round(m["shoulder_w_px"], 1),
            "颈宽": m["neck_w_px"],
        },
        "结果": {
            "头身比": round(m["ratio"], 2),
            "头身比不确定区间": [round(m["ratio_lo"], 2), round(m["ratio_hi"], 2)],
            "头宽肩宽比": round(m["head_shoulder"], 3),
        },
        "下巴线来源": m["chin_src"],
        "脚底线来源": m["foot_src"],
        "肩宽来源": m["shoulder_src"],
        "剪影肩宽_px": round(m["shoulder_sil_px"], 1),
        "骨架肩宽_px": round(m["pose_shoulder_px"], 1) if m.get("pose_shoulder_px") else None,
        "是否采用骨架": bool(m.get("used_pose")),
        "标注图": out_path,
        "警告": m["warnings"],
    }

    if a.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    ln = lambda t="": (("── %s " % t) + "─" * max(0, 56 - len(t) * 2)) if t else "─" * 60
    print(ln("读图"))
    print("  %s" % a.image)
    print("  读取：%s    抠图：%s" % (how, mhow))
    print("  工作分辨率 %d × %d（原图约为其 %.2f 倍）" % (w, h, scale))

    print()
    print(ln("量到的线（工作图 y 坐标）"))
    print("  头顶 %d   下巴 %d   颈最窄 %d   脚底 %d"
          % (m["top_y"], m["chin_y"], m["neck_y"], m["foot_y"]))
    print("  全身高 %d px   头高 %d px   头宽 %.0f px   肩宽 %.0f px   颈宽 %d px"
          % (m["total_px"], m["head_h_px"], m["head_w_px"],
             m["shoulder_w_px"], m["neck_w_px"]))
    print("  下巴线来源：%s" % result["下巴线来源"])
    print("  脚底线来源：%s" % result["脚底线来源"])
    print("  肩宽来源：  %s" % result["肩宽来源"])
    if m.get("pose_shoulder_px") is not None:
        print("  骨架肩宽 %.0f px vs 剪影肩线 %.0f px → %s"
              % (m["pose_shoulder_px"], m["shoulder_sil_px"],
                 "本次采用骨架" if m.get("used_pose") else "骨架判为不可信，已回退剪影"))

    print()
    print(ln("★ 结果"))
    print("  头身比        N = 全身高 / 头高 = %d / %d = %.2f"
          % (m["total_px"], m["head_h_px"], m["ratio"]))
    if m["ratio_hi"] - m["ratio_lo"] > 0.01:
        print("                下巴线取在 [下巴, 颈最窄] 之间何处会让 N 落在 %.2f ~ %.2f，"
              % (m["ratio_lo"], m["ratio_hi"]))
        print("                这是本方法的主要误差来源，对着标注图确认再用")
    print("  头宽/肩宽     r = 头宽 / 肩宽   = %.0f / %.0f = %.3f"
          % (m["head_w_px"], m["shoulder_w_px"], m["head_shoulder"]))
    print("  （头宽取的是剪影最大宽，天然含发，与主脚本 r 的定义一致）")

    if out_path:
        print()
        print(ln("标注图"))
        print("  %s" % out_path)
        print("  红线=头顶/下巴   蓝线=脚底   灰细线=颈最窄   绿=头宽   橙=肩宽(采用值)")
        print("  紫=骨架肩线（--pose 时画出，供对照；和橙色对不上就说明骨架这次跑偏了）")
        print("  左边灰曲线是逐行剪影宽度剖面，四条线对不对一眼就能看出来。")

    if m["warnings"]:
        print()
        print(ln("⚠ 警告"))
        for s in m["warnings"]:
            print("  · " + s)

    print()
    print(ln("接着跑主脚本"))
    print("  python3 kigurumi_head_calc.py \\")
    print("    --height 168 --weight 52 --shoulder-width 38 \\")
    print("    --head-width 15.5 --head-height 22.5 \\")
    print("    --ratio %.2f --head-shoulder %.3f \\" % (m["ratio"], m["head_shoulder"]))
    print("    --wig 3 --mount chin")
    print("  （前两行换成演员实测数据）")
    print(ln())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
