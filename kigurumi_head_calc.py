#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kigurumi 头壳尺寸计算器
======================

根据「演员身体数据 + 角色设定比例 + 假发厚度」反推头壳模型应有的
长(深) × 宽 × 高，并做可戴性校验。

核心思路
--------
1. 头壳底沿（角色下巴线）落在演员身上的某个高度 h_mount（肩峰 / 下巴 / 自定义）。
2. 角色是 N 头身，即  总高 = N × 单个头高。
   而  总高 = h_mount + 头壳可视高。
   → 头壳可视高 = h_mount / (N - 1)              （头身比含发时）
   → 净壳高      = (h_mount + 发顶厚) / (N - 1)   （头身比不含发时）
3. 角色 头宽/肩宽 = r，而着装后肩宽 = 演员肩宽 + 2×肩部填充
   → 头壳可视宽 = r × 着装肩宽
4. 减去各方向假发留量，得到实际要建模 / 打印的「净壳」尺寸。
   留量 = 从该侧头壳基底材质的外表面，垂直量到该方向发丝外缘的距离
   （发套底网 + 胶带 + 蓬松量都在内；壁厚在壳外皮以内，不重复计算）。
5. 用演员头围尺寸 + 内部余量校验净壳是否塞得下人头，不够则给出补救方案
   （增高鞋、改肩宽、调头身比等）。

所有长度单位：厘米(cm)；重量：千克(kg)。

用法
----
    python3 kigurumi_head_calc.py                  # 交互式问答
    python3 kigurumi_head_calc.py --height 168 ... # 命令行参数
    python3 kigurumi_head_calc.py --explain        # 打印公式与假设
    python3 kigurumi_head_calc.py ... --json       # 输出 JSON
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# 人体测量学经验常数（成年人平均值，仅在用户未提供实测值时作为估算兜底）
# ---------------------------------------------------------------------------
ACROMION_RATIO = 0.818      # 肩峰高 / 身高
EYE_FROM_VERTEX = 0.44      # 眼中心到头顶距离 / 头高
HEAD_DEPTH_PER_WIDTH = 1.25 # 真人头深(前后) / 头宽(左右)

# 头壳经验区间（动漫风格头壳，非写实）
ASPECT_MIN, ASPECT_MAX = 0.68, 1.00   # 净壳 宽/高 的常见范围
ASPECT_TARGET = 0.82                  # 折中目标值
EYE_FROM_TOP_MIN, EYE_FROM_TOP_MAX = 0.50, 0.65  # 角色眼位到壳顶 / 净壳高

MATERIAL_DENSITY = {        # g/cm^3
    "pla": 1.24,
    "petg": 1.27,
    "abs": 1.04,
    "resin": 1.15,
    "glassfiber": 1.60,     # 玻璃钢层压件等效密度
}


# ---------------------------------------------------------------------------
# 输入数据结构
# ---------------------------------------------------------------------------
@dataclass
class Body:
    """演员身体数据"""
    height: float                              # 身高
    weight: float                              # 体重
    shoulder_width: float                      # 肩宽（两肩峰点直线距离）
    head_width: float                          # 头宽（左右最大，含耳廓外缘）
    head_height: float                         # 头高（下巴尖到头顶）
    head_depth: Optional[float] = None         # 头深（前后最大），缺省按头宽估算
    shoulder_height: Optional[float] = None    # 肩峰离地高，缺省按身高估算


@dataclass
class Character:
    """角色设定比例"""
    head_body_ratio: float          # N 头身
    head_shoulder_ratio: float      # 头宽 / 肩宽
    depth_ratio: float = 1.15       # 净壳 深/宽（动漫头侧看偏长，1.05~1.30）
    hair_in_ratio: bool = True      # 头身比里的「头」是否把发型算进去


@dataclass
class Build:
    """制作参数"""
    mount: str = "shoulder"         # shoulder | chin | custom
    rim_drop: float = 0.0           # mount=custom 时：底沿低于演员下巴多少 cm
    lift: float = 0.0               # 增高鞋 / 内增高厚度

    # 假发留量 = 从该侧「头壳基底材质的外表面」垂直量到该方向发丝外缘的距离。
    # 不是发丝本身有多厚，也不是发套自身的厚度：发套底网、定位胶带、发丝蓬松量
    # 全都算在这一个数里，因为参与计算的只有「壳外皮到轮廓最外沿」这一段。
    # 壁厚 wall 不参与，它在壳外皮以内，不会重复计算。
    wig: float = 3.0                # 基准留量（未单独指定的方向 = 它 × WIG_FACTOR）
    wig_top: Optional[float] = None   # 壳顶外表面 → 发丝最高点
    wig_side: Optional[float] = None  # 单侧：最宽处壳外表面 → 该侧发丝最外沿（左右方向量）
    wig_front: Optional[float] = None # 前额壳外表面 → 刘海最前沿（前后方向量）
    wig_back: Optional[float] = None  # 后脑壳外表面 → 后发最后沿（前后方向量）

    # 海绵留量 = 贴在壳内壁上的衬垫标称厚度（买来时的厚度）。
    # 它和下面的「空气余量」是两回事：海绵是实体、可压缩、负责夹住头；
    # 空气余量是留给风扇/走线/通风/进出头的空隙，不可压缩。
    foam: float = 2.0                  # 基准厚度（未单独指定的方向 = 它 × FOAM_FACTOR）
    foam_top: Optional[float] = None   # 头顶（头台/顶垫，通常最厚）
    foam_side: Optional[float] = None  # 单侧（夹持头部的主要衬垫）
    foam_face: Optional[float] = None  # 面部方向（要让出视线，通常最薄）
    foam_back: Optional[float] = None  # 后脑方向
    foam_compress: float = 0.30        # 戴上后的压缩率：实占厚度 = 标称 × (1 - 它)
    foam_density: float = 0.05         # g/cm³，软质 EVA/海绵约 0.03~0.10

    side_clear: float = 1.0         # 单侧空气余量（风扇/走线/通风）
    top_clear: float = 1.5          # 头顶空气余量
    face_clear: float = 2.0         # 面部方向空气余量（视窗/网纱/睫毛内构）
    back_clear: float = 1.5         # 后脑方向空气余量
    wall: float = 0.5               # 壳体壁厚

    shoulder_pad: Optional[float] = None  # 单侧肩部填充，缺省按 BMI 自动给
    material: str = "pla"


# 未单独指定时，各方向假发留量 = wig × 系数。
# 侧向系数取 1.0 只是「没有更好信息时按顶部同量处理」，侧向留量和顶部蓬松量
# 是两个独立的物理量，能量就用 --wig-side 直接给实测值。
WIG_FACTOR = {"top": 1.0, "side": 1.0, "front": 0.6, "back": 1.4}

# 未单独指定时，各方向海绵厚度 = foam × 系数。
# 顶部要做头台承重所以最厚，面部要让出视线所以最薄。
FOAM_FACTOR = {"top": 1.5, "side": 1.0, "face": 0.4, "back": 1.0}

# 海绵不会铺满整个内壁（面窗、颈口、机构位都不贴），估重时按这个覆盖率打折。
FOAM_COVERAGE = 0.65


# ---------------------------------------------------------------------------
# 计算
# ---------------------------------------------------------------------------
def auto_shoulder_pad(bmi: float) -> float:
    """按体型给单侧肩部填充建议：偏瘦的人需要更多填充才能撑出角色轮廓。"""
    if bmi < 18.5:
        return 1.5
    if bmi < 24.0:
        return 1.0
    return 0.5


def ellipsoid_area(a: float, b: float, c: float) -> float:
    """Knud Thomsen 近似公式，椭球表面积。"""
    p = 1.6075
    return 4 * math.pi * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3) ** (1 / p)


def compute(body: Body, char: Character, build: Build) -> Dict[str, Any]:
    notes: List[str] = []       # 估算/假设说明
    warnings: List[str] = []    # 需要注意的问题
    advice: List[str] = []      # 补救建议

    if char.head_body_ratio <= 1.0:
        raise ValueError("头身比必须大于 1（否则身体高度为零或负）")

    # --- 1. 补全缺失的人体数据 ---------------------------------------------
    head_depth = body.head_depth
    if head_depth is None:
        head_depth = body.head_width * HEAD_DEPTH_PER_WIDTH
        notes.append(
            "头深未实测，按 头宽 × %.2f 估算 = %.1f cm" % (HEAD_DEPTH_PER_WIDTH, head_depth)
        )

    shoulder_height = body.shoulder_height
    if shoulder_height is None:
        shoulder_height = body.height * ACROMION_RATIO
        notes.append(
            "肩峰高未实测，按 身高 × %.3f 估算 = %.1f cm" % (ACROMION_RATIO, shoulder_height)
        )

    bmi = body.weight / (body.height / 100.0) ** 2

    shoulder_pad = build.shoulder_pad
    if shoulder_pad is None:
        shoulder_pad = auto_shoulder_pad(bmi)
        notes.append(
            "肩部填充未指定，按 BMI %.1f 自动取单侧 %.1f cm" % (bmi, shoulder_pad)
        )

    # --- 2. 假发各方向厚度 --------------------------------------------------
    wig_top = build.wig_top if build.wig_top is not None else build.wig * WIG_FACTOR["top"]
    wig_side = build.wig_side if build.wig_side is not None else build.wig * WIG_FACTOR["side"]
    wig_front = build.wig_front if build.wig_front is not None else build.wig * WIG_FACTOR["front"]
    wig_back = build.wig_back if build.wig_back is not None else build.wig * WIG_FACTOR["back"]

    # --- 3. 底沿高度与内部净空需求 -----------------------------------------
    chin_height = body.height - body.head_height          # 演员下巴离地
    if build.mount == "shoulder":
        rim_base = shoulder_height
        mount_desc = "肩支撑（底沿在肩峰线）"
    elif build.mount == "chin":
        rim_base = chin_height
        mount_desc = "颈支撑（底沿在演员下巴线）"
    elif build.mount == "custom":
        rim_base = chin_height - build.rim_drop
        mount_desc = "自定义（底沿低于下巴 %.1f cm）" % build.rim_drop
    else:
        raise ValueError("mount 只能是 shoulder / chin / custom")

    if rim_base <= 0:
        raise ValueError("底沿高度算出来 <= 0，请检查身高 / 头高 / rim_drop")
    if rim_base > chin_height + 0.01:
        warnings.append(
            "底沿(%.1f cm)高于演员下巴(%.1f cm)，壳体会卡住下颌，请改用 chin 或 custom 模式"
            % (rim_base, chin_height)
        )

    h_mount = rim_base + build.lift                  # 穿鞋后底沿离地高
    vertex = body.height + build.lift                # 穿鞋后头顶离地高
    rim_to_vertex = vertex - h_mount                 # 底沿到演员头顶（与增高无关）

    # --- 海绵：标称厚度 → 戴上后的实占厚度 ---
    fm = build.foam
    f_top = build.foam_top if build.foam_top is not None else fm * FOAM_FACTOR["top"]
    f_side = build.foam_side if build.foam_side is not None else fm * FOAM_FACTOR["side"]
    f_face = build.foam_face if build.foam_face is not None else fm * FOAM_FACTOR["face"]
    f_back = build.foam_back if build.foam_back is not None else fm * FOAM_FACTOR["back"]
    keep = 1.0 - build.foam_compress
    if not (0.0 <= build.foam_compress < 1.0):
        raise ValueError("海绵压缩率必须在 [0, 1) 之间")
    e_top, e_side = f_top * keep, f_side * keep
    e_face, e_back = f_face * keep, f_back * keep

    # 净壳最小可戴尺寸 = 人头 + 海绵实占 + 空气余量 + 壁厚
    hs_min = rim_to_vertex + e_top + build.top_clear + build.wall
    ws_min = body.head_width + 2 * (e_side + build.side_clear + build.wall)
    ds_min = (head_depth + e_face + build.face_clear
              + e_back + build.back_clear + 2 * build.wall)

    # --- 4. 由角色比例反推目标尺寸 -----------------------------------------
    n = char.head_body_ratio
    if char.hair_in_ratio:
        h_visual = h_mount / (n - 1)
        hs = h_visual - wig_top
    else:
        hs = (h_mount + wig_top) / (n - 1)
        h_visual = hs + wig_top
    total_height = h_mount + h_visual                # 角色成品总身高

    shoulder_cos = body.shoulder_width + 2 * shoulder_pad   # 着装后肩宽
    w_visual = char.head_shoulder_ratio * shoulder_cos
    ws = w_visual - 2 * wig_side

    ds = ws * char.depth_ratio
    d_visual = ds + wig_front + wig_back

    # --- 5. 可行性校验 ------------------------------------------------------
    ok_h = hs >= hs_min
    ok_w = ws >= ws_min
    ok_d = ds >= ds_min

    # 高度不够 → 需要多少增高
    if char.hair_in_ratio:
        need_mount_h = (n - 1) * (hs_min + wig_top)
        n_max = 1 + h_mount / (hs_min + wig_top)
    else:
        need_mount_h = (n - 1) * hs_min - wig_top
        n_max = 1 + (h_mount + wig_top) / hs_min
    extra_lift = need_mount_h - h_mount

    if not ok_h:
        warnings.append(
            "净壳高 %.1f cm < 可戴最小值 %.1f cm，差 %.1f cm：角色头身比对这位演员偏「写实」了"
            % (hs, hs_min, hs_min - hs)
        )
        advice.append(
            "把增高鞋/内增高加到 %.1f cm（现 %.1f cm，需再加 %.1f cm），或把头身比降到 %.2f 以下"
            % (build.lift + extra_lift, build.lift, extra_lift, n_max)
        )
    if hs > 45:
        warnings.append("净壳高 %.1f cm 超过 45 cm，重量与颈部负荷会很吃力" % hs)

    # 宽度不够 → 需要多少肩宽
    pad_needed = ((ws_min + 2 * wig_side) / char.head_shoulder_ratio - body.shoulder_width) / 2
    if not ok_w:
        warnings.append(
            "净壳宽 %.1f cm < 可戴最小值 %.1f cm，差 %.1f cm"
            % (ws, ws_min, ws_min - ws)
        )
        advice.append(
            "单侧肩部填充加到 %.1f cm（现 %.1f cm），或减薄侧面假发/内衬"
            % (pad_needed, shoulder_pad)
        )
    if not ok_d:
        warnings.append(
            "净壳深 %.1f cm < 可戴最小值 %.1f cm，差 %.1f cm" % (ds, ds_min, ds_min - ds)
        )
        advice.append("把 深/宽 比从 %.2f 提到 %.2f 以上" % (char.depth_ratio, ds_min / ws))

    # --- 6. 形状合理性 ------------------------------------------------------
    aspect = ws / hs if hs > 0 else float("nan")
    if hs > 0 and not (ASPECT_MIN <= aspect <= ASPECT_MAX):
        shape = "偏窄长" if aspect < ASPECT_MIN else "偏宽扁"
        warnings.append(
            "净壳 宽/高 = %.2f，%s（动漫头壳常见 %.2f~%.2f）"
            % (aspect, shape, ASPECT_MIN, ASPECT_MAX)
        )
        pad_balance = (
            (ASPECT_TARGET * hs + 2 * wig_side) / char.head_shoulder_ratio - body.shoulder_width
        ) / 2
        advice.append(
            "若想把 宽/高 拉回 %.2f：单侧肩部填充改为 %.1f cm（正数=垫宽，负数=收窄肩线）"
            % (ASPECT_TARGET, pad_balance)
        )
        # 另一条路：宽度不动，改头身比把高度调过来
        hs_balance = ws / ASPECT_TARGET
        if char.hair_in_ratio:
            n_balance = 1 + h_mount / (hs_balance + wig_top)
        else:
            n_balance = 1 + (h_mount + wig_top) / hs_balance
        if n_balance > 1:
            advice.append(
                "或保持肩宽不动，把头身比改成 %.2f（当前 %.2f）——头身比与 头宽/肩宽 这两个"
                "设定对这具身体是互相冲突的，必须让一个" % (n_balance, n)
            )

    # --- 7. 眼位 ------------------------------------------------------------
    eye_from_vertex = EYE_FROM_VERTEX * body.head_height
    eye_floor = vertex - eye_from_vertex
    eye_from_rim = eye_floor - h_mount              # 演员眼睛离壳底
    eye_from_top = (h_mount + hs) - eye_floor       # 演员眼睛离壳顶（净壳）
    eye_ratio = eye_from_top / hs if hs > 0 else float("nan")
    eye_ok = EYE_FROM_TOP_MIN <= eye_ratio <= EYE_FROM_TOP_MAX
    if hs > 0 and not eye_ok:
        warnings.append(
            "演员眼位在净壳高的 %.0f%% 处（自壳顶下量），角色眼位一般在 %.0f%%~%.0f%%，"
            "视线可能对不上眼窗，需要靠内部抬高/降低头台调整"
            % (eye_ratio * 100, EYE_FROM_TOP_MIN * 100, EYE_FROM_TOP_MAX * 100)
        )

    # --- 8. 重量估算 --------------------------------------------------------
    density = MATERIAL_DENSITY.get(build.material, MATERIAL_DENSITY["pla"])
    area = ellipsoid_area(max(0.1, ws / 2), max(0.1, ds / 2), max(0.1, hs / 2)) * 0.92   # 扣除颈口开孔
    shell_volume = area * build.wall                      # cm^3
    shell_mass = shell_volume * density / 1000.0          # kg（纯壳体，不含涂装/假发）

    # 海绵按标称厚度算重（压缩不减少材料，只是占位变薄）
    area_in = ellipsoid_area(max(0.1, ws / 2 - build.wall),
                             max(0.1, ds / 2 - build.wall),
                             max(0.1, hs / 2 - build.wall)) * 0.92
    foam_mean = (f_top + 2 * f_side + f_face + f_back) / 5.0
    foam_mass = area_in * foam_mean * FOAM_COVERAGE * build.foam_density / 1000.0
    total_mass = shell_mass + foam_mass

    mass_limit = min(2.5, 0.035 * body.weight)
    if total_mass > mass_limit:
        warnings.append(
            "壳体 %.2f + 海绵 %.2f = %.2f kg，已超过建议上限 %.2f kg"
            "（涂装+假发+风扇还要再加 0.4~1.2 kg）"
            % (shell_mass, foam_mass, total_mass, mass_limit)
        )
        if shell_mass > foam_mass:
            advice.append(
                "壁厚从 %.1f mm 减到 %.1f mm 可把总重压到 %.2f kg，或改用更轻的材料"
                % (build.wall * 10,
                   max(0.1, build.wall * 10 * (mass_limit - foam_mass) / max(0.01, shell_mass)),
                   mass_limit)
            )
        else:
            advice.append(
                "重量主要在海绵上（%.2f kg）：改用更薄或更低密度的衬垫，"
                "或只在受力点贴海绵而不是满铺" % foam_mass
            )

    feasible = ok_h and ok_w and ok_d

    return {
        "输入": {
            "身高": body.height, "体重": body.weight, "肩宽": body.shoulder_width,
            "头宽": body.head_width, "头高": body.head_height, "头深": round(head_depth, 1),
            "头身比": n, "头宽肩宽比": char.head_shoulder_ratio,
            "假发厚度_基准": build.wig,
        },
        "人体基准": {
            "BMI": round(bmi, 1),
            "肩峰高": round(shoulder_height, 1),
            "下巴高": round(chin_height, 1),
            "增高鞋": build.lift,
            "支撑方式": mount_desc,
            "底沿离地": round(h_mount, 1),
            "底沿到头顶": round(rim_to_vertex, 1),
            "单侧肩部填充": round(shoulder_pad, 1),
            "着装后肩宽": round(shoulder_cos, 1),
        },
        "假发厚度": {
            "顶": round(wig_top, 2), "侧(单侧)": round(wig_side, 2),
            "前": round(wig_front, 2), "后": round(wig_back, 2),
        },
        "净壳尺寸": {"深": round(ds, 1), "宽": round(ws, 1), "高": round(hs, 1)},
        "含发外观尺寸": {"深": round(d_visual, 1), "宽": round(w_visual, 1), "高": round(h_visual, 1)},
        "净壳比例_深宽高": [round(ds / ws, 3), 1.0, round(hs / ws, 3)] if ws > 0 else None,
        "含发比例_深宽高": [round(d_visual / w_visual, 3), 1.0, round(h_visual / w_visual, 3)]
        if w_visual > 0 else None,
        "成品总身高": round(total_height, 1),
        "可戴最小净壳": {"深": round(ds_min, 1), "宽": round(ws_min, 1), "高": round(hs_min, 1)},
        "海绵": {
            "标称_顶": round(f_top, 2), "标称_侧单侧": round(f_side, 2),
            "标称_面": round(f_face, 2), "标称_后": round(f_back, 2),
            "压缩率": build.foam_compress,
            "实占_顶": round(e_top, 2), "实占_侧单侧": round(e_side, 2),
            "实占_面": round(e_face, 2), "实占_后": round(e_back, 2),
        },
        "尺寸链": {
            "宽": [("演员头宽", round(body.head_width, 1)),
                   ("海绵实占 2×%.2f" % e_side, round(2 * e_side, 1)),
                   ("空气余量 2×%.2f" % build.side_clear, round(2 * build.side_clear, 1)),
                   ("壁厚 2×%.2f" % build.wall, round(2 * build.wall, 1))],
            "高": [("底沿到演员头顶", round(rim_to_vertex, 1)),
                   ("海绵实占(顶)", round(e_top, 1)),
                   ("空气余量(顶)", round(build.top_clear, 1)),
                   ("壁厚", round(build.wall, 1))],
            "深": [("演员头深", round(head_depth, 1)),
                   ("海绵实占 面%.2f+后%.2f" % (e_face, e_back), round(e_face + e_back, 1)),
                   ("空气余量 面%.2f+后%.2f" % (build.face_clear, build.back_clear),
                    round(build.face_clear + build.back_clear, 1)),
                   ("壁厚 2×%.2f" % build.wall, round(2 * build.wall, 1))],
        },
        "余量": {
            "深": round(ds - ds_min, 1), "宽": round(ws - ws_min, 1), "高": round(hs - hs_min, 1),
        },
        "校验": {"深": ok_d, "宽": ok_w, "高": ok_h, "整体可行": feasible},
        "极限": {
            "该演员最大可行头身比": round(n_max, 2),
            "达成当前头身比所需增高": round(max(0.0, extra_lift), 1),
            "达成当前头宽所需单侧肩部填充": round(pad_needed, 1),
        },
        "眼位": {
            "演员眼睛离壳底": round(eye_from_rim, 1),
            "演员眼睛离壳顶": round(eye_from_top, 1),
            "占净壳高比例": round(eye_ratio, 3),
            "建议区间": [EYE_FROM_TOP_MIN, EYE_FROM_TOP_MAX],
            "合格": bool(eye_ok),
        },
        "重量估算": {
            "材料": build.material,
            "壁厚_mm": round(build.wall * 10, 1),
            "表面积_cm2": round(area, 0),
            "壳体估重_kg": round(shell_mass, 2),
            "海绵估重_kg": round(foam_mass, 2),
            "合计_kg": round(total_mass, 2),
            "建议上限_kg": round(mass_limit, 2),
        },
        "建模包围盒_mm": {
            "净壳": [round(ds * 10, 1), round(ws * 10, 1), round(hs * 10, 1)],
            "含发": [round(d_visual * 10, 1), round(w_visual * 10, 1), round(h_visual * 10, 1)],
        },
        "说明": notes,
        "警告": warnings,
        "建议": advice,
    }


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------
def dwidth(s: str) -> int:
    """终端显示宽度：CJK 全角字符算 2 列。"""
    return sum(2 if ord(c) > 0x2E7F else 1 for c in s)


def dpad(s: str, width: int) -> str:
    """按显示宽度左对齐补空格。"""
    return s + " " * max(0, width - dwidth(s))


def drpad(s: str, width: int) -> str:
    """按显示宽度右对齐补空格。"""
    return " " * max(0, width - dwidth(s)) + s


def line(title: str = "") -> str:
    if not title:
        return "─" * 60
    head = "── %s " % title
    return head + "─" * max(0, 60 - dwidth(head))


def render(r: Dict[str, Any]) -> str:
    o: List[str] = []
    a = o.append

    a(line("输入"))
    i = r["输入"]
    a("  身高 %.1f cm   体重 %.1f kg   肩宽 %.1f cm" % (i["身高"], i["体重"], i["肩宽"]))
    a("  头宽 %.1f cm   头高 %.1f cm   头深 %.1f cm" % (i["头宽"], i["头高"], i["头深"]))
    a("  角色 %.2f 头身   头宽/肩宽 = %.3f   假发基准厚 %.1f cm"
      % (i["头身比"], i["头宽肩宽比"], i["假发厚度_基准"]))

    a("")
    a(line("人体基准"))
    b = r["人体基准"]
    a("  BMI %.1f   肩峰高 %.1f   下巴高 %.1f   增高 %.1f cm"
      % (b["BMI"], b["肩峰高"], b["下巴高"], b["增高鞋"]))
    a("  支撑方式：%s" % b["支撑方式"])
    a("  底沿离地 %.1f cm   底沿到演员头顶 %.1f cm" % (b["底沿离地"], b["底沿到头顶"]))
    a("  单侧肩部填充 %.1f cm → 着装后肩宽 %.1f cm" % (b["单侧肩部填充"], b["着装后肩宽"]))

    a("")
    a(line("假发留量（自头壳基底材质外表面量起）"))
    w = r["假发厚度"]
    a("  顶 %.2f   侧(单侧) %.2f   前 %.2f   后 %.2f cm"
      % (w["顶"], w["侧(单侧)"], w["前"], w["后"]))
    a("  正面看：含发头宽 = 净壳宽 + 2 × 侧向留量（壁厚在壳外皮以内，不重复计）")

    a("")
    a(line("★ 头壳尺寸"))
    n, v = r["净壳尺寸"], r["含发外观尺寸"]
    a("  %s%s %s %s" % (dpad("", 20), drpad("深(前后)", 9),
                        drpad("宽(左右)", 9), drpad("高(上下)", 9)))
    a("  %s%9.1f %9.1f %9.1f  cm" % (dpad("净壳（建模用）", 20), n["深"], n["宽"], n["高"]))
    a("  %s%9.1f %9.1f %9.1f  cm" % (dpad("含发外观（成品）", 20), v["深"], v["宽"], v["高"]))
    if r["净壳比例_深宽高"]:
        p = r["净壳比例_深宽高"]
        a("  净壳比例 深:宽:高 = %.3f : %.3f : %.3f" % (p[0], p[1], p[2]))
    if r["含发比例_深宽高"]:
        p = r["含发比例_深宽高"]
        a("  含发比例 深:宽:高 = %.3f : %.3f : %.3f" % (p[0], p[1], p[2]))
    a("  成品总身高（角色视觉身高）：%.1f cm" % r["成品总身高"])

    a("")
    a(line("海绵留量（贴在壳内壁的衬垫）"))
    fo = r["海绵"]
    a("  标称厚度  顶 %.2f   侧(单侧) %.2f   面 %.2f   后 %.2f cm"
      % (fo["标称_顶"], fo["标称_侧单侧"], fo["标称_面"], fo["标称_后"]))
    a("  压缩率 %.0f%% → 实占  顶 %.2f   侧(单侧) %.2f   面 %.2f   后 %.2f cm"
      % (fo["压缩率"] * 100, fo["实占_顶"], fo["实占_侧单侧"], fo["实占_面"], fo["实占_后"]))
    a("  （海绵按标称厚度买，戴上后被压薄；占位算实占，重量算标称）")

    a("")
    a(line("尺寸链：可戴最小净壳是怎么堆出来的"))
    for axis in ("宽", "高", "深"):
        parts = r["尺寸链"][axis]
        expr = " + ".join("%s %.1f" % (nm, v) for nm, v in parts)
        a("  %s：%s" % (axis, expr))
        # 显示未取整的真值，避免和「可戴性校验」里的数字差 0.1
        a("      = %.1f cm  ← 最小%s" % (r["可戴最小净壳"][axis], axis))

    a("")
    a(line("可戴性校验"))
    mn, mg, ck = r["可戴最小净壳"], r["余量"], r["校验"]
    for k in ("深", "宽", "高"):
        a("  %s：净壳 %.1f  ≥ 最小 %.1f ?  %s   余量 %+.1f cm"
          % (k, r["净壳尺寸"][k], mn[k], "通过" if ck[k] else "不通过", mg[k]))
    a("  结论：%s" % ("✅ 可行" if ck["整体可行"] else "❌ 当前参数做不出来，见下方建议"))
    lim = r["极限"]
    a("  该演员最大可行头身比 ≈ %.2f 头身" % lim["该演员最大可行头身比"])
    if lim["达成当前头身比所需增高"] > 0:
        a("  达成当前头身比还需增高 %.1f cm" % lim["达成当前头身比所需增高"])
    a("  达成当前头宽所需单侧肩部填充 %.1f cm" % lim["达成当前头宽所需单侧肩部填充"])

    a("")
    a(line("眼位"))
    e = r["眼位"]
    a("  演员眼睛：离壳底 %.1f cm，离壳顶 %.1f cm（净壳高的 %.0f%%）"
      % (e["演员眼睛离壳底"], e["演员眼睛离壳顶"], e["占净壳高比例"] * 100))
    a("  角色眼位建议落在自壳顶下量 %.0f%%~%.0f%% 处 → %s"
      % (e["建议区间"][0] * 100, e["建议区间"][1] * 100, "对得上" if e["合格"] else "需内部调整头台高度"))

    a("")
    a(line("重量估算"))
    m = r["重量估算"]
    a("  材料 %s   壁厚 %.1f mm   表面积 ≈ %.0f cm²" % (m["材料"], m["壁厚_mm"], m["表面积_cm2"]))
    a("  壳体 %.2f + 海绵 %.2f = %.2f kg（建议上限 %.2f kg）"
      % (m["壳体估重_kg"], m["海绵估重_kg"], m["合计_kg"], m["建议上限_kg"]))
    a("  涂装 + 假发 + 风扇另计 0.4~1.2 kg，不在上面这个数里")

    a("")
    a(line("建模包围盒"))
    bb = r["建模包围盒_mm"]
    a("  净壳 D×W×H = %.1f × %.1f × %.1f mm" % tuple(bb["净壳"]))
    a("  含发 D×W×H = %.1f × %.1f × %.1f mm" % tuple(bb["含发"]))

    if r["说明"]:
        a("")
        a(line("估算说明"))
        for s in r["说明"]:
            a("  · " + s)
    if r["警告"]:
        a("")
        a(line("⚠ 警告"))
        for s in r["警告"]:
            a("  · " + s)
    if r["建议"]:
        a("")
        a(line("→ 建议"))
        for s in r["建议"]:
            a("  · " + s)
    a(line())
    return "\n".join(o)


EXPLAIN = """
计算模型与假设
==============

【符号】
  H  演员身高      hh 演员头高      hw 演员头宽      hd 演员头深
  S  演员肩宽      p  单侧肩部填充  L  增高鞋厚度
  N  角色头身比    r  角色 头宽/肩宽
  Tt/Ts/Tf/Tb 假发在 顶/侧/前/后 的厚度
  Hs/Ws/Ds    净壳（不含发）的 高/宽/深

【1. 底沿高度 h_mount】
  肩支撑：h_mount = 0.818·H + L
  颈支撑：h_mount = (H - hh) + L
  自定义：h_mount = (H - hh - rim_drop) + L

【2. 高度（由头身比反推）】
  角色总高 = h_mount + 头壳可视高，且 总高 = N × 一个头高
  · 头身比含发：可视高 = h_mount / (N-1)，  Hs = 可视高 - Tt
  · 头身比不含发：Hs = (h_mount + Tt) / (N-1)
  注意：h_mount 与 L 同增同减，所以增高鞋会把「需要的头」变大。
  这也是为什么高头身角色难做——它要求一个小到塞不进人头的壳。

【3. 宽度（由头宽/肩宽反推）】
  着装后肩宽 = S + 2p
  可视宽 = r × (S + 2p)
  Ws = 可视宽 - 2·Ts

【4. 深度】
  Ds = Ws × depth_ratio（动漫头壳 1.05~1.30，默认 1.15）
  可视深 = Ds + Tf + Tb

【5. 可戴最小净壳 —— 尺寸链】
  海绵和空气余量是两个不同的东西，分开算：
  · 海绵：贴在内壁上的实体衬垫，负责夹住头，可压缩。
    按标称厚度买，戴上后被压薄 → 实占厚度 = 标称 × (1 - 压缩率)，缺省压缩率 0.30。
    缺省方向系数：顶 1.5 / 侧 1.0 / 面 0.4 / 后 1.0
    （顶部要做头台承重所以最厚，面部要让出视线所以最薄）。
  · 空气余量：留给风扇、走线、通风、进出头的空隙，不可压缩，不能被海绵顶掉。

  Hs_min = (底沿到演员头顶) + 顶部海绵实占 + 顶部空气余量 + 壁厚
  Ws_min = hw + 2×(侧海绵实占 + 单侧空气余量 + 壁厚)
  Ds_min = hd + (面海绵实占 + 面空气余量) + (后海绵实占 + 后空气余量) + 2×壁厚

  注意海绵只推高「最小可戴尺寸」，不改变由角色比例定出来的目标尺寸。
  所以海绵加厚 = 余量变小 = 更容易做不出来，这正是它该有的效果。

【6. 假发校准 —— 留量怎么量】
  留量的定义：从该侧「头壳基底材质的外表面」，沿该方向垂直量到发丝外缘的距离。
  · 侧向 Ts = 在头壳最宽处，沿左右方向，从壳的外皮量到该侧发丝最外沿。
    发套底网、定位胶带、发丝蓬松量全部包含在这一个数里，因为参与计算的
    只有「壳外皮 → 轮廓最外沿」这一段。
  · 壁厚 wall 不参与假发这几个减法：它在壳外皮以内，不会被重复计算。
  · 所以正面看：含发头宽 = Ws + 2·Ts，程序反过来用 Ws = 含发头宽 - 2·Ts。

  留量不是拿来「加大」头壳的，而是从角色外观尺寸里「扣掉」的：
  你按角色比例算出的是戴上假发后的外轮廓，实际要建模/打印的净壳
  必须比它小一圈，否则装上假发就超比例了。
  各方向缺省系数：顶 1.0 / 侧 1.0 / 前 0.6 / 后 1.4（前发贴额、后发蓬松）。
  侧向的 1.0 只是没有实测值时的兜底，它和顶部蓬松量本是两个独立的量，
  能量到就用 --wig-side 给实测值。

【7. 经验值来源（请按实际情况覆盖）】
  · 肩峰高 = 0.818 × 身高、眼位 = 头顶下 0.44 × 头高、头深 = 1.25 × 头宽
    来自成年人人体测量平均值，个体差异可达 ±5%，能实测就实测。
  · 净壳 宽/高 0.68~1.00、角色眼位 壳顶下 50%~65%、壳体重量上限
    是动漫风格头壳的经验区间，不是硬性标准。
"""


# ---------------------------------------------------------------------------
# 交互式输入
# ---------------------------------------------------------------------------
def ask(prompt: str, default: Optional[float] = None, optional: bool = False) -> Optional[float]:
    tip = " [%g]" % default if default is not None else (" [回车=自动估算]" if optional else "")
    while True:
        s = input("%s%s: " % (prompt, tip)).strip()
        if not s:
            if default is not None:
                return default
            if optional:
                return None
            print("   ! 此项必填")
            continue
        try:
            return float(s)
        except ValueError:
            print("   ! 请输入数字")


def ask_choice(prompt: str, options: List[str], default: str) -> str:
    while True:
        s = input("%s (%s) [%s]: " % (prompt, "/".join(options), default)).strip().lower()
        if not s:
            return default
        if s in options:
            return s
        print("   ! 请从 %s 中选择" % "/".join(options))


def ask_bool(prompt: str, default: bool) -> bool:
    d = "y" if default else "n"
    s = input("%s (y/n) [%s]: " % (prompt, d)).strip().lower()
    if not s:
        return default
    return s.startswith("y")


def interactive():
    print("=" * 60)
    print("  kigurumi 头壳尺寸计算器   （单位 cm / kg，回车用默认值）")
    print("=" * 60)

    print("\n【演员数据】")
    height = ask("身高")
    weight = ask("体重(kg)")
    shoulder = ask("肩宽（两肩峰点直线距离）")
    hw = ask("头宽（左右最大，含耳廓）")
    hh = ask("头高（下巴尖到头顶）")
    hd = ask("头深（前后最大）", optional=True)
    sh = ask("肩峰离地高", optional=True)

    print("\n【角色设定】")
    n = ask("头身比（X 头身）", 6.0)
    r = ask("角色 头宽 / 肩宽", 0.65)
    dr = ask("角色头 深/宽 比（1.05~1.30）", 1.15)
    hair = ask_bool("头身比里的「头」包含发型吗", True)

    print("\n【假发留量】")
    print("  留量 = 从头壳基底材质的外表面，垂直量到该方向发丝外缘的距离；")
    print("  发套底网 + 定位胶带 + 发丝蓬松量都算在内，壁厚不算（它在壳外皮以内）。")
    wig = ask("基准留量（壳顶外表面 → 发丝最高点）", 3.0)
    fine = ask_bool("侧/前/后 要分别给实测值吗（侧向缺省会照抄顶部值）", False)
    wt = ws_ = wf = wb = None
    if fine:
        wt = ask("  顶：壳顶外表面 → 发丝最高点", wig)
        ws_ = ask("  单侧：最宽处壳外表面 → 该侧发丝最外沿", wig)
        wf = ask("  前：前额壳外表面 → 刘海最前沿", round(wig * WIG_FACTOR["front"], 1))
        wb = ask("  后：后脑壳外表面 → 后发最后沿", round(wig * WIG_FACTOR["back"], 1))

    print("\n【海绵留量】")
    print("  贴在壳内壁的衬垫标称厚度（买来时的厚度）。它和「空气余量」不是一回事：")
    print("  海绵是实体、可压缩、负责夹住头；空气余量是风扇/走线/通风的空隙。")
    foam = ask("基准厚度（顶部会按 1.5 倍派生）", 2.0)
    ffine = ask_bool("顶/侧/面/后 要分别给吗", False)
    ft = fs = ff = fb = None
    if ffine:
        ft = ask("  顶（头台/顶垫）", round(foam * FOAM_FACTOR["top"], 1))
        fs = ask("  单侧（主要夹持衬垫）", round(foam * FOAM_FACTOR["side"], 1))
        ff = ask("  面（要让出视线）", round(foam * FOAM_FACTOR["face"], 1))
        fb = ask("  后脑", round(foam * FOAM_FACTOR["back"], 1))
    fcomp = ask("戴上后的压缩率（0.30 = 压掉 30%）", 0.30)

    print("\n【制作参数】")
    mount = ask_choice("支撑方式", ["shoulder", "chin", "custom"], "shoulder")
    drop = ask("  底沿低于下巴多少 cm", 5.0) if mount == "custom" else 0.0
    lift = ask("增高鞋/内增高厚度", 0.0)
    adv = ask_bool("要调空气余量 / 壁厚 / 肩部填充吗", False)
    kw = {}
    if adv:
        kw["side_clear"] = ask("  单侧空气余量", 1.0)
        kw["top_clear"] = ask("  头顶空气余量", 1.5)
        kw["face_clear"] = ask("  面部方向空气余量", 2.0)
        kw["back_clear"] = ask("  后脑方向空气余量", 1.5)
        kw["wall"] = ask("  壁厚", 0.5)
        kw["foam_density"] = ask("  海绵密度 g/cm³", 0.05)
        kw["shoulder_pad"] = ask("  单侧肩部填充", optional=True)
        kw["material"] = ask_choice("  材料", list(MATERIAL_DENSITY), "pla")

    body = Body(height, weight, shoulder, hw, hh, hd, sh)
    char = Character(n, r, dr, hair)
    build = Build(mount=mount, rim_drop=drop, lift=lift, wig=wig,
                  wig_top=wt, wig_side=ws_, wig_front=wf, wig_back=wb,
                  foam=foam, foam_top=ft, foam_side=fs, foam_face=ff, foam_back=fb,
                  foam_compress=fcomp, **kw)
    print()
    print(render(compute(body, char, build)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="kigurumi 头壳长宽高计算器（不带参数运行 = 交互式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_argument_group("演员数据 (cm/kg)")
    g.add_argument("--height", type=float, help="身高")
    g.add_argument("--weight", type=float, help="体重")
    g.add_argument("--shoulder-width", type=float, help="肩宽")
    g.add_argument("--head-width", type=float, help="头宽")
    g.add_argument("--head-height", type=float, help="头高（下巴到头顶）")
    g.add_argument("--head-depth", type=float, help="头深，缺省按头宽估算")
    g.add_argument("--shoulder-height", type=float, help="肩峰离地高，缺省按身高估算")

    g = p.add_argument_group("角色设定")
    g.add_argument("--ratio", type=float, help="头身比 N")
    g.add_argument("--head-shoulder", type=float, help="角色 头宽/肩宽")
    g.add_argument("--depth-ratio", type=float, default=1.15, help="头 深/宽 比 (默认 1.15)")
    g.add_argument("--no-hair-in-ratio", action="store_true",
                   help="头身比里的「头」不含发型（默认含）")

    g = p.add_argument_group("假发留量（自头壳基底材质外表面量到发丝外缘，cm）")
    g.add_argument("--wig", type=float, default=3.0,
                   help="基准留量，未单独指定的方向由它派生 (默认 3.0)")
    g.add_argument("--wig-top", type=float, help="壳顶外表面 → 发丝最高点")
    g.add_argument("--wig-side", type=float,
                   help="单侧：最宽处壳外表面 → 该侧发丝最外沿（左右方向量）")
    g.add_argument("--wig-front", type=float, help="前额壳外表面 → 刘海最前沿")
    g.add_argument("--wig-back", type=float, help="后脑壳外表面 → 后发最后沿")

    g = p.add_argument_group("海绵留量（贴壳内壁的衬垫标称厚度，cm）")
    g.add_argument("--foam", type=float, default=2.0,
                   help="基准厚度，未单独指定的方向由它派生 (默认 2.0)")
    g.add_argument("--foam-top", type=float, help="头顶（头台/顶垫，通常最厚）")
    g.add_argument("--foam-side", type=float, help="单侧（夹持头部的主要衬垫）")
    g.add_argument("--foam-face", type=float, help="面部方向（要让出视线，通常最薄）")
    g.add_argument("--foam-back", type=float, help="后脑方向")
    g.add_argument("--foam-compress", type=float, default=0.30,
                   help="戴上后的压缩率，实占厚度 = 标称 × (1-它) (默认 0.30)")
    g.add_argument("--foam-density", type=float, default=0.05,
                   help="g/cm³，软质 EVA/海绵约 0.03~0.10 (默认 0.05)")

    g = p.add_argument_group("制作参数")
    g.add_argument("--mount", choices=["shoulder", "chin", "custom"], default="shoulder")
    g.add_argument("--rim-drop", type=float, default=0.0, help="mount=custom 时底沿低于下巴的距离")
    g.add_argument("--lift", type=float, default=0.0, help="增高鞋厚度")
    g.add_argument("--side-clear", type=float, default=1.0, help="单侧空气余量（风扇/走线）")
    g.add_argument("--top-clear", type=float, default=1.5, help="头顶空气余量")
    g.add_argument("--face-clear", type=float, default=2.0, help="面部方向空气余量")
    g.add_argument("--back-clear", type=float, default=1.5, help="后脑方向空气余量")
    g.add_argument("--wall", type=float, default=0.5, help="壁厚 cm")
    g.add_argument("--shoulder-pad", type=float, help="单侧肩部填充，缺省按 BMI 自动")
    g.add_argument("--material", choices=sorted(MATERIAL_DENSITY), default="pla")

    p.add_argument("--json", action="store_true", help="输出 JSON")
    p.add_argument("--explain", action="store_true", help="打印公式与假设后退出")
    return p


def main(argv: List[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.explain:
        print(EXPLAIN)
        return 0

    required = ["height", "weight", "shoulder_width", "head_width", "head_height",
                "ratio", "head_shoulder"]
    if not any(getattr(args, k) is not None for k in required):
        try:
            interactive()
        except (KeyboardInterrupt, EOFError):
            print("\n已取消")
            return 130
        return 0

    missing = [k for k in required if getattr(args, k) is None]
    if missing:
        parser.error("缺少参数：" + ", ".join("--" + m.replace("_", "-") for m in missing))

    body = Body(args.height, args.weight, args.shoulder_width, args.head_width,
                args.head_height, args.head_depth, args.shoulder_height)
    char = Character(args.ratio, args.head_shoulder, args.depth_ratio,
                     not args.no_hair_in_ratio)
    build = Build(mount=args.mount, rim_drop=args.rim_drop, lift=args.lift,
                  wig=args.wig, wig_top=args.wig_top, wig_side=args.wig_side,
                  wig_front=args.wig_front, wig_back=args.wig_back,
                  foam=args.foam, foam_top=args.foam_top, foam_side=args.foam_side,
                  foam_face=args.foam_face, foam_back=args.foam_back,
                  foam_compress=args.foam_compress, foam_density=args.foam_density,
                  side_clear=args.side_clear, top_clear=args.top_clear,
                  face_clear=args.face_clear, back_clear=args.back_clear,
                  wall=args.wall, shoulder_pad=args.shoulder_pad,
                  material=args.material)

    try:
        result = compute(body, char, build)
    except ValueError as e:
        print("错误：%s" % e, file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
