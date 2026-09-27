with open("char_ratio_from_image.py", "r") as f:
    code = f.read()

import re

# We need to completely replace the logic for top_y, chin_y, and head_h if face_rect is present.
# Let's find the section.
start_marker = 'top_y, foot_y = rows[0], rows[-1]'
end_marker = 'head_h = chin_y - top_y + 1'

new_logic = """top_y, foot_y = rows[0], rows[-1]
    
    # --- 核心：基于面部识别锚定人类头骨（无视兽耳/呆毛/帽子） ---
    chin_src = "检不出，回退到颈最窄线"
    if face_rect is not None:
        fx, fy, fw, fh = face_rect
        chin_y = int(fy + fh)
        # 动漫人脸框通常是从额头到下巴，人类头骨顶部大约在框顶上方 25% 的位置
        inferred_top = int(fy - fh * 0.25)
        top_y = max(0, inferred_top)
        chin_src = "OpenCV 面部锚定 (无视兽耳/帽子)"
        neck_y = chin_y
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
        
    head_h = chin_y - top_y + 1"""

pattern = re.compile(re.escape(start_marker) + r'.*?' + re.escape(end_marker), re.DOTALL)
code = pattern.sub(new_logic.replace('\\', '\\\\'), code)

with open("char_ratio_from_image.py", "w") as f:
    f.write(code)
