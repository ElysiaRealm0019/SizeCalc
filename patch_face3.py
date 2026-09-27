with open("char_ratio_from_image.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if line.strip() == "auto_chin = None":
        if i + 1 < len(lines) and "chin_src =" in lines[i+1]:
            # This is the start of the block.
            skip = True
    
    if skip and line.strip().startswith("chin_y = int(ov[\"chin_y\"])"):
        skip = False
        new_lines.append("""    auto_chin = None
    chin_src = "检不出，回退到颈最窄线"
    
    if face_rect is not None:
        auto_chin = int(face_rect[1] + face_rect[3])
        chin_src = "OpenCV 动漫面部识别"
    elif neck_w >= head_top_max * 0.80:
        warn.append(
            "找不到明显的颈部收窄（最窄处 %d px 已是头宽 %d px 的 %.0f%%）——"
            "多半是长发盖住了脖子。已回退为面部识别或人工修正。"
            % (neck_w, head_top_max, 100.0 * neck_w / max(1, head_top_max))
        )
    else:
        widest_y = max(range(top_y, neck_y + 1), key=lambda y: width[y])
        thr = neck_w * 1.08
        auto_chin = neck_y
        for y in range(widest_y, neck_y + 1):
            if width[y] <= thr:
                auto_chin = y
                break
        chin_src = "轮廓几何寻颈"

""")
    
    if not skip:
        new_lines.append(line)

with open("char_ratio_from_image.py", "w") as f:
    f.writelines(new_lines)
