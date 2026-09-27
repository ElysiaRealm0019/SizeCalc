import re
import sys

with open("char_ratio_from_image.py", "r") as f:
    code = f.read()

# Add cv2 import
code = code.replace("import argparse\nimport json", "import argparse\nimport json\nimport cv2")

# We need to detect face inside main() or load_image(), but it's best done in main() right after loading image.
# Let's add face detection logic in main().
face_logic = """
    # --- 面部识别回退逻辑 ---
    face_rect = None
    try:
        import cv2
        import numpy as np
        cascade_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lbpcascade_animeface.xml")
        if os.path.exists(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
            # rgba to grayscale for cv2
            np_img = np.array(rgba).reshape((h, w, 4))
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGBA2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(24, 24))
            if len(faces) > 0:
                # 找最靠上的脸（如果有多个）
                faces = sorted(faces, key=lambda f: f[1])
                face_rect = faces[0]
    except Exception as e:
        pass
"""

# Insert face_logic before analyse() call in main()
code = code.replace("m = analyse(w, h, mask, ov, a.shoulder_band)", face_logic + "\n    m = analyse(w, h, mask, ov, a.shoulder_band, face_rect)")

# Modify analyse signature
code = code.replace("def analyse(w: int, h: int, mask: bytearray, ov: Dict[str, Optional[float]], shoulder_band: float = 0.6) -> Dict[str, Any]:", 
                    "def analyse(w: int, h: int, mask: bytearray, ov: Dict[str, Optional[float]], shoulder_band: float = 0.6, face_rect=None) -> Dict[str, Any]:")

# Modify chin logic in analyse()
old_chin_logic = """    auto_chin = None
    if head_w_y is not None and neck_y is not None:
        auto_chin = neck_y  # 简单取颈部最窄处为下巴"""

new_chin_logic = """    auto_chin = None
    chin_src = "检不出，回退到颈最窄线"
    if face_rect is not None:
        auto_chin = int(face_rect[1] + face_rect[3])
        chin_src = "OpenCV 动漫面部识别"
    elif head_w_y is not None and neck_y is not None:
        auto_chin = neck_y
        chin_src = "轮廓几何寻颈"
"""
code = code.replace(old_chin_logic, new_chin_logic)

# Remove the old dict entry for chin_src
old_chin_src_dict = """        "chin_src": ("手动指定" if ov.get("chin_y") is not None
                     else ("自动检出" if auto_chin is not None else "检不出，回退到颈最窄线")),"""

new_chin_src_dict = """        "chin_src": "手动指定" if ov.get("chin_y") is not None else chin_src,"""
code = code.replace(old_chin_src_dict, new_chin_src_dict)

with open("char_ratio_from_image.py", "w") as f:
    f.write(code)
