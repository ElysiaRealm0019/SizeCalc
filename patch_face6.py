with open("char_ratio_from_image.py", "r") as f:
    code = f.read()

code = code.replace('neck_y = chin_y', 'neck_y = chin_y\n        neck_w = width[neck_y] if neck_y < h else 0')

with open("char_ratio_from_image.py", "w") as f:
    f.write(code)
