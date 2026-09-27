with open("char_ratio_from_image.py", "r") as f:
    code = f.read()

import re
code = code.replace('head_h = chin_y - top_y + 1', 'head_h = chin_y - top_y + 1\n    total = foot_y - top_y + 1')

with open("char_ratio_from_image.py", "w") as f:
    f.write(code)
