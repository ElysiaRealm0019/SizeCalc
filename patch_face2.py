with open("char_ratio_from_image.py", "r") as f:
    code = f.read()

# Replace the signature correctly
code = code.replace("def analyse(w: int, h: int, mask: bytearray, ov: Dict[str, Optional[float]],\n            shoulder_band: float = 0.6) -> Dict[str, Any]:", 
                    "def analyse(w: int, h: int, mask: bytearray, ov: Dict[str, Optional[float]],\n            shoulder_band: float = 0.6, face_rect=None) -> Dict[str, Any]:")

with open("char_ratio_from_image.py", "w") as f:
    f.write(code)
