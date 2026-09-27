from PIL import Image
import sys

img = Image.open("/Users/qianhelingxue/.gemini/antigravity/brain/9e9f636b-8435-4c19-a095-25c5c9f7e3d7/.user_uploaded/media_1787860209933.png")
w, h = img.size
# The main character is in the center. Let's crop the middle 40% width.
crop_box = (w * 0.25, 0, w * 0.75, h)
img_cropped = img.crop(crop_box)
img_cropped.save("yinlin_cropped.png")
