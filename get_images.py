import urllib.request
import os

urls = {
    "Nahida": "https://raw.githubusercontent.com/FortOfFans/GI/main/characters/Nahida/Portrait.png",
    "Zhongli": "https://raw.githubusercontent.com/FortOfFans/GI/main/characters/Zhongli/Portrait.png",
    "Raiden": "https://raw.githubusercontent.com/FortOfFans/GI/main/characters/Raiden_Shogun/Portrait.png"
}
os.makedirs("test_images", exist_ok=True)
for name, url in urls.items():
    try:
        urllib.request.urlretrieve(url, f"test_images/{name}.png")
        print(f"Downloaded {name}")
    except Exception as e:
        print(f"Failed {name}: {e}")
