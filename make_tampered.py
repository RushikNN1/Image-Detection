import os
import random
from PIL import Image

random.seed(1)
src, dst = "dataset/real", "dataset/tampered"
os.makedirs(dst, exist_ok=True)

names = sorted(n for n in os.listdir(
    src) if n.lower().endswith((".jpg", ".jpeg", ".png")))

for i, name in enumerate(names):
    img = Image.open(os.path.join(src, name)).convert("RGB")
    donor = Image.open(os.path.join(
        src, names[(i + 1) % len(names)])).convert("RGB")
    w, h = img.size

    # Take a patch from a DIFFERENT image and paste it in (splicing)
    pw, ph = w // 3, h // 3
    dw, dh = donor.size
    x0, y0 = random.randint(0, dw - dw // 3), random.randint(0, dh - dh // 3)
    patch = donor.crop((x0, y0, x0 + dw // 3, y0 + dh // 3)).resize((pw, ph))
    x, y = random.randint(0, w - pw), random.randint(0, h - ph)
    img.paste(patch, (x, y))

    # Save only once, at a similar quality, so the splice isn't wiped out
    img.save(os.path.join(dst, "fake_" + name), "JPEG", quality=98)
    print("Created fake_" + name)
