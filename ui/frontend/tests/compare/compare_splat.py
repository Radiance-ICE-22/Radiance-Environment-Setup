"""Diff drei's <Splat> against our SplatMesh on the same synthetic splat (cloud check).
Serve tests/compare with vite on :5199 first. Prints per-pixel stats and writes both PNGs."""
import io, sys, time
from playwright.sync_api import sync_playwright
from PIL import Image, ImageChops
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "."
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1000, "height": 400})
    logs = []
    pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
    pg.goto("http://127.0.0.1:5199/")
    pg.wait_for_function("window.__ours === true", timeout=60000)
    time.sleep(4)
    shots = [Image.open(io.BytesIO(pg.locator(f"#{s} canvas").screenshot())).convert("RGB") for s in ("left", "right")]
    b.close()
for s, im in zip(("drei", "ours"), shots):
    im.save(f"{out}/compare_{s}.png")
a, c = (np.asarray(x).astype(int) for x in shots)
d = np.abs(a - c)
print("mean abs diff", round(d.mean(), 3), "| pixels differing > 8:", round(100 * (d.max(2) > 8).mean(), 3), "% | max", d.max())
print("non-black pixels: drei", round(100 * (a.max(2) > 10).mean(), 1), "% ours", round(100 * (c.max(2) > 10).mean(), 1), "%")
for l in logs[:10]: print(l)
