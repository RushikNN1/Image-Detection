import os
import csv
import time
from PIL import Image
from analyzer import analyze_image

VALID = (".jpg", ".jpeg", ".png")
rows = []
correct = total = crashes = 0

for folder, expected_tampered in [("real", False), ("tampered", True)]:
    path = os.path.join("dataset", folder)
    for name in sorted(os.listdir(path)):
        if not name.lower().endswith(VALID):
            continue
        try:
            start = time.time()
            r = analyze_image(Image.open(os.path.join(path, name)))
            seconds = time.time() - start

            risk = r["risk"]
            predicted_tampered = risk is not None and risk >= 40
            ok = predicted_tampered == expected_tampered
            correct += ok
            total += 1

            if r["heatmap"] is not None:
                r["heatmap"].save(os.path.join(
                    "results", f"{folder}_{name}.png"))

            ela = f"{r['ela_score']:.0f}" if r["ela_score"] is not None else "N/A"
            ai = f"{r['ai_score']:.0f}" if r["ai_score"] is not None else "N/A"
            rows.append([folder, name, ela, ai, f"{risk:.0f}", r["verdict"],
                         "OK" if ok else "WRONG", f"{seconds:.1f}s"])
            print(f"{folder:9} {name:22} ELA={ela:>3} AI={ai:>3} risk={risk:>3.0f} "
                  f"{r['verdict']:20} {'OK' if ok else 'WRONG'}")
        except Exception as e:
            crashes += 1
            rows.append([folder, name, "", "", "", "CRASH", str(e), ""])
            print(f"{folder:9} {name:22} CRASH: {e}")

with open("results/results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["type", "image", "ela", "ai",
                    "risk", "verdict", "correct", "time"])
    writer.writerows(rows)

print(f"\nAccuracy: {correct}/{total}   Crashes: {crashes}")
print("Table saved to results/results.csv, heatmaps saved in results/")
