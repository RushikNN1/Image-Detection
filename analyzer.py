from PIL import Image
from ela import perform_ela
from ai_detector import predict_image


def analyze_image(image: Image.Image) -> dict:
    image = image.convert("RGB")
    errors = []

    # ---- Step 1: ELA (returns heatmap + score between 0 and 1) ----
    heatmap, ela_score = None, None
    try:
        heatmap, ela_score = perform_ela(image)
        ela_score = float(ela_score) * 100      # turn 0-1 into 0-100
    except Exception as e:
        errors.append(f"ELA failed: {e}")

    # ---- Step 2: AI model (returns label + confidence 0-100) ----
    ai_score = None   # = chance the image is TAMPERED
    try:
        result = predict_image(image)
        conf = float(result["confidence"])
        # confidence is for the predicted label, so flip it if label is "real"
        ai_score = conf if result["label"] == "tampered" else 100 - conf
    except Exception as e:
        errors.append(f"AI failed: {e}")

    # ---- Step 3: combine into one risk score ----
    if ai_score is not None and ela_score is not None:
        risk = 0.6 * ai_score + 0.4 * ela_score
    elif ela_score is not None:        # fallback: AI broke
        risk = ela_score
    elif ai_score is not None:
        risk = ai_score
    else:
        risk = None

    # ---- Step 4: verdict + evidence ----
    if risk is None:
        verdict = "ANALYSIS FAILED"
    elif risk >= 65:
        verdict = "POTENTIAL TAMPERING"
    elif risk >= 40:
        verdict = "SUSPICIOUS"
    else:
        verdict = "LIKELY REAL"

    evidence = []
    if ela_score is not None and ela_score >= 50:
        evidence.append("Compression inconsistency")
    if ai_score is not None and ai_score >= 50:
        evidence.append("AI model flagged manipulation")
    if not evidence:
        evidence.append("No strong signs of manipulation")

    return {
        "heatmap": heatmap,
        "ela_score": ela_score,
        "ai_score": ai_score,
        "risk": risk,
        "verdict": verdict,
        "evidence": evidence,
        "errors": errors,
    }
