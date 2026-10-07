import streamlit as st
from PIL import Image
from analyzer import analyze_image

st.set_page_config(page_title="TRUTHPIX", page_icon="🔍", layout="wide")

st.title("TRUTHPIX — Image Tampering Detector")
st.caption("Upload an image to check whether it may have been edited.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)

    with st.spinner("Analyzing image..."):
        r = analyze_image(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("ELA Heatmap")
        if r["heatmap"] is not None:
            st.image(r["heatmap"], use_container_width=True)
        else:
            st.warning("Heatmap not available.")

    st.divider()

    st.subheader("AI Analysis")
    c1, c2 = st.columns(2)
    c1.metric(
        "AI Score", f"{r['ai_score']:.0f}%" if r["ai_score"] is not None else "N/A")
    c2.metric(
        "ELA Score", f"{r['ela_score']:.0f}%" if r["ela_score"] is not None else "N/A")

    st.divider()

    st.subheader("Final Result")
    if r["risk"] is not None:
        icon = "🔴" if r["risk"] >= 65 else "🟠" if r["risk"] >= 40 else "🟢"
        st.header(f"{icon} {r['verdict']}")
        st.write(f"**Tampering Risk: {r['risk']:.0f}%**")
        st.progress(min(int(r["risk"]), 100))
    else:
        st.error("Analysis failed.")

    st.write("**Evidence:**")
    for e in r["evidence"]:
        st.write(f"✓ {e}")

    for err in r["errors"]:
        st.warning(err)

    st.info("This is a screening tool, not proof. Images that were resized or "
            "recompressed (WhatsApp, screenshots, social media) can give false alarms. "
            "Look at the heatmap for regions that glow differently from their surroundings.")
