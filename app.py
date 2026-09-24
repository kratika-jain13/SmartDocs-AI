"""SmartDocs AI Streamlit application entry point."""

import streamlit as st
from PIL import Image, UnidentifiedImageError

from scripts.prediction_pipeline import predict_document


def confidence_interpretation(confidence_percentage):
    """Return a plain-language interpretation of model confidence."""
    if confidence_percentage >= 80:
        return "High confidence"
    if confidence_percentage >= 50:
        return "Medium confidence"
    return "Low confidence"


def display_prediction_result(result):
    """Render the prediction result with ranked class probabilities."""
    confidence_percentage = result["confidence_percentage"]
    st.divider()
    st.subheader("Prediction Result")
    result_columns = st.columns(2)
    with result_columns[0]:
        st.metric("Predicted Document Type", result["predicted_class"].title())
    with result_columns[1]:
        st.metric("Model Confidence", f"{confidence_percentage:.2f}%")
    st.caption(
        f"{confidence_interpretation(confidence_percentage)}. "
        "This is model confidence, not model accuracy."
    )

    st.write("Class probability distribution")
    ranked_probabilities = sorted(
        result["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True,
    )
    for class_name, probability in ranked_probabilities:
        st.progress(
            float(probability),
            text=f"{class_name.title()}: {probability * 100:.2f}%",
        )


def main():
    st.set_page_config(page_title="SmartDocs AI", page_icon="📄", layout="centered")
    st.title("SmartDocs AI")
    st.subheader("Document classification")
    st.write("Upload a document image to classify it with the final SmartDocs model.")

    uploaded_file = st.file_uploader(
        "Upload a document image",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, and JPEG.",
    )
    if uploaded_file is None:
        st.info("Choose a PNG, JPG, or JPEG image to begin.")
        return

    try:
        image = Image.open(uploaded_file)
        image.load()
        image = image.convert("RGB")
        st.image(image, caption=uploaded_file.name, use_container_width=True)
        # prediction_pipeline keeps its predictor at module scope, so the model
        # is loaded once per Streamlit process and reused across reruns.
        result = predict_document(image)
    except (UnidentifiedImageError, OSError) as exc:
        st.error(f"Could not read the uploaded image: {exc}")
        return
    except (RuntimeError, TypeError, ValueError) as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.success("Prediction complete")
    display_prediction_result(result)


if __name__ == "__main__":
    main()
