
import streamlit as st
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO
from io import BytesIO
import os

# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)

# --------------------------------
# 2. MODEL CONFIGURATION
# --------------------------------

MODEL_PATH = "/content/gdrive/MyDrive/Road_Damage_Project/models/final_road_damage.pt"


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return YOLO(MODEL_PATH)


# --------------------------------
# 3. APPLICATION HEADER
# --------------------------------

st.title("🚧 Road Damage Detection System")

st.markdown("""
### AI-Powered Road Inspection

Upload a road image to identify road damage using
a custom-trained YOLOv8 object detection model.
""")

st.divider()


# --------------------------------
# 4. SIDEBAR SETTINGS
# --------------------------------

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05
)

image_size = st.sidebar.selectbox(
    "Image Size",
    [320, 480, 640, 800],
    index=2
)

st.sidebar.info(
    "The model identifies four road damage categories."
)


# --------------------------------
# 5. CLASS INFORMATION
# --------------------------------

CLASS_NAMES = {
    0: "D00 - Longitudinal Crack",
    1: "D10 - Transverse Crack",
    2: "D20 - Alligator Crack",
    3: "D40 - Pothole"
}


# --------------------------------
# 6. LOAD MODEL
# --------------------------------

try:

    model = load_model()

    st.sidebar.success(
        "YOLOv8 model loaded successfully!"
    )

except Exception as e:

    st.error(f"Model loading error: {e}")
    st.stop()


# --------------------------------
# 7. IMAGE UPLOAD
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Road Image")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )


    # --------------------------------
    # 8. OBJECT DETECTION
    # --------------------------------

    if st.button(
        "Detect Road Damage",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing road image..."):

            # Run YOLOv8 prediction
            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=image_size,
                iou=0.5,
                max_det=100,
                agnostic_nms=False,
                verbose=False
            )

            result = results[0]

            # Generate annotated image
            annotated_image = result.plot()

            annotated_image = cv2.cvtColor(
                annotated_image,
                cv2.COLOR_BGR2RGB
            )


            # --------------------------------
            # 9. DISPLAY PREDICTIONS
            # --------------------------------

            with col2:

                st.image(
                    annotated_image,
                    caption="Detected Road Damage",
                    use_container_width=True
                )


            # --------------------------------
            # 10. EXTRACT DETECTION DETAILS
            # --------------------------------

            detections = []

            for box in result.boxes:

                class_id = int(box.cls[0].item())

                confidence_score = float(
                    box.conf[0].item()
                )

                coordinates = (
                    box.xyxy[0].cpu().numpy().tolist()
                )

                detections.append({

                    "Damage Type": CLASS_NAMES.get(
                        class_id,
                        "Unknown"
                    ),

                    "Confidence": round(
                        confidence_score * 100,
                        2
                    ),

                    "X1": round(coordinates[0], 2),
                    "Y1": round(coordinates[1], 2),
                    "X2": round(coordinates[2], 2),
                    "Y2": round(coordinates[3], 2)

                })


            # --------------------------------
            # 11. DETECTION SUMMARY
            # --------------------------------

            st.divider()

            st.subheader("Detection Summary")

            total_detections = len(detections)


            if total_detections > 0:

                detection_df = pd.DataFrame(detections)

                unique_damage_types = (
                    detection_df["Damage Type"].nunique()
                )

                avg_confidence = (
                    detection_df["Confidence"].mean()
                )


                # Display metrics
                metric1, metric2, metric3 = st.columns(3)

                metric1.metric(
                    "Total Detections",
                    total_detections
                )

                metric2.metric(
                    "Damage Categories",
                    unique_damage_types
                )

                metric3.metric(
                    "Average Confidence",
                    f"{avg_confidence:.2f}%"
                )


                # Detection table
                st.subheader(
                    "Individual Detection Results"
                )

                st.dataframe(
                    detection_df,
                    use_container_width=True,
                    hide_index=True
                )


                # Category distribution
                st.subheader(
                    "Damage Category Distribution"
                )

                category_counts = (
                    detection_df["Damage Type"]
                    .value_counts()
                )

                st.bar_chart(category_counts)


            else:

                st.warning(
                    "No road damage detected at the selected "
                    "confidence threshold. Try lowering the threshold."
                )


            # --------------------------------
            # 12. DOWNLOAD ANNOTATED IMAGE
            # --------------------------------

            image_buffer = BytesIO()

            Image.fromarray(annotated_image).save(
                image_buffer,
                format="PNG"
            )

            st.download_button(
                label="Download Annotated Image",
                data=image_buffer.getvalue(),
                file_name="road_damage_detection.png",
                mime="image/png"
            )


# --------------------------------
# 13. FOOTER
# --------------------------------

st.divider()

st.caption(
    "Road Damage Detection System | "
    "YOLOv8n | PyTorch | OpenCV | Streamlit"
)
