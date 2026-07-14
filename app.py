"""
app.py
======
Main Streamlit entry point for the AI Human Activity Recognition system.
Provides two prediction modes (uploaded video, live webcam), a sidebar
with model/dataset/performance info, and the bonus UX features (timeline,
CSV export, report download, prediction history, dark theme).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

import config
from deployment.streamlit_utils import (
    inject_dark_theme,
    render_activity_timeline,
    render_metric_cards,
    render_probability_bar_chart,
    build_text_report,
    predictions_to_csv,
)
from deployment.video_predict import (
    cleanup_temp_file,
    predict_uploaded_video,
    render_annotated_video,
    save_uploaded_file,
)
from deployment.webcam import WebcamSession
from models.utils import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title=config.APP_CONFIG.page_title,
    page_icon=config.APP_CONFIG.page_icon,
    layout=config.APP_CONFIG.layout,
)
inject_dark_theme()


# --------------------------------------------------------------------------- #
# Session state initialization
# --------------------------------------------------------------------------- #
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False
if "predictor" not in st.session_state:
    st.session_state.predictor = None


@st.cache_resource(show_spinner="Loading model...")
def get_predictor():
    """Load (and cache across reruns) the ``ActivityPredictor``.

    Returns:
        A ready-to-use ``ActivityPredictor``, or ``None`` if no trained
        model checkpoint exists yet.
    """
    from models.inference import ActivityPredictor

    model_path = config.TRAINING_CONFIG.checkpoint_path
    if not os.path.exists(model_path):
        return None
    try:
        return ActivityPredictor(model_path=model_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load predictor: %s", exc)
        return None


def log_prediction(activity: str, confidence: float, source: str) -> None:
    """Append a prediction to the in-session history table.

    Args:
        activity: Predicted activity label.
        confidence: Confidence score (0-1).
        source: Where the prediction came from (``"video"`` or
            ``"webcam"``).
    """
    st.session_state.prediction_history.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "activity": activity,
            "confidence": round(confidence, 4),
            "source": source,
        }
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🏃 HAR System")
    st.caption("MediaPipe Pose + LSTM")

    st.subheader("📦 Model Information")
    st.markdown(
        f"""
        - **Architecture:** `{config.MODEL_CONFIG.architecture.upper()}`
        - **Sequence length:** {config.SEQUENCE_LENGTH} frames
        - **Features/frame:** {config.FEATURES_PER_FRAME} (33 landmarks × 4)
        - **Classes:** {config.NUM_CLASSES}
        """
    )

    st.subheader("🗂️ Dataset Information")
    st.markdown(
        f"""
        - **Source:** {config.DATASET_CONFIG.name}
        - **Activities:** {", ".join(config.ACTIVITY_CLASSES)}
        """
    )

    st.subheader("📊 Performance Metrics")
    metrics_path = Path(config.REPORTS_DIR) / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            saved_metrics = json.load(f)
        render_metric_cards(
            {
                "Accuracy": saved_metrics.get("accuracy", 0),
                "Precision": saved_metrics.get("precision_macro", 0),
                "Recall": saved_metrics.get("recall_macro", 0),
                "F1": saved_metrics.get("f1_macro", 0),
            }
        )
    else:
        st.info("No evaluation report yet. Run `training/evaluate.py` after training.")

    st.divider()
    confidence_threshold = st.slider(
        "Confidence Threshold", 0.0, 1.0, config.APP_CONFIG.default_confidence_threshold, 0.05
    )

    if st.session_state.prediction_history:
        st.subheader("🕘 Prediction History")
        st.dataframe(st.session_state.prediction_history, use_container_width=True, height=200)
        csv_bytes = predictions_to_csv(st.session_state.prediction_history)
        st.download_button("⬇️ Export History CSV", csv_bytes, "prediction_history.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.markdown(
    "<h1 style='margin-bottom:0'>AI Human Activity Recognition</h1>"
    "<p style='color:#9ca3af'>Deep learning-powered activity classification from video or webcam</p>",
    unsafe_allow_html=True,
)

predictor = get_predictor()
if predictor is None:
    st.warning(
        "⚠️ No trained model found at `saved_models/best_model.keras`. "
        "Run the preprocessing pipeline and `training/train.py` first, then reload this app. "
        "The UI below is fully functional and will activate automatically once a model exists."
    )

tab_video, tab_webcam = st.tabs(["📹 Upload Video", "🎥 Live Webcam"])

# ------------------------------ Video tab --------------------------------- #
with tab_video:
    st.subheader("Upload a video for activity recognition")
    uploaded_file = st.file_uploader(
        "Choose a video file", type=["mp4", "avi", "mov", "mkv", "webm"]
    )
    show_timeline = st.checkbox("Generate activity timeline (sliding window)", value=False)
    save_annotated = st.checkbox("Save annotated prediction video", value=False)

    if uploaded_file is not None:
        st.video(uploaded_file)

    predict_clicked = st.button("🔮 Predict", type="primary", disabled=(uploaded_file is None or predictor is None))

    if predict_clicked and uploaded_file is not None and predictor is not None:
        video_path = save_uploaded_file(uploaded_file)
        progress = st.progress(0, text="Extracting pose landmarks...")

        try:
            progress.progress(30, text="Running inference...")
            overall, timeline = predict_uploaded_video(predictor, video_path, timeline=show_timeline)
            progress.progress(80, text="Rendering results...")

            activity, confidence = overall["activity"], overall["confidence"]
            log_prediction(activity, confidence, source="video")

            flag = "✅" if confidence >= confidence_threshold else "⚠️ low confidence"
            st.markdown(
                f"<div class='prediction-badge'>{activity} — {confidence:.1%} {flag}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Inference time: {overall['inference_time_sec']}s")

            st.subheader("Class Probabilities")
            render_probability_bar_chart(overall["probabilities"])

            if timeline:
                st.subheader("Activity Timeline")
                render_activity_timeline(timeline)

            report_text = build_text_report(
                activity, confidence, overall["probabilities"],
                extra={"filename": uploaded_file.name, "inference_time_sec": overall["inference_time_sec"]},
            )
            st.download_button("⬇️ Download Report", report_text, "prediction_report.txt", "text/plain")

            if save_annotated:
                out_path = str(Path(config.OUTPUTS_DIR) / f"annotated_{Path(video_path).stem}.mp4")
                with st.spinner("Rendering annotated video..."):
                    render_annotated_video(video_path, out_path, activity, confidence)
                with open(out_path, "rb") as f:
                    st.download_button("⬇️ Download Annotated Video", f, Path(out_path).name, "video/mp4")

            progress.progress(100, text="Done")
        finally:
            cleanup_temp_file(video_path)

# ------------------------------ Webcam tab --------------------------------- #
with tab_webcam:
    st.subheader("Real-time activity recognition")
    col_start, col_stop = st.columns(2)
    start_clicked = col_start.button("▶️ Start Webcam", disabled=predictor is None)
    stop_clicked = col_stop.button("⏹️ Stop Webcam")

    if start_clicked:
        st.session_state.webcam_running = True
    if stop_clicked:
        st.session_state.webcam_running = False

    frame_placeholder = st.empty()
    info_placeholder = st.empty()
    chart_placeholder = st.empty()

    if st.session_state.webcam_running and predictor is not None:
        try:
            with WebcamSession(predictor, confidence_threshold=confidence_threshold) as session:
                while st.session_state.webcam_running:
                    frame, prediction, fps = session.read_and_predict()
                    if frame is None:
                        st.error("Webcam feed unavailable.")
                        break

                    frame_placeholder.image(frame, channels="RGB", use_container_width=True)

                    if prediction is not None:
                        activity, confidence = prediction["activity"], prediction["confidence"]
                        flag = "✅" if confidence >= confidence_threshold else "⚠️"
                        info_placeholder.markdown(
                            f"**Activity:** {activity} {flag}  |  **Confidence:** {confidence:.1%}  |  **FPS:** {fps:.1f}"
                        )
                        with chart_placeholder.container():
                            render_probability_bar_chart(prediction["probabilities"])
                        if confidence >= confidence_threshold:
                            log_prediction(activity, confidence, source="webcam")
                    else:
                        info_placeholder.markdown(f"Buffering frames... | **FPS:** {fps:.1f}")

                    time.sleep(max(0, 1.0 / config.APP_CONFIG.webcam_fps_target - 0.001))
        except IOError as exc:
            st.error(f"Could not access webcam: {exc}")
            st.session_state.webcam_running = False
    else:
        st.info("Click **Start Webcam** to begin real-time recognition.")


st.divider()
st.caption(
    "AI Human Activity Recognition · MediaPipe Pose + LSTM · Built with TensorFlow & Streamlit"
)
