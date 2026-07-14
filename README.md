# AI Human Activity Recognition

Deep-learning system that recognizes human activities — **Walking, Running,
Standing, Sitting, Jumping, Clapping, Waving, Falling** — from uploaded
videos or a live webcam, using **MediaPipe Pose + LSTM**, deployed as a
**Streamlit** app.

---

## ✨ Features

- Pose-based activity classification (33 MediaPipe landmarks → LSTM → softmax)
- Upload-a-video prediction with confidence score, probability bar chart, and optional activity timeline
- Live webcam prediction with skeleton overlay, FPS counter, and adjustable confidence threshold
- Swappable model backbone: LSTM, CNN+LSTM, 3D CNN, or MoveNet+LSTM — no app changes required
- Full preprocessing pipeline: dataset download → frame extraction → pose landmark extraction → sequence generation (with augmentation) → train/val/test split
- Training pipeline with early stopping, LR scheduling, checkpointing, resume support, and TensorBoard
- Evaluation: accuracy, precision, recall, F1, confusion matrix, ROC curves, classification report
- Bonus UX: dark theme, CSV export, downloadable text report, downloadable annotated video, prediction history

---

## 🏗️ Architecture

```
Video / Webcam
      │
      ▼
Frame Extraction
      │
      ▼
MediaPipe Pose Detection  ──► 33 body landmarks (x, y, z, visibility)
      │
      ▼
Normalization (hip-centered, torso-scaled)
      │
      ▼
Sequence Generator (30 frames × 132 features)
      │
      ▼
LSTM(128) → Dropout → LSTM(64) → Dense(64) → Dropout → Dense(num_classes)
      │
      ▼
Softmax Classification
      │
      ▼
Predicted Activity + Confidence
```

The backbone is pluggable via `config.MODEL_CONFIG.architecture`
(`"lstm"`, `"cnn_lstm"`, `"conv3d"`, `"movenet_lstm"`) — see
`models/lstm_model.py::ModelFactory`.

---

## 📁 Folder Structure

```
HumanActivityRecognition/
├── app.py                     # Streamlit entry point
├── config.py                  # All paths, hyperparameters, class list
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/{raw,processed,train,val,test}
│
├── models/
│   ├── lstm_model.py           # Model architecture factory
│   ├── mediapipe_extractor.py  # Pose extraction
│   ├── inference.py            # ActivityPredictor (batch + streaming)
│   └── utils.py                # Logging, seeding, pickle I/O, normalization
│
├── preprocessing/
│   ├── download_dataset.py
│   ├── extract_frames.py
│   ├── generate_landmarks.py
│   ├── create_sequences.py     # + augmentation
│   └── split_dataset.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── callbacks.py
│
├── deployment/
│   ├── webcam.py
│   ├── video_predict.py
│   └── streamlit_utils.py
│
├── saved_models/{best_model.keras, label_encoder.pkl}
└── outputs/{confusion_matrix.png, accuracy.png, loss.png, reports/}
```

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd HumanActivityRecognition
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A GPU (CUDA-enabled) is strongly recommended for training but not required
for inference or running the app with a pre-trained model.

---

## 📦 Dataset Preparation

1. **Download** (large — several GB; run somewhere with disk space and a
   stable connection):
   ```bash
   python -m preprocessing.download_dataset --dataset ucf101
   ```
   This downloads, extracts, and trims the archive down to only the class
   folders mapped in `config.DATASET_CONFIG`. **Note:** UCF101/HMDB51 don't
   natively contain every class in `config.ACTIVITY_CLASSES` (e.g.
   "Falling" isn't a UCF101 category) — update `ucf101_class_map` in
   `config.py` or supplement with your own clips per class folder.

2. **Extract frames:**
   ```bash
   python -m preprocessing.extract_frames --input data/raw/ucf101 --frames-per-video 60
   ```

3. **Generate pose landmarks:**
   ```bash
   python -m preprocessing.generate_landmarks
   ```

4. **Build fixed-length sequences (with augmentation):**
   ```bash
   python -m preprocessing.create_sequences
   ```

5. **Split into train/val/test:**
   ```bash
   python -m preprocessing.split_dataset
   ```

### Bring your own data
Skip step 1 and instead organize your own clips as:
```
data/raw/my_dataset/<ActivityName>/<video files>
```
then run steps 2–5 pointing `--input` at `data/raw/my_dataset`.

---

## 🏋️ Training

```bash
python -m training.train --epochs 100 --batch-size 32 --lr 1e-3
```

Resume from a checkpoint:
```bash
python -m training.train --resume saved_models/best_model.keras --epochs 50
```

Monitor with TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```

Training automatically applies early stopping, LR reduction on plateau,
and saves the best checkpoint to `saved_models/best_model.keras`.

---

## 📊 Evaluation

```bash
python -m training.evaluate
```

Generates `outputs/confusion_matrix.png`, `outputs/reports/roc_curves.png`,
`outputs/reports/test_metrics.json`, and prints the full classification
report.

---

## 🔮 Inference

Programmatic use:
```python
from models.inference import ActivityPredictor

predictor = ActivityPredictor()
result = predictor.predict_video("path/to/video.mp4")
print(result["activity"], result["confidence"])
```

---

## 🚀 Deployment (Streamlit)

```bash
streamlit run app.py
```

Open the printed local URL. Use the **Upload Video** tab for file-based
prediction or the **Live Webcam** tab for real-time recognition. The
sidebar shows model/dataset info and — once you've run
`training/evaluate.py` — live performance metrics.

---

## 🖼️ Screenshots

*(Add screenshots here after running the app, e.g. `docs/screenshot_upload.png`, `docs/screenshot_webcam.png`)*

---

## 🔭 Future Improvements

- Multi-person tracking and per-person activity streams
- Transformer-based temporal head (replace/augment LSTM)
- On-device export (TFLite) for mobile/edge inference
- Active-learning loop to mine hard examples from prediction history
- Expand class map with a proper "Falling" dataset (e.g. UR Fall Detection Dataset)

---

## ⚠️ Known Limitations

- The UCF101/HMDB51 class mapping in `config.py` is a starting point, not
  a verified 1:1 match for every activity (notably "Falling" and
  "Waving") — verify/replace mappings or supply your own clips before
  training for production use.
- Webcam prediction in Streamlit uses a synchronous read loop (Streamlit
  doesn't support persistent background threads across reruns cleanly);
  FPS will depend on your machine and model complexity.
