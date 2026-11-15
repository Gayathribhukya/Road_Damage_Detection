
# Road Damage Detection System Using YOLOv8

An AI-powered road damage detection application that identifies and classifies road surface damage using YOLOv8.

## Features
- Detects four road damage categories.
- Displays bounding boxes and confidence scores.
- Shows original and annotated images.
- Provides detection summaries and category charts.
- Supports annotated image downloads.

## Technologies
- Python
- YOLOv8
- PyTorch
- OpenCV
- Streamlit
- NumPy
- Pandas

## Damage Categories

| Class | Category |
|---|---|
| D00 | Longitudinal Crack |
| D10 | Transverse Crack |
| D20 | Alligator Crack |
| D40 | Pothole |

## Dataset
Roboflow Road Damage Dataset.

- Training images: 3,360
- Validation images: 716
- Testing images: 727

## Model Evaluation

| Metric | Score |
|---|---:|
| Precision | 41.45% |
| Recall | 54.17% |
| mAP@50 | 47.76% |
| mAP@50-95 | 24.25% |

## Dashboard Screenshots

### Main Dashboard
![Dashboard](screenshots/dashboard.png)

### Pothole Detection
![Pothole Detection](screenshots/pothole_detection.png)

### Crack Detection
![Crack Detection](screenshots/crack_detection.png)

## Installation

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Road-Damage-Detection
pip install -r requirements.txt
streamlit run app.py
