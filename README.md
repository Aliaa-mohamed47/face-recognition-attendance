# Face Recognition Attendance System

Automated attendance system using computer vision and machine learning.
Detects faces, extracts embeddings, classifies identities, and logs attendance automatically.

---

## 📊 Dataset

The model is trained and evaluated using the **AT&T Database of Faces**.

🔗 https://www.kaggle.com/datasets/kasikrit/att-database-of-faces


---

## Project Structure

```
face-recognition-attendance/
│
├── data/
│   ├── raw/                        # Original dataset (ORL/AT&T — 40 subjects × 10 images)
│   └── processed/                  # Filtered images output from Module 1
│
├── modules/                        # Jupyter Notebooks (development & analysis)
│   ├── preprocessing.py            # Module 1 — Ahmed Khaled
│   ├── face_detection.ipynb        # Module 2 — Sama
│   ├── embeddings.ipynb            # Module 3 — Alaa
│   ├── matching.ipynb              # Module 4 — Mayar
│   ├── classification.ipynb        # Module 5 — Waad
│   ├── attendance.py               # Module 6 — Sameh
│   └── database_manager.ipynb      # Module 7 — Ahmed Alaa
│
├── pipeline/                       # Production scripts (integration-ready)
│   ├── main_pipeline.py            # Module 8 — Aliaa  ← RUN THIS
│   ├── database_manager.py         # Module 7 (class-based, importable)
│   └── run_pipeline.py             # Legacy single-image script
│
├── outputs/
│   ├── models/
│   │   └── svm_classifier.pkl      # Trained SVM + LabelEncoder
│   ├── database/
│   │   └── face_embeddings.pkl     # FaceNet512 embeddings for all students
│   ├── cropped_faces/              # Haar Cascade output (Module 2)
│   ├── logs/
│   │   └── attendance_YYYY-MM-DD.csv
│   └── results/
│       ├── preprocessing/          # metrics.csv + filter comparison plots
│       ├── face_detection/         # detection_results.csv + bar chart
│       └── matching/               # matching_results.csv + accuracy plots
│
└── README.md
```

---

## Pipeline Flow

```
Raw Images
    │
    ▼
[Module 1] Preprocessing       → Gaussian/Median filter, PSNR/MSE metrics
    │
    ▼
[Module 2] Face Detection      → Haar Cascade + CLAHE, cropped faces saved
    │
    ▼
[Module 3] Feature Extraction  → FaceNet512 embeddings → face_embeddings.pkl
    │
    ▼
[Module 4] Matching            → Cosine similarity verification
    │
    ▼
[Module 5] Classification      → SVM (RBF, C=10) → svm_classifier.pkl
    │
    ▼
[Module 6] Attendance          → Daily CSV log, duplicate prevention
    │
    ▼
[Module 8] Integration         → main_pipeline.py (image / camera / test)
```

---

## Setup

### Requirements

```bash
pip install opencv-python deepface scikit-learn numpy pandas matplotlib seaborn tqdm
```

### Dataset

Download the [ORL/AT&T Face Database](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces) and place it in `data/raw/` with this structure:

```
data/raw/
├── s1/   (1.pgm, 2.pgm ... 10.pgm)
├── s2/
...
└── s40/
```

---

## How to Run

### Step 1 — Run notebooks in order (first time only)

| Order | Notebook | Output |
|-------|----------|--------|
| 1 | `modules/preprocessing.py` | `data/processed/` |
| 2 | `modules/face_detection.ipynb` | `outputs/cropped_faces/` |
| 3 | `modules/embeddings.ipynb` | `outputs/database/face_embeddings.pkl` |
| 4 | `modules/matching.ipynb` | `outputs/results/matching/` |
| 5 | `modules/classification.ipynb` | `outputs/models/svm_classifier.pkl` |

### Step 2 — Run the main pipeline

```bash
# Test on database (no camera needed)
python pipeline/main_pipeline.py --test

# Run on a single image
python pipeline/main_pipeline.py --image data/raw/s1/1.pgm

# Run live webcam
python pipeline/main_pipeline.py --camera
```

### Attendance log location

```
outputs/logs/attendance_2025-01-15.csv
```

---

## Results Summary

| Module | Method | Metric | Score |
|--------|--------|--------|-------|
| Preprocessing | Gaussian Filter | Avg PSNR | ~31 dB |
| Face Detection | Haar Cascade | Detection Rate | ~85–95% |
| Feature Extraction | FaceNet512 | Embedding Dim | 512-D |
| Matching | Cosine Distance | Accuracy | ~90%+ |
| Classification | SVM (RBF, C=10) | Accuracy | **96.67%** |
| Classification | SVM 5-Fold CV | CV Score | **98.50% ± 0.50%** |

---

## Team

| Module | Task | Member |
|--------|------|--------|
| 1 | Preprocessing | Ahmed Khaled |
| 2 | Face Detection | Sama |
| 3 | Feature Extraction | Alaa |
| 4 | Matching & Verification | Mayar |
| 5 | Classification | Waad |
| 6 | Attendance System | Sameh |
| 7 | Database Management | Ahmed Alaa |
| 8 | Integration & Leadership | Aliaa |