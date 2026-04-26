"""
Module 8 — Integration & Main Pipeline
Author: Aliaa
Description: Connects all modules end-to-end.
             Supports both image-file mode and live webcam mode.

Usage:
    python main_pipeline.py --image path/to/image.jpg
    python main_pipeline.py --camera
    python main_pipeline.py --test          (runs on database mean embeddings)
"""

import os
import sys
import cv2
import pickle
import argparse
import numpy as np
from datetime import datetime
from sklearn.preprocessing import normalize
from deepface import DeepFace

# ─── Project root (works from any working directory) ─────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(ROOT) == "pipeline":
    ROOT = os.path.dirname(ROOT)

sys.path.insert(0, os.path.join(ROOT, "pipeline"))

# ─── Paths ───────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "outputs", "models",   "svm_classifier.pkl")
DB_PATH    = os.path.join(ROOT, "outputs", "database", "face_embeddings.pkl")
LOG_DIR    = os.path.join(ROOT, "outputs", "logs")
TEMP_FACE  = os.path.join(ROOT, "outputs", "temp_face.jpg")

os.makedirs(LOG_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# Step 1 — Load model & database
# ════════════════════════════════════════════════════════════════════════════
def load_model():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["label_encoder"]


def load_database():
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)


# ════════════════════════════════════════════════════════════════════════════
# Step 2 — Face Detection (Haar Cascade, same params as Module 2)
# ════════════════════════════════════════════════════════════════════════════
_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def detect_face(gray_img):
    """
    Returns (x, y, w, h) of the first detected face, or None.
    Uses the same padding trick as Module 2 for small/tight images.
    """
    PAD    = 30
    padded = cv2.copyMakeBorder(gray_img, PAD, PAD, PAD, PAD,
                                cv2.BORDER_CONSTANT, value=0)
    enhanced = _clahe.apply(padded)
    faces = _cascade.detectMultiScale(
        enhanced, scaleFactor=1.05, minNeighbors=3,
        minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    h_orig, w_orig = gray_img.shape
    x2 = max(0, x - PAD)
    y2 = max(0, y - PAD)
    w2 = min(w, w_orig - x2)
    h2 = min(h, h_orig - y2)
    return (x2, y2, w2, h2)


# ════════════════════════════════════════════════════════════════════════════
# Step 3 — Feature Extraction (FaceNet512, same as Module 3)
# ════════════════════════════════════════════════════════════════════════════
def extract_embedding(img_path):
    rep = DeepFace.represent(
        img_path=img_path,
        model_name="Facenet512",
        enforce_detection=False
    )
    return np.array(rep[0]["embedding"])


# ════════════════════════════════════════════════════════════════════════════
# Step 4 — Classification (SVM, same as Module 5)
# ════════════════════════════════════════════════════════════════════════════
def predict_face(embedding, model, label_encoder):
    emb  = normalize(np.array(embedding).reshape(1, -1), norm="l2")
    pred = model.predict(emb)[0]
    return label_encoder.inverse_transform([pred])[0]


# ════════════════════════════════════════════════════════════════════════════
# Step 5 — Attendance (same logic as Module 6)
# ════════════════════════════════════════════════════════════════════════════
def mark_attendance(name):
    import csv
    today     = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(LOG_DIR, f"attendance_{today}.csv")

    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            if name in f.read():
                print(f"⚠  {name} already marked today")
                return False

    write_header = not os.path.isfile(file_path) or os.stat(file_path).st_size == 0
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Name", "Time", "Date"])
        writer.writerow([name, datetime.now().strftime("%H:%M:%S"), today])

    print(f"✅ Attendance marked: {name}")
    return True


# ════════════════════════════════════════════════════════════════════════════
# Full pipeline for a single BGR frame
# ════════════════════════════════════════════════════════════════════════════
def run_on_frame(bgr_frame, model, label_encoder, draw=True):
    """
    Process one frame end-to-end.
    Returns (annotated_frame, predicted_name or None).
    """
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    face = detect_face(gray)

    if face is None:
        if draw:
            cv2.putText(bgr_frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
        return bgr_frame, None

    x, y, w, h = face
    cropped    = gray[y:y+h, x:x+w]
    cropped    = cv2.resize(cropped, (92, 112))

    cv2.imwrite(TEMP_FACE, cropped)
    embedding  = extract_embedding(TEMP_FACE)
    name       = predict_face(embedding, model, label_encoder)
    mark_attendance(name)

    if draw:
        cv2.rectangle(bgr_frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
        cv2.putText(bgr_frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    return bgr_frame, name


# ════════════════════════════════════════════════════════════════════════════
# Run modes
# ════════════════════════════════════════════════════════════════════════════
def run_image_mode(image_path, model, label_encoder):
    print(f"\nProcessing image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: cannot read image at {image_path}")
        return
    annotated, name = run_on_frame(frame, model, label_encoder)
    print(f"Result: {name if name else 'No face detected'}")
    cv2.imwrite(os.path.join(ROOT, "outputs", "last_result.jpg"), annotated)
    print(f"Annotated image saved → outputs/last_result.jpg")


def run_camera_mode(model, label_encoder):
    print("\nStarting webcam... press Q to quit.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, _ = run_on_frame(frame, model, label_encoder)
        cv2.imshow("Face Recognition Attendance", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_test_mode(model, label_encoder):
    """Test pipeline using mean embeddings from the database (no camera needed)."""
    print("\nTest mode — running on database mean embeddings\n")
    database = load_database()
    correct  = 0

    for person, data in database.items():
        emb  = data["mean"]
        name = predict_face(emb, model, label_encoder)
        ok   = (name == person)
        correct += int(ok)
        status   = "✅" if ok else "❌"
        print(f"  {status}  True: {person:6s}  Predicted: {name}")

    total = len(database)
    print(f"\nAccuracy on mean embeddings: {correct}/{total} = {correct/total*100:.1f}%")


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition Attendance System")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to an image file")
    group.add_argument("--camera", action="store_true", help="Use live webcam")
    group.add_argument("--test",   action="store_true", help="Test on database embeddings")
    args = parser.parse_args()

    print("=" * 50)
    print("  Face Recognition Attendance System")
    print("=" * 50)

    model, label_encoder = load_model()

    if args.image:
        run_image_mode(args.image, model, label_encoder)
    elif args.camera:
        run_camera_mode(model, label_encoder)
    elif args.test:
        run_test_mode(model, label_encoder)

    # Clean up temp file
    if os.path.exists(TEMP_FACE):
        os.remove(TEMP_FACE)

    print("\nDone ✅")