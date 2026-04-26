import os
import pickle
import numpy as np

class DatabaseManager:
    def __init__(self, db_path="outputs/database/face_embeddings.pkl"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def load_database(self):
        if not os.path.exists(self.db_path):
            return {}
        try:
            with open(self.db_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Failed to load database: {e}")
            return {}

    def save_database(self, db):
        with open(self.db_path, "wb") as f:
            pickle.dump(db, f)

    def add_student(self, student_id, embeddings):
        db = self.load_database()
        if student_id in db:
            print(f"Student '{student_id}' already exists.")
            return
        db[student_id] = {
            "mean": np.mean(embeddings, axis=0),
            "all": list(embeddings)
        }
        self.save_database(db)

    def remove_student(self, student_id):
        db = self.load_database()
        if student_id not in db:
            print(f"Student '{student_id}' not found.")
            return
        del db[student_id]
        self.save_database(db)

    def update_student(self, student_id, new_embeddings):
        db = self.load_database()
        if student_id not in db:
            print(f"Student '{student_id}' not found.")
            return
        db[student_id]["all"].extend(new_embeddings)
        db[student_id]["mean"] = np.mean(db[student_id]["all"], axis=0)
        self.save_database(db)

    def get_student(self, student_id):
        db = self.load_database()
        return db.get(student_id)

    def list_students(self):
        db = self.load_database()
        return sorted(db.keys())

    def validate_database(self):
        db = self.load_database()
        issues = []
        for sid, entry in db.items():
            if "mean" not in entry or "all" not in entry:
                issues.append(f"'{sid}': missing keys")
                continue
            if len(entry["all"]) == 0:
                issues.append(f"'{sid}': empty embeddings")
            if entry["mean"].shape[0] != 512:
                issues.append(f"'{sid}': wrong embedding size {entry['mean'].shape}")
        return issues

    def prepare_data(self, use_mean_only=False):
        db = self.load_database()
        if not db:
            return np.array([]), np.array([])

        X, y = [], []
        for student_id, entry in db.items():
            if use_mean_only:
                X.append(entry["mean"])
                y.append(student_id)
            else:
                for emb in entry["all"]:
                    X.append(emb)
                    y.append(student_id)

        return np.array(X), np.array(y)