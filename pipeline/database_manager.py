"""
Module 7 — Database Manager (Pipeline Script)
Author: Ahmed Alaa
Description: Centralized CRUD operations for the face embeddings database.
             Used by all other modules as an import.
"""

import os
import pickle
import numpy as np

# ─── Path resolution ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "outputs", "database", "face_embeddings.pkl")


class DatabaseManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._cache = None                      # ← in-memory cache (faster pipeline)

    # ── Internal ──────────────────────────────────────────────────────────────
    def _load(self):
        """Load from cache if available, else from disk."""
        if self._cache is not None:
            return self._cache
        if not os.path.exists(self.db_path):
            self._cache = {}
            return self._cache
        try:
            with open(self.db_path, "rb") as f:
                self._cache = pickle.load(f)
        except Exception as e:
            print(f"Failed to load database: {e}")
            self._cache = {}
        return self._cache

    def _save(self, db):
        """Persist to disk and update cache."""
        self._cache = db
        with open(self.db_path, "wb") as f:
            pickle.dump(db, f)

    # ── Public API ────────────────────────────────────────────────────────────
    def load_database(self):
        return self._load()

    def save_database(self, db):
        self._save(db)

    def add_student(self, student_id, embeddings):
        db = self._load()
        if student_id in db:
            print(f"Student '{student_id}' already exists.")
            return
        embeddings = list(embeddings)
        db[student_id] = {
            "mean": np.mean(embeddings, axis=0),
            "all":  embeddings
        }
        self._save(db)
        print(f"Added: {student_id} ✅")

    def remove_student(self, student_id):
        db = self._load()
        if student_id not in db:
            print(f"Student '{student_id}' not found.")
            return
        del db[student_id]
        self._save(db)
        print(f"Removed: {student_id} ✅")

    def update_student(self, student_id, new_embeddings):
        db = self._load()
        if student_id not in db:
            print(f"Student '{student_id}' not found.")
            return
        db[student_id]["all"].extend(list(new_embeddings))
        db[student_id]["mean"] = np.mean(db[student_id]["all"], axis=0)
        self._save(db)
        print(f"Updated: {student_id} ✅")

    def get_student(self, student_id):
        db = self._load()
        if student_id not in db:
            print(f"Student '{student_id}' not found.")
            return None
        return db[student_id]

    def list_students(self):
        return sorted(self._load().keys())

    def validate_database(self):
        db     = self._load()
        issues = []
        for sid, entry in db.items():
            if "mean" not in entry or "all" not in entry:
                issues.append(f"'{sid}': missing keys")
                continue
            if len(entry["all"]) == 0:
                issues.append(f"'{sid}': empty embeddings list")
            if entry["mean"].shape[0] != 512:
                issues.append(f"'{sid}': wrong embedding size {entry['mean'].shape}")
        return issues

    def prepare_data(self, use_mean_only=False):
        """Return (X, y) arrays ready for sklearn classifiers."""
        db = self._load()
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

    def summary(self):
        db      = self._load()
        issues  = self.validate_database()
        X, _    = self.prepare_data()
        print("=" * 40)
        print("  Database Summary")
        print("=" * 40)
        print(f"  Students  : {len(db)}")
        print(f"  Embeddings: {len(X)}")
        print(f"  Issues    : {issues if issues else 'None'}")
        print("=" * 40)


# ─── Quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    db_mgr = DatabaseManager()
    db_mgr.summary()

    # CRUD test with dummy data
    dummy = [np.random.randn(512) for _ in range(10)]
    db_mgr.add_student("s_test", dummy)
    db_mgr.update_student("s_test", [np.random.randn(512) for _ in range(5)])
    s = db_mgr.get_student("s_test")
    print(f"s_test mean shape: {s['mean'].shape}")
    db_mgr.remove_student("s_test")
    print("CRUD test passed ✅")