from pathlib import Path
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


# หาโฟลเดอร์ของไฟล์นี้
BASE_DIR = Path(__file__).resolve().parent

# ชี้ไปที่โฟลเดอร์ data-bluelock
DATA_FILE = BASE_DIR.parent / "data-bluelock" / "blue_lock_knn_dataset.csv"


def load_and_preprocess_data(file_path=None):

    # ถ้าไม่ได้ส่ง path เข้ามา ใช้ DATA_FILE
    if file_path is None:
        file_path = DATA_FILE

    df = pd.read_csv(file_path)

    feature_cols = [
        "Shooting",
        "Speed",
        "Physical",
        "Vision",
        "Dribbling",
        "Passing",
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    X_tf = tf.cast(X_scaled, tf.float32)

    return df, X_tf, scaler, feature_cols