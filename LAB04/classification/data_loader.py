from pathlib import Path
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "data-bluelock" / "blue_lock_knn_dataset.csv"


def load_and_preprocess_data(test_size=0.3, random_state=42):
    # โหลด CSV
    df = pd.read_csv(DATA_FILE)

    # แยก Features และ Label
    feature_cols = ['Shooting', 'Speed', 'Physical', 'Vision', 'Dribbling', 'Passing']
    X = df[feature_cols].values

    # แปลง Label เป็นตัวเลข
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Position_Style'])

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # แปลงเป็น TensorFlow Tensor
    X_train_tf = tf.cast(X_train_scaled, tf.float32)
    X_test_tf = tf.cast(X_test_scaled, tf.float32)
    y_train_tf = tf.cast(y_train, tf.int32)
    y_test_tf = tf.cast(y_test, tf.int32)

    return X_train_tf, X_test_tf, y_train_tf, y_test_tf, label_encoder, feature_cols