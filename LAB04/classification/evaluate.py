import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from knn_tf import KNNTensorFlow
file_path = "../data-bluelock/blue_lock_knn_dataset.csv"

def evaluate_and_save_outputs(X_train, X_test, y_train, y_test, label_encoder, output_dir='outputs'):
    # สร้างโฟลเดอร์ outputs ถ้ายังไม่มี
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. วาดรูป K-Accuracy Curve (01_k_curve.png)
    k_range = range(1, max(2, len(X_train)))
    accuracies = []
    
    for k in k_range:
        knn = KNNTensorFlow(k=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        acc = accuracy_score(y_test.numpy(), preds.numpy())
        accuracies.append(acc)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), accuracies, marker='o', color='b')
    plt.title('KNN K-Value vs Accuracy')
    plt.xlabel('Value of K')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_k_curve.png'))
    plt.close()

    # เลือก K ที่ดีที่สุดมาวัด Confusion Matrix
    best_k = list(k_range)[accuracies.index(max(accuracies))]
    best_knn = KNNTensorFlow(k=best_k)
    best_knn.fit(X_train, y_train)
    final_preds = best_knn.predict(X_test)

    # 2. วาดรูป Confusion Matrix (02_confusion_matrix.png)
    cm = confusion_matrix(y_test.numpy(), final_preds.numpy())
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix (K={best_k})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_confusion_matrix.png'))
    plt.close()

    # 3. บันทึกผลทำนายลง CSV (predictions.csv)
    results_df = pd.DataFrame({
        'True_Class': label_encoder.inverse_transform(y_test.numpy()),
        'Predicted_Class': label_encoder.inverse_transform(final_preds.numpy())
    })
    results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    print(f"บันทึกไฟล์ผลลัพธ์ลงในโฟลเดอร์ '{output_dir}/' สำเร็จเรียบร้อย!")