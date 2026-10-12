from data_loader import load_and_preprocess_data
from evaluate import evaluate_and_save_outputs
file_path = "../data-bluelock/blue_lock_knn_dataset.csv"

def main():
    print("🚀 กำลังเริ่มกระบวนการ ML Pipeline (TensorFlow KNN)...")
    
    # 1. โหลดและเตรียมข้อมูล
    X_train, X_test, y_train, y_test, label_encoder, _ = load_and_preprocess_data()
    print(f"โหลดข้อมูลสำเร็จ! [Train samples: {len(X_train)}, Test samples: {len(X_test)}]")
    
    # 2. ประเมินโมเดลและสร้าง Outputs
    evaluate_and_save_outputs(X_train, X_test, y_train, y_test, label_encoder)
    
    print("✨ ทำงานเสร็จสิ้น! สามารถตรวจสอบไฟล์ในโฟลเดอร์ outputs/ ได้เลยครับ")

if __name__ == '__main__':
    main()