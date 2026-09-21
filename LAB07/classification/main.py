import os
import json
import numpy as np
import tensorflow as tf
from split_data import prepare_data
from cnn_model import create_cnn_model
from evaluate import evaluate_model, plot_history
from test_cnn import generate_prediction_sample  

def main():
    data_dir = r"C:\ML-CPE\LAB07\archive\Training"
    output_dir = "outputs"
    
    # 1. เตรียมและแบ่งข้อมูล (หากรันแล้วข้ามขั้นตอนนี้ได้)
    if not os.path.exists(os.path.join(output_dir, "X_train.npy")):
        prepare_data(data_dir, output_dir)
        
    # 2. โหลดข้อมูลเข้าสู่แรม
    X_train = np.load(os.path.join(output_dir, "X_train.npy"))
    y_train = np.load(os.path.join(output_dir, "y_train.npy"))
    X_val = np.load(os.path.join(output_dir, "X_val.npy"))
    y_val = np.load(os.path.join(output_dir, "y_val.npy"))
    
    with open(os.path.join(output_dir, "classes.json"), "r") as f:
        class_names = json.load(f)
        
    # 3. สร้างและเทรนโมเดล
    model = create_cnn_model(input_shape=X_train.shape[1:], num_classes=len(class_names))
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32
    )
    
    # 4. บันทึกโมเดลและกราฟ
    model.save(os.path.join(output_dir, "cnn_model.keras"))
    plot_history({'My CNN Model': history}, output_dir)
    
    # 5. ประเมินผล
    evaluate_model(output_dir)
    
    # 6. สร้างรูปตัวอย่างการทำนาย
    print("Generating prediction sample...")
    generate_prediction_sample(output_dir)
    
    print("Training and Evaluation Complete!")

if __name__ == "__main__":
    main()