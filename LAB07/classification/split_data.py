import os
import numpy as np
from sklearn.model_selection import train_test_split
import json
from data_loader import load_images_from_folder

def prepare_data(data_dir, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # โหลดรูปภาพทั้งหมด
    X, y, class_names = load_images_from_folder(data_dir)
    
    # Normalize ข้อมูลภาพให้อยู่ในช่วง [0, 1]
    X = X.astype("float32") / 255.0
    
    # แบ่ง Train (70%), ส่วนที่เหลือไปทำ Val/Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # แบ่ง Val (15%) และ Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # บันทึกไฟล์ numpy และชื่อคลาส
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    with open(os.path.join(output_dir, "classes.json"), "w") as f:
        json.dump(class_names, f)
        
    print("Data preparation and splitting completed!")

if __name__ == "__main__":
    prepare_data(r"..\archive\Training")