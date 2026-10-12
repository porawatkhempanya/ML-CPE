import os
import cv2
import numpy as np
from tqdm import tqdm

def load_images_from_folder(data_dir, img_size=(128, 128)):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        print(f"Loading class: {class_name}")
        for img_name in tqdm(os.listdir(class_path)):
            img_path = os.path.join(class_path, img_name)
            try:
                # โหลดรูปและแปลง BGR เป็น RGB
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                
                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                # ข้ามไฟล์ที่อ่านไม่ได้หรือเสียหาย
                continue
                
    return np.array(images), np.array(labels), class_names