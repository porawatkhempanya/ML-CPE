import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def generate_prediction_sample(output_dir="outputs", num_images=4):
    X_test = np.load(os.path.join(output_dir, "X_test.npy"))
    y_test = np.load(os.path.join(output_dir, "y_test.npy"))
    
    with open(os.path.join(output_dir, "classes.json"), "r") as f:
        class_names = json.load(f)
        
    model = tf.keras.models.load_model(os.path.join(output_dir, "cnn_model.keras"))
    
    indices = np.random.choice(len(X_test), num_images, replace=False)
    
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices):
        img = X_test[idx]
        true_label = class_names[y_test[idx]]
        
        pred_prob = model.predict(np.expand_dims(img, axis=0))
        pred_idx = np.argmax(pred_prob)
        pred_label = class_names[pred_idx]
        
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
        
    plt.tight_layout()
    # บันทึกภาพตัวอย่างการทำนายลง outputs -> ได้ prediction_sample.png
    plt.savefig(os.path.join(output_dir, "prediction_sample.png"))
    plt.close()