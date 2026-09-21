import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

def evaluate_model(output_dir="outputs"):
    """
    ฟังก์ชันสำหรับประเมินผลโมเดลและสร้าง Confusion Matrix
    """
    X_test = np.load(os.path.join(output_dir, "X_test.npy"))
    y_test = np.load(os.path.join(output_dir, "y_test.npy"))
    
    with open(os.path.join(output_dir, "classes.json"), "r") as f:
        class_names = json.load(f)
        
    model = tf.keras.models.load_model(os.path.join(output_dir, "cnn_model.keras"))
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # สร้างและบันทึก Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

def plot_history(histories_dict, output_dir="outputs"):
    """
    ฟังก์ชันพล็อตประวัติการเทรนแบบละเอียด (มี Marker วงกลม, Grid และเปอร์เซ็นต์ที่ Legend)
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#b15928', '#e377c2', '#2ca02c', '#1f77b4', '#7f7f7f', '#bcbd22']
    
    for idx, (model_name, history) in enumerate(histories_dict.items()):
        color = colors[idx % len(colors)]
        
        acc = history.history.get('accuracy', [])
        epochs = range(len(acc))
        
        if len(acc) == 0:
            continue
            
        final_score = acc[-1] * 100  # แปลงเป็นเปอร์เซ็นต์
        
        ax.plot(
            epochs, 
            [a * 100 for a in acc], 
            label=f'{model_name}: {final_score:.2f}%', 
            color=color, 
            marker='o',         # มีจุดวงกลมทุกจุด
            markersize=4,       # ขนาดจุด
            linewidth=1.5,      # ความหนาเส้น
            alpha=0.85
        )

    ax.set_title('Training Performance', fontsize=14, pad=12, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, "training_performance_detailed.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved detailed plot to {save_path}")