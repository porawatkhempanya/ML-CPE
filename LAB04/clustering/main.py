import sys
import os

# เพิ่ม Path ป้องกันปัญหา ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_and_preprocess_data
from knn_tools import find_nearest_neighbors
from visualize import run_visualization_and_save

def main():
    print("🚀 กำลังเริ่มกระบวนการ Blue Lock Clustering Pipeline...")

    # 1. โหลดข้อมูล
    df, X_tf, scaler, feature_cols = load_and_preprocess_data()
    player_col = 'Player' if 'Player' in df.columns else df.columns[0]
    print(f"โหลดข้อมูลสำเร็จ! [จำนวนนักเตะ: {len(df)} คน]")

    # 2. หาเพื่อนบ้านที่มีสไตล์การเล่นใกล้เคียงกันที่สุด
    neighbors_idx = find_nearest_neighbors(X_tf, k_neighbors=2)
    df['Nearest_Player_1'] = [df[player_col].iloc[idx[0]] for idx in neighbors_idx]
    df['Nearest_Player_2'] = [df[player_col].iloc[idx[1]] for idx in neighbors_idx]

    # 3. รัน K-Means และบันทึกผลลัพธ์
    run_visualization_and_save(df, X_tf, feature_cols)

    print("✨ ดำเนินการเสร็จสิ้น! สามารถตรวจสอบผลลัพธ์ในโฟลเดอร์ outputs/ ได้เลยครับ")

if __name__ == '__main__':
    main()