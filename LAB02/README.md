# Data Preprocessing Laboratory

## รายวิชา
Machine Learning

---

## รายละเอียดใบงาน

ใบงานนี้เป็นการศึกษาการเตรียมข้อมูล (Data Preprocessing) โดยใช้ภาษา Python และไลบรารีสำหรับวิเคราะห์ข้อมูล เพื่อเตรียมข้อมูลก่อนนำไปใช้ในงาน Machine Learning ประกอบด้วยการสำรวจข้อมูล การแสดงผลข้อมูล การทำความสะอาดข้อมูล และการแปลงข้อมูลให้อยู่ในรูปแบบที่พร้อมสำหรับการสร้างแบบจำลอง

---

## วัตถุประสงค์

- ศึกษาการใช้งานภาษา Python สำหรับการวิเคราะห์ข้อมูล
- เรียนรู้การใช้ไลบรารี Pandas
- ตรวจสอบลักษณะของชุดข้อมูล
- วิเคราะห์ข้อมูลเบื้องต้นด้วยสถิติ
- แสดงผลข้อมูลด้วยกราฟ
- ทำความสะอาดข้อมูล (Data Cleaning)
- แปลงข้อมูล (Feature Engineering)
- เตรียมข้อมูลสำหรับ Machine Learning

---

## เครื่องมือที่ใช้

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## การติดตั้ง

ติดตั้งไลบรารีที่จำเป็น

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel
```

หรือหากใช้ uv

```bash
uv pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel
```

---

## โครงสร้างโปรเจกต์

```
ML-CPE/
│
├── data.csv
├── cleaned_data.csv
├── LAB2_code.ipynb
├── report.pdf
├── README.md
└── images/
```

---

# ขั้นตอนการทดลอง

## Part 1 : Dataset Exploration

ดำเนินการตรวจสอบข้อมูลเบื้องต้น

- Load Dataset
- Display Shape
- Display Data Types
- Display Summary Statistics
- Display Missing Values
- Display Duplicate Records
- Display Class Distribution (ถ้ามี)

---

## Part 2 : Data Visualization

สร้างกราฟเพื่อวิเคราะห์ข้อมูล

- Histogram
- Correlation Heatmap

---

## Part 3 : Data Cleaning

ทำความสะอาดข้อมูล

- Missing Value Handling
- Duplicate Removal
- Incorrect Data Correction
- Data Type Conversion

---

## Part 4 : Feature Engineering

แปลงข้อมูลเพื่อใช้กับ Machine Learning

- Mean
- Median
- Label Encoding
- One-Hot Encoding

---

## ตัวอย่างโค้ด

```python
import pandas as pd

df = pd.read_csv("data.csv")

print(df.head())
print(df.info())
print(df.shape)
```

---

## ผลการทดลอง

จากการทดลองสามารถ

- โหลดข้อมูลจากไฟล์ CSV ได้สำเร็จ
- ตรวจสอบจำนวนแถวและคอลัมน์ของข้อมูล
- ตรวจสอบชนิดข้อมูลของแต่ละคอลัมน์
- วิเคราะห์ค่าสถิติเบื้องต้นของข้อมูล
- ตรวจสอบ Missing Values และ Duplicate Records
- ทำ Data Cleaning ได้สำเร็จ
- สร้างกราฟ Histogram และ Correlation Heatmap
- แปลงข้อมูลด้วย Label Encoding และ One-Hot Encoding
- เตรียมข้อมูลพร้อมสำหรับการนำไปสร้างโมเดล Machine Learning

---

## ไลบรารีที่ใช้

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
```

---

## วิธีรันโปรแกรม

เปิด Jupyter Notebook

```bash
jupyter notebook
```

หรือเปิดด้วย VS Code แล้วรันไฟล์

```
LAB2_code.ipynb
```

---

## ผู้จัดทำ

ชื่อ : นายปรวัต เข็มปัญญา

รหัสนักศึกษา : 116710462030-3

สาขาวิชา : วิศวกรรมคอมพิวเตอร์

รายวิชา : Machine Learning

---

## สรุป

ใบงานนี้ช่วยให้เข้าใจขั้นตอนการเตรียมข้อมูลก่อนนำไปใช้ในงาน Machine Learning ตั้งแต่การสำรวจข้อมูล การวิเคราะห์ข้อมูล การทำความสะอาดข้อมูล และการแปลงข้อมูลให้อยู่ในรูปแบบที่เหมาะสมสำหรับการสร้างแบบจำลอง ซึ่งเป็นพื้นฐานสำคัญของการวิเคราะห์ข้อมูลและการพัฒนาโมเดล Machine Learning