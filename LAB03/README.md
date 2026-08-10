# Machine Learning Lab 3
## Linear Regression and Logistic Regression

### ผู้จัดทำ
- ชื่อ: นายปรวัต เข็มปัญญา
- รหัสนักศึกษา: 116710462030-3

---

## รายละเอียดใบงาน

ใบงานนี้เป็นการทดลองเกี่ยวกับการสร้างโมเดล Machine Learning โดยใช้ภาษา Python และไลบรารี Scikit-learn เพื่อศึกษาการทำงานของ

- Simple Linear Regression
- Multiple Linear Regression
- Logistic Regression
- Model Performance Metrics
- Confusion Matrix
- Decision Boundary
- Training vs Testing Performance
- Regression vs Classification

ชุดข้อมูลที่ใช้คือ **data.csv** ซึ่งประกอบด้วยข้อมูลการออกกำลังกาย ได้แก่

- Duration
- Pulse
- Maxpulse
- Calories

---

## เครื่องมือที่ใช้

- Python 3
- Jupyter Notebook / VS Code
- pandas
- numpy
- matplotlib
- scikit-learn

---

## โครงสร้างไฟล์

```
Project/
│── data.csv
│── lab3.ipynb
│── README.md
```

---

## ขั้นตอนการทดลอง

### 1. เตรียมข้อมูล (Preparing Data)

- อ่านข้อมูลจากไฟล์ `data.csv`
- ตรวจสอบข้อมูลที่หายไป (Missing Values)
- ลบข้อมูลที่มีค่า NaN ในคอลัมน์ Calories

---

### 2. Simple Linear Regression

ใช้ตัวแปร

- Duration

เพื่อทำนาย

- Calories

พร้อมแสดงกราฟ Regression Line

---

### 3. Multiple Linear Regression

ใช้ตัวแปร

- Duration
- Pulse
- Maxpulse

เพื่อทำนาย

- Calories

---

### 4. Logistic Regression

สร้างคอลัมน์ Class จาก Calories

- Calories ≥ 300 = High Calories (1)
- Calories < 300 = Low Calories (0)

แล้วใช้

- Duration
- Pulse
- Maxpulse

ในการจำแนกประเภท

---

### 5. Decision Boundary

แสดงเส้นแบ่งการจำแนกของ Logistic Regression โดยใช้ข้อมูล 2 Features

---

### 6. Confusion Matrix

ประเมินผลการจำแนกข้อมูลด้วย

- Accuracy
- Confusion Matrix
- Classification Report

---

### 7. Model Performance Metrics

Regression

- R² Score
- MAE
- MSE
- RMSE

Classification

- Accuracy
- Precision
- Recall
- F1-score

---

### 8. Training vs Testing Performance

เปรียบเทียบประสิทธิภาพของโมเดลระหว่าง

- Training Set
- Testing Set

---

### 9. Regression vs Classification

เปรียบเทียบความแตกต่างระหว่าง

Regression

- ใช้ทำนายค่าตัวเลข เช่น Calories

Classification

- ใช้จำแนกประเภท เช่น High Calories และ Low Calories

---

## ผลการทดลอง

จากการทดลองพบว่า

- Multiple Linear Regression ให้ผลการทำนายดีกว่า Simple Linear Regression เนื่องจากใช้หลายตัวแปรในการพยากรณ์
- Logistic Regression สามารถจำแนกข้อมูลออกเป็น High Calories และ Low Calories ได้
- การประเมินผลด้วย R² Score, MAE, MSE, RMSE, Accuracy และ Confusion Matrix ช่วยวิเคราะห์ประสิทธิภาพของโมเดลได้อย่างเหมาะสม
- การจัดการ Missing Values ก่อนสร้างโมเดลเป็นขั้นตอนสำคัญที่ช่วยให้โมเดลทำงานได้อย่างถูกต้อง

---

## สรุป

ใบงานนี้ช่วยให้เข้าใจหลักการของ Machine Learning ทั้งในรูปแบบ Regression และ Classification รวมถึงการเตรียมข้อมูล การสร้างโมเดล การประเมินผล และการเปรียบเทียบประสิทธิภาพของโมเดลด้วย Scikit-learn