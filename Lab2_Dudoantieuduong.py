"""
Bài 2: Xây dựng và huấn luyện mạng nơ-ron cơ bản để dự đoán khả năng mắc bệnh tiểu đường
Tập dữ liệu: Pima Indians Diabetes Database (Chỉ số sức khỏe)
Môn học: Deep Learning
"""

import sys
import io
import os

# Đảm bảo in tiếng Việt Unicode trên Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. TẢI VÀ KHÁM PHÁ DỮ LIỆU
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(CURRENT_DIR, "diabetes.csv")

# Nếu file chưa tồn tại cục bộ, tải từ nguồn online
if not os.path.exists(csv_path):
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    df.to_csv(csv_path, index=False)
else:
    df = pd.read_csv(csv_path)

print("=" * 65)
print("1. KHÁM PHÁ TẬP DỮ LIỆU CHỈ SỐ SỨC KHỎE TIỂU ĐƯỜNG")
print("=" * 65)
print(f"- Kích thước tập dữ liệu: {df.shape[0]} hàng, {df.shape[1]} cột")
print("- 5 dòng dữ liệu đầu tiên:")
print(df.head())
print("\n- Phân phối nhãn Outcome (0: Không mắc bệnh, 1: Mắc bệnh):")
print(df['Outcome'].value_counts())

# 2. TIỀN XỬ LÝ DỮ LIỆU
print("\n" + "=" * 65)
print("2. TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING)")
print("=" * 65)

# Trong dữ liệu y tế, các giá trị 0 ở Glucose, BloodPressure, SkinThickness, Insulin, BMI là giá trị khuyết
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    median_val = df[df[col] > 0][col].median()
    df[col] = df[col].replace(0, median_val)

X = df.drop(columns=['Outcome']).values
y = df['Outcome'].values

# Phân chia tập dữ liệu: 80% Train, 20% Test (sử dụng stratify để giữ tỷ lệ nhãn)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Chuẩn hóa dữ liệu bằng StandardScaler (Z-score Normalization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"- Kích thước tập huấn luyện (Train): {X_train.shape}")
print(f"- Kích thước tập kiểm tra (Test): {X_test.shape}")
print(f"- Số đặc trưng đầu vào (Features): {X_train.shape[1]}")

# 3. XÂY DỰNG KIẾN TRÚC MẠNG NƠ-RON CƠ BẢN (MLP)
print("\n" + "=" * 65)
print("3. XÂY DỰNG MẠNG NƠ-RON TRUYỀN THẲNG (MLP) CHO PHÂN LOẠI NHỊ PHÂN")
print("=" * 65)

model = models.Sequential([
    # Lớp ẩn 1: 32 nơ-ron, đầu vào 8 đặc trưng, hàm kích hoạt ReLU
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],), name="hidden_layer_1"),
    layers.Dropout(0.2, name="dropout_1"),
    
    # Lớp ẩn 2: 16 nơ-ron, hàm kích hoạt ReLU
    layers.Dense(16, activation='relu', name="hidden_layer_2"),
    layers.Dropout(0.1, name="dropout_2"),
    
    # Lớp ẩn 3: 8 nơ-ron, hàm kích hoạt ReLU
    layers.Dense(8, activation='relu', name="hidden_layer_3"),
    
    # Lớp ngõ ra: 1 nơ-ron, hàm kích hoạt Sigmoid xuất xác suất [0, 1]
    layers.Dense(1, activation='sigmoid', name="output_layer")
])

model.summary()

# 4. BIÊN DỊCH VÀ HUẤN LUYỆN
print("\n" + "=" * 65)
print("4. BIÊN DỊCH VÀ HUẤN LUYỆN MÔ HÌNH")
print("=" * 65)

model.compile(
    optimizer=Adam(learning_rate=0.002),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Áp dụng EarlyStopping để dừng sớm nếu mô hình bắt đầu học vẹt trên tập val
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

EPOCHS = 80
BATCH_SIZE = 16

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 5. ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST
print("\n" + "=" * 65)
print("5. ĐÁNH GIÁ HIỆU SUẤT TRÊN TẬP KIỂM THỬ (TEST SET)")
print("=" * 65)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
y_pred_probs = model.predict(X_test, verbose=0).ravel()
y_pred = (y_pred_probs >= 0.5).astype(int)

auc_score = roc_auc_score(y_test, y_pred_probs)

print(f"-> Test Loss: {test_loss:.4f}")
print(f"-> Test Accuracy: {test_acc * 100:.2f}%")
print(f"-> ROC-AUC Score: {auc_score:.4f}")

print("\n--- Ma trận nhầm lẫn (Confusion Matrix) ---")
cm = confusion_matrix(y_test, y_pred)
print(f"True Negative (Không bệnh đúng): {cm[0, 0]}")
print(f"False Positive (Dương tính giả): {cm[0, 1]}")
print(f"False Negative (Bỏ sót bệnh):   {cm[1, 0]}")
print(f"True Positive (Phát hiện bệnh): {cm[1, 1]}")

print("\n--- Báo cáo phân loại chi tiết (Classification Report) ---")
print(classification_report(y_test, y_pred, target_names=["Không bệnh (0)", "Mắc bệnh (1)"], digits=4))

# 6. DỰ ĐOÁN MẪU THỰC TẾ
print("\n" + "=" * 65)
print("6. DỰ ĐOÁN NGUY CƠ TIỂU ĐƯỜNG TRÊN CÁC MẪU BỆNH NHÂN")
print("=" * 65)

feature_names = df.columns[:-1].tolist()
sample_indices = [0, 5, 10, 15, 20]

for idx in sample_indices:
    prob = y_pred_probs[idx] * 100
    actual = "Mắc bệnh" if y_test[idx] == 1 else "Không bệnh"
    predict = "Mắc bệnh" if y_pred[idx] == 1 else "Không bệnh"
    status = "ĐÚNG" if y_test[idx] == y_pred[idx] else "SAI"
    print(f"Bệnh nhân #{idx:03d}: Xác suất = {prob:5.2f}% | Dự đoán: {predict:<10} | Thực tế: {actual:<10} -> {status}")

# Vẽ và lưu biểu đồ Loss & Accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', lw=2)
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange', lw=2)
plt.title('Độ chính xác mô hình (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue', lw=2)
plt.plot(history.history['val_loss'], label='Val Loss', color='red', lw=2)
plt.title('Hàm mất mát (Binary Crossentropy Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plot_save_path = os.path.join(CURRENT_DIR, "diabetes_training_history.png")
plt.tight_layout()
plt.savefig(plot_save_path, dpi=300)
plt.close()

print(f"\n[+] Biểu đồ huấn luyện đã được lưu tại: {plot_save_path}")
print("Hoàn tất chương trình Bài 2 thành công!")
