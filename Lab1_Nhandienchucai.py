"""
Bài 1: Implementation mạng nơ-ron cơ bản cho bài toán nhận diện chữ số/ký tự (MNIST Dataset)
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
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Thiết lập đường dẫn thư mục hiện tại
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("1. TẢI VÀ KHÁM PHÁ TẬP DỮ LIỆU MNIST")
print("=" * 60)

# Tải tập dữ liệu MNIST (60,000 ảnh huấn luyện và 10,000 ảnh kiểm tra kích thước 28x28)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"- Kích thước tập huấn luyện (X_train): {X_train.shape}")
print(f"- Số lượng nhãn huấn luyện (y_train): {y_train.shape}")
print(f"- Kích thước tập kiểm tra (X_test): {X_test.shape}")
print(f"- Số lượng nhãn kiểm tra (y_test): {y_test.shape}")
print(f"- Các nhãn lớp phân loại (0-9): {np.unique(y_train)}")

print("\n" + "=" * 60)
print("2. TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING)")
print("=" * 60)

# Chuẩn hóa giá trị pixel từ [0, 255] về [0.0, 1.0] để mô hình hội tụ nhanh hơn
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"- Giá trị pixel sau chuẩn hóa: min = {X_train.min()}, max = {X_train.max()}")

print("\n" + "=" * 60)
print("3. XÂY DỰNG MẠNG NƠ-RON CƠ BẢN (MULTI-LAYER PERCEPTRON - MLP)")
print("=" * 60)

# Xây dựng mô hình tuần tự (Sequential)
model = models.Sequential([
    # Lớp phẳng hóa ảnh 28x28 thành vector 784 phần tử
    layers.Flatten(input_shape=(28, 28), name="input_flatten"),
    
    # Ẩn tầng 1: 128 nơ-ron, hàm kích hoạt ReLU
    layers.Dense(128, activation='relu', name="hidden_layer_1"),
    
    # Lớp Dropout giảm thiểu hiện tượng Overfitting
    layers.Dropout(0.2, name="dropout_1"),
    
    # Ẩn tầng 2: 64 nơ-ron, hàm kích hoạt ReLU
    layers.Dense(64, activation='relu', name="hidden_layer_2"),
    
    # Lớp ngõ ra: 10 nơ-ron tương ứng 10 lớp nhãn, kích hoạt Softmax để tính phân phối xác suất
    layers.Dense(10, activation='softmax', name="output_layer")
])

# Hiển thị cấu trúc mô hình
model.summary()

print("\n" + "=" * 60)
print("4. BIÊN DỊCH VÀ HUẤN LUYỆN MÔ HÌNH")
print("=" * 60)

# Biên dịch mô hình với thuật toán tối ưu Adam và hàm mất mát Sparse Categorical Crossentropy
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Huấn luyện mô hình trong 10 epochs, trích 10% tập train làm validation
EPOCHS = 10
BATCH_SIZE = 64

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

print("\n" + "=" * 60)
print("5. ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM THỬ (TEST SET)")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"-> Test Loss: {test_loss:.4f}")
print(f"-> Test Accuracy: {test_accuracy * 100:.2f}%")

# Dự đoán trên toàn bộ tập test để tạo báo cáo chi tiết
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n--- Báo cáo phân loại chi tiết (Classification Report) ---")
print(classification_report(y_test, y_pred, digits=4))

print("\n" + "=" * 60)
print("6. THỰC HIỆN DỰ ĐOÁN MẪU VÀ TRỰC QUAN HÓA")
print("=" * 60)

# Dự đoán ngẫu nhiên một vài mẫu từ tập kiểm thử
num_samples = 5
indices = np.random.choice(len(X_test), num_samples, replace=False)

for idx in indices:
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    confidence = y_pred_probs[idx][pred_label] * 100
    status = "ĐÚNG" if true_label == pred_label else "SAI"
    print(f"Mẫu #{idx:05d}: Thực tế = {true_label} | Dự đoán = {pred_label} (Độ tin cậy: {confidence:.2f}%) -> {status}")

# Vẽ và lưu biểu đồ Loss & Accuracy qua các Epochs
plt.figure(figsize=(12, 4))

# Biểu đồ Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', lw=2)
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange', lw=2)
plt.title('Độ chính xác qua các Epochs (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Biểu đồ Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue', lw=2)
plt.plot(history.history['val_loss'], label='Val Loss', color='red', lw=2)
plt.title('Hàm mất mát qua các Epochs (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plot_save_path = os.path.join(CURRENT_DIR, "training_history.png")
plt.tight_layout()
plt.savefig(plot_save_path, dpi=300)
plt.close()
print(f"\n[+] Biểu đồ huấn luyện đã được lưu tại: {plot_save_path}")
print("Hoàn tất chương trình thành công!")
