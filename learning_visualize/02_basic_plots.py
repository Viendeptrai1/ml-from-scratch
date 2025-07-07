"""
===========================================
GIAI ĐOẠN 1: NỀN TẢNG CƠ BẢN - BÀI 2
Các loại biểu đồ cơ bản
===========================================

Bài này sẽ dạy bạn 4 loại biểu đồ cơ bản:
1. plt.plot() - Vẽ đường thẳng
2. plt.scatter() - Vẽ scatter plot  
3. plt.bar() - Biểu đồ cột
4. plt.hist() - Histogram
"""

import matplotlib.pyplot as plt
import numpy as np

print("📊 HỌC CÁC LOẠI BIỂU ĐỒ CƠ BẢN")
print("="*50)

# ===========================================
# 1. PLT.PLOT() - VẼ ĐƯỜNG THẲNG
# ===========================================
print("\n🔵 1. PLT.PLOT() - Vẽ đường thẳng")

# Dữ liệu cho linear regression (giống ví dụ của bạn)
x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = np.array([300, 500, 700, 900, 1100])

# Giả sử w=200, b=100 (model parameters)
w, b = 200, 100
y_predicted = w * x_train + b

plt.figure(figsize=(8, 6))
plt.plot(x_train, y_predicted, c='b', label='Our Prediction')
plt.title("Linear Regression - Giống ví dụ của bạn!")
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (in 1000s of dollars)')
plt.legend()
plt.grid(True)
plt.show()

print("✅ Đây chính xác là cú pháp trong ví dụ của bạn!")

# ===========================================
# 2. PLT.SCATTER() - VẼ SCATTER PLOT
# ===========================================
print("\n🔴 2. PLT.SCATTER() - Vẽ scatter plot")

plt.figure(figsize=(8, 6))
# Vẽ cả predicted line và actual data points
plt.plot(x_train, y_predicted, c='b', label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values', s=100)

plt.title("Housing Prices - Hoàn chỉnh như ví dụ của bạn")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.grid(True)
plt.show()

print("✅ Đây là full version của ví dụ bạn muốn học!")

# Các tùy chọn cho scatter plot
print("\n📝 Tùy chọn cho plt.scatter():")
print("- marker: 'o', 'x', 's', '^', 'v', '*', '+', 'D'")
print("- s: kích thước điểm")
print("- c: màu sắc")
print("- alpha: độ trong suốt (0-1)")

# ===========================================
# 3. PLT.BAR() - BIỂU ĐỒ CỘT
# ===========================================
print("\n🟡 3. PLT.BAR() - Biểu đồ cột")

# Dữ liệu để so sánh model performance
models = ['Linear Reg', 'Polynomial', 'Neural Net', 'Random Forest']
accuracy = [85, 92, 96, 94]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracy, color=['blue', 'green', 'red', 'orange'])
plt.title('So sánh độ chính xác các model ML')
plt.ylabel('Accuracy (%)')
plt.xlabel('Model')

# Thêm giá trị lên đầu mỗi cột
for bar, acc in zip(bars, accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{acc}%', ha='center')

plt.grid(axis='y', alpha=0.3)
plt.show()

print("✅ Dùng để so sánh performance các model!")

# ===========================================
# 4. PLT.HIST() - HISTOGRAM  
# ===========================================
print("\n🟢 4. PLT.HIST() - Histogram")

# Tạo dữ liệu giả cho phân phối lỗi của model
np.random.seed(42)
prediction_errors = np.random.normal(0, 10, 1000)  # Mean=0, std=10

plt.figure(figsize=(8, 6))
plt.hist(prediction_errors, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
plt.title('Phân phối lỗi dự đoán của model')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.3)

# Thêm đường thẳng đứng ở mean
plt.axvline(np.mean(prediction_errors), color='red', linestyle='--', 
            label=f'Mean: {np.mean(prediction_errors):.2f}')
plt.legend()
plt.show()

print("✅ Dùng để phân tích distribution của data!")

# ===========================================
# TỔNG KẾT
# ===========================================
print("\n" + "="*50)
print("🎯 TỔNG KẾT BÀI 2")
print("="*50)
print("""
✅ Đã học 4 loại biểu đồ cơ bản:

1. plt.plot() ➜ Vẽ đường (predictions, trends)
2. plt.scatter() ➜ Vẽ điểm (actual data points)  
3. plt.bar() ➜ So sánh categories
4. plt.hist() ➜ Phân phối dữ liệu

🔥 CÚ PHÁP QUAN TRỌNG từ ví dụ của bạn:
- plt.plot(x, y, c='b', label='Our Prediction')
- plt.scatter(x, y, marker='x', c='r', label='Actual Values')
- plt.title(), plt.xlabel(), plt.ylabel()
- plt.legend(), plt.show()
""")

print("🚀 Tiếp theo: Làm bài tập thực hành!") 