"""
===========================================
GIAI ĐOẠN 1: NỀN TẢNG CƠ BẢN - SOLUTIONS
Đáp án chi tiết cho các bài tập
===========================================

🎯 HÃY LÀM BÀI TẬP TRƯỚC KHI XEM SOLUTIONS!

File này chứa đáp án đầy đủ với giải thích để bạn học hỏi.
"""

import matplotlib.pyplot as plt
import numpy as np

print("🔍 SOLUTIONS CHO CÁC BÀI TẬP MATPLOTLIB")
print("="*50)

# ===========================================
# SOLUTION 1: VẼ ĐƯỜNG CƠ BẢN
# ===========================================
print("\n✅ SOLUTION 1: Vẽ đường y = 2x + 3")
print("-" * 40)

# Tạo dữ liệu
x = np.linspace(0, 10, 50)  # 50 điểm từ 0 đến 10
y = 2 * x + 3               # Công thức y = 2x + 3

# Vẽ biểu đồ
plt.figure(figsize=(8, 6))
plt.plot(x, y, color='red')  # hoặc c='r'
plt.title("Đường thẳng y = 2x + 3")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

print("💡 Giải thích:")
print("- np.linspace(0, 10, 50): tạo 50 điểm đều nhau từ 0 đến 10")
print("- color='red' hoặc c='r': đặt màu đỏ cho đường")
print("- grid(True): hiển thị lưới để dễ đọc")

# ===========================================
# SOLUTION 2: SCATTER PLOT
# ===========================================
print("\n✅ SOLUTION 2: Scatter plot điểm thi")
print("-" * 40)

# Dữ liệu
hours_studied = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
scores = [65, 70, 75, 80, 82, 85, 88, 90, 92, 95]

# Vẽ scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(hours_studied, scores, marker='o', c='blue', s=80)
plt.title("Mối quan hệ giữa giờ học và điểm thi")
plt.xlabel("Giờ học")
plt.ylabel("Điểm thi")
plt.grid(alpha=0.3)
plt.show()

print("💡 Giải thích:")
print("- marker='o': điểm tròn")
print("- s=80: kích thước điểm")
print("- alpha=0.3: grid mờ để không che data")

# ===========================================
# SOLUTION 3: BIỂU ĐỒ CỘT
# ===========================================
print("\n✅ SOLUTION 3: Biểu đồ cột doanh thu")
print("-" * 40)

# Dữ liệu
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
revenue = [120, 150, 180, 200, 170]

# Vẽ biểu đồ cột
plt.figure(figsize=(8, 6))
bars = plt.bar(months, revenue, color='green')
plt.title("Doanh thu theo tháng")
plt.xlabel("Tháng")
plt.ylabel("Doanh thu (triệu VND)")

# Thêm giá trị lên đầu mỗi cột
for i, v in enumerate(revenue):
    plt.text(i, v + 5, str(v), ha='center')

plt.xticks(rotation=45)  # Xoay labels 45 độ
plt.tight_layout()       # Tự động điều chỉnh layout
plt.show()

print("💡 Giải thích:")
print("- enumerate(): lấy cả index và value")
print("- plt.text(x, y, text, ha='center'): thêm text tại vị trí (x,y)")
print("- ha='center': căn giữa text theo horizontal")
print("- tight_layout(): tránh bị cắt labels")

# ===========================================
# SOLUTION 4: HISTOGRAM
# ===========================================
print("\n✅ SOLUTION 4: Histogram phân phối điểm")
print("-" * 40)

# Tạo dữ liệu
np.random.seed(42)
test_scores = np.random.normal(75, 10, 200)

# Vẽ histogram
plt.figure(figsize=(8, 6))
plt.hist(test_scores, bins=20, color='lightblue', alpha=0.7, edgecolor='black')
plt.title("Phân phối điểm thi")
plt.xlabel("Điểm")
plt.ylabel("Số học sinh")

# Thêm đường mean
mean_score = np.mean(test_scores)
plt.axvline(mean_score, color='red', linestyle='--', 
            label=f'Mean: {mean_score:.1f}')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()

print("💡 Giải thích:")
print("- bins=20: chia thành 20 khoảng")
print("- edgecolor='black': viền đen cho mỗi bin")
print("- axvline(): vẽ đường thẳng đứng")
print("- f'Mean: {mean_score:.1f}': format số thập phân 1 chữ số")

# ===========================================
# SOLUTION 5: LINEAR REGRESSION CHALLENGE
# ===========================================
print("\n✅ SOLUTION 5: 🔥 Linear Regression Challenge")
print("-" * 40)

# Dữ liệu
x_train = [1, 2, 3, 4, 5]
y_train = [300, 500, 700, 900, 1100]

# Tính w và b tối ưu (dùng least squares)
x_array = np.array(x_train)
y_array = np.array(y_train)

# Công thức least squares
n = len(x_train)
w = (n * np.sum(x_array * y_array) - np.sum(x_array) * np.sum(y_array)) / \
    (n * np.sum(x_array**2) - np.sum(x_array)**2)
b = (np.sum(y_array) - w * np.sum(x_array)) / n

print(f"📊 Tính toán tối ưu: w = {w:.1f}, b = {b:.1f}")

# Hoặc có thể thử nghiệm: w=200, b=100 cũng khá tốt!
w_simple, b_simple = 200, 100

# Compute model output
def compute_model_output(x, w, b):
    return w * np.array(x) + b

tmp_f_wb = compute_model_output(x_train, w, b)

# Vẽ biểu đồ CHÍNH XÁC như ví dụ của bạn
plt.figure(figsize=(8, 6))
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

print("💡 Giải thích:")
print(f"- Dùng least squares: w={w:.1f}, b={b:.1f}")
print("- Hoặc estimate đơn giản: w=200, b=100")
print("- c='b' = color='blue'")
print("- marker='x': dấu X cho actual data")
print("- Đây chính xác là code mà bạn muốn học!")

# ===========================================
# BONUS: KẾT HỢP TẤT CẢ TRONG 1 FIGURE
# ===========================================
print("\n🎁 BONUS: Kết hợp nhiều subplot trong 1 figure")
print("-" * 40)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Line plot
x = np.linspace(0, 10, 50)
y = 2 * x + 3
ax1.plot(x, y, 'r-')
ax1.set_title('Line Plot')
ax1.grid(True)

# Plot 2: Scatter plot  
ax2.scatter(hours_studied, scores, c='blue', s=80)
ax2.set_title('Scatter Plot')
ax2.grid(alpha=0.3)

# Plot 3: Bar plot
ax3.bar(months, revenue, color='green')
ax3.set_title('Bar Plot')
ax3.set_xticklabels(months, rotation=45)

# Plot 4: Histogram
ax4.hist(test_scores, bins=20, color='lightblue', alpha=0.7)
ax4.axvline(np.mean(test_scores), color='red', linestyle='--')
ax4.set_title('Histogram')

plt.tight_layout()
plt.show()

print("💡 Subplots giúp so sánh nhiều biểu đồ cùng lúc!")

# ===========================================
# TỔNG KẾT VÀ TIPS
# ===========================================
print("\n" + "="*50)
print("🎯 TỔNG KẾT GIAI ĐOẠN 1")
print("="*50)
print("""
✅ ĐÃ HỌC XONG:
1. Import matplotlib.pyplot as plt
2. 4 loại biểu đồ cơ bản: plot, scatter, bar, hist
3. Labels, titles, legends
4. Colors, markers, styling cơ bản
5. Grid và layout

🔥 CÚ PHÁP QUAN TRỌNG NHẤT:
- plt.plot(x, y, c='b', label='Our Prediction')
- plt.scatter(x, y, marker='x', c='r', label='Actual Values')  
- plt.title(), plt.xlabel(), plt.ylabel()
- plt.legend(), plt.show()

💡 TIPS KHI CODE:
1. Luôn import: import matplotlib.pyplot as plt
2. Đặt figsize: plt.figure(figsize=(8,6))
3. Thêm grid: plt.grid(True) hoặc plt.grid(alpha=0.3)
4. Đừng quên: plt.show() ở cuối!
5. Dùng tight_layout() nếu bị cắt labels

🚀 TIẾP THEO: Giai đoạn 2 - Tùy chỉnh biểu đồ nâng cao!
""")

print("🎉 Chúc mừng! Bạn đã hoàn thành Giai đoạn 1!") 