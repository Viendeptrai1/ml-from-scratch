"""
===========================================
GIAI ĐOẠN 1: NỀN TẢNG CƠ BẢN - BÀI 1
Setup và Import Matplotlib
===========================================

Bài này sẽ dạy bạn:
1. Cách import matplotlib.pyplot
2. Hiểu về figure và axes
3. Cách hiển thị biểu đồ với plt.show()
"""

# 1. IMPORT CƠ BẢN
import matplotlib.pyplot as plt
import numpy as np

print("📚 Chào mừng đến với khóa học Matplotlib!")
print("✅ Import matplotlib.pyplot thành công!")

# 2. TẠO BIỂU ĐỒ ĐƠN GIẢN NHẤT
# Tạo dữ liệu đơn giản
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Vẽ biểu đồ
plt.plot(x, y)
plt.title("Biểu đồ đầu tiên của tôi!")
plt.show()

print("🎉 Bạn vừa tạo biểu đồ đầu tiên!")

# 3. HIỂU VỀ FIGURE VÀ AXES
print("\n" + "="*50)
print("📖 KIẾN THỨC CƠ BẢN")
print("="*50)

print("""
🔍 CÁC KHÁI NIỆM QUAN TRỌNG:

1. FIGURE: 
   - Là cả cửa sổ chứa biểu đồ
   - Có thể chứa nhiều biểu đồ con (subplots)
   
2. AXES: 
   - Là khu vực vẽ biểu đồ thực tế
   - Chứa data, labels, title...
   
3. plt.show():
   - Hiển thị biểu đồ ra màn hình
   - Luôn gọi cuối cùng!
""")

# 4. VÍ DỤ VỚI NHIỀU THÔNG TIN HỚN
print("\n🚀 Ví dụ có comment chi tiết:")

# Tạo dữ liệu
x_data = np.linspace(0, 10, 50)  # 50 điểm từ 0 đến 10
y_data = np.sin(x_data)          # Hàm sin

# Vẽ biểu đồ
plt.figure(figsize=(8, 6))       # Tạo figure với kích thước 8x6 inch
plt.plot(x_data, y_data)         # Vẽ đường
plt.title("Hàm Sin(x)")          # Tiêu đề
plt.xlabel("x")                  # Nhãn trục x
plt.ylabel("sin(x)")             # Nhãn trục y
plt.grid(True)                   # Hiển thị lưới
plt.show()                       # Hiển thị biểu đồ

print("✨ Tuyệt vời! Bạn đã học xong setup cơ bản!")
print("➡️  Tiếp theo: Học các loại biểu đồ cơ bản...") 