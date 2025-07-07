"""
===========================================
GIAI ĐOẠN 1: NỀN TẢNG CƠ BẢN - BÀI TẬP
Thực hành các biểu đồ cơ bản
===========================================

🎯 MỤC TIÊU: Tự mình code các biểu đồ mà không nhìn solutions!

Làm từng bài một, sau đó so với solutions để học hỏi.
Đừng copy paste - hãy gõ lại để nhớ cú pháp!
"""

import matplotlib.pyplot as plt
import numpy as np

print("💪 BÀI TẬP THỰC HÀNH MATPLOTLIB!")
print("="*50)
print("Hãy uncomment từng bài và code theo yêu cầu")
print("="*50)

# ===========================================
# BÀI TẬP 1: VẼ ĐƯỜNG CƠ BẢN 
# ===========================================
print("\n📝 BÀI TẬP 1: Vẽ đường y = 2x + 3")
print("-" * 30)
print("""
YÊU CẦU:
1. Tạo x từ 0 đến 10 (dùng np.linspace với 50 điểm)
2. Tính y = 2*x + 3
3. Vẽ đường với màu đỏ
4. Đặt title: "Đường thẳng y = 2x + 3"
5. Đặt xlabel: "x", ylabel: "y"
6. Hiển thị grid
""")

# CODE CỦA BẠN Ở ĐÂY:
# x = ?
# y = ?
# plt.plot(?)
# ...

# ===========================================
# BÀI TẬP 2: SCATTER PLOT VỚI DỮ LIỆU THẬT
# ===========================================
print("\n📝 BÀI TẬP 2: Scatter plot điểm thi")
print("-" * 30)
print("""
YÊU CẦU:
Dữ liệu: Giờ học vs Điểm thi
hours_studied = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
scores = [65, 70, 75, 80, 82, 85, 88, 90, 92, 95]

1. Vẽ scatter plot với marker 'o', màu xanh, size=80
2. Title: "Mối quan hệ giữa giờ học và điểm thi"  
3. xlabel: "Giờ học", ylabel: "Điểm thi"
4. Thêm grid với alpha=0.3
""")

# Dữ liệu cho bạn
hours_studied = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
scores = [65, 70, 75, 80, 82, 85, 88, 90, 92, 95]

# CODE CỦA BẠN Ở ĐÂY:
# plt.figure(figsize=(8,6))
# plt.scatter(?)
# ...

# ===========================================
# BÀI TẬP 3: BIỂU ĐỒ CỘT
# ===========================================
print("\n📝 BÀI TẬP 3: Biểu đồ cột doanh thu")
print("-" * 30)
print("""
YÊU CẦU:
Dữ liệu doanh thu theo tháng:
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
revenue = [120, 150, 180, 200, 170]

1. Vẽ biểu đồ cột với màu 'green'
2. Title: "Doanh thu theo tháng"
3. xlabel: "Tháng", ylabel: "Doanh thu (triệu VND)"
4. Thêm giá trị lên đầu mỗi cột (hint: dùng plt.text)
5. Xoay labels trục x 45 độ (hint: plt.xticks(rotation=45))
""")

# Dữ liệu cho bạn  
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
revenue = [120, 150, 180, 200, 170]

# CODE CỦA BẠN Ở ĐÂY:
# bars = plt.bar(?)
# for i, v in enumerate(revenue):
#     plt.text(i, v + 5, str(v), ha='center')
# ...

# ===========================================
# BÀI TẬP 4: HISTOGRAM
# ===========================================
print("\n📝 BÀI TẬP 4: Histogram phân phối điểm")
print("-" * 30)
print("""
YÊU CẦU:
Tạo dữ liệu điểm thi random:
np.random.seed(42)
test_scores = np.random.normal(75, 10, 200)  # mean=75, std=10, 200 students

1. Vẽ histogram với 20 bins
2. Màu 'lightblue', alpha=0.7, edgecolor='black'  
3. Title: "Phân phối điểm thi"
4. xlabel: "Điểm", ylabel: "Số học sinh"
5. Thêm đường thẳng đứng màu đỏ ở vị trí mean
6. Thêm legend cho đường mean
""")

# Tạo dữ liệu cho bạn
np.random.seed(42)
test_scores = np.random.normal(75, 10, 200)

# CODE CỦA BẠN Ở ĐÂY:
# plt.hist(?)
# plt.axvline(np.mean(test_scores), color='red', linestyle='--', ?)
# ...

# ===========================================
# BÀI TẬP 5: KẾT HỢP TẤT CẢ (CHALLENGE!)
# ===========================================
print("\n📝 BÀI TẬP 5: 🔥 CHALLENGE - Linear Regression giống ví dụ của bạn!")
print("-" * 30)
print("""
YÊU CẦU:
Tái tạo CHÍNH XÁC ví dụ mà bạn đưa ra:

x_train = [1, 2, 3, 4, 5]
y_train = [300, 500, 700, 900, 1100]

Tự tính w và b để fit tốt nhất (dùng công thức hoặc thử nghiệm)
Sau đó vẽ:
1. Đường prediction màu xanh với label 'Our Prediction'
2. Scatter plot actual values màu đỏ, marker 'x', label 'Actual Values'
3. Title: "Housing Prices"  
4. ylabel: 'Price (in 1000s of dollars)'
5. xlabel: 'Size (1000 sqft)'
6. legend và show
""")

# Dữ liệu cho bạn
x_train = [1, 2, 3, 4, 5]  
y_train = [300, 500, 700, 900, 1100]

# CODE CỦA BẠN Ở ĐÂY:
# w = ?  # Hãy tính hoặc thử để tìm w tốt nhất
# b = ?  # Hãy tính hoặc thử để tìm b tốt nhất  
# tmp_f_wb = w * np.array(x_train) + b
# 
# plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
# plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
# plt.title("Housing Prices")
# plt.ylabel('Price (in 1000s of dollars)')
# plt.xlabel('Size (1000 sqft)')  
# plt.legend()
# plt.show()

print("\n" + "="*50)
print("🎯 HƯỚNG DẪN HOÀN THÀNH BÀI TẬP")
print("="*50)
print("""
1. Làm từng bài một cách tuần tự
2. Uncomment code và điền vào chỗ trống
3. Chạy từng section để test
4. So sánh với solutions sau khi làm xong
5. Gặp khó khăn? Xem lại bài học trước!

🚀 Sau khi làm xong, chạy: python 04_solutions.py để so sánh!
""")

print("💪 Chúc bạn làm bài tốt!") 