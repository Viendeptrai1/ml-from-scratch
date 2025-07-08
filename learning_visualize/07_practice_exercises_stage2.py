"""
===========================================
GIAI ĐOẠN 2: TÙY CHỈNH BIỂU ĐỒ - BÀI TẬP
Thực hành Styling Nâng cao
===========================================

🎯 MỤC TIÊU: Master các kỹ thuật styling nâng cao!

Làm từng bài một, tự gõ code (không copy-paste) để nhớ cú pháp.
So sánh với solutions sau khi hoàn thành!
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

print("💪 BÀI TẬP STYLING NÂNG CAO!")
print("="*50)
print("Hãy uncomment từng bài và code theo yêu cầu")
print("="*50)

# ===========================================
# BÀI TẬP 1: FONT STYLING VÀ LATEX
# ===========================================
print("\n📝 BÀI TẬP 1: Font styling và LaTeX symbols")
print("-" * 40)
print("""
YÊU CẦU:
1. Vẽ hàm y = sin(x) và y = cos(x) từ -π đến π
2. Title: "Trigonometric Functions: $\\sin(x)$ and $\\cos(x)$" 
   - fontsize=16, fontweight='bold', color='darkblue'
3. xlabel: "$x$ (radians)", ylabel: "$f(x)$"  
4. Legend với 2 cột (ncol=2), title="Functions"
5. Thêm text box ở (π/2, 0.5) với nội dung: "$\\sin(\\frac{\\pi}{2}) = 1$"
6. Grid với alpha=0.3, đường axis tại x=0 và y=0
""")

# Dữ liệu cho bạn:
x = np.linspace(-np.pi, np.pi, 1000)
y_sin = np.sin(x)
y_cos = np.cos(x)

# CODE CỦA BẠN Ở ĐÂY:
# plt.figure(figsize=(10, 6))
# plt.plot(x, y_sin, ?, label=r'$\sin(x)$')
# plt.plot(x, y_cos, ?, label=r'$\cos(x)$')
# plt.title(?)
# plt.xlabel(?)
# plt.ylabel(?)
# plt.legend(?)
# plt.text(?, ?, r'$\sin(\frac{\pi}{2}) = 1$', ?)
# plt.grid(?)
# plt.axhline(?)
# plt.axvline(?)
# plt.show()

# ===========================================
# BÀI TẬP 2: HEX COLORS VÀ LINE STYLES
# ===========================================
print("\n📝 BÀI TẬP 2: Hex colors và custom line styles")
print("-" * 40)
print("""
YÊU CẦU:
Vẽ 4 đường với data polynomial:
y1 = x, y2 = x², y3 = x³, y4 = x⁴ (x từ 0 đến 2)

1. Colors: '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'
2. Line styles: solid, dashed, dashdot, custom pattern (0, (5, 2, 1, 2))
3. Line widths: 2, 2.5, 3, 3.5
4. Labels: 'Linear', 'Quadratic', 'Cubic', 'Quartic'
5. Title: "Polynomial Functions" với fontsize=16, color='#2C3E50'
6. Legend outside plot (bbox_to_anchor=(1.05, 1))
7. Grid với alpha=0.2
""")

# Dữ liệu cho bạn:
x_poly = np.linspace(0, 2, 100)
y1 = x_poly
y2 = x_poly**2
y3 = x_poly**3
y4 = x_poly**4

colors_hex = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
line_styles = ['-', '--', '-.', (0, (5, 2, 1, 2))]
line_widths = [2, 2.5, 3, 3.5]
labels = ['Linear', 'Quadratic', 'Cubic', 'Quartic']

# CODE CỦA BẠN Ở ĐÂY:
# plt.figure(figsize=(10, 6))
# data_sets = [y1, y2, y3, y4]
# for i, (y_data, color, ls, lw, label) in enumerate(zip(?)):
#     plt.plot(?, ?, color=?, linestyle=?, linewidth=?, label=?)
# plt.title(?)
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.legend(?)
# plt.grid(?)
# plt.tight_layout()
# plt.show()

# ===========================================
# BÀI TẬP 3: SCATTER PLOT VỚI TRANSPARENCY
# ===========================================
print("\n📝 BÀI TẬP 3: Advanced scatter plot với transparency")
print("-" * 40)
print("""
YÊU CẦU:
Tạo scatter plot với overlapping data:

1. Tạo 3 datasets với np.random (seed=42):
   - Group A: x~N(2,1), y~N(2,1), 200 points
   - Group B: x~N(3,1), y~N(3,1), 200 points  
   - Group C: x~N(2.5,1), y~N(2.5,1), 200 points

2. Colors: '#E74C3C', '#3498DB', '#2ECC71'
3. Markers: 'o', 's', '^'
4. Sizes: 80, 100, 60
5. Alpha: 0.6 cho tất cả
6. Edge colors: 'white', linewidth=0.5
7. Title: "Overlapping Data with Transparency"
8. Labels: 'Group A', 'Group B', 'Group C'
9. Legend với framealpha=0.9, fancybox=True
""")

# Setup random data cho bạn:
np.random.seed(42)

# CODE CỦA BẠN Ở ĐÂY:
# # Group A
# x_a = np.random.normal(2, 1, 200)
# y_a = np.random.normal(2, 1, 200)
# 
# # Group B  
# x_b = ?
# y_b = ?
# 
# # Group C
# x_c = ?
# y_c = ?
# 
# plt.figure(figsize=(10, 8))
# plt.scatter(?, ?, s=?, color=?, marker=?, alpha=?, 
#             edgecolors=?, linewidth=?, label=?)
# plt.scatter(?, ?, ?) # Group B
# plt.scatter(?, ?, ?) # Group C
# 
# plt.title(?)
# plt.xlabel('X values')
# plt.ylabel('Y values')
# plt.legend(?)
# plt.grid(True, alpha=0.3)
# plt.show()

# ===========================================
# BÀI TẬP 4: COLORMAP VÀ PROFESSIONAL STYLING  
# ===========================================
print("\n📝 BÀI TẬP 4: 🔥 CHALLENGE - Professional Linear Regression")
print("-" * 40)
print("""
YÊU CẦU:
Tái tạo professional version của linear regression với:

Data: x_train = [1,2,3,4,5], y_train = [300,500,700,900,1100]

1. Prediction line với gradient color (dùng plt.cm.plasma colormap)
   - Chia line thành segments và color từ 0 đến 1
   - linewidth=4, alpha=0.9

2. Actual data points:
   - color='#E74C3C', marker='X', s=200  
   - edgecolors='white', linewidth=3
   - label='Actual Values'

3. Predicted points:
   - color='#8E44AD', marker='o', s=120
   - edgecolors='white', linewidth=2  
   - label='Predictions', alpha=0.8

4. Title: "Professional Housing Price Analysis"
   - fontsize=20, fontweight='bold', color='#2C3E50'
   - pad=25

5. Labels với fontsize=14, fontweight='semibold'
6. Legend: frameon=True, shadow=True, fancybox=True
7. Grid: alpha=0.25, linestyle='--', color='gray'
""")

# Dữ liệu cho bạn:
x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = np.array([300, 500, 700, 900, 1100])
w, b = 200, 100
y_predicted = w * x_train + b

# CODE CỦA BẠN Ở ĐÂY:
# plt.figure(figsize=(12, 8))
# 
# # Gradient line segments
# x_smooth = np.linspace(1, 5, 100)
# y_smooth = w * x_smooth + b
# colors = np.linspace(0, 1, len(x_smooth))
# 
# for i in range(len(x_smooth)-1):
#     plt.plot([x_smooth[i], x_smooth[i+1]], [y_smooth[i], y_smooth[i+1]], 
#              color=plt.cm.plasma(?), linewidth=?, alpha=?)
# 
# # Actual data points
# plt.scatter(?, ?, s=?, c=?, marker=?, 
#            edgecolors=?, linewidth=?, label=?, zorder=5)
# 
# # Predicted points  
# plt.scatter(?, ?, s=?, c=?, marker=?, alpha=?,
#            edgecolors=?, linewidth=?, label=?, zorder=5)
# 
# plt.title(?)
# plt.xlabel(?, fontsize=?, fontweight=?)
# plt.ylabel(?, fontsize=?, fontweight=?)
# plt.legend(?)
# plt.grid(?)
# plt.tight_layout()
# plt.show()

# ===========================================
# BÀI TẬP 5: CUSTOM COLORMAP
# ===========================================
print("\n📝 BÀI TẬP 5: 🌈 Custom colormap và subplots")
print("-" * 40)
print("""
YÊU CẦU:
Tạo figure với 2x2 subplots:

1. Custom colormap từ colors: ['#FF9999', '#66B2FF', '#99FF99', '#FFD700']

2. Subplot 1 (top-left): Heatmap 10x10 với random data
   - Title: "Custom Heatmap"
   - Colorbar

3. Subplot 2 (top-right): Scatter plot 100 points  
   - x, y random normal, colors random [0,1]
   - Custom colormap, s=50, alpha=0.8
   - Title: "Custom Scatter"

4. Subplot 3 (bottom-left): 3 line plots với custom colors
   - y1=sin(x), y2=cos(x), y3=sin(2x) 
   - Dùng first 3 colors của custom palette
   - Title: "Custom Lines"

5. Subplot 4 (bottom-right): Bar chart
   - data = [10, 25, 30, 15]
   - labels = ['A', 'B', 'C', 'D']
   - Custom colors, Title: "Custom Bars"

6. Figure title: "Custom Colormap Showcase", fontsize=16
""")

# Setup cho bạn:
custom_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700']

# CODE CỦA BẠN Ở ĐÂY:
# # Create custom colormap
# cmap_custom = LinearSegmentedColormap.from_list(?, ?, N=100)
# 
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
# 
# # Subplot 1: Heatmap
# data_heat = np.random.rand(10, 10)
# im1 = ax1.imshow(?, cmap=?)
# ax1.set_title(?)
# plt.colorbar(?, ax=?)
# 
# # Subplot 2: Scatter
# np.random.seed(42)
# x_scatter = ?
# y_scatter = ?
# c_scatter = ?
# scatter = ax2.scatter(?, ?, s=?, c=?, cmap=?, alpha=?)
# ax2.set_title(?)
# 
# # Subplot 3: Lines
# x_line = np.linspace(0, 2*np.pi, 100)
# ax3.plot(?, ?, color=?, linewidth=2, label='sin(x)')
# ax3.plot(?, ?, color=?, linewidth=2, label='cos(x)')  
# ax3.plot(?, ?, color=?, linewidth=2, label='sin(2x)')
# ax3.set_title(?)
# ax3.legend()
# 
# # Subplot 4: Bars
# data_bar = [10, 25, 30, 15]
# labels_bar = ['A', 'B', 'C', 'D'] 
# ax4.bar(?, ?, color=?)
# ax4.set_title(?)
# 
# plt.suptitle(?, fontsize=?)
# plt.tight_layout()
# plt.show()

print("\n" + "="*50)
print("🎯 HƯỚNG DẪN HOÀN THÀNH BÀI TẬP")
print("="*50)
print("""
1. Làm từng bài theo thứ tự
2. Tự gõ lại code, đừng copy-paste
3. Chạy từng section để kiểm tra
4. Gặp khó khăn? Review lại bài học
5. So sánh với solutions sau khi xong

🚀 Tips:
- r'$\LaTeX$' cho math symbols
- alpha=0.6-0.8 cho overlapping data
- edgecolors='white' làm nổi bật markers
- bbox_to_anchor=(1.05, 1) cho legend outside
- plt.cm.colormap_name(value) cho gradient

🎯 Mục tiêu: Tự tin styling professional plots!
""")

print("💪 Chúc bạn làm bài tốt!") 