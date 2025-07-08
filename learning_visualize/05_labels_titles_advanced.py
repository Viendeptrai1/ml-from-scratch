"""
===========================================
GIAI ĐOẠN 2: TÙY CHỈNH BIỂU ĐỒ - BÀI 1
Labels, Titles và Font Styling Nâng cao
===========================================

Bài này sẽ dạy bạn:
1. Tùy chỉnh fonts, sizes, colors cho text
2. Positioning và rotation của labels
3. Math symbols và special characters
4. Legend styling nâng cao
5. Annotations và text boxes
"""

import matplotlib.pyplot as plt
import numpy as np

print("🎨 LABELS, TITLES VÀ FONT STYLING NÂNG CAO")
print("="*50)

# ===========================================
# 1. FONT SIZES VÀ STYLES CƠ BẢN
# ===========================================
print("\n📝 1. Font sizes và styles cơ bản")

# Dữ liệu linear regression quen thuộc
x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = np.array([300, 500, 700, 900, 1100])
w, b = 200, 100
y_predicted = w * x_train + b

plt.figure(figsize=(10, 7))
plt.plot(x_train, y_predicted, c='b', label='Our Prediction', linewidth=2)
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values', s=100)

# Font styling nâng cao
plt.title("Housing Prices - Advanced Styling", 
          fontsize=18,           # Kích thước font
          fontweight='bold',     # Độ đậm
          color='darkblue',      # Màu
          pad=20)                # Khoảng cách từ plot

plt.xlabel('Size (1000 sqft)', 
           fontsize=14, 
           fontweight='semibold', 
           color='darkgreen')

plt.ylabel('Price (in 1000s of dollars)', 
           fontsize=14, 
           fontweight='semibold', 
           color='darkgreen')

# Legend styling
plt.legend(fontsize=12, 
           loc='upper left',           # Vị trí
           frameon=True,               # Có khung
           fancybox=True,              # Khung bo tròn
           shadow=True,                # Có đổ bóng
           framealpha=0.9,             # Độ trong suốt khung
           facecolor='lightgray')      # Màu nền

plt.grid(True, alpha=0.3)
plt.show()

print("✅ Advanced styling cho titles và labels!")

# ===========================================
# 2. FONT FAMILIES VÀ CUSTOM FONTS
# ===========================================
print("\n🔤 2. Font families và custom fonts")

plt.figure(figsize=(10, 6))

# Thử các font families khác nhau
fonts = ['serif', 'sans-serif', 'monospace', 'fantasy']
colors = ['red', 'blue', 'green', 'purple']

x = np.linspace(0, 10, 100)
for i, (font, color) in enumerate(zip(fonts, colors)):
    y = np.sin(x + i)
    plt.plot(x, y, label=f'{font} font', color=color, linewidth=2)

plt.title('Different Font Families Demo', 
          fontfamily='serif',      # Font family
          fontsize=16, 
          style='italic')          # Style: normal, italic, oblique

plt.xlabel('X values', fontfamily='sans-serif', fontsize=12)
plt.ylabel('Y values', fontfamily='monospace', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()

print("💡 Font families: serif, sans-serif, monospace, fantasy, cursive")

# ===========================================
# 3. MATH SYMBOLS VÀ LATEX
# ===========================================
print("\n🧮 3. Math symbols và LaTeX")

plt.figure(figsize=(10, 6))

# Vẽ các hàm toán học
x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.exp(-x**2/10) * np.sin(x)

plt.plot(x, y1, 'b-', label=r'$y = \sin(x)$', linewidth=2)
plt.plot(x, y2, 'r--', label=r'$y = \cos(x)$', linewidth=2)  
plt.plot(x, y3, 'g:', label=r'$y = e^{-x^2/10} \sin(x)$', linewidth=2)

# LaTeX trong titles và labels
plt.title(r'Mathematical Functions: $f(x) = \sin(x), \cos(x), e^{-x^2/10}\sin(x)$', 
          fontsize=14, pad=20)
plt.xlabel(r'$x$ (radians)', fontsize=12)
plt.ylabel(r'$f(x)$', fontsize=12)

# Greek letters và symbols
plt.text(-5, 0.8, r'$\alpha = \frac{\pi}{4}$', fontsize=14, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
plt.text(3, -0.8, r'$\beta = \sqrt{2}$', fontsize=14,
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()

print("✅ LaTeX math symbols: $, \\alpha, \\beta, \\pi, \\frac{}{}, \\sqrt{}, ^{}, _{}")

# ===========================================
# 4. TEXT POSITIONING VÀ ROTATION
# ===========================================
print("\n🔄 4. Text positioning và rotation")

plt.figure(figsize=(10, 8))

# Tạo scatter plot với annotations
np.random.seed(42)
x_points = np.random.randn(20)
y_points = np.random.randn(20)
colors = np.random.rand(20)

scatter = plt.scatter(x_points, y_points, c=colors, s=100, alpha=0.7, cmap='viridis')

# Annotations với arrows
for i in range(5):  # Chỉ annotate 5 điểm đầu
    plt.annotate(f'Point {i+1}', 
                xy=(x_points[i], y_points[i]),     # Điểm cần annotate
                xytext=(10, 10),                   # Offset từ điểm
                textcoords='offset points',        # Tọa độ relative
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'))

# Rotated labels
plt.xlabel('X-axis with normal label', fontsize=12)
plt.ylabel('Y-axis with\nrotated label', fontsize=12, rotation=0, labelpad=40)

# Title với multi-line
plt.title('Scatter Plot with Annotations\nand Custom Text Positioning', 
          fontsize=14, pad=20)

# Text ở các vị trí khác nhau
plt.text(-2.5, 2.5, 'Top Left', fontsize=12, ha='left', va='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.3))
plt.text(2.5, 2.5, 'Top Right', fontsize=12, ha='right', va='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.3))
plt.text(0, -2.5, 'Bottom Center', fontsize=12, ha='center', va='bottom',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.3))

plt.colorbar(scatter, label='Color Scale')
plt.grid(True, alpha=0.3)
plt.show()

print("✅ Text positioning: ha (left/center/right), va (top/center/bottom)")

# ===========================================
# 5. LEGEND NÂNG CAO
# ===========================================
print("\n🏷️ 5. Legend styling nâng cao")

plt.figure(figsize=(12, 8))

# Multiple datasets với different styles
x = np.linspace(0, 10, 100)
datasets = [
    ('Linear', x, 'red', '-', 'o', 4),
    ('Quadratic', x**1.5, 'blue', '--', 's', 4),
    ('Exponential', np.exp(x/5), 'green', ':', '^', 4),
    ('Logarithmic', np.log(x+1)*10, 'purple', '-.', 'D', 4)
]

for name, y_data, color, ls, marker, ms in datasets:
    # Chỉ sample một số điểm cho markers
    plt.plot(x, y_data, color=color, linestyle=ls, linewidth=2, label=name)
    plt.plot(x[::10], y_data[::10], color=color, marker=marker, 
             markersize=ms, linestyle='None', markerfacecolor='white', 
             markeredgecolor=color, markeredgewidth=2)

# Legend với multiple columns và custom positioning
legend1 = plt.legend(loc='upper left', ncol=2,           # 2 cột
                    fontsize=11,
                    title='Function Types',              # Tiêu đề legend
                    title_fontsize=12,
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    framealpha=0.9,
                    facecolor='lightgray',
                    edgecolor='black')

# Tùy chỉnh title của legend
legend1.get_title().set_fontweight('bold')
legend1.get_title().set_color('darkred')

plt.title('Advanced Legend Styling Example', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('X values', fontsize=12)
plt.ylabel('Y values', fontsize=12)
plt.grid(True, alpha=0.3)

# Thêm text box với thông tin
textstr = 'Key Points:\n• Multiple line styles\n• Custom markers\n• 2-column legend'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.7, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.show()

print("✅ Legend nâng cao: ncol, title, custom positioning!")

# ===========================================
# TỔNG KẾT BÀI 1
# ===========================================
print("\n" + "="*50)
print("🎯 TỔNG KẾT BÀI 1 - LABELS & TITLES NÂNG CAO")
print("="*50)
print("""
✅ ĐÃ HỌC:
1. Font styling: fontsize, fontweight, color, fontfamily
2. Math symbols với LaTeX: $\\alpha$, $\\frac{}{}$, $\\sqrt{}$
3. Text positioning: ha, va, rotation, annotations
4. Legend nâng cao: ncol, title, custom styling
5. Text boxes và arrows

🔥 CÚ PHÁP QUAN TRỌNG:
- plt.title('Title', fontsize=16, fontweight='bold', color='blue')
- plt.xlabel(r'$x$ (units)', fontsize=12)  # LaTeX
- plt.legend(ncol=2, title='Legend Title', loc='upper left')
- plt.text(x, y, 'Text', ha='center', va='top')
- plt.annotate('Label', xy=(x,y), xytext=(10,10))

💡 TIPS:
- Dùng r'$...$' cho LaTeX math
- pad=20 để tăng khoảng cách title
- ha='center', va='top' để căn chỉnh text
- bbox=dict() để tạo text box
""")

print("🚀 Tiếp theo: Học colors và styles nâng cao!") 