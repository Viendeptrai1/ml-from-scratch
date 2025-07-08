"""
===========================================
GIAI ĐOẠN 2: TÙY CHỈNH BIỂU ĐỒ - SOLUTIONS
Đáp án chi tiết cho styling nâng cao
===========================================

🎯 HÃY LÀM BÀI TẬP TRƯỚC KHI XEM SOLUTIONS!

File này chứa đáp án đầy đủ với giải thích để bạn master styling nâng cao.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

print("🔍 SOLUTIONS CHO BÀI TẬP STYLING NÂNG CAO")
print("="*50)

# ===========================================
# SOLUTION 1: FONT STYLING VÀ LATEX
# ===========================================
print("\n✅ SOLUTION 1: Font styling và LaTeX symbols")
print("-" * 40)

# Dữ liệu
x = np.linspace(-np.pi, np.pi, 1000)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y_sin, 'b-', linewidth=2, label=r'$\sin(x)$')
plt.plot(x, y_cos, 'r--', linewidth=2, label=r'$\cos(x)$')

# Advanced title styling
plt.title(r"Trigonometric Functions: $\sin(x)$ and $\cos(x)$", 
          fontsize=16, fontweight='bold', color='darkblue', pad=20)

# LaTeX labels
plt.xlabel(r'$x$ (radians)', fontsize=12)
plt.ylabel(r'$f(x)$', fontsize=12)

# Legend với 2 cột
plt.legend(ncol=2, title="Functions", title_fontsize=11, 
           loc='upper right', frameon=True, fancybox=True)

# Text box với LaTeX
plt.text(np.pi/2, 0.5, r'$\sin\left(\frac{\pi}{2}\right) = 1$', 
         fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
         ha='center')

# Grid và axis lines
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

print("💡 Giải thích:")
print("- r'$\\sin(x)$': LaTeX math mode với raw string")
print("- ncol=2: legend với 2 cột")
print("- bbox=dict(): tạo text box với bo góc")
print("- pad=20: khoảng cách title với plot")

# ===========================================
# SOLUTION 2: HEX COLORS VÀ LINE STYLES  
# ===========================================
print("\n✅ SOLUTION 2: Hex colors và custom line styles")
print("-" * 40)

# Dữ liệu
x_poly = np.linspace(0, 2, 100)
y1 = x_poly
y2 = x_poly**2
y3 = x_poly**3
y4 = x_poly**4

colors_hex = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
line_styles = ['-', '--', '-.', (0, (5, 2, 1, 2))]
line_widths = [2, 2.5, 3, 3.5]
labels = ['Linear', 'Quadratic', 'Cubic', 'Quartic']

plt.figure(figsize=(10, 6))
data_sets = [y1, y2, y3, y4]

for y_data, color, ls, lw, label in zip(data_sets, colors_hex, line_styles, line_widths, labels):
    plt.plot(x_poly, y_data, color=color, linestyle=ls, linewidth=lw, label=label)

plt.title("Polynomial Functions", fontsize=16, color='#2C3E50', fontweight='bold')
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)

# Legend outside plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

print("💡 Giải thích:")
print("- zip(): iterate qua multiple lists cùng lúc")
print("- (0, (5, 2, 1, 2)): custom dash pattern")
print("- bbox_to_anchor=(1.05, 1): legend outside plot")
print("- color='#2C3E50': hex color code")

# ===========================================
# SOLUTION 3: SCATTER PLOT VỚI TRANSPARENCY
# ===========================================
print("\n✅ SOLUTION 3: Advanced scatter plot với transparency")
print("-" * 40)

# Setup random data
np.random.seed(42)

# Group A
x_a = np.random.normal(2, 1, 200)
y_a = np.random.normal(2, 1, 200)

# Group B  
x_b = np.random.normal(3, 1, 200)
y_b = np.random.normal(3, 1, 200)

# Group C
x_c = np.random.normal(2.5, 1, 200)
y_c = np.random.normal(2.5, 1, 200)

plt.figure(figsize=(10, 8))

# Plot với different styles
plt.scatter(x_a, y_a, s=80, color='#E74C3C', marker='o', alpha=0.6, 
           edgecolors='white', linewidth=0.5, label='Group A')
plt.scatter(x_b, y_b, s=100, color='#3498DB', marker='s', alpha=0.6,
           edgecolors='white', linewidth=0.5, label='Group B')
plt.scatter(x_c, y_c, s=60, color='#2ECC71', marker='^', alpha=0.6,
           edgecolors='white', linewidth=0.5, label='Group C')

plt.title("Overlapping Data with Transparency", fontsize=14, fontweight='bold')
plt.xlabel('X values', fontsize=12)
plt.ylabel('Y values', fontsize=12)

# Advanced legend
plt.legend(framealpha=0.9, fancybox=True, shadow=True, 
          loc='upper left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("💡 Giải thích:")
print("- alpha=0.6: transparency để thấy overlapping")
print("- edgecolors='white': viền trắng nổi bật")
print("- framealpha=0.9: legend semi-transparent")
print("- np.random.normal(mean, std, size): generate normal distribution")

# ===========================================
# SOLUTION 4: PROFESSIONAL LINEAR REGRESSION
# ===========================================
print("\n✅ SOLUTION 4: 🔥 Professional Linear Regression")
print("-" * 40)

# Dữ liệu
x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = np.array([300, 500, 700, 900, 1100])
w, b = 200, 100
y_predicted = w * x_train + b

plt.figure(figsize=(12, 8))

# Gradient line segments
x_smooth = np.linspace(1, 5, 100)
y_smooth = w * x_smooth + b
colors = np.linspace(0, 1, len(x_smooth))

for i in range(len(x_smooth)-1):
    plt.plot([x_smooth[i], x_smooth[i+1]], [y_smooth[i], y_smooth[i+1]], 
             color=plt.cm.plasma(colors[i]), linewidth=4, alpha=0.9)

# Actual data points
plt.scatter(x_train, y_train, s=200, c='#E74C3C', marker='X', 
           edgecolors='white', linewidth=3, label='Actual Values', zorder=5)

# Predicted points  
plt.scatter(x_train, y_predicted, s=120, c='#8E44AD', marker='o', alpha=0.8,
           edgecolors='white', linewidth=2, label='Predictions', zorder=5)

plt.title("Professional Housing Price Analysis", 
          fontsize=20, fontweight='bold', color='#2C3E50', pad=25)
plt.xlabel('Size (1000 sqft)', fontsize=14, fontweight='semibold')
plt.ylabel('Price (in 1000s of dollars)', fontsize=14, fontweight='semibold')

# Professional legend
plt.legend(frameon=True, shadow=True, fancybox=True, 
          fontsize=12, loc='upper left')
plt.grid(True, alpha=0.25, linestyle='--', color='gray')
plt.tight_layout()
plt.show()

print("💡 Giải thích:")
print("- plt.cm.plasma(value): colormap gradient")
print("- zorder=5: draw order (higher = on top)")
print("- marker='X': capital X marker")
print("- linestyle='--': dashed grid lines")
print("- Gradient effect: multiple line segments với different colors")

# ===========================================
# SOLUTION 5: CUSTOM COLORMAP VÀ SUBPLOTS
# ===========================================
print("\n✅ SOLUTION 5: 🌈 Custom colormap và subplots")
print("-" * 40)

# Setup
custom_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700']

# Create custom colormap
cmap_custom = LinearSegmentedColormap.from_list('custom', custom_colors, N=100)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Heatmap
data_heat = np.random.rand(10, 10)
im1 = ax1.imshow(data_heat, cmap=cmap_custom)
ax1.set_title('Custom Heatmap', fontweight='bold')
plt.colorbar(im1, ax=ax1)

# Subplot 2: Scatter
np.random.seed(42)
x_scatter = np.random.randn(100)
y_scatter = np.random.randn(100)
c_scatter = np.random.rand(100)
scatter = ax2.scatter(x_scatter, y_scatter, s=50, c=c_scatter, 
                     cmap=cmap_custom, alpha=0.8, edgecolors='white', linewidth=0.5)
ax2.set_title('Custom Scatter', fontweight='bold')
plt.colorbar(scatter, ax=ax2)

# Subplot 3: Lines
x_line = np.linspace(0, 2*np.pi, 100)
ax3.plot(x_line, np.sin(x_line), color=custom_colors[0], linewidth=2, label='sin(x)')
ax3.plot(x_line, np.cos(x_line), color=custom_colors[1], linewidth=2, label='cos(x)')  
ax3.plot(x_line, np.sin(2*x_line), color=custom_colors[2], linewidth=2, label='sin(2x)')
ax3.set_title('Custom Lines', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Bars
data_bar = [10, 25, 30, 15]
labels_bar = ['A', 'B', 'C', 'D'] 
bars = ax4.bar(labels_bar, data_bar, color=custom_colors)
ax4.set_title('Custom Bars', fontweight='bold')

# Add value labels on bars
for bar, value in zip(bars, data_bar):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             str(value), ha='center', va='bottom')

plt.suptitle('Custom Colormap Showcase', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("💡 Giải thích:")
print("- LinearSegmentedColormap.from_list(): tạo custom colormap")
print("- plt.subplots(2, 2): 2x2 grid của subplots")
print("- plt.colorbar(im, ax=ax): colorbar cho specific subplot")
print("- plt.suptitle(): title cho toàn bộ figure")
print("- custom_colors[0]: access first color từ palette")

# ===========================================
# BONUS: COMBINED PROFESSIONAL VISUALIZATION
# ===========================================
print("\n🎁 BONUS: Combined Professional Visualization")
print("-" * 40)

fig = plt.figure(figsize=(16, 10))

# Main plot với gradient background
ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)

# Linear regression data
x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = np.array([300, 500, 700, 900, 1100])
w, b = 200, 100

# Professional gradient line
x_smooth = np.linspace(0.5, 5.5, 200)
y_smooth = w * x_smooth + b

# Create gradient effect
for i in range(len(x_smooth)-1):
    color_val = i / len(x_smooth)
    ax_main.plot([x_smooth[i], x_smooth[i+1]], [y_smooth[i], y_smooth[i+1]], 
                color=plt.cm.viridis(color_val), linewidth=3, alpha=0.8)

# Data points với professional styling
ax_main.scatter(x_train, y_train, s=150, c='#E74C3C', marker='D', 
               edgecolors='white', linewidth=2, label='Actual Prices', zorder=5)

predicted = w * x_train + b
ax_main.scatter(x_train, predicted, s=100, c='#3498DB', marker='o', 
               alpha=0.8, edgecolors='white', linewidth=1.5, 
               label='Model Predictions', zorder=5)

ax_main.set_title('Housing Price Prediction Model', 
                 fontsize=18, fontweight='bold', color='#2C3E50', pad=20)
ax_main.set_xlabel('House Size (1000 sqft)', fontsize=12, fontweight='semibold')
ax_main.set_ylabel('Price ($1000s)', fontsize=12, fontweight='semibold')
ax_main.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax_main.grid(True, alpha=0.3, linestyle=':', color='gray')

# Residuals plot
ax_residual = plt.subplot2grid((3, 3), (0, 2))
residuals = y_train - predicted
ax_residual.scatter(x_train, residuals, c='#E74C3C', s=60, alpha=0.7)
ax_residual.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax_residual.set_title('Residuals', fontsize=12, fontweight='bold')
ax_residual.set_xlabel('Size')
ax_residual.set_ylabel('Error')
ax_residual.grid(True, alpha=0.3)

# Error histogram
ax_hist = plt.subplot2grid((3, 3), (1, 2))
ax_hist.hist(residuals, bins=5, color='#3498DB', alpha=0.7, edgecolor='white')
ax_hist.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax_hist.set_xlabel('Error')
ax_hist.set_ylabel('Frequency')

# Model metrics text
ax_text = plt.subplot2grid((3, 3), (2, 0), colspan=3)
ax_text.axis('off')

# Calculate metrics
mse = np.mean(residuals**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(residuals))

metrics_text = f"""
📊 MODEL PERFORMANCE METRICS:
• Root Mean Square Error (RMSE): {rmse:.2f}
• Mean Absolute Error (MAE): {mae:.2f}
• Model Parameters: w = {w}, b = {b}
• R² Score: {1 - (np.sum(residuals**2) / np.sum((y_train - np.mean(y_train))**2)):.3f}
"""

ax_text.text(0.02, 0.8, metrics_text, fontsize=11, fontweight='semibold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#ECF0F1', alpha=0.8),
            verticalalignment='top')

plt.tight_layout()
plt.show()

print("✅ Professional dashboard với multiple subplots!")

# ===========================================
# TỔNG KẾT GIAI ĐOẠN 2
# ===========================================
print("\n" + "="*50)
print("🎯 TỔNG KẾT GIAI ĐOẠN 2 - STYLING NÂNG CAO")
print("="*50)
print("""
✅ ĐÃ MASTER:
1. Font styling: fontsize, fontweight, color, LaTeX
2. Color systems: hex, RGB, named colors
3. Line patterns: solid, dashed, custom patterns
4. Transparency effects: alpha blending, overlapping
5. Professional markers: sizes, edges, styles
6. Colormaps: viridis, plasma, custom colormaps
7. Advanced legends: ncol, positioning, styling
8. Complex layouts: subplots, annotations

🔥 PROFESSIONAL TECHNIQUES:
- Gradient lines với colormap segments
- Custom colormaps cho branding
- Multi-panel dashboards
- Error visualization
- Professional typography
- Edge styling cho contrast

💡 KEY LEARNINGS:
- Alpha=0.6-0.8 optimal cho overlapping data
- edgecolors='white' tạo professional look
- zorder controls drawing order
- bbox_to_anchor cho flexible legend positioning
- LaTeX với r'$...$' cho math expressions

🚀 BẠN ĐÃ SẴN SÀNG CHO GIAI ĐOẠN 3!
""")

print("🎉 Chúc mừng! Bạn đã master styling nâng cao!") 