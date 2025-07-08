"""
===========================================
GIAI ĐOẠN 2: TÙY CHỈNH BIỂU ĐỒ - BÀI 2
Colors, Styles và Transparency Nâng cao
===========================================

Bài này sẽ dạy bạn:
1. Color palettes và hex codes
2. Line styles và markers nâng cao
3. Transparency (alpha) và gradients
4. Colormaps cho visualization
5. Custom color schemes
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

print("🌈 COLORS, STYLES VÀ TRANSPARENCY NÂNG CAO")
print("="*50)

# ===========================================
# 1. COLOR SYSTEMS - HEX, RGB, NAMES
# ===========================================
print("\n🎨 1. Color systems - Hex, RGB, Names")

# Dữ liệu để demo
x = np.linspace(0, 10, 100)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Method 1: Named colors
colors_named = ['red', 'blue', 'green', 'orange', 'purple']
for i, color in enumerate(colors_named):
    y = np.sin(x + i)
    ax1.plot(x, y, color=color, linewidth=2, label=f'{color}')
ax1.set_title('Named Colors')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Method 2: Hex colors
colors_hex = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
for i, color in enumerate(colors_hex):
    y = np.sin(x + i)
    ax2.plot(x, y, color=color, linewidth=2, label=f'{color}')
ax2.set_title('Hex Colors')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Method 3: RGB tuples (0-1 range)
colors_rgb = [(1.0, 0.4, 0.4), (0.3, 0.8, 0.3), (0.3, 0.3, 1.0), 
              (0.8, 0.6, 0.8), (1.0, 0.8, 0.3)]
for i, color in enumerate(colors_rgb):
    y = np.sin(x + i)
    ax3.plot(x, y, color=color, linewidth=2, label=f'RGB{color}')
ax3.set_title('RGB Colors')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ 3 cách định màu: named ('red'), hex ('#FF6B6B'), RGB ((1,0,0))")

# ===========================================
# 2. LINE STYLES VÀ PATTERNS NÂNG CAO
# ===========================================
print("\n📏 2. Line styles và patterns nâng cao")

plt.figure(figsize=(12, 8))

# Custom line styles
line_styles = [
    ('-', 'solid'),
    ('--', 'dashed'), 
    ('-.', 'dashdot'),
    (':', 'dotted'),
    ((0, (5, 5)), 'custom dashed'),
    ((0, (1, 1)), 'densely dotted'),
    ((0, (5, 1, 1, 1)), 'dashdotted'),
    ((0, (3, 5, 1, 5, 1, 5)), 'custom pattern')
]

x = np.linspace(0, 10, 100)
colors = plt.cm.tab10(np.linspace(0, 1, len(line_styles)))

for i, ((style, name), color) in enumerate(zip(line_styles, colors)):
    y = np.sin(x) + i * 0.5
    plt.plot(x, y, linestyle=style, color=color, linewidth=2.5, label=name)

plt.title('Advanced Line Styles and Patterns', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('X values', fontsize=12)
plt.ylabel('Y values (offset)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("✅ Custom line patterns: (offset, (dash_length, gap_length, ...))")

# ===========================================
# 3. MARKERS VÀ MARKER STYLING
# ===========================================
print("\n🔴 3. Markers và marker styling nâng cao")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Basic markers
markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h', 'H', 'x', '+']
x_pos = range(len(markers))
y_pos = [1] * len(markers)

ax1.scatter(x_pos, y_pos, s=200, c=range(len(markers)), 
           marker='o', cmap='viridis', alpha=0.8)
for i, marker in enumerate(markers):
    ax1.scatter(i, 0.5, s=300, marker=marker, c='red', edgecolors='black', linewidth=2)
ax1.set_title('Marker Types')
ax1.set_ylim(0, 1.5)

# Marker sizes
sizes = [50, 100, 200, 400, 800]
ax2.scatter(range(len(sizes)), [1]*len(sizes), s=sizes, 
           c='blue', alpha=0.6, edgecolors='black')
ax2.set_title('Marker Sizes')
ax2.set_ylim(0.5, 1.5)

# Marker colors and edges
np.random.seed(42)
x_scatter = np.random.randn(50)
y_scatter = np.random.randn(50)
colors_scatter = np.random.rand(50)

scatter = ax3.scatter(x_scatter, y_scatter, s=100, c=colors_scatter, 
                     cmap='plasma', alpha=0.7, edgecolors='white', linewidth=1)
ax3.set_title('Colored Markers with Edges')
plt.colorbar(scatter, ax=ax3)

# Mixed markers
marker_types = ['o', 's', '^', 'D', 'v']
for i, marker in enumerate(marker_types):
    x_group = np.random.randn(10) + i*2
    y_group = np.random.randn(10) + i
    ax4.scatter(x_group, y_group, s=80, marker=marker, 
               alpha=0.7, label=f'Group {i+1}')
ax4.set_title('Multiple Marker Types')
ax4.legend()

plt.tight_layout()
plt.show()

print("✅ Marker styling: s (size), edgecolors, linewidth, alpha")

# ===========================================
# 4. TRANSPARENCY VÀ OVERLAPPING
# ===========================================
print("\n👻 4. Transparency và overlapping effects")

plt.figure(figsize=(12, 6))

# Left subplot: Alpha blending
plt.subplot(1, 2, 1)
np.random.seed(42)
for i in range(5):
    x_data = np.random.randn(100) + i
    y_data = np.random.randn(100) + i
    plt.scatter(x_data, y_data, s=80, alpha=0.6, label=f'Dataset {i+1}')

plt.title('Alpha Blending for Overlapping Data', fontsize=12)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid(True, alpha=0.3)

# Right subplot: Gradient effects
plt.subplot(1, 2, 2)
x = np.linspace(0, 10, 100)
colors = np.linspace(0.1, 1.0, 100)

# Create gradient effect với multiple lines
for i in range(100):
    alpha_val = colors[i]
    plt.plot([x[i], x[i]], [0, np.sin(x[i])], 
             color='blue', alpha=alpha_val, linewidth=2)

plt.title('Gradient Effect with Alpha', fontsize=12)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Alpha blending giúp hiển thị overlapping data rõ ràng hơn!")

# ===========================================
# 5. COLORMAPS VÀ COLOR SCALES
# ===========================================
print("\n🗺️ 5. Colormaps và color scales")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Colormap examples
cmaps = ['viridis', 'plasma', 'inferno', 'cool']
data = np.random.rand(20, 20)

for i, (ax, cmap) in enumerate(zip(axes.flat, cmaps)):
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    ax.set_title(f'Colormap: {cmap}')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

# Linear regression với colormap
print("\n🔥 Linear Regression với Professional Colormap")
plt.figure(figsize=(10, 7))

x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = np.array([300, 500, 700, 900, 1100])
w, b = 200, 100
y_predicted = w * x_train + b

# Prediction line với gradient
x_smooth = np.linspace(1, 5, 100)
y_smooth = w * x_smooth + b
colors = np.linspace(0, 1, len(x_smooth))

# Create segments với colors
for i in range(len(x_smooth)-1):
    plt.plot([x_smooth[i], x_smooth[i+1]], [y_smooth[i], y_smooth[i+1]], 
             color=plt.cm.coolwarm(colors[i]), linewidth=3, alpha=0.8)

# Actual data points với professional styling
plt.scatter(x_train, y_train, s=150, c='#2E86AB', marker='x', 
           linewidth=4, label='Actual Values', edgecolors='white', zorder=5)

# Predicted points
plt.scatter(x_train, y_predicted, s=100, c='#A23B72', marker='o', 
           alpha=0.8, label='Predictions', edgecolors='white', linewidth=2, zorder=5)

plt.title('Professional Housing Price Prediction', 
          fontsize=18, fontweight='bold', color='#2C3E50', pad=25)
plt.xlabel('Size (1000 sqft)', fontsize=14, fontweight='semibold', color='#34495E')
plt.ylabel('Price (in 1000s of dollars)', fontsize=14, fontweight='semibold', color='#34495E')

# Advanced legend
plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, 
          framealpha=0.9, facecolor='#ECF0F1', edgecolor='#BDC3C7')

plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='gray')
plt.tight_layout()
plt.show()

print("✅ Professional visualization với colormaps và advanced styling!")

# ===========================================
# 6. CUSTOM COLOR PALETTES
# ===========================================
print("\n🎨 6. Custom color palettes")

# Create custom colormap
colors_custom = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
n_bins = 100
cmap_custom = LinearSegmentedColormap.from_list('custom', colors_custom, N=n_bins)

# Demo custom colormap
plt.figure(figsize=(12, 8))

# Subplot 1: Heatmap với custom colormap
plt.subplot(2, 2, 1)
data_heat = np.random.rand(10, 10)
im1 = plt.imshow(data_heat, cmap=cmap_custom)
plt.title('Custom Colormap Heatmap')
plt.colorbar(im1)

# Subplot 2: Scatter với custom colors
plt.subplot(2, 2, 2)
np.random.seed(42)
x_scatter = np.random.randn(100)
y_scatter = np.random.randn(100)
c_scatter = np.random.rand(100)
scatter = plt.scatter(x_scatter, y_scatter, s=60, c=c_scatter, 
                     cmap=cmap_custom, alpha=0.7, edgecolors='white', linewidth=0.5)
plt.title('Custom Colormap Scatter')
plt.colorbar(scatter)

# Subplot 3: Line plots với custom palette
plt.subplot(2, 2, 3)
x = np.linspace(0, 10, 100)
for i, color in enumerate(colors_custom):
    y = np.sin(x + i) + i * 0.5
    plt.plot(x, y, color=color, linewidth=3, label=f'Line {i+1}')
plt.title('Custom Color Palette')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 4: Gradient bar
plt.subplot(2, 2, 4)
gradient = np.linspace(0, 1, 256).reshape(1, -1)
plt.imshow(gradient, aspect='auto', cmap=cmap_custom)
plt.title('Custom Gradient Bar')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()

print("✅ Custom colormaps với LinearSegmentedColormap.from_list()!")

# ===========================================
# TỔNG KẾT BÀI 2
# ===========================================
print("\n" + "="*50)
print("🎯 TỔNG KẾT BÀI 2 - COLORS & STYLES NÂNG CAO")
print("="*50)
print("""
✅ ĐÃ HỌC:
1. Color systems: named, hex (#FF6B6B), RGB ((1,0,0))
2. Line styles: solid, dashed, custom patterns
3. Marker styling: sizes, edges, transparency
4. Alpha blending cho overlapping data
5. Colormaps: viridis, plasma, custom colormaps
6. Professional styling techniques

🔥 CÚ PHÁP QUAN TRỌNG:
- color='#FF6B6B' hoặc color=(1,0,0)  # Hex và RGB
- linestyle='--' hoặc linestyle=(0, (5,5))  # Custom patterns
- alpha=0.7  # Transparency
- s=100, edgecolors='white', linewidth=2  # Marker styling
- cmap='viridis'  # Colormaps
- LinearSegmentedColormap.from_list('name', colors)  # Custom

💡 TIPS:
- Dùng alpha=0.6-0.8 cho overlapping data
- edgecolors='white' làm markers nổi bật
- Custom colormaps cho branding
- plt.cm.colormap_name(value) để lấy color từ colormap
""")

print("🚀 Tiếp theo: Thực hành styling nâng cao!") 