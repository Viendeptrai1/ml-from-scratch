# 📚 Khóa học Matplotlib.pyplot từ Cơ bản đến Nâng cao

## 🎯 Mục tiêu
Học cú pháp matplotlib.pyplot để có thể tự vẽ biểu đồ Machine Learning như:
```python
tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
```

---

## 📖 GIAI ĐOẠN 1: NỀN TẢNG CƠ BẢN (2-3 ngày)

### 🎓 Thứ tự học tập:

1. **[01_setup_basics.py](01_setup_basics.py)** - Bắt đầu ở đây!
   - Import matplotlib.pyplot as plt
   - Hiểu figure và axes
   - plt.show() cơ bản

2. **[02_basic_plots.py](02_basic_plots.py)** - Học 4 loại biểu đồ cơ bản
   - plt.plot() - Vẽ đường thẳng  
   - plt.scatter() - Scatter plot
   - plt.bar() - Biểu đồ cột
   - plt.hist() - Histogram

3. **[03_practice_exercises.py](03_practice_exercises.py)** - Thực hành!
   - 5 bài tập từ dễ đến khó
   - Bài cuối là recreation chính xác ví dụ của bạn
   - **HÃY TỰ LÀM TRƯỚC KHI XEM SOLUTIONS!**

4. **[04_solutions.py](04_solutions.py)** - So sánh và học hỏi
   - Đáp án chi tiết có giải thích
   - Tips và tricks
   - Bonus: subplots

---

## 🚀 Cách học hiệu quả:

### Bước 1: Chạy các file học
```bash
python 01_setup_basics.py
python 02_basic_plots.py
```

### Bước 2: Làm bài tập
```bash
# Mở file và làm từng bài
code 03_practice_exercises.py
```

### Bước 3: So sánh đáp án
```bash
python 04_solutions.py
```

---

## 💡 Tips quan trọng:

✅ **Luôn import đầu tiên:**
```python
import matplotlib.pyplot as plt
import numpy as np
```

✅ **Template cơ bản:**
```python
plt.figure(figsize=(8, 6))
plt.plot(x, y)  # hoặc scatter, bar, hist
plt.title("Your Title")
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.legend()
plt.grid(True)
plt.show()
```

✅ **Cú pháp từ ví dụ của bạn:**
```python
plt.plot(x, y, c='b', label='Our Prediction')
plt.scatter(x, y, marker='x', c='r', label='Actual Values')
```

---

## 🎯 Sau khi hoàn thành Giai đoạn 1:

Bạn sẽ biết:
- ✅ Import và setup matplotlib
- ✅ Vẽ 4 loại biểu đồ cơ bản
- ✅ Thêm labels, titles, legends  
- ✅ Tùy chỉnh màu sắc và markers
- ✅ **Recreate chính xác ví dụ linear regression của bạn!**

---

## 📊 Tiến độ học tập:

- [ ] 01_setup_basics.py
- [ ] 02_basic_plots.py  
- [ ] 03_practice_exercises.py (5 bài tập)
- [ ] 04_solutions.py
- [ ] Tự tin vẽ biểu đồ ML cơ bản!

---

## 🎨 GIAI ĐOẠN 2: TÙY CHỈNH BIỂU ĐỒ NÂNG CAO (3-4 ngày)

### 🎓 Thứ tự học tập:

5. **[05_labels_titles_advanced.py](05_labels_titles_advanced.py)** - Font styling & LaTeX
   - Advanced font styling: sizes, weights, colors
   - LaTeX math symbols và Greek letters
   - Text positioning và annotations
   - Legend styling nâng cao

6. **[06_colors_styles_advanced.py](06_colors_styles_advanced.py)** - Colors & Styles
   - Color systems: hex, RGB, named colors
   - Custom line patterns và markers
   - Transparency effects (alpha blending)
   - Colormaps và custom color schemes

7. **[07_practice_exercises_stage2.py](07_practice_exercises_stage2.py)** - Advanced Practice!
   - 5 bài tập styling nâng cao
   - Professional linear regression với gradient
   - Custom colormaps và subplots
   - **CHALLENGE: Tái tạo professional visualization!**

8. **[08_solutions_stage2.py](08_solutions_stage2.py)** - Professional Solutions
   - Solutions với pro techniques
   - Dashboard-style multi-panel plots
   - BONUS: Complete ML visualization dashboard

---

## 🎯 Sau khi hoàn thành Giai đoạn 2:

Bạn sẽ biết:
- ✅ Professional font styling với LaTeX
- ✅ Advanced color schemes và transparency
- ✅ Custom line patterns và markers
- ✅ Gradient effects và colormaps
- ✅ Multi-panel layouts
- ✅ **Tạo professional ML visualizations!**

---

## 📊 Tiến độ học tập FULL:

**Giai đoạn 1: Cơ bản**
- [ ] 01_setup_basics.py
- [ ] 02_basic_plots.py  
- [ ] 03_practice_exercises.py (5 bài tập)
- [ ] 04_solutions.py

**Giai đoạn 2: Nâng cao**
- [ ] 05_labels_titles_advanced.py
- [ ] 06_colors_styles_advanced.py
- [ ] 07_practice_exercises_stage2.py (5 bài tập pro)
- [ ] 08_solutions_stage2.py + BONUS dashboard

---

## 🔥 Ready cho Giai đoạn 3?

Sau khi master cả 2 giai đoạn, chúng ta sẽ học:
- Subplots và complex layouts
- Biểu đồ chuyên dụng cho ML (confusion matrix, ROC curves)
- Animations và interactive plots
- Publication-ready figures

**Chúc bạn học tốt! 🚀** 