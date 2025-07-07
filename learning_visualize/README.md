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

## 🔥 Ready cho Giai đoạn 2?

Sau khi master Giai đoạn 1, chúng ta sẽ học:
- Tùy chỉnh biểu đồ nâng cao
- Subplots và layouts
- Biểu đồ chuyên dụng cho ML
- Animations và interactive plots

**Chúc bạn học tốt! 🚀** 