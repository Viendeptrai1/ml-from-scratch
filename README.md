# Machine Learning từ đầu

Dự án machine learning hoàn chỉnh từ thu thập dữ liệu đến deploy model.

## Cấu trúc dự án

```
project/
├── data/
│   ├── raw/              # Dữ liệu gốc chưa xử lý
│   └── processed/        # Dữ liệu đã tiền xử lý
├── notebooks/
│   ├── EDA.ipynb         # Phân tích dữ liệu khám phá
│   ├── baseline_models.ipynb  # Các mô hình ML cơ bản
│   └── deep_learning.ipynb    # Mô hình deep learning
├── src/
│   ├── data_utils.py     # Tiện ích xử lý dữ liệu
│   ├── models.py         # Định nghĩa các model
│   ├── train.py          # Training và cross validation
│   ├── evaluate.py       # Đánh giá model
│   └── visualization.py  # Visualization
├── outputs/
│   ├── models/           # Model đã train
│   ├── logs/             # Training logs
│   └── figures/          # Hình ảnh biểu đồ
├── app/
│   ├── app.py            # Web app serving model
│   └── Dockerfile        # Docker container
├── requirements.txt      # Dependencies
└── README.md             # Tài liệu dự án
```

## Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd ml-from-scratch
```

2. Tạo virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

3. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

## Cách sử dụng

1. **Thu thập dữ liệu**: Đặt dữ liệu gốc vào thư mục `data/raw/`
2. **Khám phá dữ liệu**: Chạy notebook `notebooks/EDA.ipynb`
3. **Baseline models**: Thử nghiệm với `notebooks/baseline_models.ipynb`
4. **Deep learning**: Phát triển model với `notebooks/deep_learning.ipynb`
5. **Training**: Sử dụng `src/train.py` để train model
6. **Đánh giá**: Dùng `src/evaluate.py` để đánh giá performance
7. **Deploy**: Chạy `app/app.py` để serve model

## Tác giả

[Tên của bạn]

## License

MIT License 