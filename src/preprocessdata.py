import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os
from pathlib import Path
import random

# --- PHƯƠNG PHÁP 1: KÉO DÃN / NÉN (STRETCH) ---
def resize_stretch(image_np, target_size):
    """
    Ép ảnh về kích thước mục tiêu, làm méo hình dạng.
    
    Args:
        image_np (np.array): Mảng NumPy của ảnh.
        target_size (tuple): (width, height) mục tiêu.
    
    Returns:
        np.array: Ảnh đã được thay đổi kích thước.
    """
    # Chuyển NumPy array sang ảnh PIL để dùng hàm resize có sẵn
    image_pil = Image.fromarray(image_np)
    
    # PIL.Image.resize yêu cầu kích thước là (width, height)
    # Thuật toán Image.Resampling.LANCZOS cho chất lượng tốt nhất khi resize
    resized_pil = image_pil.resize(target_size, Image.Resampling.LANCZOS)
    
    # Chuyển lại sang NumPy array để trả về
    return np.array(resized_pil)

# --- PHƯƠNG PHÁP 2: THAY ĐỔI KÍCH THƯỚC VÀ CHÈN ĐỆM (RESIZE & PAD) ---
def resize_pad(image_np, target_size):
    """
    Resize ảnh giữ nguyên tỉ lệ và chèn đệm (padding) để đạt kích thước mục tiêu.
    Đây là phương pháp được khuyên dùng nhất.
    
    Args:
        image_np (np.array): Mảng NumPy của ảnh.
        target_size (tuple): (width, height) mục tiêu.
    
    Returns:
        np.array: Ảnh đã được thay đổi kích thước và chèn đệm.
    """
    target_w, target_h = target_size
    orig_h, orig_w, _ = image_np.shape

    # 1. Tính toán tỉ lệ để thu nhỏ ảnh mà không làm méo
    ratio_w = target_w / orig_w
    ratio_h = target_h / orig_h
    scale_factor = min(ratio_w, ratio_h)
    
    # 2. Kích thước mới của ảnh sau khi thu nhỏ
    new_w = int(orig_w * scale_factor)
    new_h = int(orig_h * scale_factor)
    
    # 3. Thực hiện resize ảnh về kích thước mới
    image_pil = Image.fromarray(image_np)
    resized_pil = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_np = np.array(resized_pil)
    
    # 4. Tạo một khung (canvas) màu đen có kích thước mục tiêu
    # Kiểu dữ liệu uint8 là cho các giá trị pixel từ 0-255
    padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # 5. Tính toán vị trí để đặt ảnh đã resize vào giữa khung
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    
    # 6. Dùng slicing của NumPy để đặt ảnh vào khung
    padded_image[top:top + new_h, left:left + new_w] = resized_np
    
    return padded_image

# --- PHƯƠG PHÁP 3: THAY ĐỔI KÍCH THƯỚC VÀ CẮT CÚP (RESIZE & CROP) ---
def resize_crop(image_np, target_size):
    """
    Resize ảnh giữ nguyên tỉ lệ sao cho nó lấp đầy kích thước mục tiêu,
    sau đó cắt phần thừa ở trung tâm.
    
    Args:
        image_np (np.array): Mảng NumPy của ảnh.
        target_size (tuple): (width, height) mục tiêu.
    
    Returns:
        np.array: Ảnh đã được thay đổi kích thước và cắt cúp.
    """
    target_w, target_h = target_size
    orig_h, orig_w, _ = image_np.shape

    # 1. Tính toán tỉ lệ để resize ảnh lớn hơn hoặc bằng kích thước mục tiêu
    ratio_w = target_w / orig_w
    ratio_h = target_h / orig_h
    scale_factor = max(ratio_w, ratio_h)

    # 2. Kích thước mới của ảnh sau khi resize
    new_w = int(orig_w * scale_factor)
    new_h = int(orig_h * scale_factor)
    
    # 3. Thực hiện resize
    image_pil = Image.fromarray(image_np)
    resized_pil = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_np = np.array(resized_pil)
    
    # 4. Tính toán vị trí để cắt từ trung tâm
    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    
    # 5. Dùng slicing của NumPy để cắt ảnh
    cropped_image = resized_np[top:top + target_h, left:left + target_w]
    
    return cropped_image

# --- PHƯƠNG PHÁP DATA AUGMENTATION ---
def rotate_image(image_np, angle):
    """
    Xoay ảnh theo góc cho trước.
    
    Args:
        image_np (np.array): Mảng NumPy của ảnh.
        angle (float): Góc xoay (độ).
    
    Returns:
        np.array: Ảnh đã được xoay.
    """
    image_pil = Image.fromarray(image_np)
    rotated_pil = image_pil.rotate(angle, expand=False, fillcolor=(0, 0, 0))
    return np.array(rotated_pil)

def flip_image(image_np, mode='horizontal'):
    """
    Lật ảnh theo chiều ngang hoặc dọc.
    
    Args:
        image_np (np.array): Mảng NumPy của ảnh.
        mode (str): 'horizontal' hoặc 'vertical'.
    
    Returns:
        np.array: Ảnh đã được lật.
    """
    image_pil = Image.fromarray(image_np)
    if mode == 'horizontal':
        flipped_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
    elif mode == 'vertical':
        flipped_pil = image_pil.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        raise ValueError("Mode phải là 'horizontal' hoặc 'vertical'")
    return np.array(flipped_pil)

def adjust_brightness(image_np, factor):
    """
    Thay đổi độ sáng của ảnh.
    
    Args:
        image_np (np.array): Mảng NumPy của ảnh.
        factor (float): Hệ số độ sáng (1.0 = không đổi, >1 = sáng hơn, <1 = tối hơn).
    
    Returns:
        np.array: Ảnh đã được điều chỉnh độ sáng.
    """
    image_pil = Image.fromarray(image_np)
    enhancer = ImageEnhance.Brightness(image_pil)
    enhanced_pil = enhancer.enhance(factor)
    return np.array(enhanced_pil)

def adjust_contrast(image_np, factor):
    """
    Thay đổi độ tương phản của ảnh.
    
    Args:
        image_np (np.array): Mảng NumPy của ảnh.
        factor (float): Hệ số độ tương phản (1.0 = không đổi).
    
    Returns:
        np.array: Ảnh đã được điều chỉnh độ tương phản.
    """
    image_pil = Image.fromarray(image_np)
    enhancer = ImageEnhance.Contrast(image_pil)
    enhanced_pil = enhancer.enhance(factor)
    return np.array(enhanced_pil)

def add_noise(image_np, noise_factor=0.1):
    """
    Thêm nhiễu ngẫu nhiên vào ảnh.
    
    Args:
        image_np (np.array): Mảng NumPy của ảnh.
        noise_factor (float): Mức độ nhiễu (0.0 - 1.0).
    
    Returns:
        np.array: Ảnh đã được thêm nhiễu.
    """
    noise = np.random.normal(0, noise_factor * 255, image_np.shape)
    noisy_image = image_np.astype(np.float64) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def zoom_image(image_np, zoom_factor):
    """
    Zoom ảnh và cắt về kích thước gốc.
    
    Args:
        image_np (np.array): Mảng NumPy của ảnh.
        zoom_factor (float): Hệ số zoom (>1 = zoom in, <1 = zoom out).
    
    Returns:
        np.array: Ảnh đã được zoom.
    """
    h, w, _ = image_np.shape
    image_pil = Image.fromarray(image_np)
    
    if zoom_factor > 1:
        # Zoom in: resize lớn hơn rồi cắt giữa
        new_size = (int(w * zoom_factor), int(h * zoom_factor))
        resized_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)
        resized_np = np.array(resized_pil)
        
        # Cắt từ giữa
        new_h, new_w, _ = resized_np.shape
        top = (new_h - h) // 2
        left = (new_w - w) // 2
        zoomed_image = resized_np[top:top + h, left:left + w]
    else:
        # Zoom out: resize nhỏ hơn rồi thêm padding
        new_size = (int(w * zoom_factor), int(h * zoom_factor))
        resized_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)
        resized_np = np.array(resized_pil)
        
        # Thêm padding
        zoomed_image = np.zeros((h, w, 3), dtype=np.uint8)
        new_h, new_w, _ = resized_np.shape
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        zoomed_image[top:top + new_h, left:left + new_w] = resized_np
    
    return zoomed_image

def apply_random_augmentation(image_np):
    """
    Áp dụng ngẫu nhiên một hoặc nhiều phép augmentation.
    
    Args:
        image_np (np.array): Mảng NumPy của ảnh.
    
    Returns:
        np.array: Ảnh đã được augment.
    """
    augmented = image_np.copy()
    
    # Danh sách các augmentation với xác suất
    augmentations = [
        (lambda img: rotate_image(img, random.uniform(-15, 15)), 0.5),  # Xoay ±15°
        (lambda img: flip_image(img, 'horizontal'), 0.3),  # Lật ngang
        (lambda img: adjust_brightness(img, random.uniform(0.8, 1.2)), 0.4),  # Độ sáng
        (lambda img: adjust_contrast(img, random.uniform(0.8, 1.2)), 0.4),  # Độ tương phản
        (lambda img: add_noise(img, random.uniform(0.01, 0.05)), 0.2),  # Nhiễu nhẹ
        (lambda img: zoom_image(img, random.uniform(0.9, 1.1)), 0.3),  # Zoom nhẹ
    ]
    
    # Áp dụng từng augmentation với xác suất
    for augment_func, probability in augmentations:
        if random.random() < probability:
            try:
                augmented = augment_func(augmented)
            except Exception as e:
                print(f"    Lỗi khi áp dụng augmentation: {e}")
                continue
    
    return augmented

# --- HÀM XỬ LÝ TOÀN BỘ DATASET ---
def process_dataset(raw_dir="data/raw", processed_dir="data/processed", target_size=(224, 224), 
                   method="pad", enable_augmentation=False, augment_factor=2):
    """
    Xử lý toàn bộ dataset và lưu vào thư mục processed.
    
    Args:
        raw_dir (str): Đường dẫn thư mục dữ liệu gốc
        processed_dir (str): Đường dẫn thư mục dữ liệu đã xử lý
        target_size (tuple): Kích thước mục tiêu (width, height)
        method (str): Phương pháp xử lý ("stretch", "pad", "crop")
        enable_augmentation (bool): Có áp dụng data augmentation không
        augment_factor (int): Số lượng ảnh augmented tạo ra cho mỗi ảnh gốc
    """
    # Tạo thư mục processed nếu chưa có
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    
    # Định nghĩa các phương pháp xử lý
    methods = {
        "stretch": resize_stretch,
        "pad": resize_pad,
        "crop": resize_crop
    }
    
    if method not in methods:
        raise ValueError(f"Phương pháp '{method}' không hợp lệ. Chọn từ: {list(methods.keys())}")
    
    process_func = methods[method]
    
    # Lấy danh sách các thư mục con (classes)
    raw_path = Path(raw_dir)
    class_dirs = [d for d in raw_path.iterdir() if d.is_dir()]
    
    total_processed = 0
    total_augmented = 0
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Bỏ qua thư mục processed nếu có
        if class_name == "processed":
            continue
            
        print(f"Đang xử lý class: {class_name}")
        
        # Tạo thư mục cho class trong processed
        processed_class_dir = Path(processed_dir) / class_name
        processed_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Lấy danh sách các file ảnh
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_files.extend(class_dir.glob(ext))
            image_files.extend(class_dir.glob(ext.upper()))
        
        print(f"  Tìm thấy {len(image_files)} ảnh")
        if enable_augmentation:
            print(f"  Sẽ tạo {augment_factor} ảnh augmented cho mỗi ảnh gốc")
        
        # Xử lý từng ảnh
        for img_idx, img_file in enumerate(image_files):
            try:
                # Đọc ảnh
                image_np = np.array(Image.open(img_file))
                
                # Kiểm tra xem ảnh có 3 channels không (RGB)
                if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                    # Xử lý ảnh gốc
                    processed_image = process_func(image_np, target_size)
                    
                    # Lưu ảnh gốc đã xử lý
                    output_path = processed_class_dir / f"{img_file.stem}.jpg"
                    Image.fromarray(processed_image).save(output_path, "JPEG", quality=95)
                    total_processed += 1
                    
                    # Tạo ảnh augmented nếu được bật
                    if enable_augmentation:
                        for aug_idx in range(augment_factor):
                            # Áp dụng augmentation trên ảnh gốc
                            augmented_image = apply_random_augmentation(image_np)
                            
                            # Resize ảnh augmented
                            processed_augmented = process_func(augmented_image, target_size)
                            
                            # Lưu ảnh augmented
                            aug_output_path = processed_class_dir / f"{img_file.stem}_aug_{aug_idx+1}.jpg"
                            Image.fromarray(processed_augmented).save(aug_output_path, "JPEG", quality=95)
                            total_augmented += 1
                    
                else:
                    print(f"  Bỏ qua {img_file.name} (không phải RGB)")
                    
            except Exception as e:
                print(f"  Lỗi khi xử lý {img_file.name}: {e}")
        
        print(f"  Hoàn thành xử lý {class_name}")
    
    print(f"\nTổng cộng đã xử lý:")
    print(f"  - Ảnh gốc: {total_processed}")
    if enable_augmentation:
        print(f"  - Ảnh augmented: {total_augmented}")
        print(f"  - Tổng cộng: {total_processed + total_augmented}")
    print(f"Kết quả được lưu tại: {processed_dir}")

# --- HÀM MAIN ĐỂ CHẠY THỬ NGHIỆM ---
if __name__ == '__main__':
    print("=== XỬ LÝ DỮ LIỆU ẢNH ===")
    print("Phương pháp: Resize và Pad (giữ nguyên tỉ lệ)")
    print("Kích thước mục tiêu: 224x224")
    print()
    
    # Xử lý toàn bộ dataset
    process_dataset(
        raw_dir="data/raw",
        processed_dir="data/processed", 
        target_size=(224, 224),
        method="pad",  # Sử dụng phương pháp pad (khuyên dùng)
        enable_augmentation=True, # Bật augmentation
        augment_factor=3 # Tạo 3 ảnh augmented cho mỗi ảnh gốc
    )
    
    print("\n=== HOÀN THÀNH ===")