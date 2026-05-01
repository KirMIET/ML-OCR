import os
import random
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import albumentations as A

USE_GLARE = True               # Блики от ламп
USE_CHROMATIC = True           # Хроматические аберрации (разнос цвета)
USE_VIGNETTING = True          # Затемнение углов (виньетка)
USE_EXPOSURE_CONTRAST = True   # Случайная яркость и контрастность
USE_SENSOR_NOISE = True        # Цифровой шум (зернистость матрицы)
USE_ALBUMENTATIONS = True      # Геометрические искажения и блюр (Albumentations)
USE_GLOBAL_BLUR = True         # Общее размытие всего изображения
USE_RANDOM_LINES = True        # НОВЫЙ: Случайные темные линии (вертикальные/горизонтальные)

NUM_IMAGES = 10000
OUTPUT_DIR = "dataset/sint_data"
OUTPUT_CSV_DIR = "dataset"
CSV_FILENAME = "sint_data.csv"

MAX_WIDTH, MIN_WIDTH = 240, 40
MAX_HEIGHT, MIN_HEIGHT = 180, 25

FONT_PATH = "type.ttf" 
BASE_FONT_SIZE = 80        
BASE_SCALE = 2             

BG_COLORS = [
    (174, 184, 176), (255, 255, 255), (240, 240, 240), (200, 200, 200),
    (220, 220, 220), (220, 232, 106), (185, 172, 57), (213, 132, 69),
    (150, 195, 214), (227, 224, 131), (124, 75, 45), (211, 145, 113),
    (226, 206, 93), (212, 187, 105), (226, 226, 126)
]


def apply_random_lines(img):
    """Создает случайные темные полупрозрачные линии"""
    if not USE_RANDOM_LINES: return img
    if random.random() > 0.3:
        h, w = img.shape[:2]
        res = img.astype(np.float32)
        # Генерируем от 1 до 4 линий
        for _ in range(random.randint(1, 10)):
            alpha = random.uniform(0.1, 0.7) # Прозрачность линии (чем выше, тем темнее)
            is_vertical = random.choice([True, False])
            thickness = random.randint(2, 15) # Толщина линии
            
            if is_vertical:
                x = random.randint(0, w - thickness)
                # Затемняем вертикальную полосу
                res[:, x : x + thickness] *= (1 - alpha)
            else:
                y = random.randint(0, h - thickness)
                # Затемняем горизонтальную полосу
                res[y : y + thickness, :] *= (1 - alpha)
                
        return res.astype(np.uint8)
    return img

def apply_global_blur(img):
    if not USE_GLOBAL_BLUR: return img
    if random.random() > 0.3:
        k_size = random.choice([5, 7, 9, 11])
        if random.random() > 0.5:
            return cv2.GaussianBlur(img, (k_size, k_size), 0)
        else:
            return cv2.blur(img, (k_size, k_size))
    return img

def apply_chromatic_aberration(img):
    if not USE_CHROMATIC: return img
    if random.random() > 0.4:
        shift = random.randint(1, 2)
        b, g, r = cv2.split(img)
        r = np.roll(r, shift, axis=0)
        b = np.roll(b, -shift, axis=1)
        return cv2.merge([b, g, r])
    return img

def apply_vignetting(img):
    if not USE_VIGNETTING: return img
    if random.random() > 0.4:
        rows, cols = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/2)
        kernel_y = cv2.getGaussianKernel(rows, rows/2)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        img = img.astype(np.float32)
        for i in range(3):
            img[:,:,i] *= (mask * 0.4 + 0.6)
        return img.astype(np.uint8)
    return img

def apply_exposure_contrast(img):
    if not USE_EXPOSURE_CONTRAST: return img
    alpha = random.uniform(0.6, 1.7) 
    beta = random.randint(-25, 25)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def apply_sensor_noise(img):
    if not USE_SENSOR_NOISE: return img
    if random.random() > 0.5:
        noise = np.random.normal(0, random.uniform(3, 12), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def add_complex_glare(image_cv):
    if not USE_GLARE: return image_cv
    if random.random() > 0.4:
        h, w = image_cv.shape[:2]
        cx, cy = random.randint(0, w), random.randint(0, h)
        radius = random.randint(w // 3, w)
        glare_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(glare_mask, (cx, cy), radius, 255, -1)
        glare_mask = cv2.GaussianBlur(glare_mask, (101, 101), 0)
        for i in range(3):
            image_cv[:,:,i] = cv2.addWeighted(image_cv[:,:,i], 1.0, glare_mask, 0.2, 0).squeeze()
    return image_cv

# ==========================================
# --- ALBUMENTATIONS ---
# ==========================================
albumentations_transform = A.Compose([
    A.Affine(
        scale=(0.3, 1), 
        translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)}, 
        rotate=(-30, 30), 
        border_mode=cv2.BORDER_REPLICATE, 
        p=0.8
    ),
    A.Perspective(
        scale=(0.04, 0.1), 
        border_mode=cv2.BORDER_REPLICATE, 
        p=0.8
    ),
    A.RandomShadow(
        shadow_roi=(0, 0, 1, 1), 
        num_shadows_limit=(1, 2), 
        shadow_dimension=4, 
        p=0.4
    ),
    A.ElasticTransform(
        alpha=2, 
        sigma=20, 
        p=0.3, 
        border_mode=cv2.BORDER_REPLICATE
    ),
    A.OpticalDistortion(distort_limit=0.3, p=0.3),
    A.GaussNoise(std_range=(0.05, 0.2), p=0.5), 
    # A.OneOf([
    #     A.Blur(blur_limit=(11, 17), p=1.0),
    #     A.GaussianBlur(blur_limit=(11, 17), p=1.0),
    #     A.MedianBlur(blur_limit=(7, 13), p=1.0),
    #     A.MotionBlur(blur_limit=(15, 25), p=1.0),
    #     A.Downscale(scale_range=(0.2, 0.4), p=1.0), 
    # ], p=1),

    A.Blur(blur_limit=(11, 17), p=0.8),
    A.GaussianBlur(blur_limit=(11, 17), p=0.8),
    A.MedianBlur(blur_limit=(7, 13), p=0.8),
    A.MotionBlur(blur_limit=(15, 25), p=0.8),
    A.Downscale(scale_range=(0.2, 0.4), p=0.8), 
    A.Defocus(radius=(1, 4), alias_blur=(0.1, 0.5), p=0.8),

    A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.05, p=0.6),
    A.ImageCompression(quality_range=(20, 90), p=0.5)
])

# ==========================================
# --- ГЕНЕРАЦИЯ ---
# ==========================================

def generate_image(idx):
    bg_color = random.choice(BG_COLORS)
    
    if random.random() < 0.7:
        price_value = str(random.randint(1, 1000))
    else:
        price_value = str(random.randint(1001, 99999))
    
    num_digits = len(price_value)
    font_scale_factor = 1.0 if num_digits <= 3 else (0.8 if num_digits == 4 else 0.65)
    
    canvas_w = MAX_WIDTH * BASE_SCALE
    canvas_h = MAX_HEIGHT * BASE_SCALE
    
    try:
        font = ImageFont.truetype(FONT_PATH, int(BASE_FONT_SIZE * BASE_SCALE * font_scale_factor))
    except:
        font = ImageFont.load_default()

    img_pil = Image.new('RGB', (canvas_w, canvas_h), color=bg_color)
    draw = ImageDraw.Draw(img_pil)
    bbox = draw.textbbox((0, 0), price_value, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((canvas_w - tw) // 2, (canvas_h - th) // 2), price_value, fill=(0, 0, 0), font=font)
    
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # --- ПРИМЕНЕНИЕ ФИЛЬТРОВ ---
    img_cv = add_complex_glare(img_cv)
    img_cv = apply_chromatic_aberration(img_cv)
    img_cv = apply_vignetting(img_cv)
    
    # Геометрия и блюр (Albumentations)
    if USE_ALBUMENTATIONS:
        img_cv = albumentations_transform(image=img_cv)['image']
    
    # ПРИМЕНЕНИЕ ЛИНИЙ (после геометрии)
    img_cv = apply_random_lines(img_cv)
    
    # СТОРОННЕЕ РАЗМЫТИЕ
    img_cv = apply_global_blur(img_cv)
    
    img_cv = apply_exposure_contrast(img_cv)
    img_cv = apply_sensor_noise(img_cv)
    
    # Финальный ресайз
    final_w = random.randint(MIN_WIDTH, MAX_WIDTH)
    aspect_ratio = random.uniform(0.5, 0.75)
    final_h = int(final_w * aspect_ratio)
    final_h = max(MIN_HEIGHT, min(MAX_HEIGHT, final_h))
    
    img_final = cv2.resize(img_cv, (final_w, final_h), interpolation=cv2.INTER_LANCZOS4)
    
    filename = f"image_{idx:05d}.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img_final)
    
    return filename, price_value

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    csv_filepath = os.path.join(OUTPUT_CSV_DIR, CSV_FILENAME)
    
    with open(csv_filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Price"])
        for i in range(1, NUM_IMAGES + 1):
            fn, pr = generate_image(i)
            writer.writerow([fn, pr])
            if i % 100 == 0: print(f"Сгенерировано {i}...")
                
    print(f"Готово! Все параметры применены. Данные в {OUTPUT_DIR}")

if __name__ == "__main__":
    main()