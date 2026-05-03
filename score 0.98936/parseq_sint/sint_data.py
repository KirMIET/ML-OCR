import os
import random
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import albumentations as A

NUM_IMAGES = 30000
OUTPUT_DIR = "dataset/only_sint_data"

OUTPUT_CSV_DIR = "dataset"
CSV_FILENAME = "only_sint_data.csv"

MAX_WIDTH, MIN_WIDTH = 240, 60
MAX_HEIGHT, MIN_HEIGHT = 180, 40

FONT_PATH = "type.ttf" 
BASE_FONT_SIZE = 80        
BASE_SCALE = 2             

BG_COLORS = [
    (174, 184, 176),  # Светло-серый
    (255, 255, 255),  # Белый
    (240, 240, 240),  # Очень светло-серый
    (200, 200, 200),  # Средне-серый
    (220, 220, 220),   # Светло-серый
    (220, 232, 106),  # Светло-желтый
    (185, 172, 57),   # Бледно-желтый
    (213, 132, 69),   # Бледно-оранжевый
    (150, 195, 214),  # Бледно-голубой
    (227, 224, 131),  # Бледно-зеленый
    (124, 75, 45),     # Бледно-коричневый
    (211, 145, 113), # Бледно-розовый
    (226, 206, 93),   # Бледно-лимонный
    (212, 187, 105),  # Бледно-горчичный
    (226, 226, 126)
]

transform = A.Compose([
    # Affine: mode -> border_mode
    A.Affine(
        scale=(0.3, 1), 
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, 
        rotate=(-30, 30), 
        border_mode=cv2.BORDER_REPLICATE, 
        p=0.8
    ),
    
    # Perspective: mode -> border_mode
    A.Perspective(
        scale=(0.04, 0.1), 
        border_mode=cv2.BORDER_REPLICATE, 
        p=0.8
    ),
    
    A.RandomShadow(
        shadow_roi=(0, 0, 1, 1), 
        num_shadows_limit=(1, 3), 
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
    
    A.GaussNoise(std_range=(0.1, 0.5), p=0.5), 
    
    A.OneOf([
        A.Blur(blur_limit=(11, 17), p=1.0),
        
        A.GaussianBlur(blur_limit=(11, 17), p=1.0),
        
        A.MedianBlur(blur_limit=(7, 13), p=1.0),
        
        A.MotionBlur(blur_limit=(15, 25), p=1.0),
        
        A.Downscale(scale_range=(0.2, 0.4), p=1.0), 
    ], p=0.8),
    
    A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.05, p=0.6),
    
    A.ImageCompression(quality_range=(20, 90), p=0.5)
])

def add_complex_glare(image_cv):
    """Имитация сильного точечного блика от лампы"""
    if random.random() > 0.6:
        h, w = image_cv.shape[:2]
        overlay = image_cv.copy()
        cx, cy = random.randint(0, w), random.randint(0, h)
        radius = random.randint(w // 4, w // 1)
        
        glare_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(glare_mask, (cx, cy), radius, 255, -1)
        glare_mask = cv2.GaussianBlur(glare_mask, (101, 101), 0)
        
        for i in range(3):
            overlay[:,:,i] = np.where(glare_mask > 0, 
                                     cv2.addWeighted(overlay[:,:,i], 0.7, glare_mask, 0.3, 0).squeeze(), 
                                     overlay[:,:,i])
        return overlay
    return image_cv

def generate_image(idx):
    bg_color = random.choice(BG_COLORS)
    
    # Логика вероятности цен
    if random.random() < 0.7:
        price_value = str(random.randint(1, 1000))
    else:
        price_value = str(random.randint(1001, 99999))
    
    # 1. АДАПТИВНЫЙ ШРИФТ: уменьшаем кегль, если цифр много
    # Если 5 знаков — шрифт меньше, если 1 знак — шрифт крупнее
    num_digits = len(price_value)
    font_scale_factor = 1.0 if num_digits <= 3 else (0.8 if num_digits == 4 else 0.65)
    
    canvas_w = MAX_WIDTH * BASE_SCALE
    canvas_h = MAX_HEIGHT * BASE_SCALE
    
    # Создаем временный шрифт
    try:
        current_size = int(BASE_FONT_SIZE * BASE_SCALE * font_scale_factor)
        font = ImageFont.truetype(FONT_PATH, current_size)
    except IOError:
        font = ImageFont.load_default()

    # 2. РАСЧЕТ ГРАНИЦ
    img_pil = Image.new('RGB', (canvas_w, canvas_h), color=bg_color)
    draw = ImageDraw.Draw(img_pil)
    
    # Получаем точные размеры текста
    bbox = draw.textbbox((0, 0), price_value, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Центрируем СТРОГО посередине (без начального рандома)
    # Это дает запас места для последующих аугментаций (Affine/Perspective)
    x = (canvas_w - tw) // 2
    y = (canvas_h - th) // 2
    
    draw.text((x, y), price_value, fill=(0, 0, 0), font=font)
    
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_cv = add_complex_glare(img_cv)
    
    # 3. КОНТРОЛИРУЕМЫЕ АУГМЕНТАЦИИ
    # В transform (Affine) ограничь translate_percent до 0.05, чтобы сдвиг был небольшим
    augmented = transform(image=img_cv)
    img_cv = augmented['image']
    
    # Стабилизация размеров
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
                
    print(f"Готово! Данные в {OUTPUT_DIR}")

if __name__ == "__main__":
    main()