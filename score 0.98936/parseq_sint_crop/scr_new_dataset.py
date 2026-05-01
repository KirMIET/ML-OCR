import os
import random
import csv
import cv2
import numpy as np
import uuid
from tqdm import tqdm

DIGITS_DIR = r"dataset\left_label"
DIGITS_CSV = r"dataset\left_label\labels.csv"
GARBAGE_DIR = r"dataset\left_label"
OUTPUT_DIR = r"dataset\crop_train"
OUTPUT_CSV = r"dataset\crop_train.csv"

GEN_CONFIG = {
    2: 900,
    3: 1100,
    4: 1500,
    5: 500
}
GARBAGE_PROBABILITY = 0.1 # 10% шанс

def imread_unicode(path):
    """Безопасное чтение изображений с русскими символами в путях Windows"""
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)

def imwrite_unicode(path, img):
    """Безопасное сохранение изображений с русскими символами в путях"""
    is_success, im_buf_arr = cv2.imencode(".jpg", img)
    im_buf_arr.tofile(path)

def load_data():
    """Загружает пути к цифрам и мусору"""
    digit_data = []
    
    # Читаем разметку цифр
    if not os.path.exists(DIGITS_CSV):
        raise FileNotFoundError(f"Файл {DIGITS_CSV} не найден!")
        
    with open(DIGITS_CSV, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                filename, label = row[0].strip(), row[1].strip()
                # Пропускаем строку заголовка, если она есть
                if label.lower() == 'price': 
                    continue
                full_path = os.path.join(DIGITS_DIR, filename)
                if os.path.exists(full_path):
                    digit_data.append((full_path, label))
    
    # Читаем мусор
    garbage_files = []
    if os.path.exists(GARBAGE_DIR):
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')
        garbage_files = [
            os.path.join(GARBAGE_DIR, f) for f in os.listdir(GARBAGE_DIR) 
            if f.lower().endswith(valid_ext)
        ]
        
    return digit_data, garbage_files

def resize_to_height(img, target_height):
    """Меняет размер изображения под нужную высоту, сохраняя пропорции"""
    h, w = img.shape[:2]
    if h == target_height:
        return img
    
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LANCZOS4)

def blend_images(img1, img2):
    """Соединяет 2 картинки через 1 пиксель и применяет медианный фильтр к шву"""
    h = img1.shape[0]
    
    # Создаем 1 пиксель-разделитель (берем среднее значение цветов краев)
    gap = np.zeros((h, 1, 3), dtype=np.uint8)
    gap[:, 0] = (img1[:, -1] / 2 + img2[:, 0] / 2).astype(np.uint8)
    
    # Склеиваем
    combined = np.hstack((img1, gap, img2))
    
    # Применяем медианный фильтр ТОЛЬКО к шву (полоска шириной 3 пикселя: край_картинки1 + gap + край_картинки2)
    seam_x = img1.shape[1] # Индекс пикселя gap в склеенной картинке
    roi = combined[:, seam_x-1 : seam_x+2] # Вырезаем область шва
    blurred_roi = cv2.medianBlur(roi, 3) # Блюрим
    combined[:, seam_x-1 : seam_x+2] = blurred_roi # Вставляем обратно
    
    return combined

def generate_price_tag(digit_data, garbage_files, length):
    """Генерирует 1 ценник заданной длины"""
    # 1. Выбираем случайные цифры
    chosen_items = random.choices(digit_data, k=length)
    
    images = []
    label_str = ""
    max_height = 0
    
    # 2. Загружаем картинки и ищем максимальную высоту
    for path, label in chosen_items:
        img = imread_unicode(path)
        images.append(img)
        label_str += str(label)
        if img.shape[0] > max_height:
            max_height = img.shape[0]
            
    # 3. Подстраиваем размеры всех цифр под max_height
    images = [resize_to_height(img, max_height) for img in images]
    
    # 4. Соединяем цифры с блюром шва
    final_img = images[0]
    for i in range(1, len(images)):
        final_img = blend_images(final_img, images[i])
        
    # 5. Прикрепляем мусор (10% шанс)
    if garbage_files and random.random() < GARBAGE_PROBABILITY:
        garbage_path = random.choice(garbage_files)
        garbage_img = imread_unicode(garbage_path)
        garbage_img = resize_to_height(garbage_img, max_height)
        
        # Мусор просто приклеиваем слева (без сложного шва, так как мусор бывает резким)
        final_img = np.hstack((garbage_img, final_img))
        
    return final_img, label_str

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Загрузка исходных данных...")
    digit_data, garbage_files = load_data()
    print(f"Загружено цифр: {len(digit_data)}")
    print(f"Загружено мусора: {len(garbage_files)}")
    
    if not digit_data:
        print("Ошибка: Нет данных для генерации!")
        return

    # Подготавливаем файл разметки
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Price"])

        # Начинаем генерацию
        for length, count in GEN_CONFIG.items():
            print(f"\nГенерация ценников из {length} цифр (Всего: {count} шт.)...")
            
            for _ in tqdm(range(count)):
                img, label = generate_price_tag(digit_data, garbage_files, length)
                
                # Уникальное имя (чтобы не было перезаписи, если метки совпадают)
                uid = uuid.uuid4().hex[:8]
                filename = f"price_{label}_{uid}.jpg"
                save_path = os.path.join(OUTPUT_DIR, filename)
                
                # Сохраняем картинку и пишем в CSV
                imwrite_unicode(save_path, img)
                writer.writerow([filename, label])

    print("\nГотово! Датасет успешно сгенерирован.")
    print(f"Папка: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()