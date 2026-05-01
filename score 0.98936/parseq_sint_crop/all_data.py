import os
import csv
import shutil

REAL_IMG_DIR = "dataset/train/train"
REAL_CSV_PATH = "dataset/train.csv"

SINT_IMG_DIR = "dataset/sint_data"
SINT_CSV_PATH = "dataset/sint_data.csv"

OUTPUT_IMG_DIR = "dataset/full_data"
OUTPUT_CSV_PATH = "dataset/full_labels.csv"

def process_dataset(source_img_dir, source_csv_path, output_dir, csv_writer, prefix):
    print(f"\nНачинаю обработку датасета: {source_img_dir} (Префикс: '{prefix}_')")
    
    if not os.path.exists(source_csv_path):
        print(f"ОШИБКА: Файл {source_csv_path} не найден!")
        return 0

    copied_count = 0
    missing_count = 0

    # Используем utf-8-sig для автоматического удаления невидимого символа BOM
    with open(source_csv_path, mode='r', encoding='utf-8-sig') as file:
        # Автоматически определяем разделитель (запятая или точка с запятой)
        sample = file.read(2048)
        file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel # Если не удалось определить, используем стандартный
            
        reader = csv.DictReader(file, dialect=dialect)
        
        # Получаем сырые заголовки
        raw_fieldnames = reader.fieldnames
        if not raw_fieldnames:
            print("ОШИБКА: Файл пуст!")
            return 0
            
        norm_fields = {str(f).strip().lower(): f for f in raw_fieldnames}
        
        # Ищем колонки независимо от регистра и пробелов
        if 'filename' not in norm_fields or 'price' not in norm_fields:
            print(f"ОШИБКА: Не найдены нужные заголовки.")
            print(f"Скрипт увидел следующие заголовки: {raw_fieldnames}")
            print(f"Разделитель, который определил скрипт: '{dialect.delimiter}'")
            return 0

        # Получаем точные (оригинальные) ключи для словаря
        actual_filename_key = norm_fields['filename']
        actual_price_key = norm_fields['price']

        for row in reader:
            original_filename = row[actual_filename_key].strip()
            price = row[actual_price_key].strip()
            
            src_img_path = os.path.join(source_img_dir, original_filename)
            new_filename = f"{prefix}_{original_filename}"
            dst_img_path = os.path.join(output_dir, new_filename)
            
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
                csv_writer.writerow([new_filename, price])
                copied_count += 1
                
                if copied_count % 1000 == 0:
                    print(f"  Скопировано {copied_count} файлов...")
            else:
                missing_count += 1

    print(f"Готово: {copied_count} скопировано. Не найдено картинок (битых ссылок): {missing_count}")
    return copied_count

def main():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    total_copied = 0
    
    # Итоговый файл всегда сохраняем со стандартной запятой
    with open(OUTPUT_CSV_PATH, mode='w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file, delimiter=',')
        writer.writerow(["Filename", "Price"])
        
        count1 = process_dataset(REAL_IMG_DIR, REAL_CSV_PATH, OUTPUT_IMG_DIR, writer, "real")
        total_copied += count1
        
        count2 = process_dataset(SINT_IMG_DIR, SINT_CSV_PATH, OUTPUT_IMG_DIR, writer, "sint")
        total_copied += count2
        
    print(f"\n=========================================")
    print(f"ОБЪЕДИНЕНИЕ ЗАВЕРШЕНО!")
    print(f"Всего изображений в новом датасете: {total_copied}")
    print(f"=========================================")

if __name__ == "__main__":
    main()