import os
import random
import csv
import time
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

# --- НАСТРОЙКИ ПУТЕЙ ---
INPUT_DIR = r"dataset\train\train"
TRAIN_CSV = r"dataset\train.csv" # Путь к оригинальной разметке
OUTPUT_DIR = r"dataset\left_label"
CSV_FILE = os.path.join(OUTPUT_DIR, "labels.csv") # Файл, куда сохраняются наши кропы

# Максимальный размер окна для отображения
MAX_WIDTH = 1200
MAX_HEIGHT = 800
# Желаемый размер по бОльшей стороне для мелких картинок
TARGET_SIZE = 500 

class ImageCropperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Digit Cropper")
        
        # Переменные для логики
        self.image_files = []
        self.ground_truth = {} # Словарь для хранения разметки из train.csv
        self.current_image_path = ""
        self.original_image = None
        self.display_image = None
        self.scale_factor = 1.0
        
        self.rect = None
        self.start_x = None
        self.start_y = None

        self._setup_directories()
        self._load_ground_truth() # Загружаем разметку до загрузки картинок
        self._load_image_list()
        self._setup_gui()
        
        self.load_random_image()

    def _setup_directories(self):
        """Создает папку вывода и CSV файл с заголовками, если их нет."""
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Filename", "Price"]) 

    def _load_ground_truth(self):
        """Загружает оригинальную разметку из train.csv в словарь."""
        if not os.path.exists(TRAIN_CSV):
            messagebox.showwarning("Внимание", f"Файл разметки не найден:\n{TRAIN_CSV}\nПодсказки выводиться не будут.")
            return

        try:
            with open(TRAIN_CSV, mode='r', encoding='utf-8') as f:
                # Определяем разделитель (запятая или точка с запятой)
                sample = f.read(1024)
                f.seek(0)
                dialect = csv.Sniffer().sniff(sample)
                reader = csv.reader(f, dialect)
                
                header = next(reader, None) # Пропускаем заголовок
                for row in reader:
                    if len(row) >= 2:
                        filename = row[0].strip()
                        label = row[1].strip()
                        self.ground_truth[filename] = label
        except Exception as e:
            messagebox.showerror("Ошибка чтения CSV", f"Не удалось прочитать {TRAIN_CSV}\n{e}")

    def _load_image_list(self):
        """Загружает список изображений из входной директории."""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        if not os.path.exists(INPUT_DIR):
            messagebox.showerror("Ошибка", f"Папка не найдена:\n{INPUT_DIR}")
            self.root.destroy()
            return

        self.image_files = [
            os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) 
            if f.lower().endswith(valid_extensions)
        ]
        
        if not self.image_files:
            messagebox.showerror("Ошибка", "В папке нет изображений!")
            self.root.destroy()

    def _setup_gui(self):
        """Создает элементы интерфейса."""
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.btn_next = tk.Button(btn_frame, text="Следующее фото", command=self.load_random_image, font=("Arial", 12))
        self.btn_next.pack(side=tk.LEFT, padx=10)

        self.btn_exit = tk.Button(btn_frame, text="Выход", command=self.root.destroy, font=("Arial", 12), fg="red")
        self.btn_exit.pack(side=tk.RIGHT, padx=10)

        self.label_info = tk.Label(btn_frame, text="Загрузка...", font=("Arial", 12, "bold"))
        self.label_info.pack(expand=True)

        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def load_random_image(self):
        """Загружает случайное фото из списка и УМНО масштабирует под экран."""
        if not self.image_files:
            return

        self.canvas.delete("all")
        self.current_image_path = random.choice(self.image_files)
        self.original_image = Image.open(self.current_image_path)

        orig_w, orig_h = self.original_image.size
        
        # --- ЛОГИКА МАСШТАБИРОВАНИЯ ---
        if max(orig_w, orig_h) < TARGET_SIZE:
            self.scale_factor = TARGET_SIZE / max(orig_w, orig_h)
        else:
            scale_w = MAX_WIDTH / orig_w
            scale_h = MAX_HEIGHT / orig_h
            self.scale_factor = min(1.0, scale_w, scale_h)

        new_w = int(orig_w * self.scale_factor)
        new_h = int(orig_h * self.scale_factor)

        if self.scale_factor > 1.0:
            resample_method = Image.Resampling.NEAREST
        else:
            resample_method = Image.Resampling.LANCZOS

        resized_img = self.original_image.resize((new_w, new_h), resample_method)
        self.display_image = ImageTk.PhotoImage(resized_img)

        # Обновляем холст
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.create_image(0, 0, anchor="nw", image=self.display_image)
        
        # --- ПОИСК РАЗМЕТКИ ДЛЯ ВЫВОДА НА ЭКРАН ---
        filename = os.path.basename(self.current_image_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Пробуем найти разметку: сначала по полному имени, потом без расширения
        true_label = self.ground_truth.get(filename, self.ground_truth.get(name_without_ext, "Неизвестно"))
        
        self.label_info.config(
            text=f"Файл: {filename}  |  На ценнике: {true_label}  |  Зум: x{self.scale_factor:.1f}",
            fg="blue"
        )

    def on_button_press(self, event):
        """Начало рисования рамки."""
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, 1, 1, outline='red', width=2)

    def on_move_press(self, event):
        """Растягивание рамки."""
        curX, curY = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        """Конец выделения, вызов окна сохранения."""
        end_x, end_y = (event.x, event.y)
        
        if abs(end_x - self.start_x) < 5 or abs(end_y - self.start_y) < 5:
            self.canvas.delete(self.rect)
            return

        digit = simpledialog.askstring("Ввод", "Какая цифра выделена? (оставь пустым для отмены)", parent=self.root)
        
        if digit is not None and digit.strip() != "":
            self.save_crop(self.start_x, self.start_y, end_x, end_y, digit.strip())
        
        self.canvas.delete(self.rect)

    def save_crop(self, x1, y1, x2, y2, digit):
        """Вырезает кусок из ОРИГИНАЛЬНОГО изображения и сохраняет."""
        orig_x1 = int(min(x1, x2) / self.scale_factor)
        orig_y1 = int(min(y1, y2) / self.scale_factor)
        orig_x2 = int(max(x1, x2) / self.scale_factor)
        orig_y2 = int(max(y1, y2) / self.scale_factor)

        cropped_img = self.original_image.crop((orig_x1, orig_y1, orig_x2, orig_y2))

        orig_filename = os.path.basename(self.current_image_path)
        orig_name_only, orig_ext = os.path.splitext(orig_filename)
        timestamp = int(time.time() * 1000)
        
        filename = f"{orig_name_only}_{digit}_{timestamp}{orig_ext}"
        save_path = os.path.join(OUTPUT_DIR, filename)

        cropped_img.save(save_path)

        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([filename, digit])

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCropperApp(root)
    root.mainloop()