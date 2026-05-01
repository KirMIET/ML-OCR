import os
import pandas as pd
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ==========================================
# НАСТРОЙКИ
# ==========================================
SUB_1_PATH = './submissions/ensemble_weighted_and_gamma_v4.csv'  # Путь к первому CSV
SUB_2_PATH = './submissions/ensemble_weighted_and_gamma_v5.csv' # Путь ко второму CSV
IMG_DIR = './dataset/test/test'        # Папка с картинками теста

class OCRComparator:
    def __init__(self, sub1_path, sub2_path, img_dir):
        self.img_dir = img_dir
        
        # Загружаем данные
        df1 = pd.read_csv(sub1_path)
        df2 = pd.read_csv(sub2_path)
        
        # Объединяем по имени файла
        merged = df1.merge(df2, on='Filename', suffixes=('_Model1', '_Model2'))
        
        # Находим только те строки, где цены различаются
        self.diffs = merged[merged['Price_Model1'] != merged['Price_Model2']].to_dict('records')
        self.current_idx = 0
        
        if not self.diffs:
            print("🎉 Отличий не найдено! Файлы идентичны.")
            return

        print(f"🔍 Найдено отличий: {len(self.diffs)}")

        # Создаем GUI
        self.root = tk.Tk()
        self.root.title("OCR Comparison Tool")
        self.root.geometry("800x500")

        # Настройка интерфейса
        self.setup_ui()
        self.show_current()
        self.root.mainloop()

    def setup_ui(self):
        # Контейнер для изображения
        self.img_label = tk.Label(self.root)
        self.img_label.pack(pady=20)

        # Контейнер для текста
        text_frame = tk.Frame(self.root)
        text_frame.pack(pady=10)

        # Модель 1
        self.label1_title = tk.Label(text_frame, text="Model 1:", font=("Arial", 12, "bold"))
        self.label1_title.grid(row=0, column=0, padx=20)
        self.label1_val = tk.Label(text_frame, text="", font=("Arial", 14), fg="blue")
        self.label1_val.grid(row=1, column=0)

        # Модель 2
        self.label2_title = tk.Label(text_frame, text="Model 2:", font=("Arial", 12, "bold"))
        self.label2_title.grid(row=0, column=1, padx=20)
        self.label2_val = tk.Label(text_frame, text="", font=("Arial", 14), fg="red")
        self.label2_val.grid(row=1, column=1)

        # Имя файла
        self.fname_label = tk.Label(self.root, text="", font=("Arial", 10, "italic"))
        self.fname_label.pack()

        # Кнопки управления
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side="bottom", pady=30)

        self.btn_prev = ttk.Button(btn_frame, text="⬅ Назад", command=self.prev_diff)
        self.btn_prev.pack(side="left", padx=10)

        self.counter_label = tk.Label(btn_frame, text="")
        self.counter_label.pack(side="left", padx=20)

        self.btn_next = ttk.Button(btn_frame, text="Вперед ➡", command=self.next_diff)
        self.btn_next.pack(side="left", padx=10)

        # Горячие клавиши
        self.root.bind('<Left>', lambda e: self.prev_diff())
        self.root.bind('<Right>', lambda e: self.next_diff())

    def show_current(self):
        item = self.diffs[self.current_idx]
        img_path = os.path.join(self.img_dir, item['Filename'])
        
        # Отображение текста
        self.label1_val.config(text=item['Price_Model1'])
        self.label2_val.config(text=item['Price_Model2'])
        self.fname_label.config(text=f"File: {item['Filename']}")
        self.counter_label.config(text=f"{self.current_idx + 1} / {len(self.diffs)}")

        # Загрузка и ресайз изображения
        if os.path.exists(img_path):
            img = Image.open(img_path)
            # Масштабируем картинку, чтобы она влезала в окно
            img.thumbnail((600, 200))
            img_tk = ImageTk.PhotoImage(img)
            self.img_label.config(image=img_tk)
            self.img_label.image = img_tk
        else:
            self.img_label.config(image='', text="ИЗОБРАЖЕНИЕ НЕ НАЙДЕНО")

    def next_diff(self):
        if self.current_idx < len(self.diffs) - 1:
            self.current_idx += 1
            self.show_current()

    def prev_diff(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current()

if __name__ == "__main__":
    # Убедитесь, что пути указаны верно
    OCRComparator(SUB_1_PATH, SUB_2_PATH, IMG_DIR)