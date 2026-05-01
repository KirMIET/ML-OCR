import os
import re
import cv2
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np


PARSE_PATH = r'./parseq'
TEST_DIR = './dataset/test/test'
OUTPUT_CSV = './submissions/ensemble_weighted_and_gamma_v4.csv'

MODELS_CONFIG = {
    './outputs/saved_models/parseq_swa.pth': 1.0, 
    './outputs/saved_models/parseq_sint.pth':    0.9,
    './outputs/saved_models/parseq_sint_crop.pth': 1.1   
}

USE_AUTO_CROP = False        
USE_GAMMA_CORRECTION = True  
GAMMA_VALUE = 5.0            

SAVE_VISUALIZATION = False    
VIS_DIR = './debug_final'
VIS_LIMIT = 50

CUSTOM_VOCAB = "0123456789"
IMG_W, IMG_H = 192, 64
REFINE_STEPS = 7           
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append(PARSE_PATH)
from strhub.models.utils import create_model


def apply_gamma(image, gamma=1.0):
    """ Коррекция яркости: делает темные участки насыщеннее """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def auto_crop(img):
    """ Обрезка лишнего фона по контурам текста """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is None: return img
    x, y, w, h = cv2.boundingRect(coords)
    mx, my = int(w * 0.05), int(h * 0.05) # 5% отступа
    x, y = max(0, x - mx), max(0, y - my)
    w, h = min(img.shape[1] - x, w + 2 * mx), min(img.shape[0] - y, h + 2 * my)
    return img[y:y+h, x:x+w]

def smart_resize_and_pad(image, target_w, target_h):
    """ Ресайз Lanczos4 + заливка полей медианным цветом края """
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    res = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    
    # Собираем все пиксели границ для вычисления фона
    edge_pixels = np.concatenate([
        res[0, :].reshape(-1, 3), res[-1, :].reshape(-1, 3), 
        res[:, 0].reshape(-1, 3), res[:, -1].reshape(-1, 3)
    ])
    bg_color = np.median(edge_pixels, axis=0).astype(np.uint8)
    
    padded = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    y_off, x_off = (target_h - nh) // 2, (target_w - nw) // 2
    padded[y_off:y_off+nh, x_off:x_off+nw] = res
    return padded

def generate_tta_images(img):
    """ Генерация вариантов изображения для теста (TTA) """
    variants = [img] # Оригинал
    
    # 1. CLAHE (Локальный контраст)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    th, tw = max(2, min(8, l.shape[0]//4)), max(2, min(8, l.shape[1]//4))
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(th, tw))
    variants.append(cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB))
    
    if USE_GAMMA_CORRECTION:
        # 2. Глубокое затемнение (Gamma)
        variants.append(apply_gamma(img, gamma=GAMMA_VALUE))
        # 3. Гамма + Резкость
        dark = apply_gamma(img, gamma=1.3)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        variants.append(cv2.filter2D(dark, -1, kernel))
    else:
        # Если гамма выключена, используем стандартные фильтры
        variants.append(cv2.convertScaleAbs(img, alpha=1.2, beta=10))
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        variants.append(cv2.filter2D(img, -1, kernel))
        
    return variants


class PARSeqInfer:
    def __init__(self, weights_path, weight_coeff):
        self.coeff = weight_coeff
        self.model = create_model('parseq', pretrained=False, img_size=[IMG_H, IMG_W])
        new_tokenizer = type(self.model.tokenizer)(CUSTOM_VOCAB)
        core = self.model.model if hasattr(self.model, 'model') else self.model
        core.text_embed.embedding = nn.Embedding(len(new_tokenizer), core.text_embed.embedding.embedding_dim)
        core.head = nn.Linear(core.head.in_features, len(new_tokenizer))
        
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        self.model.to(DEVICE).eval()
        
        self.model.tokenizer = core.tokenizer = new_tokenizer
        core.bos_id, core.eos_id, core.pad_id = new_tokenizer.bos_id, new_tokenizer.eos_id, new_tokenizer.pad_id
        core.refine_steps = REFINE_STEPS

    @torch.no_grad()
    def predict(self, tensors):
        batch = torch.stack(tensors).to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=='cuda'), dtype=torch.float16):
            logits = self.model(batch)
            if isinstance(logits, (list, tuple)): logits = logits[-1]
            preds, char_probs = self.model.tokenizer.decode(logits.softmax(-1))
            # Сразу применяем вес модели к уверенности
            confs = [cp.mean().item() * self.coeff if (cp is not None and len(cp)>0) else 0.0 for cp in char_probs]
        return preds, confs


def clean_text(text):
    """ Только цифры, исправляем частые ошибки OCR """
    text = text.upper().replace('O', '0').replace('О', '0')
    return re.sub(r'[^\d]', '', text).strip()

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    if SAVE_VISUALIZATION: os.makedirs(VIS_DIR, exist_ok=True)

    print(f"Инициализация ансамбля ({len(MODELS_CONFIG)} моделей)...")
    ensemble = [PARSeqInfer(p, c) for p, c in MODELS_CONFIG.items()]
    
    test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    results = []

    for idx, name in enumerate(tqdm(test_files, desc="OCR Ensemble")):
        path = os.path.join(TEST_DIR, name)
        img_raw = cv2.imread(path)
        if img_raw is None: continue
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        
        # Обработка
        work_img = auto_crop(img_rgb) if USE_AUTO_CROP else img_rgb
        tta_variants = generate_tta_images(work_img)
        processed = [smart_resize_and_pad(v, IMG_W, IMG_H) for v in tta_variants]
        tensors = [torch.from_numpy(i).permute(2, 0, 1).float().sub_(127.5).div_(127.5) for i in processed]
        
        if SAVE_VISUALIZATION and idx < VIS_LIMIT:
            vis_row = cv2.hconcat([cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in processed])
            cv2.imwrite(os.path.join(VIS_DIR, f"debug_{name}"), vis_row)

        pred_scores = {}
        for model in ensemble:
            try:
                texts, confs = model.predict(tensors)
                for t, c in zip(texts, confs):
                    t_clean = clean_text(t)
                    if t_clean:
                        # Накопление взвешенной уверенности
                        pred_scores[t_clean] = pred_scores.get(t_clean, 0.0) + c
            except: pass

        # Победитель по сумме весов
        final_p = max(pred_scores.items(), key=lambda x: x[1])[0] if pred_scores else ""
        results.append({'Filename': name, 'Price': final_p})

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"\n Результаты сохранены: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()