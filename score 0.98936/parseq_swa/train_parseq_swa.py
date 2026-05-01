import os
import gc
import cv2
import sys
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

PARSE_PATH = './parseq' 
CSV_PATH = './dataset/train.csv'
IMG_DIR = './dataset/train/train'
OUT_DIR = './outputs/saved_models'

CUSTOM_VOCAB = "0123456789"
IMG_W = 192
IMG_H = 64
BATCH_SIZE = 32
VAL_SPLIT = 0.15
SEED = 42

TOTAL_EPOCHS = 20
WARMUP_EPOCHS = 2
SWA_START_EPOCH = 15      
BASE_LR = 3e-4            
SWA_LR = 5e-5             

sys.path.append(PARSE_PATH) 
from strhub.models.utils import create_model

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def smart_resize_and_pad(image, target_w=IMG_W, target_h=IMG_H):
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    res = cv2.resize(image, (nw, nh))
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off, x_off = (target_h - nh) // 2, (target_w - nw) // 2
    padded[y_off:y_off+nh, x_off:x_off+nw] = res
    return padded

class OCRDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Filename'])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = smart_resize_and_pad(img, IMG_W, IMG_H)
        if self.transforms: img = self.transforms(image=img)['image']
        img = (img / 255.0).astype(np.float32)
        img = (img - 0.5) / 0.5  
        img = torch.tensor(img).permute(2, 0, 1)
        return img, str(row['Price'])

def train_parseq(train_df, val_df, img_dir, save_dir):
    print("\n--- Инициализация PARSeq ---")
    
    # 1. Загружаем модель
    model = create_model('parseq', pretrained=True).to(device)
    core = model.model if hasattr(model, 'model') else model
    
    # 2. Интерполяция (уже проверено, работает)
    if IMG_H != 32 or IMG_W != 128:
        patch_size = core.encoder.patch_embed.patch_size 
        old_h, old_w = 32 // patch_size[0], 128 // patch_size[1]
        new_h, new_w = IMG_H // patch_size[0], IMG_W // patch_size[1]
        pos_embed = core.encoder.pos_embed 
        embed_dim = pos_embed.shape[-1]
        pos_embed_tokens = pos_embed.reshape(1, old_h, old_w, embed_dim).permute(0, 3, 1, 2)
        pos_embed_tokens = F.interpolate(pos_embed_tokens, size=(new_h, new_w), mode='bicubic', align_corners=False)
        new_pos_embed = pos_embed_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        core.encoder.pos_embed = nn.Parameter(new_pos_embed)
        core.encoder.patch_embed.img_size = (IMG_H, IMG_W)
        core.encoder.patch_embed.grid_size = (new_h, new_w)
        core.encoder.patch_embed.num_patches = new_h * new_w

    # 3. ЗАМЕНА СЛОВАРЯ (Исправленная логика)
    new_tokenizer = type(model.tokenizer)(CUSTOM_VOCAB)
    new_vocab_size = len(new_tokenizer)
    print(f"Новый размер словаря: {new_vocab_size}")

    # Прямая замена слоев по их именам в архитектуре PARSeq
    embed_dim = core.text_embed.embedding.embedding_dim
    core.text_embed.embedding = nn.Embedding(new_vocab_size, embed_dim)
    
    head_in_features = core.head.in_features
    core.head = nn.Linear(head_in_features, new_vocab_size)

    # Обновляем токенизатор везде
    model.tokenizer = new_tokenizer
    core.tokenizer = new_tokenizer
    model.bos_id = new_tokenizer.bos_id
    model.eos_id = new_tokenizer.eos_id
    model.pad_id = new_tokenizer.pad_id
    
    # Заглушки для Lightning
    model.log = lambda *args, **kwargs: None
    model.log_dict = lambda *args, **kwargs: None
    class DummyTrainer:
        global_step = 0
        current_epoch = 0
    model.trainer = DummyTrainer()
    model.to(device)

    # 4. Оптимизация
    train_trans = A.Compose([
        # Заменили ShiftScaleRotate на Affine
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # Сдвиг (shift)
            scale=(0.9, 1.1),                                          # Масштаб (scale)
            rotate=(-5, 5),                                            # Поворот (rotate)
            p=0.5
        ),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        A.Perspective(scale=(0.02, 0.05), p=0.3), # Полезно для ценников под углом
        A.MotionBlur(blur_limit=3, p=0.2)
    ])
    
    loader = DataLoader(OCRDataset(train_df, img_dir, train_trans), batch_size=BATCH_SIZE, shuffle=True)
    v_loader = DataLoader(OCRDataset(val_df, img_dir), batch_size=BATCH_SIZE)
    
    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
    scaler = torch.amp.GradScaler()
    
    warmup = LinearLR(opt, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(opt, T_max=(SWA_START_EPOCH - WARMUP_EPOCHS))
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(opt, swa_lr=SWA_LR)

    best_em = 0.0
    for ep in range(1, TOTAL_EPOCHS + 1):
        model.train()
        t_loss = 0
        for imgs, labels in tqdm(loader, desc=f"Epoch {ep}/{TOTAL_EPOCHS}"):
            opt.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
                # Важно: передаем метки для расчета лосса
                loss = model.training_step((imgs.to(device), labels), 0)
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            t_loss += loss.item()
        
        if ep >= SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # --- ВАЛИДАЦИЯ ---
        eval_m = swa_model if ep >= SWA_START_EPOCH else model
        eval_m.eval()
        preds_list, targets_list = [], []
        
        with torch.no_grad():
            for imgs, labels in tqdm(v_loader, desc="Validating"):
                # Получаем логиты
                logits = eval_m(imgs.to(device))
                
                probs = logits.softmax(-1)
                decoded, _ = new_tokenizer.decode(probs)
                
                preds_list.extend(decoded)
                targets_list.extend(labels)
        
        # Считаем метрику
        em = sum([1 for p, t in zip(preds_list, targets_list) if str(p).strip() == str(t).strip()]) / len(targets_list)
        print(f"Ep {ep} | Loss: {t_loss/len(loader):.4f} | EM: {em:.4f} {'(SWA)' if ep >= SWA_START_EPOCH else ''}")
        
        if em > best_em and ep < SWA_START_EPOCH:
            best_em = em
            print("🚀 Сохранение лучшей модели")
            torch.save(model.state_dict(), os.path.join(save_dir, "parseq_swa_best.pth"))

    print(" Финализация SWA (BatchNorm Update)...")
    swa_model.train()
    with torch.no_grad():
        for imgs, _ in loader:
            swa_model(imgs.to(device))
            
    torch.save(swa_model.module.state_dict(), os.path.join(save_dir, "parseq_swa.pth"))

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=SEED)
    train_parseq(train_df, val_df, IMG_DIR, OUT_DIR)