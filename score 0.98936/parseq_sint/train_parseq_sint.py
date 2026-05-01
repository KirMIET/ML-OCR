import os
import cv2
import sys
import torch
import random
import gc
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, SWALR

PARSE_PATH = './parseq' 
CSV_PATH = './dataset/full_labels.csv'
IMG_DIR = './dataset/full_data'
OUT_DIR = './outputs/saved_models'

CUSTOM_VOCAB = "0123456789"
IMG_W, IMG_H = 192, 64
BATCH_SIZE = 64
VAL_SPLIT = 0.15
SEED = 42

TOTAL_EPOCHS = 20
WARMUP_EPOCHS = 2
SWA_START_EPOCH = 15      
BASE_LR = 7e-4            
SWA_LR = 5e-5             

NUM_WORKERS = 4 

sys.path.append(PARSE_PATH) 
from strhub.models.utils import create_model

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True 

seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OCRDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        self.filenames = df['Filename'].tolist()
        self.labels = df['Price'].astype(str).tolist()

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = (img / 127.5) - 1.0
        return img, self.labels[idx]

def cleanup_resources():
    """Очистка памяти GPU и сборка мусора"""
    print("\n--- Очистка ресурсов ---")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("Память очищена.")

def train_parseq(train_df, val_df, img_dir, save_dir):
    # Объявляем переменные заранее, чтобы они были доступны в блоке finally
    model = None
    opt = None
    loader = None
    v_loader = None

    try:
        print(f"\n--- Инициализация модели на устройстве: {device} ---")
        
        model = create_model('parseq', pretrained=True).to(device)
        core = model.model if hasattr(model, 'model') else model
        
        # --- ПОДАВЛЕНИЕ WARNINGS LIGHTNING ---
        class MockTrainer:
            def __init__(self):
                self.logger = None
                self.global_step = 0
                self.current_epoch = 0
                self.fast_dev_run = False
        
        # Заменяем методы логирования на пустые функции
        model.log = lambda *args, **kwargs: None
        model.log_dict = lambda *args, **kwargs: None
        # Прикрепляем фиктивный тренер
        model.trainer = MockTrainer()
        # --------------------------------------

        # Далее идет ваш код ресайза эмбеддингов и замены словаря...
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

        # Замена словаря (фикс device)
        new_tokenizer = type(model.tokenizer)(CUSTOM_VOCAB)
        new_vocab_size = len(new_tokenizer)
        embed_dim = core.text_embed.embedding.embedding_dim
        
        core.text_embed.embedding = nn.Embedding(new_vocab_size, embed_dim).to(device)
        core.head = nn.Linear(core.head.in_features, new_vocab_size).to(device)
        
        model.tokenizer = core.tokenizer = new_tokenizer
        model.bos_id, model.eos_id, model.pad_id = new_tokenizer.bos_id, new_tokenizer.eos_id, new_tokenizer.pad_id
        
        model.to(device)
        model.collect_cache_offsets = lambda x: x 

        # Подготовка данных
        train_trans = A.Compose([
            A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, rotate=(-5, 5), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            A.Perspective(scale=(0.02, 0.05), p=0.3)
        ])
        
        loader_params = {
            'batch_size': BATCH_SIZE,
            'num_workers': NUM_WORKERS,
            'pin_memory': True,
            'persistent_workers': True if NUM_WORKERS > 0 else False
        }
        
        loader = DataLoader(OCRDataset(train_df, img_dir, train_trans), shuffle=True, **loader_params)
        v_loader = DataLoader(OCRDataset(val_df, img_dir), shuffle=False, **loader_params)
        
        opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
        scaler = torch.amp.GradScaler('cuda')
        
        warmup = LinearLR(opt, start_factor=0.1, total_iters=WARMUP_EPOCHS)
        cosine = CosineAnnealingLR(opt, T_max=(SWA_START_EPOCH - WARMUP_EPOCHS))
        scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
        
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(opt, swa_lr=SWA_LR)

        best_em = 0.0
        
        # ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
        for ep in range(1, TOTAL_EPOCHS + 1):
            model.train()
            t_loss = 0
            pbar = tqdm(loader, desc=f"Epoch {ep}/{TOTAL_EPOCHS}")
            for imgs, labels in pbar:
                opt.zero_grad(set_to_none=True)
                imgs = imgs.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    loss = model.training_step((imgs, labels), 0)
                
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                
                t_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            if ep >= SWA_START_EPOCH:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

            # Валидация
            eval_m = swa_model if ep >= SWA_START_EPOCH else model
            eval_m.eval()
            preds_list, targets_list = [], []
            
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                for imgs, labels in v_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    logits = eval_m(imgs)
                    probs = logits.softmax(-1)
                    decoded, _ = new_tokenizer.decode(probs)
                    preds_list.extend(decoded)
                    targets_list.extend(labels)
            
            em = sum([1 for p, t in zip(preds_list, targets_list) if str(p).strip() == str(t).strip()]) / len(targets_list)
            print(f"Ep {ep} | Loss: {t_loss/len(loader):.4f} | EM: {em:.4f}")
            
            if em > best_em and ep < SWA_START_EPOCH:
                best_em = em
                torch.save(model.state_dict(), os.path.join(save_dir, "parseq_sint_best.pth"))

        # Сохранение финала
        final_model = swa_model.module if hasattr(swa_model, 'module') else swa_model
        torch.save(final_model.state_dict(), os.path.join(save_dir, "parseq_sint.pth"))
        print("\nОбучение успешно завершено.")

    except KeyboardInterrupt:
        print("\n[!] Обучение прервано пользователем.")
    except Exception as e:
        print(f"\n[!!!] КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    finally:
        del model
        del swa_model
        del opt
        del loader
        del v_loader
        cleanup_resources()

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=SEED)
    
    train_parseq(train_df, val_df, IMG_DIR, OUT_DIR)