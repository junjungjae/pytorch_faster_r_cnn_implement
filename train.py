import time
import torch

import albumentations as A
import numpy as np

from albumentations.pytorch import transforms
from torch.utils.data import DataLoader

import conf as cfg

from custom_dataset import CustomDataset
from model import TwoStageDetector
from utils import EarlyStopping

import warnings
warnings.filterwarnings("ignore")


img_width, img_height = (640, 480)

voc_path = "/pascal voc path"
name2idx = cfg.CLASSES_DICT
idx2name = {v:k for k, v in name2idx.items()}


# albumentation 이용하여 간단한 rescale 및 tensor 변환
A_Resize = A.Compose([
    A.Resize(img_height, img_width),
    A.Normalize(),
    transforms.ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

train_dataset = CustomDataset(dataset_dir=voc_path, name2idx=name2idx, dataset_div='train',
                                       apply_transform=A_Resize)

valid_dataset = CustomDataset(dataset_dir=voc_path, name2idx=name2idx, dataset_div='val',
                                       apply_transform=A_Resize)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=True, pin_memory=True)

out_c, out_h, out_w = (2048, 15, 20)

width_scale_factor = img_width // out_w
height_scale_factor = img_height // out_h

img_size = (img_height, img_width)
out_size = (out_h, out_w)
n_classes = len(name2idx) - 1  # background(padding) 인덱스 제거
roi_size = (7, 7)

model = TwoStageDetector(img_size, out_size, out_c, n_classes, roi_size).to(cfg.DEVICE)

learning_rate = 0.001
n_epochs = 1000
    
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
earlystopper = EarlyStopping(patience=100, verbose=True)

for i in range(n_epochs):
    train_epoch_loss = 0
    
    model.train()
    for train_batch_idx, (train_img_batch, train_gt_bboxes_batch, train_gt_classes_batch) in enumerate(train_dataloader):
        
        start_time = time.time()
        train_img_batch = train_img_batch.float().to(cfg.DEVICE)
        train_gt_bboxes_batch = train_gt_bboxes_batch.float().to(cfg.DEVICE)
        train_gt_classes_batch = train_gt_classes_batch.float().to(cfg.DEVICE)
        
        train_loss = model(train_img_batch, train_gt_bboxes_batch, train_gt_classes_batch)
        train_epoch_loss = (train_epoch_loss * train_batch_idx + train_loss.item()) / (train_batch_idx + 1)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        print(f"Epochs: {i + 1}\tBatch:{train_batch_idx + 1}/{len(train_dataloader)}\tTime: {np.round(time.time() - start_time, 2)}\tTrain Loss: {train_epoch_loss:.2f}", end='\r')
        
    print()
    valid_epoch_loss = 0
    
    model.eval()
    with torch.no_grad():
        for valid_batch_idx, (valid_img_batch, valid_gt_bboxes_batch, valid_gt_classes_batch) in enumerate(valid_dataloader):
            start_time = time.time()
            valid_img_batch = valid_img_batch.float().to(cfg.DEVICE)
            valid_gt_bboxes_batch = valid_gt_bboxes_batch.float().to(cfg.DEVICE)
            valid_gt_classes_batch = valid_gt_classes_batch.float().to(cfg.DEVICE)
            
            valid_loss = model(valid_img_batch, valid_gt_bboxes_batch, valid_gt_classes_batch)
            valid_epoch_loss = (valid_epoch_loss * valid_batch_idx + valid_loss.item()) / (valid_batch_idx + 1)

            print(f"Epochs: {i + 1}\tBatch:{valid_batch_idx + 1}/{len(train_dataloader)}\tTime: {np.round(time.time() - start_time, 2)}\tvalid Loss: {valid_epoch_loss:.2f}", end='\r')
            
    print()
    earlystopper(target_score=valid_epoch_loss, model=model)
    
    if earlystopper.early_stop:
        print("EarlyStopping Activate")
        break