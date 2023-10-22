import torch
import os

import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

class CustomDataset(Dataset):
    def __init__(self, dataset_dir, name2idx, dataset_div, apply_transform):
        self.dataset_dir = dataset_dir
        self.name2idx = name2idx
        self.dataset_div = dataset_div
        self.transform = apply_transform
        
        self.img_data_all, self.gt_bboxes_all, self.gt_classes_all = self.get_data()
        
    def __len__(self):
        return self.img_data_all.size(dim=0)
    
    def __getitem__(self, idx):
        return self.img_data_all[idx], self.gt_bboxes_all[idx], self.gt_classes_all[idx]
        
    def get_data(self):
        img_data_all = []
        gt_idxs_all = []
        
        gt_boxes_all, gt_classes_all, img_paths = parse_annotation(self.dataset_dir, self.dataset_div)
        gt_augmented_all = []
        
        for i, img_path in tqdm(enumerate(img_paths)):
            
            if (not img_path) or (not os.path.exists(img_path)):
                continue
            
            img = np.array(Image.open(img_path))
            gt_idx = torch.tensor([self.name2idx[name] for name in gt_classes_all[i]])

                
            augmentation_data = self.transform(image=img, bboxes=gt_boxes_all[i], class_labels=gt_idx)
            
            img_data_all.append(augmentation_data['image'])
            gt_idxs_all.append(gt_idx)
            gt_augmented_all.append(torch.tensor(augmentation_data['bboxes']))

        # 각 이미지마다 bbox의 개수가 다르기에 통일 필요
        gt_bboxes_pad = pad_sequence(gt_augmented_all, batch_first=True, padding_value=-1)
        gt_classes_pad = pad_sequence(gt_idxs_all, batch_first=True, padding_value=-1)
        
        img_data_stacked = torch.stack(img_data_all, dim=0)
        
        return img_data_stacked, gt_bboxes_pad, gt_classes_pad
    

def parse_annotation(dataset_dir, dataset_div):
    """
    데이터셋 구분(class 1, class 2... train, val, test)에 따른 bounding box, class 정보 및 이미지 경로 파싱.
    """
    filedir = os.path.join(dataset_dir, 'ImageSets', 'Main', f'{dataset_div}.txt')
    with open(filedir, "r") as t:
        fileList = t.read().splitlines()
        xmlList = [os.path.join(dataset_dir, 'Annotations', f'{filename.split()[0]}.xml') for filename in fileList]
        imgList = [os.path.join(dataset_dir, 'JPEGImages', f'{filename.split()[0]}.jpg') for filename in fileList]
    gt_boxes_all = []
    gt_classes_all = []

    for single_xml in xmlList:
        with open(single_xml, "r") as f:
            tree = ET.parse(f)

        root = tree.getroot()

        groundtruth_boxes = []
        groundtruth_classes = []

        for object_ in root.findall('object'):
            for box_ in object_.findall('bndbox'):
                label = object_.find("name").text             
                
                # bounding box 좌표
                # 1을 빼주는 이유 <- 컴퓨터의 인덱스 시작값은 0인 반면 Pascal VOC의 시작값은 1이라
                xmin = float(box_.find("xmin").text) - 1
                ymin = float(box_.find("ymin").text) - 1
                xmax = float(box_.find("xmax").text) - 1
                ymax = float(box_.find("ymax").text) - 1

                # 입력 이미지 크기와 모델 내 처리 이미지 크기가 다르기 때문에 bounding box scale 보정
                bbox = torch.Tensor([xmin, ymin, xmax, ymax])
                groundtruth_boxes.append(bbox.tolist())                
                groundtruth_classes.append(label)

        gt_boxes_all.append(torch.Tensor(groundtruth_boxes))
        gt_classes_all.append(groundtruth_classes)

    return gt_boxes_all, gt_classes_all, imgList