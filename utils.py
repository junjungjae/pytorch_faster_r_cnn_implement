import xml.etree.ElementTree as ET
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

import torch
from torchvision import ops
import torch.nn.functional as F
import torch.optim as optim
import conf as cfg


def calc_gt_offsets(pos_anc_coords, gt_bbox_mapping):
    """
    ground truth와 positive anchor box coordinate 간 offset 계산
    이미지 width 및 height의 경우 object 및 class에 따라 값의 범위가 천차만별이며, 이는 학습에 부정적인 효과를 주기 때문에 log 연산 수행
    x, y 좌표의 경우 이미지 scale에 비례하여 변하기 때문에 별도 연산 필요없음. 오히려 log 연산 시 학습에 부정적인 효과 발생
    """
    pos_anc_coords = ops.box_convert(pos_anc_coords, in_fmt='xyxy', out_fmt='cxcywh')
    gt_bbox_mapping = ops.box_convert(gt_bbox_mapping, in_fmt='xyxy', out_fmt='cxcywh')

    gt_cx, gt_cy, gt_w, gt_h = gt_bbox_mapping[:, 0], gt_bbox_mapping[:, 1], gt_bbox_mapping[:, 2], gt_bbox_mapping[:, 3]
    anc_cx, anc_cy, anc_w, anc_h = pos_anc_coords[:, 0], pos_anc_coords[:, 1], pos_anc_coords[:, 2], pos_anc_coords[:, 3]

    tx_ = (gt_cx - anc_cx)/anc_w
    ty_ = (gt_cy - anc_cy)/anc_h
    tw_ = torch.log(gt_w / anc_w)
    th_ = torch.log(gt_h / anc_h)

    return torch.stack([tx_, ty_, tw_, th_], dim=-1)

def gen_anc_centers(out_size):
    """
    anchor box들의 기본 기점이 되는 좌표들 생성
    """
    out_h, out_w = out_size
    
    anc_pts_x = torch.arange(0, out_w).to(cfg.DEVICE) + 0.5
    anc_pts_y = torch.arange(0, out_h).to(cfg.DEVICE) + 0.5
    
    return anc_pts_x, anc_pts_y

def convert_bboxes_scale(bboxes, width_scale_factor, height_scale_factor, mode='train'):
    """
    원본 스케일 <-> 학습 스케일 변환 시 사용
    train: 원본 스케일 -> 학습 스케일
    real: 학습 스케일 -> 원본 스케일(검증, 시각화 etc)
    padding 여부 고려해서 작성.
    """
    assert mode in ['real', 'train']
    
    batch_size = bboxes.size(dim=0)
    proj_bboxes = bboxes.clone().reshape(batch_size, -1, 4)
    invalid_bbox_mask = (proj_bboxes == -1)
    
    if mode == 'real':
        proj_bboxes[:, :, [0, 2]] *= width_scale_factor
        proj_bboxes[:, :, [1, 3]] *= height_scale_factor
        
    else:
        proj_bboxes[:, :, [0, 2]] /= width_scale_factor
        proj_bboxes[:, :, [1, 3]] /= height_scale_factor
        
    proj_bboxes.masked_fill_(invalid_bbox_mask, -1)
    proj_bboxes.resize_as_(bboxes)
    
    return proj_bboxes

def generate_proposals(anchors, offsets):
    """
    anchor: positive anchor box
    offset: positive anchor box에 대한 offset regression
    위 calc_gt_offset method의 경우 anchor와 ground truth를 가지고 offset을 계산하는 방식이었으므로
    이 method에서 구하고자 하는 proposal에 맞게 계산식 재구성    
    """
    anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')

    proposals_ = torch.zeros_like(anchors).to(cfg.DEVICE)
    proposals_[:,0] = anchors[:,0] + offsets[:,0]*anchors[:,2]
    proposals_[:,1] = anchors[:,1] + offsets[:,1]*anchors[:,3]
    proposals_[:,2] = anchors[:,2] * torch.exp(offsets[:,2])
    proposals_[:,3] = anchors[:,3] * torch.exp(offsets[:,3])

    proposals = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')

    return proposals

def gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, out_size):
    """
    gen_anc_centers에서 생성한 anchor 기반 좌표들과 기정의된 anchor box의 종횡비 및 scale을 활용하여
    각 point 별 anchor box의 bbox coordinate 생성
    """
    n_anc_boxes = len(anc_scales) * len(anc_ratios)
    anc_base = torch.zeros(1, anc_pts_x.size(dim=0), anc_pts_y.size(dim=0), n_anc_boxes, 4).to(cfg.DEVICE)
    
    for ix, xc in enumerate(anc_pts_x):
        for jx, yc in enumerate(anc_pts_y):
            anc_boxes = torch.zeros((n_anc_boxes, 4)).to(cfg.DEVICE)
            c = 0
            for scale in anc_scales:
                for ratio in anc_ratios:
                    w = scale * ratio
                    h = scale
                    
                    xmin = xc - w / 2
                    ymin = yc - h / 2
                    xmax = xc + w / 2
                    ymax = yc + h / 2

                    anc_boxes[c, :] = torch.Tensor([xmin, ymin, xmax, ymax]).to(cfg.DEVICE)
                    c += 1

            # 이미지의 범위를 넘어가는 좌표들에 대한 보정
            anc_base[:, ix, jx, :] = ops.clip_boxes_to_image(anc_boxes, size=out_size)
            
    return anc_base

def get_iou_mat(batch_size, anc_boxes_all, gt_bboxes_all):
    """
    anchor box와 ground truth를 입력받아 iou 계산
    """
    anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)
    tot_anc_boxes = anc_boxes_flat.size(dim=1)
    
    ious_mat = torch.zeros((batch_size, tot_anc_boxes, gt_bboxes_all.size(dim=1))).to(cfg.DEVICE)

    for i in range(batch_size):
        gt_bboxes = gt_bboxes_all[i]
        anc_boxes = anc_boxes_flat[i]
        ious_mat[i, :] = ops.box_iou(anc_boxes, gt_bboxes)
        
    return ious_mat

def get_req_anchors(anc_boxes_all, gt_bboxes_all, gt_classes_all, pos_thresh=0.7, neg_thresh=0.2):
    """
    학습에 필요한 정보 전반을 가공
    positive anchor box indices, positive anchor box coordinates
    negative anchor box indices, negative anchor box coordiantes
    ground truth class, ground truth offset
    """
    batch_size, anchor_width, anchor_height, anchor_num, _ = anc_boxes_all.shape
    max_gt_len = gt_bboxes_all.shape[1] # batch 내 단일 이미지 기준 최다 ground truth 개수
    total_anchor_boxes = anchor_num * anchor_width * anchor_height

    # batch 내 포함된 anchor와 ground truth간 IoU 계산
    # iou_mat -> (batch_size, anchor_num, batch's max ground_truth_num)
    iou_mat = get_iou_mat(batch_size, anc_boxes_all, gt_bboxes_all)
    
    # 이미지 내 각 ground truth에 대한 anchor box의 최대 IoU
    max_iou_per_gt_box, _ = iou_mat.max(dim=1, keepdim=True)
    
    # (조건 1) IoU matrix에서 최대 IoU에 해당 & max iou가 0을 초과하는 조건을 만족하는 마스크 생성
    positive_anc_mask = torch.logical_and(iou_mat == max_iou_per_gt_box, max_iou_per_gt_box > 0) 

    # (조건 2) 조건 1에서 구한 anchor mask | IoU matrix에서 positive threshold를 충족하는 조건을 만족하는 마스크 
    # 바로 threshold 안걸고 2번에 걸쳐 마스크 생성하는 이유?
    # -> 만약 anchor box들의 max IoU가 positive threshold 미만이면 생성되는 positive anchor box가 없음
    positive_anc_mask = torch.logical_or(positive_anc_mask, iou_mat > pos_thresh)
    
    # mask 값이 True인 anchor의 batch index 반환
    # -> 추후 batch 별 proposal 탐색 시 사용
    positive_anc_ind_per_batch = torch.where(positive_anc_mask)[0]
    
    # bacth로 구분된 mask flatten 후 index 추출
    # -> 밑의 positive anchor box 추출 시 사용
    positive_anc_mask = positive_anc_mask.flatten(start_dim=0, end_dim=1)
    positive_anc_ind = torch.where(positive_anc_mask)[0]
    
    # max iou를 가지는 anchor box indices와 그 값을 추출    
    max_iou_per_anc, max_iou_per_anc_ind = iou_mat.max(dim=-1)
    max_iou_per_anc = max_iou_per_anc.flatten(start_dim=0, end_dim=1)
    
    # anchor box shape에 맞게 차원 조정 및 expand 수행
    gt_classes_expand = gt_classes_all.view(batch_size, 1, max_gt_len).expand(batch_size, total_anchor_boxes, max_gt_len)
    
    # 재조정한 gt_classes에서 max_iou 값에 해당하는 class 추출
    GT_class = torch.gather(gt_classes_expand, -1, max_iou_per_anc_ind.unsqueeze(-1)).squeeze(-1)
    GT_class = GT_class.flatten(start_dim=0, end_dim=1)
    GT_class_pos = GT_class[positive_anc_ind]
    
    # anchor box shape에 맞게 차원 조정 및 expand 수행
    gt_bboxes_expand = gt_bboxes_all.view(batch_size, 1, max_gt_len, 4).expand(batch_size, total_anchor_boxes, max_gt_len, 4)

    # 재조정한 gt_bboxes에서 max_iou 값에 해당하는 anchor box 추출
    GT_bboxes = torch.gather(gt_bboxes_expand, -2, max_iou_per_anc_ind.reshape(batch_size, total_anchor_boxes, 1, 1).repeat(1, 1, 1, 4))
    GT_bboxes = GT_bboxes.flatten(start_dim=0, end_dim=2)
    GT_bboxes_pos = GT_bboxes[positive_anc_ind]
    
    # positive anchor box 추출
    anc_boxes_flat = anc_boxes_all.flatten(start_dim=0, end_dim=-2) # flatten all the anchor boxes
    positive_anc_coords = anc_boxes_flat[positive_anc_ind]
    
    # positive anchor box, ground truth anchor box 간 offset 추출
    GT_offsets = calc_gt_offsets(positive_anc_coords, GT_bboxes_pos)
    
    
    # negative anchor box의 indices 및 coordinates 추출
    negative_anc_mask = (max_iou_per_anc < neg_thresh)
    negative_anc_ind = torch.where(negative_anc_mask)[0]

    # negative anchor box 개수가 positive anchor box 개수에 비해 많음
    # 그러므로 전체 negative anchor box의 indices 및 coordinate 중 일부를 sampling함.
    negative_anc_ind = negative_anc_ind[torch.randint(0, negative_anc_ind.shape[0], (positive_anc_ind.shape[0],))]
    negative_anc_coords = anc_boxes_flat[negative_anc_ind]
    
    return positive_anc_ind, negative_anc_ind, GT_offsets, GT_class_pos, \
         positive_anc_coords, negative_anc_coords, positive_anc_ind_per_batch

# -------------- 학습관련 유틸 ----------------

class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.weights_save_dir = cfg.SAVE_WEIGHT_DIR
        
        if not os.path.exists(self.weights_save_dir):
            os.mkdir(self.weights_save_dir)
        
        self.metric_min = np.inf

    def __call__(self, target_score, model):    
        self._save_checkpoint(target_score, model, islast=True)

        if not self.best_score:
            self._save_checkpoint(target_score, model)
            self.best_score = target_score
            
        elif (target_score > self.best_score + self.delta):
            print(f"Current score: {target_score}, best score: {self.best_score}")
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self._save_checkpoint(target_score, model)
            self.counter = 0

    def _save_checkpoint(self, target_score, model, islast=False):        
        
        if not os.path.isdir(self.weights_save_dir):
            os.mkdir(self.weights_save_dir)
        
        if islast:
            torch.save(model.state_dict(), f"{self.weights_save_dir}/last_weights.pt")
            
        else:
            if self.verbose and self.best_score:
                print(f'Validation loss improved ({self.best_score:.6f} --> {target_score:.6f}).  Saving model ...')
                torch.save(model.state_dict(), f"{self.weights_save_dir}/best_weights.pt")
                self.best_score = target_score
            
            else:
                torch.save(model.state_dict(), f"{self.weights_save_dir}/best_weights.pt")
                self.best_score = target_score