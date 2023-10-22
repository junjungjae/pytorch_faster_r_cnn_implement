import torch
import torchvision
from torchvision import ops
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import conf as cfg

from utils import *


class FeatureExtractor(nn.Module):
    """
    Backbone network.
    Pytorch의 pretrained 된 resnet50 모델을 사용했으며
    fully-connected 이전 layer만 feature extractor로 사용
    """
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(pretrained=True).to(cfg.DEVICE)
        req_layers = list(model.children())[:8]
        self.backbone = nn.Sequential(*req_layers)
        
        for param in self.backbone.named_parameters():
            param[1].requires_grad = False
        
    def forward(self, img_data):
        return self.backbone(img_data)
    
class ProposalModule(nn.Module):
    """
    Region Proposal Network 중 anchor box 생성과 관련된 모듈
    """
    def __init__(self, in_features, hidden_dim=512, n_anchors=9, p_dropout=0.3):
        super().__init__()
        self.n_anchors = n_anchors
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p_dropout)
        self.conf_head = nn.Conv2d(hidden_dim, n_anchors, kernel_size=1)
        self.reg_head = nn.Conv2d(hidden_dim, n_anchors * 4, kernel_size=1)
        
    def forward(self, feature_map, pos_anc_ind=None, neg_anc_ind=None, pos_anc_coords=None):
        # inference 최적화를 위해 모델 학습과 inference 파트를 구별하여 작성
        if pos_anc_ind is None or neg_anc_ind is None or pos_anc_coords is None:
            mode = 'eval'
        else:
            mode = 'train'
            
        out = self.conv1(feature_map)
        out = F.relu(self.dropout(out))
        
        reg_offsets_pred = self.reg_head(out)
        conf_scores_pred = self.conf_head(out)
        
        if mode == 'train': 
            conf_scores_pos = conf_scores_pred.flatten()[pos_anc_ind]
            conf_scores_neg = conf_scores_pred.flatten()[neg_anc_ind]
            offsets_pos = reg_offsets_pred.contiguous().view(-1, 4)[pos_anc_ind]
            proposals = generate_proposals(pos_anc_coords, offsets_pos)
            
            return conf_scores_pos, conf_scores_neg, offsets_pos, proposals
            
        elif mode == 'eval':
            return conf_scores_pred, reg_offsets_pred
        
class RegionProposalNetwork(nn.Module):
    def __init__(self, img_size, out_size, out_channels):
        super().__init__()
        
        self.img_height, self.img_width = img_size
        self.out_h, self.out_w = out_size
        
        self.width_scale_factor = self.img_width // self.out_w
        self.height_scale_factor = self.img_height // self.out_h 
        
        # 사전정의된 scale, aspect ratio
        self.anc_scales = [2, 4, 6]
        self.anc_ratios = [0.5, 1, 1.5]
        self.n_anc_boxes = len(self.anc_scales) * len(self.anc_ratios)
        
        self.pos_thresh = 0.7
        self.neg_thresh = 0.3
        
        self.w_conf = 1
        self.w_reg = 5
        
        self.feature_extractor = FeatureExtractor().to(cfg.DEVICE)
        self.proposal_module = ProposalModule(out_channels, n_anchors=self.n_anc_boxes).to(cfg.DEVICE)
        
    def forward(self, images, gt_bboxes, gt_classes):
        batch_size = images.size(dim=0)
        feature_map = self.feature_extractor(images)
        
        # 기반 anchor 정보 생성(anchor box 중심좌표, anchor box coordinates)
        anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(self.out_h, self.out_w))
        anc_base = gen_anc_base(anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
        anc_boxes_all = anc_base.repeat(batch_size, 1, 1, 1, 1)
        
        # ground truth box를 학습셋에 맞게 rescaling
        gt_bboxes_proj = convert_bboxes_scale(gt_bboxes, self.width_scale_factor, self.height_scale_factor, mode='train')
        
        # positive, negative, ground truth 관련 필요정보 추출
        positive_anc_ind, negative_anc_ind, \
        GT_offsets, GT_class_pos, positive_anc_coords, \
        negative_anc_coords, positive_anc_ind_sep = get_req_anchors(anc_boxes_all, gt_bboxes_proj, gt_classes)
        
        # feature map에 대해 positive, negative anchor 정보를 통해 proposal 및 관련 정보 생성
        conf_scores_pos, conf_scores_neg, offsets_pos, proposals = self.proposal_module(feature_map, positive_anc_ind, \
                                                                                        negative_anc_ind, positive_anc_coords)
        
        # 가공한 정보들 기반 classifcation, regression loss 계산
        cls_loss = calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size)
        reg_loss = calc_bbox_reg_loss(GT_offsets, offsets_pos, batch_size)
        
        # 통합 loss의 경우 사전정의된 각 파트별 가중치로 계산
        total_rpn_loss = self.w_conf * cls_loss + self.w_reg * reg_loss
        
        return total_rpn_loss, feature_map, proposals, positive_anc_ind_sep, GT_class_pos
    
    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):
        with torch.no_grad():
            batch_size = images.size(dim=0)
            feature_map = self.feature_extractor(images)

            anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(self.out_h, self.out_w))
            anc_base = gen_anc_base(anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
            anc_boxes_all = anc_base.repeat(batch_size, 1, 1, 1, 1)
            anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)

            conf_scores_pred, offsets_pred = self.proposal_module(feature_map)
            conf_scores_pred = conf_scores_pred.reshape(batch_size, -1)
            offsets_pred = offsets_pred.reshape(batch_size, -1, 4)

            proposals_final = []
            conf_scores_final = []
            
            for i in range(batch_size):
                conf_scores = torch.sigmoid(conf_scores_pred[i])
                
                offsets = offsets_pred[i]
                anc_boxes = anc_boxes_flat[i]
                
                proposals = generate_proposals(anc_boxes, offsets)
                
                conf_idx = torch.where(conf_scores >= conf_thresh)[0]
                conf_scores_pos = conf_scores[conf_idx]
                proposals_pos = proposals[conf_idx]
                
                nms_idx = ops.nms(proposals_pos, conf_scores_pos, nms_thresh)
                conf_scores_pos = conf_scores_pos[nms_idx]
                proposals_pos = proposals_pos[nms_idx]

                proposals_final.append(proposals_pos)
                conf_scores_final.append(conf_scores_pos)
            
        return proposals_final, conf_scores_final, feature_map
    
class ClassificationModule(nn.Module):
    def __init__(self, out_channels, n_classes, roi_size, hidden_dim=512, p_dropout=0.3):
        super().__init__()        
        self.roi_size = roi_size

        self.avg_pool = nn.AvgPool2d(self.roi_size)
        self.fc = nn.Linear(out_channels, hidden_dim)
        self.dropout = nn.Dropout(p_dropout)
        
        self.cls_head = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, feature_map, proposals_list, gt_classes=None):
        
        if gt_classes is None:
            mode = 'eval'
        else:
            mode = 'train'
        
        # 가변적인 크기의 feature map 및 box를 지정된 크기에 맞게 pooling(for classification)
        roi_out = ops.roi_pool(feature_map, proposals_list, self.roi_size)
        roi_out = self.avg_pool(roi_out)
        
        roi_out = roi_out.squeeze(-1).squeeze(-1)
        
        out = self.fc(roi_out)
        out = F.relu(self.dropout(out))
        
        cls_scores = self.cls_head(out)
        
        if mode == 'eval':
            return cls_scores
        
        cls_loss = F.cross_entropy(cls_scores, gt_classes.long())
        
        return cls_loss
    
class TwoStageDetector(nn.Module):
    def __init__(self, img_size, out_size, out_channels, n_classes, roi_size):
        super().__init__() 
        self.rpn = RegionProposalNetwork(img_size, out_size, out_channels).to(cfg.DEVICE)
        self.classifier = ClassificationModule(out_channels, n_classes, roi_size).to(cfg.DEVICE)
        
    def forward(self, images, gt_bboxes, gt_classes):
        total_rpn_loss, feature_map, proposals, \
        positive_anc_ind_sep, GT_class_pos = self.rpn(images, gt_bboxes, gt_classes)
        
        pos_proposals_list = []
        batch_size = images.size(dim=0)
        
        for idx in range(batch_size):
            proposal_idxs = torch.where(positive_anc_ind_sep == idx)[0]
            proposals_sep = proposals[proposal_idxs].detach().clone()
            pos_proposals_list.append(proposals_sep)
        
        cls_loss = self.classifier(feature_map, pos_proposals_list, GT_class_pos)
        total_loss = cls_loss + total_rpn_loss
        
        return total_loss
    
    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):
        # inference의 경우 train과 필요 정보, 반환 정보가 다름
        # 특정 모듈에 대한 inference 별도 구현하여 불필요 구간을 생략하여 최적화
        batch_size = images.size(dim=0)
        proposals_final, conf_scores_final, feature_map = self.rpn.inference(images, conf_thresh, nms_thresh)
        cls_scores = self.classifier(feature_map, proposals_final)
        
        cls_probs = F.softmax(cls_scores, dim=-1)
        classes_all = torch.argmax(cls_probs, dim=-1)
        
        classes_final = []
        c = 0
        for i in range(batch_size):
            n_proposals = len(proposals_final[i])
            classes_final.append(classes_all[c: c+n_proposals])
            c += n_proposals
            
        return proposals_final, conf_scores_final, classes_final


def calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size):
    target_pos = torch.ones_like(conf_scores_pos).to(cfg.DEVICE)
    target_neg = torch.zeros_like(conf_scores_neg).to(cfg.DEVICE)
    
    target = torch.cat((target_pos, target_neg))
    inputs = torch.cat((conf_scores_pos, conf_scores_neg))
     
    loss = F.binary_cross_entropy_with_logits(inputs, target, reduction='sum') * 1. / batch_size
    
    return loss

def calc_bbox_reg_loss(gt_offsets, reg_offsets_pos, batch_size):
    loss = F.smooth_l1_loss(reg_offsets_pos, gt_offsets, reduction='sum') * 1. / batch_size
    return loss