import argparse

import albumentations as A
import albumentations.pytorch.transforms as A_transform

from PIL import Image, ImageDraw
from torchvision.ops import clip_boxes_to_image
from torchvision.transforms.functional import to_pil_image

import conf as cfg

from utils import *
from model import TwoStageDetector



def detect(img, conf_thres, nms_thres, model, transfer_module):
    transformed_img = transfer_module(image=np.array(img))['image'].to(cfg.DEVICE)
    proposals_final, _, classes_final = model.inference(transformed_img.float().unsqueeze(0), conf_thresh=conf_thres, nms_thresh=nms_thres)

    prop_proj = convert_bboxes_scale(proposals_final[0].to('cpu'), 32, 32, mode='real')

    classes_pred = [cfg.IDX2CLASSES[cls] for cls in classes_final[0].tolist()]

    draw = ImageDraw.Draw(img)

    for bbox_pred, label in zip(prop_proj, classes_pred):
        bbox_pred = bbox_pred.tolist()
        print(bbox_pred, label)
        
        draw.rectangle(bbox_pred, outline=(0, 255, 0))
        draw.text(bbox_pred[:2], label, (255, 0, 0))

    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", dest="img_path")
    parser.add_argument("--weights", dest="weights_path")
    args = parser.parse_args()
    
    model = TwoStageDetector(img_size=(640, 480), out_size=(15, 20), out_channels=2048, n_classes=20, roi_size=(7, 7)).to(cfg.DEVICE)
    model.load_state_dict(torch.load(args.weights_path))
    model.eval()
    
    A_inference = A.Compose([A.Resize(640, 480),
                             A.Normalize(),
                             A_transform.ToTensorV2()])
    
    img = Image.open(args.img_path)
    
    
    detect(img, conf_thres=0.95, nms_thres=0.05, model=model, transfer_module=A_inference).show()
    
    
    
    