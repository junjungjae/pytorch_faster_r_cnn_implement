import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_WEIGHT_DIR = './weights'
CLASSES_DICT = {'pad': -1, 'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                'bus': 5, 'car': 6, 'chair': 7, 'cow': 8, 'diningtable': 9,
                'dog': 10, 'horse': 11, 'motorbike': 12, 'person': 13, 'pottedplant': 14,
                'sheep': 15, 'sofa': 16, 'train': 17, 'tvmonitor': 18, 'cat': 19}

IDX2CLASSES = {v:k for k, v in CLASSES_DICT.items()}