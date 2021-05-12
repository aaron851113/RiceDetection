# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from matplotlib import colors
import glob
from backbone import EfficientDetBackbone
import cv2
import os
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

compound_coef = 0
force_input_size = None  # set None to use default size
img_path = './datasets/Rice/train/DSC080454 H:1 W:5.JPG'

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (0.75, 1.25), (1.25, 0.75)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.3
iou_threshold = 0.3
use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['rice']

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
#model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location='cpu'))

weights = sorted(glob.glob(f'logs/Rice/efficientdet-d{compound_coef}'+'*.pth'))
weights.sort(key=os.path.getmtime)

for weight in weights:
    print('Load weight :',weight)
    model.load_state_dict(torch.load(weight, map_location='cpu'))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)
        print(out)

    def display(preds, imgs, imshow=False, imwrite=True):
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                continue

            imgs[i] = imgs[i].copy()

            for j in range(len(preds[i]['rois'])):
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
                obj = obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])
                w, h = int((x1+x2)/2), int((y1+y2)/2)
                #cv2.circle(imgs[i],(w, h), 5, (255, 0, 0), -1)
                plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])


            if imshow:
                cv2.imshow('img', imgs[i])
                cv2.waitKey(0)

            if imwrite:
                weight_name = weight.split('-')[-1].split('.')[0]            
                cv2.imwrite(f'test/Rice_test_{weight_name}.jpg', imgs[i])
                #cv2.imwrite('./test/Rice_test.jpg', imgs[i])


    out = invert_affine(framed_metas, out)
    display(out, ori_imgs, imshow=False, imwrite=True)
    
    """
    print('running speed test...')
    with torch.no_grad():
        print('test1: model inferring and postprocessing')
        print('inferring image for 10 times...')
        t1 = time.time()
        for _ in range(10):
            _, regression, classification, anchors = model(x)

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)
            out = invert_affine(framed_metas, out)

        t2 = time.time()
        tact_time = (t2 - t1) / 10
        print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')
    """
