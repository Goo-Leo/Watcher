import cv2
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import Function
from rknnlite.api import RKNNLite

import Config as config
from detection import *
from voc0712 import *
import utils
import matplotlib.pyplot as plt


RK3588_RKNN_MODEL = 'models/ssd_voc_5000_plus.rknn'
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229), (158, 218, 229), (158, 218, 229)]

if __name__ == '__main__':
    rknn_model = RK3588_RKNN_MODEL
    rknn_lite = RKNNLite()

    # load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    img = cv2.imread('plmm.jpg', cv2.IMREAD_COLOR)
    x = cv2.resize(img, (300, 300)).astype(np.float16)
    x -= (104.0, 117.0, 123.0)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)
    xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
    x_ = xx.data.numpy()
    x_ = x_.astype(np.float16)

    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    softmax = nn.Softmax(dim=-1)
    detect = Detect.apply
    priors = utils.default_prior_box()
    print('--> Running model')
    outputs = rknn_lite.inference(inputs=[x_])
    y = [torch.tensor(lst) for lst in outputs]
    t1 = tuple(y[:6])
    t2 = tuple(y[6:12])
    result = [t1, t2]
    loc, conf = result
    loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

    detections = detect(
        loc.view(loc.size(0), -1, 4),
        softmax(conf.view(conf.size(0), -1, config.class_num)),
        torch.cat([o.view(-1, 4) for o in priors], 0),
        config.class_num,
        200,
        0.7,
        0.45
    ).data

    labels = VOC_CLASSES
    top_k = 10

    # scale each detection back up to the image
    scale = torch.Tensor(img.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.4:
            score = detections[0, i, j, 0]
            label_name = labels[i - 1]
            display_txt = '%s: %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            color = colors_tableau[i]
            cv2.rectangle(img, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), color, 2)
            cv2.putText(img, display_txt, (int(pt[0]), int(pt[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255),
                        1, 8)
            j += 1

    cv2.imshow('test', img)
    cv2.waitKey(100000)
    print("------end-------")
    cv2.imwrite('./done.jpg', img)
    print('done')

    rknn_lite.release()
