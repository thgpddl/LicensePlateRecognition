from __future__ import print_function
import torch
import numpy as np
from LicensePlateDetector.data import cfg_mnet
from .layers.functions.prior_box import PriorBox
from .utils.nms.py_cpu_nms import py_cpu_nms
from .utils.box_utils import decode, decode_landm


class PreProcess:
    def __init__(self):
        pass

    def __call__(self, img_raw):
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        return img


class PostProcess:
    def __init__(self, confidence_threshold=0.02, nms_threshold=0.4):
        self.cfg = cfg_mnet
        self.resize = 1
        self.variance = [0.1, 0.2]
        self.confidence_threshold = confidence_threshold
        self.top_k = 1000
        self.nms_threshold = nms_threshold
        self.keep_top_k = 500

    def __call__(self, loc, conf, landms, img_raw_shape, img_shape):
        scale = torch.Tensor([img_raw_shape[1], img_raw_shape[0], img_raw_shape[1], img_raw_shape[0]])
        im_height, im_width, _ = img_raw_shape
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.variance)
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms = decode_landm(landms.data.squeeze(0), prior_data, self.variance)
        scale1 = torch.Tensor([img_shape[3], img_shape[2], img_shape[3], img_shape[2],
                               img_shape[3], img_shape[2],
                               img_shape[3], img_shape[2]])
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        return dets
