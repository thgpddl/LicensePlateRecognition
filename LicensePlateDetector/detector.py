from .model import LicensePlateDetector
from .dataprocess import PostProcess, PreProcess


class Detector:
    def __init__(self):
        self.LPDetector = LicensePlateDetector()  # 模型实例化
        self.prp = PreProcess()  # 预处理实例化
        self.pop = PostProcess(confidence_threshold=0.5)  # 后处理实例化

    def __call__(self, img_raw):
        img = self.prp(img_raw)  # 预处理
        loc, conf, landms = self.LPDetector(img)  # 推理
        dets = self.pop(loc, conf, landms, img_raw.shape, img.shape)  # 后处理
        # 一般来讲，传入车身图，最多检测到一个车牌
        if dets.__len__():
            # 坐上，右下，置信度，关键点
            # p = list(map(int, [det[0], det[1], det[2], det[3], det[5], det[6], det[7], det[8], det[9], det[10], det[11],det[12]]))
            return dets[0]

