from LicensePlateDetector.detector import Detector
from LicensePlateRecognizer.recognizer import Recognizer
from utils.getPlateArea import getPlateArea


class LicensePlateAPI:
    def __init__(self, getdet=False):
        """

        :param getdet: 是否一起返回车牌定位信息
        """
        self.detector = Detector()
        self.recognizer = Recognizer()
        self.getdet = getdet

    def __call__(self, img):
        """
        传入车身图片，返回车牌内容和置信度
        :param img: 车身图片[彩色 尺寸任意]
        :return:
        """
        # 预测车牌位置并得到裁剪后车牌图像
        det = self.detector(img)  # 检测车牌位置
        crop = getPlateArea(img, det)  # 裁剪车牌

        # 车牌识别
        res = self.recognizer(crop)  # 识别得到内容和置信度
        if self.getdet:
            return det, res
        return res
