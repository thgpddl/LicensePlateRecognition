from .dataprocess import PreProcess, PostProcess
from .model import LicensePlateRecognizer


class Recognizer:
    def __init__(self):
        self.ppocr = LicensePlateRecognizer(onnx_path="LicensePlateRecognizer/assets/ppocr.onnx")
        self.prep = PreProcess()  # 预处理实例化
        self.postp = PostProcess(character_dict_path='LicensePlateRecognizer/assets/plate_dict.txt')  # 后处理实例化

    def __call__(self, img):
        imp = self.prep(img)  # 预处理
        out = self.ppocr(imp)  # 推理
        rec_result = self.postp(out[0])  # 后处理
        return rec_result
