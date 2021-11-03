#  0. 前言
这是一个集车牌检测和车牌识别一体的车牌识别项目
- 车牌检测项目参考了：[License Plate Detection with RetinaFace](https://github.com/zeusees/License-Plate-Detector)
- 车牌识别项目参考了：[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

接收一个仅包含车身（比如YOLO检测到的车身）图片，然后程序内部完成车牌检测和车牌识别后，输出车牌内容和置信度

# 1. 如何使用？
**1. 安装依赖**
requirements.txt定义了需要的库以及版本，你需要运行以下代码安装：
`pip install -r requirements.txt`
**2. 使用**
你可以从test.py脚本中看到使用实例
```python
from LicensePlateAPI import LicensePlateAPI
import cv2


lpapi=LicensePlateAPI()     # 实例化车牌识别器

img=cv2.imread("imgs/2.jpg")

out=lpapi(img)        # 车牌识别并获取结果

print(out)    # 输出结果
```
