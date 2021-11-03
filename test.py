from LicensePlateAPI import LicensePlateAPI
import cv2


lpapi=LicensePlateAPI()     # 实例化车牌识别器

img=cv2.imread("imgs/2.jpg")
out=lpapi(img)      # 车牌识别并获取结果
print(out)

