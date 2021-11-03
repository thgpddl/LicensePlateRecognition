
def getPlateArea(img, det):
    p = list(
        map(int, [det[0], det[1], det[2], det[3], det[5], det[6], det[7], det[8], det[9], det[10], det[11], det[12]]))
    crop = img[p[1]:p[3],p[0]:p[2]]
    return crop
