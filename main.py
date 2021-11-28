import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS,  60)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
listImg = os.listdir("New_folder")
print(listImg)
imglist = []
for imgpath in listImg:
    img = cv2.imread(f'New_folder/{imgpath}')
    imglist.append(img)


indeximg = 1
while True:
    secc , img = cap.read()
    imgout = segmentor.removeBG(img,imglist[indeximg] , threshold=0.50)


    imGStacked = cvzone.stackImages([img,imgout],2,1)
    _, imGStacked = fpsReader.update(imGStacked,color = (0,0,255))

    cv2.imshow("Image", imGStacked)
    key = cv2.waitKey(1)
    if key == ord('a'):
        if indeximg > 0:
            indeximg -= 1
            print(indeximg)
    elif key == ord('d'):
        if indeximg < len(imglist) - 1:
            indeximg += 1
            print(indeximg)
    elif key == ord('q'):
        break
