import cv2
from random import randint
from time import time

MARKER_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
#TAG_ID = randint(0, 10)
TAG_ID = 7
SIZE_PX = 600

img = cv2.aruco.generateImageMarker(MARKER_DICT, TAG_ID, SIZE_PX)

cv2.namedWindow('MARKER_WINDOW')
cv2.imshow('MARKER_WINDOW', img)
cv2.waitKey(0)
cv2.destroyWindow('MARKER_WINDOW')

timestamp = str(int(time()*10))
TAG_NAME = f'tag_{TAG_ID}_{timestamp}.png'
cv2.imwrite(TAG_NAME, img)
print('Tag generated', TAG_NAME)