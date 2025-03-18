import cv2 as cv
import numpy as np

map_image = cv.imread("videos/agv.png")

hsv = cv.cvtColor(map_image, cv.COLOR_BGR2HSV)

lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])

mask = cv.inRange(hsv, lower_green, upper_green)

map_image[mask > 0] = [255, 255, 255]

cv.imwrite("processes_map.png",map_image)

cv.imshow('a',map_image)


cv.waitKey(0)




