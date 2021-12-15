import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def stitching():
    img1 = cv.imread('./data/milanDst.jpg')
    img2 = cv.imread('./data/milanS.jpg')
    img_mask = cv.imread('./data/mask.jpg')
    kernel = np.ones((100, 100), np.uint8)
    img_mask = cv.dilate(img_mask, kernel, iterations = 1)
    img_mask = cv.bitwise_not(img_mask)

    akaze = cv.AKAZE_create()

    kp1, des1 = akaze.detectAndCompute(img1, img_mask)
    kp2, des2 = akaze.detectAndCompute(img2, img_mask)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < 0.55*n.distance:
            print(m.queryIdx, m.trainIdx, sep=' ')
            position =[kp1[m.queryIdx].pt[i]-kp2[m.trainIdx].pt[i] for i in range(2)]
            good_matches.append([m])
    position =[int(position[i]/len(good_matches)) for i in range(2)]
    return position