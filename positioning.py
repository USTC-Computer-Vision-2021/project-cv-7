import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def positioning(dst_path, src_path, mask):
    img1 = cv.imread(dst_path)
    img2 = cv.imread(src_path)
    img_mask = cv.imread(mask)
    kernel = np.ones((100, 100), np.uint8)
    #img_mask = cv.dilate(img_mask, kernel, iterations = 1)
    img_mask = cv.bitwise_not(img_mask)

    akaze = cv.AKAZE_create()

    kp1, des1 = akaze.detectAndCompute(img1, img_mask)
    kp2, des2 = akaze.detectAndCompute(img2, img_mask)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    shifts = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            p =[kp1[m.queryIdx].pt[i]-kp2[m.trainIdx].pt[i] for i in range(2)]
            shifts.append(p)
            good_matches.append([m])

    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite('./data/matches.jpg', img3)
    shifts = abs(np.array(shifts))

    shift =[int(min(shifts[:,1])), int(min(shifts[:,0]))]
    plt.imshow(img3),plt.colorbar(),plt.show()

    return shift

if __name__ == "__main__":
    mask_path = './input/3dst.jpg'
    src_path='./input/3src.jpg'
    dst_path = './input/3dst.jpg'
    p = positioning(dst_path, src_path, mask_path)
    print(p)