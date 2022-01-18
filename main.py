from skimage.io.manage_plugins import reset_plugins
from blending import create_mask, poisson_blend
import positioning
import cv2
import numpy as np
from skimage import io
from scipy.sparse.linalg import spsolve
#import pyamg
import matplotlib.pyplot as plt
mask_path = './input/1mask.jpg'
src_path='./input/1src.jpg'
dst_path = './input/1dst.jpg'
result_path = './output/1.jpg'

if __name__ == "__main__":
    img_mask = io.imread(mask_path, as_gray=True)
    img_src = io.imread(src_path).astype(np.float64)
    img_target = io.imread(dst_path)

    position = positioning.positioning(dst_path, src_path, mask_path)
    offset = (position[0], position[1])
    print("estimated shift of position:", offset)    
    blend_mask, img_src, offset_adj \
        = create_mask(img_mask.astype(np.float64),
                      img_target, img_src, offset=offset)

    img_pro = poisson_blend(blend_mask, img_src, img_target,
                            method='normal', offset_adj=offset_adj)



    def BoundaryPro(img_mask, img_pro):
        kernel = np.ones((10, 10), np.uint8)
        img_mask1 = cv2.dilate(img_mask, kernel, iterations = 1)
        img_mask2 = cv2.erode(img_mask, kernel, iterations = 1)

        cv2.absdiff(img_mask1, img_mask2, img_mask)
        mask = img_mask.astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        img_pro = cv2.addWeighted(img_pro, 1, mask, 0.1, 0)
        return img_pro

    img_pro=BoundaryPro(img_mask, img_pro)
    plt.imshow(img_pro)
    plt.show()
    #io.imsave(result_path, img_pro)