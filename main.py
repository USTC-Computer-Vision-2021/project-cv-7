from blending import create_mask, poisson_blend
import positioning
import cv2
import numpy as np
from skimage import io
from scipy.sparse.linalg import spsolve
#import pyamg
import matplotlib.pyplot as plt

if __name__ == "__main__":
    offset = (56,19)     #(135, 35) # (605,400) ∂‘”¶ (700,550)
    img_mask = io.imread('./data/mask.jpg', as_gray=True)
    img_src = io.imread('./data/milanS.jpg').astype(np.float64)
    img_target = io.imread('./data/milanDst.jpg')

    img_mask, img_src, offset_adj \
        = create_mask(img_mask.astype(np.float64),
                      img_target, img_src, offset=offset)

    img_pro = poisson_blend(img_mask, img_src, img_target,
                            method='normal', offset_adj=offset_adj)
    plt.imshow(img_pro)
    plt.show()
    io.imsave('./data/normalclone.png', img_pro)