# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:54:20 2021

@author: matsunaga
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:41:04 2021

@author: matsunaga
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import os
import skimage.util
from decimal import Decimal, ROUND_HALF_UP
import copy

from my_packege import tool 
from my_packege.watermark import MultipleDWTandBlock as mdb
from my_packege.watermark import MyMultipleDWTandBlock as mmdb
import pywt

img_name = 'Man.bmp'
#img_name = 'Honeyview_image-name-ec.png'
#img_name = 'Honeyview_image-name-ec (4).png'
img_original = cv2.imread('image\\' + img_name , cv2.IMREAD_GRAYSCALE)
img_original = img_original.clip(10,235)

#watermarks = tool.get_image('Watermark\\o\\', '*.png')
watermarks = tool.get_image('Watermark\\16_16\\', '*.png')

""" 埋め込みの設定 """
N = 3
Q = 9
BLOCK_SIZE = (2, 2)



img_z = mmdb.preembed(img_original, BLOCK_SIZE, N)
#img_z = img_original
"""
psnr = np.array([None] * len(watermarks[0]) * Q).reshape([Q,len(watermarks[0])])
for q in np.arange(0, Q, 1):
    print("q =", q)
    for i in range(len(watermarks[0])):  
        img_embed = mdb.embed_bad(img_z, watermarks[i][2], q, BLOCK_SIZE, N, point = [0,0])
        psnr[q][i] = cv2.PSNR(img_original, img_embed)

#img_extract0 = mdb.extract(img_embed2, Q, BLOCK_SIZE, N)
psnr_mean = np.sum(psnr,1) / psnr.shape[1]


x = np.arange(0, Q, 1)
plt.scatter(x, psnr_mean, marker = '.')
plt.grid(True)
plt.show()
"""

img_embed, LM = mmdb.embed(img_z, watermarks[1][0], Q, BLOCK_SIZE, N, point = [768,768])
img_ext = mmdb.extract(img_embed, Q, BLOCK_SIZE, N, LM)

print("img_original's mean", np.sum(np.sum(img_original)) / img_original.size )
print("img_z's mean", np.sum(np.sum(img_z)) / img_z.size )
print("img_embed's mean", np.sum(np.sum(img_embed)) / img_embed.size )

plt.figure(1)
plt.subplot(221)
plt.imshow(img_original, vmin=0, vmax=255, cmap = 'gray')
plt.subplot(222)
plt.imshow(img_z, vmin=0, vmax=255, cmap = 'gray')
plt.subplot(223)
plt.imshow(img_embed, vmin=0, vmax=255, cmap = 'gray')
plt.subplot(224)
plt.imshow(img_ext, vmin=0, vmax=255, cmap = 'gray')

cv2.imwrite('man.bmp', img_embed )

