# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 03:38:13 2022

@author: takehiko
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 01:36:06 2022

@author: takehiko
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import re
import os
import skimage.util
import csv
import time

from my_packege import tool
from my_packege.watermark import MultipleDWTandBlock as mdbo
from my_packege.watermark import MyMultipleDWTandBlock2 as mdb

img_original = cv2.imread('image/Man.bmp' , cv2.IMREAD_GRAYSCALE)
img_original = img_original.clip(12,243)
#img_original = img_original.clip(55,200)
#w = cv2.imread('../../Watermark/16_16/0.png',cv2.IMREAD_GRAYSCALE)
w = cv2.imread('Watermark/watermark1.png',cv2.IMREAD_GRAYSCALE)

""" 埋め込みの設定 """
N = 3
Q = 1
block_size = (2, 2)
target = [0.1, 0.6]
modulo = 1


img_originalMy = mdb.preembed(img_original, block_size, N, target, modulo)
img_originalO = mdbo.preembed(img_original, Q, block_size, N)

start_timeEmbM = time.perf_counter()
img_embmy = mdb.embed(img_originalMy, w, Q, block_size, N, point = [0,0])
diff = mdb.calcdiff(img_embmy, img_embmy)
end_timeEmbM = time.perf_counter()

start_timeExtM = time.perf_counter()
img_extm, _ = mdb.extract(img_embmy, Q, block_size, N, diff)
end_timeExtM = time.perf_counter()


start_timeEmbO = time.perf_counter()
img_embo, LM = mdbo.embed(img_originalO, w, Q, block_size, N, point = [0,0])
end_timeEmbO = time.perf_counter()

start_timeExtO = time.perf_counter()
img_exto =  mdbo.extract(img_embo, Q, block_size, N, LM)
end_timeExtO = time.perf_counter()
    

elapsed_timeEmbM = end_timeEmbM - start_timeEmbM
elapsed_timeExtM = end_timeExtM - start_timeExtM
elapsed_timeEmbO = end_timeEmbO - start_timeEmbO
elapsed_timeExtO = end_timeExtO - start_timeExtO
print()

"""
x = [x for x in range(1,Q)]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, PSNRMy[1:])

ax.set_xlabel('埋め込み強度Q')
ax.set_ylabel('PSNR(db)')
#ax.set_ylim(0, 2)
plt.show()
"""










#cv2.imwrite('my.bmp', img_emb)


plt.figure(1)
plt.subplot(231)
plt.imshow(img_original, vmin=0, vmax=255, cmap = 'gray')
plt.subplot(232)
plt.imshow(img_embmy, vmin=0, vmax=255, cmap = 'gray')
plt.subplot(233)
plt.imshow(img_extm , vmin=0, vmax=255, cmap = 'gray')

plt.subplot(234)
plt.imshow(img_original, vmin=0, vmax=255, cmap = 'gray')
plt.subplot(235)
plt.imshow(img_embo, vmin=0, vmax=255, cmap = 'gray')
plt.subplot(236)
plt.imshow(img_exto , vmin=0, vmax=255, cmap = 'gray')
