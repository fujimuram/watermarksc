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

from my_packege import tool
from my_packege.watermark import MultipleDWTandBlock as mdbo
from my_packege.watermark import MyMultipleDWTandBlock2 as mdb

img_name = 'man.bmp'
img_original = cv2.imread('image\\' + img_name , cv2.IMREAD_GRAYSCALE)
#img_original = cv2.imread('image/Man.bmp' , cv2.IMREAD_GRAYSCALE)
img_original = img_original.clip(11,244)
#img_original = img_original.clip(55,200)
#w = cv2.imread('../../Watermark/16_16/0.png',cv2.IMREAD_GRAYSCALE)
w = cv2.imread('Watermark/watermark1.png',cv2.IMREAD_GRAYSCALE)

""" 埋め込みの設定 """
N = 3
Q = 10
block_size = (2, 2)
target = [0.25, 0.75]
modulo = 1


img_originalMy = mdb.preembed(img_original, block_size, N, target, modulo)
#w = np.zeros(w.shape)

PSNRMy = [None] * Q
PSNRO = [None] * Q
for q in range(1,Q):
    print(q, (q / Q) * 100, "%")
    img_embmy = mdb.embed(img_originalMy, w, q, block_size, N, point = [0,0])
    diff = mdb.calcdiff(img_embmy, img_embmy)
    img_extm, _ = mdb.extract(img_embmy, q, block_size, N, diff)
    PSNRMy[q] = cv2.PSNR(img_original, img_embmy)
    
    img_originalO = mdbo.preembed(img_original, q, block_size, N)
    img_embo, LM = mdbo.embed(img_originalO, w, q, block_size, N, point = [0,0])
    img_exto =  mdbo.extract(img_embo,q, block_size, N, LM)
    PSNRO[q] = cv2.PSNR(img_original, img_embo)
    

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


plt.figure(2)
x = np.arange(1, Q, 1)
plt.scatter(x, PSNRMy[1:], marker = '.')
plt.scatter(x, PSNRO[1:], marker = '.')
plt.grid(True)
plt.show()





f = open('_PSNR_myemb2.csv', 'w')
writer = csv.writer(f)
writer.writerow(PSNRMy)
f.close()

f = open('_PSNR_Original.csv', 'w')
writer = csv.writer(f)
writer.writerow(PSNRO)
f.close()


cv2.imwrite('BlackOnlyOriginal' + str(q) + 'PSNR' + str(PSNRO[q]) + img_name, img_embo)
cv2.imwrite('BlackOnlyMyemb' + str(q) + 'PSNR' + str(PSNRMy[q]) + img_name, img_embmy)

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
