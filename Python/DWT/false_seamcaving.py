# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 14:21:38 2021

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
from pathlib import Path
import re
import os
import skimage.util
from decimal import Decimal, ROUND_HALF_UP
import copy

from my_packege import tool 
from my_packege.watermark import MultipleDWTandBlock as mdb
import pywt

img_name = 'Man.bmp'
#img_name = 'Honeyview_image-name-ec.png'
#img_name = 'Honeyview_image-name-ec (4).png'
img_original = cv2.imread('image\\' + img_name , cv2.IMREAD_GRAYSCALE)
img_original = img_original.clip(1,244)
watermark = cv2.imread('Watermark\\255.png',cv2.IMREAD_GRAYSCALE)
watermark0 = cv2.imread('Watermark\\watermark1.png',cv2.IMREAD_GRAYSCALE)
watermark1 = cv2.imread('Watermark\\watermark2.png',cv2.IMREAD_GRAYSCALE)


watermarks = tool.get_image('Watermark\\16_16\\', '*.png')

""" 埋め込みの設定 """
N = 3
Q = 1
BLOCK_SIZE = (2, 2) 

#ループの回数
kaisuu = 16


#watermark0 = watermark
#watermark1 = watermark

""" 元画像にあらかじめ白か黒を埋め込んでおく """
img_original, a = mdb.embed(img_original, np.zeros(watermark0.shape), Q, BLOCK_SIZE, N, point = [0,0])


"""
img_embed0, LM0 = mdb.embed(img_original, watermark0, Q, BLOCK_SIZE, N, point = [0,0])
img_embed1, LM1 = mdb.embed(img_original, watermark1, Q, BLOCK_SIZE, N, point = [0,0])
img_embed2, LM1 = mdb.embed(img_original, watermarks[1][2], Q, BLOCK_SIZE, N, point = [0,0])
"""



""" 埋め込み----------------------------------------------------------------------------------------------------------------------- """

""" 変数の初期化 """
imgs_emb = [None] * len(watermarks[0])
LM = [None] * len(watermarks[0])
diffs = [None] * len(watermarks[0])

for i in range(kaisuu):
    
    """ z,z2は元画像をいくつずらすか """
    z = (i % 4) * 4
    z2 = 4 * (i // 4)
    print(i, z, z2)
    
    imgs_emb[i], LM[i] = mdb.embed(img_original[z2:z2+256,z:z+256], watermarks[1][0], Q, BLOCK_SIZE, N, point = [0,0])
    #diffs[i] = img_original.astype(np.int16) - imgs_emb[i].astype(np.int16)
    #print(sum(sum(diffs[i])))
    plt.figure(3)
    plt.subplot(4,4,i+1)
    plt.imshow(imgs_emb[i], vmin=0, vmax=255, cmap = 'gray')
    

""" 平均値攻撃---------------------------------------------------------------------------------------------------------------------------  """
    
""" 変数の初期化 """
img_sum = np.zeros( imgs_emb[0].shape, dtype = float)
imgs_ext = [None] * len(watermarks[0])
imgs_mean_ext = [None] * len(watermarks[0])

for i in range(kaisuu):
    img_sum += imgs_emb[i]
    img_mean = img_sum / (i + 1)
    img_mean = np.round(np.clip(img_mean,0,255)).astype('uint8')
    
    imgs_ext[i] = mdb.extract(imgs_emb[i], Q, BLOCK_SIZE, LM[i], N)
    imgs_mean_ext[i] = mdb.extract(img_mean, Q, BLOCK_SIZE, LM[i], N)
    
    
    plt.figure(1)
    plt.subplot(4,4,i+1)
    plt.imshow(imgs_ext[i][0:30,0:30], vmin=0, vmax=255, cmap = 'gray')
    
    plt.figure(2)
    plt.subplot(4,4,i+1)
    plt.imshow(imgs_mean_ext[i][0:30,0:30], vmin=0, vmax=255, cmap = 'gray')
    
    
    
"""
imgs_emb = [None] * len(watermarks[0])
LM = [None] * len(watermarks[0])
diffs = [None] * len(watermarks[0])

for i in range(kaisuu):
    
    z = (i % 4) * 4
    z2 = 4 * (i // 4)
    print(i, z, z2)
    
    imgs_emb[i] = copy.deepcopy(img_original)
    imgs_emb[i][z2:z2+256,z:z+256], LM[i] = mdb.embed(img_original[z2:z2+256,z:z+256], watermarks[1][0], Q, BLOCK_SIZE, N, point = [0,0])
    diffs[i] = img_original.astype(np.int16) - imgs_emb[i].astype(np.int16)
    print(sum(sum(diffs[i])))
  

img_sum = np.zeros( imgs_emb[0].shape, dtype = float)
imgs_ext = [None] * len(watermarks[0])
imgs_mean_ext = [None] * len(watermarks[0])

for i in range(kaisuu):
    img_sum += imgs_emb[i]
    img_mean = img_sum / (i + 1)
    img_mean = np.round(np.clip(img_mean,0,255)).astype('uint8')
    
    imgs_ext[i] = mdb.extract(imgs_emb[i], Q, BLOCK_SIZE, LM[i], N)
    imgs_mean_ext[i] = mdb.extract(img_mean, Q, BLOCK_SIZE, LM[i], N)
    
    
    plt.figure(1)
    plt.subplot(4,4,i+1)
    plt.imshow(imgs_ext[i][0:30,0:30], vmin=0, vmax=255, cmap = 'gray')
    
    plt.figure(2)
    plt.subplot(4,4,i+1)
    plt.imshow(imgs_mean_ext[i][0:30,0:30], vmin=0, vmax=255, cmap = 'gray')
    
"""

"""
img_extract0 = mdb.extract(img_embed0, Q, BLOCK_SIZE, LM0, N)
img_extract1 = mdb.extract(img_embed1, Q, BLOCK_SIZE, LM1, N)
"""


#diff = img_original.astype(np.int16) - img_embed.astype(np.int16)
#diff_ext  = watermark.astype(np.float64) - img_extract
#print('diff_ext の合計 = ', sum(sum(diff_ext)))


"""
plt.figure(1)
plt.subplot(221)
plt.imshow(watermark, vmin=0, vmax=255, cmap = 'gray')
plt.subplot(222)
plt.imshow(img_extract0, vmin=0, vmax=255, cmap = 'gray')
plt.subplot(224)
plt.imshow(img_extract1, vmin=0, vmax=255, cmap = 'gray')
"""
#cv2.imwrite('man_leveling.png', img_embed)

