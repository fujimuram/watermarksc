# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 16:53:54 2022

@author: matsunaga
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from my_packege import tool



savepath = 'Watermark\\overlap'
os.makedirs(savepath, exist_ok=True)

w = cv2.imread('Watermark\\16_16\\0.png',cv2.IMREAD_GRAYSCALE)
_, ws = tool.get_image('Watermark\\16_16\\', '*.png')

#w_big = make_watermark(w, [64, 64], [22, 36])
#plt.imshow(w_big, vmin=0, vmax=255, cmap = 'gray')

w_bigs = tool.make_watermarks(ws, [64, 64])
for i in range(16):
    cv2.imwrite(str(i) + '.png', w_bigs[i])
    plt.figure(1)
    plt.subplot(4,4,i+1)
    plt.imshow(w_bigs[i], vmin=0, vmax=255, cmap = 'gray')




"""
w_big = np.ones([4, 4, w.shape[0], w.shape[1]], np.uint8) * 255
w_zeros = np.ones(w.shape, np.uint8) * 255
ws = [None] * 16


for i in range(4):
    for j in range(4):
        w_big[i,j] = w 


for h in range(0,16):
    for i in range(4):
        for j in range(4):
            if h > 4 * i + j:
                w_big[i,j] = w_zeros 
    tmp = np.concatenate(np.concatenate(w_big,1),1)
    ws[h] = tmp
    
    filename = str(h) + '.png'
    cv2.imwrite(savepath + '\\' + filename, ws[h])

for i in range(16):
    plt.figure(1)
    plt.subplot(4,4,i+1)
    plt.imshow(ws[i], vmin=0, vmax=255, cmap = 'gray')
    
"""



    
    

    
    
