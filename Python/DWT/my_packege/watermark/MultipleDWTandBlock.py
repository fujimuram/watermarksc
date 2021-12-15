# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:30:08 2021

@author: matsunaga
"""

import numpy as np
import math
from decimal import Decimal, ROUND_HALF_UP
import skimage.util
import copy
import sys
sys.path.append('../../')

from my_packege import tool 




def embed(img, w, Q, block_size, n, point = [0,0]):
    
    point_cor = point // what_size_need(n, block_size)
    
    #w = np.ravel(w)
    
    coefficients = tool.dwt_haarN(img, n)
    
    embed_block = skimage.util.view_as_blocks(copy.deepcopy(coefficients[n-1][0]), tuple(block_size))
    
    
    """ ループの回数を透かし画像wと埋め込み領域embed_blockのどちらか小さいほうに設定する """
    loop_num0 = embed_block.shape[0]
    loop_num1 = embed_block.shape[1]
    
    if loop_num0 > w.shape[0]:
        loop_num0 = w.shape[0]
        
    if loop_num1 > w.shape[1]:
        loop_num1 = w.shape[1]
        
        
    """ 埋め込み開始 """
    for i in range(loop_num0):
        for j in range(loop_num1):
            
            #print("\n")

            i_cor = i + point_cor[0]
            j_cor = j + point_cor[1]
            
            
            #print("i = ", i, "j = ", j)
            #print("i = ", i, "j = ", j)
            #imgの平均を計算する
            mean = sum(sum( embed_block[i_cor][j_cor] )) / embed_block[i_cor][j_cor].size
            
            """
            (1)
            """
            #diff = 四捨五入した(平均/Q) と　切り捨て(平均/Q)　の差
            #print("mean = ", mean)
            q = int(Decimal(str(mean / Q)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
            
            """
            (2)
            """
            diff = q - math.floor(mean / Q)
            
            #print(diff, q, math.floor(mean) , mean)
            #print(diff, '{:3} {:3}'.format(q,math.floor(mean)), mean)
            
            """
            (3)
            """
            #埋め込み作業。詳しくはwebで（上記)
            #ごみpythonがくそみたいなエラーをはきやがるので、仕方なくここにもqを定義しておく
            #print("q = ", q, "w[", i, ",", j,  "] = ", w[i][j])
            _q = q
            if ( (w[i][j] == 0 and q % 2 == 1) or (w[i][j] == 255 and q % 2 == 0) ):
                #print("値を変える")
                
                if diff == 0:
                    _q = q + 1
                elif diff == 1:
                    _q = q - 1
                else:
                    _q = q
            """
            (4)
            """
            #この辺は参考にした論文通りにやってるが、もっと効率のいい方法がある気がした。
            _mean = _q * Q
            diff_M = _mean - mean
            
            #print(diff_M)
            #print(embed_block[i,j,0,0:3])
            embed_block[i_cor][j_cor] += diff_M
            #print(embed_block[i,j,0,0:3])
            
            
    coefficients[n-1][0] = np.concatenate(np.concatenate(embed_block, 1), 1)
    
    """ LMを画像全体で出す（上）か埋め込んだ場所だけで出すか（下） """
    tmp = coefficients[n-1,0][point_cor[0]:point_cor[0]+w.shape[0],point_cor[1]:point_cor[1]+w.shape[1]]
    LM1 = sum(sum(tmp)) / tmp.size
    LM2 = sum(sum( coefficients[n-1,0] )) / coefficients[n-1,0].size
    #print(_LM)
    
    
    img_ = tool.inv_dwt_haarN(coefficients)
    
    img_ = np.round(img_)
    img_ = img_.clip(0, 255)
    img_ = img_.astype(np.uint8)
    
    return img_, LM1




def extract(img, Q, block_size, LM_, n):
    
    """
    (1)
    """
    coefficients = tool.dwt_haarN(img, n)
    
    """
    (2)
    """
    LM = sum(sum( coefficients[n-1][0]) ) / coefficients[n-1][0].size
    diff_mean = LM - LM_
    diff_mean = 0
    
    #print(diff_mean)
    
    """
    (3)
    (4)
    """
    embed_block = skimage.util.view_as_blocks(copy.deepcopy(coefficients[n-1][0]), tuple(block_size))
    w = np.zeros(embed_block.shape[0:2], np.uint8)

    for i in range(embed_block.shape[0]):
        for j in range(embed_block.shape[1]):
            
            
            block_mean = sum(sum( embed_block[i][j] )) /  embed_block[i][j].size
            S = (block_mean - diff_mean) / Q
            
            
            S = int(Decimal( str(S) ).quantize( Decimal('0'), rounding=ROUND_HALF_UP) )
            #print(S)
            if S % 2 == 0:
                w[i,j] = 0
            else :
                w[i,j] = 255
                
    return w

def what_size_need(n, block_size):
    
    bs = np.array(block_size)
    
    return (2**n) * bs

"""
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

img = cv2.imread('../../image/Man.bmp' , cv2.IMREAD_GRAYSCALE)
w = cv2.imread('../../Watermark/16_16/0.png',cv2.IMREAD_GRAYSCALE)

Q = 1
block_size = [2,2]
n = 3

embed_img , LM_ = embed(img, w, Q, block_size, n, [0,0])
extract_w =  extract(embed_img, Q, block_size, LM_, n)



aaa =  what_size_need(n, block_size)

plt.figure(1)
plt.subplot(221)
plt.imshow(img, cmap = 'gray')
plt.subplot(222)
plt.imshow(embed_img , cmap = 'gray')
plt.subplot(224)
plt.imshow(extract_w , cmap = 'gray')
"""

