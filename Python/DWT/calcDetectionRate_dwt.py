# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 16:28:25 2022

@author: matsunaga
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 16:05:36 2022

@author: matsunaga
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 01:49:05 2022

@author: takehiko
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:04:50 2022

@author: matsunaga
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:33:13 2021

@author: matsunaga

平均値攻撃を行うプログラム
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import re
import os
import skimage.util
import csv
import inspect
local_vars= {}

from my_packege import tool
#from my_packege.my_class.ClassMultipleDWTandBlock import ClassMultipleDWTandBlock as CMDB
#from my_packege.my_class.ClassMultipleDWTandBlock import ClassMultipleDWTandBlock as CMDB
import watermarkDWT as WDWT

def calc_failrate(img_original_pre, ws, Q):
    
    """" 埋め込みに使うインスタンスを作成 """
    #print(cmdb.status())
    white = np.ones(ws[0].shape) * 255
    img_original = WDWT.emb_DWT(img_original_pre, white, 1, Start = [0,0])
    
    
    """ 埋め込み&保存 ------------------------------------------------------------------------------------------------------"""
    
    img_embs = [None] * len(ws)
    extract_imgs = [None] * len(ws)
    LM = [None] * len(ws)
    psnr = [None for x in range(len(ws))]
    emb_block = 0
    for i in range(len(ws)):
        #print(i)
         
        img_embs[i] = WDWT.emb_DWT(img_original, ws[i], Q, Start = [0,0])
        
        psnr[i] = cv2.PSNR(img_original_pre, img_embs[i])
        #diff = cmdb.calcdiff(img_embs[i], img_embs[i])
        #extract_imgs[i], coe = cmdb.extract(img_embs[i], diff)
        extract_imgs[i] = WDWT.ext_DWT(img_embs[i])
        
    psnr_mean = sum(psnr) / len(psnr)
    
    """ 平均値攻撃 ------------------------------------------------------------------------------------------------------"""
    
    """ 前準備 """
    img_sum = np.zeros( img_embs[0].shape, dtype = float)
    w_mean = np.ones( ws[0].shape, dtype = np.uint8) * 255
    fail_rate = np.ones( len(ws), dtype = np.float32) * -1
    fail_block_rate = np.ones( len(ws), dtype = np.float32) * -1
    fail_white_rate = np.ones( len(ws), dtype = np.float32) * -1
    for i in range(len(img_embs)):  
        """ 
        平均値攻撃の実行
        img_meanにはimg_embsの0からi番目までの平均が入る。
        よってi番目のループではi枚の平均値攻撃が行われる。
        """
        img_sum += img_embs[i]
        img_mean = img_sum / (i + 1)
        img_mean = np.round(np.clip(img_mean,0,255)).astype('uint8')
        
        #diff = cmdb.calcdiff(img_mean, img_embs[i])
        w_mean_ext = WDWT.ext_DWT(img_mean)
        
        """
        w_mean_extが平均値攻撃で作成された画像から抽出された透かし情報。
        w_meanは抽出したかった透かし情報。もしくは平均値攻撃に使われた透かし入り画像に埋め込まれた透かし情報の黒の部分を合わせた物とも言える。
        w - w_ext = ?
          0 -   0 =   0 ......正確
          0 - 255 =   1 ......黒を白と検出した
        255 -   0 = 255 ......白を黒と検出した
        255 - 255 =   0 ......正確
        であることを利用している。
        
        BlackPixcelSumを毎回計算するのは無駄だけど面倒だから変えない
        """
        w_mean += ws[i] + 1 
        w_mean = w_mean.astype(np.uint8)

        fail_rate[i], fail_block_rate[i], fail_white_rate[i] = tool.calc_failrate(w_mean_ext, w_mean)
        plt.imshow(w_mean_ext, vmin=0, vmax=255, cmap = 'gray')
        print(i)
        print()
        """
        plt.figure(1)
        plt.subplot(4,4,i+1)
        plt.imshow(w_mean_ext, vmin=0, vmax=255, cmap = 'gray')     
        """      
    """ 平均値攻撃。ここまで------------------------------------------------------------------------------------------------------ """
    
    
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(img_original, cmap = 'gray')
    plt.subplot(222)
    plt.imshow(img_embs[i] , cmap = 'gray')
    plt.subplot(224)
    plt.imshow(extract_imgs[i] , cmap = 'gray')
    
    return fail_rate, fail_block_rate, fail_white_rate, psnr_mean

def main():
    global local_vars
    
    """ 準備 """
    img_name = 'LENNA512.tiff'
    img_original_ = cv2.imread('image/' + img_name, cv2.IMREAD_GRAYSCALE)
    img_original_pre = cv2.imread('image\\' + img_name , cv2.IMREAD_GRAYSCALE)
    img_original_pre = img_original_pre.clip(12,243)
    
    watermarks = tool.get_image('Watermark\\64_64\\', '*.png')
    ws = tool.make_watermarks(watermarks[1], [256,256])
    
    for i in range(len(ws)):
        plt.figure(2)
        plt.subplot(4,4,i+1)
        plt.imshow(ws[i], vmin=0, vmax=255, cmap = 'gray')     
    
    fail_rate = []
    fail_block_rate = []
    fail_white_rate = []
    psnrs = []
    
    """ 埋め込みの設定 """
    #N = 3
    Q = 10
    #BLOCK_SIZE = (2, 2)
    
    #target = [0.1, 0.6]
    #modulo = 1
    for q in range(1,Q):
        print(q, (q/Q) * 100, "%")
        tmp_fail_rate, tmp_fail_block_rate, tmp_fail_white_rate, tmp_psnr = calc_failrate(img_original_pre, ws, q)
        fail_rate.append(tmp_fail_rate * 100)
        fail_block_rate.append((1 - tmp_fail_block_rate) * 100)
        fail_white_rate.append(tmp_fail_white_rate * 100)
        psnrs.append(tmp_psnr)
        
    np.savetxt("BlackDetectionRate_dwt.csv", fail_block_rate, delimiter=",", fmt="%d")
    
    f = open('PSNR_dwt.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(psnrs)
    f.close()
    
    local_vars= inspect.currentframe().f_locals
if __name__ == "__main__":
    main()
