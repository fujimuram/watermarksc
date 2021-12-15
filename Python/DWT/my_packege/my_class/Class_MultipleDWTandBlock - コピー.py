# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:19:14 2021

@author: 松永健彦
何か困ったことがあったとき、連絡くれたら対応します。多分。

参考サイト
https://www.jstage.jst.go.jp/article/itej1997/52/12/52_12_1832/_pdf/-char/ja

最小構成を意識した
"""

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP
import skimage.util
import copy
import os
import glob
from natsort import natsorted
import sys

sys.path.append('../../')

from my_packege import tool
from my_packege.watermark import MultipleDWTandBlock as mdb


class MultipleDWTandBlock:
    
    #元画像
    img_original = None
    #何回DWTするか
    N = None
    #埋め込み強度を表すらしい。詳しくは埋め込みor抽出の実装を見て考えてください。もしくはwebで
    Q = None
    #N回DWTして得られるLL_Nに対して行うブロック分割の際の１ブロックの大きさ
    block_size = None
    watermark_size = None
    #透かし画像を埋めるのに必要な元画像の最低限のサイズ。透かし画像のサイズとDWTの回数Nとblock_sizeに依存する
    required_capacity = None
    divisions_size = None
    
    def __init__(self, img_shape, N, Q, block_size, watermark_shape):
        
        if N < 2:
            print("クラス製作者からの警告 : Nが１以下のときは絶対バグるから考え直して")
        
        self.N = N
        self.Q = Q
        self.block_size = block_size
        self.watermark_size = np.array(watermark_shape)
        self.required_capacity = np.array(watermark_shape) * np.array(block_size) ** N
        self.divisions_size = self.how_many_divisions(img_shape, self.required_capacity)
        
    
    def how_many_divisions(self, img_size, required_capacity):
    
        img_size = np.array(img_size)
        required_capacity = np.array(required_capacity)
        
        return img_size // required_capacity
    
    
    def emb(self, img, watermark):
        
        img_, LM_ = mdb.emb(img, watermark, self.Q, self.block_size, self.N)
        
        return img_, LM_
    
    def extract(self, img, LM_):
        
        w_ = mdb.extract(img, self.Q, self.block_size, LM_, self.N)
        
        return w_
        
        

img_original = cv2.imread('../../image/Man.bmp' , cv2.IMREAD_GRAYSCALE)
watermark = cv2.imread('../../Watermark/16_16/0.png',cv2.IMREAD_GRAYSCALE)

wdtn0 = MultipleDWTandBlock(img_original.shape, 3, 1, (2,2), watermark.shape)

embed_img , LM_ = wdtn0.emb(img_original, watermark)
extract_w = wdtn0.extract(embed_img, LM_)

plt.figure(1)
plt.subplot(221)
plt.imshow(img_original, cmap = 'gray')
plt.subplot(222)
plt.imshow(embed_img , cmap = 'gray')
plt.subplot(224)
plt.imshow(extract_w , cmap = 'gray')

