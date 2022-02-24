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

from my_packege import tool
#from my_packege.my_class.ClassMultipleDWTandBlock import ClassMultipleDWTandBlock as CMDB
from my_packege.my_class.ClassMyMultipleDWTandBlock2 import ClassMultipleDWTandBlock as CMDB

""" 準備 """
savepath_pre = "result\\main3\\overlap\\"

img_name = 'man.bmp'
#img_name = 'man_leveling.png'
#img_name = 'Honeyview_image-name-ec.png'
#img_name = 'Honeyview_image-name-ec (4).png'
img_original_ = cv2.imread('image/man.bmp', cv2.IMREAD_GRAYSCALE)
img_original_pre = cv2.imread('image\\' + img_name , cv2.IMREAD_GRAYSCALE)
img_original_pre = img_original_pre.clip(10,245)
watermark = cv2.imread('Watermark\\16_16\\0.png',cv2.IMREAD_GRAYSCALE)
watermarks = tool.get_image('Watermark\\overlap\\', '*.png')

""" 埋め込みの設定 """
N = 3
Q = 1
BLOCK_SIZE = (2, 2)

target = [0.25, 0.75]
modulo = 1

diff_block = [4, 4]

""" 埋め込みにどの透かし画像から始めるかを決める。※watermaeks[0]の順番から。 """
start_num_embed = 0
""" 平均値攻撃を行う際の最初の場所の設定 """
start_num = 0

"""" 埋め込みに使うインスタンスを作成 """
cmdb = CMDB(N, Q, BLOCK_SIZE, target, modulo)

img_original = cmdb.preembed(img_original_pre)


""" 埋め込み&保存 ------------------------------------------------------------------------------------------------------"""

"""" ファイル関係 """

filename_status = 'N' + str(N).zfill(2) + 'Q' + str(Q).zfill(2) \
    + 'BS' + str(BLOCK_SIZE[0]).zfill(2) + 'x' + str(BLOCK_SIZE[1]).zfill(2)
savepass = savepath_pre + Path(img_name).stem + '\\' + Path(img_name).stem + ',' + cmdb.status()
savepass = re.sub(r"\s", "", savepass)
savepass = re.sub("\|", ",", savepass)
os.makedirs(savepass, exist_ok=True)
os.makedirs(savepass + '\\ext', exist_ok=True)

block_number =  img_original.shape // cmdb.need_size // watermark.shape
img_embs = [None] * len(watermarks[0])
extract_imgs = [None] * len(watermarks[0])

psnr = [None] * len(watermarks[0])

plt.figure(5)
for i in range(block_number[0] * block_number[1]):
    
    """ 用意した透かし画像の枚数が埋め込み可能名ブロック数より少ない場合に必要な処理 """
    if i >= len(watermarks[0]):
        break
    #print(i)
    
    """ １次元と２次元表現の橋渡し的なやつ """
    row = i // block_number[0]
    col = i % block_number[0]
    #emb_block = cmdb.need_size * watermark.shape * [row,col]
    emb_block = [0,0]
    i_shift = (i + start_num_embed) % len(watermarks[0])
    
    img_embs[i] = cmdb.embed(img_original, watermarks[1][i_shift], emb_block)
    
    psnr[i] = cv2.PSNR(img_original_, img_embs[i])
    diff = cmdb.calcdiff(img_embs[i], img_embs[i])
    extract_imgs[i], _ = cmdb.extract(img_embs[i], diff)
    
    filename = Path(img_name).stem + ',W' + Path(watermarks[0][i_shift].split('\\')[-1]).stem.zfill(3) + \
        '(' + str(row) + ',' + str(col) + ')' + filename_status + '.png'
    cv2.imwrite(savepass + '\\' + filename, img_embs[i])
    cv2.imwrite(savepass + '\\ext\\ext_[' + str(i) + ',' + str(i_shift) + ']' +  filename, extract_imgs[i])
    #print('!')
    
    
    plt.subplot(4,4,i+1)
    plt.imshow(extract_imgs[i], vmin=0, vmax=255, cmap = 'gray')
    #print(i, i_shift)
    
    
    
    
""" 平均値攻撃 ------------------------------------------------------------------------------------------------------"""

""" 前準備 """
img_sum = np.zeros( img_embs[0].shape, dtype = float)

filename_pre = 'ExtractWatermakAfterMeanAttack' 
filename_status = 'N' + str(N).zfill(2) + 'Q' + str(Q).zfill(2) \
    + 'BS' + str(BLOCK_SIZE[0]).zfill(2) + 'x' + str(BLOCK_SIZE[1]).zfill(2)
savepass = savepath_pre + Path(img_name).stem + '\\' + Path(img_name).stem + ',' + cmdb.status()
savepass = re.sub(r"\s", "", savepass)
savepass = re.sub("\|", ",", savepass)
savepass = savepass + '\\' + 'ExtractWatermakAfterMeanAttack'  
filename_middle  = '[' + str(start_num) + ','+ str(start_num_embed) + ']_' + Path(img_name).stem
os.makedirs(savepass, exist_ok=True)

plt.figure(1)


diffh = [None] * 16

#print("\n\n抽出")
for i in range(len(img_embs)):  

    i_pre = i
    i = (i + start_num) % len(img_embs)
    #print("i = ", i)
    
    """ 
    平均値攻撃の実行
    img_meanにはimg_embsの0からi番目までの平均が入る。
    よってi番目のループではi枚の平均値攻撃が行われる。
    """
    img_sum += img_embs[i]
    img_mean = img_sum / (i_pre + 1)
    img_mean = np.round(np.clip(img_mean,0,255)).astype('uint8')
     
    
    """ 
    抽出
    """
    diff = cmdb.calcdiff(img_mean, img_embs[i])
    extract_w, aaa = cmdb.extract(img_mean, diff)
    
    
    

    
    

    
    """ 保存関係 """
    filename_middle += str(i) + '+'   
    filename = filename_pre +  filename_middle + '.png'
    cv2.imwrite(savepass + '\\' + filename, extract_w)
    
    plt.figure(1)
    plt.subplot(4,4,i_pre+1)
    plt.imshow(extract_w, vmin=0, vmax=255, cmap = 'gray')

""" 平均値攻撃。ここまで------------------------------------------------------------------------------------------------------ """

"""
plt.figure(1)
plt.subplot(221)
plt.imshow(img_original, cmap = 'gray')
plt.subplot(222)
plt.imshow(embed_img , cmap = 'gray')
plt.subplot(224)
plt.imshow(extract_w , cmap = 'gray')
"""







