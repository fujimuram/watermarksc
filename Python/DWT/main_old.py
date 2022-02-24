# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 17:42:22 2022

@author: matsunaga
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 14:21:39 2022

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
from my_packege.watermark import watermarkDWT as WDWT




""" 準備 """
savepath_pre = "result\\main_old\\overlap\\"

img_name = 'LENNA128.png'
#img_name = 'man_leveling.png'
#img_name = 'Honeyview_image-name-ec.png'
#img_name = 'Honeyview_image-name-ec (4).png'
img_original_pre = cv2.imread('image\\' + img_name , cv2.IMREAD_GRAYSCALE)
img_original = img_original_pre.clip(2,253)
cv2.imwrite('man_clip.png', img_original_pre)

watermark = cv2.imread('Watermark\\16_16\\0.png',cv2.IMREAD_GRAYSCALE)
watermarks = tool.get_image('Watermark\\overlap\\', '*.png')
#watermarks = tool.make_watermarks(watermarks[1], [64,64])

""" 埋め込みの設定 """
N = 3
Q = 1
BLOCK_SIZE = (2, 2)

""" 埋め込みにどの透かし画像から始めるかを決める。※watermaeks[0]の順番から。 """
start_num_embed = 0
""" 平均値攻撃を行う際の最初の場所の設定 """
start_num = 0

"""" 埋め込みに使うインスタンスを作成 """


#img_original = cmdb.preembed(img_original_pre)

""" 埋め込み&保存 ------------------------------------------------------------------------------------------------------"""

"""" ファイル関係 """
filename_status = 'N' + str(N).zfill(2) + 'Q' + str(Q).zfill(2) \
    + 'BS' + str(BLOCK_SIZE[0]).zfill(2) + 'x' + str(BLOCK_SIZE[1]).zfill(2)
savepass = savepath_pre + Path(img_name).stem + '\\' + Path(img_name).stem 
savepass = re.sub(r"\s", "", savepass)
savepass = re.sub("\|", ",", savepass)
os.makedirs(savepass, exist_ok=True)
os.makedirs(savepass + '\\ext', exist_ok=True)

block_number =  img_original.shape // np.array([2,2]) // watermark.shape

embed_imgs = []
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
    emb_block = [0, 0]
    i_shift = (i + start_num_embed) % len(watermarks[0])
    
    embed_img = WDWT.embed(img_original, watermarks[1][i_shift], Q, [0,0])
    embed_imgs.append(embed_img)
    psnr[i] = cv2.PSNR(img_original, embed_img)
    
    
    extract_img = WDWT.extract(embed_img)
    
    
    filename = Path(img_name).stem + ',W' + Path(watermarks[0][i_shift].split('\\')[-1]).stem.zfill(3) + \
        '(' + str(row) + ',' + str(col) + ')' + filename_status + '.png'
    cv2.imwrite(savepass + '\\' + filename, embed_img)
    
    cv2.imwrite(savepass + '\\ext\\ext_[' + str(i) + ',' + str(i_shift) + ']' +  filename, extract_img)
    #print('!')
    
    plt.subplot(4,4,i+1)
    plt.imshow(extract_img, vmin=0, vmax=255, cmap = 'gray')
    #print(i, i_shift)
    
means = np.sum(np.sum(embed_imgs,axis=1),axis=1) / embed_imgs[0].size
    
""" 平均値攻撃 ------------------------------------------------------------------------------------------------------"""

""" 前準備 """
img_sum = np.zeros( embed_imgs[0].shape, dtype = float)

filename_pre = 'ExtractWatermakAfterMeanAttack' 
filename_status = 'N' + str(N).zfill(2) + 'Q' + str(Q).zfill(2) \
    + 'BS' + str(BLOCK_SIZE[0]).zfill(2) + 'x' + str(BLOCK_SIZE[1]).zfill(2)
savepass = savepath_pre + Path(img_name).stem + '\\' + Path(img_name).stem
savepass = re.sub(r"\s", "", savepass)
savepass = re.sub("\|", ",", savepass)
savepass = savepass + '\\' + 'ExtractWatermakAfterMeanAttack'  
filename_middle  = '[' + str(start_num) + ','+ str(start_num_embed) + ']_' + Path(img_name).stem
os.makedirs(savepass, exist_ok=True)

plt.figure(1)

#print("\n\n抽出")
for i in range(len(embed_imgs)):  

    i_pre = i
    i = (i + start_num) % len(embed_imgs)
    #print("i = ", i)
    
    """ 
    平均値攻撃の実行
    img_meanにはembed_imgsの0からi番目までの平均が入る。
    よってi番目のループではi枚の平均値攻撃が行われる。
    """
    img_sum += embed_imgs[i]
    img_mean = img_sum / (i_pre + 1)
    img_mean = np.round(np.clip(img_mean,0,255)).astype('uint8')
    

     
    """ 
    抽出
    """
    extract_w = WDWT.extract(img_mean)
    
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






