# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 16:41:59 2021

@author: matsunaga


DWTなどの便利な関数を記述しておく

"""

import numpy as np
import cv2
import copy
import os
import glob
from natsort import natsorted
import skimage.util



def dwt_haar(Img_):
    """
    ハール基底を用いたDWTを行う
    """
    
    #引数は変更しないようにした。念のため。
    Img = copy.deepcopy(Img_)
    Img = Img.astype(np.float32)
    
    #画像のサイズ等の制約を書こう
    if Img.shape[0] % 2 == 1 or Img.shape[1] % 2 == 1:
        print("画像のサイズは偶数にして下さい。バグってもしらない")
     
    """
    画像を2×2のブロックに分割したときの、
    あるブロックを
    [a1 b1
     c1 d1]みたいな感じで表すことにする
    """
    a = Img[0::2,0::2]
    b = Img[0::2,1::2]
    c = Img[1::2,0::2]
    d = Img[1::2,1::2]
    
    LL = (a + b + c + d) / 4
    LH = (a - b + c - d) / 4
    HL = (a + b - c - d) / 4
    HH = (a - b - c + d) / 4
    
    return LL,HL,LH,HH

def inv_dwt_haar(arg):
    
    LL, HL, LH, HH = arg
    
    """
    ハール基底を用いた逆DWTを行う
    """
    Img = np.zeros((LL.shape[0] * 2, LL.shape[1] * 2))
    
    Img[0::2,0::2] = LL + HL + LH + HH
    Img[1::2,0::2] = LL - HL + LH - HH
    Img[0::2,1::2] = LL + HL - LH - HH
    Img[1::2,1::2] = LL - HL - LH + HH
    
    return Img


def dwt_haarN(img_, n):
    """
    引数nで指定した回数DWTを行う。
    ２回目以降は前のDWTで得られたLL領域に対して行う。
    """
    
    img = copy.deepcopy(img_)
    
    
    coefficients = []
    tmp = [img,0,0,0]
    
    for i in range(n):
        tmp = list(dwt_haar(tmp[0]))
        coefficients.append(tmp)
    
    return np.array(coefficients, dtype=object)



def inv_dwt_haarN(coefficients_):
    """
    係数から逆DWTを行う。
    ２回以上のDWT係数を用意していないとエラーが出るはず。
    """
    
    coefficients = copy.deepcopy(coefficients_)
    
    for i in range( -1 , -len(coefficients) , -1 ):
        tmp = inv_dwt_haar(coefficients[i])
        #print(i,i-1)
        coefficients[i -1][0] = tmp
        
    rv = inv_dwt_haar(coefficients[i -1])
    
    return rv

"""


"""
def get_image(image_dir = 'Watermark\\16_16', search_pattern = '*.png'):
    """
    引数で指定したパスにある域数で指定した拡張子の画像をグレースケールで読み込んでくる。
    返り値は１次元のリスト。　
    """
    #(image_dir)
    
    image_paths = []
    datas = []
    
    for image_path in natsorted(glob.glob(os.path.join(image_dir,search_pattern))):
        data = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        image_paths.append(image_path)
        datas.append(data)
        #print(image_path)

    return image_paths, datas

def split_img(img_, block_size):
    
    img = copy.deepcopy(img_)
    
    
    img_blocks = skimage.util.view_as_blocks( img, tuple(block_size.tolist()) )
    
    
    return img_blocks


def split_image_cut(img, block_size ):
    div_v = block_size[0]
    div_h = block_size[1]
    h, w = img.shape[:2]
    block_h, out_h = divmod(h, div_v)
    block_w, out_w = divmod(w, div_h)
    block_shape = (block_h, block_w, 3) if len(img.shape) == 3 else (block_h, block_w)
    return skimage.util.view_as_blocks(img[:h - out_h, :w - out_w], block_shape)

def how_many_divisions(img_size, need_size):
    
    img_size = np.array(img_size)
    need_size = np.array(need_size)
    
    return img_size // need_size 




