# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:23:39 2021

@author: matsunaga
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 13:10:12 2021

@author: 松永健彦
https://www.jstage.jst.go.jp/article/itej1997/52/12/52_12_1832/_pdf/-char/ja
参考サイト
"""


import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP
import skimage.util
import copy


def main():
    """
    img_original = cv2.imread('image/LENNA512.tiff' , cv2.IMREAD_GRAYSCALE)
    
    r = dwt_haarN(copy.deepcopy(img_original), 3)
    
    #dwt_haar(img_original)
    
    re_img_n = inv_dwt_haarN(r)
    
    
    plt.figure(1)
    plt.subplot(231)
    plt.imshow(re_img_n, cmap = 'gray')
    
    
    dwt1 = dwt_haar(img_original)
    dwt2 = dwt_haar(dwt1[0])
    dwt3 = dwt_haar(dwt2[0])

    
    re2     = inv_dwt_haar(*dwt3)
    re1     = inv_dwt_haar(re2 , *dwt2[1:])
    re_img  = inv_dwt_haar(re1 , *dwt1[1:])
    
    diff = re_img_n - re_img
    
    goukei = sum(sum(diff))
    
    pass

    """
    
    BLOCK_SIZE = (2,2)
    Q = 2.0
    n = 3
    
    
    #画像読み込み
    #img_original = cv2.imread('image/LENNA512.tiff' , cv2.IMREAD_GRAYSCALE)
    img_original = cv2.imread('image/Man.bmp' , cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread('Watermark/watermark1.png',cv2.IMREAD_GRAYSCALE)
    #watermark = np.array([[1,0,0,1],[0,0,0,0],[1,1,1,1],[1,1,0,0]]) * 255
    #watermark = np.zeros((4,4))    
    
    
    #埋め込みと抽出
    img_embed, _LM = emb_dwtN(copy.deepcopy(img_original),watermark, Q, BLOCK_SIZE, n)    
    watermark_extracted = extract_dwtN(copy.deepcopy(img_embed), Q, BLOCK_SIZE, _LM, n)


    diff = img_embed - img_original
    
    """    
    plt.figure(2)
    plt_size = int(SPLIT_SIZE / 4)
    for tmp in range(SPLIT_SIZE):
        plt.subplot(plt_size, plt_size, tmp + 1)
        #plt.imshow(haarLL3_[tmp], cmap = 'gray')
    """
    #haarLL3_ = np.concatenate(np.concatenate(haarLL3_, 1), 1)
    #表示
    
    
    plt.figure(1)
    plt.subplot(231)
    plt.imshow(img_original, cmap = 'gray')
    plt.subplot(232)
    plt.imshow(img_embed , cmap = 'gray')
    plt.subplot(233)
    plt.imshow(watermark_extracted , cmap = 'gray')
    
    pass


"""
ハール基底を用いたDWTを行う
"""
def dwt_haar(Img_):
    
    #引数は変更しないようにした。念のため。
    Img = copy.deepcopy(Img_)
    Img = Img.astype(np.float32)
    
    #画像のサイズ等の制約を書こう
    if Img.shape[0] % 2 == 1 or Img.shape[1] % 2 == 1:
        print("画像のサイズは偶数にしろ。バグがあっても知らない")
     
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


"""
ハール基底を用いた逆DWTを行う
"""
def inv_dwt_haar(LL,HL,LH,HH):
    
    Img = np.zeros( (LL[0].size * 2, LL[1].size * 2) )
    
    Img[0::2,0::2] = LL + HL + LH + HH
    Img[1::2,0::2] = LL - HL + LH - HH
    Img[0::2,1::2] = LL + HL - LH - HH
    Img[1::2,1::2] = LL - HL - LH + HH
    
    
    return Img.clip(0, 255).astype(np.uint8)

def dwt_haarN(img_, n):
    
    img = copy.deepcopy(img_)
    
    coefficients = []
    
    tmp = [img,0,0,0]
    
    for i in range(n):
        tmp = list(dwt_haar(tmp[0]))
        coefficients.append(tmp)
    
    return coefficients

def inv_dwt_haarN(coefficients_):
    
    
    coefficients = copy.deepcopy(coefficients_)
      
    for i in range( -1 , -len(coefficients) , -1 ) :
        tmp = inv_dwt_haar(*coefficients[i])
        #print(i,i-1)
        coefficients[i -1][0] = tmp
        
    
    return inv_dwt_haar(*coefficients[i -1])


def emb_dwtN(img, w, Q, block_size, n):
    
    #w = np.ravel(w)
    
    """
    haarLL1, haarHL1, haarLH1, haarHH1 = dwt_haar(img)
    haarLL2, haarHL2, haarLH2, haarHH2 = dwt_haar(haarLL1)
    haarLL3, haarHL3, haarLH3, haarHH3 = dwt_haar(haarLL2)
    """
    
    
    coefficients = dwt_haarN(img, n)
    
    embed_block = skimage.util.view_as_blocks(copy.deepcopy(coefficients[n-1][0]), block_size)
    
    
    
    
    for i in range(embed_block.shape[0]):
        for j in range(embed_block.shape[1]):
            
            #print("\n")
            
            if w.shape[0] <= i or w.shape[1] <= j:
                continue
            
            #print("i = ", i, "j = ", j)
            #print("i = ", i, "j = ", j)
            #imgの平均を計算する
            mean = sum(sum( embed_block[i][j] )) / embed_block[i][j].size
            
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
            embed_block[i][j] += diff_M
            #print(embed_block[i,j,0,0:3])
            
            
    """
    plt.figure(0)   
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            plt.subplot(imgs.shape[0], imgs.shape[1], i * imgs.shape[1] + j + 1)
            plt.imshow(imgs[i][j], cmap = 'gray')
    """

    
    coefficients[n-1][0] = np.concatenate(np.concatenate(embed_block, 1), 1)
    
    _LM = sum(sum( coefficients[n-1][0] )) / coefficients[n-1][0].size
    #print(_LM)
    
    """
    _haarLL2 = inv_dwt_haar(_haarLL3, haarHL3, haarLH3 , haarHH3)
    _haarLL1 = inv_dwt_haar(_haarLL2, haarHL2, haarLH2 , haarHH2)
    _img     = inv_dwt_haar(_haarLL1, haarHL1, haarLH1 , haarHH1)
    """
    
    _img = inv_dwt_haarN(coefficients)
    
    
    return _img, _LM

def extract_dwtN(img, Q, block_size, _LM, n):
    
    """
    haarLL1, haarHL1, haarLH1, haarHH1 = dwt_haar(img)
    haarLL2, haarHL2, haarLH2, haarHH2 = dwt_haar(haarLL1)
    haarLL3, haarHL3, haarLH3, haarHH3 = dwt_haar(haarLL2)
    """
    
    """
    (1)
    """
    coefficients = dwt_haarN(img, n)
    
    """
    (2)
    """
    LM = sum(sum( coefficients[n-1][0]) ) / coefficients[n-1][0].size
    diff_mean = LM - _LM
    
    #print(diff_mean)
    
    """
    (3)
    (4)
    """
    embed_block = skimage.util.view_as_blocks(copy.deepcopy(coefficients[n-1][0]), block_size)
    w = np.zeros(embed_block.shape[0:2])

    for i in range(embed_block.shape[0]):
        for j in range(embed_block.shape[1]):
            
            
            block_mean = sum(sum( embed_block[i][j] )) /  embed_block[i][j].size
            S = (block_mean - diff_mean) / Q
            
            
            S = int(Decimal( str(S) ).quantize( Decimal('0'), rounding=ROUND_HALF_UP) )
            #print(S)
            if S % 2 == 0:
                w[i,j] = 0
            else :
                w[i,j] = 1
                
                
    return w



if __name__ == "__main__":
    main()
