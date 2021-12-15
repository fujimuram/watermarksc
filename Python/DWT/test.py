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
import matplotlib.pyplot as plt
import skimage.util
import copy


from my_packege import tool 
from my_packege.watermark import MultipleDWTandBlock as mdb

def main():
    
    block_size = [2,2]
    Q = 2.0
    N = 3
    number_big_block = None
    #透かし画像を埋めるのに必要な元画像の最低限のサイズ。透かし画像のサイズとDWTの回数Nとblock_sizeに依存する
    required_capacity = None
    
    
    img_original = cv2.imread('image/Man.bmp' , cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread('Watermark/16_16/0.png',cv2.IMREAD_GRAYSCALE)
    
    
    required_capacity = np.array(watermark.shape) * np.array(block_size) ** N
    
    
    size_need = np.array(block_size) * (2 ** N) * watermark.shape
    img_blocks = skimage.util.view_as_blocks(copy.deepcopy(img_original), tuple(size_need.tolist())) 
    
    
    _img, _LM = mdb.emb(img_original, watermark, Q, block_size, N)
    
    _w = mdb.extract(_img, Q, block_size, _LM, N)
    
    
    
    """
    LM_ = np.zeros(img_blocks.shape[0:2])
    
    LM_ = get_LM(img_blocks, N)
    
    
    
    #img_embed, LM_ = emb_several(img_original, watermark, Q, block_size, N)
    img_embed, LM_[0,0] = emb_to_point(img_original, watermark, Q, block_size, N, [0,0])
    #img_embed, _ = make_all_embed_img(img_blocks, watermarks, Q, block_size, n)
    
    watermark_extracted = extract_dwtN(img_embed, Q, block_size, LM_[0,0], N)
    #watermark_extracted = extract_dwtN_point(img_embed, Q, block_size, LM_, B, watermarks[0].shape)
    """
    
    
    print("main ")
    print(mdb.what_size_need(watermark.shape, N, block_size))
    
    
    plt.figure(1)
    plt.subplot(231)
    plt.imshow(img_original, cmap = 'gray')
    plt.subplot(232)
    plt.imshow(_img , cmap = 'gray')
    plt.subplot(233)
    plt.imshow(_w , cmap = 'gray')
    #plt.imshow(watermark_extracted , cmap = 'gray')
    
   
    
    """    
    
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
    """    
    plt.figure(2)
    plt_size = int(SPLIT_SIZE / 4)
    for tmp in range(SPLIT_SIZE):
        plt.subplot(plt_size, plt_size, tmp + 1)
        #plt.imshow(haarLL3_[tmp], cmap = 'gray')
    """
    
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
    """
    
    pass








def emb_several(img, watermarks, Q, block_size, n):
    
    #元画像をブロックに分割する
    #ブロックサイズの指定はタプルじゃないといけないらしいので、変換&変換というめんどくさいことをやってる
    
    
    size_need = block_size[0] * (2 ** n)
    img_block_size = np.array(watermarks[0].shape) * size_need
    img_block_size = tuple(img_block_size.tolist())
    img_blocks = skimage.util.view_as_blocks(copy.deepcopy(img), img_block_size )
    
    
    img_embed = copy.deepcopy(img_blocks)
    LM_ = np.zeros(img_blocks.shape[0:2])
    
    #print(img_block_size)
    
    for row in range(img_blocks.shape[0]):
        if img_blocks.shape[0] * row >= len(watermarks):
                break
        for col in range(img_blocks.shape[0]):
            if img_blocks.shape[0] * row + col >= len(watermarks):
                break
            
            print(row,col,img_blocks.shape[0] * row + col)
            img_embed[row,col], LM_[row,col] = emb_dwtN(img_blocks[row][col],watermarks[img_blocks.shape[0] * row + col], Q, block_size, n)
        
    
    
    
    img_embed_ = np.concatenate(np.concatenate(img_embed, 1), 1)
    #print("aaaaa")
    
    return img_embed_, LM_


def emb_to_point(img, watermark, Q, block_size, n, point = [0,0]):
    
    #元画像をブロックに分割する
    #ブロックサイズの指定はタプルじゃないといけないらしいので、変換&変換というめんどくさいことをやってる
        
    size_need = block_size[0] * (2 ** n)
    img_block_size = np.array(watermark.shape) * size_need
    img_block_size = tuple(img_block_size.tolist())
    img_blocks = skimage.util.view_as_blocks(copy.deepcopy(img), img_block_size )
    img_blocks = np.array(img_blocks)
    
    img_embed, LM_ = emb_dwtN(img_blocks[point[0],point[1]],watermark, Q, block_size, n)
    img_blocks[point[0],point[1]] = img_embed
    
    img_ = np.concatenate(np.concatenate(img_blocks, 1), 1)
    
    
    """
    watermark_extracted1 = extract_dwtN(img_embed, Q, block_size, LM_, n)
    watermark_extracted2 = extract_dwtN(img_[0:256,0:256], Q, block_size, LM_, n)
    
    plt.figure(4)
    plt.subplot(231)
    plt.imshow(watermark_extracted1, cmap = 'gray')
    plt.subplot(232)
    plt.imshow(watermark_extracted2, cmap = 'gray')
    """
    
    return img_, LM_

def extract_dwtN_point(img, Q, block_size, LM_, n, watermark_size):
    
    size_need = block_size[0] * (2 ** n)
    img_block_size = np.array(watermark_size) * size_need
    img_block_size = tuple(img_block_size.tolist())
    img_blocks = skimage.util.view_as_blocks(copy.deepcopy(img), tuple(img_block_size) )
    img_blocks = np.array(img_blocks)
            
    w_blocks = np.zeros(LM_.shape + watermark_size)
    
    for row in range(img_blocks.shape[0]):
        for col in range(img_blocks.shape[0]):
            w_blocks[row,col] = extract_dwtN(img_blocks[row,col], Q, block_size, LM_[row,col], n)
            
    return w_blocks

def get_LM(imgs, n):
    
    LM = np.zeros(imgs.shape[0:2])
    
    for row in range(imgs.shape[0]):
        for col in range(imgs.shape[1]):
            #print(row,col)
            coefficient = dwt_haarN(imgs[row,col], n)
            LM[row,col] = sum(sum(coefficient[n-1,0])) / coefficient[n-1,0].size
            
    return LM

def make_all_embed_img(img, watermarks, Q, block_size, n):
    
    print("aaaaaa")
    
    
    
    for row in range(imgs.shape[0]):
        for col in range(imgs.shape[1]):
            img_embed[row,col], LM_[row,col] = emb_to_point(img[row,col],watermarks[a * row + col], Q, block_size, n)
    
    
    
    #img_embed, LM_[0,0] = emb_to_point(img, watermarks[0], Q, BLOCK_SIZE, n, [0,0])
    

    
    return img_embed, LM_



if __name__ == "__main__":
    main()
