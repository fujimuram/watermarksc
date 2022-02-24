# -*- coding: utf-8 -*-
"""
2021/6/3
2021/6/14　完成+改良中

"""
 
import numpy as np
import cv2
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP

"""
ハール基底を用いたDWTを行う
"""
def dwt_haar(Img):
    
    if Img.shape[0] % 2 == 1 or Img.shape[1] % 2 == 1:
        print("画像のサイズは偶数にしろ。バグがあっても知らない")
    
    
    """
    １ブロックを
    [a b
     c d]と表すことにする
    """
    Img = Img.astype(np.float32)
    
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
    
    

"""
ハール基底のDWTを用いた透かしの埋め込みを行う関数
HL,LH,HHの最大値-最小値を四捨五入した値の偶数,奇数を制御することで情報を埋め込む。
埋め込み位置は元画像のどこから透かし情報を埋め込むかということを示す。
"""
def emb_DWT(Img,Watermark,Start = [0,0]):
    
    if Img.shape[0] / 2 - Start[0] < Watermark.shape[0] or Img.shape[1] / 2 - Start[1] < Watermark.shape[1]:
        print("エラー。透かし画像に対して元画像のサイズが足りないか、埋め込み位置が悪いと思います。ちなみに元画像は最低でも透かし画像のサイズの２倍は必要です。")
        return Img
    
    if Start[0] % 2 == 1 or Start[1] % 2 == 1 :
        print("埋め込み位置の指定は偶数でよろしく。この表示が出ているときは埋め込みはしていません。")
        return Img
    
    HaarLL, HaarHL, HaarLH , HaarHH = dwt_haar(Img)
    #埋め込み位置を調整するための計算
    Start[0] = int(Start[0] / 2)
    Start[1] = int(Start[1] / 2)
    
    for i in range(Watermark.shape[0]):
        for j in range(Watermark.shape[1]):
            
            #print("\n\n")

            Max = max([ HaarHL[i+Start[0],j+Start[1]],HaarLH[i+Start[0],j+Start[1]],HaarHH[i+Start[0],j+Start[1]] ]) 
            Min = min([ HaarHL[i+Start[0],j+Start[1]],HaarLH[i+Start[0],j+Start[1]],HaarHH[i+Start[0],j+Start[1]] ]) 
            
            
            #最大値-最小値を四捨五入した値を計算する
            diff = int(Decimal(str(Max - Min)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
            
            #print("max",Max,",min",Min,",diff", diff % 2,"  ,",Watermark[i,j])
            
            
            #透かし情報と"元々の偶数か奇数か"が一致していれば何もしないので次のイタレーションへ
            if (diff % 2 == 0 and Watermark[i,j] == 0) or ( diff % 2 == 1 and Watermark[i,j] == 255):
                continue
            
            #最大値が複数ある場合は以下のような優先順位で処理をする。今回はこうしたけど、どれを先にしても良い。
            if HaarHL[i+Start[0],j+Start[1]] == Max:
                HaarHL[i+Start[0],j+Start[1]] +=1
            elif HaarLH[i+Start[0],j+Start[1]] == Max:
                HaarLH[i+Start[0],j+Start[1]] +=1
            elif HaarHH[i+Start[0],j+Start[1]] == Max:
                HaarHH[i+Start[0],j+Start[1]] +=1
                
    
    return inv_dwt_haar(HaarLL, HaarHL, HaarLH , HaarHH)

"""
ハール基底のDWTを用いた透かしの抽出を行う関数
HL,LH,HHの最大値-最小値を四捨五入した値の偶数,奇数を参照することで埋め込まれた情報を取り出す。
"""
def ext_DWT(Img):

    Inv_Watermark = np.zeros( (int(Img[0].size / 2), int(Img[1].size / 2)) )
    
    haarLL, haarHL, haarLH , haarHH = dwt_haar(Img)
    
    
    for i in range(int(Img.shape[0] / 2)):
        for j in range(int(Img.shape[1] / 2)):
            
            Max = max([ haarHL[i,j],haarLH[i,j],haarHH[i,j] ]) 
            Min = min([ haarHL[i,j],haarLH[i,j],haarHH[i,j] ]) 
            
            #最大値-最小値を四捨五入した値を計算する
            diff = int(Decimal(str(Max - Min)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
            
            #最大値-最小値を四捨五入した値が奇数の場合は白しする
            if diff % 2 == 1:
                Inv_Watermark[i,j] = 255
                
    return Inv_Watermark


#画像読み込み
Img_original = cv2.imread('image/LENNA512.tiff' , cv2.IMREAD_GRAYSCALE)
Watermark = cv2.imread('Watermark/watermark1.png',cv2.IMREAD_GRAYSCALE)

zzzzgomi, Watermark = cv2.threshold(Watermark, 128, 255, cv2.THRESH_BINARY)


#元画像に対してハール基底のDWTを行う
HaarLL, HaarHL, HaarLH , HaarHH = dwt_haar(Img_original)


#haar基底のdwtを用いた埋め込みを行う
Img_embed = emb_DWT(Img_original,Watermark,[128,128])

#抽出
Img_extract = ext_DWT(Img_embed )

#表示
plt.figure
plt.subplot(231)
plt.imshow(Img_original, cmap = 'gray')
plt.subplot(232)
plt.imshow(Watermark, cmap = 'gray')
plt.subplot(233)
plt.imshow(Img_extract, cmap = 'gray')