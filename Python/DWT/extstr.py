# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:54:54 2022

@author: matsunaga
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 16:45:22 2021

@author: matsunaga
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:41:04 2021

@author: matsunaga
"""

import numpy as np
import cv2
from pathlib import Path
import os
import csv

from my_packege import tool 
from my_packege.watermark import MyMultipleDWTandBlock2 as mdb
#from my_packege.watermark import MyMultipleDWTandBlock as mdb




""" 埋め込みの設定 """
N = 3
Q = 1
BLOCK_SIZE = (2, 2)

target = [0.1, 0.6]
diff_block = [4, 4]


foldername = "MyWatermark2Q1"
savepass = "result\\StirMark\\" + foldername
os.makedirs(savepass, exist_ok=True)

img_original = cv2.imread('image\\man.bmp' , cv2.IMREAD_GRAYSCALE)
imgs = tool.get_image('image\\' + foldername + '\\', '*.bmp')
w = cv2.imread('Watermark/watermark1.png',cv2.IMREAD_GRAYSCALE)

i = 0
"""
for i in range(len(imgs[0])):
    print(i)
    if imgs[1][i].shape == (1024,1024):
        img_ext = mdb.extract(imgs[1][i], Q, BLOCK_SIZE, N)
        savepass = "result\\StirMark" 
        filename = Path(imgs[0][i]).stem + '.bmp'
        cv2.imwrite(savepass + '\\' + filename, img_ext)
        
    i += 1
"""   

fail_rate = np.ones( len(imgs[0]), dtype = np.float32) * -1
fail_block_rate = np.ones( len(imgs[0]), dtype = np.float32) * -1
fail_white_rate = np.ones( len(imgs[0]), dtype = np.float32) * -1

if True:  
    for i in range(len(imgs[0])):
        print(i)
        if imgs[1][i].shape != img_original.shape:
            print("skip!!")
            continue
        
        tmp = imgs[1][i]
        tmp = tmp[0:(tmp.shape[0]//16)*16,0:(tmp.shape[1]//16)*16]
        if(np.sum(np.sum(tmp)) > 0 ):
            diff = mdb.calcdiff(tmp, img_original)
            #diff = 0
            img_ext, _ = mdb.extract(tmp, Q, BLOCK_SIZE, N, diff)
            fail_rate[i], fail_block_rate[i], fail_white_rate[i] = tool.calc_failrate(img_ext, w)
            filename = Path(imgs[0][i]).stem + '.bmp'
            cv2.imwrite(savepass + '\\' + filename, img_ext)
            
        i += 1
        

f = open('datection_rate_stirMy.csv', 'w', newline="")
writer = csv.writer(f)
writer.writerow(100 - fail_rate * 100)
writer.writerow(100 - fail_block_rate * 100)
writer.writerow(100 - fail_white_rate * 100)
f.close()        
#aaadwt = tool.dwt_haarN(imgs[1][i], N)
#aaa = tool.inv_dwt_haarN(aaadwt)

