# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 15:48:51 2021

@author: matsunaga
"""

"""

import numpy as np
import sys
sys.path.append('../../')

from my_packege import tool
from my_packege.my_class import Class_MultipleDWTandBlock as c_mdb



class Class_MultipleDWTandBlock_deriving():

    def __init__(self):
        pass

"""

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from my_packege.my_class.Class_MultipleDWTandBlock import Class_MultipleDWTandBlock as c_mdb


class Class_MultipleDWTandBlock_deriving(c_mdb):

    image_original = None
    image_embed = None
    image_blocks = None
    watermark_shape = None
    one_block_size = None


    def __init__(self, N, Q, block_size, image_original, watermark_shape):
        super().__init__(N, Q, block_size)
        self.image_original = image_original
        self.image_embed = image_original
        
        self.watermark_shape = watermark_shape
        self.one_block_size = watermark_shape * self.required_capacity_per_1bit


    def status(self):
        return "N:{} | Q:{} | block_size:{} | required_capacity:{} | watermark_shape:{} | one_block_shape:{}".format\
            (self.N, self.Q, self.block_size, self.required_capacity_per_1bit, self.watermark_shape, self.one_block_size)



img_original = cv2.imread('../../image/Man.bmp' , cv2.IMREAD_GRAYSCALE)
watermark = cv2.imread('../../Watermark/16_16/0.png',cv2.IMREAD_GRAYSCALE)



wdtn0 = Class_MultipleDWTandBlock_deriving(3, 1, (2,2), img_original, watermark.shape)
#wdtn0 = c_mdb(3, 1, (2,2))

embed_img , LM_ = wdtn0.emb(img_original, watermark)
extract_w = wdtn0.extract(embed_img, LM_)

print(wdtn0.status())

plt.figure(1)
plt.subplot(221)
plt.imshow(img_original, cmap = 'gray')
plt.subplot(222)
plt.imshow(embed_img , cmap = 'gray')
plt.subplot(224)
plt.imshow(extract_w , cmap = 'gray')
