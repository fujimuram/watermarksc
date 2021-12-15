# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 19:03:26 2021

@author: matsunaga

透かしのクラスの抽象メソッドを定義する
"""

from abc import ABC

class ABCWatermark(ABC):
    
    def embed(self, listdataA):
        #listdataは元画像、透かし情報、埋め込み場所が想定される。増えてもいいようにlist型で記述した
        pass
        return listdataR
    
    def extract(self, listdataA):
        #listdataは元画像、透かし情報、埋め込み場所が想定される。増えてもいいようにlist型で記述した
        pass
        return listdataR
    