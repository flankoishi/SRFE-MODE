import cv2
import os
import chaifen
import bianli
import numpy as np
Filelist = []
dir_path = 'D:\yangben\Test_ds\Xikapbzn_test'
tar_path = 'D:\yangben\Test_ds\Xikapbzn_test\qiege\ '
#chaifen.chaifen(tar_path, dir_path, 1200, 1600)

Filelist_1 = bianli.getFileList(dir_path, Filelist)

print(bianli.getFileList(dir_path, Filelist))
for i in Filelist_1:
    chaifen.chaifen(tar_path, i, 1200, 1600)


