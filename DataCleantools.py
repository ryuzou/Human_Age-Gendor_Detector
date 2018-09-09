import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
from timeit import default_timer as timer
import os
import glob
import re
import shutil

AGE_LABLES = ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-40', '41-50', '51-60',
              '61-']  # 0,1,2,3,4,5,6,7,8,9
GENDER_LABLES = ['MALE', 'FEMALE']  # 0,1
# LABLES = [('1-5', 'MALE'), ('1-5', 'FEMALE')...('61-', 'MALE', ('61-', 'FEMALE'))]#0,1,2,..18,19

counter = 0

if __name__ == '__main__':
    num = 0
    print(num)
    filepaths = glob.glob('Dataset/Data/*.jpg')

    datasets = []
    for FP in filepaths:
        num = num + 1
        error = 0
        # if num >= 2000:
        #    break
        try:
            label1 = int(re.split(r'_', re.split(r'/', FP)[2])[0])
            label2 = int(re.split(r'_', re.split(r'/', FP)[2])[1])
        except Exception:
            continue
        if label2 == 0:
            sex = "Male"
        elif label2 == 1:
            sex = "FeMale"
        else:
            continue
        if 1 <= label1 <= 5:
            CPpath = sex + "_1-5"
        elif 6 <= label1 <= 10:
            CPpath = sex + "_6-10"
        elif 11 <= label1 <= 15:
            CPpath = sex + "_11-15"
        elif 16 <= label1 <= 20:
            CPpath = sex + "_16-20"
        elif 21 <= label1 <= 25:
            CPpath = sex + "_21-25"
        elif 26 <= label1 <= 30:
            CPpath = sex + "_26-30"
        elif 31 <= label1 <= 40:
            CPpath = sex + "_31-40"
        elif 41 <= label1 <= 50:
            CPpath = sex + "_41-50"
        elif 51 <= label1 <= 60:
            CPpath = sex + "_51-60"
        elif 61 <= label1:
            CPpath = sex + "_61-"
        dst = FP
        filename = re.split(r'_', re.split(r'/', FP)[2])[-1]
        Finalpath = "Dataset/" + "traindat/" + CPpath + "/" + filename
        if num % 8 == 0:
            Finalpath = "Dataset/" + "testdat" + "/" + CPpath + "/" + filename
            try:
                os.remove(Finalpath)
            except Exception:
                error = error + 1
                f = shutil.copyfile(FP, Finalpath)
                continue
        f = shutil.copyfile(FP, Finalpath)
        print(f)
