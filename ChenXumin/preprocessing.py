#  -*- coding: utf-8 -*-
'''
===================================================
@Time    : 2019/11/28  12:15 上午
@Author  : Simon Chen Xumin
@IDE     : PyCharm
===================================================
'''


def outlier_detect(lists, min=0, max=5):
    for i in lists:
        if (i == "" or i is None):
            # there is a null value,so we have to delete this line
            # log function
            print(i + " is null, so this might be processed")
        elif (i < 0 or i > 5):
            # there is a outlier value,so we have to delete this line
            # log function
            print(i + " is outlier, so this might be processed")
        else:
            return i