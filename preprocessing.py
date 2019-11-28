#  -*- coding: utf-8 -*-
'''
===================================================
@Time    : 2019/11/28  12:15 上午
@Author  : Simon Chen Xumin
@IDE     : PyCharm
===================================================
'''
import logging

def outlier_detect(num, min=0, max=5):
    if (num == "" or num is None):
        # there is a null value,so we have to delete this line
        logging.warning("There is a null, so this might be processed")
        return False
    elif (num < min or num > max):
        # there is a outlier value,so we have to delete this line
        logging.warning("There is outlier, so this might be processed")
        return False
    else:
        return True