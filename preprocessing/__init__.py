#  -*- coding: utf-8 -*-
'''
===================================================
@Time    : 2019/11/28  12:15 上午
@Author  : Simon Chen Xumin
@IDE     : PyCharm
===================================================
'''
import logging


def outlier_detect(list, min=0., max=5.):
    flag = True
    new_list = []

    # check userid, movieid and rating is null
    for i in range(len(list) - 1):
        if list[i].isalpha():
            flag = False
            break
        elif (list[i] == "" or list[i] is None):
            # there is a null value,so we have to delete this line
            logging.warning( "There is a null, so this might be processed")
            flag=False
            break
        else:
            new_list.append(list[i])

    if (not list[2].isalpha()) and ((float(list[2]) < min) or (float(list[2]) > max)):
        # there is a outlier value,so we have to delete this line
        logging.warning("There is outlier, so this might be processed")
        flag=False

    if flag is True:
        new_list.append(list[2])
        return new_list
    else:
        return []
