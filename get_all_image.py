# coding:utf-8
import cv2
import numpy as np
from PIL import Image,ImageFilter
from matplotlib import pyplot as plt
import thread
import time
import os
#链接区间
# save_path = "tst_img/"

def tst_img(img):
    cv2.imshow("tst",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

mode = "train"#train or val or gen
is_test = False
def get_segment(l,low_limit,seg_limit,len_limit):
    tmp_segments = []
    state = False
    tmp_i = len(l) - 1
    for i in xrange(len(l)):
        if state == False:
            if l[i] >= low_limit:
                state = True
                tmp_i = i
        if state == True:
            if l[i] < low_limit or i == len(l) - 1:
                state = False
                # if len_limit < i - tmp_i:
                tmp_segments.append([tmp_i,i])

    if len(tmp_segments) == 0:
        return []
    res = [tmp_segments[0]]
    for i in xrange(1,len(tmp_segments)):
        if tmp_segments[i][0] - res[-1][1] < seg_limit:
            res[-1][1] = tmp_segments[i][1]
        else:
            res.append(tmp_segments[i])
    return res

def process_single_img(img,img_path):
    # tmp_time = time.time()
    #腐蚀膨胀
    kernel = np.ones((3, 3), np.uint8)
    pos_img = cv2.dilate(img, kernel, iterations=2)
    pos_img = cv2.erode(pos_img, kernel, iterations=6)
    if is_test:
        tst_img(pos_img)
        cv2.imwrite("/home/night/1.png", pos_img)
    # print 1, time.time() - tmp_time
    # tmp_time = time.time()
    #确定背景色，左上角采样
    bg_color_list = {}
    bg_test_width = 16
    bg_test_height = 16
    for i in xrange(bg_test_width):
        for j in xrange(bg_test_height):
            tmp_color = tuple(pos_img[i, j])
            if bg_color_list.has_key(tmp_color) == False:
                bg_color_list[tmp_color] = 1
            else:
                tmp_num = bg_color_list[tmp_color]
                tmp_num += 1
                bg_color_list[tmp_color] = tmp_num
    sort_bg_color_list = sorted(bg_color_list.iteritems(), key=lambda a: a[1], reverse=True)
    bg_color = list(sort_bg_color_list[0][0])
    # print 2, time.time() - tmp_time
    # tmp_time = time.time()
    #xy投影
    # IMG_WIDTH = img.shape[1]
    # IMG_HEIGHT = img.shape[0]
    # x_tou = [0 for i in range(IMG_WIDTH)]
    # y_tou = [0 for i in range(IMG_HEIGHT)]
    # for i in xrange(IMG_WIDTH):
    #     for j in xrange(IMG_HEIGHT):
    #         if (pos_img[j, i] != bg_color).any():
    #             x_tou[i] += 1
    #             y_tou[j] += 1
    # cv2.imshow("1", pos_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    bg_color_sum = sum(bg_color)
    IMG_HEIGHT = pos_img.shape[0]
    IMG_WIDTH = pos_img.shape[1]
    pos_img_np = np.asarray(pos_img, np.int32)
    pos_img_np = pos_img_np[:, :, 0] + pos_img_np[:, :, 1] + pos_img_np[:, :, 2]
    tmp_ones = np.ones((IMG_WIDTH, 1))
    y_tou = np.dot(pos_img_np, tmp_ones)
    y_tou = np.reshape(y_tou, (IMG_HEIGHT))
    y_tou = np.abs(y_tou - bg_color_sum * IMG_WIDTH)

    # x_tou = cv2.calcHist()
    y_segs = get_segment(y_tou, 5, 25, 16)

    # print 3, time.time() - tmp_time
    # tmp_time = time.time()
    #处理碎块
    may_wrong = False
    i = 0
    while i < len(y_segs):
        if y_segs[i][1] - y_segs[i][0] < 20:
            may_wrong = True
            del y_segs[i]
        else:
            i += 1
    i = 0
    x_segs = []
    for tmp_y_segs in y_segs:
        tmp_ones = np.ones((1, tmp_y_segs[1] - tmp_y_segs[0]))
        # tst_img(pos_img[tmp_y_segs[0]:tmp_y_segs[1], :, :] )
        # cv2.imshow("1", pos_img[tmp_y_segs[0]:tmp_y_segs[1], :, :])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        x_tou = np.dot(tmp_ones, pos_img_np[tmp_y_segs[0]:tmp_y_segs[1],:])
        x_tou = np.reshape(x_tou, (IMG_WIDTH))
        x_tou = np.abs(x_tou - bg_color_sum * (tmp_y_segs[1] - tmp_y_segs[0]))
        # for t in range(len(x_tou)):
        #     if x_tou[t] != 0:
        #         print t,x_tou[t],tmp_y_segs
        tmp_x_segs = get_segment(x_tou, 5, 40, 20)
        if is_test:
            print "here", tmp_x_segs
        tmp_v = 0
        tmp_i = 0
        if len(tmp_x_segs) > 1:
            s_i = 0
            for s in tmp_x_segs:
                if s[1] - s[0] > tmp_v:
                    tmp_v = s[1] - s[0]
                    tmp_i = s_i
                s_i += 1
        x_segs.append(tmp_x_segs[tmp_i])
    if is_test:
        print "here2", x_segs

    while i < len(x_segs):
        if x_segs[i][1] - x_segs[i][0] < 20:
            may_wrong = True
            del x_segs[i]
        else:
            i += 1
    # print x_segs
    # print 4, time.time() - tmp_time
    # tmp_time = time.time()
    # 处理不合理块
    while len(y_segs) > 3:
        may_wrong = True
        tmp_min_v = y_segs[0][1] - y_segs[0][0]
        tmp_min_i = 0
        for i in range(1,len(y_segs)):
            if y_segs[i][1] - y_segs[i][0] < tmp_min_v:
                tmp_min_v = y_segs[i][1] - y_segs[i][0]
                tmp_min_i = i
        del y_segs[tmp_min_i]


    if len(x_segs) > 3:
        may_wrong = True
        tmp_min_v = x_segs[0][1] - x_segs[0][0]
        tmp_min_i = 0
        for i in range(1, len(x_segs)):
            if x_segs[i][1] - x_segs[i][0] < tmp_min_v:
                tmp_min_v = x_segs[i][1] - x_segs[i][0]
                tmp_min_i = i
        del x_segs[tmp_min_i]
    # print 5, time.time() - tmp_time
    # tmp_time = time.time()
    #根据区间划分

    bias = 5
    if may_wrong:
        f = open("{0}_may_wrong_img.txt".format(mode),"a")
        f.write("{0}\n".format(img_path))
        f.close()

    tmp_x_min = x_segs[-1][0] - bias
    if tmp_x_min < 0:
        tmp_x_min = 0
    tmp_x_max = x_segs[-1][1] + bias+15

    if tmp_x_max >= IMG_WIDTH:
        tmp_x_max = IMG_WIDTH - 1
    tmp_y_min = y_segs[0][0] - bias
    if tmp_y_min < 0:
        tmp_y_min = 0
    tmp_y_max = y_segs[-1][1] + bias
    if tmp_y_max >= IMG_HEIGHT:
        tmp_y_max = IMG_HEIGHT - 1

    tmp_img = img[tmp_y_min:tmp_y_max, tmp_x_min:tmp_x_max, :]
    f = open("{0}_img_shape.txt".format(mode), "a")
    f.write("{0} {1} {2}\n".format(img_path, i, tmp_img.shape))
    f.close()
    return tmp_img
    # print 6, time.time() - tmp_time
    # tmp_time = time.time()

if __name__ == '__main__':
    if os.path.exists("/home/night/data/{0}_all/".format(mode)) == False:
        os.mkdir("/home/night/data/{0}_all/".format(mode))

    if mode == "train":
        file_path = "/home/night/data/image_contest_level_2/"
    else:
        file_path = "/home/night/data/image_contest_level_2_validate/"
    save_path = "/home/night/data/{0}_all/".format(mode)
    # tst_index = 4635
    # is_test = True
    # for i in range(tst_index,tst_index+1):
    for i in range(80000, 100000):
        print i
        # tmp_line = tmp_lines[i]
        # tmp_strs = tmp_line.split(";")
        # segs = len(tmp_strs)
        tmp_path = file_path + "{0}.png".format(i)
        tmp_png = cv2.imread(tmp_path)
        tmp_img = process_single_img(tmp_png, tmp_path)
        f = open("{0}_all_img_shape.txt".format(mode), "a")
        f.write("{0} {1}\n".format(i ,tmp_img.shape))
        f.close()
        # tst_img(tmp_img)
        # print segs_num,len(tmp_imgs)
        cv2.imwrite("{0}{1}.png".format(save_path, i), tmp_img)







