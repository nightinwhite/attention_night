#coding:utf-8
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import sys
reload(sys)
sys.setdefaultencoding('utf8')
def tst_img(img):
    cv2.imshow("tst",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

char_to_label_bac = {
u'君':0,u'不':1,u'见':2,u'黄':3,u'河':4,u'之':5,u'水':6,u'天':7,u'上':8,
    u'来':9,u'奔':10,u'流':11,u'到':12,u'海':13,u'复':14,u'回':15,u'烟':16,
    u'锁':17,u'池':18,u'塘':19,u'柳':20,u'深':21,u'圳':22,u'铁':23,u'板':24,
    u'烧':25,u'0':26,u'1':27,u'2':28,u'3':29,u'4':30,u'5':31,u'6':32,
    u'7':33,u'8':34,u'9':35,u'+':36,u'-':37,u'*':38,u'/':39,
    u'(':40,u')':41,u'=':42,u';':43,u'#':43
}

mode = "train"
fill_label = 43
seq_length = 51
file_dir = "/home/night/data/train_all/"
txt_dir = "baiducontest_real_labels.txt"
IMG_WIDTH = 600
IMG_HEIGHT = 300
tf_writer = tf.python_io.TFRecordWriter("tfrecords/{0}_all.records".format(mode))
f = open(txt_dir,"r")
tmp_str = f.readline()
tmp_strs = []
i = 0
# wrong_list = [1165, 2167, 4635, 6567, 13563, 17420, 21324, 24410, 25287, 25665, 27840, 28741, 29128, 32549, 47946, 48385, 53371, 56252, 56559, 58873, 59050, 59973, 71219, 74667, 77247, 77547, 77757, 78031, 78859, 83896, 86433, 87429, 87994, 89789, 91619, 92750, 93458, 95612, 96175]
while tmp_str!="":
    tmp_str = unicode(tmp_str)
    # tmp_str = tmp_str[:-1]
    # print tmp_str
    tmp_strs.append([tmp_str,i])
    i+=1
    tmp_str = f.readline()
np.random.seed(1234)
np.random.shuffle(tmp_strs)

if mode == "test":
    tmp_strs = tmp_strs[90000:]
    i = 90000
else:
    tmp_strs = tmp_strs[:90000]
    i = 0

for tmp_line in tmp_strs:
    print i
    tmp_str = tmp_line[0].split(" ")[0]
    tmp_chars = tmp_str
    fen_index = -1

    # print tmp_chars
    tmp_labels = []
    for c in tmp_chars:
        tmp_labels.append(char_to_label_bac[c])
    while len(tmp_labels) < seq_length:
        tmp_labels.append(fill_label)
    try:
        # print tmp_line[:-1]
        # print len(tmp_chars_group)
        # print "{0}{1}_{2}_{3}_.png".format(file_dir, i, g,len(tmp_chars_group))
        tmp_img = Image.open("{0}{1}.png".format(file_dir, tmp_line[1]))
    except IOError :
        print "!!!!!!!!!!!!!!!!!!!", i
        f = open("{0}_tf_wrong.txt".format(mode), "a")
        f.write("{0},".format(i))
        f.close()
        i+=1
        continue
    tmp_img = tmp_img.resize((IMG_WIDTH,IMG_HEIGHT))
    tmp_img = np.asarray(tmp_img,np.uint8)
    # tst_img(tmp_img)
    tmp_img_raw = tmp_img.tobytes()
    tmp_example = tf.train.Example(features=tf.train.Features(feature={
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmp_img_raw])),
        'labels':tf.train.Feature(int64_list=tf.train.Int64List(value=tmp_labels)),
        }))
    tf_writer.write(tmp_example.SerializeToString())
    i += 1
tf_writer.close()
