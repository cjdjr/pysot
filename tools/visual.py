from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.model_load import load_pretrain

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--gt_path', type=str, help='config file')
parser.add_argument('--path1', type=str, help='model name')
parser.add_argument('--path2', type=str, help='config file')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        print("ok")
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            # cv2.imwrite("/gdata/wangmr/pysot/img/test.jpg",frame)
            yield frame


def main():

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    index = 0
    gt_file = open(args.gt_path, "r")
    file1 =open(args.path1, "r")
    file2 =open(args.path2, "r")
    gt = gt_file.readlines()

    # print(gt)
    # return 
    ans1 = file1.readlines()
    ans2 = file2.readlines()
    trans = lambda answer: [ list(map(int,list(map(float,x.split(','))))) for x in answer]
    gt = trans(gt)
    ans1 = trans(ans1)
    ans2 = trans(ans2)
    # ans1 = [ list(map(float,x.split(','))) for x in ans1]


    for frame in get_frames(args.video_name):
        print(index)
        draw = lambda frame,bbox,color: cv2.rectangle(frame, (bbox[0], bbox[1]),(bbox[0]+bbox[2], bbox[1]+bbox[3]),color, 3)
        draw(frame,gt[index],(255,0,0))
        draw(frame,ans1[index],(0,255,0))
        draw(frame,ans2[index],(0,0,255))

            #cv2.imshow(video_name, frame)
            # print("ok")
        cv2.imwrite("/gdata/wangmr/pysot/img/"+str(index)+"_2.jpg",frame)
            # cv2.waitKey(40)
        index+=1
        


if __name__ == '__main__':
    main()
