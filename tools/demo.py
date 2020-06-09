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
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--config_1', type=str, help='config file')
parser.add_argument('--snapshot_1', type=str, help='model name')
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

    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    # load config
    # old_cfg = cfg
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    # create model
    model = ModelBuilder()
    model = load_pretrain (model,args.snapshot)
    model.eval().to(device)




    # load model
    # model.load_state_dict(torch.load(args.snapshot,
    #     map_location=lambda storage, loc: storage.cpu()))
    # model = load_pretrain (model,args.snapshot)
    # model.eval().to(device)

    # model_1 = load_pretrain (model_1,args.snapshot_1)
    # build tracker
    tracker = build_tracker(model)


    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    # print("ok")
    # print(args.video_name)
    # return 
    #cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    index = 0
    for frame in get_frames(args.video_name):
        print(index)
        if first_frame:
            # print(frame)
            # cv2.imwrite("/gdata/wangmr/pysot/img/"+str(frame)+".jpg",frame)
            try:
                init_rect = (198.5,214.5,34.0,81.0)
                # init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (255, 0, 0), 3)
            #cv2.imshow(video_name, frame)
            # print("ok")
            cv2.imwrite("/gdata/wangmr/pysot/img/"+str(index)+"_2.jpg",frame)
            # cv2.waitKey(40)
        index+=1


if __name__ == '__main__':
    main()
