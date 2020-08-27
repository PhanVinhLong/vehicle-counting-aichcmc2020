import os
import argparse
import json
from tqdm import tqdm
import cv2
import timeit
from pathlib import Path
import numpy as np
from PIL import Image

import sys
sys.path.append(os.path.realpath('yolov4'))

from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
from shapely.geometry import Polygon
from utils import *
from config import config

"""hyper parameters"""
use_cuda = True
remove_not_intersec_moi = config['remove_not_intersec_moi']

def detect_yolov4(model, class_names, imgs, cam_name='', batch=4):
    sizeds = []
    height, width, _ = imgs[0].shape
    for img in tqdm(imgs, desc='Resize images {}'.format(cam_name)):
        sized = cv2.resize(img, (model.width, model.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        sizeds.append(sized)
    # sizeds = np.array(sizeds)
    
    start = 0
    end = batch
    result = []
    with tqdm(total=len(imgs), desc='Detecting {}'.format(cam_name)) as progress_bar:
        while True:
            for i in range(2):
                boxes = do_detect(model, np.array(sizeds[start:end]), 0.4, 0.6, use_cuda)
                if i==1:
                    for i in range(end-start):
                        img_boxes = []
                        for box in boxes[i]:
                            box[0] = int(box[0] * width)
                            box[1] = int(box[1] * height)
                            box[2] = int(box[2] * width)
                            box[3] = int(box[3] * height)
                            img_boxes.append(box)
                        result.append(img_boxes)
            start += batch
            end += batch
            if end > len(imgs):
                end = len(imgs)
            if start >= end:
                break
            progress_bar.update(end - start)
        return result

def parse_args():
    argparser = argparse.ArgumentParser(description='Data preparation for vehicle counting')
    argparser.add_argument('-j', '--json_dir', type=str,
                           default='data/json/', help='Json directory')
    argparser.add_argument('-v', '--video_dir', type=str,
                           default='data/video/', help='Video directory')
    argparser.add_argument('-s', '--save_dir', type=str, default='data/detect', help='Save result')
    args = vars(argparser.parse_args())
    return args

def check_intersect_box(boxes, moi_poly):
    checked_boxes = []
    for box in boxes:
        x1, y1 = box[0], box[1]
        x2, y2 = box[2], box[3]
        box_poly = Polygon([(x1,y1), (x2, y1), (x2, y2), (x1, y2)])
        if box_poly.intersects(moi_poly):
            checked_boxes.append(box)
    return checked_boxes

def detect(json_dir, video_dir, save_dir):
    starttime = timeit.default_timer()

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    cfgfile = 'yolov4/cfg/yolov4.cfg'
    weightfile = 'yolov4/yolov4.weights'

    model = Darknet(cfgfile)
    # model.print_network()
    model.load_weights(weightfile)
    model.cuda()

    num_classes = model.num_classes
    if num_classes == 20:
        namesfile = 'yolov4/data/voc.names'
    elif num_classes == 80:
        namesfile = 'yolov4/data/coco.names'
    else:
        namesfile = 'yolov4/data/x.names'
    class_names = load_class_names(namesfile)

    cam_datas = get_list_data(json_dir)
    
    for cam_data in cam_datas:
        cam_name = cam_data['camName']
        moi_poly =  Polygon(cam_data['shapes'][0]['points'])

        video_path = os.path.join(video_dir, cam_name + '.mp4')
        video_cap = cv2.VideoCapture(video_path)
        num_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        imgs = []
        for i in tqdm(range(num_frames), desc='Extracting {}'.format(cam_name)):
            success, img = video_cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            imgs.append(img)

        boxes = detect_yolov4(model, class_names, imgs, cam_name, 4)

        # remove bboxes out of MOI
        if remove_not_intersec_moi:
            boxes = [check_intersect_box(box_list, moi_poly) for box_list in boxes]

        if save_dir:
            filepath = os.path.join(save_dir, cam_name)
            boxes = np.array(boxes)
            np.save(filepath, boxes)

    endtime = timeit.default_timer()
    
    print('Detect time: {} seconds'.format(endtime - starttime))

if __name__=='__main__':
    args = parse_args()

    json_dir = args['json_dir']
    video_dir = args['video_dir']
    save_dir = args['save_dir']

    detect(json_dir, video_dir, save_dir)
    