import os
import argparse
import json
from tqdm import tqdm
import cv2
import timeit
from pathlib import Path
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import torch

from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from utils import *
from config import config

def setup_cfg(cfgfile, confidence_threshold):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(cfgfile)
    # cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.MODEL.WEIGHTS = config['detector']['weightfile']
    cfg.freeze()
    return cfg

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

def detect_detectron2(model, cpu_device, imgs, cam_name='', batch=4):
    height, width, _ = imgs[0].shape
    # sizeds = np.array(sizeds)
    
    start = 0
    end = batch
    result = []
    with tqdm(total=len(imgs), desc='Detecting {}'.format(cam_name)) as progress_bar:
        progress_bar.update(batch)
        while True:
            inputs = [{'image': torch.from_numpy(np.transpose(img, (2, 0, 1)))} for img in imgs[start:end]]
            model.eval()
            with torch.no_grad():
                predictions = model(inputs)

            for i in range(end-start):
                instance = predictions[i]["instances"].to(cpu_device)
                pred_bboxes = instance.pred_boxes.tensor.numpy().tolist()
                pred_scores = instance.scores.numpy().tolist()
                pred_classes = instance.pred_classes.numpy().tolist()
                img_boxes = []
                for idx, box in enumerate(pred_bboxes):
                    box[0] = int(box[0])
                    box[1] = int(box[1])
                    box[2] = int(box[2])
                    box[3] = int(box[3])
                    box.append(pred_scores[idx])
                    box.append(pred_scores[idx])
                    box.append(pred_classes[idx])
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

def detect(json_dir, video_dir, save_dir):
    starttime = timeit.default_timer()

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    cfgfile = config['detector']['cfgfile']
    weightfile = config['detector']['weightfile']
    confidence_threshold = config['detector']['confidencethreshold']

    cfg = setup_cfg(cfgfile, confidence_threshold)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(weightfile)

    cpu_device = torch.device("cpu")

    class_names = config['detector']['originclassnames']
    cam_datas = get_list_data(json_dir)
    
    for cam_data in cam_datas:
        cam_name = cam_data['camName']
        roi_poly =  Polygon(cam_data['shapes'][0]['points'])

        video_path = os.path.join(video_dir, cam_name + '.mp4')
        video_cap = cv2.VideoCapture(video_path)
        num_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        imgs = []
        for i in tqdm(range(num_frames), desc='Extracting {}'.format(cam_name)):
            success, img = video_cap.read()
            imgs.append(img)

        boxes = detect_detectron2(model, cpu_device, imgs, cam_name, config['detector']['batchsize'])

        # remove bboxes out of MOI
        if config['remove_not_intersec_moi']:
            boxes = [check_intersect_box(box_list, roi_poly) for box_list in boxes]

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
    