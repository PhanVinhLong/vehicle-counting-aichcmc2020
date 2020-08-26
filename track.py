#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
import cv2
import numpy as np
from PIL import Image
import argparse
import timeit
import os
from tqdm import tqdm
import moviepy.editor as moviepy
from pathlib import Path

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet
import imutils.video
from utils import *
from config import config

warnings.filterwarnings('ignore')


def parse_args():
    argparser = argparse.ArgumentParser(
        description='Data preparation for vehicle counting')
    argparser.add_argument('-j', '--json_dir', type=str,
                           default='data/json/', help='Json directory')
    argparser.add_argument('-v', '--video_dir', type=str,
                           default='data/video/', help='Video directory')
    argparser.add_argument('-d', '--detect_dir', type=str,
                           default='data/detect', help='Detection result directory')
    argparser.add_argument('-s', '--save_dir', type=str,
                           default='data/track', help='Save result')
    args = vars(argparser.parse_args())
    return args


def map_boxes(boxes, classes_map, idx):
    # boxes: [[x1, y1, x2, y2, _, confident, class_id]...]
    mapped_boxes = []
    mapped_confs = []
    mapped_classes = []
    for box in boxes:
        x = box[0]
        y = box[1]
        w = box[2] - box[0]
        h = box[3] - box[1]
        conf = box[5]
        class_id = box[6]

        if class_id in classes_map[idx]:
            mapped_boxes.append([x, y, w, h])
            mapped_confs.append(conf)
            mapped_classes.append(idx)

    return mapped_boxes, mapped_confs, mapped_classes

def track(json_dir, video_dir, detect_dir, save_dir):
    starttime = timeit.default_timer()

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    cam_datas = get_list_data(json_dir)

    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    model_filename = 'deep_sort/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=16)

    classes_map = [[1, 3], [2], [5], [7]]
    class_names = ['type 1', 'type 2', 'type 3', 'type 4']

    for cam_data in cam_datas:
        cam_name = cam_data['camName']
        width = int(cam_data['imageWidth'])
        height = int(cam_data['imageHeight'])
        FPS = 10.0

        video_path = os.path.join(video_dir, cam_name + '.mp4')
        video_cap = cv2.VideoCapture(video_path)
        num_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        detect_res_path = os.path.join(detect_dir, cam_name + '.npy')
        bboxes = np.load(detect_res_path, allow_pickle=True)

        trackers = []
        filtered_tracks = []
        for _ in range(len(classes_map)):
            metric = nn_matching.NearestNeighborDistanceMetric(
                "cosine", max_cosine_distance, nn_budget)
            tracker = Tracker(metric)
            trackers.append(tracker)
            filtered_tracks.append([])

        for i in tqdm(range(num_frames), desc='Tracking {}'.format(cam_name)):
            success, frame = video_cap.read()

            for class_id in range(len(classes_map)):
                boxes, confidence, classes = map_boxes(
                    bboxes[i], classes_map, class_id)

                features = encoder(frame, boxes)
                detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                              zip(boxes, confidence, classes, features)]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(
                    boxes, nms_max_overlap, scores)
                detections = [detections[k] for k in indices]

                # tracking
                trackers[class_id].predict()
                trackers[class_id].update(detections)

                tracks = []
                for track in trackers[class_id].tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    tracks.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), class_id, track.track_id])
                filtered_tracks[class_id].append(tracks)

        filtered_tracks = np.array(filtered_tracks)
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            filepath = os.path.join(save_dir, cam_name)
            filtered_tracks = np.array(filtered_tracks)
            np.save(filepath, filtered_tracks)

    endtime = timeit.default_timer()

    print('Track time: {} seconds'.format(endtime - starttime))

if __name__ == '__main__':
    args = parse_args()

    json_dir = args['json_dir']
    video_dir = args['video_dir']
    detect_dir = args['detect_dir']
    save_dir = args['save_dir']

    track(json_dir, video_dir, detect_dir, save_dir)
