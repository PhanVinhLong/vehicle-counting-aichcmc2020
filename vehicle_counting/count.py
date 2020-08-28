import numpy as np
import json
import timeit
import os
import argparse
from utils import *
from pathlib import Path
from config import config
import sys
from shapely.geometry import Polygon

sys.path.append(os.path.realpath('../yolov4'))

from tool.utils import *

def parse_args():
    argparser = argparse.ArgumentParser(
        description='Data preparation for vehicle counting')
    argparser.add_argument('-j', '--json_dir', type=str,
                           default='data/json/', help='Json directory')
    argparser.add_argument('-v', '--video_dir', type=str,
                           default='data/video/', help='Video directory')
    argparser.add_argument('-t', '--track_dir', type=str,
                           default='data/track', help='Detection result directory')
    argparser.add_argument('-s', '--save_dir', type=str,
                           default='data/count', help='Save result')
    args = vars(argparser.parse_args())
    return args

def load_zone_anno(json_filename):
    with open(json_filename) as jsonfile:
        dd = json.load(jsonfile)
        polygon = [(int(x), int(y)) for x, y in dd['shapes'][0]['points']]
        paths = {}
        for it in dd['shapes'][1:]:
            kk = str(int(it['label'][-2:]))
            paths[kk] = [(int(x), int(y)) for x, y in it['points']]
    return polygon, paths

def check_bbox_overlap_with_roi(box, roi):
    roi_poly =  Polygon(roi)
    x1, y1 = box[0], box[1]
    x2, y2 = box[2], box[3]
    box_poly = Polygon([(x1,y1), (x2, y1), (x2, y2), (x1, y2)])
    return box_poly.intersects(roi_poly)

def is_same_direction(traj1, traj2, angle_thr):
    vec1 = np.array([traj1[-1][0] - traj1[0][0], traj1[-1][1] - traj1[0][1]])
    vec2 = np.array([traj2[-1][0] - traj2[0][0], traj2[-1][1] - traj2[0][1]])
    L1 = np.sqrt(vec1.dot(vec1))
    L2 = np.sqrt(vec2.dot(vec2))
    if L1 == 0 or L2 == 0:
        return False
    cos = vec1.dot(vec2)/(L1*L2)
    angle = np.arccos(cos) * 360/(2*np.pi)
    
    return angle < angle_thr

def count(json_dir, video_dir, track_dir, save_dir):
	starttime = timeit.default_timer()

	remove_wrong_classes = config['remove_wrong_classes']
	min_track_len = config['tracker']['min_len']

	Path(save_dir).mkdir(parents=True, exist_ok=True)

	cam_datas = get_list_data(json_dir)



    # save result
	result_filename = os.path.join(save_dir, 'result.txt')
	with open(result_filename, 'w') as result_file:
		for video_name in sorted(total_moi_detections.keys()):
			for frame_id, movement_id, vehicle_class_id in total_moi_detections[video_name]:
				if not frame_id + 1 or not movement_id or not vehicle_class_id + 1:
					continue
				result_file.write('{} {} {} {}\n'.format('video_' + video_name, frame_id + 1, movement_id, vehicle_class_id + 1))
				
	endtime = timeit.default_timer()
	print('Track time: {} seconds'.format(endtime - starttime))

if __name__=='__main__':
	args = parse_args()

	json_dir = args['json_dir']
	video_dir = args['video_dir']
	track_dir = args['track_dir']
	save_dir = args['save_dir']

	count(json_dir, video_dir, track_dir, save_dir)