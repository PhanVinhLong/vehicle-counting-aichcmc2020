import numpy as np
import json
import timeit
import os
import argparse
from pathlib import Path
import sys
from shapely.geometry import Polygon

sys.path.append(os.path.realpath('count2'))
from hausdorff_dist import hausdorff_distance

sys.path.append(os.path.realpath('yolov4'))
from tool.utils import *

from config import config
from utils import *

def parse_args():
	argparser = argparse.ArgumentParser(
		description='Data preparation for vehicle counting')
	argparser.add_argument('-j', '--json_dir', type=str,
						   default='../data/json/', help='Json directory')
	argparser.add_argument('-v', '--video_dir', type=str,
						   default='../data/video/', help='Video directory')
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

	results = []
	
	for cam_data in cam_datas:
		cam_name = cam_data['camName']
		width = int(cam_data['imageWidth'])
		height = int(cam_data['imageHeight'])
		track_res_path = os.path.join(track_dir, cam_name + '.npy')
		tracks = np.load(track_res_path, allow_pickle=True)

		mm_track = {}
		tipical_trajs = {}
		for mm_id, mm in enumerate(cam_data['shapes'][1:]):
			if 'tracklets' in mm.keys():
				tipical_trajs[mm_id] = [mm['tracklets']]
			else:
				tipical_trajs[mm_id] = [mm['points']]

		track_dict = []
		for class_id, class_tracks in enumerate(tracks):
			track_dict.append({})
			for frame_id, vehicle_tracks in enumerate(class_tracks):
				for track in vehicle_tracks:
					x1 = track[0]
					y1 = track[1]
					x2 = track[2]
					y2 = track[3]
					cx = int((x1 + x2) / 2)
					cy = int((y1 + y2) / 2)
					track_id = int(track[5])
					if track_id in track_dict[class_id]:
						track_dict[class_id][track_id]['endframe'] = frame_id
						track_dict[class_id][track_id]['bbox'].append([frame_id, x1, y1, x2, y2, class_id])
						track_dict[class_id][track_id]['tracklet'].append([cx, cy])
					else:
						track_dict[class_id][track_id] = {'startframe' : frame_id,
										'endframe' : frame_id,
										'bbox' : [[frame_id, x1, y1, x2, y2, class_id]],
										'tracklet' : [[cx, cy]]}
						
		for class_id, _ in enumerate(track_dict):
			mm_track[class_id] = {}
			track_ids = sorted([k for k in track_dict[class_id].keys()])
			for track_id in track_ids:
				if len(track_dict[class_id][track_id]['tracklet']) < config['tracker']['min_len']:
					continue
				track_traj = track_dict[class_id][track_id]['tracklet']

				# calc hausdorff dist with tipical trajs, assign the movement with the min dist
				all_dists_dict = {k: float('inf') for k in tipical_trajs}
				for m_id, m_t in tipical_trajs.items():
					for t in m_t:
						tmp_dist = hausdorff_distance(np.array(track_traj), np.array(t), distance='euclidean')
						if tmp_dist < all_dists_dict[m_id]:
							all_dists_dict[m_id] = tmp_dist
				
				# check direction
				all_dists = sorted(all_dists_dict.items(), key=lambda k: k[1])
				min_idx, min_dist = None, config['counter']['dist_thr']
				for i in range(0, len(all_dists)):
					m_id = all_dists[i][0]
					m_dist = all_dists[i][1]
					if m_dist >= config['counter']['dist_thr']: #if min dist > dist_thr, will not assign to any movement
						break
					else:
						if is_same_direction(track_traj, tipical_trajs[m_id][0], config['counter']['angle_thr']): #check direction
							min_idx = m_id
							min_dist = m_dist
							break # if match, end
						else:
							continue # direction not matched, find next m_id

				if min_idx == None and min_dist >= config['counter']['dist_thr']:
					continue
				#save counting results
				mv_idx = min_idx
				#get last frameid in roi
				bboxes = track_dict[class_id][track_id]['bbox']
				bboxes.sort(key=lambda x: x[0])

				dst_frame = bboxes[0][0]
				last_bbox = bboxes[-1]
				roi = cam_data['shapes'][0]['points']
				if check_bbox_overlap_with_roi(last_bbox, roi) == True:
					dst_frame = last_bbox[0]
				else:
					for i in range(len(bboxes) - 2, 0, -1):
						bbox = bboxes[i]
						if check_bbox_overlap_with_roi(bbox, roi) == True:
							dst_frame = bbox[0]
							break
						else:
							continue

				track_types = [k[5] for k in bboxes]
				track_type = max(track_types, key=track_types.count)

				mm_track[class_id][track_id] = mv_idx
				results.append([cam_name, dst_frame, mv_idx, class_id])


		filepath = os.path.join(save_dir, cam_name + '.json')
		with open(filepath, 'w') as f:
			json.dump(mm_track, f)

	results.sort(key=lambda x: ([x[0], x[1], x[2], x[3]]))
	
	result_filename = os.path.join(save_dir, 'result.txt')
	with open(result_filename, 'w') as result_file:
		for result in results:
			result_file.write('{} {} {} {}\n'.format(result[0], result[1] + 1, result[2] + 1, result[3] + 1))

	endtime = timeit.default_timer()
	print('Track time: {} seconds'.format(endtime - starttime))

if __name__=='__main__':
	args = parse_args()

	json_dir = args['json_dir']
	video_dir = args['video_dir']
	track_dir = args['track_dir']
	save_dir = args['save_dir']

	count(json_dir, video_dir, track_dir, save_dir)