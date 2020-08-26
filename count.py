import numpy as np
import json
import timeit
import os
import argparse
from utils import *
from pathlib import Path

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
      paths[kk] = [(int(x), int(y)) for x, y
              in it['points']]
  return polygon, paths

def cosin_similarity(a2d, b2d):  
	a = np.array((a2d[1][0] - a2d[0][0], a2d[1][1 ]- a2d[0][1]))
	b = np.array((b2d[1][0] - b2d[0][1], b2d[1][1] - b2d[1][0]))
	return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def counting_moi(paths, moto_vector_list, class_id):

	moi_detection_list = []
	for moto_vector in moto_vector_list:
		max_cosin = -2
		movement_id = ''
		last_frame = 0
		for movement_label, movement_vector in paths.items():
			cosin = cosin_similarity(movement_vector, moto_vector)
			if cosin > max_cosin:
				max_cosin = cosin
				movement_id = movement_label
				last_frame = moto_vector[2]
		moi_detection_list.append((last_frame, movement_id, class_id))
	return moi_detection_list

def count(json_dir, video_dir, track_dir, save_dir):
	starttime = timeit.default_timer()

	remove_wrong_class = True

	Path(save_dir).mkdir(parents=True, exist_ok=True)

	cam_datas = get_list_data(json_dir)

	total_moi_detections = {}
	for cam_data in cam_datas:
		cam_name = cam_data['camName']
		width = int(cam_data['imageWidth'])
		height = int(cam_data['imageHeight'])
		track_res_path = os.path.join(track_dir, cam_name + '.npy')
		tracks = np.load(track_res_path, allow_pickle=True)
		
		polygon, paths = load_zone_anno(os.path.join(json_dir, cam_name + '.json'))
		
		num_classes = 4

		track_dict = []

		for class_id, class_tracks in enumerate(tracks):
			track_dict.append({})
			for frame_id, vehicle_tracks in enumerate(class_tracks):
				for track in vehicle_tracks:
					x1 = track[0]
					y1 = track[1]
					x2 = track[2]
					y2 = track[3]
					track_id = int(track[5])
					if track_id not in track_dict[class_id].keys():
						track_dict[class_id][track_id] = [(x1, y1, x2, y2, frame_id)]
					else:
						track_dict[class_id][track_id].append((x1, y1, x2, y2, frame_id))

		if remove_wrong_class:
			for class_id, class_tracks in enumerate(tracks):
				for frame_id, vehicle_tracks in enumerate(class_tracks):
					for track in vehicle_tracks:
						pass

		vector_list = []
		for class_id, class_track_dict in enumerate(track_dict):
			vector_list.append([])
			for tracker_id, tracker_list in class_track_dict.items():
				if len(tracker_list) > 1:
					first = tracker_list[0]
					last = tracker_list[-1]
					first_point = ((first[2] - first[0])/2, (first[3] - first[1])/2)
					last_point = ((last[2] - last[0])/2, (last[3] - last[1])/2)
					vector_list[class_id].append((first_point, last_point, last[4]))
			
		moi_detections = []
		for class_id, vehicle_vector_list in enumerate(vector_list):
			moi_detections.extend(counting_moi(paths, vehicle_vector_list, class_id))

		def custom_sort(e):
			return e[0]
		moi_detections.sort(key=custom_sort)
		total_moi_detections[cam_name] = moi_detections

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