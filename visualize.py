import argparse
import numpy as np
import os
import timeit
import json
import cv2
import random
import colorsys

import moviepy.editor as moviepy

from tqdm import tqdm
from pathlib import Path

from utils import *


def parse_args():
	argparser = argparse.ArgumentParser(
	    description='Data preparation for vehicle counting')
	argparser.add_argument('-j', '--json_dir', type=str, default='data/json/', help='Json directory')
	argparser.add_argument('-v', '--video_dir', type=str, default='data/video/', help='Video directory')
	argparser.add_argument('-d', '--detect_dir', type=str, default='data/detect', help='Detection result directory')
	argparser.add_argument('-t', '--track_dir', type=str, default='data/track', help='Tracking result directory')
	argparser.add_argument('-c', '--count_dir', type=str, default='', help='Counting result directory')
	argparser.add_argument('-s', '--save_dir', type=str, default='data/visualize', help='Save result')
	args = vars(argparser.parse_args())
	return args

def get_count(countfile):
	count_dict = {}
	with open(countfile, 'r') as result_file:
		while True:
			line = result_file.readline()
			if not line:
				break
			cam_name, frame, mm_id, class_id = line.split(' ')
			ccam_name = cam_name.replace('video_', '')
			cframe = int(frame)
			cmm_id = int(mm_id)
			cclass_id = int(class_id) - 1
			if ccam_name not in count_dict.keys():
				count_dict[ccam_name] = {}
				count_dict[ccam_name][cframe] = [(cmm_id, cclass_id)]
			elif cframe not in count_dict[ccam_name]:
				count_dict[ccam_name][cframe] = [(cmm_id, cclass_id)]
			else:
				count_dict[ccam_name][cframe].append((cmm_id, cclass_id))
	return count_dict

def cal_frame_count(count, count_dict, frame_id):
	if frame_id in count_dict.keys():
		for c in count_dict[frame_id]:
			count[c[0] - 1][c[1]] = count[c[0] - 1][c[1]] + 1
	return count

def visualize(json_dir, video_dir, detect_dir, track_dir, count_dir, save_dir):
	starttime = timeit.default_timer()

	class_names_track=['1', '2', '3', '4']
	class_names = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

	Path(save_dir).mkdir(parents=True, exist_ok=True)
	
	cam_datas = get_list_data(json_dir)

	max_color = 200
	hsv_tuples = [(1.0 * x / max_color, 1., 1.) for x in range(max_color)]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(
		map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

	random.seed(0)
	random.shuffle(colors)
	random.seed(None)
	
	count_dict = get_count(os.path.join(count_dir, 'result.txt'))

	for cam_data in cam_datas:
		cam_name = cam_data['camName']
		width = int(cam_data['imageWidth'])
		height = int(cam_data['imageHeight'])
		FPS = 10.0

		video_path = os.path.join(video_dir, cam_name + '.mp4')
		video_cap = cv2.VideoCapture(video_path)
		num_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
		
		if detect_dir:
			detect_res_path = os.path.join(detect_dir, cam_name + '.npy')
			bboxes = np.load(detect_res_path, allow_pickle=True)
			video_writer_det = cv2.VideoWriter(os.path.join(save_dir, cam_name + '_detect.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (width,height))

		if track_dir:
			track_res_path = os.path.join(track_dir, cam_name + '.npy')
			track_bboxes = np.load(track_res_path, allow_pickle=True)
			video_writer_track = cv2.VideoWriter(os.path.join(save_dir, cam_name + '_track.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (width,height))

		num_lines = len(cam_data['shapes']) - 1
		num_colors = num_lines + 1
		hsv_tuples = [(1.0 * x / num_colors, 1., 1.) for x in range(num_colors)]
		color_lines = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
		color_lines = list(
			map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color_lines))

		random.seed(0)
		random.shuffle(color_lines)
		random.seed(None)


		count = []
		for class_id in range(num_lines):
			count.append([0 for _ in range(len(class_names_track))])

		for i in tqdm(range(num_frames), desc='Visualizing {}'.format(cam_name)):
			success, img = video_cap.read()
			img2 = np.copy(img)
			
			# visualize detection
			if detect_dir:
				img = draw_bbox(img, bboxes[i], class_names)
				img = draw_moi(img, cam_data['shapes'])
				img = draw_frame(img, i)
				video_writer_det.write(img)

			# visualize tracking
			if track_dir:
				for class_id in range(len(class_names_track)):
					img2 = draw_bbox_track(img2, track_bboxes[class_id][i], class_names_track, colors)
				img2 = draw_frame(img2, i)
				img2 = draw_moi(img2, cam_data['shapes'], color_lines)
				if cam_name in count_dict.keys():
					count = cal_frame_count(count, count_dict[cam_name], i)
					img2 = draw_count(img2, count, color_lines, class_names_track)
				video_writer_track.write(img2)

		if detect_dir:
			video_writer_det.release()
			clip = moviepy.VideoFileClip(os.path.join(save_dir, cam_name + '_detect.avi'), verbose=False)
			clip.write_videofile(os.path.join(save_dir, cam_name + '_detect.mp4'))
			os.remove(os.path.join(save_dir, cam_name + '_detect.avi'))
		
		if track_dir:
			video_writer_track.release()
			clip = moviepy.VideoFileClip(os.path.join(save_dir, cam_name + '_track.avi'), verbose=False)
			clip.write_videofile(os.path.join(save_dir, cam_name + '_track.mp4'))
			os.remove(os.path.join(save_dir, cam_name + '_track.avi'))

	endtime = timeit.default_timer()
	
	print('Visualize time: {} seconds'.format(endtime - starttime))

if __name__=='__main__':
	args = parse_args()

	json_dir = args['json_dir']
	video_dir = args['video_dir']
	detect_dir = args['detect_dir']
	track_dir = args['track_dir']
	count_dir = args['count_dir']
	save_dir = args['save_dir']

	visualize(json_dir, video_dir, detect_dir, track_dir, count_dir, save_dir)
