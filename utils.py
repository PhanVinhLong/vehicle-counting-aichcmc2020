import numpy as np
import cv2
import math
import random
import colorsys
import os
import json


def get_list_data(jsondir):
	cam_datas = []
	for (dirpath, dirnames, filenames) in os.walk(jsondir):
		for filename in filenames:
			ext = filename.split('.')[-1]
			if ext == 'json':
				cam_name = filename.split('.' + ext)[0]
				cam_data = {}
				with open(os.path.join(dirpath, filename)) as f:
					cam_data = json.load(f)
					cam_data['camName'] = cam_name
					for idx, mm in enumerate(cam_data['shapes']):
						if mm['label'] == 'zone':
							cam_data['shapes'][idx]['id'] = '00'
						else:
							cam_data['shapes'][idx]['id'] = mm['label'][-2:]
					cam_data['shapes'].sort(key=lambda x: x['id'])
					cam_datas.append(cam_data)
	cam_datas.sort(key=lambda x: x['camName'])
	return cam_datas


def plot_boxes(img, boxes, class_names=None, color=None):
	img = np.copy(img)
	colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [
		1, 1, 0], [1, 0, 0]], dtype=np.float32)

	def get_color(c, x, max_val):
		ratio = float(x) / max_val * 5
		i = int(math.floor(ratio))
		j = int(math.ceil(ratio))
		ratio = ratio - i
		r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
		return int(r * 255)

	# width = img.shape[1]
	# height = img.shape[0]
	for i in range(len(boxes)):
		box = boxes[i]
		x1 = int(box[0])
		y1 = int(box[1])
		x2 = int(box[2])
		y2 = int(box[3])

		if color:
			rgb = color
		else:
			rgb = (255, 0, 0)
		if len(box) >= 7 and class_names:
			cls_conf = box[5]
			cls_id = box[6]
			classes = len(class_names)
			offset = cls_id * 123457 % classes
			red = get_color(2, offset, classes)
			green = get_color(1, offset, classes)
			blue = get_color(0, offset, classes)
			if color is None:
				rgb = (red, green, blue)
			img = cv2.putText(
				img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
		img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
	return img


def draw_bbox(img, bboxes, classes):
	show_label = True
	image_h, image_w, _ = img.shape
	num_classes = len(classes)
	hsv_tuples = [(1.0 * x / num_classes + 10, 1., 1.)
				  for x in range(num_classes + 10)]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(
		map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

	# out_boxes, out_scores, out_classes, num_boxes = bboxes
	for i in range(len(bboxes)):
		x1 = int(bboxes[i][0])
		y1 = int(bboxes[i][1])
		x2 = int(bboxes[i][2])
		y2 = int(bboxes[i][3])
		conf = float(bboxes[i][5])
		class_id = int(bboxes[i][6])
		if class_id < 0 or class_id > num_classes:
			continue

		fontScale = 0.5
		score = conf
		class_ind = class_id
		bbox_color = colors[class_id]
		bbox_thick = int(0.6 * (image_h + image_w) / 600)
		c1, c2 = (x1, y1), (x2, y2)
		cv2.rectangle(img, c1, c2, bbox_color, bbox_thick)

		if show_label:
			bbox_mess = '%s: %.2f' % (classes[class_ind], score)
			t_size = cv2.getTextSize(
				bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
			c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
			cv2.rectangle(img, c1, (np.float32(c3[0]), np.float32(
				c3[1])), bbox_color, -1)  # filled

			cv2.putText(img, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
						fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
	return img


def draw_bbox_track(image, bboxes, classes, colors, mm_track, mm_colors):
	show_label = True
	image_h, image_w, _ = image.shape

	# out_boxes, out_scores, out_classes, num_boxes = bboxes
	for i in range(len(bboxes)):
		x1 = int(bboxes[i][0])
		y1 = int(bboxes[i][1])
		x2 = int(bboxes[i][2])
		y2 = int(bboxes[i][3])
		ox = int((x2 + x1) / 2)
		oy = int((y2 + y1) / 2)
		class_id = int(bboxes[i][4])
		track_id = int(bboxes[i][5])

		fontScale = 0.5
		class_ind = class_id
		bbox_color = colors[track_id % 200]

		bbox_thick = int(0.6 * (image_h + image_w) / 600)
		c1, c2 = (x1, y1), (x2, y2)
		cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

		if show_label:
			bbox_mess = '%s - %d' % (classes[class_ind], track_id)
			t_size = cv2.getTextSize(
				bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
			c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
			cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(
				c3[1])), bbox_color, -1)  # filled

			cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
						fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
			if str(track_id) in mm_track[str(class_id)].keys():
				mm_color_id = mm_track[str(class_id)][str(track_id)] + 1
				cv2.circle(image, (ox, oy), bbox_thick * 5, mm_colors[mm_color_id], -1)
	return image


def draw_moi(img, moi, colors=None):
	image_h, image_w, _ = img.shape

	max_iter = 30
	hsv_tuples = [(1.0 * x / max_iter, 1., 1.) for x in range(max_iter)]

	if colors == None:
		colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
		colors = list(
			map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

		random.seed(1)
		random.shuffle(colors)
		random.seed(None)

	fontScale = 0.5
	bbox_thick = int(0.6 * (image_h + image_w) / 600)

	for i, data in enumerate(moi):
		bbox_color = colors[i]
		if data['label'] == 'zone':
			img = cv2.polylines(
				img, [np.array(data['points'], dtype=np.int32)], True, bbox_color, bbox_thick)
		else:
			points = []
			if 'tracklets' in data.keys():
				points = data['tracklets']
			else:
				points = data['points']

			img = cv2.arrowedLine(img, tuple(np.array(data['points'][-2], dtype=np.int32)), tuple(
				np.array(data['points'][-1], dtype=np.int32)), bbox_color, bbox_thick, 8, 0, 0.02)
			if len(points) > 2:
				for idx in range(len(points) - 2):
					img = cv2.line(img, tuple(np.array(data['points'][idx], dtype=np.int32)), tuple(
						np.array(data['points'][idx + 1], dtype=np.int32)), bbox_color, bbox_thick, 8, 0)
	return img


def draw_count(img, counts, colors=None, class_names=None):

	image_h, image_w, _ = img.shape
	fontScale = 0.5
	bbox_thick = int(0.6 * (image_h + image_w) / 600)

	x, y, w, h = image_w - 190, 5, 185, 25 + 15 * len(counts)
	sub_img = img[y:y+h, x:x+w]
	white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
	res = cv2.addWeighted(sub_img, 0.6, white_rect, 0.6, 1.0)
	img[y:y+h, x:x+w] = res

	# draw classes label
	if class_names:
		mess = ''
		for class_name in class_names:
			num_space = 4
			mess = mess + str(class_name) + ' ' * num_space
		cv2.putText(img, mess, (image_w - 140, 20), cv2.FONT_HERSHEY_SIMPLEX,
					0.4, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

	# draw counting result
	for i, line_count in enumerate(counts):
		for j, count in enumerate(line_count):
			cv2.putText(img, str(count), (image_w - 140 + j * 33, 35 + i * 15), cv2.FONT_HERSHEY_SIMPLEX,
						0.4, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
		cv2.rectangle(img, (image_w - 180, 25 + i * 15),
					  (image_w - 165 + 15, 35 + i * 15), colors[i+1], bbox_thick)
	return img


def draw_frame(img, frame_id):
	image_h, image_w, _ = img.shape
	fontScale = 0.5
	bbox_thick = int(0.6 * (image_h + image_w) / 600)
	cv2.putText(img, str(frame_id), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
				fontScale, (255, 255, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
	return img
