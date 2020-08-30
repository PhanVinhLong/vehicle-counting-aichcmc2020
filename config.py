detector = {
	'detector-yolo-152': {
		'type': 'yolo',
		'classesmap': [[0], [1], [2], [3]],
		# 'classesmap': [[0], [1], [2], [3]],
		'classnames': ['1', '2', '3', '4'],
		# 'originclassnames': ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'],
		'originclassnames': ['loai 1', 'loai 2', 'loai 3', 'loai 4', 'loai 5'],
		'cfgfile': 'yolov4/152-model/yolov4.cfg',
		'weightfile': 'yolov4/152-model/yolov4.weights',
		'classnamefile': 'yolov4/data/coco.names',
		'batchsize': 4
	},
	'detector-faster-pretrain': {
		'type': 'detectron',
		'classesmap': [[1, 3], [2], [5], [7]],
		'classnames': ['1', '2', '3', '4'],
		'originclassnames': ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'],
		'cfgfile': 'detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
		'weightfile': 'detectron2/model/faster_rcnn_R_101_FPN_3x.pkl',
		'batchsize': 20,
		'confidencethreshold': 0.5
	},
		'detector-yolo-pretrain': {
		'type': 'yolo',
		'classesmap': [[1, 3], [2], [5], [7]],
		'classnames': ['1', '2', '3', '4'],
		'originclassnames': ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'],
		'originclassnames': ['loai 1', 'loai 2', 'loai 3', 'loai 4', 'loai 5'],
		'cfgfile': 'yolov4/pretrained/yolov4.cfg',
		'weightfile': 'yolov4/pretrained/yolov4.weights',
		'classnamefile': 'yolov4/data/coco.names',
		'batchsize': 4
	}
}

config = {
	'detector': detector['detector-faster-pretrain'],
	'tracker': {
		'modelfile': 'deep_sort/model_data/mars-small128.pb',
		'max_cosine_distance': 0.3,
		'nn_budget': None,
		'nms_max_overlap': 1.0,
		'min_len': 5
	},
	'counter': {
		'dist_thr': 300,
		'angle_thr': 30,
	},
	'remove_wrong_classes': True,
	'remove_not_intersec_moi': True
}
