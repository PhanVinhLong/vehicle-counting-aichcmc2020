config = {
    'detector': {
        'classesmap': [[1, 3], [2], [5], [7]],
        'classnames': ['type 1', 'type 2', 'type 3', 'type 4'],
        'cfgfile': 'yolov4/cfg/yolov4.cfg',
        'weightfile': 'yolov4/yolov4.weights',
        'classnamefile': 'yolov4/data/coco.names'
    },
    'tracker': {
        'modelfile': 'deep_sort/model_data/mars-small128.pb',
        'max_cosine_distance': 0.3,
        'nn_budget': None,
        'nms_max_overlap': 1.0,
        'min_length': 5
    },
    'counter': {
        'dist_thr': 300,
        'angle_thr': 30,
    },
    'remove_wrong_classes': True,
    'remove_not_intersec_moi': True
}