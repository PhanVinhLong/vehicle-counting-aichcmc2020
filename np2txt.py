import os
import numpy as np

detect_dir = 'data/detect'
json_dir = '../data/json-custom/'
save_dir = 'data/detect-txt'

from utils import *

cam_datas = get_list_data(json_dir)

for cam_data in cam_datas:
    cam_name = cam_data['camName']
    savepath = os.path.join(save_dir, cam_name + '.txt')
    detect_res_path = os.path.join(detect_dir, cam_name + '.npy')
    bboxes = np.load(detect_res_path, allow_pickle=True)
    num_frame = bboxes.shape[0]
    with open(savepath, 'w') as f:
        for i in range(num_frame):
            for bbox in bboxes[i]:
                txtboxes = str(i) + ' ' + str(bbox[6]) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + ' ' + str(bbox[5]) + '\n'
                f.write(txtboxes)