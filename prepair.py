import os
import argparse
import json
from tqdm import tqdm
import cv2
import timeit
from pathlib import Path

def parse_args():
    argparser = argparse.ArgumentParser(description='Data preparation for vehicle counting')
    argparser.add_argument('-d', '--data_dir', required=True, type=str, default='data', help='Data directory')
    args = vars(argparser.parse_args())
    return args

def get_list_data(jsondir: str):
    cam_datas = []
    for (dirpath, dirnames, filenames) in os.walk(jsondir):
        for filename in filenames:
            ext = filename.split('.')[-1]
            if ext == 'json':
                cam_name = filename.split('.' + ext)[0]
                cam_data = {}
                with open(dirpath + filename) as f:
                    cam_data = json.load(f)
                    cam_data['camName'] = cam_name
                    cam_datas.append(cam_data)
    return cam_datas

def prepair():
    starttime = timeit.default_timer()

    args = parse_args()

    data_dir = args['data_dir']

    cam_datas = get_list_data(data_dir)
    
    for cam_data in cam_datas:
        cam_name = cam_data['camName']
        img_dir = os.path.join(data_dir, "frames", cam_name)
        
        Path(img_dir).mkdir(parents=True, exist_ok=True)

        video_path = os.path.join(data_dir, cam_name + '.mp4')
        video_cap = cv2.VideoCapture(video_path)
        num_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for i in tqdm(range(num_frames), desc='Extracting {}'.format(cam_name)):
            success, img = video_cap.read()
            img_name = str(i) + '.jpg'
            img_path = os.path.join(img_dir, img_name)
            cv2.imwrite(img_path, img)

    endtime = timeit.default_timer()
    
    print('Prepair time: {} seconds'.format(endtime - starttime))

if __name__=="__main__":
    prepair()
    