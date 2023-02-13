import os
import os.path as osp
import numpy as np
import pandas as pd
import pickle
import cv2 as cv
import shutil
from tqdm import tqdm


def vis_debug(INPUT_PATH, fps, remove_captures_afterward = True):
    os.makedirs(osp.join(INPUT_PATH, 'debug'), exist_ok=True)
    print('INFO - saving TEMP images at {}'.format(osp.join(INPUT_PATH, 'debug')))
    
    with open(osp.join(INPUT_PATH, 'csv', 'df1_face.pickle'), 'rb') as f:
        df1 = pickle.load(f)
    
    with open(osp.join(INPUT_PATH, 'csv', 'pred.pickle'), 'rb') as f:
        pred = pickle.load(f)
        
    MCOLOR = ((255,0,0), (0,255,0), (0,0,255), (127,127,0), (127,0,127), (0,127,127))

    all_members = dict()
    _NEXT = 0
    for k, v in pred.items():
        if v not in all_members.keys() and v != 'NO_FACE_DETECTED':
            all_members[v] = _NEXT
            _NEXT += 1
            
    
    all_imgs = sorted(os.listdir(osp.join(INPUT_PATH, 'captures')))
    
    for filename in tqdm(all_imgs):
        img = cv.imread(osp.join(INPUT_PATH, 'captures', filename))

        # Pallette
        for i, (member, num) in enumerate(all_members.items()):
            color = MCOLOR[num]
            img = cv.rectangle(img, (60, 60+60*i), (120, 120+60*i), color, -1)
            img = cv.putText(img, member, (120, 120+60*i), fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=3, color=color, thickness=2)
            
            
        _detections = df1[df1['filename']==filename]

        for row in _detections.itertuples():
            bbox = list(map(int, [row.track_body_xmin, row.track_body_ymin, row.track_body_xmax, row.track_body_ymax]))
            tID = int(row.track_id)
            if tID in pred.keys():
                label = pred[tID]
                if label == 'NO_FACE_DETECTED':
                    boxcolor = (30, 30, 30)
                    label = 'NONE_NOFACE'
                else:
                    boxcolor = MCOLOR[all_members[label]]
            else:
                label = 'NONE_NONE'
                boxcolor = (30, 30, 30)
                
            try:
                facebboxes = row.face_bbox.astype(int)
                facelabels = row.face_pred
            except:
                facebboxes = False
            
            img = cv.rectangle(img, bbox[:2], bbox[2:], boxcolor, thickness=6)
            img = cv.putText(img, f'{tID}, {label.split("_")[1]}', [bbox[0], bbox[1]-10], fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=5, color=boxcolor, thickness=3)
            
            if type(facebboxes) != bool:
                for fb, fp in zip(facebboxes, facelabels):
                    img = cv.rectangle(img, fb[:2], fb[2:4], MCOLOR[all_members[fp]], 5)
        
        cv.resize(img, (1920, 1080))
        cv.imwrite(osp.join(INPUT_PATH, 'debug', filename), img)
    
    os.system('ffmpeg -f image2 -r {} -i {}/%6d.jpg ./debugged.mp4'.format(fps, osp.join(INPUT_PATH, 'debug')))
    print('output file created at {}'.format(osp.join('.', 'debugged.mp4')))
    
    if remove_captures_afterward:
        shutil.rmtree(osp.join(INPUT_PATH, 'debug'))
    print('INFO - removed TEMP folder at {}'.format(osp.join(INPUT_PATH, 'debug')))
    
    
    
if __name__ == '__main__':
    fps = 30
    
    # 원본 영상이 4k 정도일때 예쁘게 나옵니다. fhd 수준이라면 문의주십쇼..
    
    vis_debug('/opt/ml/torchkpop/result/YLy4iXLu0qI/0_185', fps)
    
    
    
    