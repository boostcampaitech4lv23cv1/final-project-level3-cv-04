import json
import numpy as np
import pandas as pd
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

import face_embedding

import cv2 as cv
from tqdm import tqdm



class GroupRecognizer():
    def __init__(self,
                 meta_info: dict,
                 anchors: dict,                     # 비교할 anchors 있는 dict 파일
                 num_samples_to_use: int = 50,      # TODO : 사용할 샘플 수 제한하는건 아직 안함. 그냥 모든 샘플 다 활용
                 every_samples: np.ndarray = None,
                 ):
        self.meta_info = meta_info
        self.num_samples_to_use = num_samples_to_use
        self.every_samples = every_samples
        self.anchors = anchors
        self.groups = self.__get_groups(self.anchors)
        print('INFO: Group Recognizer initialized. loaded groups : {}'.format(self.groups))
    
    
    # 본 프로젝트에 맞게 하려면, 해당 함수를 실행시켜서
    # df1, df2 를 인자로 넣어주면 됨
    def register_dataframes(self,
                            df1: pd.DataFrame,
                            df2: pd.DataFrame):
        
        # 전체 샘플의 갯수가 뽑고자 하는 샘플의 수보다 적다면
        if len(df2) < self.num_samples_to_use:
            self.num_samples_to_use = len(df2)
        
        _every_samples = df2['df1_index'].values
        _every_samples = df1.loc[_every_samples, ['filename', 'track_body_xmin', 'track_body_ymin', 'track_body_xmax', 'track_body_ymax']]
        _every_samples['filename'] = _every_samples['filename'].apply(lambda x : osp.join(self.meta_info['image_root'], x))

        print('INFO: Group Recognizer - reading {} raw image files'.format(len(_every_samples)))
        self.every_samples = list(map(lambda x : cv.imread(x[1])[int(x[3]):int(x[5]), int(x[2]):int(x[4])], tqdm(_every_samples.itertuples())))
        
        
    def __get_groups(self,
                   anchors):
        groups = set()
        for k, v in anchors.items():
            _groupname = k.split('_')[0]
            groups.add(_groupname)
            
        return list(groups)
            
        
    def guess_group(self,
                    ):
        print('INFO: Group Recognizer - extracting features from {} bbox images. . .'.format(len(self.every_samples)))
        every_features = list(map(face_embedding.detect_face_and_extract_feature, tqdm(self.every_samples)))
        detected_features = []
        for f in every_features:
            if type(f) == str and f == 'FLAG':
                continue
            else:
                detected_features.append(f)
        print('INFO: Group Recognizer - filtered out Non-Face images, {} bbox images remained. . .'.format(len(detected_features)))
        
        print('INFO: Group Recognizer - computing similarity with {} face anchors. . .'.format(len(self.anchors)))
        every_guesses = list(map(lambda x : face_embedding.compute_face_confidence_all(x, self.anchors), tqdm(detected_features)))
        
        votes = dict()
        for name in self.anchors.keys():
            votes[name] = 0
        
        
        for guess in every_guesses:
            guess = guess[0]
            top_name = None
            top_conf = 0
            for k, v in guess.items():
                if v > top_conf:
                    top_conf = v
                    top_name = k
            if top_name:
                votes[top_name] += 1
            
        self.votes = votes
        
        group_votes = {k : 0 for k in self.groups}
        for k, v in self.votes.items():
            _group = k.split('_')[0]
            group_votes[_group] += v
            
        max_votes = 0
        result = None
        for k, v in group_votes.items():
            if v > max_votes:
                max_votes = v
                result = k
            
        print('-'*10)
        print('INFO: Group Recognizer - result')
        for k, v in group_votes.items():
            print('{} : matches for {} bboxes'.format(k, v))
        print('')
        print('result : {} confirmed'.format(result))
        print('-'*10)
        
        all_members = set()
        for member_name in self.anchors.keys():
            if member_name.split('_')[0] == result:
                all_members.add(member_name)
        all_members = list(all_members)
        
        self.meta_info['group'] = result
        self.meta_info['member_list'] = all_members
        
        return self.meta_info