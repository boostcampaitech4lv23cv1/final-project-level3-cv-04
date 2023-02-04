import natsort
from glob import glob
import face_embedding
import os
import cv2
from video_generator.NewVideo import video_generator

def app_video_maker(df1, meta_info, pred, save_dir):
    member_list = meta_info['member_list']
    member_video_paths = []
    for member in member_list:
        member_video_path = video_generator(df1, meta_info, member, pred, save_dir, face_loc=3, video_size=0.4)
        member_video_paths.append(member_video_path)
                        
    return member_video_paths