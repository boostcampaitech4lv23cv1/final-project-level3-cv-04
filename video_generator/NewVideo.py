import os
from matplotlib import pyplot as plt 
from glob import glob
import cv2
from tqdm import tqdm
import csv
import pandas as pd
import natsort
import numpy as np
from itertools import chain

def crop_img(idx, px,mx,py,my,path,newfolder):
    img = cv2.imread(path + '{0:06d}.jpg'.format(idx))
    h, w, _ = img.shape # 이미지 크기 받기
    # 사진 padding
    if px > w or mx < 0 or py > h or my < 0: # 사진 범위 벗어나면
        img,px,mx,py,my = img_padding(img,px,mx,py,my,w,h)
    
    cropped_img = img[my:py, mx:px]
    
    # 이미지 저장
    cv2.imwrite(newfolder+str(idx)+'.jpg', cropped_img)
    return

def full_img(idx,video_size_w,video_size_h,path,newfolder):
    # 이미지 불러오고 resize
    img = cv2.imread(path + '{0:06d}.jpg'.format(idx))
    resize_img = cv2.resize(img,(video_size_w,video_size_h))
    # 이미지 저장
    cv2.imwrite(newfolder+str(idx)+'.jpg',resize_img)
    return

def img_padding(img,px,mx,py,my,w,h):   #top, bottom, left, right
    if px > w:
        img = cv2.copyMakeBorder(img, 0,0,0,px-w,cv2.BORDER_CONSTANT)
        
    if mx < 0:
        img = cv2.copyMakeBorder(img, 0,0,-mx,0,cv2.BORDER_CONSTANT)
        px = px - mx
        mx = 0
    if py > h:
        img = cv2.copyMakeBorder(img, 0,py-h,0,0,cv2.BORDER_CONSTANT)
        
    if my < 0:
        img = cv2.copyMakeBorder(img, -my,0,0,0,cv2.BORDER_CONSTANT)
        py = py - my
        my = 0
        
    return img,px,mx,py,my

def video_df(df1,pred,member):
    df1 = df1.drop('face_embedding',axis=1)
    df1 = df1.drop('face_confidence',axis=1)
    trackID_by_member = []
    for k, v in pred.items():
        if v == member:
            trackID_by_member.append(k)

    video_df = df1[df1['track_id'].isin(trackID_by_member)]
    
    return video_df


def video_generator(df1,meta_info,member,pred):
    '''
    input
        df1 : filename,bbox,track_id
        img_list : image들의 path
        member(str형) : user가 원하는 member
        pred : predictor 거쳐서 나온 prediction -> 구조: pred = {'track_id' : 'aespa_karina'}
        full_video(boolean) ->  True  : ex) 카리나 없는 부분은 전체화면으로
                                False : ex) 카리나 없는 부분은 skip

    output
        얼굴 좌표 고려해서 항상 정중앙에 올 수 있게 + 상반신 보이게 crop해서
        output으로 .mp4파일 내뱉음
    '''
    
    ######################################################
    video_size_w = 1280 # 최종 video 크기 (가로)
    video_size_h = 720  # 최종 video 크기 (세로)
    newfolder =  './result/' + meta_info["image_root"].split('/')[-1] + '/img/'  # 사진 저장할 폴더
    video_path = './result/' + meta_info["image_root"].split('/')[-1] + '/video/'   # 비디오 저장할 폴더
    frame = meta_info["fps"]  # 비디오 프레임
    os.makedirs(newfolder,exist_ok=True)
    os.makedirs(video_path,exist_ok=True)
    ######################################################
    
    
    print('video_generator실행중')
    face_df = video_df(df1,pred,member)
    prev_px = 0  
    prev_mx = 0
    prev_py = 0
    prev_my = 0
    
    #이미지 주소
    path = meta_info["image_root"]+'/'
    img_list = glob(meta_info["image_root"]+'/*.jpg')
    img_list = natsort.natsorted(img_list)

    ###'aespa_karina' -> member 변수로 받아오게 수정
    img_len = int(((img_list[-1].split('/'))[-1].split('.'))[0])
    idx = 1
    while True:
        if idx > img_len:   # img 범위 벗어나면 while문 탈출
            break
        else:   #img 범위 내
            if '{0:06d}.jpg'.format(idx) in face_df['filename'].unique():    #해당 이미지가 face_df에 있으면
                if ('{0:06d}.jpg'.format(idx) in face_df['filename'].unique()) and (member in chain.from_iterable(face_df[face_df['filename'] == '{0:06d}.jpg'.format(idx)]['face_pred'].values)):
                    _series = face_df[face_df['filename'] == '{0:06d}.jpg'.format(idx)].iloc[0]
                    face_bbox = list(_series['face_bbox'][_series['face_pred'].index(member)])   # xmin, ymin xmax,ymax
                    center_x = (face_bbox[0]+face_bbox[2])/2    
                    center_y = (face_bbox[1]+face_bbox[3])/2
                    px = int(center_x + video_size_w/2)
                    mx = int(center_x - video_size_w/2)
                    py = int(center_y + video_size_h/2)
                    my = int(center_y - video_size_h/2)

                    #좌표 저장
                    prev_px = px    
                    prev_mx = mx
                    prev_py = py
                    prev_my = my
                                       
                    crop_img(idx,px,mx,py,my,path,newfolder)
                    idx += 1
                    
                else:   #filename같은데 karina 없으면 -> 이전 좌표 사용
                    if prev_px == 0 and prev_py == 0:
                        full_img(idx,video_size_w,video_size_h,path,newfolder)
                        idx += 1
                    else:
                        px = prev_px
                        mx = prev_mx
                        py = prev_py
                        my = prev_my
                        crop_img(idx,px,mx,py,my,path,newfolder)
                        idx += 1
                    
            else:   #해당 이미지가 face_df에 없으면->여기서 카리나 없는 이미지 작업하고 idx도 늘려줘서 두번 작업안하게
                fidx = idx+1
                fcount = 1
                #몇 frame동안 카리나 없는지 확인 -> fcount에 저장
                while True:
                    if fidx > img_len:    #총 이미지 수 보다 커지면 while문 탈출
                        break
                    elif '{0:06d}.jpg'.format(fidx) in face_df['filename'].unique(): # 카리나 있으면
                        break
                    else:   # 카리나 없으면
                        fcount += 1
                        fidx += 1

                #fcount 결과 가지고 full image or crop image 적용
                if fcount > (frame):  ### 1초보다 오래 full image 잡혀야 되면 줌인 줌아웃 효과
                    for _ in range(fcount):
                        full_img(idx,video_size_w,video_size_h,path,newfolder)
                        idx += 1
                else:   # 1초보다 짧게 full image 잡혀야 되면 기존에 True에서 잡았던 bbox의 중심 좌표를 계속해서 이용
                    # prev 좌표가 0,0인 경우 -> 전체화면
                    if prev_px == 0 and prev_py == 0:
                        for _ in range(fcount):
                            full_img(idx,video_size_w,video_size_h,path,newfolder)
                            idx += 1
                    #prev 좌표가 0,0이 아닌 경우 -> 이전 좌표로 crop        
                    else:
                        print('fcount : ',fcount)
                        for _ in range(fcount):
                            # 이전 center_x, center_y좌표 불러와서 crop
                            px = prev_px
                            mx = prev_mx
                            py = prev_py
                            my = prev_my
                            crop_img(idx,px,mx,py,my,path,newfolder)
                            idx += 1
    
    print('video 생성중...')
    img_list = glob(newfolder+"*.jpg")
    img_list = natsort.natsorted(img_list)
    out = cv2.VideoWriter(video_path+member+'_output.mp4',cv2.VideoWriter_fourcc(*'mp4v'),frame,(video_size_w,video_size_h))
    for path in img_list:
        img = cv2.imread(path)
        out.write(img)
    out.release()
    
    video_path = video_path+member+'_output.mp4'
        
    return video_path