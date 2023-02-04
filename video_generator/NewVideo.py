import os
import os.path as osp
from glob import glob
import cv2
import csv
import pandas as pd
import natsort
import numpy as np
from itertools import chain


def crop_img(idx, px,mx,py,my,path,make_video_img_dir,video_size_w,video_size_h):
    '''
    이미지 crop해서 저장까지
    '''
    img = cv2.imread(path + '{0:06d}.jpg'.format(idx))
    h, w, _ = img.shape # 이미지 크기 받기
    # 사진 padding
    if px > w or mx < 0 or py > h or my < 0: # 사진 범위 벗어나면
        img,px,mx,py,my = img_padding(img,px,mx,py,my,w,h)
    
    cropped_img = img[my:py, mx:px]
    resize_img = cv2.resize(cropped_img,(video_size_w,video_size_h))
    # 이미지 저장
    cv2.imwrite(make_video_img_dir+str(idx)+'.jpg', resize_img)
    return

def full_img(idx,video_size_w,video_size_h,path,make_video_img_dir):
    # 이미지 불러오고 resize
    img = cv2.imread(path + '{0:06d}.jpg'.format(idx))
    resize_img = cv2.resize(img,(video_size_w,video_size_h))
    # 이미지 저장
    cv2.imwrite(make_video_img_dir+str(idx)+'.jpg',resize_img)
    return

def img_padding(img,px,mx,py,my,w,h):   # img 사이즈 범위 밖으로 벗어나면 그에맞게 padding
    '''
    px : top
    mx : bottom
    py : left
    my : right
    '''
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

def video_df(df1,pred,member):  # 들어온 df를 깔끔하게 정리
    df1 = df1.drop('face_embedding',axis=1)
    df1 = df1.drop('track_body_xmin',axis=1)
    df1 = df1.drop('track_body_ymin',axis=1)
    df1 = df1.drop('track_body_xmax',axis=1)
    df1 = df1.drop('track_body_ymax',axis=1)
    df1 = df1.drop('num_overlap_bboxes',axis=1)
    df1 = df1.drop('intercept_iou',axis=1)
    df1 = df1.drop('isfront',axis=1)
    trackID_by_member = []
    for k, v in pred.items():
        if v == member:
            trackID_by_member.append(k)

    video_df = df1[df1['track_id'].isin(trackID_by_member)]
    
    return video_df

def interpolation(start,end,frame):
    '''
    input
        start : (s_xmin,s_ymin,s_xmax,s_ymax)
        end : (e_xmin,e_ymin,e_xmax,e_ymax)
    
    output
        start좌표와 end좌표를 frame수로 평균낸만큼 각 frame별로 이동시킨 좌표를 2차원리스트에 저장
    '''
    
    s_xmin,s_ymin,s_xmax,s_ymax = start
    e_xmin,e_ymin,e_xmax,e_ymax = end
    
    #평균 이동 거리
    av_xmin = (e_xmin - s_xmin)/frame
    av_ymin = (e_ymin - s_ymin)/frame
    av_xmax = (e_xmax - s_xmax)/frame
    av_ymax = (e_ymax - s_ymax)/frame
    
    coordinates = []
    for _ in range(frame-1):
        s_xmin = int(s_xmin+av_xmin)
        s_ymin = int(s_ymin+av_ymin)
        s_xmax = int(s_xmax+av_xmax)
        s_ymax = int(s_ymax+av_ymax)
        coordinates.append([s_xmin,s_ymin,s_xmax,s_ymax])
    coordinates.append([e_xmin,e_ymin,e_xmax,e_ymax])
        
    return coordinates


def video_generator(df1,meta_info,member,pred, save_dir,face_loc=3,video_size=0.4):
    '''
    input
        df1 : filename,face_keypoint,track_id,face_pred
        meta_info : 영상 정보(image_root, width, height, frame)
        save_dir : 이미지, 영상 저장 경로
        member(str형) : user가 원하는 member
        pred : predictor 거쳐서 나온 prediction -> 구조: pred = {'track_id' : 'aespa_karina'}
        face_loc : face_location (1~10)
        video_size : crop된 영상의 비율 조절 (0~1)
    output
        얼굴 좌표 고려해서 항상 정중앙에 올 수 있게 + 상반신 보이게 crop해서
        output으로 영상 저장경로 내뱉음.
    '''
    
    ######################################################
    video_size_w = int(meta_info['width'] * video_size)
    video_size_h = int(meta_info['height'] * video_size)
    make_video_img_dir = osp.join(save_dir, f'make_video_img_{member}') + '/'
    save_video_dir = osp.join(save_dir, f'make_video_video_{member}') + '/'
    frame = meta_info["fps"]  # 비디오 프레임
    os.makedirs(make_video_img_dir,exist_ok=True)
    os.makedirs(save_video_dir,exist_ok=True)
    ######################################################
    
    
    print('video_generator실행중')
    face_df = video_df(df1,pred,member)
    prev_px,prev_mx,prev_py,prev_my = 0,0,0,0   # 시작 좌표
    end_px,end_mx,end_py,end_my= 0,0,0,0    # 끝 좌표
    
    #이미지 주소
    path = meta_info["image_root"]+'/'
    img_list = glob(meta_info["image_root"]+'/*.jpg')
    img_list = natsort.natsorted(img_list)
    img_len = int(((img_list[-1].split('/'))[-1].split('.'))[0])
    
    idx = 1
    
    while True:
        if idx > img_len:   # img 범위 벗어나면 while문 탈출
            break
        else:   #img 범위 내
            mem_in_img = list(face_df['face_pred'][(face_df['filename']=='{0:06}.jpg'.format(idx))])
            mem_in_img.append(['temp'])
            if member in mem_in_img[0]: #image에 member가 있을 경우
                _series = face_df[face_df['filename'] == '{0:06d}.jpg'.format(idx)].iloc[0]
                face_keypoints = list(_series['face_keypoint'][_series['face_pred'].index(member)])   # xmin, ymin xmax,ymax
                # print(face_keypoints)
                eye = face_keypoints[0] + face_keypoints[1]
                center_x = (float(eye[0]))/2    
                center_y = (float(eye[1]))/2
                px = int(center_x + video_size_w/2)                 # 오른쪽 아래 x 좌표
                mx = int(center_x - video_size_w/2)                 # 왼쪽 위 x 좌표
                py = int(center_y + video_size_h*(10-face_loc)/10)  # 오른쪽 아래 y 좌표
                my = int(center_y - video_size_h*face_loc/10)       # 왼쪽 위 y 좌표


                #좌표 저장
                prev_px = px    
                prev_mx = mx
                prev_py = py
                prev_my = my
                
                crop_img(idx,px,mx,py,my,path,make_video_img_dir,video_size_w,video_size_h)
                idx += 1

            else:   #그 외의 모든 경우
                fidx = idx  
                fcount = 0
                #몇 frame동안 해당 멤버 없는지 확인 -> fcount에 저장
                while True:
                    mem_in_img = list(face_df['face_pred'][(face_df['filename']=='{0:06}.jpg'.format(fidx))])
                    mem_in_img.append(['temp'])
                    if fidx >= img_len:    #총 이미지 수 보다 커지면 while문 탈출
                        break
                    elif member in mem_in_img[0]: # image에 member가 존재하면
                        _series = face_df[face_df['filename'] == '{0:06d}.jpg'.format(fidx)].iloc[0]
                        face_keypoints = list(_series['face_keypoint'][_series['face_pred'].index(member)])   # xmin, ymin xmax,ymax
                        eye = face_keypoints[0] + face_keypoints[1]
                        center_x = (float(eye[0]))/2    
                        center_y = (float(eye[1]))/2
                        px = int(center_x + video_size_w/2)
                        mx = int(center_x - video_size_w/2)
                        py = int(center_y + video_size_h*(10-face_loc)/10)
                        my = int(center_y - video_size_h*face_loc/10)
                        #좌표 저장
                        end_px = px
                        end_mx = mx
                        end_py = py
                        end_my = my
                        break
                    else:   # 해당 멤버 없으면
                        fcount += 1
                        fidx += 1

                #fcount 결과 가지고 full image or crop image 적용
                if fcount > (frame):  ### 1초보다 오래 full image 잡혀야 되면 줌인 줌아웃 효과
                    for _ in range(fcount):
                        full_img(idx,video_size_w,video_size_h,path,make_video_img_dir)
                        idx += 1
                else:   # 1초보다 짧게 full image 잡혀야 되면 기존에 True에서 잡았던 bbox의 중심 좌표를 계속해서 이용
                    # prev 좌표가 0,0인 경우 -> 전체화면
                    if prev_px == 0 and prev_py == 0:
                        for _ in range(fcount):
                            full_img(idx,video_size_w,video_size_h,path,make_video_img_dir)
                            idx += 1
                    #prev 좌표가 0,0이 아닌 경우 -> 이전 좌표로 crop        
                    else:
                        start = (prev_mx,prev_my,prev_px,prev_py)
                        end = (end_mx,end_my,end_px,end_py)
                        coordinates = interpolation(start,end,fcount+1)
                        for c in coordinates:
                            mx,my,px,py = c[0],c[1],c[2],c[3]
                            # 이전 center_x, center_y좌표 불러와서 crop
                            crop_img(idx,px,mx,py,my,path,make_video_img_dir,video_size_w,video_size_h)

                            idx += 1
    
    print('video 생성중...')
    img_list = glob(make_video_img_dir+"*.jpg")
    img_list = natsort.natsorted(img_list)
    video_path = osp.join(save_video_dir, f'{member}_output.mp4')
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame, (video_size_w,video_size_h))
    for path in img_list:
        img = cv2.imread(path)
        out.write(img)
    out.release()
    
    return video_path