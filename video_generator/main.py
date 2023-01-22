import sys
sys.path.append('/opt/ml/torchkpop/face_embedding/')
import face_embedding
from collections import defaultdict
from matplotlib import pyplot as plt 
from glob import glob
import json
import os
import cv2
import subprocess
from tqdm import tqdm
import csv
import pandas as pd
import natsort
import copy
import numpy as np


'''
df1
# idx   framenum,       image,   bbox,      unit_video_id
#  1        1      out1.jpg      xmin, ...         1
#  2        1      out1.jpg      xmin, ...         2
#  3        1      out1.jpg      xmin, ...         3
-> df1에 image의 path가 따로 있는것도 아니고, 다른 정보를 쓰는것도 아니라서 그냥 image_path를 바로 받는게 메모리, 속도면에서 나을 것 같음


1. img_path에서 image주소를 받아서 image를 cv2로 불러옴
2. member에서 어떤 멤버에 대한 영상을 만들어줄지 받아옴 -> 멤버별 idx로 받아오는게 나을 듯?
3. for문으로 모든 image를 한번씩 불러오면서 member의 bbox정보가 없을 때
    full_video = True -> 해당 image 전체를 그대로 저장
    full_video = False -> 해당 image 생략
4. for문으로 모든 image를 한번식 불러오면서 member의 bbox정보가 있을 때 
   df2에서 image별 bbox의 정보를 받아온 뒤 얼굴이 정중앙에 올 수 있게 좌표 계산
5. image를 crop하고 저장

'''
def img_padding(img,px,mx,py,my,w,h):   #top, bottom, left, right
    if px > w:
        # print('px 들어감')
        img = cv2.copyMakeBorder(img, 0,0,0,px-w,cv2.BORDER_CONSTANT)
        
    if mx < 0:
        # print('mx 들어감 : ', mx)
        img = cv2.copyMakeBorder(img, 0,0,-mx,0,cv2.BORDER_CONSTANT)
        px = px - mx
        mx = 0
    if py > h:
        # print('py 들어감')
        img = cv2.copyMakeBorder(img, 0,py-h,0,0,cv2.BORDER_CONSTANT)
        
    if my < 0:
        # print('my 들어감')
        img = cv2.copyMakeBorder(img, -my,0,0,0,cv2.BORDER_CONSTANT)
        py = py - my
        my = 0
        
    return img,px,mx,py,my

def video_generator(df1,img_list,member,pred,full_video=True):
    '''
    input
        df2 : video_id 별 sample 들에 대한 정보
        df2_idx       unit_video_id      df1_index         body_samples       face_sample       face_pred   face__confidence     body_pred     body_conf                pred
           0              1                 2                  (512,)             (512)               kari       0.99              kari        [0.8, 0.4, 0.1, 0.5]      1
           1              1                 6                  (512,)             (512)               kari       0.80              kari                                  1
           2              1                 10                 (512,)             (512)               ning       0.77                                                    3
           3              2                 3                  (512,)             (512)               wint       0.89                                                    2
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
    crop_img_w = 16 # 전체 이미지 크기 조절(가로)
    crop_img_h = 9 #전체 이미지 크기 조절(세로)
    video_size_w = 1280 # 최종 video 크기 (가로)
    video_size_h = 720  # 최종 video 크기 (세로)
    #얼굴을 화면 어디에 둘지 비율 조정(세로만 조정 가능, 숫자가 낮을 수록 얼굴 위로 올라감)
    face_loc = 3
    path = '/opt/ml/data/frame_1080p/' # 사진 저장되어 있는 폴더
    newfolder = '/opt/ml/data/result/'  # 사진 저장할 폴더
    video_path = '/opt/ml/torchkpop/'   # 비디오 저장할 폴더
    os.makedirs(newfolder,exist_ok=True)
    
    aespa = {'aespa_giselle' : 0,
         'aespa_ningning' : 1,
         'aespa_winter':2,
         'aespa_karina':3}
    ######################################################
    df1 = pd.read_csv(df1)
    
    
    
    print('video_generator실행중')
    
    
    
    
    
    for idx,img_name in enumerate(img_list, start=1):
        img_path = path + img_name
        print(img_path)
        filename_df = df1[df1['filename']== img_name]    #파일명이 같은 애들끼리 df하나 만들어주고
        print(filename_df.head())
        count = 0
        for row in filename_df.itertuples():
            
            print(pred[int(row.track_id)],member)  ###########추후 track_id 0.0 -> 0으로 바뀌면 변경
            print(type(pred[int(row.track_id)]),type(member))
            if pred[int(row.track_id)] == member: # 해당 member 사진이 있으면
                count += 1
                x_min = int(row.track_body_xmin)
                x_max = int(row.track_body_xmax)
                y_max = int(row.track_body_ymax)
                y_min = int(row.track_body_ymin)
                print('bbox좌표 : ',x_min,x_max,y_max,y_min)
                
                # retinaface로 얼굴 좌표 구하기
                temp_img = cv2.imread(img_path)
                temp_img = temp_img[y_min:y_max, x_min:x_max]
                
                # plt.imshow(temp_img)
                # plt.show
                # assert False
                face_x_min,face_y_min,face_x_max,face_y_max,_ = face_embedding.detect_face(temp_img)[0]
                face_x_min,face_y_min,face_x_max,face_y_max = int(face_x_min+x_min),int(face_y_min+y_min),int(face_x_max+x_min),int(face_y_max+y_min)#얼굴좌표
                face_img = temp_img[face_y_min:face_y_max, face_x_min:face_x_max]
                # plt.imshow(face_img)
                
                
                center_x = (face_x_max + face_x_min) / 2   #얼굴 중심 x좌표
                center_y = (face_y_max + face_x_max) / 2   #얼굴 중심 y좌표
                face_w = face_x_max - face_x_min    # 얼굴 가로 길이
                face_h = face_y_max - face_y_min    # 얼굴 세로 길이
                img = cv2.imread(img_path)  #이미지 불러와서 얼굴 좌표 구함
                h, w, _ = img.shape # 이미지 크기 받기
                px = int(center_x + (face_h/2) * (crop_img_w/2)) # x plus 방향 (오른쪽)
                mx = int(center_x - (face_h/2) * (crop_img_w/2)) # x minus 방향 (왼쪽)
                py = int(center_y + (face_h/2) * (crop_img_h - face_loc))  # y plus 방향 (아래)
                my = int(center_y - (face_h/2) * (face_loc))  # y minus 방향 (위)
                # 사진 padding
                if px > w or mx < 0 or py > h or my < 0: # 사진 범위 벗어나면
                    print('padding')
                    img,px,mx,py,my = img_padding(img,px,mx,py,my,w,h)
                    plt.imshow(img)
                    
                # 사진 crop
                print('crop')
                cropped_img = img[my:py, mx:px]
                plt.imshow(cropped_img)
                
                # 이미지 resize
                #interpolation : 기법 다른거로 바꿔서 품질, 속도 조절 가능
                resize_img = cv2.resize(cropped_img,(video_size_w,video_size_h),interpolation=cv2.INTER_CUBIC)

                # 이미지 저장
                cv2.imwrite(newfolder+str(idx)+'.jpg', resize_img)
                print('crop사진 저장')

        if count == 0 & full_video:
            # 이미지 resize
            #interpolation : 전체사진이라 넣지 않음
            img = cv2.imread(img_path)
            resize_img = cv2.resize(img,(video_size_w,video_size_h))

            # 이미지 저장
            print('full_image저장')
            cv2.imwrite(newfolder+str(idx)+'.jpg', resize_img)        

    
    #video 생성 -> out 부분을 위로 올리면 나중에 for문 한번 아낄 수 있고 사진도 저장 안해도 됨
    print('video 생성중,,,')
    img_list = glob(f"/opt/ml/data/result/*.jpg")
    img_list = natsort.natsorted(img_list)
    out = cv2.VideoWriter(video_path+'output.mp4',cv2.VideoWriter_fourcc(*'mp4v'),24(video_size_w,video_size_h))
    for path in img_list:
        img = cv2.imread(path)
        out.write(img)
    out.release()

if __name__ == '__main__':
    df1_path = '/opt/ml/df1.csv'
    path = '/opt/ml/aespa/'
    img_list = os.listdir(f"/opt/ml/aespa/")
    img_list = natsort.natsorted(img_list)
    pred = {'0': '1', '1':'2','2':'3','3':'0'}
    video_generator(df1=df1_path, img_list=img_list, member=1, pred=pred ,full_video=True)