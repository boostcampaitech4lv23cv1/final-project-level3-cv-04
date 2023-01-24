import os
import sys
sys.path.append('/opt/ml/torchkpop/face_embedding/')
import face_embedding
from glob import glob
import cv2
from tqdm import tqdm
import pandas as pd
import natsort
import numpy as np

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

def video_generator(df1,img_list,member,pred,full_video=True):
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
    crop_img_w = 16 # 전체 이미지 크기 조절(가로)
    crop_img_h = 9 #전체 이미지 크기 조절(세로)
    video_size_w = 1280 # 최종 video 크기 (가로)
    video_size_h = 720  # 최종 video 크기 (세로)
    #얼굴을 화면 어디에 둘지 비율 조정(세로만 조정 가능, 숫자가 낮을 수록 얼굴 위로 올라감)
    face_loc = 3
    newfolder = '/opt/ml/torchkpop/data/result/'  # 사진 저장할 폴더
    video_path = '/opt/ml/torchkpop/'   # 비디오 저장할 폴더
    frame = 24  # 비디오 프레임
    os.makedirs(newfolder,exist_ok=True)
    ######################################################   
    prev_px = 0
    prev_mx = 0
    prev_py = 0
    prev_my = 0
    
    print('video_generator실행중')
    
    for idx,img_path in enumerate(img_list, start=1):
        print(img_path)
        filename_df = df1[df1['filename']== img_path.split('/')[-1]]    #파일명이 같은 애들끼리 df하나 만들어주고
        count = 0
        for row in filename_df.itertuples():
            if pred[row.track_id] == member: # 해당 member 사진이 있으면
                print(row.track_id, pred[row.track_id],member)  ###########추후 track_id 0.0 -> 0으로 바뀌면 변경
                count += 1
                x_min = int(row.track_body_xmin)
                x_max = int(row.track_body_xmax)
                y_max = int(row.track_body_ymax)
                y_min = int(row.track_body_ymin)
                print('bbox좌표 : ',x_min,',',x_max,',',y_min,',',y_max)
                
                # retinaface로 얼굴 좌표 구하기
                temp_img = cv2.imread(img_path)
                temp_img = temp_img[y_min:y_max, x_min:x_max]
                
                
                ### face embedding에서 얼굴이 있는 경우와 없는 경우로 나눠야 함
                face_embedding_result = face_embedding.detect_face(temp_img)
                del temp_img    #temp_img 메모리 할당 해제
                print('face_enbedding_result :',face_embedding_result)
                
                if len(face_embedding_result) == 2:   # 얼굴이 없는 경우
                    print('얼굴 없음')
                    ## prev 좌표가 0,0인 경우 -> 전체화면
                    if prev_px == 0 and prev_py == 0:
                        count -= 1
                        break
                    ## 그 외 -> 이전 좌표 값 불러오기
                    else:
                        px = prev_px
                        mx = prev_mx
                        py = prev_py
                        my = prev_my
                
                else:   # 얼굴이 있는 경우
                    face_x_min,face_y_min,face_x_max,face_y_max = face_embedding_result[0][0],face_embedding_result[0][1],face_embedding_result[0][2],face_embedding_result[0][3]
                    face_x_min,face_y_min,face_x_max,face_y_max = face_x_min+x_min,face_y_min+y_min,face_x_max+x_min,face_y_max+y_min#얼굴좌표         
                    center_x = (face_x_max + face_x_min) / 2   #얼굴 중심 x좌표
                    center_y = (face_y_max + face_y_max) / 2   #얼굴 중심 y좌표
                
                    face_w = face_x_max - face_x_min    # 얼굴 가로 길이
                    face_h = face_y_max - face_y_min    # 얼굴 세로 길이
                    px = int(center_x + (face_h/2) * (crop_img_w/2)) # x plus 방향 (오른쪽)
                    mx = int(center_x - (face_h/2) * (crop_img_w/2)) # x minus 방향 (왼쪽)
                    py = int(center_y + (face_h/2) * (crop_img_h - face_loc))  # y plus 방향 (아래)
                    my = int(center_y - (face_h/2) * (face_loc))  # y minus 방향 (위)
                    prev_px = px
                    prev_mx = mx
                    prev_py = py
                    prev_my = my
                    
                img = cv2.imread(img_path)  #이미지 불러와서 얼굴 좌표 구함
                h, w, _ = img.shape # 이미지 크기 받기
                # 사진 padding
                if px > w or mx < 0 or py > h or my < 0: # 사진 범위 벗어나면
                    img,px,mx,py,my = img_padding(img,px,mx,py,my,w,h)
                    
                # 사진 crop
                cropped_img = img[my:py, mx:px]
                del img     #img 메모리 할당 해제
                
                # 이미지 resize
                #interpolation : 기법 다른거로 바꿔서 품질, 속도 조절 가능
                resize_img = cv2.resize(cropped_img,(video_size_w,video_size_h),interpolation=cv2.INTER_CUBIC)
                del cropped_img # cropped_img 메모리 할당 해제
                # 이미지 저장
                cv2.imwrite(newfolder+str(idx)+'.jpg', resize_img)
                del resize_img  #resize_img 메모리 할당 해제

                break
        del filename_df # 데이터 프레임 메모리 할당 해제 
        
        if count == 0 & full_video:
            # 이미지 resize
            #interpolation : 전체사진이라 넣지 않음
            img = cv2.imread(img_path)
            resize_img = cv2.resize(img,(video_size_w,video_size_h))

            # 이미지 저장
            cv2.imwrite(newfolder+str(idx)+'.jpg', resize_img)  

            del img
            del resize_img   
           
    
    del df1 # df1 데이터프레임 메모리 할당 해제
    del img_list    #함수에서 받은 img_list 메모리 할당 해제
    
    #video 생성 -> out 부분을 위로 올리면 나중에 for문 한번 아낄 수 있고 사진도 저장 안해도 됨
    print('video 생성중...')
    img_list = glob(f"/opt/ml/torchkpop/data/result/*.jpg")
    img_list = natsort.natsorted(img_list)
    out = cv2.VideoWriter(video_path+member+'_output.mp4',cv2.VideoWriter_fourcc(*'mp4v'),frame,(video_size_w,video_size_h))
    for path in img_list:
        img = cv2.imread(path)
        out.write(img)
    out.release()

if __name__ == '__main__':
    df1_path = '/opt/ml/torchkpop/df1.csv'
    df1 = pd.read_csv(df1_path)
    df1['track_id'].fillna(-1, inplace=True)
    df1['track_id'] = df1['track_id'].map(lambda x: int(x))
    img_list = glob(f"/opt/ml/torchkpop/data/frame_1080p/*.jpg")
    img_list = natsort.natsorted(img_list)
    pred = {-1:'plz',
        0: 'aespa_giselle',
        1: 'aespa_ningning',
        2: 'aespa_winter',
        3: 'aespa_karina',
        4: 'aespa_ningning',
        5: 'aespa_giselle',
        8: 'aespa_winter',
        12: 'aespa_karina',
        14: 'aespa_karina',
        19: 'aespa_ningning',
        25: 'aespa_ningning',
        26: 'aespa_karina',
        27: 'aespa_winter',
        32: 'aespa_ningning',
        34: 'aespa_giselle',
        43: 'aespa_winter',
        48: 'aespa_winter',
        53: 'aespa_karina',
        62: 'aespa_winter'}
    video_generator(df1=df1, img_list=img_list, member='aespa_ningning', pred=pred ,full_video=True)