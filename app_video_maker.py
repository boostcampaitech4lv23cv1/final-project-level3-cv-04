import natsort
from glob import glob
import face_embedding
import os
import cv2

def video_generator(df1,meta_info,member,pred,full_video=True):
    ######################################################
    crop_img_w = 16 # 전체 이미지 크기 조절(가로)
    crop_img_h = 9 #전체 이미지 크기 조절(세로)
    video_size_w = 1280 # 최종 video 크기 (가로)
    video_size_h = 720  # 최종 video 크기 (세로)
    #얼굴을 화면 어디에 둘지 비율 조정(세로만 조정 가능, 숫자가 낮을 수록 얼굴 위로 올라감)
    face_loc = 3
    newfolder = './streamlit_output/' + meta_info["image_root"].split('/')[-1] + '/img/'  # 사진 저장할 폴더
    video_path = './streamlit_output/' + meta_info["image_root"].split('/')[-1] + '/video/'   # 비디오 저장할 폴더
    frame = meta_info["fps"]  # 비디오 프레임
    os.makedirs(newfolder,exist_ok=True)
    os.makedirs(video_path,exist_ok=True)
    ######################################################   
    prev_px = 0
    prev_mx = 0
    prev_py = 0
    prev_my = 0
    
    df1['track_id'].fillna(-1, inplace=True)
    df1['track_id'] = df1['track_id'].map(lambda x: int(x))
    
    img_list = glob(meta_info["image_root"]+'/*.jpg')
    img_list = natsort.natsorted(img_list)
    
    print('video_generator 실행 중')
    for idx,img_path in enumerate(img_list, start=1):
        # print(img_path)
        filename_df = df1[df1['filename']== img_path.split('/')[-1]]
        count = 0
        for row in filename_df.itertuples():
            if pred[row.track_id] == member:
                count += 1
                x_min = int(row.track_body_xmin)
                x_max = int(row.track_body_xmax)
                y_max = int(row.track_body_ymax)
                y_min = int(row.track_body_ymin)
                
                temp_img = cv2.imread(img_path)
                temp_img = temp_img[y_min:y_max, x_min:x_max]
                         
                face_embedding_result = face_embedding.detect_face(temp_img)[0]
                del temp_img
                
                # print(face_embedding_result)  # [386.82208    37.99035   495.5613    153.07664     0.8939557]
                if face_embedding_result[0][0] == -1.0 and face_embedding_result[0][1] == -1.0:
                    if prev_px == 0 and prev_py == 0:
                        count -= 1
                        break
                    else:
                        px = prev_px
                        mx = prev_mx
                        py = prev_py
                        my = prev_my
                else:
                    face_x_min,face_y_min,face_x_max,face_y_max = face_embedding_result[0][0],face_embedding_result[0][1],face_embedding_result[0][2],face_embedding_result[0][3]
                    face_x_min,face_y_min,face_x_max,face_y_max = face_x_min+x_min,face_y_min+y_min,face_x_max+x_min,face_y_max+y_min#얼굴좌표         
                    center_x = (face_x_max + face_x_min) / 2
                    center_y = (face_y_max + face_y_max) / 2
                    face_w = face_x_max - face_x_min
                    face_h = face_y_max - face_y_min
                    px = int(center_x + video_size_w/2)
                    mx = int(center_x - video_size_w/2)
                    py = int(center_y + video_size_h/2)
                    my = int(center_y - video_size_h/2)

                    
                img = cv2.imread(img_path)
                h, w, _ = img.shape
                if px > w or mx < 0 or py > h or my < 0:
                    img,px,mx,py,my = img_padding(img,px,mx,py,my,w,h)
                    
                cropped_img = img[my:py, mx:px]

                del img
                del cropped_img 
                cv2.imwrite(newfolder+str(idx)+'.jpg', cropped_img)

                break
        del filename_df
        
        if count == 0 & full_video:
            img = cv2.imread(img_path)
            resize_img = cv2.resize(img,(video_size_w,video_size_h))
            cv2.imwrite(newfolder+str(idx)+'.jpg', resize_img)  
            del img
            del resize_img   
           
    
    del df1
    del img_list
    
    print('video 생성 중...')
    img_list = glob(newfolder+"*.jpg")
    img_list = natsort.natsorted(img_list)
    
    # 🙏🏻 fourcc = cv2.VideoWriter_fourcc(*'MP4V') -> cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(video_path+member+'_output.mp4',cv2.VideoWriter_fourcc(*'H264'),frame,(video_size_w,video_size_h))
    for path in img_list:
        img = cv2.imread(path)
        out.write(img)
    out.release()
    
    video_full_path = video_path+member+'_output.mp4'
    return video_full_path # 🙏🏻 video_full_path return

def app_video_maker(df1_name_tagged_GT, meta_info, pred):
    video_path_karina = video_generator(df1_name_tagged_GT, meta_info, member='aespa_karina', pred=pred, full_video = True)
    video_path_winter = video_generator(df1_name_tagged_GT, meta_info, member='aespa_winter', pred=pred, full_video = True)
    video_path_ningning = video_generator(df1_name_tagged_GT, meta_info, member='aespa_ningning', pred=pred, full_video = True)
    video_path_giselle = video_generator(df1_name_tagged_GT, meta_info, member='aespa_giselle', pred=pred, full_video = True)
    return [video_path_karina, video_path_winter, video_path_ningning, video_path_giselle]


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



