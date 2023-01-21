
import argparse
import cv2
import os
import os.path as osp
import glob
from pathlib import Path
import re
from tqdm import tqdm


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]



def padding(img, set_size):
    try:
        h,w,c = img.shape
    except:
        print('shape error')
        raise
    # if width가 height보다 더 크면
    if h<w: 
        # set_size를 새로운 width로 선언
        new_width = set_size 
        # 기존 ratio에 맞도록 new_height 선언
        new_height = int(new_width * (h/w)) 
    else:
        new_height = set_size
        new_width = int(new_height*(w/h))

    # set_size가 더 클 경우
    if max(h,w) < set_size:
        # new_width, new_height로 맞춤 cubic으로 보간
        img = cv2.resize(img, (new_width, new_height), cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img, (new_width, new_height), cv2.INTER_AREA)
    
    # 다시 확인
    try:
        h,w,c = img.shape
    except:
        print("file error")
        raise

    delta_h = set_size - h # height 패딩 pixel
    delta_w = set_size - w # width 패딩 pixel
    top, bottom = delta_h//2, delta_h-(delta_h//2) # height 증가량
    left, right = delta_w//2, delta_w-(delta_w//2) # width 증가량
    # 패딩
    new_img = cv2.copyMakeBorder(img, 
                                 top, 
                                 bottom, 
                                 left, 
                                 right,
                                 cv2.BORDER_CONSTANT,
                                 value=[0,0,0])
    return new_img

def make_mp4(target_dir, output_dir):
    # output each person dir과 상위 dir을 생성
    for people_id in sorted(os.listdir(target_dir)):
        os.makedirs(osp.join(output_dir, people_id), exist_ok=True)

    # each person(person_id)의 dir에 방문
    # each person(person_id)의 video 생성
    for people_id in tqdm(os.listdir(target_dir)):
        print(osp.join(target_dir, people_id))
        person_img_paths =  [ osp.join(target_dir,people_id,i) for i in os.listdir(osp.join(target_dir, people_id)) ]
        
        # 오름차순 정렬
        person_img_paths.sort(key=natural_keys)

        img_array = []
        heights = []
        widths = []
        frames_array = []

        # 이미지들을 돌면서 img_array에 저장한다.
        for path in person_img_paths:
            img = cv2.imread(path)
            height, width, _ = img.shape
            # 이미지 저장과 동시에 frame 번호를 리스트에 저장한다
            img_array.append(img)
            frames_array.append(osp.basename(path).split("_")[1].split(".")[0])
            heights.append(height)
            widths.append(width)

        max_height = max(heights) # height 최대값
        max_width = max(widths) # width 최대값
        max_squre = max([max_height, max_width]) # height, width 중 의 최대값
        out = cv2.VideoWriter(f"{output_dir}/{people_id}.avi", # 여기에다가
                              cv2.VideoWriter_fourcc(*'DIVX'), # 어떤 양식으로
                              24, # fps는 이것으로
                              (max_squre, max_squre)) # height, width
        
        # 모든 이미지들에 패딩을 한다
        for img, frame_no in zip(img_array,frames_array):
            
            # 하나씩 video를 만들고
            img = padding(img, max_squre)
            out.write(img)

            # 🐬 시간이 너무 길어서 주석처리
            # 패딩처리된 이미지들을 하나씩 outputdir에 복사한다. 이름은 {people_id}_{order}.jpg (ex)1_1.jpg
            # copy_dir = osp.join(output_dir, people_id, str(people_id) + "_" + frame_no + ".jpg") 
            # cv2.imwrite(copy_dir, img)
        
        out.release()
    return None

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--target_dir', default="./runs/track/input_strongsort_sample_conf_640_crop/crops/person", type=str)
    parser.add_argument('--output_dir', default="./video_output", type=str)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser('from jpg root to mp4', parents=[get_args_parser()])
    args = parser.parse_args()
    # argument들 출력
    for arg in vars(args):
        print("--"+arg, getattr(args, arg))
    
    make_mp4(args.target_dir, args.output_dir)