# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser
import shutil
import pandas as pd
import mmcv
import cv2
from mmtrack.apis import inference_mot, init_model
import argparse


## 클립을 하기위해서 만든 툴
def clip(num, min_value, max_value):
   return max(min(num, max_value), min_value)


def main(): # mata_info:dict, output:str
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether show the results on the fly')
    parser.add_argument(
        '--backend',
        choices=['cv2', 'plt'],
        default='cv2',
        help='the backend to visualize the results')
    parser.add_argument('--fps', help='FPS of the output video')
    args = parser.parse_args()
    assert args.output or args.show


    # input_path = mata_info["image_root"]
    # fps = mata_info["fps"]

    # 원본 그대로의 자료 출력을 위해
    raw_data = {
                'frame':[], 
                'filename':[], 
                'det_body_xmin':[], 
                'det_body_ymin':[], 
                'det_body_xmax':[], 
                'det_body_ymax':[],
                'det_conf':[],
                'track_id':[], 
                'track_body_xmin':[],
                'track_body_ymin':[],
                'track_body_xmax':[],
                'track_body_ymax':[],
                'track_conf':[]}
    
    # 클립된 자료 출력을 위해
    clipped_data = {
                    'frame':[], 
                    'filename':[], 
                    'det_body_xmin':[], 
                    'det_body_ymin':[], 
                    'det_body_xmax':[], 
                    'det_body_ymax':[],
                    'det_conf':[],
                    'track_id':[], 
                    'track_body_xmin':[],
                    'track_body_ymin':[],
                    'track_body_xmax':[],
                    'track_body_ymax':[],
                    'track_conf':[]}
    # pd.DataFrame(clipped_data) 이렇게 만들면 됨


    # load images
    if osp.isdir(args.input):
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(args.input)),
            key=lambda x: int(x.split('.')[0]))
        filenames = [osp.basename(i) for i in imgs]
        IN_VIDEO = False
    else:
        imgs = mmcv.VideoReader(args.input)
        IN_VIDEO = True

    _out = args.output.rsplit(os.sep, 1)
    # define output
    if args.output is not None:
        if args.output.endswith('.mp4'): # 출력으로 mp4를 지정했으면
            OUT_VIDEO = True
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
                os.makedirs(osp.join(_out[0], "crop_imgs", "det", "per_frame"), exist_ok=True) # 크롭 이미지를 저장하기 위한 디렉토리 생성
                os.makedirs(osp.join(_out[0], "crop_imgs", "track", "per_frame"), exist_ok=True) # 크롭 이미지를 저장하기 위한 디렉토리 생성
                os.makedirs(osp.join(_out[0], "crop_imgs", "track", "per_id"), exist_ok=True) # 크롭 이미지를 저장하기 위한 디렉토리 생성
                det_img_save_path = osp.join(_out[0], "crop_imgs", "det", "per_frame")
                track_img_save_path_per_frame = osp.join(_out[0], "crop_imgs", "track", "per_frame")
                track_img_save_path_per_id = osp.join(_out[0], "crop_imgs", "track", "per_id")
        else: # 출력으로 img를 지정했으면
            OUT_VIDEO = False
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)
            os.makedirs(osp.join(_out[0], "crop_imgs", "det", "per_frame"), exist_ok=True) # 크롭 이미지를 저장하기 위한 디렉토리 생성
            os.makedirs(osp.join(_out[0], "crop_imgs", "track", "per_frame"), exist_ok=True) # 크롭 이미지를 저장하기 위한 디렉토리 생성
            os.makedirs(osp.join(_out[0], "crop_imgs", "track", "per_id"), exist_ok=True) # 크롭 이미지를 저장하기 위한 디렉토리 생성
            det_img_save_path = osp.join(_out[0], "crop_imgs", "det", "per_frame")
            track_img_save_path_per_frame = osp.join(_out[0], "crop_imgs", "track", "per_frame")
            track_img_save_path_per_id = osp.join(_out[0], "crop_imgs", "track", "per_id")

    fps = args.fps
    if args.show or OUT_VIDEO:
        if fps is None and IN_VIDEO: # 비디오 입력이 아니면 fps를 수동으로 반드시 지정해주어야 합니다
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    prog_bar = mmcv.ProgressBar(len(imgs))
    # test and show/save the images

    unmatching_cnt = 0 # det와 tracker의 언매칭된 개수를 세는 counter

    for i, img in enumerate(imgs):
        frame_idx = i+1 # frame 번호

        # !IN_VIDEO일경우 img는 str이므로 cv를 통해 np로 로드
        if isinstance(img, str): 

            img = osp.join(args.input, img) # filename to path
            img = cv2.imread(img)

        input_img_height = img.shape[0] # for clip
        input_img_width = img.shape[1] # for clip

        # 이미지 한장씩 inference
        result = inference_mot(model, img, frame_id=i) 

        # 일단 가장 최대 박스의 개수를 기준으로 append
        # 일단 이거 히든해봅시다
        # frame_max_row = max(result["det_bboxes"][0].shape[0], result["track_bboxes"][0].shape[0])
        # print(f"Frame {frame_idx}'s num of detbox 🐬: {result['det_bboxes'][0].shape[0]}, num of trk box: {result['track_bboxes'][0].shape[0]}")
        # raw_data["frame"].extend([frame_idx] * frame_max_row)
        # clipped_data["frame"].extend([frame_idx] * frame_max_row)

        # raw_data["filename"].extend([filenames[i]]*frame_max_row)
        # clipped_data["filename"].extend([filenames[i]]*frame_max_row)
        
        # 한장의 이미지의 det의 박스개수와 tracker의 박스개수가 다르면 언매칭의 경우로 카운트한다
        # if result["det_bboxes"][0].shape[0] != result["track_bboxes"][0].shape[0]:
            # unmatching_cnt+=1     

        if args.output is not None:
            # argparser로 입력된 입력 형식이 비디오면 frame별 이미지를 생성
            if IN_VIDEO or OUT_VIDEO:
                out_file = osp.join(out_path, f'{i:06d}.jpg')
            # 단지 이미지면
            else:
                out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
        else:
            out_file = None

        for img_order, detected_info in enumerate(result["det_bboxes"][0]): # iter는 하나의 det_bbox
            raw_xmin_d = detected_info[0]
            raw_ymin_d = detected_info[1]
            raw_xmax_d = detected_info[2]
            raw_ymax_d = detected_info[3]

            xmin_d = clip(raw_xmin_d, 0, input_img_width)
            ymin_d = clip(raw_ymin_d, 0, input_img_height)
            xmax_d = clip(raw_xmax_d, 0, input_img_width)
            ymax_d = clip(raw_ymax_d, 0, input_img_height)
            det_confidence_score = detected_info[4]

            # raw_data["det_conf"].append(det_confidence_score)
            # clipped_data["det_conf"].append(det_confidence_score)

            bbox_width = xmax_d - xmin_d
            bbox_height = ymax_d - ymin_d
            
            # 이미지 크롭 [H,W,C]
            cropped_det_img = img[int(ymin_d):int(ymin_d+bbox_height), int(xmin_d):int(xmin_d+bbox_width), : ]
            
            # raw는 무조건 데이터 저장
            # raw_data["det_body_xmin"].append(raw_xmin_d)
            # raw_data["det_body_ymin"].append(raw_ymin_d)
            # raw_data["det_body_xmax"].append(raw_xmax_d)
            # raw_data["det_body_ymax"].append(raw_ymax_d)

            # 클립된 이미지의 정상여부를 확인합니다. width or height가 0이면 정상적으로 디텍션 되지 않았다고 가정합니다
            if 0 not in cropped_det_img.shape:
                # 정상일 경우 저장합니다.
                cv2.imwrite(osp.join(det_img_save_path, f"{frame_idx}_{img_order}.jpg"), cropped_det_img)
                # clipped_data["det_body_xmin"].append(xmin_d)
                # clipped_data["det_body_ymin"].append(ymin_d)
                # clipped_data["det_body_xmax"].append(xmax_d)
                # clipped_data["det_body_ymax"].append(ymax_d)
            else:
                print()
                print(f"🤕 {frame_idx} 프레임의 {img_order}번째의 det bbox는 너무 작아 사용하지 않습니다 HEIGHT:{int(bbox_height)}, WIDTH:{int(bbox_width)}")
                # clipped_data["det_body_xmin"].append(None)
                # clipped_data["det_body_ymin"].append(None)
                # clipped_data["det_body_xmax"].append(None)
                # clipped_data["det_body_ymax"].append(None)

        # print(f"result['det_bboxes'][0] {result['det_bboxes'][0].shape}")
        # print(f"{frame_idx}'s frame_max_row is {frame_max_row}")
        # n_missbox = frame_max_row - result["det_bboxes"][0].shape[0] # 부족한 박수의 개수
        # if n_missbox != 0:
        #     print(f"num of miss track box: {n_missbox} so append empty row")
        #     for i in range(n_missbox):
        #         raw_data["det_conf"].append(None)
        #         clipped_data["det_conf"].append(None)
        #         raw_data["det_body_xmin"].append(None)
        #         raw_data["det_body_ymin"].append(None)
        #         raw_data["det_body_xmax"].append(None)
        #         raw_data["det_body_ymax"].append(None)
        #         clipped_data["det_body_xmin"].append(None)
        #         clipped_data["det_body_ymin"].append(None)
        #         clipped_data["det_body_xmax"].append(None)
        #         clipped_data["det_body_ymax"].append(None)


        
        for tracked_info in result["track_bboxes"][0]:
            id = tracked_info[0] # for mkdirs
            # raw_data["track_id"].append(id)
            # clipped_data["track_id"].append(id)

            raw_xmin_t = tracked_info[1]
            raw_ymin_t = tracked_info[2]
            raw_xmax_t = tracked_info[3]
            raw_ymax_t = tracked_info[4]

            xmin_t = clip(raw_xmin_t, 0, input_img_width)
            ymin_t = clip(raw_ymin_t, 0, input_img_height)
            xmax_t = clip(raw_xmax_t, 0, input_img_width)
            ymax_t = clip(raw_ymax_t, 0, input_img_height)
            track_confidence_score = tracked_info[5]
            # raw_data["track_conf"].append(track_confidence_score)
            # clipped_data["track_conf"].append(track_confidence_score)

            bbox_width = xmax_t-xmin_t
            bbox_height = ymax_t-ymin_t

            # 프레임에서 bbodx의 영역 크롭
            cropped_track_img = img[int(ymin_t):int(ymin_t+bbox_height), int(xmin_t):int(xmin_t+bbox_width), :]
            
            # 프레임마다의 이미지 저장 패스 지정
            per_frame_save_path = osp.join(track_img_save_path_per_frame, f"{frame_idx}_{int(id)}.jpg")

            # raw는 무조건 데이터 저장
            # raw_data["track_body_xmin"].append(raw_xmin_t)
            # raw_data["track_body_ymin"].append(raw_ymin_t)
            # raw_data["track_body_xmax"].append(raw_xmax_t)
            # raw_data["track_body_ymax"].append(raw_ymax_t)

            # 아아디별로 dir를 생성하여 저장할 이미지 패스 지정
            per_id_save_path = osp.join(track_img_save_path_per_id, f'{int(id)}', f'{int(id)}'+"_"+f'{frame_idx}'+".jpg")
            # id 폴더 생성
            os.makedirs(osp.dirname(per_id_save_path), exist_ok=True)

            # 정상여부 검사
            if 0 not in cropped_track_img.shape:
                cv2.imwrite(per_frame_save_path, cropped_track_img)
                # 정상이면 저장
                # clipped_data["track_body_xmin"].append(xmin_t)
                # clipped_data["track_body_ymin"].append(ymin_t)
                # clipped_data["track_body_xmax"].append(xmax_t)
                # clipped_data["track_body_ymax"].append(ymax_t)
            else:
                print()
                print(f"🤕 {frame_idx} 프레임의 {id}의 track bbox는 너무 작아 사용하지 않습니다 HEIGHT:{int(bbox_height)}, WIDTH:{int(bbox_width)}")
                # clipped_data["track_body_xmin"].append(None)
                # clipped_data["track_body_ymin"].append(None)
                # clipped_data["track_body_xmax"].append(None)
                # clipped_data["track_body_ymax"].append(None)

        # n_missbox = frame_max_row - result["track_bboxes"][0].shape[0] # 부족한 박수의 개수
        # if n_missbox != 0:
        #     print(f"→ num of miss det box appear {n_missbox} so that we append empty row in csv")
        #     for i in range(n_missbox):
        #         raw_data["track_id"].append(None)
        #         clipped_data["track_id"].append(None)
        #         raw_data["track_conf"].append(None)
        #         clipped_data["track_conf"].append(None)
        #         raw_data["track_body_xmin"].append(None)
        #         raw_data["track_body_ymin"].append(None)
        #         raw_data["track_body_xmax"].append(None)
        #         raw_data["track_body_ymax"].append(None)
        #         clipped_data["track_body_xmin"].append(None)
        #         clipped_data["track_body_ymin"].append(None)
        #         clipped_data["track_body_xmax"].append(None)
        #         clipped_data["track_body_ymax"].append(None)
        
        model.show_result(
            img,
            result,
            score_thr=args.score_thr,
            show=args.show,
            thickness=5, # 가시성을 위해서 변경
            font_scale=1.0, # 가시성을 위해서 변경
            wait_time=int(1000. / fps) if fps else 0,
            out_file=out_file,
            backend=args.backend)
        prog_bar.update()
    
    # 인퍼런스 끝
    # 결과 출력
    print(f"→🐬 num of unmatching bbox frame: {str(unmatching_cnt)} Frame")
    # df1 저장
    # bbox 0~1로 스케일링 하고 저장

    
    
    if args.output:
        print()
        if OUT_VIDEO: # .avi입력 이면 비디오 만듬
            print(f'making the output video 📺 at {args.output} with a FPS of {fps}')
            mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
        img_dir_path = osp.join(osp.dirname(args.output),"tracked_imgs") # 트래킹 된 전체 이미지를 저장할 dir 생성
        if osp.isdir(img_dir_path):
            print(f"dir이 이미 존재하므로 overwriting을 위해서 삭제합니다")
            shutil.rmtree(img_dir_path)
        os.makedirs(img_dir_path, exist_ok=True)
        print()
        for file_name in os.listdir(out_path):
            new_file_name = str(int(file_name.split(".")[0])+1).zfill(6) + ".jpg"
            shutil.copy(osp.join(out_path, file_name),osp.join(img_dir_path,new_file_name))

        out_dir.cleanup() # temp폴더 삭제(mmtrack 디폴트로 만들어서 저장함)


    """for check
    
    print("result 📝")
    print("raw_data")
    print(f"frame {len(raw_data['frame'])}")
    print(f"filename {len(raw_data['filename'])}")
    print(f"det_body_xmin {len(raw_data['det_body_xmin'])}")
    print(f"det_body_ymin {len(raw_data['det_body_ymin'])}")
    print(f"det_body_xmax {len(raw_data['det_body_xmax'])}")
    print(f"det_body_ymax {len(raw_data['det_body_ymax'])}")
    print(f"det_conf {len(raw_data['det_conf'])}")
    print(f"track_id {len(raw_data['track_id'])}")
    print(f"track_body_xmin {len(raw_data['track_body_xmin'])}")
    print(f"track_body_ymin {len(raw_data['track_body_ymin'])}")
    print(f"track_body_xmax {len(raw_data['track_body_xmax'])}")
    print(f"track_body_ymax {len(raw_data['track_body_ymax'])}")
    print(f"track_conf {len(raw_data['track_conf'])}")
    print()
    print("clipped_data")
    print(f"frame {len(clipped_data['frame'])}")
    print(f"filename {len(clipped_data['filename'])}")
    print(f"det_body_xmin {len(clipped_data['det_body_xmin'])}")
    print(f"det_body_ymin {len(clipped_data['det_body_ymin'])}")
    print(f"det_body_xmax {len(clipped_data['det_body_xmax'])}")
    print(f"det_body_ymax {len(clipped_data['det_body_ymax'])}")
    print(f"det_conf {len(clipped_data['det_conf'])}")
    print(f"track_id {len(clipped_data['track_id'])}")
    print(f"track_body_xmin {len(clipped_data['track_body_xmin'])}")
    print(f"track_body_ymin {len(clipped_data['track_body_ymin'])}")
    print(f"track_body_xmax {len(clipped_data['track_body_xmax'])}")
    print(f"track_body_ymax {len(clipped_data['track_body_ymax'])}")
    print(f"track_conf {len(clipped_data['track_conf'])}")
    """

    # vanila_df1 = pd.DataFrame(raw_data) # 바닐라 predict result
    # vanila_df1.to_csv(osp.dirname(args.output) + "/" + "vanila_df1.csv")

    # df1 = pd.DataFrame(clipped_data) # 최종적으로 아웃할 자료
    # df1.to_csv(osp.dirname(args.output) + "/" + "df1.csv")

    
    # return df1, vanila_df1

if __name__ == '__main__':
    main()
