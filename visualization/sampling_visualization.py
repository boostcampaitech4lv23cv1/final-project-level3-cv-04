import pandas as pd
import os
from PIL import Image
import json

def visualize_sample(df1_postprocessed, df2_sampled, save_dir, meta_info=None):
    '''
        sampling 된 tracked bbox를 살펴봅니다.
        intput : 
            df2_sampled : 샘플링된 df2 (face, body model의 예측 결과 반영한)
            df1_postprocessed : 후처리된 df1
            save_dir : 샘플링된 이미지 저장 경로
        output : images
        save_dir에 어떤 sampled 이미지들이 저장되어있는지 확인하세요.
    '''
    # save_dir 만들기
    sample_save_dir = os.path.join(save_dir, 'sampled_images')
    if not os.path.exists(sample_save_dir):
         os.makedirs(sample_save_dir)
    
    for i in range(len(df2_sampled)):
        # body_img 가져오기
        track_id = df2_sampled.iloc[i]['track_id']
        df1_idx = df2_sampled.iloc[i]['df1_index']

        x_min = df1_postprocessed.loc[df1_idx]['track_body_xmin']
        y_min = df1_postprocessed.loc[df1_idx]['track_body_ymin']
        x_max = df1_postprocessed.loc[df1_idx]['track_body_xmax']
        y_max = df1_postprocessed.loc[df1_idx]['track_body_ymax']

        filename = df1_postprocessed.loc[df1_idx]['filename']        
        
        img_path = os.path.join(meta_info["image_root"], filename) # ✅ meta_info를 이용하세요
        img = Image.open(img_path).convert('RGB')

        # 이미지 자르기 crop함수 이용 ex. crop(left,up, rigth, down)
        body_img=img.crop((x_min, y_min, x_max, y_max))
        
        face_pred = df2_sampled.iloc[i]['face_pred']
        body_pred = df2_sampled.iloc[i]['body_pred']
        
        # 이미지 저장
        filename = os.path.splitext(filename)[0]
        save_path = os.path.join(sample_save_dir, f'{track_id}_{filename}_{df1_idx}_{face_pred}_{body_pred}.jpg')
        body_img.save(save_path, "JPEG")


if __name__ == '__main__': # main 전체 실행하지 않고 debuggin 하려면 아래 수정 후 실행 -> python sampling_visualization.py
    df1_postprocessed = pd.read_csv('/opt/ml/torchkpop/result/20230128_1736/csv/df1_postprocessed.csv')
    df2_sampled = pd.read_csv("/opt/ml/torchkpop/result/20230128_1736/csv/df2_out_of_body_embedding.csv")
    with open('/opt/ml/torchkpop/data/20230128_1736.json') as meta_info_file:
        meta_info = json.load(meta_info_file)
    save_dir = '/opt/ml/torchkpop/result/20230128_1736'
    visualize_sample(df1_postprocessed, df2_sampled, save_dir, meta_info=meta_info)