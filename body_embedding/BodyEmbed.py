import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "torchreid"))
import pandas as pd
import numpy as np
from torchreid.utils import load_pretrained_weights
import torchreid
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip
)
import torch
from torch.utils.data import DataLoader, Dataset

norm_mean = [0.485, 0.456, 0.406] # imagenet mean
norm_std = [0.229, 0.224, 0.225] # imagenet std
normalize = Normalize(mean=norm_mean, std=norm_std)
transform_te = Compose([
        Resize((800, 400)), # (800,400) -> l2_norm : 300~700 정도, (160, 80) -> l2_norm : 5e-12~7e-12 정도
        ToTensor(),
        # normalize,
    ])

### generate_body_anchor
def make_body_img_list(df1, df2_member_sorted, meta_info=None, img_num=None):
    if not img_num:
        img_num = len(df2_member_sorted)
    body_img_list = []
    # for df2_idx in range(len(df2)):
    for i in range(img_num):
        # body_img 가져오기
        df1_idx = df2_member_sorted.iloc[i]['df1_index']
        
        # x_min_rate = df1.loc[df1_idx]['track_body_xmin'] if df1.loc[df1_idx]['track_body_xmin'] >= 0 else np.float64(0)
        # y_min_rate = df1.loc[df1_idx]['track_body_ymin'] if df1.loc[df1_idx]['track_body_ymin'] >= 0 else np.float64(0)
        # x_max_rate = df1.loc[df1_idx]['track_body_xmax'] if df1.loc[df1_idx]['track_body_xmin'] <= 1 else np.float64(1)
        # y_max_rate = df1.loc[df1_idx]['track_body_ymax'] if df1.loc[df1_idx]['track_body_xmin'] <= 1 else np.float64(1)


        x_min = df1.loc[df1_idx]['track_body_xmin'] # if df1.loc[df1_idx]['track_body_xmin'] >= 0 else np.float64(0)
        y_min = df1.loc[df1_idx]['track_body_ymin'] # if df1.loc[df1_idx]['track_body_ymin'] >= 0 else np.float64(0)
        x_max = df1.loc[df1_idx]['track_body_xmax'] # if df1.loc[df1_idx]['track_body_xmin'] <= 1 else np.float64(1)
        y_max = df1.loc[df1_idx]['track_body_ymax'] # if df1.loc[df1_idx]['track_body_xmin'] <= 1 else np.float64(1)

        filename = df1.loc[df1_idx]['filename']        
        
        img_path = os.path.join(meta_info["image_root"], filename) # ✅ meta_info를 이용하세요
        img = Image.open(img_path).convert('RGB')
        
        width = img.size[0]
        height = img.size[1]



        
        # x_min = width * x_min_rate
        # y_min = height * y_min_rate
        # x_max = width * x_max_rate
        # y_max = height * y_max_rate
        
        # 이미지 자르기 crop함수 이용 ex. crop(left,up, rigth, down)
        body_img=img.crop((x_min, y_min, x_max, y_max))
        # display(body_img)

        body_img_list.append(body_img)
        
    return body_img_list

def make_anchor(body_img_list, model):
    
    batches = []
    for img in body_img_list:
        batch = transform_te(img).unsqueeze(0)
        batches.append(batch)
    concat = torch.concat(batches, dim=0) # [10, 3, 800, 400]
    concat_pred = model(concat) # [10, 512]
    pred_mean = torch.mean(concat_pred, dim=0) # [512]
    
    return pred_mean

def generate_body_anchor(df1, df2, group_name='aespa', meta_info=None):
    # model load
    model = torchreid.models.build_model( # 이거 나중에 문제될듯 지금은 나의 bpbreid 가상환경의 파이썬 실행 환경변수가 opt/ml/torchreid로 잘 되어 있어서 잘 되는 듯
        name="osnet_ain_x1_0",
        num_classes=4, # 어차피 classifier 안써서 몇이든 상관없음
        loss="softmax",
        pretrained=True,
        use_gpu=True
    )
    load_weights = './pretrained_weight/osnet_ain_ms_d_c.pth.tar' # ✅ meta_info 를 이용해주세요
    load_pretrained_weights(model, load_weights)
    model.training = False
    
    
    # csv 저장과정 dict->str 이슈 해결
    if type(df2['face_confidence'].loc[0]) == str:
        # df2['face_confidence'] type : str -> dict
        face_confidence_list = []
        for i in range(len(df2)):
            face_confidence_list.append(eval(df2['face_confidence'].loc[i]))
        
        face_confidence_series = pd.Series(face_confidence_list)
        df2['face_confidence'] = face_confidence_series
    
    # face_pred_confi 컬럼 추가 - pred한 인물에 대한 confi
    face_pred_confi_list = []
    for i in range(len(df2)):
        face_pred = df2['face_pred'].loc[i]
        face_confi_i = df2['face_confidence'].loc[i]
        face_confi = face_confi_i[face_pred]
        face_pred_confi_list.append(face_confi)
    face_confidence_series = pd.Series(face_pred_confi_list)
    df2['face_pred_confi'] = face_confidence_series
    
    # df2 멤버별로 분리
    df2_karina = df2[df2['face_pred'] == 'aespa_karina']
    df2_winter = df2[df2['face_pred'] == 'aespa_winter']
    df2_ningning = df2[df2['face_pred'] == 'aespa_ningning']
    df2_giselle = df2[df2['face_pred'] == 'aespa_giselle']
    
    # df2_member confi_score 순으로 정렬
    df2_karina_sorted = df2_karina.sort_values(by='face_pred_confi', ascending=False)
    df2_winter_sorted = df2_winter.sort_values(by='face_pred_confi', ascending=False)
    df2_ningning_sorted = df2_ningning.sort_values(by='face_pred_confi', ascending=False)
    df2_giselle_sorted = df2_giselle.sort_values(by='face_pred_confi', ascending=False)
    
    
    # print("karina 대표 이미지 10장 뽑는 중...")
    body_img_list_karina = make_body_img_list(df1, df2_karina_sorted, meta_info, img_num=6)
    # print("winter 대표 이미지 10장 뽑는 중...")
    body_img_list_winter = make_body_img_list(df1, df2_winter_sorted, meta_info, img_num=6)
    # print("ningning 대표 이미지 10장 뽑는 중...")
    body_img_list_ningning = make_body_img_list(df1, df2_ningning_sorted, meta_info, img_num=6)
    # print("giselle 대표 이미지 10장 뽑는 중...")
    body_img_list_giselle = make_body_img_list(df1, df2_giselle_sorted, meta_info, img_num=6)
    
    anchor_karina = make_anchor(body_img_list_karina, model)
    anchor_winter = make_anchor(body_img_list_winter, model)
    anchor_ningning = make_anchor(body_img_list_ningning, model)
    anchor_giselle = make_anchor(body_img_list_giselle, model)
    
    anchors = {
        'aespa_karina' : anchor_karina, 
        'aespa_winter' : anchor_winter, 
        'aespa_ningning' : anchor_ningning, 
        'aespa_giselle' : anchor_giselle
    }
    
    return anchors



### body_embedding_extractor
class MyDataset(Dataset):
    def __init__(self, img_list, transform_te):
        super(MyDataset, self).__init__()
        self.img_list = img_list
        self.transform_te = transform_te

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        return self.transform_te(img)

def compute_cossim(rprst, body_feature): # rprst : [512]
    output = torch.nn.CosineSimilarity(dim=0)(rprst, body_feature)
    return output

def my_softmax(scores):
    exp_scores = np.exp(scores) # [4]    
    for i in range(len(exp_scores)):
        exp_scores[i] = exp_scores[i]*100
    exp_scores = np.exp(exp_scores)
    sum_exp_scores = np.sum(exp_scores) # 값
    output = exp_scores / sum_exp_scores # [4] / 값
    return output # [4]

   
def body_embedding_extractor(df1, df2, body_anchors, meta_info):
    '''
        model : 함수 호출할 때 마다 모델 빌드하면 안되니까 인자로 받아줌
    
        1. body_img_list 생성
        2. body_img -> [model] -> body_embedding
        3. df2 에 저장
    '''
    transform_te = Compose([
        Resize((800, 400)), # (800,400) -> l2_norm : 300~700 정도, (160, 80) -> l2_norm : 5e-12~7e-12 정도
        ToTensor(),
        # normalize,
    ])
    # model load
    model = torchreid.models.build_model( # 이거 나중에 문제될듯 지금은 나의 bpbreid 가상환경의 파이썬 실행 환경변수가 opt/ml/torchreid로 잘 되어 있어서 잘 되는 듯
        name="osnet_ain_x1_0",
        num_classes=4, # 어차피 classifier 안써서 몇이든 상관없음
        loss="softmax",
        pretrained=True,
        use_gpu=True
    )

    load_weights = './pretrained_weight/osnet_ain_ms_d_c.pth.tar' # ✅ meta_info 를 이용해주세요
    load_pretrained_weights(model, load_weights)
    model.training = False
    
    
    
    body_img_list = make_body_img_list(df1, df2, meta_info) # ⛔️ <- 이거 메모리에 적재하다가 문제될 수 있겠다
    
    my_dataset = MyDataset(body_img_list, transform_te)
    my_dataloader = DataLoader(my_dataset, batch_size=32)
    
    body_embedding = []
    body_confidence = []
    body_pred = []
    
    
    pred = None
    len_body_img_list = len(body_img_list)
    for i, data in enumerate(my_dataloader): # data : torch.Size([32, 3, 800, 400])
        batch_pred = model(data) # [batch_size, 512]
        # pred = torch.concat([pred, batch_pred], dim=0) if pred is not None else batch_pred
    
    
        for i in range(len(batch_pred)):
            # display(body_img_list[i])
            # print(concat_pred[i].detach().numpy())
            body_embedding.append(batch_pred[i].detach().numpy())
            
            karina_score = compute_cossim(body_anchors['aespa_karina'], batch_pred[i])
            winter_score = compute_cossim(body_anchors['aespa_winter'], batch_pred[i])
            ningning_score = compute_cossim(body_anchors['aespa_ningning'], batch_pred[i])
            giselle_score = compute_cossim(body_anchors['aespa_giselle'], batch_pred[i]) # torch.Tensor, size : [1]
            
            scores = [karina_score.item(), winter_score.item(), ningning_score.item(), giselle_score.item()]
            softmax_scores = my_softmax(scores) # [4] 

            scores_dict = {'aespa_karina':round(softmax_scores[0],4), 'aespa_winter':round(softmax_scores[1],4), 
                        'aespa_ningning':round(softmax_scores[2],4), 'aespa_giselle':round(softmax_scores[3],4)}
            '''
                성능 확인
            # display(body_img_list[i])
            # print(scores_dict)
            '''
            
            body_confidence.append(scores_dict)
            
            winner = max(scores_dict, key=scores_dict.get)
            body_pred.append(winner)
            # print(f"### model 예측 : {winner}") 
        
    body_embedding = pd.Series(body_embedding)
    body_confidence = pd.Series(body_confidence)
    body_pred = pd.Series(body_pred)
    df2['body_embedding'] = body_embedding
    df2['body_confidence'] = body_confidence
    df2['body_pred'] = body_pred
        
    return df2

