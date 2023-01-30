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

transform_te = Compose([
        Resize((300, 150)), # (800,400) -> l2_norm : 300~700 정도, (160, 80) -> l2_norm : 5e-12~7e-12 정도
        ToTensor(),
    ])

def load_model():
    model = torchreid.models.build_model( # 이거 나중에 문제될듯 지금은 나의 bpbreid 가상환경의 파이썬 실행 환경변수가 opt/ml/torchreid로 잘 되어 있어서 잘 되는 듯
        name="osnet_ain_x1_0",
        num_classes=4, # 어차피 classifier 안써서 몇이든 상관없음
        loss="softmax",
        pretrained=False,
        use_gpu=True
    )
    load_weights = './pretrained_weight/osnet_ain_ms_d_c.pth.tar' # ✅ meta_info 를 이용해주세요
    load_pretrained_weights(model, load_weights)
    model.training = False
    return model

model = load_model()

### generate_body_anchor
def make_body_img_list(df1, df2_member_sorted, meta_info=None, img_num=None):
    '''
        df1 : tracked bbox info about all frame
        df2_member_sorted : 특정 member face_confi 순으로 정렬한 df2
        img_num : output list 원소 개수, none 이면 df2_member_sorted 통째로 img_list 만들어줌
        return - [img1, img2, img3, ...]
    '''
    if not img_num:
        img_num = len(df2_member_sorted)
    body_img_list = []
    # for df2_idx in range(len(df2)):
    for i in range(img_num):
        # body_img 가져오기
        df1_idx = df2_member_sorted.iloc[i]['df1_index']

        x_min = df1.loc[df1_idx]['track_body_xmin']
        y_min = df1.loc[df1_idx]['track_body_ymin']
        x_max = df1.loc[df1_idx]['track_body_xmax']
        y_max = df1.loc[df1_idx]['track_body_ymax']

        filename = df1.loc[df1_idx]['filename']        
        
        img_path = os.path.join(meta_info["image_root"], filename) # ✅ meta_info를 이용하세요
        img = Image.open(img_path).convert('RGB')

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
    del body_img_list
    concat = torch.cat(batches, dim=0) # [10, 3, 800, 400]
    del batches
    concat_pred = model(concat) # [10, 512]
    del concat
    pred_mean = torch.mean(concat_pred, dim=0) # [512]
    del concat_pred
    
    return pred_mean

def str_to_dict(df2):
    # df2['face_confidence'] type : str -> dict
    face_confidence_list = []
    for i in range(len(df2)):
        face_confidence_list.append(eval(df2['face_confidence'].loc[i]))
    
    face_confidence_series = pd.Series(face_confidence_list)
    del face_confidence_list
    df2['face_confidence'] = face_confidence_series
    return df2

def mkcol_face_pred_confi(df2):
    face_pred_confi_list = []
    for i in range(len(df2)):
        face_pred = df2['face_pred'].loc[i]
        face_confi_i = df2['face_confidence'].loc[i]
        face_confi = face_confi_i[face_pred]
        face_pred_confi_list.append(face_confi)
    face_confidence_series = pd.Series(face_pred_confi_list)
    df2['face_pred_confi'] = face_confidence_series
    return df2

def make_anchor_save_dir(save_dir):
    anchor_save_dir = os.path.join(save_dir, 'anchor_images')
    if not os.path.exists(anchor_save_dir):
         os.makedirs(anchor_save_dir)
    # 이미지 저장 경로 생성
    karina_anchor_dir = os.path.join(anchor_save_dir, 'karina_anchor')
    winter_anchor_dir = os.path.join(anchor_save_dir, 'winter_anchor')
    ningning_anchor_dir = os.path.join(anchor_save_dir, 'ningning_anchor')
    giselle_anchor_dir = os.path.join(anchor_save_dir, 'giselle_anchor')
    if not os.path.exists(karina_anchor_dir) or not os.path.exists(winter_anchor_dir) or not os.path.exists(ningning_anchor_dir) or not os.path.exists(giselle_anchor_dir):
        os.makedirs(karina_anchor_dir)
        os.makedirs(winter_anchor_dir)
        os.makedirs(ningning_anchor_dir)
        os.makedirs(giselle_anchor_dir)
    return karina_anchor_dir, winter_anchor_dir, ningning_anchor_dir, giselle_anchor_dir

def save_anchor_images(img_list_karina, img_list_winter, img_list_ningning, img_list_giselle,
                       karina_anchor_dir, winter_anchor_dir, ningning_anchor_dir, giselle_anchor_dir):
    for i, img in enumerate(img_list_karina):
        save_path = os.path.join(karina_anchor_dir, f'karina_anchor_{i+1}.jpg')
        img.save(save_path, "JPEG")
    for i, img in enumerate(img_list_winter):
        save_path = os.path.join(winter_anchor_dir, f'winter_anchor_{i+1}.jpg')
        img.save(save_path, "JPEG")
    for i, img in enumerate(img_list_ningning):
        save_path = os.path.join(ningning_anchor_dir, f'ningning_anchor_{i+1}.jpg')
        img.save(save_path, "JPEG")
    for i, img in enumerate(img_list_giselle):
        save_path = os.path.join(giselle_anchor_dir, f'giselle_anchor_{i+1}.jpg')
        img.save(save_path, "JPEG")

def make_anchor_img_lists(df1, 
                       df2_karina_sorted, df2_winter_sorted, df2_ningning_sorted, df2_giselle_sorted, 
                       meta_info):
    try: # karina 대표 이미지 9장으로 뽑는 중...
        body_img_list_karina = make_body_img_list(df1, df2_karina_sorted, meta_info, img_num=9)
    except: # karina 대표 이미지 6장으로 뽑는 중...
        body_img_list_karina = make_body_img_list(df1, df2_karina_sorted, meta_info, img_num=6)
    try: # winter 대표 이미지 9장 뽑는 중...
        body_img_list_winter = make_body_img_list(df1, df2_winter_sorted, meta_info, img_num=9)
    except: # winter 대표 이미지 6장 뽑는 중...
        body_img_list_winter = make_body_img_list(df1, df2_winter_sorted, meta_info, img_num=6)
    try:
        body_img_list_ningning = make_body_img_list(df1, df2_ningning_sorted, meta_info, img_num=9)
    except:
        body_img_list_ningning = make_body_img_list(df1, df2_ningning_sorted, meta_info, img_num=6)
    try:
        body_img_list_giselle = make_body_img_list(df1, df2_giselle_sorted, meta_info, img_num=9)
    except:
        body_img_list_giselle = make_body_img_list(df1, df2_giselle_sorted, meta_info, img_num=6)
    return body_img_list_karina, body_img_list_winter, body_img_list_ningning, body_img_list_giselle

def make_anchors(model, 
                 body_img_list_karina, body_img_list_winter, body_img_list_ningning, body_img_list_giselle
                 ):
    anchor_karina = make_anchor(body_img_list_karina, model)
    del body_img_list_karina
    anchor_winter = make_anchor(body_img_list_winter, model)
    del body_img_list_winter
    anchor_ningning = make_anchor(body_img_list_ningning, model)
    del body_img_list_ningning
    anchor_giselle = make_anchor(body_img_list_giselle, model)
    del body_img_list_giselle
    
    anchors = {
        'aespa_karina' : anchor_karina, 
        'aespa_winter' : anchor_winter, 
        'aespa_ningning' : anchor_ningning, 
        'aespa_giselle' : anchor_giselle
    }
    return anchors

def generate_body_anchor(df1, df2, save_dir, group_name='aespa', meta_info=None):    
    # csv 저장과정 dict->str 이슈 해결
    if type(df2['face_confidence'].loc[0]) == str:
        df2 = str_to_dict(df2)
    
    # face_pred_confi 컬럼 추가 - pred한 인물에 대한 confi
    df2 = mkcol_face_pred_confi(df2)
    
    # df2 멤버별로 분리 후 confi_score 순으로 정렬
    df2_karina_sorted = df2[df2['face_pred'] == 'aespa_karina'].sort_values(by='face_pred_confi', ascending=False)
    df2_winter_sorted = df2[df2['face_pred'] == 'aespa_winter'].sort_values(by='face_pred_confi', ascending=False)
    df2_ningning_sorted = df2[df2['face_pred'] == 'aespa_ningning'].sort_values(by='face_pred_confi', ascending=False)
    df2_giselle_sorted = df2[df2['face_pred'] == 'aespa_giselle'].sort_values(by='face_pred_confi', ascending=False)
    
    # make save_dir for anchor images
    karina_anchor_dir, winter_anchor_dir, ningning_anchor_dir, giselle_anchor_dir = make_anchor_save_dir(save_dir)
    
    img_list_karina, img_list_winter, img_list_ningning, img_list_giselle = make_anchor_img_lists(df1, df2_karina_sorted, df2_winter_sorted, df2_ningning_sorted, df2_giselle_sorted, meta_info)
    
    # body anchor에 쓰인 이미지 저장
    save_anchor_images(img_list_karina, img_list_winter, img_list_ningning, img_list_giselle,
                       karina_anchor_dir, winter_anchor_dir, ningning_anchor_dir, giselle_anchor_dir)
    
    # 멤버별 대표 이미지로 anchor embedding 생성 [512]
    anchors = make_anchors(model, img_list_karina, img_list_winter, img_list_ningning, img_list_giselle)
    
    return anchors

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

def compute_cossim(anchor, body_feature): # anchor : [512]
    output = torch.nn.CosineSimilarity(dim=0)(anchor, body_feature)
    return output

def my_softmax(scores):
    exp_scores = np.exp(scores) # [4]    
    for i in range(len(exp_scores)):
        exp_scores[i] = exp_scores[i]*100
    exp_scores = np.exp(exp_scores)
    sum_exp_scores = np.sum(exp_scores) # 값
    output = exp_scores / sum_exp_scores # [4] / 값
    return output # [4]

def compute_scores(body_anchors, pred):
    karina_score = compute_cossim(body_anchors['aespa_karina'], pred)
    winter_score = compute_cossim(body_anchors['aespa_winter'], pred)
    ningning_score = compute_cossim(body_anchors['aespa_ningning'], pred)
    giselle_score = compute_cossim(body_anchors['aespa_giselle'], pred) # torch.Tensor, size : [1]
    
    scores = [karina_score.item(), winter_score.item(), ningning_score.item(), giselle_score.item()]
    softmax_scores = my_softmax(scores) # [4] 
    scores_dict = {'aespa_karina':round(softmax_scores[0],4), 'aespa_winter':round(softmax_scores[1],4), 
                        'aespa_ningning':round(softmax_scores[2],4), 'aespa_giselle':round(softmax_scores[3],4)}
    return scores_dict

def add_body_columns(df2, body_embedding, body_confidence, body_pred):
    body_embedding = pd.Series(body_embedding)
    body_confidence = pd.Series(body_confidence)
    body_pred = pd.Series(body_pred)
    df2['body_embedding'] = body_embedding
    df2['body_confidence'] = body_confidence
    df2['body_pred'] = body_pred
    
    return df2

### body_embedding_extractor
def body_embedding_extractor(df1, df2, body_anchors, meta_info):
    '''
        add (body_embedding, body_confidence, body_pred) columns to df2
        df2의 모든 row(sampled bboxes)를 body_anchors와 비교해 어떤 anchor와 가장 비슷한지 prediction 하는 함수
    '''
    body_img_list = make_body_img_list(df1, df2, meta_info)
    my_dataset = MyDataset(body_img_list, transform_te)
    my_dataloader = DataLoader(my_dataset, batch_size=16)
    
    body_embedding = []
    body_confidence = []
    body_pred = []
    
    for i, data in enumerate(my_dataloader): # data : torch.Size([{batch_size}, 3, 800, 400])
        batch_pred = model(data) # [{batch_size}, 512]
        
        for i in range(len(batch_pred)): # batch_pred[i] : [512]
            body_embedding.append(batch_pred[i].detach().numpy())
            scores_dict = compute_scores(body_anchors, batch_pred[i])
            body_confidence.append(scores_dict)
            winner = max(scores_dict, key=scores_dict.get)
            body_pred.append(winner)
    
    df2 = add_body_columns(df2, body_embedding, body_confidence, body_pred)
    return df2