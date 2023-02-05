import argparse
import pandas as pd
import pickle
from copy import deepcopy
import json

def make_timeline(df1:pd.DataFrame, pred:dict, save=False) -> tuple:
    
    # for streamlit
    timeline_info = {}

    # for videogenerator
    df = df1.copy()

    # fillna to -1
    df['track_id'].fillna(-1, inplace=True)

    # track_id type change for matching pred's type
    df['track_id'] = df['track_id'].astype('int')

    # per unique track_id assign str_name
    for id in df['track_id'].unique():
        if id == -1: # if id is -1
            df.loc[df['track_id'] == -1,'name'] = "background"
        else: # if else input name
            df.loc[df['track_id'] == id,'name'] = pred[id]
    
    # per name change to list
    for name in df['name'].unique():
        if name == 'background':
            continue
        timeline_info[name] = df.loc[df['name']==name, 'frame'].tolist()

    return df, timeline_info



def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--df1_path', default="/opt/ml/final-project-level3-cv-04/test_chj/df1.csv", type=str)
    parser.add_argument('--pred_path', default="/opt/ml/final-project-level3-cv-04/test_chj/pred.pickle", type=str)
    return parser



if __name__ == "__main__":
    parser = argparse.ArgumentParser('from df1, json to timeline', parents=[get_args_parser()])
    args = parser.parse_args()

    # print args
    for arg in vars(args):
        print("--"+arg, getattr(args, arg))

    # read csv
    df1 = pd.read_csv(args.df1_path, index_col=0)
    # read pickle
    with open(args.pred_path, 'rb') as fr:
        pred = pickle.load(fr)
    
    # form change
    df1_out, timeline_info = make_timeline(df1, pred, save=True)