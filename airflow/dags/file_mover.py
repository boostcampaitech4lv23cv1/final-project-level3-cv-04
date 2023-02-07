from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

import os
import os.path as osp
import shutil


def executer_filterout(target_path, remainder_names):
    for item in os.listdir(target_path):
        path = os.path.join(target_path, item)
        if os.path.isdir(path):
            if item not in remainder_names:
                shutil.rmtree(path)
                print("removed folder :", path)
        else:
            os.remove(path)
            print("removed file:", path)


def executer_getsamples(target_path, samples_foldername, save_path):
    target_path = osp.join(target_path, samples_foldername)
    if not osp.isdir(target_path):
        print("INFO - no sampled images folder found!")
        return
    
    os.makedirs(save_target_path, exist_ok=True)
    
    for filename in os.listdir(target_path):
        tags = filename.rstrip(".jpg").split("_")
        trackID = tags[0]
        dfIndex = tags[2]
        groupname = tags[3]
        pred1 = tags[4]
        pred2 = tags[6]
        if pred1 == pred2:
            src = osp.join(target_path, filename)
            newname = trackID + "_" + dfIndex + "_" + groupname + "_" + pred1
            os.makedirs(osp.join(save_path, pred1), exist_ok=True)
            dest = osp.join(save_path, pred1, newname + ".jpg")
            os.replace(src, dest)
            print("sample saved at", dest)

    shutil.rmtree(target_path)
    print("removed folder :", target_path)


def executer(*args):
    # executer should take 'result' folder's path
    # executer should take Destination folder's path to save train images
    TARGET_PATH = args[0]
    SAVE_PATH = args[1]
    for linkID in os.listdir(TARGET_PATH):
        if osp.isdir(osp.join(TARGET_PATH, linkID)) and len(linkID) == 11:
            current_linkID = osp.join(TARGET_PATH, linkID)
            for vidSEC in os.listdir(current_linkID):
                if osp.isdir(osp.join(current_linkID, vidSEC)):
                    current_vidSEC = osp.join(current_linkID, vidSEC)
                    
                    # csv 폴더와 sampled_images 폴더만 남기고 지움
                    executer_filterout(current_vidSEC, ["csv", "sampled_images"])
                    
                    save_target_path = osp.join(SAVE_PATH, linkID+'_'+vidSEC)
                    
                    # sampled_images 폴더 안에서 save_target_path 폴더로 샘플 이미지 가져옴
                    executer_getsamples(current_vidSEC, "sampled_images", save_target_path)


default_args = dict(
    owner="astron8t",
    depends_on_past=False,
    start_date=datetime(2023, 1, 31),
    retires=1,  # 실패시 재시도 횟수
    retry_delay=timedelta(minutes=5),  # 실패하면 5분뒤 재실행
)


with DAG(
    dag_id="get_samples_and_remove_results",
    default_args=default_args,
    schedule_interval="0 0 * * *",
    tags=["my_dags"],
) as dag:
    execute = PythonOperator(
        task_id="get_samples_and_remove_results",
        python_callable=executer,
        op_args=[
            "/opt/ml/torchkpop/result",
            "/opt/ml/torchkpop/body_embedding/train_images",
        ],
    )

    execute
