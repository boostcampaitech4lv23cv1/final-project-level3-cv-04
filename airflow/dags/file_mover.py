from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

import os
import os.path as osp
import shutil


def executer_filterout(target_path, remainder_names):
    for item in os.listdir(target_path):
        path = osp.join(target_path, item)
        if osp.isdir(path):
            if item not in remainder_names:
                shutil.rmtree(path)
                print("removed folder :", path)
        else:
            os.remove(path)
            print("removed file:", path)


def executer_getsamples(target_path, linkID, vidSEC, samples_foldername, save_path):
    target_path = osp.join(target_path, samples_foldername)
    if not osp.isdir(target_path):
        print("INFO - no sampled images folder found!")
        return

    os.makedirs(save_path, exist_ok=True)

    for filename in os.listdir(target_path):
        tags = filename.rstrip(".jpg").split("_")
        trackID = tags[0]
        dfIndex = tags[2]
        groupname = tags[3]
        pred1 = tags[4]
        pred2 = tags[6]
        if pred1 == pred2:
            src = osp.join(target_path, filename)

            foldername = linkID + "_" + vidSEC + "_" + groupname + "_" + pred1

            newname = trackID + "_" + dfIndex + "_" + groupname + "_" + pred1

            os.makedirs(osp.join(save_path, foldername), exist_ok=True)

            dest = osp.join(save_path, foldername, newname + ".jpg")

            os.replace(src, dest)
            print("sample saved at", dest)

    shutil.rmtree(target_path)
    print("removed folder :", target_path)


def executer(target_path, save_path):
    # executer should take 'result' folder's path
    # executer should take Destination folder's path to save train images

    # save_path 내부의 파일 모두 삭제
    for filename in os.listdir(save_path):
        current = osp.join(save_path, filename)
        if osp.isdir(current):
            shutil.rmtree(current)
            print("removed folder:", current)
        else:
            os.remove(current)
            print("removed file:", current)

    for linkID in os.listdir(target_path):
        if osp.isdir(osp.join(target_path, linkID)) and len(linkID) == 11:
            current_linkID = osp.join(target_path, linkID)
            for vidSEC in os.listdir(current_linkID):
                if osp.isdir(osp.join(current_linkID, vidSEC)):
                    current_vidSEC = osp.join(current_linkID, vidSEC)

                    # csv 폴더와 sampled_images 폴더만 남기고 지움
                    executer_filterout(current_vidSEC, ["csv", "sampled_images"])

                    # sampled_images 폴더 안에서 save_target_path 폴더로 샘플 이미지 가져옴
                    executer_getsamples(
                        current_vidSEC, linkID, vidSEC, "sampled_images", save_path
                    )


def formatter(target_path):

    train_folder = osp.join(target_path, "train")
    os.makedirs(train_folder, exist_ok=True)
    query_folder = osp.join(target_path, "query")
    os.makedirs(query_folder, exist_ok=True)
    gallery_folder = osp.join(target_path, "gallery")
    os.makedirs(gallery_folder, exist_ok=True)

    for folder in os.listdir(target_path):
        if folder in ["train", "gallery", "query"]:
            continue

        current_path = osp.join(target_path, folder)
        if not osp.isdir(current_path):
            continue

        linkID, vidsec = (
            folder.split("_")[0],
            folder.split("_")[1] + "_" + folder.split("_")[2],
        )

        len_images = len(os.listdir(current_path))

        for i, filename in enumerate(os.listdir(current_path)):
            now_file = osp.join(current_path, filename)
            dest_filename = linkID + "_" + vidsec + "_" + filename

            ratio = (i + 1) / len_images
            # gallery
            if ratio < 0.5:
                os.replace(now_file, osp.join(gallery_folder, dest_filename))

            # train
            elif 0.5 <= ratio < 0.83:
                os.replace(now_file, osp.join(train_folder, dest_filename))

            # query
            elif ratio >= 0.83:
                os.replace(now_file, osp.join(query_folder, dest_filename))

        shutil.rmtree(osp.join(target_path, folder))


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
            "/opt/ml/torchkpop/body_embedding/data/kpop",
        ],
    )

    formatter = PythonOperator(
        task_id="format_moved_images",
        python_callable=formatter,
        op_args=["/opt/ml/torchkpop/body_embedding/data/kpop"],
    )

    execute >> formatter

