import os
import cv2
import pandas as pd
import numpy as np
import onnxruntime
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX


def detect_face(image: str):
    image = cv2.imread(image)
    bboxes, _ = detector.autodetect(image, max_num=1)
    if bboxes.shape[0] == 0:
        return -1.0, "Face not found in Image"
    else:
        return bboxes


def compute_face_feature(image: str) -> np.ndarray:
    image = cv2.imread(image)
    bboxes, kpss = detector.autodetect(image, max_num=1)
    if bboxes.shape[0] == 0:
        return np.zeros(512)

    kps = kpss[0]
    face_feature = rec.get(image, kps)
    # face_feature = np.reshape(face_feature, (1, -1))

    return face_feature


def compute_face_confidence(
    query_feature: np.ndarray, anchor_face_embedding: dict
) -> dict:

    face_feature = query_feature
    face_confidence = {
        key: round(rec.compute_sim(np.array(value), face_feature), 4)
        for key, value in anchor_face_embedding.items()
    }
    return face_confidence


def face_embedding_extractor(
    df1: pd.DataFrame, df2: pd.DataFrame, anchor_face_embedding: dict
) -> pd.DataFrame:

    df2["face_embedding"] = (
        df1.loc[df2["df1_index"]]["filename"]
        .map(lambda x: os.path.join("/opt/ml/data/frame_1080p", x))
        .map(compute_face_feature)
        .values
    )
    df2["face_confidence"] = df2["face_embedding"].map(
        lambda x: compute_face_confidence(x, anchor_face_embedding)
    )
    df2["face_pred"] = df2["face_confidence"].map(lambda x: max(x, key=x.get))

    # df2.to_csv("/opt/ml/torchkpop/df2.csv", sep=",")
    return df2


# if __name__ == "__main__":

onnxruntime.set_default_logger_severity(3)

root_dir = "/opt/ml/torchkpop/face_embedding/"

detector = SCRFD(os.path.join(root_dir, "model/det_10g.onnx"))
detector.prepare(0)

model_path = os.path.join(root_dir, "model", "w600k_r50.onnx")

rec = ArcFaceONNX(model_path)
rec.prepare(0)
