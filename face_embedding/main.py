import os
import math
import cv2
import json
import pandas as pd
import numpy as np
import onnxruntime
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX

def detect_face(image: np.ndarray) -> np.ndarray:
    # image = cv2.imread(image)
    bboxes, keypoints = detector.autodetect(image, max_num=1)
    if bboxes.shape[0] == 0:
        return ([[-1.0,-1.0,-1.0,-1.0,-1.0]], [[-1.0,-1.0,-1.0,-1.0,-1.0]])
    else:
        return (bboxes, keypoints)

def simple_softmax(confidence: dict) -> dict:
    confidence = {key: pow(math.e, value) for key, value in confidence.items()}
    denominator = sum(confidence.values())
    confidence = {
        key: round(value / denominator, 4) for key, value in confidence.items()
    }
    return confidence


def compute_face_feature(row) -> np.ndarray:

    image = cv2.imread(row["filename"])

    ## normalized scaling
    # H, W, _ = image.shape
    # xmin = int(abs(row["track_body_xmin"]) * W)
    # xmax = int(abs(row["track_body_xmax"]) * W)
    # ymin = int(abs(row["track_body_ymin"]) * H)
    # ymax = int(abs(row["track_body_ymax"]) * H)

    xmin = int(row["track_body_xmin"])
    xmax = int(row["track_body_xmax"])
    ymin = int(row["track_body_ymin"])
    ymax = int(row["track_body_ymax"])

    image = image[
        ymin:ymax,
        xmin:xmax,
    ]

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
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    anchor_face_embedding: dict,
    meta_info: dict,
) -> pd.DataFrame:

    df2[
        [
            "filename",
            "track_body_xmin",
            "track_body_ymin",
            "track_body_xmax",
            "track_body_ymax",
        ]
    ] = df1.loc[df2["df1_index"]][
        [
            "filename",
            "track_body_xmin",
            "track_body_ymin",
            "track_body_xmax",
            "track_body_ymax",
        ]
    ].reset_index(
        drop=True
    )

    # df2["filename"] = df2["filename"].map(
    #     lambda x: os.path.join("/opt/ml/data/frame_1080p", x)
    # )
    df2["filename"] = df2["filename"].map(
        lambda x: os.path.join(meta_info["image_root"], x)
    )

    df2["face_embedding"] = df2.apply(compute_face_feature, axis=1)

    df2.drop(
        [
            "filename",
            "track_body_xmin",
            "track_body_ymin",
            "track_body_xmax",
            "track_body_ymax",
        ],
        axis=1,
        inplace=True,
    )

    # df2["face_embedding"] = (
    #     df1.loc[df2["df1_index"]]["filename"]
    #     .map(lambda x: os.path.join("/opt/ml/data/frame_1080p", x))
    #     .map(compute_face_feature)
    #     .reset_index(drop=True)
    # )

    df2["face_confidence"] = (
        df2["face_embedding"]
        .map(lambda x: compute_face_confidence(x, anchor_face_embedding))
        .map(simple_softmax)
    )

    df2["face_pred"] = df2["face_confidence"].map(lambda x: max(x, key=x.get))

    # df2.to_csv("/opt/ml/torchkpop/df2.csv", sep=",")
    return df2


# if __name__ == "__main__":

# for meta_json in os.listdir("./data/"):
#     if os.path.splitext(meta_json)[-1] == ".json":
#         break

# with open(os.path.join("./data", meta_json), "r", encoding="utf-8") as f:
#     meta_json = json.load(f)


onnxruntime.set_default_logger_severity(3)

# root_dir = "./face_embedding"
root_dir = "./pretrained_weight"

detector = SCRFD(os.path.join(root_dir, "det_10g.onnx"))
detector.prepare(0)

model_path = os.path.join(root_dir, "w600k_r50.onnx")

rec = ArcFaceONNX(model_path)
rec.prepare(0)
