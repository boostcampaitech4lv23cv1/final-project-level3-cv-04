import os
import math
import cv2
import json
import pandas as pd
import numpy as np
import onnxruntime
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX
from tqdm import tqdm


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import onnx
from onnx2torch import convert
from collections import Counter

tqdm.pandas()



def detect_face(image: np.ndarray) -> np.ndarray:
    # image = cv2.imread(image)
    bboxes, keypoints = detector.autodetect(image, max_num=1)
    if bboxes.shape[0] == 0:
        return ([[-1.0, -1.0, -1.0, -1.0, -1.0]], [[-1.0, -1.0, -1.0, -1.0, -1.0]])
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
        return np.zeros(512), "FLAG"

    kps = kpss[0]
    face_feature = rec.get(image, kps)
    # face_feature = np.reshape(face_feature, (1, -1))

    return face_feature, round(bboxes[0][-1], 4)


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


    df2["filename"] = df2["filename"].map(
        lambda x: os.path.join(meta_info["image_root"], x)
    )

    df2["face_embedding-face_det_confidence"] = df2.apply(compute_face_feature, axis=1)
    df2["face_embedding"] = df2['face_embedding-face_det_confidence'].map(lambda x: x[0])
    df2["face_det_confidence"] = df2['face_embedding-face_det_confidence'].map(lambda x: x[1])
    df2.drop(["face_embedding-face_det_confidence"], axis=1, inplace=True)

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


    df2["face_confidence"] = (
        df2["face_embedding"]
        .map(lambda x: compute_face_confidence(x, anchor_face_embedding))
        .map(simple_softmax)
    )

    df2["face_pred"] = df2["face_confidence"].map(lambda x: max(x, key=x.get))

    return df2


########################################################################################
########################################################################################


def simple_softmax_all(confidence: dict) -> dict:

    if type(confidence) == str:
        return "FLAG"

    else:
        confidence_all = []
        for i in range(len(confidence)):
            confidence[i] = {
                key: pow(math.e, value) for key, value in confidence[i].items()
            }
            denominator = sum(confidence[i].values())
            confidence[i] = {
                key: round(value / denominator, 4)
                for key, value in confidence[i].items()
            }
            confidence_all.append(confidence[i])
        return np.array(confidence_all)


def compute_face_feature_all(row) -> np.ndarray:

    image = cv2.imread(row["full_filename"])

    if np.isnan(row["track_body_xmin"]):
        return ("FLAG", "FLAG")

    else:
        xmin = int(row["track_body_xmin"])
        xmax = int(row["track_body_xmax"])
        ymin = int(row["track_body_ymin"])
        ymax = int(row["track_body_ymax"])

        image = image[
            ymin:ymax,
            xmin:xmax,
        ]

        bboxes, kpss = detector.autodetect(image, max_num=5, metric="center_high")
        if bboxes.shape[0] == 0:
            return ("FLAG", "FLAG")

        else:

            for i in range(len(bboxes)):
                bboxes[i][0] = int(bboxes[i][0] + xmin)
                bboxes[i][1] = int(bboxes[i][1] + ymin)
                bboxes[i][2] = int(bboxes[i][2] + xmin)
                bboxes[i][3] = int(bboxes[i][3] + ymin)

            kpss[:, :, 0] = kpss[:, :, 0].astype(int) + xmin
            kpss[:, :, 1] = kpss[:, :, 1].astype(int) + ymin

            return (bboxes, kpss)


def compute_face_confidence_all(
    query_feature: np.ndarray, anchor_face_embedding: dict
) -> dict:

    if type(query_feature) == str:
        return "FLAG"

    else:
        face_feature = query_feature
        face_confidence = np.array(
            [
                {
                    key: round(rec.compute_sim(np.array(value), single_face_feature), 4)
                    for key, value in anchor_face_embedding.items()
                }
                for single_face_feature in face_feature
            ]
        )

        return face_confidence


def face_embedding_extractor_all(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    anchor_face_embedding: dict,
    meta_info: dict,
) -> pd.DataFrame:

    df1["full_filename"] = df1["filename"].map(
        lambda x: os.path.join(meta_info["image_root"], x)
    )

    print('face detection & face recognition in progress')

    #----------------------------------------------------------------------------------------------------

    df1["face_bbox-face_keypoint"] = df1.progress_apply(compute_face_feature_all, axis=1)
    df1["face_bbox"] = df1["face_bbox-face_keypoint"].map(lambda x: x[0])
    df1["face_keypoint"] = df1["face_bbox-face_keypoint"].map(lambda x: x[1])
    df1.drop(["face_bbox-face_keypoint"], axis=1, inplace=True)


    MyDataset = CustomDataset(df1["full_filename"].values, df1["face_bbox"], df1['face_keypoint'],transform)
    MyDataLoader = DataLoader(MyDataset, batch_size=200, shuffle=False, num_workers=5)

    embeddings = []
    for i, batch in enumerate(tqdm(MyDataLoader)):
        result = model(batch.cuda())
        result = result.detach().cpu().numpy()
        for j in range(len(result)):
            embeddings.append(result[j][np.newaxis, :])

    flag_checker = df1["face_bbox"].map(lambda x: True if type(x[0]) != str else False).values
    embeddings = np.array(embeddings)


    _series = []
    tick = 0
    for count, flag in zip(MyDataset.counts.tolist(), flag_checker):
        if flag == True:
            _series.append(np.squeeze(embeddings[tick: tick+count], 1))
        else:
            _series.append("FLAG")
        tick += count

    df1['face_embedding'] = _series

    #----------------------------------------------------------------------------------------------------


    df1["face_confidence"] = (
        df1["face_embedding"]
        .map(lambda x: compute_face_confidence_all(x, anchor_face_embedding))
        .map(simple_softmax_all)
    )

    
    used_in_df2 = df2['df1_index'].unique()
    df1['face_embedding'].loc[np.isin(df1.index, used_in_df2, invert=True)] = "FLAG"

    df1["face_pred"] = df1["face_confidence"].map(
        lambda x: [max(y, key=y.get) for y in x] if type(x) != str else "FLAG"
    )

    df1.drop(["full_filename"], axis=1, inplace=True)

    return df1


## group recognizer 용
def detect_face_and_extract_feature(
    image: np.ndarray
) -> np.ndarray:
    bboxes, keypoints = detector.autodetect(image, max_num=1)
    if bboxes.shape[0] == 0:
        # return (np.array([-1.0 for _ in range(512)]))
        return 'FLAG'
    else:
        face_feature = np.array([rec.get(image, kps) for kps in keypoints])
    return face_feature




########################################################################################
########################################################################################




onnxruntime.set_default_logger_severity(3)

root_dir = "./pretrained_weight"

detector = SCRFD(os.path.join(root_dir, "det_10g.onnx"))
detector.prepare(0)

model_path = os.path.join(root_dir, "w600k_r50.onnx")

rec = ArcFaceONNX(model_path)
rec.prepare(0)


onnx_model = onnx.load(model_path)
model = convert(onnx_model)
model.eval()
model.cuda()


transform = transforms.Compose([transforms.ToTensor()])


class CustomDataset(Dataset):
    def __init__(self, full_filename, face_bbox, face_keypoint, transform):

        self.counts = face_bbox.map(lambda x: len(x) if type(x[0]) != str else 1).values
        self.full_filename = np.repeat(full_filename, self.counts)

        temp = []
        for y in face_bbox:
            if type(y[0]) == str:
                temp.append("FLAG")
            else:
                for x in y:
                    temp.append(x)
        self.face_bbox = temp
        
        temp = []
        for y in face_keypoint:
            if type(y[0]) == str:
                temp.append("FLAG")
            else:
                for x in y:
                    temp.append(x)
        self.face_keypoint = temp
        
        self.transform = transform
        

    def __len__(self):
        return len(self.full_filename)
    

    def __getitem__(self, idx):
        if type(self.face_bbox[idx]) == str:
            img = np.zeros(shape=(112, 112, 3), dtype=np.uint8)

        else:
            keypoint = self.face_keypoint[idx]
            img = cv2.imread(self.full_filename[idx])
            img = rec.get(img, keypoint, mode='dataloader')

        return self.transform(img)