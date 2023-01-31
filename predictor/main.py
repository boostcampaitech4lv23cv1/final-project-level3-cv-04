import pandas as pd
import numpy as np

from itertools import chain
from collections import Counter


def body_or_face(row, face_coefficient, body_coefficient, threshold_1, threshold_2):

    if type(row["face_det_confidence"]) == str:
        return Counter()

    face_det_conf = float(row["face_det_confidence"])

    if face_det_conf < threshold_1:
        return Counter(
            {k: v * face_coefficient * 0 for k, v in row["face_confidence"].items()}
        ) + Counter(
            {k: v * body_coefficient * 1 for k, v in row["face_confidence"].items()}
        )
    elif threshold_1 <= face_det_conf <= threshold_2:
        return Counter(
            {k: v * face_coefficient * 0.5 for k, v in row["face_confidence"].items()}
        ) + Counter(
            {k: v * body_coefficient * 0.5 for k, v in row["face_confidence"].items()}
        )
    else:
        return Counter(
            {k: v * face_coefficient * 1 for k, v in row["face_confidence"].items()}
        ) + Counter(
            {k: v * body_coefficient * 0 for k, v in row["face_confidence"].items()}
        )


def predictor(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    face_coefficient: float = 1,
    body_coefficient: float = 1,
    threshold_1: float = 0.6,
    threshold_2: float = 0.8,
    no_duplicate: bool = False,
) -> dict:

    if no_duplicate == False:
        pred = {}
        for track in df2["track_id"].unique():
            # face_score = (
            #     df2.loc[df2["track_id"] == track]["face_confidence"]
            #     .map(lambda x: Counter(x))
            #     .sum()
            # )
            # body_score = (
            #     df2.loc[df2["track_id"] == track]["body_confidence"]
            #     .map(lambda x: Counter(x))
            #     .sum()
            # )
            # face_series = pd.Series(face_score, dtype=float).map(
            #     lambda x: face_coefficient * x
            # )
            # body_series = pd.Series(body_score, dtype=float).map(
            #     lambda x: body_coefficient * x
            # )

            # face_series = pd.to_numeric(face_series)
            # body_series = pd.to_numeric(body_series)

            # prediction = (face_series + body_series).idxmax()
            # pred[track] = prediction

            score = (
                df2.loc[df2["track_id"] == track]
                .apply(
                    lambda x: body_or_face(
                        x, face_coefficient, body_coefficient, threshold_1, threshold_2
                    ),
                    axis=1,
                )
                .sum()
            )
            prediction = pd.to_numeric(pd.Series(score, dtype=float)).idxmax()
            pred[track] = prediction

        pred[-1] = "NO_FACE_DETECTED"
        return pred

    elif no_duplicate == True:
        _df1 = df1[["frame", "track_id"]]
        _df1.dropna(axis=0, inplace=True)
        _df1["track_id"] = _df1["track_id"].map(lambda x: int(x))

        unique_sets = (
            _df1.groupby("frame")["track_id"]
            .apply(lambda x: tuple(sorted(x.values)))
            .unique()
        )

        coincident = {}
        pred_df = pd.DataFrame(
            {
                "track_id": np.repeat(
                    df2["track_id"].unique(), repeats=len(df2["face_pred"].unique())
                ),
                "member": -1,
                "confidence": -1,
            }
        )

        for track in df2["track_id"].unique():
            coincident[track] = set(
                chain.from_iterable([x for x in unique_sets if track in x])
            )
            # # coincident[track].discard(track)

            # face_score = (
            #     df2.loc[df2["track_id"] == track]["face_confidence"]
            #     .map(lambda x: Counter(x))
            #     .sum()
            # )

            # body_score = (
            #     df2.loc[df2["track_id"] == track]["body_confidence"]
            #     .map(lambda x: Counter(x))
            #     .sum()
            # )

            # face_series = pd.Series(face_score, dtype=float).map(
            #     lambda x: round(
            #         face_coefficient * x / len(df2.loc[df2["track_id"] == track]), 4
            #     )
            # )

            # body_series = pd.Series(body_score, dtype=float).map(
            #     lambda x: round(
            #         body_coefficient * x / len(df2.loc[df2["track_id"] == track]), 4
            #     )
            # )

            # pred_df["member"].loc[pred_df["track_id"] == track] = face_series.index
            # pred_df["confidence"].loc[pred_df["track_id"] == track] = (
            #     face_series.values + body_series.values
            # )
            score = (
                df2.loc[df2["track_id"] == track]
                .apply(
                    lambda x: body_or_face(
                        x, face_coefficient, body_coefficient, threshold_1, threshold_2
                    ),
                    axis=1,
                )
                .sum()
            )
            series = pd.Series(score, dtype=float)

            pred_df["member"].loc[pred_df["track_id"] == track] = series.index
            pred_df["confidence"].loc[pred_df["track_id"] == track] = series.values

        pred_df_og = pred_df.copy()

        pred = {}
        counter = 0

        while pred_df["confidence"].sum() != 0:
            member, track_id = pred_df[["member", "track_id"]].loc[
                pred_df["confidence"].idxmax()
            ]
            pred[track_id] = member
            pred_df["confidence"].loc[pred_df["track_id"] == track_id] = 0

            bitmask = np.isin(pred_df["track_id"], list(coincident[track_id]))
            temp_index = (
                pred_df.loc[bitmask].loc[pred_df.loc[bitmask]["member"] == member].index
            )
            pred_df["confidence"].loc[temp_index] = 0

        for missing_track_id in set(df2["track_id"].unique()) - set(pred.keys()):
            temp_index = pred_df_og.loc[pred_df_og["track_id"] == missing_track_id][
                "confidence"
            ].idxmax()
            pred[missing_track_id] = pred_df_og["member"].loc[temp_index]

        pred[-1] = "NO_FACE_DETECTED"

        return pred
