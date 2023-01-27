import pandas as pd
import numpy as np

from itertools import chain
from collections import Counter


def predictor(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    face_coefficient: float,
    body_coefficient: float,
    no_duplicate=False,
) -> dict:
    
    if no_duplicate == False:
        pred = {}
        for track in df2["track_id"].unique():
            face_score = (
                df2.loc[df2["track_id"] == track]["face_confidence"]
                .map(lambda x: Counter(x))
                .sum()
            )
            body_score = (
                df2.loc[df2["track_id"] == track]["body_confidence"]
                .map(lambda x: Counter(x))
                .sum()
            )
            face_series = pd.Series(face_score, dtype=float).map(
                lambda x: face_coefficient * x
            )
            body_series = pd.Series(body_score, dtype=float).map(
                lambda x: body_coefficient * x
            )

            face_series = pd.to_numeric(face_series)
            body_series = pd.to_numeric(body_series)

            prediction = (face_series + body_series).idxmax()
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
            # coincident[track].discard(track)

            face_score = (
                df2.loc[df2["track_id"] == track]["face_confidence"]
                .map(lambda x: Counter(x))
                .sum()
            )

            body_score = (
                df2.loc[df2["track_id"] == track]["body_confidence"]
                .map(lambda x: Counter(x))
                .sum()
            )

            face_series = pd.Series(face_score, dtype=float).map(
                lambda x: round(
                    face_coefficient * x / len(df2.loc[df2["track_id"] == track]), 4
                )
            )

            body_series = pd.Series(body_score, dtype=float).map(
                lambda x: round(
                    body_coefficient * x / len(df2.loc[df2["track_id"] == track]), 4
                )
            )

            pred_df["member"].loc[pred_df["track_id"] == track] = face_series.index
            pred_df["confidence"].loc[pred_df["track_id"] == track] = (
                face_series.values + body_series.values
            )

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
