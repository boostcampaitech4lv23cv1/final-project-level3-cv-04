from collections import Counter

import pandas as pd


def predictor(
    df2: pd.DataFrame, face_coefficient: float, body_coefficient: float
) -> dict:
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
    return pred
