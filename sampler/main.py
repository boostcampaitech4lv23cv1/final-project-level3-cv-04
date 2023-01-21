import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None


def equal_distance_frame(array: np.array, num_sample: int) -> np.array:
    equidistant_df1_index = np.linspace(
        np.min(array), np.max(array), num=num_sample, endpoint=False
    )[1:]
    closest_df1_index = []
    for df1_index in equidistant_df1_index:
        closest_df1_index.append(array[np.absolute(array - df1_index).argmin()])
    return np.array(closest_df1_index)


def sampler(df1: pd.DataFrame, num_sample: int) -> pd.DataFrame:

    # _df1 = pd.read_csv(df1, sep=",", index_col=0)
    _df1 = df1[["frame", "track_id"]]
    _df1["track_id"].fillna(-1, inplace=True)
    _df1["track_id"] = _df1["track_id"].map(lambda x: int(x))
    _df1.sort_values(["track_id", "frame"], inplace=True)
    _df1 = _df1[_df1["track_id"] != -1]
    track_ids = _df1["track_id"].unique()

    df2 = pd.DataFrame({"track_id": np.repeat(track_ids, num_sample), "df1_index": 0})

    counter = 0
    for track in track_ids:
        df1_index_array = _df1[_df1["track_id"] == track].index
        closest_df1_index = equal_distance_frame(df1_index_array, num_sample + 1)
        # print(closest_df1_index)
        for idx in closest_df1_index:
            df2["df1_index"][counter] = idx
            counter += 1
        # df2[df2["track_id"] == track]['df1_index'] = closest_df1_index

    # df2.to_csv("/opt/ml/torchkpop/df2.csv", sep=",")
    return df2
