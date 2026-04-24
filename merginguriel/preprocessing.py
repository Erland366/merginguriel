import pandas as pd
import numpy as np


def find_rows_with_languagges_in_set(
    df: pd.DataFrame, feature_list: list
) -> pd.DataFrame:
    allowed_features_set = set(feature_list)

    mask = df.apply(
        lambda row: set(row[row > 0].index).issubset(allowed_features_set), axis=1
    )

    return df[mask]
