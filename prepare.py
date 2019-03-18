import pandas as pd
import numpy as np
from typing import Tuple, List


def split_data(data: pd.DataFrame, last: int,
               next_columns: bool = True) -> Tuple[List[pd.DataFrame], pd.DataFrame]:

    time_depended = [
        np.array(data[[f"Pay{k:02d}", f"Open{k:02d}"]]) for k in range(last)]

    age = data.age.div(data.age.max())
    age_column = ["age", ]

    lob = pd.get_dummies(data.Lob)
    lob_columns = [f"LoB{k}" for k in range(lob.shape[1])]

    inj_part = pd.get_dummies(data.inj_part)
    inj_part_columns = [f"inj_part{k}" for k in range(inj_part.shape[1])]

    static = pd.concat([age, lob, inj_part], axis=1)
    static.columns = age_column + lob_columns + inj_part_columns

    next_col = None
    if next_columns:
        next_col = data[["fPay{last:02d}", "fOpen{last:02d}"]]

    return [static, np.array(time_depended)], next_col
