import pandas as pd
import numpy as np
from typing import Tuple, List


def split_data(data: pd.DataFrame, last: int,
               next_columns: bool = True) -> Tuple[List[pd.DataFrame], pd.DataFrame]:

    time_depended = [
        np.array(data[[f"Pay{k:02d}", f"Open{k:02d}"]]) for k in range(last)]
    time_depended = np.swapaxes(np.array(time_depended), 0, 1)
    
    age_column = ["age",]
    age = data[["age", ]] / data.age.max()

    lob = pd.get_dummies(data.LoB)
    lob_columns = [f"LoB{k}" for k in range(lob.shape[1])]

    inj_part = pd.get_dummies(data.inj_part)
    inj_part_columns = [f"inj_part{k}" for k in range(inj_part.shape[1])]

    static = pd.concat([age, lob, inj_part], axis=1)
    static.columns = age_column + lob_columns + inj_part_columns

    next_col = None
    if next_columns:
        next_col = data[[f"Pay{last:02d}", f"Open{last:02d}"]]

    return [static.values, time_depended], next_col
