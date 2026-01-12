import numpy as np
import pandas as pd
import zeus_time as zeus_time
from typing import Tuple, Optional

import modules_zeus.constants_old as constants_old

#------------------------- sampe_time_range, code from zeus ----------------

predict_sample_range: Tuple[float, float] = (1, 25)       # Mike
start_sample_std: float = constants_old.LIVE_START_SAMPLE_STD
uniform_start_prob: float = constants_old.LIVE_UNIFORM_START_OFFSET_PROB
start_offset_range: Tuple[int, int] = constants_old.LIVE_START_OFFSET_RANGE

def sample_time_range() -> Tuple[pd.Timestamp, pd.Timestamp, int]:
    """
    Sample random start and end times according to the provided ranges.
    """
    num_predict_hours = np.random.randint(*predict_sample_range)

    # see visualisation at Zeus/static/era5_start_offset_distribution.png
    if np.random.rand() > uniform_start_prob:
        start_offset = min(
            start_offset_range[1], # don't overshoot
            np.abs(
                int(np.random.normal(0, start_sample_std))
            ) + start_offset_range[0]
        )
    else:
        start_offset = int(np.random.uniform(*start_offset_range))

    start_timestamp = zeus_time.get_today("h") + pd.Timedelta(hours=start_offset)   # get_today sticks to pd.Timestamp.now(tz="GMT+0").floor("h")
    end_timestamp = start_timestamp + pd.Timedelta(hours=num_predict_hours - 1)     # so end_timestamp is inclusive.

    return start_timestamp, end_timestamp, num_predict_hours


if __name__ == "__main__":

    start_timestamp, end_timestamp, num_predict_hours = sample_time_range()
    print("start_timestamp: {}\nend_timestamp: {}\nnum_predict_hours:{}".format(start_timestamp, end_timestamp, num_predict_hours))
