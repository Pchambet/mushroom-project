from pathlib import Path

import pandas as pd
import pytest

from mushroom.data import load_raw, prepare

FIXTURE = Path(__file__).parent / "fixtures" / "agaricus-lepiota-sample.data"


@pytest.fixture(scope="session")
def sample() -> tuple[pd.DataFrame, pd.Series]:
    """301 real UCI rows (every 27th line), decoded exactly like the full file."""
    df = prepare(load_raw(FIXTURE))
    return df.drop(columns="class"), (df["class"] == "poisonous").astype(int)
