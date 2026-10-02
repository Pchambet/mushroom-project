import pandas as pd
import pytest

from mushroom.data import CODEBOOK, COLUMNS, load_raw, prepare


def test_prepare_decodes_codes_and_keeps_missing_as_a_category(sample):
    X, y = sample
    assert set(X["odor"]) <= set(CODEBOOK["odor"].values())
    assert "missing" in set(X["stalk-root"])
    assert not X.isna().any().any()
    assert set(y) == {0, 1}


def test_prepare_drops_constant_columns(sample):
    X, _ = sample
    assert "veil-type" not in X.columns  # always "partial" in the UCI file
    assert X.shape[1] == len(COLUMNS) - 2


def test_prepare_rejects_codes_outside_codebook(tmp_path):
    row = [
        "e",
        "x",
        "s",
        "n",
        "t",
        "p",
        "f",
        "c",
        "n",
        "k",
        "e",
        "e",
        "s",
        "s",
        "w",
        "w",
        "p",
        "w",
        "o",
        "p",
        "k",
        "s",
        "u",
    ]
    row[5] = "z"  # no odour is coded "z"
    path = tmp_path / "bad.data"
    path.write_text(",".join(row) + "\n")
    with pytest.raises(ValueError, match="odor"):
        prepare(load_raw(path))


def test_codebook_covers_every_feature_column():
    assert set(CODEBOOK) == set(COLUMNS) - {"class"}
    assert CODEBOOK["odor"]["f"] == "foul"
    assert CODEBOOK["stalk-root"]["?"] == "missing"


def test_load_raw_keeps_codes_as_strings(tmp_path):
    path = tmp_path / "one.data"
    path.write_text("p,x,s,n,t,p,f,c,n,k,e,e,s,s,w,w,p,w,o,p,k,s,u\n")
    raw = load_raw(path)
    assert isinstance(raw, pd.DataFrame)
    assert raw.loc[0, "class"] == "p"
