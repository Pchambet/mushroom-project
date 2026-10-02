"""Download, verify and tidy the UCI Mushroom file."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

from mushroom.config import DATA_SHA256, DATA_URL, PROCESSED_CSV, RAW_DIR

COLUMNS = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]
TARGET = "class"
CLASS_NAMES = {"e": "edible", "p": "poisonous"}

# Codebook from agaricus-lepiota.names, kept verbatim so labels stay auditable.
_CODEBOOK = """
cap-shape: bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s
cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
bruises: bruises=t,no=f
odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
gill-attachment: attached=a,descending=d,free=f,notched=n
gill-spacing: close=c,crowded=w,distant=d
gill-size: broad=b,narrow=n
gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
stalk-shape: enlarging=e,tapering=t
stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
veil-type: partial=p,universal=u
veil-color: brown=n,orange=o,white=w,yellow=y
ring-number: none=n,one=o,two=t
ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
"""
CODEBOOK: dict[str, dict[str, str]] = {
    var: {code: word for word, code in (pair.split("=") for pair in pairs.split(","))}
    for var, pairs in (line.split(": ") for line in _CODEBOOK.strip().splitlines())
}


def download(raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Fetch the raw file once and refuse it if the checksum moved."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "agaricus-lepiota.data"
    if force or not path.exists():
        urllib.request.urlretrieve(DATA_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f"Unexpected checksum for {path}: {digest}. Re-run with --force-download.")
    return path


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Decoded labels, explicit missing category, constant columns dropped.

    ``stalk-root`` is unknown ("?") for 30.5 % of specimens. In an MCA, "unknown"
    is simply one more category; imputing the mode would invent information and,
    done before cross-validation, leak it across folds.
    """
    df = raw.copy()
    df[TARGET] = df[TARGET].map(CLASS_NAMES)
    for var, codes in CODEBOOK.items():
        df[var] = df[var].map(codes)
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Codes outside the UCI codebook in: {bad}")
    constant = [c for c in df.columns if c != TARGET and df[c].nunique() == 1]
    return df.drop(columns=constant)


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=COLUMNS, dtype=str)


def build_processed(force_download: bool = False) -> pd.DataFrame:
    df = prepare(load_raw(download(force=force_download)))
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    return df


def load_processed(path: Path = PROCESSED_CSV) -> tuple[pd.DataFrame, pd.Series]:
    """Features and a 0/1 target where 1 means poisonous (the class we must not miss)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    y = (df[TARGET] == "poisonous").astype(int)
    return df.drop(columns=TARGET), y
