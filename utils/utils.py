import json
import pickle
from collections import OrderedDict
from itertools import repeat
from pathlib import Path


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)
    return dirname


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def write_txt(content, fname):
    fname = Path(fname)
    with fname.open("w") as handle:
        handle.write(content)


def write_pickle(content, fname):
    fname = Path(fname)
    with open(fname, "wb") as f:
        pickle.dump(content, f)


def read_pickle(fname):
    fname = Path(fname)
    with open(fname, "rb") as f:
        return pickle.load(f)
