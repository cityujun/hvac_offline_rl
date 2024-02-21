"""
Helpers for script run_energyplus.py.
"""
import os
import glob
import re


def extract_energyplus_version(model_file):
    with open(model_file, "r") as mf:
        match = re.search(r'Version,[\n ]*([0-9.]+);', mf.read())
        assert match, "couldn't find EnergyPlus version in {}".format(model_file)
        energyplus_version = match.group(1)
        if len(energyplus_version.split(".")) == 2:
                energyplus_version += ".0"
        return tuple([int(p) for p in energyplus_version.split(".")])


def energyplus_locate_log_dir(index=0):
    pat_openai = energyplus_logbase_dir() + f'/openai-????-??-??-??-??-??-??????*/progress.csv'
    pat_ray = energyplus_logbase_dir() + f'/ray-????-??-??-??-??-??-??????*/*/progress.csv'
    files = [
        (f, os.path.getmtime(f))
        for pat in [pat_openai, pat_ray]
        for f in glob.glob(pat)
    ]
    newest = sorted(files, key=lambda files: files[1])[-(1 + index)][0]
    dir = os.path.dirname(newest)
    # in ray, progress.csv is in a subdir, so we need to get
    # one step upper.
    if "/ray-" in dir:
        dir = os.path.dirname(dir)
    print('energyplus_locate_log_dir: {}'.format(dir))
    return dir


def energyplus_logbase_dir():
    logbase_dir = os.getenv('ENERGYPLUS_LOGBASE')
    if logbase_dir is None:
        logbase_dir = '/tmp'
    return logbase_dir
