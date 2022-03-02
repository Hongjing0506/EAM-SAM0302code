'''
Author: ChenHJ
Date: 2022-03-02 16:45:05
LastEditors: ChenHJ
LastEditTime: 2022-03-02 17:10:17
FilePath: /chenhj/0302code/data_predealing.py
Aim: 
Mission: 
'''
# %%
import numpy as np
import xarray as xr
import os
import re
from cdo import Cdo
import shutil
import sys

sys.path.append("/home/ys17-23/chenhj/self_def/")
import plot as sepl
import cal as ca
import pandas as pd
from importlib import reload

# sd.path.append("/home/ys17-23/chenhj/1201code/self_def.py")

cdo = Cdo()

# for plot
import proplot as pplt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter
from cartopy.mpl.ticker import LatitudeFormatter
from cartopy.util import add_cyclic_point
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t
# %%
modelname = ["ACCESS-CM2", "BCC-CSM2-MR", "CAMS-CSM1-0", "CanESM5", "CESM2", "CESM2-WACCM", "CMCC-ESM2", "CNRM-CM6-1", "CNRM-ESM2-1", "EC-Earth3-Veg", "EC-Earth3", "FGOALS-g3", "GFDL-CM4", "HadGEM3-GC31-LL", "IITM-ESM", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G", "MIROC-ES2L", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0", "NESM3", "NorESM2-LM", "TaiESM1", "UKESM1-0-LL"]
rlzn = ["r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r11i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f2", "r1i1p1f2", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f3", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f2", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r3i1p1f2"]
for model,rl in zip(modelname, rlzn):
    print(model)
    srcPath = "/home/ys17-23/Extension/CMIP6/CMIP6/"+ model +"/historical/" + rl + "/Amon"
    tmpPath = "/home/ys17-23/chenhj/CMIP6/tmpPath"
    dstPath = "/home/ys17-23/chenhj/CMIP6/historical"
    variable = ["zg", "ta", "wap", "ua", "va", "pr", "hus"]
    freq = "Amon"
    bash_merge_process(srcPath, tmpPath, dstPath, variable, freq, rl)