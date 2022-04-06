'''
Author: ChenHJ
Date: 2022-04-06 10:49:49
LastEditors: ChenHJ
LastEditTime: 2022-04-06 11:55:45
FilePath: /chenhj/0302code/pick_models.py
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
from importlib import reload
reload(sepl)

import pandas as pd
import metpy.calc as mpcalc
import metpy.constants as constants
import geocat.comp
from windspharm.xarray import VectorWind


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
from scipy import signal
from scipy.interpolate import interp2d
from eofs.multivariate.standard import MultivariateEof
from eofs.standard import Eof


def patches(ax, x0, y0, width, height, proj):
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x0, y0),
        width,
        height,
        fc="none",
        ec="grey7",
        linewidth=0.8,
        zorder=1.1,
        transform=proj,
        linestyle="--",
    )
    ax.add_patch(rect)
# %%
#   calculate the ensemble mean
def cdo_ensmean(srcPath, dstPath):
    g = os.walk(srcPath)
    inputString = ""
    mergelist = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if re.search("IITM-ESM", file_name) == None and re.search("ensemble", file_name) == None:
                mergelist.append(os.path.join(path, file_name))
    for i in range(len(mergelist)):
        inputString += mergelist[i] + " "
    print(inputString)
    cdo.ensmean(input=inputString, output=os.path.join(dstPath, var + "_Amon_ensemble_historical_gn_195001-201412.nc"))
    
variables = ["ua", "va", "hus", "ta", "wap", "ps", "pr", "zg"]
for var in variables:
    srcPath = os.path.join("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models/historical", var)
    dstPath = os.path.join("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models/historical", var)
    cdo_ensmean(srcPath, dstPath)
# %%
#   delete the bad models
def pick_models(srcPath, delmodels):
    g = os.walk(srcPath)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if re.search("_"+delmodels+"_", file_name) != None:
                print(file_name)
                os.remove(os.path.join(path, file_name))

delmodels = ["TaiESM1", "IPSL-CM6A-LR", "CanESM5", "CMCC-ESM2", "CESM2", "FGOALS-g3", "INM-CM4-8", "EC-Earth3-Veg", "INM-CM5-0", "CNRM-CM6-1", "ensemble"]
srcPath = "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models"
for delm in delmodels:
    pick_models(srcPath, delm)

# %%