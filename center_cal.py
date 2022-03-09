'''
Author: ChenHJ
Date: 2022-03-09 17:17:59
LastEditors: ChenHJ
LastEditTime: 2022-03-09 20:42:07
FilePath: /chenhj/0302code/center_cal.py
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
#   read obs data
fhgt_ERA5 = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/obs/hgt_mon_r144x72_195001-201412.nc")
hgt_ERA5 = fhgt_ERA5["z"]

# %%
#   calculate JJA
hgt_ERA5_JJA = ca.p_time(hgt_ERA5, 6, 8, True)


# %%
#   select the 200hPa data and calculate the center of SAH
hgt_ERA5_JJA_200 = hgt_ERA5_JJA.sel(level = 200.0)
SAH_area = hgt_ERA5_JJA_200.loc[:,15.0:35.0,47.5:97.5]
time = SAH_area.coords["time"]
lon_axis = np.arange(47.5, 97.6, 2.5)
freq = np.zeros((len(lon_axis)))
for t in time:
    freq[int(SAH_area.sel(time=t).argmax(dim=["lat", "lon"])['lon'])] += 1.0
frequency = xr.DataArray(freq, coords=[lon_axis], dims=['lon'])
print(frequency)
# %%
