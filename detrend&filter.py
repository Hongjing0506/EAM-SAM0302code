"""
Author: ChenHJ
Date: 2022-03-21 21:07:17
LastEditors: ChenHJ
LastEditTime: 2022-03-24 20:03:34
FilePath: /chenhj/0302code/detrend&filter.py
Aim: 
Mission: 
"""
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

import metpy.calc as mpcalc
import metpy.constants as constants
import geocat.comp

reload(sepl)

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
from eofs.multivariate.standard import MultivariateEof
from eofs.standard import Eof

# %%
#   the args of butterworth bandpass
ya = 2
yb = 8
# %%
#   read obs data
fhgt_ERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/hgt_mon_r144x72_195001-201412.nc"
)
hgt_ERA5 = fhgt_ERA5["z"]
hgt_ERA5 = ca.detrend_dim(hgt_ERA5, "time", deg=1, demean=False)
hgt_ERA5_filt = ca.butterworth_filter(
    hgt_ERA5 - hgt_ERA5.mean(dim="time", skipna=True), 8, ya * 12, yb * 12, "bandpass"
)

fu_ERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/uwind_mon_r144x72_195001-201412.nc"
)
u_ERA5 = fu_ERA5["u"]
u_ERA5 = ca.detrend_dim(u_ERA5, "time", deg=1, demean=False)
u_ERA5_filt = ca.butterworth_filter(
    u_ERA5 - u_ERA5.mean(dim="time", skipna=True), 8, ya * 12, yb * 12, "bandpass"
)

fhgt_his = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/zg/zg_Amon_ensemble_historical_gn_195001-201412.nc"
)
hgt_his = fhgt_his["zg"]
hgt_his = ca.detrend_dim(hgt_his, "time", deg=1, demean=False)
hgt_his_filt = ca.butterworth_filter(
    hgt_his - hgt_his.mean(dim="time", skipna=True), 8, ya * 12, yb * 12, "bandpass"
)

fu_his = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/ua/ua_Amon_ensemble_historical_gn_195001-201412.nc"
)
u_his = fu_his["ua"]
u_his = ca.detrend_dim(u_his, "time", deg=1, demean=False)
u_his_filt = ca.butterworth_filter(
    u_his - u_his.mean(dim="time", skipna=True), 8, ya * 12, yb * 12, "bandpass"
)

fpreCRU = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/cru_ts4.01_r144x72_195001-201412.nc"
)
preCRU = fpreCRU["pre"]
preCRU = ca.detrend_dim(preCRU, "time", deg=1, demean=False)
preCRU_filt = ca.butterworth_filter(
    preCRU - preCRU.mean(dim="time", skipna=True), 8, ya * 12, yb * 12, "bandpass"
)

fprehis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/pr/pr_Amon_ensemble_historical_gn_195001-201412.nc"
)
prehis = fprehis["pr"]
prehis = ca.detrend_dim(prehis, "time", deg=1, demean=False)
prehis_filt = ca.butterworth_filter(
    prehis - prehis.mean(dim="time", skipna=True), 8, ya * 12, yb * 12, "bandpass"
)

pr_his_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/pr"
)
g = os.walk(pr_his_path)
filepath = []
modelname_pr = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_pr.append(filename[loc[1] + 1 : loc[2]])
preds_his = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
prehis_ds = xr.DataArray(preds_his["pr"])
prehis_ds.coords["models"] = modelname_pr
prehis_ds_filt = ca.butterworth_filter(
    prehis_ds - prehis_ds.mean(dim="time", skipna=True), 8, ya * 12, yb * 12, "bandpass"
)

# %%
preCRU_JJA = ca.standardize((ca.p_time(preCRU_filt, 6, 8, True)))
preCRU_India_JJA = preCRU_JJA.loc[:, 8:28, 70:86]
preCRU_EA_JJA = preCRU_JJA.loc[:, 36:42, 108:118]

prehis_JJA = ca.standardize((ca.p_time(prehis_filt, 6, 8, True)))
prehis_India_JJA = prehis_JJA.loc[:, 8:28, 70:86]
prehis_EA_JJA = prehis_JJA.loc[:, 36:42, 108:118]

# %%
preCRU_India_mean = ca.cal_lat_weighted_mean(preCRU_India_JJA).mean(
    dim="lon", skipna=True
)
preCRU_EA_mean = ca.cal_lat_weighted_mean(preCRU_EA_JJA).mean(dim="lon", skipna=True)

prehis_India_mean = ca.cal_lat_weighted_mean(prehis_India_JJA).mean(
    dim="lon", skipna=True
)
prehis_EA_mean = ca.cal_lat_weighted_mean(prehis_EA_JJA).mean(dim="lon", skipna=True)

# %%
fig = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
axs = fig.subplots(ncols=1, nrows=1)
axs[0].line(preCRU_India_mean, color="grey6", lw=1.0)
axs[0].line(preCRU_EA_mean, color="grey6", linestyle="--", lw=1.0)
axs.format(
    xrotation=0,
    ylim=(-2,2),
    ylocator=0.5,
    yminorlocator=0.25
    )

# %%
