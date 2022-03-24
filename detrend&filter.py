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
(
    CRU_India_slope,
    CRU_India_intercept,
    CRU_India_rvalue,
    CRU_India_pvalue,
    CRU_India_hypothesis,
) = ca.dim_linregress(preCRU_India_mean, preCRU_JJA)

(
    CRU_EA_slope,
    CRU_EA_intercept,
    CRU_EA_rvalue,
    CRU_EA_pvalue,
    CRU_EA_hypothesis,
) = ca.dim_linregress(preCRU_EA_mean, preCRU_JJA)


(
    his_India_slope,
    his_India_intercept,
    his_India_rvalue,
    his_India_pvalue,
    his_India_hypothesis,
) = ca.dim_linregress(prehis_India_mean, prehis_JJA)

(
    his_EA_slope,
    his_EA_intercept,
    his_EA_rvalue,
    his_EA_pvalue,
    his_EA_hypothesis,
) = ca.dim_linregress(prehis_EA_mean, prehis_JJA)


# %%
#   plot the rvalue distribution for different area precipitation
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig_rvalue = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig_rvalue.subplots(ncols=2, nrows=2, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.arange(50, 151, 10)  # 设置纬度刻度
yticks = np.arange(10, 51, 10)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], 5, 55]
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
n = 1
# ==========================
con = axs[0, 0].contourf(
    CRU_India_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    CRU_India_pvalue,
    axs[0, 0],
    n,
    np.where(CRU_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
# axs[0,0].contour(
#     CRU_India_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[0, 0].format(
    title="Pr reg IndR", rtitle="1950-2014", ltitle="CRU TS4.01",
)
sepl.patches(axs[0, 0], 70.0, 8.0, 16.0, 20.0, proj)
# ==========================
axs[1, 0].contourf(
    CRU_EA_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    CRU_EA_pvalue, axs[1, 0], n, np.where(CRU_EA_pvalue[::n, ::n] <= 0.05), "denim", 3.0
)
# axs[0,0].contour(
#     CRU_India_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[1, 0].format(
    title="Pr reg NCR", rtitle="1950-2014", ltitle="CRU TS4.01",
)
sepl.patches(axs[1, 0], 108, 36, 10.0, 6.0, proj)
# ==========================
axs[0, 1].contourf(
    his_India_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
# axs[0,2].contour(
#     his_India_pvalue,
#     color="black",
#     levels=np.array([0.01, 0.05])
# )
sepl.plt_sig(
    his_India_pvalue,
    axs[0, 1],
    n,
    np.where(his_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[0, 1].format(
    title="Pr reg IndR", rtitle="1950-2014", ltitle="historical",
)
sepl.patches(axs[0, 1], 70.0, 8.0, 16.0, 20.0, proj)
# ==========================
axs[1, 1].contourf(
    his_EA_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
# axs[0,2].contour(
#     his_India_pvalue,
#     color="black",
#     levels=np.array([0.01, 0.05])
# )
sepl.plt_sig(
    his_EA_pvalue, axs[1, 1], n, np.where(his_EA_pvalue[::n, ::n] <= 0.05), "denim", 3.0
)
axs[1, 1].format(
    title="Pr reg NCR", rtitle="1950-2014", ltitle="historical",
)
sepl.patches(axs[1, 1], 108, 36, 10.0, 6.0, proj)
# ==========================
fig_rvalue.format(abc="(a)", abcloc="l")

# %%
window = 7
freq = "AS-JUL"
CRUtime = preCRU_India_mean.coords["time"]
histime = prehis_India_mean.coords["time"]
CRU_India_EA_regress_7 = ca.rolling_reg_index(
    preCRU_India_mean, preCRU_EA_mean, CRUtime, window, freq, True
)

window = 9
CRU_India_EA_regress_9 = ca.rolling_reg_index(
    preCRU_India_mean, preCRU_EA_mean, CRUtime, window, freq, True
)
his_India_EA_regress_9 = ca.rolling_reg_index(
    prehis_India_mean, prehis_EA_mean, histime, window, freq, True
)

window = 11
CRU_India_EA_regress_11 = ca.rolling_reg_index(
    preCRU_India_mean, preCRU_EA_mean, CRUtime, window, freq, True
)
his_India_EA_regress_11 = ca.rolling_reg_index(
    prehis_India_mean, prehis_EA_mean, histime, window, freq, True
)

# %%
fig = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
axs = fig.subplots(ncols=2, nrows=1)

lw = 1.0

m1 = axs[0].line(preCRU_India_mean, color="grey7", lw=lw)
m2 = axs[0].line(preCRU_EA_mean, color="grey7", linestyle="--", lw=lw)
m3 = axs[0].line(
    CRU_India_EA_regress_7.time,
    np.array(CRU_India_EA_regress_7["rvalue"]),
    lw=lw,
    color="blue",
)
axs[0].axhline(0.6664, lw=0.8, color="grey5", linestyle="--")
axs[0].axhline(-0.6664, lw=0.8, color="grey5", linestyle="--")
axs[0].legend(handles=[m1, m2, m3], loc="ll", labels=["IndR", "NCR", "r"], ncols=1)
axs[0].format(
    xrotation=0,
    ylim=(-2, 2),
    ylocator=0.5,
    yminorlocator=0.25,
    ylabel="precip",
    xlabel="time",
    rtitle="window= {} yr".format(window),
    ltitle="CRU TS4.01",
)


fig.format(abc="(a)", abcloc="l")

# %%

