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
fhgtERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/hgt_mon_r144x72_195001-201412.nc"
)
hgtERA5 = fhgtERA5["z"]
hgtERA5 = ca.detrend_dim(hgtERA5, "time", deg=1, demean=False)
hgtERA5_filt = ca.butterworth_filter(hgtERA5, 8, ya * 12, yb * 12, "bandpass")

fuERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/uwind_mon_r144x72_195001-201412.nc"
)
uERA5 = fuERA5["u"]
uERA5 = ca.detrend_dim(uERA5, "time", deg=1, demean=False)
uERA5_filt = ca.butterworth_filter(uERA5, 8, ya * 12, yb * 12, "bandpass")

fvERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/vwind_mon_r144x72_195001-201412.nc"
)
vERA5 = fvERA5["v"]
vERA5 = ca.detrend_dim(vERA5, "time", deg=1, demean=False)
vERA5_filt = ca.butterworth_filter(vERA5, 8, ya * 12, yb * 12, "bandpass")

fhgthis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/zg/zg_Amon_ensemble_historical_gn_195001-201412.nc"
)
hgthis = fhgthis["zg"]
hgthis = ca.detrend_dim(hgthis, "time", deg=1, demean=False)
hgthis_filt = ca.butterworth_filter(hgthis, 8, ya * 12, yb * 12, "bandpass")

fuhis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/ua/ua_Amon_ensemble_historical_gn_195001-201412.nc"
)
uhis = fuhis["ua"]
uhis = ca.detrend_dim(uhis, "time", deg=1, demean=False)
uhis_filt = ca.butterworth_filter(uhis, 8, ya * 12, yb * 12, "bandpass")

fvhis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/va/va_Amon_ensemble_historical_gn_195001-201412.nc"
)
vhis = fvhis["va"]
vhis = ca.detrend_dim(vhis, "time", deg=1, demean=False)
vhis_filt = ca.butterworth_filter(vhis, 8, ya * 12, yb * 12, "bandpass")


fpreCRU = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/cru_ts4.01_r144x72_195001-201412.nc"
)
preCRU = fpreCRU["pre"]
preCRU = ca.detrend_dim(preCRU, "time", deg=1, demean=False)
preCRU_filt = ca.butterworth_filter(preCRU, 8, ya * 12, yb * 12, "bandpass")

fprehis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/pr/pr_Amon_ensemble_historical_gn_195001-201412.nc"
)
prehis = fprehis["pr"]
prehis = ca.detrend_dim(prehis, "time", deg=1, demean=False)
prehis_filt = ca.butterworth_filter(prehis, 8, ya * 12, yb * 12, "bandpass")

prhis_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/pr"
)
g = os.walk(prhis_path)
filepath = []
modelname_pr = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_pr.append(filename[loc[1] + 1 : loc[2]])
predshis = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
prehis_ds = xr.DataArray(predshis["pr"])
prehis_ds.coords["models"] = modelname_pr
prehis_ds_filt = ca.butterworth_filter(prehis_ds, 8, ya * 12, yb * 12, "bandpass")

fspERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/sp_mon_r144x72_195001-201412.nc"
)
spERA5 = fspERA5["sp"]
spERA5 = ca.detrend_dim(spERA5, "time", deg=1, demean=False)
spERA5_filt = ca.butterworth_filter(spERA5, 8, ya * 12, yb * 12, "bandpass")

fsphis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/ps/ps_Amon_ensemble_historical_gn_195001-201412.nc"
)
sphis = fsphis["ps"]
sphis = ca.detrend_dim(sphis, "time", deg=1, demean=False)
sphis_filt = ca.butterworth_filter(sphis, 8, ya * 12, yb * 12, "bandpass")

fqERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/q_mon_r144x72_195001-201412.nc"
)
qERA5 = fqERA5["q"]
qERA5 = ca.detrend_dim(qERA5, "time", deg=1, demean=False)
qERA5_filt = ca.butterworth_filter(qERA5, 8, ya * 12, yb * 12, "bandpass")

fqhis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/hus/hus_Amon_ensemble_historical_gn_195001-201412.nc"
)
qhis = fqhis["hus"]
qhis = ca.detrend_dim(qhis, "time", deg=1, demean=False)
qhis_filt = ca.butterworth_filter(qhis, 8, ya * 12, yb * 12, "bandpass")

# %%
preCRU_JJA = ca.standardize((ca.p_time(preCRU_filt, 6, 8, True)))
preCRU_India_JJA = preCRU_JJA.loc[:, 8:28, 70:86]
preCRU_EA_JJA = preCRU_JJA.loc[:, 36:42, 108:118]

prehis_JJA = ca.standardize((ca.p_time(prehis_filt, 6, 8, True)))
prehis_India_JJA = prehis_JJA.loc[:, 8:28, 70:86]
prehis_EA_JJA = prehis_JJA.loc[:, 36:42, 108:118]

uERA5_ver_JJA = (ca.p_time(uERA5_filt, 6, 8, True) + ca.p_time(uERA5, 6, 8, True)).loc[
    :, 100.0:, :, :
]
vERA5_ver_JJA = (ca.p_time(vERA5_filt, 6, 8, True) + ca.p_time(vERA5, 6, 8, True)).loc[
    :, 100.0:, :, :
]
qERA5_ver_JJA = (ca.p_time(qERA5_filt, 6, 8, True) + ca.p_time(qERA5, 6, 8, True)).loc[
    :, 100.0:, :, :
]
spERA5_ver_JJA = (
    ca.p_time(spERA5_filt, 6, 8, True) + ca.p_time(spERA5, 6, 8, True)
).loc[:, :, :]
hgtERA5_ver_JJA = (
    ca.p_time(hgtERA5_filt, 6, 8, True) + ca.p_time(hgtERA5, 6, 8, True)
).loc[:, 100.0:, :, :]

uhis_ver_JJA = (ca.p_time(uhis_filt, 6, 8, True) + ca.p_time(uhis, 6, 8, True)).loc[
    :, :10000.0, :, :
]
vhis_ver_JJA = (ca.p_time(vhis_filt, 6, 8, True) + ca.p_time(vhis, 6, 8, True)).loc[
    :, :10000.0, :, :
]
qhis_ver_JJA = (ca.p_time(qhis_filt, 6, 8, True) + ca.p_time(qhis, 6, 8, True)).loc[
    :, :10000.0, :, :
]
sphis_ver_JJA = (ca.p_time(sphis_filt, 6, 8, True) + ca.p_time(sphis, 6, 8, True)).loc[
    :, :, :
]
hgthis_ver_JJA = (
    ca.p_time(hgthis_filt, 6, 8, True) + ca.p_time(hgthis, 6, 8, True)
).loc[:, :10000.0, :, :]

preCRU_India_mean = ca.cal_lat_weighted_mean(preCRU_India_JJA).mean(
    dim="lon", skipna=True
)
preCRU_EA_mean = ca.cal_lat_weighted_mean(preCRU_EA_JJA).mean(dim="lon", skipna=True)

prehis_India_mean = ca.cal_lat_weighted_mean(prehis_India_JJA).mean(
    dim="lon", skipna=True
)
prehis_EA_mean = ca.cal_lat_weighted_mean(prehis_EA_JJA).mean(dim="lon", skipna=True)


# %%
#   calculate the water vapor vertical intergration
ptop = 100 * 100
g = 9.8
ERA5level = qERA5_ver_JJA.coords["level"] * 100.0
ERA5level.attrs["units"] = "Pa"
ERA5dp = geocat.comp.dpres_plevel(ERA5level, spERA5_ver_JJA, ptop)
ERA5dpg = ERA5dp / g
ERA5dpg.attrs["units"] = "kg/m2"
# calculate the water vapor transport
uq_ERA5 = uERA5_ver_JJA * qERA5_ver_JJA * 1000.0
vq_ERA5 = vERA5_ver_JJA * qERA5_ver_JJA * 1000.0
uq_ERA5.attrs["units"] = "[m/s][g/kg]"
vq_ERA5.attrs["units"] = "[m/s][g/kg]"
# calculate the whole levels water vapor transport
uq_dpg_ERA5 = (uq_ERA5 * ERA5dpg.data).sum(dim="level", skipna=True)
vq_dpg_ERA5 = (vq_ERA5 * ERA5dpg.data).sum(dim="level")
uq_dpg_ERA5.attrs["units"] = "[m/s][g/kg]"
vq_dpg_ERA5.attrs["units"] = "[m/s][g/kg]"


hislevel = qhis_ver_JJA.coords["plev"]
hislevel.attrs["units"] = "Pa"
hisdp = geocat.comp.dpres_plevel(hislevel, sphis_ver_JJA, ptop)
hisdpg = hisdp / g
hisdpg.attrs["units"] = "kg/m2"
# calculate the water vapor transport
uq_his = uhis_ver_JJA * qhis_ver_JJA * 1000.0
vq_his = vhis_ver_JJA * qhis_ver_JJA * 1000.0
uq_his.attrs["units"] = "[m/s][g/kg]"
vq_his.attrs["units"] = "[m/s][g/kg]"
# calculate the whole levels water vapor transport
uq_dpg_his = (uq_his * hisdpg.data).sum(dim="plev")
vq_dpg_his = (vq_his * hisdpg.data).sum(dim="plev")
uq_dpg_his.attrs["units"] = "[m/s][g/kg]"
vq_dpg_his.attrs["units"] = "[m/s][g/kg]"

# %%
preCRU_India_mean.coords["time"] = uq_dpg_ERA5.coords["time"]
(
    uq_CRU_India_slope,
    uq_CRU_India_intercept,
    uq_CRU_India_rvalue,
    uq_CRU_India_pvalue,
    uq_CRU_India_hypothesis,
) = ca.dim_linregress(preCRU_India_mean, ca.standardize(uq_dpg_ERA5))

(
    vq_CRU_India_slope,
    vq_CRU_India_intercept,
    vq_CRU_India_rvalue,
    vq_CRU_India_pvalue,
    vq_CRU_India_hypothesis,
) = ca.dim_linregress(preCRU_India_mean, ca.standardize(vq_dpg_ERA5))

(
    uq_his_India_slope,
    uq_his_India_intercept,
    uq_his_India_rvalue,
    uq_his_India_pvalue,
    uq_his_India_hypothesis,
) = ca.dim_linregress(prehis_India_mean, ca.standardize(uq_dpg_his))

(
    vq_his_India_slope,
    vq_his_India_intercept,
    vq_his_India_rvalue,
    vq_his_India_pvalue,
    vq_his_India_hypothesis,
) = ca.dim_linregress(prehis_India_mean, ca.standardize(vq_dpg_his))
# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig_rvalue = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig_rvalue.subplots(ncols=2, nrows=2, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.arange(50, 151, 10)  # ??????????????????
yticks = np.arange(10, 51, 10)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
extents = [xticks[0], xticks[-1], 5, 55]
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
n = 1
# ==========================
con = axs[0, 0].contourf(
    uq_CRU_India_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    uq_CRU_India_pvalue,
    axs[0, 0],
    n,
    np.where(uq_CRU_India_pvalue[::n, ::n] <= 0.05),
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
    title="Uq reg IndR", rtitle="1950-2014", ltitle="CRU TS4.01",
)
sepl.patches(axs[0, 0], 70.0, 8.0, 16.0, 20.0, proj)
sepl.patches(axs[0, 0], 108, 36, 10.0, 6.0, proj)
# ==========================
con = axs[0, 1].contourf(
    vq_CRU_India_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    vq_CRU_India_pvalue,
    axs[0, 1],
    n,
    np.where(vq_CRU_India_pvalue[::n, ::n] <= 0.05),
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
axs[0, 1].format(
    title="Vq reg IndR", rtitle="1950-2014", ltitle="CRU TS4.01",
)
sepl.patches(axs[0, 1], 70.0, 8.0, 16.0, 20.0, proj)
sepl.patches(axs[0, 1], 108, 36, 10.0, 6.0, proj)
# ==========================
con = axs[1, 0].contourf(
    uq_his_India_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    uq_his_India_pvalue,
    axs[1, 0],
    n,
    np.where(uq_his_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
# axs[0,0].contour(
#     his_India_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[1, 0].format(
    title="Uq reg IndR", rtitle="1950-2014", ltitle="historical",
)
sepl.patches(axs[1, 0], 70.0, 8.0, 16.0, 20.0, proj)
sepl.patches(axs[1, 0], 108, 36, 10.0, 6.0, proj)
# ==========================
con = axs[1, 1].contourf(
    vq_his_India_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    vq_his_India_pvalue,
    axs[1, 1],
    n,
    np.where(vq_his_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
# axs[0,0].contour(
#     his_India_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[1, 1].format(
    title="Vq reg IndR", rtitle="1950-2014", ltitle="historical",
)
sepl.patches(axs[1, 1], 70.0, 8.0, 16.0, 20.0, proj)
sepl.patches(axs[1, 1], 108, 36, 10.0, 6.0, proj)
fig_rvalue.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig_rvalue.format(abc="(a)", abcloc="l")
# %%
preCRU_EA_mean.coords["time"] = uq_dpg_ERA5.coords["time"]
(
    uq_CRU_EA_slope,
    uq_CRU_EA_intercept,
    uq_CRU_EA_rvalue,
    uq_CRU_EA_pvalue,
    uq_CRU_EA_hypothesis,
) = ca.dim_linregress(preCRU_EA_mean, ca.standardize(uq_dpg_ERA5))

(
    vq_CRU_EA_slope,
    vq_CRU_EA_intercept,
    vq_CRU_EA_rvalue,
    vq_CRU_EA_pvalue,
    vq_CRU_EA_hypothesis,
) = ca.dim_linregress(preCRU_EA_mean, ca.standardize(vq_dpg_ERA5))

(
    uq_his_EA_slope,
    uq_his_EA_intercept,
    uq_his_EA_rvalue,
    uq_his_EA_pvalue,
    uq_his_EA_hypothesis,
) = ca.dim_linregress(prehis_EA_mean, ca.standardize(uq_dpg_his))

(
    vq_his_EA_slope,
    vq_his_EA_intercept,
    vq_his_EA_rvalue,
    vq_his_EA_pvalue,
    vq_his_EA_hypothesis,
) = ca.dim_linregress(prehis_EA_mean, ca.standardize(vq_dpg_his))
# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig_rvalue = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig_rvalue.subplots(ncols=2, nrows=2, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.arange(50, 151, 10)  # ??????????????????
yticks = np.arange(10, 51, 10)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
extents = [xticks[0], xticks[-1], 5, 55]
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
n = 1
# ==========================
con = axs[0, 0].contourf(
    uq_CRU_EA_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    uq_CRU_EA_pvalue,
    axs[0, 0],
    n,
    np.where(uq_CRU_EA_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
# axs[0,0].contour(
#     CRU_EA_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[0, 0].format(
    title="Uq reg NCR", rtitle="1950-2014", ltitle="CRU TS4.01",
)
sepl.patches(axs[0, 0], 70.0, 8.0, 16.0, 20.0, proj)
sepl.patches(axs[0, 0], 108, 36, 10.0, 6.0, proj)
# ==========================
con = axs[0, 1].contourf(
    vq_CRU_EA_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    vq_CRU_EA_pvalue,
    axs[0, 1],
    n,
    np.where(vq_CRU_EA_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
# axs[0,0].contour(
#     CRU_EA_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[0, 1].format(
    title="Vq reg NCR", rtitle="1950-2014", ltitle="CRU TS4.01",
)
sepl.patches(axs[0, 1], 70.0, 8.0, 16.0, 20.0, proj)
sepl.patches(axs[0, 1], 108, 36, 10.0, 6.0, proj)
# ==========================
con = axs[1, 0].contourf(
    uq_his_EA_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    uq_his_EA_pvalue,
    axs[1, 0],
    n,
    np.where(uq_his_EA_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
# axs[0,0].contour(
#     his_EA_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[1, 0].format(
    title="Uq reg NCR", rtitle="1950-2014", ltitle="historical",
)
sepl.patches(axs[1, 0], 70.0, 8.0, 16.0, 20.0, proj)
sepl.patches(axs[1, 0], 108, 36, 10.0, 6.0, proj)
# ==========================
con = axs[1, 1].contourf(
    vq_his_EA_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    vq_his_EA_pvalue,
    axs[1, 1],
    n,
    np.where(vq_his_EA_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
# axs[0,0].contour(
#     his_India_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[1, 1].format(
    title="Vq reg NCR", rtitle="1950-2014", ltitle="historical",
)
sepl.patches(axs[1, 1], 70.0, 8.0, 16.0, 20.0, proj)
sepl.patches(axs[1, 1], 108, 36, 10.0, 6.0, proj)
fig_rvalue.colorbar(con, loc="b", width=0.13, length=0.7, label="")
# %%
preCRU_JJA.coords["time"] = preCRU_India_mean.time

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
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig_rvalue = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig_rvalue.subplots(ncols=2, nrows=2, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.arange(50, 151, 10)  # ??????????????????
yticks = np.arange(10, 51, 10)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
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
window = 11
freq = "AS-JUL"
CRUtime = preCRU_India_mean.coords["time"]
histime = prehis_India_mean.coords["time"]
CRU_India_EA_regress = ca.rolling_reg_index(
    preCRU_India_mean, preCRU_EA_mean, CRUtime, window, freq, True
)
his_India_EA_regress = ca.rolling_reg_index(
    prehis_India_mean, prehis_EA_mean, histime, window, freq, True
)


# %%
fig = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
axs = fig.subplots(ncols=2, nrows=1)

lw = 1.0
# ========================================
m1 = axs[0, 0].line(
    preCRU_India_mean.time.dt.year, preCRU_India_mean, color="grey7", lw=lw
)
m2 = axs[0, 0].line(
    preCRU_India_mean.time.dt.year, preCRU_EA_mean, color="grey7", linestyle="--", lw=lw
)
m3 = axs[0, 0].line(
    CRU_India_EA_regress.time.dt.year,
    np.array(CRU_India_EA_regress["rvalue"]),
    lw=lw,
    color="blue",
)
axs[0, 0].axhline(0.6664, lw=0.8, color="grey5", linestyle="--")
axs[0, 0].axhline(-0.6664, lw=0.8, color="grey5", linestyle="--")
axs[0, 0].legend(handles=[m1, m2, m3], loc="ll", labels=["IndR", "NCR", "r"], ncols=1)
axs[0, 0].format(
    xrotation=0,
    ylim=(-2, 2),
    ylocator=0.5,
    yminorlocator=0.25,
    ylabel="precip",
    xlabel="time",
    rtitle="{} yr".format(window),
    ltitle="CRU TS4.01",
)
# ===============================
m1 = axs[0, 1].line(
    preCRU_India_mean.time.dt.year, prehis_India_mean, color="grey7", lw=lw
)
m2 = axs[0, 1].line(
    preCRU_India_mean.time.dt.year, prehis_EA_mean, color="grey7", linestyle="--", lw=lw
)
m3 = axs[0, 1].line(
    his_India_EA_regress.time.dt.year,
    np.array(his_India_EA_regress["rvalue"]),
    lw=lw,
    color="blue",
)
axs[0, 1].axhline(0.6664, lw=0.8, color="grey5", linestyle="--")
axs[0, 1].axhline(-0.6664, lw=0.8, color="grey5", linestyle="--")
axs[0, 1].legend(handles=[m1, m2, m3], loc="ll", labels=["IndR", "NCR", "r"], ncols=1)
axs[0, 1].format(
    xrotation=0,
    ylim=(-2, 2),
    ylocator=0.5,
    yminorlocator=0.25,
    ylabel="precip",
    xlabel="time",
    rtitle="{} yr".format(window),
    ltitle="historical",
)

axs.format(xlocator=5)
fig.format(abc="(a)", abcloc="l")

# %%
ERA5time = uERA5_ver_JJA.coords["time"]
uERA5_ver_JJA_p1 = uERA5_ver_JJA.sel(time=(uERA5_ver_JJA.time.dt.year <= 1966))
uERA5_ver_JJA_p2 = uERA5_ver_JJA.sel(
    time=((uq_dpg_ERA5.time.dt.year >= 1967) & (uq_dpg_ERA5.time.dt.year <= 2001))
)

vERA5_ver_JJA_p1 = vERA5_ver_JJA.sel(time=(vERA5_ver_JJA.time.dt.year <= 1966))
vERA5_ver_JJA_p2 = vERA5_ver_JJA.sel(
    time=(uq_dpg_ERA5.time.dt.year >= 1967) & (uq_dpg_ERA5.time.dt.year <= 2001)
)

hgtERA5_ver_JJA_p1 = hgtERA5_ver_JJA.sel(time=(hgtERA5_ver_JJA.time.dt.year <= 1966))
hgtERA5_ver_JJA_p2 = hgtERA5_ver_JJA.sel(
    time=((uq_dpg_ERA5.time.dt.year >= 1967) & (uq_dpg_ERA5.time.dt.year <= 2001))
)

uq_dpg_ERA5_ver_JJA_p1 = uq_dpg_ERA5.sel(time=(uq_dpg_ERA5.time.dt.year <= 1966))
uq_dpg_ERA5_ver_JJA_p2 = uq_dpg_ERA5.sel(
    time=((uq_dpg_ERA5.time.dt.year >= 1967) & (uq_dpg_ERA5.time.dt.year <= 2001))
)

vq_dpg_ERA5_ver_JJA_p1 = vq_dpg_ERA5.sel(time=(vq_dpg_ERA5.time.dt.year <= 1966))
vq_dpg_ERA5_ver_JJA_p2 = vq_dpg_ERA5.sel(
    time=((uq_dpg_ERA5.time.dt.year >= 1967) & (uq_dpg_ERA5.time.dt.year <= 2001))
)

# %%
uERA5_ver_JJA_p1_mean = uERA5_ver_JJA_p1.mean(dim="time", skipna=True)
uERA5_ver_JJA_p2_mean = uERA5_ver_JJA_p2.mean(dim="time", skipna=True)

vERA5_ver_JJA_p1_mean = vERA5_ver_JJA_p1.mean(dim="time", skipna=True)
vERA5_ver_JJA_p2_mean = vERA5_ver_JJA_p2.mean(dim="time", skipna=True)

hgtERA5_ver_JJA_p1_mean = hgtERA5_ver_JJA_p1.mean(dim="time", skipna=True)
hgtERA5_ver_JJA_p2_mean = hgtERA5_ver_JJA_p2.mean(dim="time", skipna=True)

uq_dpg_ERA5_ver_JJA_p1_mean = uq_dpg_ERA5_ver_JJA_p1.mean(dim="time", skipna=True)
uq_dpg_ERA5_ver_JJA_p2_mean = uq_dpg_ERA5_ver_JJA_p2.mean(dim="time", skipna=True)

vq_dpg_ERA5_ver_JJA_p1_mean = vq_dpg_ERA5_ver_JJA_p1.mean(dim="time", skipna=True)
vq_dpg_ERA5_ver_JJA_p2_mean = vq_dpg_ERA5_ver_JJA_p2.mean(dim="time", skipna=True)
# %%
#   calculate the divergence of water vapor flux
div_uqvq_ERA5_p1 = ca.cal_divergence(
    uq_dpg_ERA5_ver_JJA_p1_mean, vq_dpg_ERA5_ver_JJA_p1_mean
)
div_uqvq_ERA5_p2 = ca.cal_divergence(
    uq_dpg_ERA5_ver_JJA_p2_mean, vq_dpg_ERA5_ver_JJA_p2_mean
)
# %%
#   calculate the difference of P1 and P2
uERA5_ver_JJA_diff_mean = uERA5_ver_JJA_p2_mean - uERA5_ver_JJA_p1_mean
vERA5_ver_JJA_diff_mean = vERA5_ver_JJA_p2_mean - vERA5_ver_JJA_p1_mean
hgtERA5_ver_JJA_diff_mean = hgtERA5_ver_JJA_p2_mean - hgtERA5_ver_JJA_p1_mean
uq_dpg_ERA5_ver_JJA_diff_mean = (
    uq_dpg_ERA5_ver_JJA_p2_mean - uq_dpg_ERA5_ver_JJA_p1_mean
)
vq_dpg_ERA5_ver_JJA_diff_mean = (
    vq_dpg_ERA5_ver_JJA_p2_mean - vq_dpg_ERA5_ver_JJA_p1_mean
)
# %%
#   calculate the check
clevel = 0.95
uERA5_mask = ca.generate_tmask(uERA5_ver_JJA_p1, uERA5_ver_JJA_p2, clevel)
vERA5_mask = ca.generate_tmask(vERA5_ver_JJA_p1, vERA5_ver_JJA_p2, clevel)
hgtERA5_mask = ca.generate_tmask(hgtERA5_ver_JJA_p1, hgtERA5_ver_JJA_p2, clevel)
uq_dpg_ERA5_mask = ca.generate_tmask(
    uq_dpg_ERA5_ver_JJA_p1, uq_dpg_ERA5_ver_JJA_p2, clevel
)
vq_dpg_ERA5_mask = ca.generate_tmask(
    vq_dpg_ERA5_ver_JJA_p1, vq_dpg_ERA5_ver_JJA_p2, clevel
)
# %%
#   calculate the wind check
wind_ERA5_mask = ca.wind_check(uERA5_mask, vERA5_mask, uERA5_mask, vERA5_mask)
vq_ERA5_mask = ca.wind_check(
    uq_dpg_ERA5_mask, vq_dpg_ERA5_mask, uq_dpg_ERA5_mask, vq_dpg_ERA5_mask
)


# %%
#   plot the different periods plots
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=(4.0, 7.0), hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=3, nrows=3, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # ??????????????????
yticks = np.arange(-30, 46, 15)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
llim_200 = 12100
hlim_200 = 12540
spacing_200 = 40

llim_500 = 5600
hlim_500 = 5920
spacing_500 = 40

llim_850 = 1320
hlim_850 = 1560
spacing_850 = 20

ski = 2
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    rect = Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1
    )
    ax.add_patch(rect)
for ax in axs[:2]:
    ax.contour(
        hgtERA5_ver_JJA.sel(level=200.0).mean(dim="time", skipna=True),
        levels=np.arange(llim_200, hlim_200 + spacing_200 / 2, spacing_200),
        lw=0.8,
        color="grey5",
        linestyle="--",
        zorder=0.9,
    )
for ax in axs[3:5]:
    ax.contour(
        hgtERA5_ver_JJA.sel(level=500.0).mean(dim="time", skipna=True),
        levels=np.arange(llim_500, hlim_500 + spacing_500 / 2, spacing_200),
        lw=0.8,
        color="grey5",
        linestyle="--",
        zorder=0.9,
    )

for ax in axs[6:8]:
    ax.contour(
        hgtERA5_ver_JJA.sel(level=850.0).mean(dim="time", skipna=True),
        levels=np.arange(llim_850, hlim_850 + spacing_850 / 2, spacing_200),
        lw=0.8,
        color="grey5",
        linestyle="--",
        zorder=0.9,
    )
# ===================================================
con = axs[0, 0].contourf(
    hgtERA5_ver_JJA_p1_mean.sel(level=200.0),
    cmap="YlOrRd",
    cmap_kw={"right": 0.77},
    extend="both",
    levels=np.arange(llim_200, hlim_200 + spacing_200 / 2, spacing_200),
    zorder=0.8,
)


m = axs[0, 0].quiver(
    uERA5_ver_JJA_p1_mean.sel(level=200.0)[::ski, ::ski],
    vERA5_ver_JJA_p1_mean.sel(level=200.0)[::ski, ::ski],
    zorder=1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=2.0,
    pivot="mid",
    color="black",
)

qk = axs[0, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=10,
    label="10 m/s",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 0].format(ltitle="1950-1966", rtitle="200hPa")
# ===========================================
con = axs[0, 1].contourf(
    hgtERA5_ver_JJA_p2_mean.sel(level=200.0),
    cmap="YlOrRd",
    cmap_kw={"right": 0.77},
    extend="both",
    levels=np.arange(llim_200, hlim_200 + spacing_200 / 2, spacing_200),
    zorder=0.8,
)


m = axs[0, 1].quiver(
    uERA5_ver_JJA_p2_mean.sel(level=200.0)[::ski, ::ski],
    vERA5_ver_JJA_p2_mean.sel(level=200.0)[::ski, ::ski],
    zorder=1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=2.0,
    pivot="mid",
    color="black",
)

qk = axs[0, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=10,
    label="10 m/s",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 1].format(ltitle="1967-2001", rtitle="200hPa")
axs[0, 1].colorbar(
    con,
    loc="r",
    ticklen=0,
    labelsize=7,
    label="gpm",
    ticklabelsize=6,
    width=0.14,
    pad=0.8,
)
# ===========================================
#   the difference of 200hPa hgt and uv
con = axs[0, 2].contourf(
    hgtERA5_ver_JJA_diff_mean.sel(level=200.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    extend="both",
    zorder=0.8,
    levels=np.arange(-60, 61, 6),
)
axs[0, 2].contour(
    hgtERA5_mask.sel(level=200.0),
    color="grey7",
    vmin=1.0,
    vmax=1.0,
    lw=1.0,
    linestyle="--",
)

axs[0, 2].quiver(
    uERA5_ver_JJA_diff_mean.sel(level=200.0)[::ski, ::ski],
    vERA5_ver_JJA_diff_mean.sel(level=200.0)[::ski, ::ski],
    zorder=1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=1.0,
    pivot="mid",
    color="grey6",
)
m = axs[0, 2].quiver(
    (uERA5_ver_JJA_diff_mean.where(wind_ERA5_mask > 0.0)).sel(level=200.0)[
        ::ski, ::ski
    ],
    (vERA5_ver_JJA_diff_mean.where(wind_ERA5_mask > 0.0)).sel(level=200.0)[
        ::ski, ::ski
    ],
    zorder=1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=1.0,
    pivot="mid",
    color="black",
)

qk = axs[0, 2].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=3,
    label="3 m/s",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)

axs[0, 2].format(ltitle="diff P2-P1", rtitle="200hPa")
cb = axs[0, 2].colorbar(
    con,
    loc="r",
    ticklen=0,
    labelsize=7,
    label="gpm",
    ticklabelsize=6,
    width=0.14,
    pad=0.8,
)
cb.set_ticks(np.arange(-60, 61, 12))


# ===========================================
#   500 hPa
con = axs[1, 0].contourf(
    hgtERA5_ver_JJA_p1_mean.sel(level=500.0),
    cmap="YlOrRd",
    cmap_kw={"right": 0.77},
    extend="both",
    levels=np.arange(llim_500, hlim_500 + spacing_500 / 2, spacing_500),
    zorder=0.8,
)


m = axs[1, 0].quiver(
    uERA5_ver_JJA_p1_mean.sel(level=500.0)[::ski, ::ski],
    vERA5_ver_JJA_p1_mean.sel(level=500.0)[::ski, ::ski],
    zorder=1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=1.3,
    pivot="mid",
    color="black",
)

qk = axs[1, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=8,
    label="8 m/s",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1, 0].format(ltitle="1950-1966", rtitle="500hPa")
# ===========================================
con = axs[1, 1].contourf(
    hgtERA5_ver_JJA_p2_mean.sel(level=500.0),
    cmap="YlOrRd",
    cmap_kw={"right": 0.77},
    extend="both",
    levels=np.arange(llim_500, hlim_500 + spacing_500 / 2, spacing_500),
    zorder=0.8,
)


m = axs[1, 1].quiver(
    uERA5_ver_JJA_p2_mean.sel(level=500.0)[::ski, ::ski],
    vERA5_ver_JJA_p2_mean.sel(level=500.0)[::ski, ::ski],
    zorder=1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=1.3,
    pivot="mid",
    color="black",
)

qk = axs[1, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=8,
    label="8 m/s",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1, 1].format(ltitle="1967-2001", rtitle="500hPa")
axs[1, 1].colorbar(
    con,
    loc="r",
    ticklen=0,
    labelsize=7,
    label="gpm",
    ticklabelsize=6,
    width=0.14,
    pad=0.8,
)
# ===========================================
#   the difference of 500hPa hgt and uv
con = axs[1, 2].contourf(
    hgtERA5_ver_JJA_diff_mean.sel(level=500.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    extend="both",
    zorder=0.8,
    levels=np.arange(-30, 31, 3),
)
axs[1, 2].contour(
    hgtERA5_mask.sel(level=500.0),
    color="grey7",
    vmin=1.0,
    vmax=1.0,
    lw=1.0,
    linestyle="--",
)

axs[1, 2].quiver(
    uERA5_ver_JJA_diff_mean.sel(level=500.0)[::ski, ::ski],
    vERA5_ver_JJA_diff_mean.sel(level=500.0)[::ski, ::ski],
    zorder=1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.7,
    pivot="mid",
    color="grey6",
)
m = axs[1, 2].quiver(
    (uERA5_ver_JJA_diff_mean.where(wind_ERA5_mask > 0.0)).sel(level=500.0)[
        ::ski, ::ski
    ],
    (vERA5_ver_JJA_diff_mean.where(wind_ERA5_mask > 0.0)).sel(level=500.0)[
        ::ski, ::ski
    ],
    zorder=1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.7,
    pivot="mid",
    color="black",
)

qk = axs[1, 2].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=3,
    label="3 m/s",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)

axs[1, 2].format(ltitle="diff P2-P1", rtitle="500hPa")
cb = axs[1, 2].colorbar(
    con,
    loc="r",
    ticklen=0,
    labelsize=7,
    label="gpm",
    ticklabelsize=6,
    width=0.14,
    pad=0.8,
)
cb.set_ticks(np.arange(-30, 31, 6))


# ===========================================
#   850 hPa
con = axs[2, 0].contourf(
    hgtERA5_ver_JJA_p1_mean.sel(level=850.0),
    cmap="YlOrRd",
    cmap_kw={"right": 0.77},
    extend="both",
    levels=np.arange(llim_850, hlim_850 + spacing_850 / 2, spacing_850),
    zorder=0.8,
)


m = axs[2, 0].quiver(
    uERA5_ver_JJA_p1_mean.sel(level=850.0)[::ski, ::ski],
    vERA5_ver_JJA_p1_mean.sel(level=850.0)[::ski, ::ski],
    zorder=1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=1.5,
    pivot="mid",
    color="black",
)

qk = axs[2, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=5,
    label="5 m/s",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[2, 0].format(ltitle="1950-1966", rtitle="850hPa")
# ===========================================
con = axs[2, 1].contourf(
    hgtERA5_ver_JJA_p2_mean.sel(level=850.0),
    cmap="YlOrRd",
    cmap_kw={"right": 0.77},
    extend="both",
    levels=np.arange(llim_850, hlim_850 + spacing_850 / 2, spacing_850),
    zorder=0.8,
)


m = axs[2, 1].quiver(
    uERA5_ver_JJA_p2_mean.sel(level=850.0)[::ski, ::ski],
    vERA5_ver_JJA_p2_mean.sel(level=850.0)[::ski, ::ski],
    zorder=1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=1.5,
    pivot="mid",
    color="black",
)

qk = axs[2, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=5,
    label="5 m/s",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[2, 1].format(ltitle="1967-2001", rtitle="850hPa")
axs[2, 1].colorbar(
    con,
    loc="r",
    ticklen=0,
    labelsize=7,
    label="gpm",
    ticklabelsize=6,
    width=0.14,
    pad=0.8,
)
# ========================================

#   the difference of 850hPa hgt and uv
con = axs[2, 2].contourf(
    hgtERA5_ver_JJA_diff_mean.sel(level=850.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    extend="both",
    zorder=0.8,
    levels=np.arange(-10, 11, 2),
)
axs[2, 2].contour(
    hgtERA5_mask.sel(level=850.0),
    color="grey7",
    vmin=1.0,
    vmax=1.0,
    lw=1.0,
    linestyle="--",
)

axs[2, 2].quiver(
    uERA5_ver_JJA_diff_mean.sel(level=850.0)[::ski, ::ski],
    vERA5_ver_JJA_diff_mean.sel(level=850.0)[::ski, ::ski],
    zorder=1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.6,
    pivot="mid",
    color="grey6",
)
m = axs[2, 2].quiver(
    (uERA5_ver_JJA_diff_mean.where(wind_ERA5_mask > 0.0)).sel(level=850.0)[
        ::ski, ::ski
    ],
    (vERA5_ver_JJA_diff_mean.where(wind_ERA5_mask > 0.0)).sel(level=850.0)[
        ::ski, ::ski
    ],
    zorder=1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.6,
    pivot="mid",
    color="black",
)

qk = axs[2, 2].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=3,
    label="3 m/s",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)

axs[2, 2].format(ltitle="diff P2-P1", rtitle="850hPa")
cb = axs[2, 2].colorbar(
    con,
    loc="r",
    ticklen=0,
    labelsize=7,
    label="gpm",
    ticklabelsize=6,
    width=0.14,
    pad=0.8,
)
cb.set_ticks(np.arange(-10, 11, 2))
fig.format(abc="(a)", abcloc="l", suptitle="hgt & UV")
# %%

# %%
