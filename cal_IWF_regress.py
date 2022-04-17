'''
Author: ChenHJ
Date: 2022-04-15 19:34:29
LastEditors: ChenHJ
LastEditTime: 2022-04-17 21:06:33
FilePath: /chenhj/0302code/cal_IWF_regress.py
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

import metpy.calc as mpcalc
import metpy.xarray as mpxr
import metpy.constants as constants
import geocat.comp
from windspharm.xarray import VectorWind

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


def patches(ax, x0, y0, width, height, proj):
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x0, y0), width, height, fc="none", ec="grey7", linewidth=0.8, zorder=1.1, transform=proj, linestyle="--",
    )
    ax.add_patch(rect)
# %%
#   read the ERA5 reanalysis data and calculate the SAM and IWF index
fhgtERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/hgt_mon_r144x72_195001-201412.nc"
)
hgtERA5 = fhgtERA5["z"]


fuERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/uwind_mon_r144x72_195001-201412.nc"
)
uERA5 = fuERA5["u"]

fvERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/vwind_mon_r144x72_195001-201412.nc"
)
vERA5 = fvERA5["v"]

fspERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/sp_mon_r144x72_195001-201412.nc"
)
spERA5 = fspERA5["sp"]

fqERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/q_mon_r144x72_195001-201412.nc")
qERA5 = fqERA5["q"]

# %%
#   read the precipitation data
fpreCRU = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/cru_ts4.01_r144x72_195001-201412.nc"
)
preCRU = fpreCRU["pre"]

# GPCP data just have 1979-2014 year
fpreGPCP = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/GPCP_r144x72_197901-201412.nc"
)
preGPCP = fpreGPCP["precip"]

# %%
hgtERA5_ver_JJA = ca.p_time(hgtERA5, 6, 8, True).loc[:, 100.0:, :, :]
uERA5_ver_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, :, :]
vERA5_ver_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, :, :]
qERA5_ver_JJA = ca.p_time(qERA5, 6, 9, True).loc[:, 100.0:, :, :]
preCRU_JJA = ca.p_time(preCRU, 6, 8, True) / 30.67
spERA5_JJA = ca.p_time(spERA5, 6, 8, True)

hgtERA5_ver_JJA = ca.detrend_dim(hgtERA5_ver_JJA, "time", deg=1, demean=False)
uERA5_ver_JJA = ca.detrend_dim(uERA5_ver_JJA, "time", deg=1, demean=False)
vERA5_ver_JJA = ca.detrend_dim(vERA5_ver_JJA, "time", deg=1, demean=False)
qERA5_ver_JJA = ca.detrend_dim(qERA5_ver_JJA, "time", deg=1, demean=False)
preCRU_JJA = ca.detrend_dim(preCRU_JJA, "time", deg=1, demean=False)
spERA5_JJA = ca.detrend_dim(spERA5_JJA, "time", deg=1, demean=False)

preGPCP_JJA = ca.p_time(preGPCP, 6, 8, True)
preGPCP_JJA = ca.detrend_dim(preGPCP_JJA, "time", deg=1, demean=False)

# %%

ERA5_IWF_index = ca.IWF(uERA5_ver_JJA, vERA5_ver_JJA)
ERA5_IWF_index = ca.detrend_dim(ERA5_IWF_index, "time", deg=1, demean=False)
ERA5_SAM_index = ca.SAM(vERA5_ver_JJA)
ERA5_SAM_index = ca.detrend_dim(ERA5_SAM_index, "time", deg=1, demean=False)

# %%
# read the SAM and IWF index of historical run and ssp585
fhis_SAM_index = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_SAM_index_1950-2014.nc")
his_SAM_index = fhis_SAM_index["SAM"]

fssp585_SAM_index = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_SAM_index_2015-2099.nc")
ssp585_SAM_index = fssp585_SAM_index["SAM"]

fhis_IWF_index = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_IWF_index_1950-2014.nc")
his_IWF_index = fhis_IWF_index["IWF"]

fssp585_IWF_index = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_IWF_index_2015-2099.nc")
ssp585_IWF_index = fssp585_IWF_index["IWF"]
# %%
#   pick up the good models in historical run and ssp585 run
gmodels = ["CNRM-CM6-1", "MIROC-ES2L", "NorESM2-LM", "HadGEM3-GC31-LL", "MRI-ESM2-0", "ACCESS-CM2", "MIROC6", "EC-Earth3", "CESM2-WACCM", "CAMS-CSM1-0"]

his_SAM_index_gmodels = his_SAM_index.sel(models=gmodels)
his_IWF_index_gmodels = his_IWF_index.sel(models=gmodels)

ssp585_SAM_index_gmodels = ssp585_SAM_index.sel(models=gmodels)
ssp585_IWF_index_gmodels = ssp585_IWF_index.sel(models=gmodels)

# %%
#   calculate the IWF regress on SAM in historical run and ssp585 run
(
    IWF_his_gmodels_SAM_slope,
    IWF_his_gmodels_SAM_intercept,
    IWF_his_gmodels_SAM_rvalue,
    IWF_his_gmodels_SAM_pvalue,
    IWF_his_gmodels_SAM_hypothesis,
) = ca.dim_linregress(his_IWF_index_gmodels, his_SAM_index_gmodels)

(
    IWF_ssp585_gmodels_SAM_slope,
    IWF_ssp585_gmodels_SAM_intercept,
    IWF_ssp585_gmodels_SAM_rvalue,
    IWF_ssp585_gmodels_SAM_pvalue,
    IWF_ssp585_gmodels_SAM_hypothesis,
) = ca.dim_linregress(ssp585_IWF_index_gmodels, ssp585_SAM_index_gmodels)

# %%
#   calculate the ensemble mean
IWF_his_gmodels_SAM_slope_ens = IWF_his_gmodels_SAM_slope.mean(dim="models", skipna=True)
IWF_ssp585_gmodels_SAM_slope_ens = IWF_ssp585_gmodels_SAM_slope.mean(dim="models", skipna=True)

print(IWF_his_gmodels_SAM_slope_ens, IWF_ssp585_gmodels_SAM_slope_ens)
print(ca.MME_reg_mask(IWF_his_gmodels_SAM_slope_ens, IWF_his_gmodels_SAM_slope.std(dim="models", skipna=True), len(IWF_his_gmodels_SAM_slope.coords["models"]), True))
print(ca.MME_reg_mask(IWF_ssp585_gmodels_SAM_slope_ens, IWF_ssp585_gmodels_SAM_slope.std(dim="models", skipna=True), len(IWF_ssp585_gmodels_SAM_slope.coords["models"]), True))

# %%
(
    IWF_his_SAM_slope,
    IWF_his_SAM_intercept,
    IWF_his_SAM_rvalue,
    IWF_his_SAM_pvalue,
    IWF_his_SAM_hypothesis,
) = ca.dim_linregress(his_IWF_index, his_SAM_index)
# %%
#   read the historical run data
fvhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/va_historical_r144x72_195001-201412.nc")
vhis_ver_JJA = fvhis_ver_JJA["va"]
fuhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/ua_historical_r144x72_195001-201412.nc")
uhis_ver_JJA = fuhis_ver_JJA["ua"]

# %%
#   calculate the vertical shear in ERA5/historical
vERA5_ver_JJA_shear = vERA5_ver_JJA.sel(level=850) - vERA5_ver_JJA.sel(level=200)

vhis_ver_JJA_shear = vhis_ver_JJA.sel(level=850) - vhis_ver_JJA.sel(level=200)
# %%
#   read the precipitation data
fpreCRU = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/cru_ts4.01_r144x72_195001-201412.nc"
)
preCRU = fpreCRU["pre"]
preCRU = fpreCRU["pre"]
preCRU_JJA = ca.p_time(preCRU, 6, 8, True)/30.67
preCRU_JJA = ca.detrend_dim(preCRU_JJA, "time", deg=1, demean=False)
preCRU_JJA.attrs["units"] = "mm/day"

# GPCP data just have 1979-2014 year
fpreGPCP = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/GPCP_r144x72_197901-201412.nc"
)
preGPCP = fpreGPCP["precip"]
preGPCP_JJA = ca.p_time(preGPCP, 6, 8, True)
preGPCP_JJA = ca.detrend_dim(preGPCP_JJA, "time", deg=1, demean=False)

fprehis = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/pr_historical_r144x72_195001-201412.nc")
prehis_JJA = fprehis["pr"]
prehis_JJA.attrs["units"] = "mm/day"
prehis_JJA.attrs["standard_name"] = "precipitation"

fpressp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/pr_ssp585_r144x72_201501-209912.nc")
pressp585_JJA = fpressp585["pr"]
pressp585_JJA.attrs["units"] = "mm/day"
pressp585_JJA.attrs["standard_name"] = "precipitation"



# %%
#   pick up the India precipitation in different data
lat = preCRU_JJA.coords["lat"]
lon = preCRU_JJA.coords["lon"]
lat_India_range = lat[(lat >= 8.0) & (lat <= 28.0)]
lon_India_range = lon[(lon >= 70.0) & (lon <= 86.0)]
preCRU_India_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)

lat = preGPCP_JJA.coords["lat"]
lon = preGPCP_JJA.coords["lon"]
lat_India_range = lat[(lat >= 8.0) & (lat <= 28.0)]
lon_India_range = lon[(lon >= 70.0) & (lon <= 86.0)]
preGPCP_India_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)

lat = prehis_JJA.coords["lat"]
lon = prehis_JJA.coords["lon"]
lat_India_range = lat[(lat >= 8.0) & (lat <= 28.0)]
lon_India_range = lon[(lon >= 70.0) & (lon <= 86.0)]
prehis_India_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)

lat = pressp585_JJA.coords["lat"]
lon = pressp585_JJA.coords["lon"]
lat_India_range = lat[(lat >= 8.0) & (lat <= 28.0)]
lon_India_range = lon[(lon >= 70.0) & (lon <= 86.0)]
pressp585_India_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)

del(lat, lon)
# %%
#   calculate the correlation coefficients between IWF and IndR
IndRGPCP_ERA5_IWF_regress = stats.linregress(preGPCP_India_JJA, ERA5_IWF_index.sel(time=ERA5_IWF_index.time.dt.year>=1979))
IndR_ERA5_IWF_regress = stats.linregress(preCRU_India_JJA, ERA5_IWF_index)
IndR_his_IWF_regress = ca.dim_linregress(prehis_India_JJA, his_IWF_index)
IndR_ssp585_IWF_regress = ca.dim_linregress(pressp585_India_JJA, ssp585_IWF_index)
# %%
#   calculate the precipitation data regression onto IWF
preCRU_JJA.coords["time"] = ERA5_IWF_index.coords["time"]
(
    IWF_ERA5_preCRU_slope,
    IWF_ERA5_preCRU_intercept,
    IWF_ERA5_preCRU_rvalue,
    IWF_ERA5_preCRU_pvalue,
    IWF_ERA5_preCRU_hypothesis,
) = ca.dim_linregress(ERA5_IWF_index, preCRU_JJA)

(
    IWF_ERA5_preGPCP_slope,
    IWF_ERA5_preGPCP_intercept,
    IWF_ERA5_preGPCP_rvalue,
    IWF_ERA5_preGPCP_pvalue,
    IWF_ERA5_preGPCP_hypothesis,
) = ca.dim_linregress(ERA5_IWF_index.sel(time=ERA5_IWF_index.time.dt.year>=1979), preGPCP_JJA)

(
    IWF_his_pre_slope,
    IWF_his_pre_intercept,
    IWF_his_pre_rvalue,
    IWF_his_pre_pvalue,
    IWF_his_pre_hypothesis,
) = ca.dim_linregress(his_IWF_index, prehis_JJA)

(
    IWF_ssp585_pre_slope,
    IWF_ssp585_pre_intercept,
    IWF_ssp585_pre_rvalue,
    IWF_ssp585_pre_pvalue,
    IWF_ssp585_pre_hypothesis,
) = ca.dim_linregress(ssp585_IWF_index, pressp585_JJA)


# %%
models = uhis_ver_JJA.coords["models"]
# %%
#   plot the precipitation regression rvalue on IWF in ERA5 and historical run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=4, nrows=7, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.arange(50, 151, 10)  # 设置纬度刻度
yticks = np.arange(10, 51, 10)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], 5, 55]
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ======================================
for ax in axs:
    #   Indian area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
    #   IWF area
    x0 = 90
    y0 = 5.0
    width = 50.0
    height = 27.5
    patches(ax, x0 - cl, y0, width, height, proj)
# ======================================
con = axs[0].contourf(
    IWF_ERA5_preCRU_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    IWF_ERA5_preCRU_rvalue, axs[0], n, np.where(IWF_ERA5_preCRU_pvalue[::n, ::n] <= 0.05), "bright purple", 3.0,
)
axs[0].format(
    rtitle="1950-2014", ltitle="ERA5 & CRU",
)
# ======================================
con = axs[1].contourf(
    IWF_ERA5_preGPCP_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    IWF_ERA5_preGPCP_rvalue, axs[1], n, np.where(IWF_ERA5_preGPCP_pvalue[::n, ::n] <= 0.05), "bright purple", 3.0,
)
axs[1].format(
    rtitle="1950-2014", ltitle="ERA5 & GPCP",
)
# ======================================
for num_mod, mod in enumerate(models):
    con = axs[num_mod+2].contourf(
        IWF_historical_pre_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-1.0, 1.1, 0.1),
        zorder=0.8,
    )
    sepl.plt_sig(
        IWF_historical_pre_rvalue.sel(models=mod), axs[num_mod+2], n, np.where(IWF_historical_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod+2].format(
        rtitle="1950-2014", ltitle="historical",
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="precip reg IWF")

# %%
