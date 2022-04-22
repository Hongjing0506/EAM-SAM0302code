'''
Author: ChenHJ
Date: 2022-04-15 19:34:29
LastEditors: ChenHJ
LastEditTime: 2022-04-19 00:43:14
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
from statsmodels.distributions.empirical_distribution import ECDF
import dask


def patches(ax, x0, y0, width, height, proj):
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x0, y0), width, height, fc="none", ec="grey7", linewidth=0.8, zorder=1.1, transform=proj, linestyle="--",
    )
    ax.add_patch(rect)
# %%
#   read the ERA5 reanalysis data and calculate the SAM and IWF index
fhgtERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/hgt_mon_r144x72_195001-201412.nc", chunks={'lat':10,'lon':10}
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
hgtERA5_ver_JJA = hgtERA5_ver_JJA - hgtERA5_ver_JJA.mean(dim="lon", skipna=True)
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
) = ca.dim_linregress(ERA5_IWF_index.sel(time=ERA5_IWF_index.time.dt.year>=1979), preCRU_JJA.sel(time=preCRU_JJA.time.dt.year>=1979))

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
) = ca.dim_linregress(his_IWF_index.sel(time=his_IWF_index.time.dt.year>=1979), prehis_JJA.sel(time=prehis_JJA.time.dt.year>=1979))

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
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
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
    rtitle="1979-2014", ltitle="ERA5 & GPCP",
)
# ======================================
for num_mod, mod in enumerate(models):
    con = axs[num_mod+2].contourf(
        IWF_his_pre_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-1.0, 1.1, 0.1),
        zorder=0.8,
    )
    sepl.plt_sig(
        IWF_his_pre_rvalue.sel(models=mod), axs[num_mod+2], n, np.where(IWF_his_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod+2].format(
        rtitle="1950-2014", ltitle="{}".format(mod.data),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="precip reg IWF")

# %%
#   plot the precipitation regression avalue on IWF in ERA5 and historical run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=4, nrows=7, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
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
    IWF_ERA5_preCRU_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-4.5e6, 4.6e6, 2.0e5),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IWF_ERA5_preCRU_slope, axs[0], n, np.where(IWF_ERA5_preCRU_pvalue[::n, ::n] <= 0.05), "bright purple", 3.0,
)
axs[0].format(
    rtitle="1979-2014", ltitle="ERA5 & CRU",
)
# ======================================
con = axs[1].contourf(
    IWF_ERA5_preGPCP_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-4.5e6, 4.6e6, 2.0e5),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IWF_ERA5_preGPCP_slope, axs[1], n, np.where(IWF_ERA5_preGPCP_pvalue[::n, ::n] <= 0.05), "bright purple", 3.0,
)
axs[1].format(
    rtitle="1979-2014", ltitle="ERA5 & GPCP",
)
# ======================================
for num_mod, mod in enumerate(models):
    con = axs[num_mod+2].contourf(
        IWF_his_pre_slope.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-4.5e6, 4.6e6, 2.0e5),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IWF_his_pre_slope.sel(models=mod), axs[num_mod+2], n, np.where(IWF_his_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod+2].format(
        rtitle="1979-2014", ltitle="{}".format(mod.data),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="precip reg IWF")
# %%
#   plot the precipitation regression rvalue on IWF in ssp585 run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 29), (7, 4))
plot_array[6,0] = 0
plot_array[6,3] = 0
axs = fig.subplots(plot_array, proj=proj)
#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
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
for num_mod, mod in enumerate(models):
    con = axs[num_mod].contourf(
        IWF_his_pre_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-1.0, 1.1, 0.1),
        zorder=0.8,
    )
    sepl.plt_sig(
        IWF_his_pre_rvalue.sel(models=mod), axs[num_mod], n, np.where(IWF_his_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod].format(
        rtitle="2015-2099", ltitle="{}".format(mod.data),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="precip reg IWF")

# %%
#   plot the precipitation regression avalue on IWF in ssp585 run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 29), (7, 4))
plot_array[6,0] = 0
plot_array[6,3] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
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
for num_mod, mod in enumerate(models):
    con = axs[num_mod].contourf(
        IWF_his_pre_slope.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-4.5e6, 4.6e6, 2.0e5),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IWF_his_pre_slope.sel(models=mod), axs[num_mod], n, np.where(IWF_his_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod].format(
        rtitle="2015-2099", ltitle="{}".format(mod.data),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="precip reg IWF")
# %%
#   read the hgt&u&v data of historicla and ssp585
fhgthis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/zg_historical_r144x72_195001-201412.nc")
hgthis_ver_JJA = fhgthis_ver_JJA["zg"]
hgthis_ver_JJA = hgthis_ver_JJA - hgthis_ver_JJA.mean(dim="lon", skipna=True)

fuhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/ua_historical_r144x72_195001-201412.nc")
uhis_ver_JJA = fuhis_ver_JJA["ua"]

fvhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/va_historical_r144x72_195001-201412.nc")
vhis_ver_JJA = fvhis_ver_JJA["va"]

fhgtssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/zg_ssp585_r144x72_201501-209912.nc")
hgtssp585_ver_JJA = fhgtssp585_ver_JJA["zg"]
hgtssp585_ver_JJA = hgtssp585_ver_JJA - hgtssp585_ver_JJA.mean(dim="lon", skipna=True)

fussp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ua_ssp585_r144x72_201501-209912.nc")
ussp585_ver_JJA = fussp585_ver_JJA["ua"]

fvssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/va_ssp585_r144x72_201501-209912.nc")
vssp585_ver_JJA = fvssp585_ver_JJA["va"]

# %%
#   calculate the hgt&u&v regress onto IWF in ERA5, historical, ssp585
(
    IWF_ERA5_hgt_slope,
    IWF_ERA5_hgt_intercept,
    IWF_ERA5_hgt_rvalue,
    IWF_ERA5_hgt_pvalue,
    IWF_ERA5_hgt_hypothesis,
) = ca.dim_linregress(ERA5_IWF_index.sel(time=ERA5_IWF_index.time.dt.year>=1979), hgtERA5_ver_JJA.sel(time=hgtERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IWF_ERA5_u_slope,
    IWF_ERA5_u_intercept,
    IWF_ERA5_u_rvalue,
    IWF_ERA5_u_pvalue,
    IWF_ERA5_u_hypothesis,
) = ca.dim_linregress(ERA5_IWF_index.sel(time=ERA5_IWF_index.time.dt.year>=1979), uERA5_ver_JJA.sel(time=uERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IWF_ERA5_v_slope,
    IWF_ERA5_v_intercept,
    IWF_ERA5_v_rvalue,
    IWF_ERA5_v_pvalue,
    IWF_ERA5_v_hypothesis,
) = ca.dim_linregress(ERA5_IWF_index.sel(time=ERA5_IWF_index.time.dt.year>=1979), vERA5_ver_JJA.sel(time=vERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

IWF_ERA5_wind_mask = ca.wind_check(
    xr.where(IWF_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IWF_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IWF_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IWF_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
)

(
    IWF_his_hgt_slope,
    IWF_his_hgt_intercept,
    IWF_his_hgt_rvalue,
    IWF_his_hgt_pvalue,
    IWF_his_hgt_hypothesis,
) = ca.dim_linregress(his_IWF_index.sel(time=his_IWF_index.time.dt.year>=1979), hgthis_ver_JJA.sel(time=hgthis_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IWF_his_u_slope,
    IWF_his_u_intercept,
    IWF_his_u_rvalue,
    IWF_his_u_pvalue,
    IWF_his_u_hypothesis,
) = ca.dim_linregress(his_IWF_index.sel(time=his_IWF_index.time.dt.year>=1979), uhis_ver_JJA.sel(time=uhis_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IWF_his_v_slope,
    IWF_his_v_intercept,
    IWF_his_v_rvalue,
    IWF_his_v_pvalue,
    IWF_his_v_hypothesis,
) = ca.dim_linregress(his_IWF_index.sel(time=his_IWF_index.time.dt.year>=1979), vhis_ver_JJA.sel(time=vhis_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

IWF_his_wind_mask = ca.wind_check(
    xr.where(IWF_his_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IWF_his_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IWF_his_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IWF_his_v_pvalue <= 0.05, 1.0, 0.0),
)

(
    IWF_ssp585_hgt_slope,
    IWF_ssp585_hgt_intercept,
    IWF_ssp585_hgt_rvalue,
    IWF_ssp585_hgt_pvalue,
    IWF_ssp585_hgt_hypothesis,
) = ca.dim_linregress(ssp585_IWF_index.sel(time=ssp585_IWF_index.time.dt.year>=1979), hgtssp585_ver_JJA.sel(time=hgtssp585_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IWF_ssp585_u_slope,
    IWF_ssp585_u_intercept,
    IWF_ssp585_u_rvalue,
    IWF_ssp585_u_pvalue,
    IWF_ssp585_u_hypothesis,
) = ca.dim_linregress(ssp585_IWF_index.sel(time=ssp585_IWF_index.time.dt.year>=1979), ussp585_ver_JJA.sel(time=ussp585_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IWF_ssp585_v_slope,
    IWF_ssp585_v_intercept,
    IWF_ssp585_v_rvalue,
    IWF_ssp585_v_pvalue,
    IWF_ssp585_v_hypothesis,
) = ca.dim_linregress(ssp585_IWF_index.sel(time=ssp585_IWF_index.time.dt.year>=1979), vssp585_ver_JJA.sel(time=vssp585_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

IWF_ssp585_wind_mask = ca.wind_check(
    xr.where(IWF_ssp585_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IWF_ssp585_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IWF_ssp585_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IWF_ssp585_v_pvalue <= 0.05, 1.0, 0.0),
)
# %%
#   create the regression dataset and output

models = IWF_his_hgt_rvalue.coords["models"]
lon = IWF_his_hgt_rvalue.coords["lon"]
lat = IWF_his_hgt_rvalue.coords["lat"]
level = IWF_his_hgt_rvalue.coords["level"]

IWF_his_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IWF_his_hgt_slope.data),
        intercept=(["models", "level", "lat", "lon"], IWF_his_hgt_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IWF_his_hgt_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IWF_his_hgt_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IWF_his_hgt_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of multi-models in historical run regress onto his_IWF_index"),
)

IWF_his_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IWF_his_u_slope.data),
        intercept=(["models", "level", "lat", "lon"], IWF_his_u_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IWF_his_u_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IWF_his_u_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IWF_his_u_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of multi-models in historical run regress onto his_IWF_index"),
)

IWF_his_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IWF_his_v_slope.data),
        intercept=(["models", "level", "lat", "lon"], IWF_his_v_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IWF_his_v_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IWF_his_v_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IWF_his_v_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of multi-models in historical run regress onto his_IWF_index"),
)
IWF_ssp585_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IWF_ssp585_hgt_slope.data),
        intercept=(["models", "level", "lat", "lon"], IWF_ssp585_hgt_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IWF_ssp585_hgt_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IWF_ssp585_hgt_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IWF_ssp585_hgt_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of multi-models in ssp585 run regress onto ssp585_IWF_index"),
)

IWF_ssp585_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IWF_ssp585_u_slope.data),
        intercept=(["models", "level", "lat", "lon"], IWF_ssp585_u_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IWF_ssp585_u_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IWF_ssp585_u_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IWF_ssp585_u_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of multi-models in ssp585 run regress onto ssp585_IWF_index"),
)

IWF_ssp585_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IWF_ssp585_v_slope.data),
        intercept=(["models", "level", "lat", "lon"], IWF_ssp585_v_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IWF_ssp585_v_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IWF_ssp585_v_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IWF_ssp585_v_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of multi-models in ssp585 run regress onto ssp585_IWF_index"),
)

IWF_his_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IWF_his_hgt_regress.nc")
IWF_his_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IWF_his_u_regress.nc")
IWF_his_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IWF_his_v_regress.nc")

IWF_ssp585_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/IWF_ssp585_hgt_regress.nc")
IWF_ssp585_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/IWF_ssp585_u_regress.nc")
IWF_ssp585_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/IWF_ssp585_v_regress.nc")


# %%
#   plot the rvalue of hgt&u&v regress onto IWF in ERA5 and historical
for lev in [200.0, 500.0, 850.0]:
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 29), (7, 4))
    plot_array[6,3] = 0
    axs = fig.subplots(plot_array, proj=proj)

    #   set the geo_ticks and map projection to the plots
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 1
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
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
        IWF_ERA5_hgt_rvalue.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-1.0e0, 1.1e0, 2.0e-1),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IWF_ERA5_hgt_rvalue.sel(level=lev), axs[0], n, np.where(IWF_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[0].quiver(
        IWF_ERA5_u_rvalue.sel(level=lev)[::ski, ::ski],
        IWF_ERA5_v_rvalue.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=0.17,
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IWF_ERA5_u_rvalue.sel(level=lev).where(IWF_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IWF_ERA5_v_rvalue.sel(level=lev).where(IWF_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=0.17,
        pivot="mid",
        color="black",
    )

    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="ERA5",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+1].contourf(
            IWF_his_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(-1.0e0, 1.1e0, 2.0e-1),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IWF_his_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod+1], n, np.where(IWF_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+1].quiver(
            IWF_his_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IWF_his_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=0.17,
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+1].quiver(
            IWF_his_u_rvalue.sel(models=mod,level=lev).where(IWF_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IWF_his_v_rvalue.sel(models=mod,level=lev).where(IWF_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=0.17,
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+1].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="1979-2014", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IWF".format(lev))
# %%
#   plot the rvalue of hgt&u&v regress onto IWF in ssp585
for lev in [200.0, 500.0, 850.0]:
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 29), (7, 4))
    plot_array[6,0] = 0
    plot_array[6,3] = 0
    axs = fig.subplots(plot_array, proj=proj)

    #   set the geo_ticks and map projection to the plots
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 1
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
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
    for num_mod, mod in enumerate(models):
        con = axs[num_mod].contourf(
            IWF_ssp585_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(-1.0e0, 1.1e0, 2.0e-1),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IWF_ssp585_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod], n, np.where(IWF_ssp585_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod].quiver(
            IWF_ssp585_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IWF_ssp585_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=0.17,
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod].quiver(
            IWF_ssp585_u_rvalue.sel(models=mod,level=lev).where(IWF_ssp585_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IWF_ssp585_v_rvalue.sel(models=mod,level=lev).where(IWF_ssp585_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=0.17,
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod].format(
            rtitle="2015-2099", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IWF".format(lev))
# %%
#   generate the historical hgt and wind mask
IWF_his_hgt_slope_ens = IWF_his_hgt_slope.mean(dim="models", skipna=True)
IWF_his_hgt_slope_ens_mask = ca.MME_reg_mask(IWF_his_hgt_slope_ens, IWF_his_hgt_slope.std(dim="models", skipna=True), len(models), True)

IWF_his_u_slope_ens = IWF_his_u_slope.mean(dim="models", skipna=True)
IWF_his_u_slope_ens_mask = ca.MME_reg_mask(IWF_his_u_slope_ens, IWF_his_u_slope.std(dim="models", skipna=True), len(models), True)

IWF_his_v_slope_ens = IWF_his_v_slope.mean(dim="models", skipna=True)
IWF_his_v_slope_ens_mask = ca.MME_reg_mask(IWF_his_v_slope_ens, IWF_his_v_slope.std(dim="models", skipna=True), len(models), True)

IWF_his_wind_slope_ens_mask = ca.wind_check(
    xr.where(IWF_his_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IWF_his_v_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IWF_his_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IWF_his_v_slope_ens_mask > 0.0, 1.0, 0.0),
)

# %%
#   plot the avalue of hgt&u&v regress onto IWF in ERA5 and historical
startlevel = [-3.0e7, -2.0e7, -1.5e7]
endlevel = [3.0e7, 2.0e7, 1.5e7]
spacinglevel = [3.0e6, 2.0e6, 1.0e6]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 29), (7, 4))
    axs = fig.subplots(plot_array, proj=proj)

    #   set the geo_ticks and map projection to the plots
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 1
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
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
        IWF_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev]/2, spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IWF_ERA5_hgt_slope.sel(level=lev), axs[0], n, np.where(IWF_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[0].quiver(
        IWF_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
        IWF_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=700000.0,
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IWF_ERA5_u_slope.sel(level=lev).where(IWF_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IWF_ERA5_v_slope.sel(level=lev).where(IWF_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=700000.0,
        pivot="mid",
        color="black",
    )

    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=5e6, label="5e6", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="ERA5",
    )
    # ======================================
    con = axs[1].contourf(
        IWF_his_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev]/2, spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IWF_his_hgt_slope_ens.sel(level=lev), axs[1], n, np.where(IWF_his_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[1].quiver(
        IWF_his_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IWF_his_v_slope_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=700000.0,
        pivot="mid",
        color="grey6",
    )

    m = axs[1].quiver(
        IWF_his_u_slope_ens.sel(level=lev).where(IWF_his_wind_slope_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IWF_his_v_slope_ens.sel(level=lev).where(IWF_his_wind_slope_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=700000.0,
        pivot="mid",
        color="black",
    )

    qk = axs[1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=5e6, label="5e6", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+2].contourf(
            IWF_his_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev]/2, spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IWF_his_hgt_slope.sel(models=mod,level=lev), axs[num_mod+2], n, np.where(IWF_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+2].quiver(
            IWF_his_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IWF_his_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=700000.0,
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+2].quiver(
            IWF_his_u_slope.sel(models=mod,level=lev).where(IWF_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IWF_his_v_slope.sel(models=mod,level=lev).where(IWF_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=700000.0,
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+2].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=5e6, label="5e6", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+2].format(
            rtitle="1979-2014", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IWF".format(lev))
# %%
#   generate the ssp585 hgt and wind mask
IWF_ssp585_hgt_slope_ens = IWF_ssp585_hgt_slope.mean(dim="models", skipna=True)
IWF_ssp585_hgt_slope_ens_mask = ca.MME_reg_mask(IWF_ssp585_hgt_slope_ens, IWF_ssp585_hgt_slope.std(dim="models", skipna=True), len(models), True)

IWF_ssp585_u_slope_ens = IWF_ssp585_u_slope.mean(dim="models", skipna=True)
IWF_ssp585_u_slope_ens_mask = ca.MME_reg_mask(IWF_ssp585_u_slope_ens, IWF_ssp585_u_slope.std(dim="models", skipna=True), len(models), True)

IWF_ssp585_v_slope_ens = IWF_ssp585_v_slope.mean(dim="models", skipna=True)
IWF_ssp585_v_slope_ens_mask = ca.MME_reg_mask(IWF_ssp585_v_slope_ens, IWF_ssp585_v_slope.std(dim="models", skipna=True), len(models), True)

IWF_ssp585_wind_slope_ens_mask = ca.wind_check(
    xr.where(IWF_ssp585_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IWF_ssp585_v_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IWF_ssp585_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IWF_ssp585_v_slope_ens_mask > 0.0, 1.0, 0.0),
)
# %%
#   plot the avalue of hgt&u&v regress onto IWF in ERA5 and ssp585
startlevel = [-3.0e7, -2.0e7, -1.5e7]
endlevel = [3.0e7, 2.0e7, 1.5e7]
spacinglevel = [3.0e6, 2.0e6, 1.0e6]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 29), (7, 4))
    plot_array[6,3] = 0
    axs = fig.subplots(plot_array, proj=proj)

    #   set the geo_ticks and map projection to the plots
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 1
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
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
        IWF_ssp585_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev]/2, spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IWF_ssp585_hgt_slope_ens.sel(level=lev), axs[0], n, np.where(IWF_ssp585_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[0].quiver(
        IWF_ssp585_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IWF_ssp585_v_slope_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=700000.0,
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IWF_ssp585_u_slope_ens.sel(level=lev).where(IWF_ssp585_wind_slope_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IWF_ssp585_v_slope_ens.sel(level=lev).where(IWF_ssp585_wind_slope_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=700000.0,
        pivot="mid",
        color="black",
    )

    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=5e6, label="5e6", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="2015-2099", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+1].contourf(
            IWF_ssp585_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev]/2, spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IWF_ssp585_hgt_slope.sel(models=mod,level=lev), axs[num_mod+1], n, np.where(IWF_ssp585_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+1].quiver(
            IWF_ssp585_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IWF_ssp585_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=700000.0,
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+1].quiver(
            IWF_ssp585_u_slope.sel(models=mod,level=lev).where(IWF_ssp585_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IWF_ssp585_v_slope.sel(models=mod,level=lev).where(IWF_ssp585_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=700000.0,
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+1].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=5e6, label="5e6", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="2015-2099", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IWF".format(lev))
# %%
#   calculate the 2064-2099yr p3 regression
(
    IWF_ssp585_p3_hgt_slope,
    IWF_ssp585_p3_hgt_intercept,
    IWF_ssp585_p3_hgt_rvalue,
    IWF_ssp585_p3_hgt_pvalue,
    IWF_ssp585_p3_hgt_hypothesis,
) = ca.dim_linregress(ssp585_IWF_index.sel(time=ssp585_IWF_index.time.dt.year>=2064), hgtssp585_ver_JJA.sel(time=hgtssp585_ver_JJA.time.dt.year>=2064, level=[200.0, 500.0, 850.0]))

(
    IWF_ssp585_p3_u_slope,
    IWF_ssp585_p3_u_intercept,
    IWF_ssp585_p3_u_rvalue,
    IWF_ssp585_p3_u_pvalue,
    IWF_ssp585_p3_u_hypothesis,
) = ca.dim_linregress(ssp585_IWF_index.sel(time=ssp585_IWF_index.time.dt.year>=2064), ussp585_ver_JJA.sel(time=ussp585_ver_JJA.time.dt.year>=2064, level=[200.0, 500.0, 850.0]))

(
    IWF_ssp585_p3_v_slope,
    IWF_ssp585_p3_v_intercept,
    IWF_ssp585_p3_v_rvalue,
    IWF_ssp585_p3_v_pvalue,
    IWF_ssp585_p3_v_hypothesis,
) = ca.dim_linregress(ssp585_IWF_index.sel(time=ssp585_IWF_index.time.dt.year>=2064), vssp585_ver_JJA.sel(time=vssp585_ver_JJA.time.dt.year>=2064, level=[200.0, 500.0, 850.0]))

IWF_ssp585_p3_wind_mask = ca.wind_check(
    xr.where(IWF_ssp585_p3_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IWF_ssp585_p3_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IWF_ssp585_p3_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IWF_ssp585_p3_v_pvalue <= 0.05, 1.0, 0.0),
)
# %%
IWF_ssp585_p3_hgt_slope_ens = IWF_ssp585_p3_hgt_slope.mean(dim="models", skipna=True)
IWF_ssp585_p3_hgt_slope_ens_mask = ca.MME_reg_mask(IWF_ssp585_p3_hgt_slope_ens, IWF_ssp585_p3_hgt_slope.std(dim="models", skipna=True), len(models), True)

IWF_ssp585_p3_u_slope_ens = IWF_ssp585_p3_u_slope.mean(dim="models", skipna=True)
IWF_ssp585_p3_u_slope_ens_mask = ca.MME_reg_mask(IWF_ssp585_p3_u_slope_ens, IWF_ssp585_p3_u_slope.std(dim="models", skipna=True), len(models), True)

IWF_ssp585_p3_v_slope_ens = IWF_ssp585_p3_v_slope.mean(dim="models", skipna=True)
IWF_ssp585_p3_v_slope_ens_mask = ca.MME_reg_mask(IWF_ssp585_p3_v_slope_ens, IWF_ssp585_p3_v_slope.std(dim="models", skipna=True), len(models), True)

IWF_ssp585_p3_wind_slope_ens_mask = ca.wind_check(
    xr.where(IWF_ssp585_p3_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IWF_ssp585_p3_v_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IWF_ssp585_p3_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IWF_ssp585_p3_v_slope_ens_mask > 0.0, 1.0, 0.0),
)
# %%
#   plot the avalue of hgt&u&v regress onto IWF in ERA5 and ssp585 p3
startlevel = [-3.0e7, -2.0e7, -1.5e7]
endlevel = [3.0e7, 2.0e7, 1.5e7]
spacinglevel = [3.0e6, 2.0e6, 1.0e6]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 29), (7, 4))
    plot_array[6,3] = 0
    axs = fig.subplots(plot_array, proj=proj)

    #   set the geo_ticks and map projection to the plots
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 1
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
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
        IWF_ssp585_p3_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev]/2, spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IWF_ssp585_p3_hgt_slope_ens.sel(level=lev), axs[0], n, np.where(IWF_ssp585_p3_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[0].quiver(
        IWF_ssp585_p3_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IWF_ssp585_p3_v_slope_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=700000.0,
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IWF_ssp585_p3_u_slope_ens.sel(level=lev).where(IWF_ssp585_p3_wind_slope_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IWF_ssp585_p3_v_slope_ens.sel(level=lev).where(IWF_ssp585_p3_wind_slope_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=700000.0,
        pivot="mid",
        color="black",
    )

    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=5e6, label="5e6", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="2064-2099", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+1].contourf(
            IWF_ssp585_p3_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev]/2, spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IWF_ssp585_p3_hgt_slope.sel(models=mod,level=lev), axs[num_mod+1], n, np.where(IWF_ssp585_p3_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+1].quiver(
            IWF_ssp585_p3_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IWF_ssp585_p3_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=700000.0,
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+1].quiver(
            IWF_ssp585_p3_u_slope.sel(models=mod,level=lev).where(IWF_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IWF_ssp585_p3_v_slope.sel(models=mod,level=lev).where(IWF_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=700000.0,
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+1].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=5e6, label="5e6", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="2064-2099", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IWF".format(lev))
# %%
#   test the bootstrap test and permutation test
a = [136.3,136.3,135.8,135.4,134.7,135.0,134.1,143.3,147.8,148.8,134.8,135.2,134.9,149.5,141.2,135.4,134.8,135.8,135.0,133.7,134.4,134.9,134.8,134.5,134.3,135.2]
b = [136.3,136.3,135.8,135.4,134.7,135.0,134.1,143.3,147.8,148.8,134.8,135.2,134.9,136.3,136.3,135.8,135.4,134.7,135.0,134.1,143.3,147.8,148.8,134.8,135.2,134.9]
K = 10000
n1 = 13
n2 = 13
loc_num = range(0,n1+n2)
# %%
re_x_sample1 = [a[i] for i in range(13)]
re_y_sample1 = [b[i] for i in range(13)]
re_x_sample2 = [a[i] for i in np.arange(13,26)]
re_y_sample2 = [b[i] for i in np.arange(13,26)]
re_d = stats.linregress(re_x_sample1, re_y_sample1)[0] - stats.linregress(re_x_sample2, re_y_sample2)[0]
# %%
bs_d = np.zeros(K)
for test_i in range(K):
    bs_sample_list1 = np.random.choice(loc_num, size=13)
    bs_sample_list2 = np.random.choice(loc_num, size=13)
    bs_x_sample1 = [a[i] for i in bs_sample_list1]
    bs_y_sample1 = [b[i] for i in bs_sample_list1]
    bs_x_sample2 = [a[i] for i in bs_sample_list2]
    bs_y_sample2 = [b[i] for i in bs_sample_list2]
    bs_regress1 = stats.linregress(bs_x_sample1, bs_y_sample1)
    bs_regress2 = stats.linregress(bs_x_sample2, bs_y_sample2)
    d = bs_regress1[0]- bs_regress2[0]
    bs_d[test_i] = d
bs_d = np.sort(bs_d)
bs_freq = stats.relfreq(bs_d, numbins=10)
bs_pdf = bs_freq.frequency
bs_cdf = np.cumsum(bs_pdf)
# %%
fi_d = np.zeros(K)
for test_i in range(K):
    fi_sample_list1 = np.random.choice(loc_num, size=13, replace=False)
    fi_sample_list2 = list(set(loc_num) ^ set(fi_sample_list1))
    fi_x_sample1 = [a[i] for i in fi_sample_list1]
    fi_y_sample1 = [b[i] for i in fi_sample_list1]
    fi_x_sample2 = [a[i] for i in fi_sample_list2]
    fi_y_sample2 = [b[i] for i in fi_sample_list2]
        
    fi_regress1 = stats.linregress(fi_x_sample1, fi_y_sample1)
    fi_regress2 = stats.linregress(fi_x_sample2, fi_y_sample2)
    d = fi_regress1[0]- fi_regress2[0]
    fi_d[test_i] = d
fi_d = np.sort(fi_d)
fi_freq = stats.relfreq(fi_d, numbins=10)
fi_pdf = fi_freq.frequency
fi_cdf = np.cumsum(fi_pdf)
# %%
bs_freq = stats.relfreq(bs_d, numbins=200)
bs_pdf = bs_freq.frequency
bs_cdf = np.cumsum(bs_pdf)
fi_freq = stats.relfreq(fi_d, numbins=200)
fi_pdf = fi_freq.frequency
fi_cdf = np.cumsum(fi_pdf)
x1 = bs_freq.lowerlimit + np.linspace(0, bs_freq.binsize*bs_freq.frequency.size, bs_freq.frequency.size)
x2 = fi_freq.lowerlimit + np.linspace(0, fi_freq.binsize*fi_freq.frequency.size, fi_freq.frequency.size)
fig = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
axs = fig.subplots(ncols=1, nrows=1)
lw = 0.8
# cycle = pplt.Cycle('Pastel1', 'Pastel2', 27, left=0.1)
# axs.format(grid=False, suptitle="ssp585 IWF & SAM", ylabel="", xlabel="year")
# ===================================================
m1 = axs[0].line(
    x1, bs_cdf, lw=lw, color="black",)
m1 = axs[0].line(
    x1, bs_pdf, lw=lw, color="black",)

m1 = axs[0].line(
    x2, fi_cdf, lw=lw, color="blue",)
m1 = axs[0].line(
    x2, fi_pdf, lw=lw, color="blue",)
axs[0].axvline(re_d)
# %%
#   calculate the avalue difference of ssp585 p3 and historical
IWF_diff_hgt_slope = IWF_ssp585_p3_hgt_slope - IWF_his_hgt_slope
IWF_diff_hgt_slope_ens = IWF_diff_hgt_slope.mean(dim="models", skipna=True)
IWF_diff_u_slope = IWF_ssp585_p3_u_slope - IWF_his_u_slope
IWF_diff_u_slope_ens = IWF_diff_u_slope.mean(dim="models", skipna=True)
IWF_diff_v_slope = IWF_ssp585_p3_v_slope - IWF_his_v_slope
IWF_diff_v_slope_ens = IWF_diff_v_slope.mean(dim="models", skipna=True)
# %%
#   plot the avalue of hgt&u&v difference of ssp585 p3 - historical
startlevel = [-2.0e7, -0.6e7, -0.6e7]
endlevel = [2.0e7, 0.6e7, 0.6e7]
spacinglevel = [1.0e6, 0.3e6, 0.3e6]
scalelevel=[1000000, 500000, 300000]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 29), (7, 4))
    plot_array[6,3] = 0
    axs = fig.subplots(plot_array, proj=proj)

    #   set the geo_ticks and map projection to the plots
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 1
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
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
        IWF_diff_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev]/2, spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )

    m = axs[0].quiver(
        IWF_diff_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IWF_diff_v_slope_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=5e6, label="5e6", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="diff", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+1].contourf(
            IWF_diff_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev]/2, spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )

        m = axs[num_mod+1].quiver(
            IWF_diff_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IWF_diff_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+1].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=5e6, label="5e6", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="diff", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IWF".format(lev))
# %%
