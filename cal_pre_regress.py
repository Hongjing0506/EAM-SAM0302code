'''
Author: ChenHJ
Date: 2022-04-14 16:32:41
LastEditors: ChenHJ
LastEditTime: 2022-04-26 13:35:43
FilePath: /chenhj/0302code/cal_pre_regress.py
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
import metpy.constants as constants
import geocat.comp
from windspharm.xarray import VectorWind
import skill_metrics as sm

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
#   read the data in CRU/historical/ssp585
fpreCRU = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/cru_ts4.01_r144x72_195001-201412.nc"
)
preCRU = fpreCRU["pre"]
preCRU_JJA = ca.p_time(preCRU, 6, 8, True)/30.67
preCRU_JJA = ca.detrend_dim(preCRU_JJA, "time", deg=1, demean=False)
preCRU_JJA.attrs["units"] = "mm/day"

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
#   calculate the precipitation regress onto India precipitation
lat = preCRU_JJA.coords["lat"]
lon = preCRU_JJA.coords["lon"]
lat_India_range = lat[(lat >= 8.0) & (lat <= 28.0)]
lon_India_range = lon[(lon >= 70.0) & (lon <= 86.0)]

preCRU_India_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)
lat = preGPCP_JJA.coords["lat"]
lon = preGPCP_JJA.coords["lon"]
lat_India_range = lat[(lat >= 8.0) & (lat <= 28.0)]
lon_India_range = lon[(lon >= 70.0) & (lon <= 86.0)]
# lat_India_range = lat[(lat >= 15.0) & (lat <= 28.0)]
# lat_India_range = lat[(lat >= 30.0) & (lat <= 35.0)]
# lon_India_range = lon[(lon >= 112.5) & (lon <= 120.0)]

preGPCP_India_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)

lat = prehis_JJA.coords["lat"]
lon = prehis_JJA.coords["lon"]

lat_India_range = lat[(lat >= 8.0) & (lat <= 28.0)]
# lat_India_range = lat[(lat >= 15.0) & (lat <= 28.0)]
lon_India_range = lon[(lon >= 70.0) & (lon <= 86.0)]
# lat_India_range = lat[(lat >= 30.0) & (lat <= 35.0)]
# lon_India_range = lon[(lon >= 112.5) & (lon <= 120.0)]

prehis_India_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)
pressp585_India_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)

# %%
(
    pre_CRU_India_pre_slope,
    pre_CRU_India_pre_intercept,
    pre_CRU_India_pre_rvalue,
    pre_CRU_India_pre_pvalue,
    pre_CRU_India_pre_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA.sel(time=preCRU_India_JJA.time.dt.year>=1979), preCRU_JJA.sel(time=preCRU_JJA.time.dt.year>=1979))

(
    pre_GPCP_India_pre_slope,
    pre_GPCP_India_pre_intercept,
    pre_GPCP_India_pre_rvalue,
    pre_GPCP_India_pre_pvalue,
    pre_GPCP_India_pre_hypothesis,
) = ca.dim_linregress(preGPCP_India_JJA, preGPCP_JJA)

(
    pre_his_India_pre_slope,
    pre_his_India_pre_intercept,
    pre_his_India_pre_rvalue,
    pre_his_India_pre_pvalue,
    pre_his_India_pre_hypothesis,
) = ca.dim_linregress(prehis_India_JJA.sel(time=prehis_India_JJA.time.dt.year>=1979), prehis_JJA.sel(time=prehis_JJA.time.dt.year>=1979))
(
    pre_ssp585_India_pre_slope,
    pre_ssp585_India_pre_intercept,
    pre_ssp585_India_pre_rvalue,
    pre_ssp585_India_pre_pvalue,
    pre_ssp585_India_pre_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA, pressp585_JJA)

# %%

pre_his_India_pre_slope_ens = pre_his_India_pre_slope.mean(dim="models", skipna=True)
pre_ssp585_India_pre_slope_ens = pre_ssp585_India_pre_slope.mean(dim="models", skipna=True)
pre_his_India_pre_slope_ens_mask = ca.MME_reg_mask(pre_his_India_pre_slope_ens, pre_his_India_pre_slope.std(dim="models", skipna=True), len(pre_his_India_pre_slope.coords["models"]), True)
pre_ssp585_India_pre_slope_ens_mask = ca.MME_reg_mask(pre_ssp585_India_pre_slope_ens, pre_ssp585_India_pre_slope.std(dim="models", skipna=True), len(pre_ssp585_India_pre_slope.coords["models"]), True)

pre_his_India_pre_rvalue_ens = pre_his_India_pre_rvalue.mean(dim="models", skipna=True)
pre_ssp585_India_pre_rvalue_ens = pre_ssp585_India_pre_rvalue.mean(dim="models", skipna=True)
pre_his_India_pre_rvalue_ens_mask = ca.MME_reg_mask(pre_his_India_pre_rvalue_ens, pre_his_India_pre_rvalue.std(dim="models", skipna=True), len(pre_his_India_pre_rvalue.coords["models"]), True)
pre_ssp585_India_pre_rvalue_ens_mask = ca.MME_reg_mask(pre_ssp585_India_pre_rvalue_ens, pre_ssp585_India_pre_rvalue.std(dim="models", skipna=True), len(pre_ssp585_India_pre_rvalue.coords["models"]), True)

# %%
#   plot the regression coefficients in CRU, GPCP, historical run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (6, 5))
plot_array[5,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_CRU_India_pre_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_CRU_India_pre_slope, axs[0], n, np.where(pre_CRU_India_pre_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="CRU",
)
# ===================================================
con = axs[1].contourf(
    pre_GPCP_India_pre_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_GPCP_India_pre_slope, axs[1], n, np.where(pre_GPCP_India_pre_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="GPCP",
)
# ===================================================
con = axs[2].contourf(
    pre_his_India_pre_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_his_India_pre_slope_ens, axs[2], n, np.where(pre_his_India_pre_slope_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[2].format(
    rtitle="1979-2014", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(pre_his_India_pre_slope.coords["models"].data):
    con = axs[num_models+3].contourf(
    pre_his_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
    sepl.plt_sig(
        pre_his_India_pre_slope.sel(models=mod), axs[num_models+3], n, np.where(pre_his_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 4.0,
    )

    axs[num_models+3].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")

# %%
#   plot the correlation coefficients rvalue in CRU, GPCP, historical run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (6, 5))
plot_array[5,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_CRU_India_pre_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_CRU_India_pre_rvalue, axs[0], n, np.where(pre_CRU_India_pre_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="CRU",
)
# ===================================================
con = axs[1].contourf(
    pre_GPCP_India_pre_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_GPCP_India_pre_rvalue, axs[1], n, np.where(pre_GPCP_India_pre_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="GPCP",
)
# # ===================================================
con = axs[2].contourf(
    pre_his_India_pre_rvalue_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    
    )
sepl.plt_sig(
    pre_his_India_pre_rvalue_ens, axs[2], n, np.where(pre_his_India_pre_rvalue_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[2].format(
    rtitle="1979-2014", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(pre_his_India_pre_slope.coords["models"].data):
    con = axs[num_models+3].contourf(
    pre_his_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
    sepl.plt_sig(
        pre_his_India_pre_rvalue.sel(models=mod), axs[num_models+3], n, np.where(pre_his_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 4.0,
    )

    axs[num_models+3].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the regression coefficients avalue in ssp585 run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 29), (7, 4))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_ssp585_India_pre_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_ssp585_India_pre_slope_ens, axs[0], n, np.where(pre_ssp585_India_pre_slope_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[0].format(
    rtitle="2015-2099", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(pre_ssp585_India_pre_slope.coords["models"].data):
    con = axs[num_models+1].contourf(
    pre_ssp585_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
    sepl.plt_sig(
        pre_ssp585_India_pre_slope.sel(models=mod), axs[num_models+1], n, np.where(pre_ssp585_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 4.0,
    )

    axs[num_models+1].format(
        rtitle="2015-2099", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the correlation coefficients rvalue in ssp585 run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 29), (7, 4))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_ssp585_India_pre_rvalue_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_ssp585_India_pre_rvalue_ens, axs[0], n, np.where(pre_ssp585_India_pre_rvalue_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[0].format(
    rtitle="2015-2099", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(pre_ssp585_India_pre_rvalue.coords["models"].data):
    con = axs[num_models+1].contourf(
    pre_ssp585_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
    sepl.plt_sig(
        pre_ssp585_India_pre_rvalue.sel(models=mod), axs[num_models+1], n, np.where(pre_ssp585_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 4.0,
    )

    axs[num_models+1].format(
        rtitle="2015-2099", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   calculate the 2064-2099 ssp585 p3 precipitation regression onto IndR
(
    pre_ssp585_p3_India_pre_slope,
    pre_ssp585_p3_India_pre_intercept,
    pre_ssp585_p3_India_pre_rvalue,
    pre_ssp585_p3_India_pre_pvalue,
    pre_ssp585_p3_India_pre_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2064), pressp585_JJA.sel(time=pressp585_JJA.time.dt.year>=2064))

# %%
models = pre_ssp585_p3_India_pre_slope.coords["models"]
pre_ssp585_p3_India_pre_slope_ens = pre_ssp585_p3_India_pre_slope.mean(dim="models", skipna=True)
pre_ssp585_p3_India_pre_slope_ens_mask = ca.MME_reg_mask(pre_ssp585_p3_India_pre_slope_ens, pre_ssp585_p3_India_pre_slope.std(dim="models", skipna=True), len(pre_ssp585_p3_India_pre_slope.coords["models"]), True)
pre_ssp585_p3_India_pre_rvalue_ens = pre_ssp585_p3_India_pre_rvalue.mean(dim="models", skipna=True)
pre_ssp585_p3_India_pre_rvalue_ens_mask = ca.MME_reg_mask(pre_ssp585_p3_India_pre_rvalue_ens, pre_ssp585_p3_India_pre_rvalue.std(dim="models", skipna=True), len(pre_ssp585_p3_India_pre_rvalue.coords["models"]), True)

pre_diff_India_pre_slope = pre_ssp585_p3_India_pre_slope-pre_his_India_pre_slope
pre_diff_India_pre_slope_ens = pre_diff_India_pre_slope.mean(dim="models", skipna=True)
pre_diff_India_pre_slope_ens_mask = ca.MME_reg_mask(pre_diff_India_pre_slope_ens, pre_diff_India_pre_slope.std(dim="models", skipna=True), len(models), True)

pre_diff_India_pre_rvalue = pre_ssp585_p3_India_pre_rvalue-pre_his_India_pre_rvalue
pre_diff_India_pre_rvalue_ens = pre_diff_India_pre_rvalue.mean(dim="models", skipna=True)
pre_diff_India_pre_rvalue_ens_mask = ca.MME_reg_mask(pre_diff_India_pre_rvalue_ens, pre_diff_India_pre_rvalue.std(dim="models", skipna=True), len(models), True)
# %%
#   calculate the significance area change
gmodels = ['MPI-ESM1-2-HR', 'EC-Earth3-Veg', 'UKESM1-0-LL', 'EC-Earth3', 'CMCC-ESM2', 'MRI-ESM2-0', 'HadGEM3-GC31-LL', 'TaiESM1', 'NorESM2-LM', 'MIROC-ES2L']
pre_diff_India_pre_sig = xr.where(pre_ssp585_p3_India_pre_pvalue <= 0.05, 1.0, 0.0) - xr.where(pre_his_India_pre_pvalue <= 0.05, 1.0, 0.0)
pre_diff_India_pre_sig_ens = pre_diff_India_pre_sig.mean(dim="models", skipna=True)
pre_diff_India_pre_sig_gmodels_ens = pre_diff_India_pre_sig.sel(models=gmodels).mean(dim="models", skipna=True)

#   plot the significance area change in precipitation regress onto IndR
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 29), (7, 4))
plot_array[-1,-1] = 0
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
    # #   IWF area
    # x0 = 90
    # y0 = 5.0
    # width = 50.0
    # height = 27.5
    # patches(ax, x0 - cl, y0, width, height, proj)    
# ======================================
con = axs[0].contourf(
    pre_diff_India_pre_sig_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.05},
    levels=np.arange(-1.0,1.1,0.1),
    zorder=0.8,
    extend="both"
)
axs[0].format(
    rtitle="diff", ltitle="MME",
)
# ======================================
for num_mod, mod in enumerate(models):
    con = axs[num_mod+1].contourf(
        pre_diff_India_pre_sig.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.05},
        levels=np.arange(-1.0,1.1,0.1),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+1].format(
        rtitle="diff", ltitle="{}".format(mod.data),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the significance area change in precipitation regress onto IndR for good models
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 13), (3, 4))
plot_array[-1,-1] = 0
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
    # #   IWF area
    # x0 = 90
    # y0 = 5.0
    # width = 50.0
    # height = 27.5
    # patches(ax, x0 - cl, y0, width, height, proj)    
# ======================================
con = axs[0].contourf(
    pre_diff_India_pre_sig_gmodels_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.05},
    levels=np.arange(-1.0,1.1,0.2),
    zorder=0.8,
    extend="both"
)
axs[0].format(
    rtitle="diff", ltitle="MME",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+1].contourf(
        pre_diff_India_pre_sig.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.05},
        levels=np.arange(-1.0,1.1,0.2),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+1].format(
        rtitle="diff", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the regression coefficients avalue in ssp585 p3
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 29), (7, 4))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_ssp585_p3_India_pre_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_ssp585_p3_India_pre_slope_ens, axs[0], n, np.where(pre_ssp585_p3_India_pre_slope_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[0].format(
    rtitle="2064-2099", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(pre_ssp585_p3_India_pre_slope.coords["models"].data):
    con = axs[num_models+1].contourf(
    pre_ssp585_p3_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
    sepl.plt_sig(
        pre_ssp585_p3_India_pre_slope.sel(models=mod), axs[num_models+1], n, np.where(pre_ssp585_p3_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 4.0,
    )

    axs[num_models+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")

# %%
#   plot the correlation coefficients rvalue in ssp585 p3
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 29), (7, 4))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_ssp585_p3_India_pre_rvalue_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_ssp585_p3_India_pre_rvalue_ens, axs[0], n, np.where(pre_ssp585_p3_India_pre_rvalue_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[0].format(
    rtitle="2064-2099", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(pre_ssp585_p3_India_pre_rvalue.coords["models"].data):
    con = axs[num_models+1].contourf(
    pre_ssp585_p3_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
    sepl.plt_sig(
        pre_ssp585_p3_India_pre_rvalue.sel(models=mod), axs[num_models+1], n, np.where(pre_ssp585_p3_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 4.0,
    )

    axs[num_models+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")

# %%
#   plot the regression coefficients avalue in ssp585 p3-historical
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 29), (7, 4))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_diff_India_pre_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )

axs[0].format(
    rtitle="diff", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(pre_diff_India_pre_slope.coords["models"].data):
    con = axs[num_models+1].contourf(
    pre_diff_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )

    axs[num_models+1].format(
        rtitle="diff", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")

# %%
#   plot the correlation coefficients rvalue in ssp585 p3-historical
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 29), (7, 4))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_diff_India_pre_rvalue_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )

axs[0].format(
    rtitle="diff", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(pre_diff_India_pre_rvalue.coords["models"].data):
    con = axs[num_models+1].contourf(
    pre_diff_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )

    axs[num_models+1].format(
        rtitle="diff", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   read the var in ERA5 and historical/ssp585
fhgtERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/hgt_mon_r144x72_195001-201412.nc")
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

hgtERA5_ver_JJA = ca.p_time(hgtERA5, 6, 8, True)
hgtERA5_ver_JJA = hgtERA5_ver_JJA-hgtERA5_ver_JJA.mean(dim="lon", skipna=True)
uERA5_ver_JJA = ca.p_time(uERA5, 6, 8, True)
vERA5_ver_JJA = ca.p_time(vERA5, 6, 8, True)
qERA5_ver_JJA = ca.p_time(qERA5, 6, 9, True)
spERA5_JJA = ca.p_time(spERA5, 6, 8, True)

hgtERA5_ver_JJA = ca.detrend_dim(hgtERA5_ver_JJA, "time", deg=1, demean=False)
uERA5_ver_JJA = ca.detrend_dim(uERA5_ver_JJA, "time", deg=1, demean=False)
vERA5_ver_JJA = ca.detrend_dim(vERA5_ver_JJA, "time", deg=1, demean=False)
qERA5_ver_JJA = ca.detrend_dim(qERA5_ver_JJA, "time", deg=1, demean=False)
spERA5_JJA = ca.detrend_dim(spERA5_JJA, "time", deg=1, demean=False)

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
#   calculate the hgt/u/v regression onto IndR in ERA5, historical, ssp585, ssp585_p3
preCRU_India_JJA.coords["time"] = hgtERA5_ver_JJA.coords["time"]
preGPCP_India_JJA.coords["time"] = hgtERA5_ver_JJA.sel(time=hgtERA5_ver_JJA.time.dt.year>=1979).coords["time"]
(
    IndRCRU_ERA5_hgt_slope,
    IndRCRU_ERA5_hgt_intercept,
    IndRCRU_ERA5_hgt_rvalue,
    IndRCRU_ERA5_hgt_pvalue,
    IndRCRU_ERA5_hgt_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA.sel(time=preCRU_India_JJA.time.dt.year>=1979), hgtERA5_ver_JJA.sel(time=hgtERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndRCRU_ERA5_u_slope,
    IndRCRU_ERA5_u_intercept,
    IndRCRU_ERA5_u_rvalue,
    IndRCRU_ERA5_u_pvalue,
    IndRCRU_ERA5_u_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA.sel(time=preCRU_India_JJA.time.dt.year>=1979), uERA5_ver_JJA.sel(time=uERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndRCRU_ERA5_v_slope,
    IndRCRU_ERA5_v_intercept,
    IndRCRU_ERA5_v_rvalue,
    IndRCRU_ERA5_v_pvalue,
    IndRCRU_ERA5_v_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA.sel(time=preCRU_India_JJA.time.dt.year>=1979), vERA5_ver_JJA.sel(time=vERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndRGPCP_ERA5_hgt_slope,
    IndRGPCP_ERA5_hgt_intercept,
    IndRGPCP_ERA5_hgt_rvalue,
    IndRGPCP_ERA5_hgt_pvalue,
    IndRGPCP_ERA5_hgt_hypothesis,
) = ca.dim_linregress(preGPCP_India_JJA.sel(time=preGPCP_India_JJA.time.dt.year>=1979), hgtERA5_ver_JJA.sel(time=hgtERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndRGPCP_ERA5_u_slope,
    IndRGPCP_ERA5_u_intercept,
    IndRGPCP_ERA5_u_rvalue,
    IndRGPCP_ERA5_u_pvalue,
    IndRGPCP_ERA5_u_hypothesis,
) = ca.dim_linregress(preGPCP_India_JJA.sel(time=preGPCP_India_JJA.time.dt.year>=1979), uERA5_ver_JJA.sel(time=uERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndRGPCP_ERA5_v_slope,
    IndRGPCP_ERA5_v_intercept,
    IndRGPCP_ERA5_v_rvalue,
    IndRGPCP_ERA5_v_pvalue,
    IndRGPCP_ERA5_v_hypothesis,
) = ca.dim_linregress(preGPCP_India_JJA.sel(time=preGPCP_India_JJA.time.dt.year>=1979), vERA5_ver_JJA.sel(time=vERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndR_his_hgt_slope,
    IndR_his_hgt_intercept,
    IndR_his_hgt_rvalue,
    IndR_his_hgt_pvalue,
    IndR_his_hgt_hypothesis,
) = ca.dim_linregress(prehis_India_JJA.sel(time=prehis_India_JJA.time.dt.year>=1979), hgthis_ver_JJA.sel(time=hgthis_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndR_his_u_slope,
    IndR_his_u_intercept,
    IndR_his_u_rvalue,
    IndR_his_u_pvalue,
    IndR_his_u_hypothesis,
) = ca.dim_linregress(prehis_India_JJA.sel(time=prehis_India_JJA.time.dt.year>=1979), uhis_ver_JJA.sel(time=uhis_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndR_his_v_slope,
    IndR_his_v_intercept,
    IndR_his_v_rvalue,
    IndR_his_v_pvalue,
    IndR_his_v_hypothesis,
) = ca.dim_linregress(prehis_India_JJA.sel(time=prehis_India_JJA.time.dt.year>=1979), vhis_ver_JJA.sel(time=vhis_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndR_ssp585_hgt_slope,
    IndR_ssp585_hgt_intercept,
    IndR_ssp585_hgt_rvalue,
    IndR_ssp585_hgt_pvalue,
    IndR_ssp585_hgt_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA, hgtssp585_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

(
    IndR_ssp585_u_slope,
    IndR_ssp585_u_intercept,
    IndR_ssp585_u_rvalue,
    IndR_ssp585_u_pvalue,
    IndR_ssp585_u_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA, ussp585_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

(
    IndR_ssp585_v_slope,
    IndR_ssp585_v_intercept,
    IndR_ssp585_v_rvalue,
    IndR_ssp585_v_pvalue,
    IndR_ssp585_v_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA, vssp585_ver_JJA.sel(level=[200.0, 500.0, 850.0]))
# %%
(
    IndR_ssp585_p3_hgt_slope,
    IndR_ssp585_p3_hgt_intercept,
    IndR_ssp585_p3_hgt_rvalue,
    IndR_ssp585_p3_hgt_pvalue,
    IndR_ssp585_p3_hgt_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2064), hgtssp585_ver_JJA.sel(time=hgtssp585_ver_JJA.time.dt.year>=2064, level=[200.0, 500.0, 850.0]))

(
    IndR_ssp585_p3_u_slope,
    IndR_ssp585_p3_u_intercept,
    IndR_ssp585_p3_u_rvalue,
    IndR_ssp585_p3_u_pvalue,
    IndR_ssp585_p3_u_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2064), ussp585_ver_JJA.sel(time=ussp585_ver_JJA.time.dt.year>=2064, level=[200.0, 500.0, 850.0]))

(
    IndR_ssp585_p3_v_slope,
    IndR_ssp585_p3_v_intercept,
    IndR_ssp585_p3_v_rvalue,
    IndR_ssp585_p3_v_pvalue,
    IndR_ssp585_p3_v_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2064), vssp585_ver_JJA.sel(time=vssp585_ver_JJA.time.dt.year>=2064, level=[200.0, 500.0, 850.0]))
# %%
#   calculate the windcheck and ensmean
IndRCRU_ERA5_wind_mask = ca.wind_check(
    xr.where(IndRCRU_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRCRU_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRCRU_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRCRU_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
)

IndRGPCP_ERA5_wind_mask = ca.wind_check(
    xr.where(IndRGPCP_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRGPCP_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRGPCP_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRGPCP_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
)

IndR_his_wind_mask = ca.wind_check(
    xr.where(IndR_his_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_his_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_his_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_his_v_pvalue <= 0.05, 1.0, 0.0),
)

IndR_ssp585_wind_mask = ca.wind_check(
    xr.where(IndR_ssp585_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_ssp585_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_ssp585_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_ssp585_v_pvalue <= 0.05, 1.0, 0.0),
)

IndR_ssp585_p3_wind_mask = ca.wind_check(
    xr.where(IndR_ssp585_p3_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_pvalue <= 0.05, 1.0, 0.0),
)

IndR_his_hgt_slope_ens = IndR_his_hgt_slope.mean(dim="models", skipna=True)
IndR_his_hgt_slope_ens_mask = ca.MME_reg_mask(IndR_his_hgt_slope_ens, IndR_his_hgt_slope.std(dim="models", skipna=True), len(models), True)

IndR_ssp585_hgt_slope_ens = IndR_ssp585_hgt_slope.mean(dim="models", skipna=True)
IndR_ssp585_hgt_slope_ens_mask = ca.MME_reg_mask(IndR_ssp585_hgt_slope_ens, IndR_ssp585_hgt_slope.std(dim="models", skipna=True), len(models), True)

IndR_ssp585_p3_hgt_slope_ens = IndR_ssp585_p3_hgt_slope.mean(dim="models", skipna=True)
IndR_ssp585_p3_hgt_slope_ens_mask = ca.MME_reg_mask(IndR_ssp585_p3_hgt_slope_ens, IndR_ssp585_p3_hgt_slope.std(dim="models", skipna=True), len(models), True)

IndR_his_u_slope_ens = IndR_his_u_slope.mean(dim="models", skipna=True)
IndR_his_u_slope_ens_mask = ca.MME_reg_mask(IndR_his_u_slope_ens, IndR_his_u_slope.std(dim="models", skipna=True), len(models), True)

IndR_ssp585_u_slope_ens = IndR_ssp585_u_slope.mean(dim="models", skipna=True)
IndR_ssp585_u_slope_ens_mask = ca.MME_reg_mask(IndR_ssp585_u_slope_ens, IndR_ssp585_u_slope.std(dim="models", skipna=True), len(models), True)

IndR_ssp585_p3_u_slope_ens = IndR_ssp585_p3_u_slope.mean(dim="models", skipna=True)
IndR_ssp585_p3_u_slope_ens_mask = ca.MME_reg_mask(IndR_ssp585_p3_u_slope_ens, IndR_ssp585_p3_u_slope.std(dim="models", skipna=True), len(models), True)

IndR_his_v_slope_ens = IndR_his_v_slope.mean(dim="models", skipna=True)
IndR_his_v_slope_ens_mask = ca.MME_reg_mask(IndR_his_v_slope_ens, IndR_his_v_slope.std(dim="models", skipna=True), len(models), True)

IndR_ssp585_v_slope_ens = IndR_ssp585_v_slope.mean(dim="models", skipna=True)
IndR_ssp585_v_slope_ens_mask = ca.MME_reg_mask(IndR_ssp585_v_slope_ens, IndR_ssp585_v_slope.std(dim="models", skipna=True), len(models), True)

IndR_ssp585_p3_v_slope_ens = IndR_ssp585_p3_v_slope.mean(dim="models", skipna=True)
IndR_ssp585_p3_v_slope_ens_mask = ca.MME_reg_mask(IndR_ssp585_p3_v_slope_ens, IndR_ssp585_p3_v_slope.std(dim="models", skipna=True), len(models), True)

IndR_his_hgt_rvalue_ens = IndR_his_hgt_rvalue.mean(dim="models", skipna=True)
IndR_his_hgt_rvalue_ens_mask = ca.MME_reg_mask(IndR_his_hgt_rvalue_ens, IndR_his_hgt_rvalue.std(dim="models", skipna=True), len(models), True)

IndR_ssp585_hgt_rvalue_ens = IndR_ssp585_hgt_rvalue.mean(dim="models", skipna=True)
IndR_ssp585_hgt_rvalue_ens_mask = ca.MME_reg_mask(IndR_ssp585_hgt_rvalue_ens, IndR_ssp585_hgt_rvalue.std(dim="models", skipna=True), len(models), True)

IndR_ssp585_p3_hgt_rvalue_ens = IndR_ssp585_p3_hgt_rvalue.mean(dim="models", skipna=True)
IndR_ssp585_p3_hgt_rvalue_ens_mask = ca.MME_reg_mask(IndR_ssp585_p3_hgt_rvalue_ens, IndR_ssp585_p3_hgt_rvalue.std(dim="models", skipna=True), len(models), True)

IndR_his_u_rvalue_ens = IndR_his_u_rvalue.mean(dim="models", skipna=True)
IndR_his_u_rvalue_ens_mask = ca.MME_reg_mask(IndR_his_u_rvalue_ens, IndR_his_u_rvalue.std(dim="models", skipna=True), len(models), True)

IndR_ssp585_u_rvalue_ens = IndR_ssp585_u_rvalue.mean(dim="models", skipna=True)
IndR_ssp585_u_rvalue_ens_mask = ca.MME_reg_mask(IndR_ssp585_u_rvalue_ens, IndR_ssp585_u_rvalue.std(dim="models", skipna=True), len(models), True)

IndR_ssp585_p3_u_rvalue_ens = IndR_ssp585_p3_u_rvalue.mean(dim="models", skipna=True)
IndR_ssp585_p3_u_rvalue_ens_mask = ca.MME_reg_mask(IndR_ssp585_p3_u_rvalue_ens, IndR_ssp585_p3_u_rvalue.std(dim="models", skipna=True), len(models), True)

IndR_his_v_rvalue_ens = IndR_his_v_rvalue.mean(dim="models", skipna=True)
IndR_his_v_rvalue_ens_mask = ca.MME_reg_mask(IndR_his_v_rvalue_ens, IndR_his_v_rvalue.std(dim="models", skipna=True), len(models), True)

IndR_ssp585_v_rvalue_ens = IndR_ssp585_v_rvalue.mean(dim="models", skipna=True)
IndR_ssp585_v_rvalue_ens_mask = ca.MME_reg_mask(IndR_ssp585_v_rvalue_ens, IndR_ssp585_v_rvalue.std(dim="models", skipna=True), len(models), True)

IndR_ssp585_p3_v_rvalue_ens = IndR_ssp585_p3_v_rvalue.mean(dim="models", skipna=True)
IndR_ssp585_p3_v_rvalue_ens_mask = ca.MME_reg_mask(IndR_ssp585_p3_v_rvalue_ens, IndR_ssp585_p3_v_rvalue.std(dim="models", skipna=True), len(models), True)

IndR_his_wind_ens_mask = ca.wind_check(
    xr.where(IndR_his_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_v_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_v_slope_ens_mask > 0.0, 1.0, 0.0),
)
IndR_ssp585_wind_ens_mask = ca.wind_check(
    xr.where(IndR_ssp585_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_v_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_v_slope_ens_mask > 0.0, 1.0, 0.0),
)
IndR_ssp585_p3_wind_ens_mask = ca.wind_check(
    xr.where(IndR_ssp585_p3_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_slope_ens_mask > 0.0, 1.0, 0.0),
)

# %%
# IndRCRU_ERA5_u_slope_load = IndRCRU_ERA5_u_slope.load()
# IndRCRU_ERA5_v_slope_load = IndRCRU_ERA5_v_slope.load()

# IndRGPCP_ERA5_u_slope_load = IndRGPCP_ERA5_u_slope.load()
# IndRGPCP_ERA5_v_slope_load = IndRGPCP_ERA5_v_slope.load()

# IndR_his_u_slope_load = IndR_his_u_slope.load()
# IndR_his_v_slope_load = IndR_his_v_slope.load()

# IndR_ssp585_u_slope_load = IndR_ssp585_u_slope.load()
# IndR_ssp585_v_slope_load = IndR_ssp585_v_slope.load()

# IndR_ssp585_p3_u_slope_load = IndR_ssp585_p3_u_slope.load()
# IndR_ssp585_p3_v_slope_load = IndR_ssp585_p3_v_slope.load()

# IndRCRU_ERA5_wind_mask_load = IndRCRU_ERA5_wind_mask.load()
# IndRGPCP_ERA5_wind_mask_load = IndRGPCP_ERA5_wind_mask.load()
# IndR_his_wind_mask_load = IndR_his_wind_mask.load()
# IndR_ssp585_wind_mask_load = IndR_ssp585_wind_mask.load()
# IndR_ssp585_p3_wind_mask_load = IndR_ssp585_p3_wind_mask.load()

# IndR_his_u_slope_ens_load = IndR_his_u_slope_ens.load()
# IndR_his_v_slope_ens_load = IndR_his_v_slope_ens.load()
# IndR_his_wind_slope_ens_mask_load = IndR_his_wind_slope_ens_mask.load()

# IndR_ssp585_u_slope_ens_load = IndR_ssp585_u_slope_ens.load()
# IndR_ssp585_v_slope_ens_load = IndR_ssp585_v_slope_ens.load()
# IndR_ssp585_wind_ens_mask_load = IndR_ssp585_wind_ens_mask.load()

# IndR_ssp585_p3_u_slope_ens_load = IndR_ssp585_p3_u_slope_ens.load()
# IndR_ssp585_p3_v_slope_ens_load = IndR_ssp585_p3_v_slope_ens.load()
# IndR_ssp585_p3_wind_ens_mask_load = IndR_ssp585_p3_wind_ens_mask.load()
# # %%
# IndRCRU_ERA5_u_rvalue_load = IndRCRU_ERA5_u_rvalue.load()
# IndRCRU_ERA5_v_rvalue_load = IndRCRU_ERA5_v_rvalue.load()

# IndRGPCP_ERA5_u_rvalue_load = IndRGPCP_ERA5_u_rvalue.load()
# IndRGPCP_ERA5_v_rvalue_load = IndRGPCP_ERA5_v_rvalue.load()

# IndR_his_u_rvalue_load = IndR_his_u_rvalue.load()
# IndR_his_v_rvalue_load = IndR_his_v_rvalue.load()

# IndR_ssp585_u_rvalue_load = IndR_ssp585_u_rvalue.load()
# IndR_ssp585_v_rvalue_load = IndR_ssp585_v_rvalue.load()

# IndR_ssp585_p3_u_rvalue_load = IndR_ssp585_p3_u_rvalue.load()
# IndR_ssp585_p3_v_rvalue_load = IndR_ssp585_p3_v_rvalue.load()

# IndR_his_u_rvalue_ens_load = IndR_his_u_rvalue_ens.load()
# IndR_his_v_rvalue_ens_load = IndR_his_v_rvalue_ens.load()

# IndR_ssp585_u_rvalue_ens_load = IndR_ssp585_u_rvalue_ens.load()
# IndR_ssp585_v_rvalue_ens_load = IndR_ssp585_v_rvalue_ens.load()

# IndR_ssp585_p3_u_rvalue_ens_load = IndR_ssp585_p3_u_rvalue_ens.load()
# IndR_ssp585_p3_v_rvalue_ens_load = IndR_ssp585_p3_v_rvalue_ens.load()
# %%
#   output the regression data
#   creatte the dataset first
models=IndR_his_hgt_slope.coords["models"]
level=IndR_his_hgt_slope.coords["level"]
lat=IndR_his_hgt_slope.coords["lat"]
lon=IndR_his_hgt_slope.coords["lon"]

IndRCRU_ERA5_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], IndRCRU_ERA5_hgt_slope.data),
        intercept=(["level", "lat", "lon"], IndRCRU_ERA5_hgt_intercept.data),
        rvalue=(["level", "lat", "lon"], IndRCRU_ERA5_hgt_rvalue.data),
        pvalue=(["level", "lat", "lon"], IndRCRU_ERA5_hgt_pvalue.data),
        hypothesis=(["level", "lat", "lon"], IndRCRU_ERA5_hgt_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of ERA5 regress onto 1979-2014 CRU IndR"),
)

IndRCRU_ERA5_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], IndRCRU_ERA5_u_slope.data),
        intercept=(["level", "lat", "lon"], IndRCRU_ERA5_u_intercept.data),
        rvalue=(["level", "lat", "lon"], IndRCRU_ERA5_u_rvalue.data),
        pvalue=(["level", "lat", "lon"], IndRCRU_ERA5_u_pvalue.data),
        hypothesis=(["level", "lat", "lon"], IndRCRU_ERA5_u_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of ERA5 regress onto 1979-2014 CRU IndR"),
)

IndRCRU_ERA5_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], IndRCRU_ERA5_v_slope.data),
        intercept=(["level", "lat", "lon"], IndRCRU_ERA5_v_intercept.data),
        rvalue=(["level", "lat", "lon"], IndRCRU_ERA5_v_rvalue.data),
        pvalue=(["level", "lat", "lon"], IndRCRU_ERA5_v_pvalue.data),
        hypothesis=(["level", "lat", "lon"], IndRCRU_ERA5_v_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of ERA5 regress onto 1979-2014 CRU IndR"),
)

IndRGPCP_ERA5_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], IndRGPCP_ERA5_hgt_slope.data),
        intercept=(["level", "lat", "lon"], IndRGPCP_ERA5_hgt_intercept.data),
        rvalue=(["level", "lat", "lon"], IndRGPCP_ERA5_hgt_rvalue.data),
        pvalue=(["level", "lat", "lon"], IndRGPCP_ERA5_hgt_pvalue.data),
        hypothesis=(["level", "lat", "lon"], IndRGPCP_ERA5_hgt_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of ERA5 regress onto 1979-2014 GPCP IndR"),
)

IndRGPCP_ERA5_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], IndRGPCP_ERA5_u_slope.data),
        intercept=(["level", "lat", "lon"], IndRGPCP_ERA5_u_intercept.data),
        rvalue=(["level", "lat", "lon"], IndRGPCP_ERA5_u_rvalue.data),
        pvalue=(["level", "lat", "lon"], IndRGPCP_ERA5_u_pvalue.data),
        hypothesis=(["level", "lat", "lon"], IndRGPCP_ERA5_u_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of ERA5 regress onto 1979-2014 GPCP IndR"),
)

IndRGPCP_ERA5_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], IndRGPCP_ERA5_v_slope.data),
        intercept=(["level", "lat", "lon"], IndRGPCP_ERA5_v_intercept.data),
        rvalue=(["level", "lat", "lon"], IndRGPCP_ERA5_v_rvalue.data),
        pvalue=(["level", "lat", "lon"], IndRGPCP_ERA5_v_pvalue.data),
        hypothesis=(["level", "lat", "lon"], IndRGPCP_ERA5_v_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of ERA5 regress onto 1979-2014 GPCP IndR"),
)

IndR_his_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IndR_his_hgt_slope.data),
        intercept=(["models", "level", "lat", "lon"], IndR_his_hgt_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IndR_his_hgt_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IndR_his_hgt_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IndR_his_hgt_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of historical regress onto 1979-2014 IndR"),
)

IndR_his_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IndR_his_u_slope.data),
        intercept=(["models", "level", "lat", "lon"], IndR_his_u_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IndR_his_u_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IndR_his_u_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IndR_his_u_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of historical regress onto 1979-2014 IndR"),
)

IndR_his_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IndR_his_v_slope.data),
        intercept=(["models", "level", "lat", "lon"], IndR_his_v_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IndR_his_v_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IndR_his_v_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IndR_his_v_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of historical regress onto 1979-2014 IndR"),
)

IndR_ssp585_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IndR_ssp585_hgt_slope.data),
        intercept=(["models", "level", "lat", "lon"], IndR_ssp585_hgt_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IndR_ssp585_hgt_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IndR_ssp585_hgt_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IndR_ssp585_hgt_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of ssp585 regress onto 2015-2099 IndR"),
)

IndR_ssp585_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IndR_ssp585_u_slope.data),
        intercept=(["models", "level", "lat", "lon"], IndR_ssp585_u_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IndR_ssp585_u_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IndR_ssp585_u_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IndR_ssp585_u_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of ssp585 regress onto 2015-2099 IndR"),
)

IndR_ssp585_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IndR_ssp585_v_slope.data),
        intercept=(["models", "level", "lat", "lon"], IndR_ssp585_v_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IndR_ssp585_v_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IndR_ssp585_v_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IndR_ssp585_v_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of ssp585 regress onto 2015-2099 IndR"),
)

IndR_ssp585_p3_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_slope.data),
        intercept=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of ssp585_p3 regress onto 2064-2099 IndR"),
)

IndR_ssp585_p3_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_slope.data),
        intercept=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of ssp585_p3 regress onto 2064-2099 IndR"),
)

IndR_ssp585_p3_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_slope.data),
        intercept=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of ssp585_p3 regress onto 2064-2099 IndR"),
)
#   output the data
IndRCRU_ERA5_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndRCRU_ERA5_hgt_regress.nc")
IndRCRU_ERA5_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndRCRU_ERA5_u_regress.nc")
IndRCRU_ERA5_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndRCRU_ERA5_v_regress.nc")

IndRGPCP_ERA5_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndRGPCP_ERA5_hgt_regress.nc")
IndRGPCP_ERA5_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndRGPCP_ERA5_u_regress.nc")
IndRGPCP_ERA5_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndRGPCP_ERA5_v_regress.nc")

IndR_his_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndR_his_hgt_regress.nc")
IndR_his_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndR_his_u_regress.nc")
IndR_his_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndR_his_v_regress.nc")

IndR_ssp585_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/IndR_ssp585_hgt_regress.nc")
IndR_ssp585_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/IndR_ssp585_u_regress.nc")
IndR_ssp585_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/IndR_ssp585_v_regress.nc")

IndR_ssp585_p3_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/IndR_ssp585_p3_hgt_regress.nc")
IndR_ssp585_p3_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/IndR_ssp585_p3_u_regress.nc")
IndR_ssp585_p3_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/IndR_ssp585_p3_v_regress.nc")

# %%
IndRCRU_ERA5_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndRCRU_ERA5_hgt_regress.nc")
IndRCRU_ERA5_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndRCRU_ERA5_u_regress.nc")
IndRCRU_ERA5_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndRCRU_ERA5_v_regress.nc")

IndRGPCP_ERA5_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndRGPCP_ERA5_hgt_regress.nc")
IndRGPCP_ERA5_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndRGPCP_ERA5_u_regress.nc")
IndRGPCP_ERA5_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndRGPCP_ERA5_v_regress.nc")

IndR_his_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndR_his_hgt_regress.nc")
IndR_his_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndR_his_u_regress.nc")
IndR_his_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/IndR_his_v_regress.nc")

IndR_ssp585_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/IndR_ssp585_hgt_regress.nc")
IndR_ssp585_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/IndR_ssp585_u_regress.nc")
IndR_ssp585_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/IndR_ssp585_v_regress.nc")

IndR_ssp585_p3_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/IndR_ssp585_p3_hgt_regress.nc")
IndR_ssp585_p3_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/IndR_ssp585_p3_u_regress.nc")
IndR_ssp585_p3_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/IndR_ssp585_p3_v_regress.nc")

IndRCRU_ERA5_hgt_slope = IndRCRU_ERA5_hgt_regress["slope"]
IndRCRU_ERA5_u_slope = IndRCRU_ERA5_u_regress["slope"]
IndRCRU_ERA5_v_slope = IndRCRU_ERA5_v_regress["slope"]
IndRCRU_ERA5_hgt_rvalue = IndRCRU_ERA5_hgt_regress["rvalue"]
IndRCRU_ERA5_u_rvalue = IndRCRU_ERA5_u_regress["rvalue"]
IndRCRU_ERA5_v_rvalue = IndRCRU_ERA5_v_regress["rvalue"]
IndRCRU_ERA5_hgt_pvalue = IndRCRU_ERA5_hgt_regress["pvalue"]
IndRCRU_ERA5_u_pvalue = IndRCRU_ERA5_u_regress["pvalue"]
IndRCRU_ERA5_v_pvalue = IndRCRU_ERA5_v_regress["pvalue"]

IndRGPCP_ERA5_hgt_slope = IndRGPCP_ERA5_hgt_regress["slope"]
IndRGPCP_ERA5_u_slope = IndRGPCP_ERA5_u_regress["slope"]
IndRGPCP_ERA5_v_slope = IndRGPCP_ERA5_v_regress["slope"]
IndRGPCP_ERA5_hgt_rvalue = IndRGPCP_ERA5_hgt_regress["rvalue"]
IndRGPCP_ERA5_u_rvalue = IndRGPCP_ERA5_u_regress["rvalue"]
IndRGPCP_ERA5_v_rvalue = IndRGPCP_ERA5_v_regress["rvalue"]
IndRGPCP_ERA5_hgt_pvalue = IndRGPCP_ERA5_hgt_regress["pvalue"]
IndRGPCP_ERA5_u_pvalue = IndRGPCP_ERA5_u_regress["pvalue"]
IndRGPCP_ERA5_v_pvalue = IndRGPCP_ERA5_v_regress["pvalue"]

IndR_his_hgt_slope = IndR_his_hgt_regress["slope"]
IndR_his_u_slope = IndR_his_u_regress["slope"]
IndR_his_v_slope = IndR_his_v_regress["slope"]
IndR_his_hgt_rvalue = IndR_his_hgt_regress["rvalue"]
IndR_his_u_rvalue = IndR_his_u_regress["rvalue"]
IndR_his_v_rvalue = IndR_his_v_regress["rvalue"]
IndR_his_hgt_pvalue = IndR_his_hgt_regress["pvalue"]
IndR_his_u_pvalue = IndR_his_u_regress["pvalue"]
IndR_his_v_pvalue = IndR_his_v_regress["pvalue"]

IndR_ssp585_hgt_slope = IndR_ssp585_hgt_regress["slope"]
IndR_ssp585_u_slope = IndR_ssp585_u_regress["slope"]
IndR_ssp585_v_slope = IndR_ssp585_v_regress["slope"]
IndR_ssp585_hgt_rvalue = IndR_ssp585_hgt_regress["rvalue"]
IndR_ssp585_u_rvalue = IndR_ssp585_u_regress["rvalue"]
IndR_ssp585_v_rvalue = IndR_ssp585_v_regress["rvalue"]
IndR_ssp585_hgt_pvalue = IndR_ssp585_hgt_regress["pvalue"]
IndR_ssp585_u_pvalue = IndR_ssp585_u_regress["pvalue"]
IndR_ssp585_v_pvalue = IndR_ssp585_v_regress["pvalue"]
# %%
#   plot the avalue of hgt&u&v regress onto IndR in ERA5 and historical
startlevel=[-15, -8, -6]
endlevel=[15, 8, 6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 31), (6, 5))
    plot_array[-1,-1] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndRCRU_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndRCRU_ERA5_hgt_slope.sel(level=lev), axs[0], n, np.where(IndRCRU_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndRCRU_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
        IndRCRU_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IndRCRU_ERA5_u_slope.sel(level=lev).where(IndRCRU_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRCRU_ERA5_v_slope.sel(level=lev).where(IndRCRU_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="CRU & ERA5",
    )
    # ======================================
    con = axs[1].contourf(
        IndRGPCP_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndRGPCP_ERA5_hgt_slope.sel(level=lev), axs[1], n, np.where(IndRGPCP_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[1].quiver(
        IndRGPCP_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
        IndRGPCP_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[1].quiver(
        IndRGPCP_ERA5_u_slope.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRGPCP_ERA5_v_slope.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="GPCP & ERA5",
    )
    # ======================================
    con = axs[2].contourf(
        IndR_his_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_ens.sel(level=lev), axs[2], n, np.where(IndR_his_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[2].quiver(
        IndR_his_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_slope_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[2].quiver(
        IndR_his_u_slope_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_slope_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[2].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[2].format(
        rtitle="1979-2014", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+3].contourf(
            IndR_his_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IndR_his_hgt_slope.sel(models=mod,level=lev), axs[num_mod+3], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+3].quiver(
            IndR_his_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_his_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+3].quiver(
            IndR_his_u_slope.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_his_v_slope.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+3].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+3].format(
            rtitle="1979-2014", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))

# %%
#   plot the rvalue of hgt&u&v regress onto IndR in ERA5 and historical
startlevel=[-1.0, -1.0, -1.0]
endlevel=[1.0, 1.0, 1.0]
spacinglevel=[0.05, 0.05, 0.05]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 31), (6, 5))
    plot_array[-1,-1] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndRCRU_ERA5_hgt_rvalue.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndRCRU_ERA5_hgt_rvalue.sel(level=lev), axs[0], n, np.where(IndRCRU_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndRCRU_ERA5_u_rvalue.sel(level=lev)[::ski, ::ski],
        IndRCRU_ERA5_v_rvalue.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IndRCRU_ERA5_u_rvalue.sel(level=lev).where(IndRCRU_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRCRU_ERA5_v_rvalue.sel(level=lev).where(IndRCRU_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="CRU & ERA5",
    )
    # ======================================
    con = axs[1].contourf(
        IndRGPCP_ERA5_hgt_rvalue.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndRGPCP_ERA5_hgt_rvalue.sel(level=lev), axs[1], n, np.where(IndRGPCP_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[1].quiver(
        IndRGPCP_ERA5_u_rvalue.sel(level=lev)[::ski, ::ski],
        IndRGPCP_ERA5_v_rvalue.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[1].quiver(
        IndRGPCP_ERA5_u_rvalue.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRGPCP_ERA5_v_rvalue.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="GPCP & ERA5",
    )
    # ======================================
    con = axs[2].contourf(
        IndR_his_hgt_rvalue_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_his_hgt_rvalue_ens.sel(level=lev), axs[2], n, np.where(IndR_his_hgt_rvalue_ens_mask.sel(level=lev)[::n, ::n] > 0.0), "bright purple", 3.0,
    )
    axs[2].quiver(
        IndR_his_u_rvalue_ens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_rvalue_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[2].quiver(
        IndR_his_u_rvalue_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_rvalue_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[2].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[2].format(
        rtitle="1979-2014", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+3].contourf(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
        )
        sepl.plt_sig(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod+3], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+3].quiver(
            IndR_his_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_his_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+3].quiver(
            IndR_his_u_rvalue.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_his_v_rvalue.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+3].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+3].format(
            rtitle="1979-2014", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))

# %%
#   plot the avalue of hgt&u&v regress onto IndR in ssp585
startlevel=[-15, -8, -6]
endlevel=[15, 8, 6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 29), (7, 4))
    plot_array[-1,-1] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)    
    # ======================================
    con = axs[0].contourf(
        IndR_ssp585_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_ssp585_hgt_slope_ens.sel(level=lev), axs[0], n, np.where(IndR_ssp585_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndR_ssp585_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IndR_ssp585_v_slope_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IndR_ssp585_u_slope_ens.sel(level=lev).where(IndR_ssp585_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_ssp585_v_slope_ens.sel(level=lev).where(IndR_ssp585_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="2015-2099", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+1].contourf(
            IndR_ssp585_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IndR_ssp585_hgt_slope.sel(models=mod,level=lev), axs[num_mod+1], n, np.where(IndR_ssp585_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+1].quiver(
            IndR_ssp585_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_ssp585_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+1].quiver(
            IndR_ssp585_u_slope.sel(models=mod,level=lev).where(IndR_ssp585_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_ssp585_v_slope.sel(models=mod,level=lev).where(IndR_ssp585_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
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
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="2015-2099", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))

# %%
#   plot the rvalue of hgt&u&v regress onto IndR in ssp585
startlevel=[-1.0, -1.0, -1.0]
endlevel=[1.0, 1.0, 1.0]
spacinglevel=[0.05, 0.05, 0.05]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 29), (7, 4))
    plot_array[-1,-1] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)    
    # ======================================
    con = axs[0].contourf(
        IndR_ssp585_hgt_rvalue_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_ssp585_hgt_rvalue_ens.sel(level=lev), axs[0], n, np.where(IndR_ssp585_hgt_rvalue_ens_mask.sel(level=lev)[::n, ::n] > 0.0), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndR_ssp585_u_rvalue_ens.sel(level=lev)[::ski, ::ski],
        IndR_ssp585_v_rvalue_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IndR_ssp585_u_rvalue_ens.sel(level=lev).where(IndR_ssp585_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_ssp585_v_rvalue_ens.sel(level=lev).where(IndR_ssp585_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="2015-2099", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+1].contourf(
            IndR_ssp585_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
        )
        sepl.plt_sig(
            IndR_ssp585_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod+1], n, np.where(IndR_ssp585_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+1].quiver(
            IndR_ssp585_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_ssp585_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+1].quiver(
            IndR_ssp585_u_rvalue.sel(models=mod,level=lev).where(IndR_ssp585_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_ssp585_v_rvalue.sel(models=mod,level=lev).where(IndR_ssp585_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
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
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="2015-2099", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the avalue of hgt&u&v regress onto IndR in ssp585_p3
startlevel=[-15, -8, -6]
endlevel=[15, 8, 6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 29), (7, 4))
    plot_array[-1,-1] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)    
    # ======================================
    con = axs[0].contourf(
        IndR_ssp585_p3_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_ssp585_p3_hgt_slope_ens.sel(level=lev), axs[0], n, np.where(IndR_ssp585_p3_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.0), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndR_ssp585_p3_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IndR_ssp585_p3_v_slope_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IndR_ssp585_p3_u_slope_ens.sel(level=lev).where(IndR_ssp585_p3_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_ssp585_p3_v_slope_ens.sel(level=lev).where(IndR_ssp585_p3_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="2064-2099", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+1].contourf(
            IndR_ssp585_p3_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IndR_ssp585_p3_hgt_slope.sel(models=mod,level=lev), axs[num_mod+1], n, np.where(IndR_ssp585_p3_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+1].quiver(
            IndR_ssp585_p3_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_ssp585_p3_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+1].quiver(
            IndR_ssp585_p3_u_slope.sel(models=mod,level=lev).where(IndR_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_ssp585_p3_v_slope.sel(models=mod,level=lev).where(IndR_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
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
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="2064-2099", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))

# %%
#   plot the rvalue of hgt&u&v regress onto IndR in ssp585_p3
startlevel=[-1.0, -1.0, -1.0]
endlevel=[1.0, 1.0, 1.0]
spacinglevel=[0.05, 0.05, 0.05]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 29), (7, 4))
    plot_array[-1,-1] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)    
    # ======================================
    con = axs[0].contourf(
        IndR_ssp585_p3_hgt_rvalue_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_ssp585_p3_hgt_rvalue_ens.sel(level=lev), axs[0], n, np.where(IndR_ssp585_p3_hgt_rvalue_ens_mask.sel(level=lev)[::n, ::n] > 0.0), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndR_ssp585_p3_u_rvalue_ens.sel(level=lev)[::ski, ::ski],
        IndR_ssp585_p3_v_rvalue_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IndR_ssp585_p3_u_rvalue_ens.sel(level=lev).where(IndR_ssp585_p3_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_ssp585_p3_v_rvalue_ens.sel(level=lev).where(IndR_ssp585_p3_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="2064-2099", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+1].contourf(
            IndR_ssp585_p3_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
        )
        sepl.plt_sig(
            IndR_ssp585_p3_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod+1], n, np.where(IndR_ssp585_p3_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+1].quiver(
            IndR_ssp585_p3_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_ssp585_p3_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+1].quiver(
            IndR_ssp585_p3_u_rvalue.sel(models=mod,level=lev).where(IndR_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_ssp585_p3_v_rvalue.sel(models=mod,level=lev).where(IndR_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
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
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="2064-2099", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   calculate the diff between ssp585_p3 and historical
IndR_diff_hgt_slope = IndR_ssp585_p3_hgt_slope-IndR_his_hgt_slope
IndR_diff_hgt_slope_ens = IndR_diff_hgt_slope.mean(dim="models", skipna=True)

IndR_diff_u_slope = IndR_ssp585_p3_u_slope-IndR_his_u_slope
IndR_diff_u_slope_ens = IndR_diff_u_slope.mean(dim="models", skipna=True)

IndR_diff_v_slope = IndR_ssp585_p3_v_slope-IndR_his_v_slope
IndR_diff_v_slope_ens = IndR_diff_v_slope.mean(dim="models", skipna=True)

IndR_diff_hgt_rvalue = IndR_ssp585_p3_hgt_rvalue-IndR_his_hgt_rvalue
IndR_diff_hgt_rvalue_ens = IndR_diff_hgt_rvalue.mean(dim="models", skipna=True)

IndR_diff_u_rvalue = IndR_ssp585_p3_u_rvalue-IndR_his_u_rvalue
IndR_diff_u_rvalue_ens = IndR_diff_u_rvalue.mean(dim="models", skipna=True)

IndR_diff_v_rvalue = IndR_ssp585_p3_v_rvalue-IndR_his_v_rvalue
IndR_diff_v_rvalue_ens = IndR_diff_v_rvalue.mean(dim="models", skipna=True)
# %%
#   plot the avalue of hgt&u&v regress onto IndR in diff
startlevel=[-22, -15, -10]
endlevel=[22, 15, 10]
spacinglevel=[1.1, 0.75, 0.5]
scalelevel=[0.14, 0.13, 0.13]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 29), (7, 4))
    plot_array[-1,-1] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)    
    # ======================================
    con = axs[0].contourf(
        IndR_diff_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.05},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )

    m = axs[0].quiver(
        IndR_diff_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IndR_diff_v_slope_ens.sel(level=lev)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="diff", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+1].contourf(
            IndR_diff_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.05},
            levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )

        m = axs[num_mod+1].quiver(
            IndR_diff_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_diff_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
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
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="diff", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))

# %%
#   plot the rvalue of hgt&u&v regress onto IndR in ssp585_p3
startlevel=[-1.0, -1.0, -1.0]
endlevel=[1.0, 1.0, 1.0]
spacinglevel=[0.05, 0.05, 0.05]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 29), (7, 4))
    plot_array[-1,-1] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)    
    # ======================================
    con = axs[0].contourf(
        IndR_diff_hgt_rvalue_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )

    m = axs[0].quiver(
        IndR_diff_u_rvalue_ens.sel(level=lev)[::ski, ::ski],
        IndR_diff_v_rvalue_ens.sel(level=lev)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="diff", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+1].contourf(
            IndR_diff_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        
        m = axs[num_mod+1].quiver(
            IndR_diff_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_diff_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
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
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="diff", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))

# %%
#   calculate the pcc and sort to reveal the rank of different models
# models = pre_his_India_pre_slope.coords["models"]
lat = prehis_JJA.coords["lat"]
lon = prehis_JJA.coords["lon"]
lat_EAM_range = lat[(lat>=22.5) & (lat<=40.0)]
lon_EAM_range = lon[(lon>=90.0) & (lon<=135.0)]

IndR_EAM_list = []
IndR_EAM_pcc = []
IndR_EAM_RMSE = []
IndR_EAM_std = []

for num_mod, mod in enumerate(models):
    IndR_EAM_list.append({"models": mod.data, "pcc": ca.cal_pcc(pre_GPCP_India_pre_rvalue.loc[22.5:40.0, 90.0:135.0], pre_his_India_pre_rvalue.sel(models=mod, lat=lat_EAM_range, lon=lon_EAM_range))})
    IndR_EAM_pcc.append(ca.cal_pcc(pre_GPCP_India_pre_rvalue.loc[22.5:40.0, 90.0:135.0], pre_his_India_pre_rvalue.sel(models=mod, lat=lat_EAM_range, lon=lon_EAM_range)))
    IndR_EAM_RMSE.append(np.sqrt(np.power((pre_his_India_pre_rvalue.sel(models=mod,lat=lat_EAM_range,lon=lon_EAM_range)-pre_GPCP_India_pre_rvalue.loc[22.5:40.0, 90.0:135.0]),2).mean(dim=["lat","lon"],skipna=True).data))
    IndR_EAM_std.append(float((pre_his_India_pre_rvalue.sel(models=mod,lat=lat_EAM_range,lon=lon_EAM_range).std(dim=["lat","lon"],skipna=True)/pre_GPCP_India_pre_rvalue.loc[22.5:40.0, 90.0:135.0].std(dim=["lat","lon"],skipna=True)).data))
#   for MME
IndR_EAM_pcc.append(ca.cal_pcc(pre_GPCP_India_pre_rvalue.loc[22.5:40.0, 90.0:135.0], pre_his_India_pre_rvalue_ens.sel(lat=lat_EAM_range, lon=lon_EAM_range)))
IndR_EAM_RMSE.append(np.sqrt(np.power((pre_his_India_pre_rvalue_ens.sel(lat=lat_EAM_range,lon=lon_EAM_range)-pre_GPCP_India_pre_rvalue.loc[22.5:40.0, 90.0:135.0]),2).mean(dim=["lat","lon"],skipna=True).data))
IndR_EAM_std.append(float((pre_his_India_pre_rvalue_ens.sel(lat=lat_EAM_range,lon=lon_EAM_range).std(dim=["lat","lon"],skipna=True)/pre_GPCP_India_pre_rvalue.loc[22.5:40.0, 90.0:135.0].std(dim=["lat","lon"],skipna=True)).data))

print(sorted(IndR_EAM_list, key=lambda x : x["pcc"]))
# %%
#   plot the Taylor plot 
labels = list(models.data)
labels.append("MME")
# plt.rcParams["figure.figsize"] = [6, 6]
# plt.rcParams["figure.facecolor"] = "white"
# plt.rcParams["figure.edgecolor"] = "white"
# plt.rcParams["figure.dpi"] = 80
# plt.rcParams['lines.linewidth'] = 1 # 
# plt.rcParams.update({'font.size': 12}) # 
# plt.close('all')
# # 开始绘图
# text_font = {'size':'10','weight':'bold','color':'black'}
# sm.taylor_diagram(np.array(IndR_EAM_std),np.array(IndR_EAM_RMSE),np.array(IndR_EAM_pcc), markerLabel=labels, markerLegend = 'on', colRMS='m', axismax=1.5, markerSize=5, titleRMS='off', titleRMSDangle=90.0, titleCOR='off', tickCOR=np.arange(-1.0, 1.1, 0.2),alpha=0.0,titleSTD='off', markerLegend)
# plt.title("pre reg IndR",fontdict=text_font,pad=35)
def tar(ax,r,std, dotlabels):
    ax.set_thetalim(thetamin=0, thetamax=180)
    r_small, r_big, r_interval=0,1.5+0.1,0.5  #横纵坐标范围，最小值 最大值 间隔
    ax.set_rlim(r_small,r_big)
    rad_list=[-1,-0.99,-0.95,-0.9,-0.8,-0.7,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.7,0.8,0.9,0.95,0.99,1] #需要显示数值的主要R的值
    # minor_rad_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.86,0.87,0.88,0.89,
                    # 0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1] #需要显示刻度的次要R的值
    angle_list = np.rad2deg(np.arccos(rad_list))
    angle_list_rad=np.arccos(rad_list)
    angle_minor_list = np.arccos(minor_rad_list)
    ax.set_thetagrids(angle_list, rad_list)
    ax.tick_params(pad=8)
    # lines, labels = plt.thetagrids(angle_list, labels=rad_list, frac=1.25)
    v = 0.11
    for i in np.arange(r_small, r_big, r_interval):
        if i == 1:
            ax.text(np.arctan(i/v)-0.5*np.pi, np.sqrt(v**2+i**2), s='REF', ha='center', va='top') 
            ax.text(1.5*np.pi-np.arctan(i/v), np.sqrt(v**2+i**2), s='REF', ha='center', va='top')
            #text的第一个坐标是角度（弧度制），第二个是距离
        elif i == 0:
            ax.text(1.5*np.pi, v, s=str(i), ha='center', va='top')
        else: 
            ax.text(np.arctan(i/v)-0.5*np.pi, np.sqrt(v**2+i**2), s=str(i), ha='center', va='top') 
            ax.text(1.5*np.pi-np.arctan(i/v), np.sqrt(v**2+i**2), s=str(i), ha='center', va='top') 
    ax.set_rgrids([])
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    ax.grid(False)
    angle_linewidth,angle_length,angle_minor_length=0.8,0.02,0.01
    tick = [ax.get_rmax(), ax.get_rmax() * (1 - angle_length)]
    # tick_minor = [ax.get_rmax(), ax.get_rmax() * (1 - angle_minor_length)]
    for t in angle_list_rad:
        ax.plot([t, t], tick, lw=angle_linewidth, color="k")  # 第一个坐标是角度（角度制），第二个是距离
    # for t in angle_minor_list:
    #     ax.plot([t, t], tick_minor, lw=angle_linewidth, color="k")  # 第一个坐标是角度（角度制），第二个是距离

    # 然后开始绘制以REF为原点的圈，可以自己加圈
    circle = plt.Circle((1, 0), 0.5, transform=ax.transData._b, facecolor=(0, 0, 0, 0), edgecolor='gray',linestyle='--', linewidth=0.8)
    ax.add_artist(circle)
    circle = plt.Circle((1, 0), 1.0, transform=ax.transData._b, facecolor=(0, 0, 0, 0), edgecolor='gray',linestyle='--', linewidth=0.8)
    ax.add_artist(circle)

    # 绘制以原点为圆点的圆圈：
    circle4 = plt.Circle((0, 0), 0.5, transform=ax.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',linestyle='--', linewidth=1.0)
    circle5 = plt.Circle((0, 0), 1, transform=ax.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',linestyle='-', linewidth=1.5)
    circle6 = plt.Circle((0, 0), 1.5, transform=ax.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',linestyle='--', linewidth=1.0)
    ax.add_artist(circle4)
    ax.add_artist(circle5)
    ax.add_artist(circle6)

    #ax.set_xlabel('Normalized')
    ax.text(np.deg2rad(40), 1.85, s='Correlation', ha='center', va='bottom', rotation=-45)  

    # 这里的网格包括：以原点为圆点的圆圈。首先绘制从原点发散的线段，长度等于半径
    ax.plot([0,np.arccos(0.4)],[0,3],lw=1,color='gray',linestyle='--')
    ax.plot([0,np.arccos(0.8)],[0,3],lw=1,color='gray',linestyle='--')
    ax.plot([0,np.arccos(0.0)],[0,3],lw=1,color='gray',linestyle='--')
    ax.plot([0,np.arccos(-0.4)],[0,3],lw=1,color='gray',linestyle='--')
    ax.plot([0,np.arccos(-0.8)],[0,3],lw=1,color='gray',linestyle='--')

    # 画点，参数一相关系数，参数二标准差 
    for i in np.arange(0,len(r)):
        ax.plot(np.arccos(r[i]), std[i], 'o',color='r',markersize=0, label='{} {}'.format(i+1,dotlabels[i]))
        ax.text(np.arccos(r[i]), std[i], s='{}'.format(i+1), c='r',fontsize=10)
        # ax.text(np.arccos(r[i]-0.05), std[i], s='2', c='r',fontsize=13)
        # ax.plot(np.arccos(r[0]), std[0], 'o',color='#FF8000',markersize=10, label='1')
        # ax.text(np.arccos(r[0]-0.05), std[0], s='1', c='#FF8000',fontsize=13)
    # ax.plot(np.arccos(r[2]), std[2], 'o',color='g',markersize=10, label='3')
    # ax.text(np.arccos(r[2]-0.05), std[2], s='3',c='g', fontsize=13)
    # ax.plot(np.arccos(r[3]), std[3], 'o',color='b',markersize=10, label='4')
    # ax.text(np.arccos(r[3]-0.05), std[3], s='4', c='b',fontsize=13)
    # ax.plot(np.arccos(r[4]), std[4], 'o',color='#E800E8',markersize=10, label='5')
    # ax.text(np.arccos(r[4]-0.05), std[4], s='5', c='#E800E8',fontsize=13)
    # ax.plot(np.arccos(r[5]), std[5], '^',color='#00AEAE',markersize=10, label='6')
    # ax.text(np.arccos(r[5]-0.05), std[5], s='6', c='#00AEAE',fontsize=13)
    ax.text(1.5*np.pi, 0.3, s='Std (Normalized)',ha='center', va='top')
    # ax.text(1.5np.pi,'Std (Normalized)',labelpad=0)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#绘图
fig=plt.figure(figsize=(12,8),dpi=300)
plt.rc('font',family='Arial',size=13)

#调用函数
ax1=fig.add_subplot(111,projection='polar')
box = ax1.get_position()
ax1.set_position([0, box.y0, box.width*1.2, box.height])
# ax1.text(0.6,0.1,'(a)',fontsize=15)
# tar(ax1,np.array(IndR_EAM_pcc),np.array(IndR_EAM_std),labels)
sepl.taylor_diagram(ax1,np.array(IndR_EAM_pcc),np.array(IndR_EAM_std),labels)
plt.legend(loc="center left", bbox_to_anchor=(1.1,0.5), ncol=2, frameon=True, numpoints=1, handlelength=0)
# plt.tight_layout()
# %%
#   pick up the good models
gmodels = ['MPI-ESM1-2-HR', 'EC-Earth3-Veg', 'UKESM1-0-LL', 'EC-Earth3', 'CMCC-ESM2', 'MRI-ESM2-0', 'HadGEM3-GC31-LL', 'TaiESM1', 'NorESM2-LM', 'MIROC-ES2L']

#   recalculate the MME of these gmodels
pre_his_India_pre_slope_gmodels = pre_his_India_pre_slope.sel(models=gmodels)

pre_ssp585_India_pre_slope_gmodels = pre_ssp585_India_pre_slope.sel(models=gmodels)

pre_ssp585_p3_India_pre_slope_gmodels = pre_ssp585_p3_India_pre_slope.sel(models=gmodels)

pre_diff_India_pre_slope_gmodels = pre_diff_India_pre_slope.sel(models=gmodels)

pre_his_India_pre_slope_gmodels_ens = pre_his_India_pre_slope.sel(models=gmodels).mean(dim="models", skipna=True)

pre_ssp585_India_pre_slope_gmodels_ens = pre_ssp585_India_pre_slope.sel(models=gmodels).mean(dim="models", skipna=True)

pre_ssp585_p3_India_pre_slope_gmodels_ens = pre_ssp585_p3_India_pre_slope.sel(models=gmodels).mean(dim="models", skipna=True)

pre_diff_India_pre_slope_gmodels_ens = pre_diff_India_pre_slope_gmodels.mean(dim="models", skipna=True)

pre_his_India_pre_slope_gmodels_ens_mask = ca.MME_reg_mask(pre_his_India_pre_slope_gmodels_ens, pre_his_India_pre_slope.sel(models=gmodels).std(dim="models", skipna=True), len(pre_his_India_pre_slope.sel(models=gmodels).coords["models"]), True)

pre_ssp585_India_pre_slope_gmodels_ens_mask = ca.MME_reg_mask(pre_ssp585_India_pre_slope_gmodels_ens, pre_ssp585_India_pre_slope.sel(models=gmodels).std(dim="models", skipna=True), len(pre_ssp585_India_pre_slope.sel(models=gmodels).coords["models"]), True)

pre_ssp585_p3_India_pre_slope_gmodels_ens_mask = ca.MME_reg_mask(pre_ssp585_p3_India_pre_slope_gmodels_ens, pre_ssp585_p3_India_pre_slope.sel(models=gmodels).std(dim="models", skipna=True), len(pre_ssp585_p3_India_pre_slope.sel(models=gmodels).coords["models"]), True)

pre_diff_India_pre_slope_gmodels_ens_mask = ca.MME_reg_mask(pre_diff_India_pre_slope_gmodels_ens, pre_diff_India_pre_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)


pre_his_India_pre_rvalue_gmodels = pre_his_India_pre_rvalue.sel(models=gmodels)

pre_ssp585_India_pre_rvalue_gmodels = pre_ssp585_India_pre_rvalue.sel(models=gmodels)

pre_ssp585_p3_India_pre_rvalue_gmodels = pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels)

pre_diff_India_pre_rvalue_gmodels = pre_diff_India_pre_rvalue.sel(models=gmodels)

pre_his_India_pre_rvalue_gmodels_ens = pre_his_India_pre_rvalue.sel(models=gmodels).mean(dim="models", skipna=True)

pre_ssp585_India_pre_rvalue_gmodels_ens = pre_ssp585_India_pre_rvalue.sel(models=gmodels).mean(dim="models", skipna=True)

pre_ssp585_p3_India_pre_rvalue_gmodels_ens = pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels).mean(dim="models", skipna=True)

pre_diff_India_pre_rvalue_gmodels_ens = pre_diff_India_pre_rvalue_gmodels.mean(dim="models", skipna=True)

pre_his_India_pre_rvalue_gmodels_ens_mask = ca.MME_reg_mask(pre_his_India_pre_rvalue_gmodels_ens, pre_his_India_pre_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(pre_his_India_pre_rvalue.sel(models=gmodels).coords["models"]), True)

pre_ssp585_India_pre_rvalue_gmodels_ens_mask = ca.MME_reg_mask(pre_ssp585_India_pre_rvalue_gmodels_ens, pre_ssp585_India_pre_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(pre_ssp585_India_pre_rvalue.sel(models=gmodels).coords["models"]), True)

pre_ssp585_p3_India_pre_rvalue_gmodels_ens_mask = ca.MME_reg_mask(pre_ssp585_p3_India_pre_rvalue_gmodels_ens, pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels).coords["models"]), True)

pre_diff_India_pre_rvalue_gmodels_ens_mask = ca.MME_reg_mask(pre_diff_India_pre_rvalue_gmodels_ens, pre_diff_India_pre_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)
# %%
#   plot the regression coefficients in CRU, GPCP, historical run for good models
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 17), (4, 4))
plot_array[-1,-3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_CRU_India_pre_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_CRU_India_pre_slope, axs[0], n, np.where(pre_CRU_India_pre_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="CRU",
)
# ===================================================
con = axs[1].contourf(
    pre_GPCP_India_pre_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_GPCP_India_pre_slope, axs[1], n, np.where(pre_GPCP_India_pre_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="GPCP",
)
# ===================================================
con = axs[2].contourf(
    pre_his_India_pre_slope_gmodels_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_his_India_pre_slope_gmodels_ens, axs[2], n, np.where(pre_his_India_pre_slope_gmodels_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[2].format(
    rtitle="1979-2014", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+3].contourf(
    pre_his_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
    sepl.plt_sig(
        pre_his_India_pre_slope.sel(models=mod), axs[num_models+3], n, np.where(pre_his_India_pre_pvalue.sel(models=mod)[::n, ::n] > 0.0), "bright purple", 4.0,
    )

    axs[num_models+3].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")

# %%
#   plot the correlation coefficients rvalue in CRU, GPCP, historical run for good models
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 17), (4, 4))
plot_array[-1,-3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_CRU_India_pre_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_CRU_India_pre_rvalue, axs[0], n, np.where(pre_CRU_India_pre_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="CRU",
)
# ===================================================
con = axs[1].contourf(
    pre_GPCP_India_pre_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_GPCP_India_pre_rvalue, axs[1], n, np.where(pre_GPCP_India_pre_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="GPCP",
)
# # ===================================================
con = axs[2].contourf(
    pre_his_India_pre_rvalue_gmodels_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    
    )
sepl.plt_sig(
    pre_his_India_pre_rvalue_gmodels_ens, axs[2], n, np.where(pre_his_India_pre_rvalue_gmodels_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[2].format(
    rtitle="1979-2014", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+3].contourf(
    pre_his_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
    sepl.plt_sig(
        pre_his_India_pre_rvalue.sel(models=mod), axs[num_models+3], n, np.where(pre_his_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 4.0,
    )

    axs[num_models+3].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the regression coefficients avalue in ssp585 run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 13), (4, 3))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_ssp585_India_pre_slope_gmodels_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_ssp585_India_pre_slope_gmodels_ens, axs[0], n, np.where(pre_ssp585_India_pre_slope_gmodels_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[0].format(
    rtitle="2015-2099", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+1].contourf(
    pre_ssp585_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
    sepl.plt_sig(
        pre_ssp585_India_pre_slope.sel(models=mod), axs[num_models+1], n, np.where(pre_ssp585_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 4.0,
    )

    axs[num_models+1].format(
        rtitle="2015-2099", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the correlation coefficients rvalue in ssp585 run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 13), (4, 3))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_ssp585_India_pre_rvalue_gmodels_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_ssp585_India_pre_rvalue_gmodels_ens, axs[0], n, np.where(pre_ssp585_India_pre_rvalue_gmodels_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[0].format(
    rtitle="2015-2099", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+1].contourf(
    pre_ssp585_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
    sepl.plt_sig(
        pre_ssp585_India_pre_rvalue.sel(models=mod), axs[num_models+1], n, np.where(pre_ssp585_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 4.0,
    )

    axs[num_models+1].format(
        rtitle="2015-2099", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the regression coefficients avalue in ssp585 p3 run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 13), (4, 3))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_ssp585_p3_India_pre_slope_gmodels_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_ssp585_p3_India_pre_slope_gmodels_ens, axs[0], n, np.where(pre_ssp585_p3_India_pre_slope_gmodels_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[0].format(
    rtitle="2064-2099", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+1].contourf(
    pre_ssp585_p3_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
    sepl.plt_sig(
        pre_ssp585_p3_India_pre_slope.sel(models=mod), axs[num_models+1], n, np.where(pre_ssp585_p3_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 4.0,
    )

    axs[num_models+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the correlation coefficients rvalue in ssp585 p3 run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 13), (4, 3))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_ssp585_p3_India_pre_rvalue_gmodels_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_ssp585_p3_India_pre_rvalue_gmodels_ens, axs[0], n, np.where(pre_ssp585_p3_India_pre_rvalue_gmodels_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[0].format(
    rtitle="2064-2099", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+1].contourf(
    pre_ssp585_p3_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
    sepl.plt_sig(
        pre_ssp585_p3_India_pre_rvalue.sel(models=mod), axs[num_models+1], n, np.where(pre_ssp585_p3_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 4.0,
    )

    axs[num_models+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the regression coefficients avalue in ssp585 p3-historical
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 13), (4, 3))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_diff_India_pre_slope_gmodels_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )

axs[0].format(
    rtitle="diff", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+1].contourf(
    pre_diff_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )

    axs[num_models+1].format(
        rtitle="diff", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the correlation coefficients rvalue in ssp585 p3-historical
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 13), (4, 3))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_diff_India_pre_rvalue_gmodels_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )

axs[0].format(
    rtitle="diff", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+1].contourf(
    pre_diff_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )

    axs[num_models+1].format(
        rtitle="diff", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
IndR_diff_hgt_slope_gmodels = (IndR_ssp585_p3_hgt_slope-IndR_his_hgt_slope).sel(models=gmodels)
IndR_diff_hgt_slope_gmodels_ens = IndR_diff_hgt_slope_gmodels.mean(dim="models", skipna=True)

IndR_diff_u_slope_gmodels = (IndR_ssp585_p3_u_slope-IndR_his_u_slope).sel(models=gmodels)
IndR_diff_u_slope_gmodels_ens = IndR_diff_u_slope_gmodels.mean(dim="models", skipna=True)

IndR_diff_v_slope_gmodels = (IndR_ssp585_p3_v_slope-IndR_his_v_slope).sel(models=gmodels)
IndR_diff_v_slope_gmodels_ens = IndR_diff_v_slope_gmodels.mean(dim="models", skipna=True)

IndR_diff_hgt_rvalue_gmodels = (IndR_ssp585_p3_hgt_rvalue-IndR_his_hgt_rvalue).sel(models=gmodels)
IndR_diff_hgt_rvalue_gmodels_ens = IndR_diff_hgt_rvalue_gmodels.mean(dim="models", skipna=True)

IndR_diff_u_rvalue_gmodels = (IndR_ssp585_p3_u_rvalue-IndR_his_u_rvalue).sel(models=gmodels)
IndR_diff_u_rvalue_gmodels_ens = IndR_diff_u_rvalue_gmodels.mean(dim="models", skipna=True)

IndR_diff_v_rvalue_gmodels = (IndR_ssp585_p3_v_rvalue-IndR_his_v_rvalue).sel(models=gmodels)
IndR_diff_v_rvalue_gmodels_ens = IndR_diff_v_rvalue_gmodels.mean(dim="models", skipna=True)
# %%
#   plot the avalue of hgt&u&v regress onto IndR in diff
startlevel=[-22, -15, -10]
endlevel=[22, 15, 10]
spacinglevel=[1.1, 0.75, 0.5]
scalelevel=[0.14, 0.13, 0.13]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 13), (4, 3))
    plot_array[-1,-1] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)    
    # ======================================
    con = axs[0].contourf(
        IndR_diff_hgt_slope_gmodels_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.05},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )

    m = axs[0].quiver(
        IndR_diff_u_slope_gmodels_ens.sel(level=lev)[::ski, ::ski],
        IndR_diff_v_slope_gmodels_ens.sel(level=lev)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="diff", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels):
        con = axs[num_mod+1].contourf(
            IndR_diff_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.05},
            levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )

        m = axs[num_mod+1].quiver(
            IndR_diff_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_diff_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
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
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="diff", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the rvalue of hgt&u&v regress onto IndR in diff
startlevel=[-1, -1, -1]
endlevel=[1, 1, 1]
spacinglevel=[0.1, 0.1, 0.1]
scalelevel=[0.14, 0.13, 0.13]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 13), (4, 3))
    plot_array[-1,-1] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)    
    # ======================================
    con = axs[0].contourf(
        IndR_diff_hgt_rvalue_gmodels_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.05},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )

    m = axs[0].quiver(
        IndR_diff_u_rvalue_gmodels_ens.sel(level=lev)[::ski, ::ski],
        IndR_diff_v_rvalue_gmodels_ens.sel(level=lev)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="diff", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels):
        con = axs[num_mod+1].contourf(
            IndR_diff_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.05},
            levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )

        m = axs[num_mod+1].quiver(
            IndR_diff_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_diff_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
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
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="diff", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   calculate the BOB precipitation
lat = preCRU_JJA.coords["lat"]
lon = preCRU_JJA.coords["lon"]
lat_BOB_range = lat[(lat >= 10.0) & (lat <= 25.0)]
lon_BOB_range = lon[(lon >= 80.0) & (lon <= 100.0)]

preCRU_BOB_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_BOB_range, lon=lon_BOB_range)).mean(dim="lon", skipna=True)
lat = preGPCP_JJA.coords["lat"]
lon = preGPCP_JJA.coords["lon"]
lat_BOB_range = lat[(lat >= 10.0) & (lat <= 25.0)]
lon_BOB_range = lon[(lon >= 80.0) & (lon <= 100.0)]
# lat_BOB_range = lat[(lat >= 15.0) & (lat <= 28.0)]
# lat_BOB_range = lat[(lat >= 30.0) & (lat <= 35.0)]
# lon_BOB_range = lon[(lon >= 112.5) & (lon <= 120.0)]

preGPCP_BOB_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_BOB_range, lon=lon_BOB_range)).mean(dim="lon", skipna=True)

lat = prehis_JJA.coords["lat"]
lon = prehis_JJA.coords["lon"]

lat_BOB_range = lat[(lat >= 10.0) & (lat <= 25.0)]
# lat_BOB_range = lat[(lat >= 15.0) & (lat <= 28.0)]
lon_BOB_range = lon[(lon >= 80.0) & (lon <= 100.0)]
# lat_BOB_range = lat[(lat >= 30.0) & (lat <= 35.0)]
# lon_BOB_range = lon[(lon >= 112.5) & (lon <= 120.0)]

prehis_BOB_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_BOB_range, lon=lon_BOB_range)).mean(dim="lon", skipna=True)
pressp585_BOB_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_BOB_range, lon=lon_BOB_range)).mean(dim="lon", skipna=True)

# %%
#   calculate the precipitation regress onto BOB precipitation
(
    pre_CRU_BOB_pre_slope,
    pre_CRU_BOB_pre_intercept,
    pre_CRU_BOB_pre_rvalue,
    pre_CRU_BOB_pre_pvalue,
    pre_CRU_BOB_pre_hypothesis,
) = ca.dim_linregress(preCRU_BOB_JJA.sel(time=preCRU_BOB_JJA.time.dt.year>=1979), preCRU_JJA.sel(time=preCRU_JJA.time.dt.year>=1979))

(
    pre_GPCP_BOB_pre_slope,
    pre_GPCP_BOB_pre_intercept,
    pre_GPCP_BOB_pre_rvalue,
    pre_GPCP_BOB_pre_pvalue,
    pre_GPCP_BOB_pre_hypothesis,
) = ca.dim_linregress(preGPCP_BOB_JJA, preGPCP_JJA)

(
    pre_his_BOB_pre_slope,
    pre_his_BOB_pre_intercept,
    pre_his_BOB_pre_rvalue,
    pre_his_BOB_pre_pvalue,
    pre_his_BOB_pre_hypothesis,
) = ca.dim_linregress(prehis_BOB_JJA.sel(time=prehis_BOB_JJA.time.dt.year>=1979), prehis_JJA.sel(time=prehis_JJA.time.dt.year>=1979))
(
    pre_ssp585_BOB_pre_slope,
    pre_ssp585_BOB_pre_intercept,
    pre_ssp585_BOB_pre_rvalue,
    pre_ssp585_BOB_pre_pvalue,
    pre_ssp585_BOB_pre_hypothesis,
) = ca.dim_linregress(pressp585_BOB_JJA, pressp585_JJA)
# %%
pre_his_BOB_pre_slope_ens = pre_his_BOB_pre_slope.mean(dim="models", skipna=True)
pre_ssp585_BOB_pre_slope_ens = pre_ssp585_BOB_pre_slope.mean(dim="models", skipna=True)
pre_his_BOB_pre_slope_ens_mask = ca.MME_reg_mask(pre_his_BOB_pre_slope_ens, pre_his_BOB_pre_slope.std(dim="models", skipna=True), len(pre_his_BOB_pre_slope.coords["models"]), True)
pre_ssp585_BOB_pre_slope_ens_mask = ca.MME_reg_mask(pre_ssp585_BOB_pre_slope_ens, pre_ssp585_BOB_pre_slope.std(dim="models", skipna=True), len(pre_ssp585_BOB_pre_slope.coords["models"]), True)

pre_his_BOB_pre_rvalue_ens = pre_his_BOB_pre_rvalue.mean(dim="models", skipna=True)
pre_ssp585_BOB_pre_rvalue_ens = pre_ssp585_BOB_pre_rvalue.mean(dim="models", skipna=True)
pre_his_BOB_pre_rvalue_ens_mask = ca.MME_reg_mask(pre_his_BOB_pre_rvalue_ens, pre_his_BOB_pre_rvalue.std(dim="models", skipna=True), len(pre_his_BOB_pre_rvalue.coords["models"]), True)
pre_ssp585_BOB_pre_rvalue_ens_mask = ca.MME_reg_mask(pre_ssp585_BOB_pre_rvalue_ens, pre_ssp585_BOB_pre_rvalue.std(dim="models", skipna=True), len(pre_ssp585_BOB_pre_rvalue.coords["models"]), True)

# %%
#   plot the regression coefficients in CRU, GPCP, historical run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (6, 5))
plot_array[5,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # BOB area
    x0 = 80
    y0 = 10.0
    width = 20.0
    height = 15.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_CRU_BOB_pre_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_CRU_BOB_pre_slope, axs[0], n, np.where(pre_CRU_BOB_pre_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="CRU",
)
# ===================================================
con = axs[1].contourf(
    pre_GPCP_BOB_pre_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_GPCP_BOB_pre_slope, axs[1], n, np.where(pre_GPCP_BOB_pre_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="GPCP",
)
# ===================================================
con = axs[2].contourf(
    pre_his_BOB_pre_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_his_BOB_pre_slope_ens, axs[2], n, np.where(pre_his_BOB_pre_slope_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[2].format(
    rtitle="1979-2014", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(pre_his_BOB_pre_slope.coords["models"].data):
    con = axs[num_models+3].contourf(
    pre_his_BOB_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
    sepl.plt_sig(
        pre_his_BOB_pre_slope.sel(models=mod), axs[num_models+3], n, np.where(pre_his_BOB_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 4.0,
    )

    axs[num_models+3].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg BOBR")

# %%
#   plot the correlation coefficients rvalue in CRU, GPCP, historical run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (6, 5))
plot_array[5,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_CRU_BOB_pre_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_CRU_BOB_pre_rvalue, axs[0], n, np.where(pre_CRU_BOB_pre_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="CRU",
)
# ===================================================
con = axs[1].contourf(
    pre_GPCP_BOB_pre_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_GPCP_BOB_pre_rvalue, axs[1], n, np.where(pre_GPCP_BOB_pre_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="GPCP",
)
# # ===================================================
con = axs[2].contourf(
    pre_his_BOB_pre_rvalue_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    
    )
sepl.plt_sig(
    pre_his_BOB_pre_rvalue_ens, axs[2], n, np.where(pre_his_BOB_pre_rvalue_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[2].format(
    rtitle="1979-2014", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(pre_his_BOB_pre_slope.coords["models"].data):
    con = axs[num_models+3].contourf(
    pre_his_BOB_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
    sepl.plt_sig(
        pre_his_BOB_pre_rvalue.sel(models=mod), axs[num_models+3], n, np.where(pre_his_BOB_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 4.0,
    )

    axs[num_models+3].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg BOBR")
# %%
#   calculate the precipitation in Northern China
lat = prehis_JJA.coords["lat"]
lon = prehis_JJA.coords["lon"]
lat_NC_range = lat[(lat>=27.5) & (lat<=37.5)]
lon_NC_range = lon[(lon>=105.0) & (lon<=125.0)]

prehis_NC_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_NC_range, lon=lon_NC_range)).mean(dim="lon", skipna=True)
pressp585_NC_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_NC_range, lon=lon_NC_range)).mean(dim="lon", skipna=True)
# %%
#   calculate the linregress
IndR_his_NCR_regress = ca.dim_linregress(prehis_India_JJA.sel(time=prehis_India_JJA.time.dt.year>=1979), prehis_NC_JJA.sel(time=prehis_NC_JJA.time.dt.year>=1979))
IndR_ssp585_NCR_regress = ca.dim_linregress(pressp585_India_JJA, pressp585_NC_JJA)
IndR_ssp585_p3_NCR_regress = ca.dim_linregress(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2064), pressp585_NC_JJA.sel(time=pressp585_NC_JJA.time.dt.year>=2064))

IndR_diff_NCR_slope = IndR_ssp585_p3_NCR_regress[0]-IndR_his_NCR_regress[0]
IndR_diff_NCR_rvalue = ca.cal_rdiff(IndR_ssp585_p3_NCR_regress[2],IndR_his_NCR_regress[2])

IndR_diff_NCR_slope_ens = IndR_diff_NCR_slope.mean(dim="models",skipna=True)
IndR_diff_NCR_rvalue_ens = ca.cal_rMME(IndR_diff_NCR_rvalue,"models")

IndR_his_NCR_slope_ens = IndR_his_NCR_regress[0].mean(dim="models",skipna=True)
IndR_ssp585_NCR_slope_ens = IndR_ssp585_NCR_regress[0].mean(dim="models",skipna=True)
IndR_ssp585_p3_NCR_slope_ens = IndR_ssp585_p3_NCR_regress[0].mean(dim="models",skipna=True)

IndR_his_NCR_rvalue_ens = ca.cal_rMME(IndR_his_NCR_regress[2],"models")
IndR_ssp585_NCR_rvalue_ens = ca.cal_rMME(IndR_ssp585_NCR_regress[2],"models")
IndR_ssp585_p3_NCR_rvalue_ens = ca.cal_rMME(IndR_ssp585_p3_NCR_regress[2],"models")
# %%
#   plot the bar-plot for regression coefficients
plot_data = np.zeros((27,3))
plot_data[:-1,0] = IndR_his_NCR_regress[0].data
plot_data[:-1,1] = IndR_ssp585_p3_NCR_regress[0].data
plot_data[:-1,2] = IndR_diff_NCR_slope.data
plot_data[-1,0] = IndR_his_NCR_slope_ens.data
plot_data[-1,1] = IndR_ssp585_p3_NCR_slope_ens.data
plot_data[-1,2] = IndR_diff_NCR_slope_ens.data

models = list(IndR_his_NCR_regress[0].coords["models"].data)
models.append("MME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=12.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(models,plot_data,width=0.6,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-0.5,0.5),ylocator=np.arange(-0.5,0.6,0.1),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and NCR")

# %%
#   plot the bar-plot for correlation coefficients
plot_data = np.zeros((27,3))
plot_data[:-1,0] = IndR_his_NCR_regress[2].data
plot_data[:-1,1] = IndR_ssp585_p3_NCR_regress[2].data
plot_data[:-1,2] = IndR_diff_NCR_rvalue.data
plot_data[-1,0] = IndR_his_NCR_rvalue_ens.data
plot_data[-1,1] = IndR_ssp585_p3_NCR_rvalue_ens.data
plot_data[-1,2] = IndR_diff_NCR_rvalue_ens.data

models = list(IndR_his_NCR_regress[0].coords["models"].data)
models.append("MME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=12.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(models,plot_data,width=0.6,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-0.7,0.7),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Cor. Coeff. IndR and NCR")
# %%
# # pick_up the good models
# gmodels = ["CNRM-CM6-1", "MIROC-ES2L", "NorESM2-LM", "HadGEM3-GC31-LL", "MRI-ESM2-0", "ACCESS-CM2", "MIROC6", "EC-Earth3", "CESM2-WACCM", "CAMS-CSM1-0"]
# prehis_JJA_gmodels = prehis_JJA.sel(models=gmodels)
# pressp_JJA_gmodels = pressp585_JJA.sel(models=gmodels)
# pre_his_India_pre_slope_gmodels = pre_his_India_pre_slope.sel(models=gmodels)
# pre_ssp585_India_pre_slope_gmodels = pre_ssp585_India_pre_slope.sel(models=gmodels)
# pre_his_India_pre_pvalue_gmodels = pre_his_India_pre_pvalue.sel(models=gmodels)
# pre_ssp585_India_pre_pvalue_gmodels = pre_ssp585_India_pre_pvalue.sel(models=gmodels)
# pre_his_India_pre_rvalue_gmodels = pre_his_India_pre_rvalue.sel(models=gmodels)

# pre_his_India_pre_slope_gmodels_ens = pre_his_India_pre_slope_gmodels.mean(dim="models", skipna=True)
# pre_ssp585_India_pre_slope_gmodels_ens = pre_ssp585_India_pre_slope_gmodels.mean(dim="models", skipna=True)

# pre_his_India_pre_slope_gmodels_ens_mask = ca.MME_reg_mask(pre_his_India_pre_slope_gmodels_ens, pre_his_India_pre_slope_gmodels.std(dim="models", skipna=True), len(gmodels), True)
# pre_ssp585_India_pre_slope_gmodels_ens_mask = ca.MME_reg_mask(pre_ssp585_India_pre_slope_gmodels_ens, pre_ssp585_India_pre_slope_gmodels.std(dim="models", skipna=True), len(gmodels), True)

# # %%
# #   bootstrap method test for the difference MME
# B = 1000
# alpha = 0.90
# dim = "models"
# pre_his_India_pre_slope_lowlim, pre_his_India_pre_slope_highlim = ca.cal_mean_bootstrap_confidence_intervals_pattern(pre_his_India_pre_slope_gmodels, B, alpha, dim)
# pre_ssp585_India_pre_slope_lowlim, pre_ssp585_India_pre_slope_highlim = ca.cal_mean_bootstrap_confidence_intervals_pattern(pre_ssp585_India_pre_slope_gmodels, B, alpha, dim)

# pre_diff_India_pre_slope_mask = ca.generate_bootstrap_mask(pre_his_India_pre_slope_lowlim, pre_his_India_pre_slope_highlim, pre_ssp585_India_pre_slope_lowlim, pre_ssp585_India_pre_slope_highlim)

# # %%
# #   plot the regression coefficients of CRU, historical ensmean, ssp585 ensmean, diff:ssp585-historical
# pplt.rc.grid = False
# pplt.rc.reso = "lo"
# cl = 0  # 设置地图投影的中心纬度
# proj = pplt.PlateCarree(central_longitude=cl)

# fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
# axs = fig.subplots(ncols=1, nrows=4, proj=proj)

# #   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
# yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
# extents = [xticks[0], xticks[-1], yticks[0], 55.0]
# sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# # ===================================================
# ski = 2
# n = 1
# w, h = 0.12, 0.14
# # ===================================================
# for ax in axs:
#     # Inida area
#     x0 = 70
#     y0 = 8.0
#     width = 16
#     height = 20.0
#     patches(ax, x0 - cl, y0, width, height, proj)
# # ===================================================
# con = axs[0].contourf(
#     pre_CRU_India_pre_slope,
#     cmap="ColdHot",
#     cmap_kw={"left": 0.06, "right": 0.94},
#     levels=np.arange(-2.0, 2.1, 0.1),
#     zorder=0.8,
#     extend="both",
#     )
# sepl.plt_sig(
#     pre_CRU_India_pre_slope, axs[0], n, np.where(pre_CRU_India_pre_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
# )

# axs[0].format(
#     rtitle="1950-2014", ltitle="CRU",
# )
# # ===================================================
# con = axs[1].contourf(
#     pre_his_India_pre_slope_gmodels_ens,
#     cmap="ColdHot",
#     cmap_kw={"left": 0.06, "right": 0.94},
#     levels=np.arange(-2.0, 2.1, 0.1),
#     zorder=0.8,
#     extend="both",
#     )
# sepl.plt_sig(
#     pre_his_India_pre_slope_gmodels_ens, axs[1], n, np.where(pre_his_India_pre_slope_gmodels_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
# )

# axs[1].format(
#     rtitle="1950-2014", ltitle="historical ensmean",
# )
# # ===================================================
# con = axs[2].contourf(
#     pre_ssp585_India_pre_slope_gmodels_ens,
#     cmap="ColdHot",
#     cmap_kw={"left": 0.06, "right": 0.94},
#     levels=np.arange(-2.0, 2.1, 0.1),
#     zorder=0.8,
#     extend="both",
#     )
# sepl.plt_sig(
#     pre_ssp585_India_pre_slope_gmodels_ens, axs[2], n, np.where(pre_ssp585_India_pre_slope_gmodels_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
# )

# axs[2].format(
#     rtitle="2015-2099", ltitle="ssp585 ensmean",
# )
# axs[2].colorbar(con, loc="b", width=0.13, length=0.7, label="")

# # ===================================================
# con = axs[3].contourf(
#     pre_ssp585_India_pre_slope_gmodels_ens - pre_his_India_pre_slope_gmodels_ens,
#     cmap="ColdHot",
#     cmap_kw={"left": 0.06, "right": 0.94},
#     levels=np.arange(-0.5, 0.55, 0.05),
#     zorder=0.8,
#     extend="both",
#     )
# sepl.plt_sig(
#     pre_ssp585_India_pre_slope_gmodels_ens - pre_his_India_pre_slope_gmodels_ens, axs[3], n, np.where(pre_diff_India_pre_slope_mask[::n, ::n] > 0.0), "bright purple", 4.0,
# )

# axs[3].format(
#     rtitle="diff", ltitle="ssp585 - historical",
# )
# # ===================================================
# cb = fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
# cb.set_ticks(np.arange(-0.5, 0.55, 0.1))
# fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   read the divuqvq/uq/vq data in historical and ssp585 run
fdivuqvqhis = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_div_uqvq.nc")
divuqvqhis_JJA = fdivuqvqhis["div_uqvq"]

fuqhis = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_uq_dpg.nc")
uqhis_JJA = fuqhis["uq_dpg"]

fvqhis = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_vq_dpg.nc")
vqhis_JJA = fvqhis["vq_dpg"]

fdivuqvqssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_div_uqvq.nc")
divuqvqssp585_JJA = fdivuqvqssp585["div_uqvq"]

fuqssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_uq_dpg.nc")
uqssp585_JJA = fuqssp585["uq_dpg"]

fvqssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_vq_dpg.nc")
vqssp585_JJA = fvqssp585["vq_dpg"]

# %%


ptop = 100 * 100
g = 9.8
#  ERA5 data
ERA5level = qERA5_ver_JJA.coords["level"] * 100.0
ERA5level.attrs["units"] = "Pa"
ERA5dp = geocat.comp.dpres_plevel(ERA5level, spERA5_JJA, ptop)
ERA5dpg = ERA5dp / g
ERA5dpg.attrs["units"] = "kg/m2"
uqERA5_ver_JJA = uERA5_ver_JJA * qERA5_ver_JJA.data * 1000.0
vqERA5_ver_JJA = vERA5_ver_JJA * qERA5_ver_JJA.data * 1000.0
uqERA5_ver_JJA.attrs["units"] = "[m/s][g/kg]"
vqERA5_ver_JJA.attrs["units"] = "[m/s][g/kg]"
uq_dpg_ERA5_JJA = (uqERA5_ver_JJA * ERA5dpg.data).sum(dim="level", skipna=True) / 1e05
vq_dpg_ERA5_JJA = (vqERA5_ver_JJA * ERA5dpg.data).sum(dim="level", skipna=True) / 1e05
uq_dpg_ERA5_JJA = ca.detrend_dim(uq_dpg_ERA5_JJA, "time", deg=1, demean=False)
vq_dpg_ERA5_JJA = ca.detrend_dim(vq_dpg_ERA5_JJA, "time", deg=1, demean=False)
uq_dpg_ERA5_JJA.attrs["units"] = "100kg/(m*s)"
vq_dpg_ERA5_JJA.attrs["units"] = "100kg/(m*s)"

divuqvqERA5_JJA = ca.cal_divergence(uq_dpg_ERA5_JJA, vq_dpg_ERA5_JJA)
divuqvqERA5_JJA = ca.detrend_dim(divuqvqERA5_JJA, "time", deg=1, demean=False)
# %%
#   pick up the good models in historical run and ssp585 run
divuqvqhis_JJA_gmodels = divuqvqhis_JJA.sel(models=gmodels)
uqhis_JJA_gmodels = uqhis_JJA.sel(models=gmodels)
vqhis_JJA_gmodels = vqhis_JJA.sel(models=gmodels)

divuqvqssp585_JJA_gmodels = divuqvqssp585_JJA.sel(models=gmodels)
uqssp585_JJA_gmodels = uqssp585_JJA.sel(models=gmodels)
vqssp585_JJA_gmodels = vqssp585_JJA.sel(models=gmodels)
# %%
#   calculate the divuqvq/uq/vq regress onto IndR
preCRU_India_JJA.coords["time"] = divuqvqERA5_JJA.coords["time"]
prehis_India_JJA_gmodels = prehis_India_JJA.sel(models=gmodels)
pressp585_India_JJA_gmodels = pressp585_India_JJA.sel(models=gmodels)
(
    pre_CRU_India_divuqvq_slope,
    pre_CRU_India_divuqvq_intercept,
    pre_CRU_India_divuqvq_rvalue,
    pre_CRU_India_divuqvq_pvalue,
    pre_CRU_India_divuqvq_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA, divuqvqERA5_JJA)

(
    pre_CRU_India_uq_slope,
    pre_CRU_India_uq_intercept,
    pre_CRU_India_uq_rvalue,
    pre_CRU_India_uq_pvalue,
    pre_CRU_India_uq_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA, uq_dpg_ERA5_JJA)

(
    pre_CRU_India_vq_slope,
    pre_CRU_India_vq_intercept,
    pre_CRU_India_vq_rvalue,
    pre_CRU_India_vq_pvalue,
    pre_CRU_India_vq_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA, vq_dpg_ERA5_JJA)

(
    pre_his_India_divuqvq_slope,
    pre_his_India_divuqvq_intercept,
    pre_his_India_divuqvq_rvalue,
    pre_his_India_divuqvq_pvalue,
    pre_his_India_divuqvq_hypothesis,
) = ca.dim_linregress(prehis_India_JJA_gmodels, divuqvqhis_JJA_gmodels)

(
    pre_his_India_uq_slope,
    pre_his_India_uq_intercept,
    pre_his_India_uq_rvalue,
    pre_his_India_uq_pvalue,
    pre_his_India_uq_hypothesis,
) = ca.dim_linregress(prehis_India_JJA_gmodels, uqhis_JJA_gmodels)

(
    pre_his_India_vq_slope,
    pre_his_India_vq_intercept,
    pre_his_India_vq_rvalue,
    pre_his_India_vq_pvalue,
    pre_his_India_vq_hypothesis,
) = ca.dim_linregress(prehis_India_JJA_gmodels, vqhis_JJA_gmodels)

(
    pre_ssp585_India_divuqvq_slope,
    pre_ssp585_India_divuqvq_intercept,
    pre_ssp585_India_divuqvq_rvalue,
    pre_ssp585_India_divuqvq_pvalue,
    pre_ssp585_India_divuqvq_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA_gmodels, divuqvqssp585_JJA_gmodels)

(
    pre_ssp585_India_uq_slope,
    pre_ssp585_India_uq_intercept,
    pre_ssp585_India_uq_rvalue,
    pre_ssp585_India_uq_pvalue,
    pre_ssp585_India_uq_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA_gmodels, uqssp585_JJA_gmodels)

(
    pre_ssp585_India_vq_slope,
    pre_ssp585_India_vq_intercept,
    pre_ssp585_India_vq_rvalue,
    pre_ssp585_India_vq_pvalue,
    pre_ssp585_India_vq_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA_gmodels, vqssp585_JJA_gmodels)

# %%
#   calculate the ensemble mean of historical run and ssp585 run
pre_his_India_divuqvq_slope_ens = pre_his_India_divuqvq_slope.mean(dim="models", skipna=True)
pre_his_India_divuqvq_ens_mask = ca.MME_reg_mask(pre_his_India_divuqvq_slope_ens, pre_his_India_divuqvq_slope.std(dim="models", skipna=True), len(pre_his_India_divuqvq_slope.coords["models"]), True)

pre_his_India_uq_slope_ens = pre_his_India_uq_slope.mean(dim="models", skipna=True)
pre_his_India_uq_ens_mask = ca.MME_reg_mask(pre_his_India_uq_slope_ens, pre_his_India_uq_slope.std(dim="models", skipna=True), len(pre_his_India_uq_slope.coords["models"]), True)

pre_his_India_vq_slope_ens = pre_his_India_vq_slope.mean(dim="models", skipna=True)
pre_his_India_vq_ens_mask = ca.MME_reg_mask(pre_his_India_vq_slope_ens, pre_his_India_vq_slope.std(dim="models", skipna=True), len(pre_his_India_vq_slope.coords["models"]), True)

pre_ssp585_India_divuqvq_slope_ens = pre_ssp585_India_divuqvq_slope.mean(dim="models", skipna=True)
pre_ssp585_India_divuqvq_ens_mask = ca.MME_reg_mask(pre_ssp585_India_divuqvq_slope_ens, pre_ssp585_India_divuqvq_slope.std(dim="models", skipna=True), len(pre_ssp585_India_divuqvq_slope.coords["models"]), True)

pre_ssp585_India_uq_slope_ens = pre_ssp585_India_uq_slope.mean(dim="models", skipna=True)
pre_ssp585_India_uq_ens_mask = ca.MME_reg_mask(pre_ssp585_India_uq_slope_ens, pre_ssp585_India_uq_slope.std(dim="models", skipna=True), len(pre_ssp585_India_uq_slope.coords["models"]), True)

pre_ssp585_India_vq_slope_ens = pre_ssp585_India_vq_slope.mean(dim="models", skipna=True)
pre_ssp585_India_vq_ens_mask = ca.MME_reg_mask(pre_ssp585_India_vq_slope_ens, pre_ssp585_India_vq_slope.std(dim="models", skipna=True), len(pre_ssp585_India_vq_slope.coords["models"]), True)

pre_CRU_India_uqvq_mask = ca.wind_check(
    xr.where(pre_CRU_India_uq_pvalue<=0.05, 1.0, 0.0),
    xr.where(pre_CRU_India_vq_pvalue<=0.05, 1.0, 0.0),
    xr.where(pre_CRU_India_uq_pvalue<=0.05, 1.0, 0.0),
    xr.where(pre_CRU_India_vq_pvalue<=0.05, 1.0, 0.0)
    )
pre_his_India_uqvq_mask = ca.wind_check(
    pre_his_India_uq_ens_mask, 
    pre_his_India_vq_ens_mask, 
    pre_his_India_uq_ens_mask, 
    pre_his_India_vq_ens_mask
    )
pre_ssp585_India_uqvq_mask = ca.wind_check(
    pre_ssp585_India_uq_ens_mask, 
    pre_ssp585_India_vq_ens_mask, 
    pre_ssp585_India_uq_ens_mask, 
    pre_ssp585_India_vq_ens_mask
    )
# %%
pre_his_India_divuqvq_slope_lowlim, pre_his_India_divuqvq_slope_highlim = ca.cal_mean_bootstrap_confidence_intervals_pattern(pre_his_India_divuqvq_slope, B, alpha, dim)
pre_ssp585_India_divuqvq_slope_lowlim, pre_ssp585_India_divuqvq_slope_highlim = ca.cal_mean_bootstrap_confidence_intervals_pattern(pre_ssp585_India_divuqvq_slope, B, alpha, dim)

pre_his_India_uq_slope_lowlim, pre_his_India_uq_slope_highlim = ca.cal_mean_bootstrap_confidence_intervals_pattern(pre_his_India_uq_slope, B, alpha, dim)
pre_ssp585_India_uq_slope_lowlim, pre_ssp585_India_uq_slope_highlim = ca.cal_mean_bootstrap_confidence_intervals_pattern(pre_ssp585_India_uq_slope, B, alpha, dim)

pre_his_India_vq_slope_lowlim, pre_his_India_vq_slope_highlim = ca.cal_mean_bootstrap_confidence_intervals_pattern(pre_his_India_vq_slope, B, alpha, dim)
pre_ssp585_India_vq_slope_lowlim, pre_ssp585_India_vq_slope_highlim = ca.cal_mean_bootstrap_confidence_intervals_pattern(pre_ssp585_India_vq_slope, B, alpha, dim)

pre_diff_India_divuqvq_slope_mask = ca.generate_bootstrap_mask(pre_his_India_divuqvq_slope_lowlim, pre_his_India_divuqvq_slope_highlim, pre_ssp585_India_divuqvq_slope_lowlim, pre_ssp585_India_divuqvq_slope_highlim)

pre_diff_India_uq_slope_mask = ca.generate_bootstrap_mask(pre_his_India_uq_slope_lowlim, pre_his_India_uq_slope_highlim, pre_ssp585_India_uq_slope_lowlim, pre_ssp585_India_uq_slope_highlim)

pre_diff_India_vq_slope_mask = ca.generate_bootstrap_mask(pre_his_India_vq_slope_lowlim, pre_his_India_vq_slope_highlim, pre_ssp585_India_vq_slope_lowlim, pre_ssp585_India_vq_slope_highlim)

pre_diff_India_uqvq_slope_mask = ca.wind_check(pre_diff_India_uq_slope_mask, pre_diff_India_vq_slope_mask, pre_diff_India_uq_slope_mask, pre_diff_India_vq_slope_mask)


# %%
#   plot the divuqvq/uq/vq regress onto IndR
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=4, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
    ax.add_patch(rect)
    # Inida area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_CRU_India_divuqvq_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-3.0e-7, 3.5e-7, 5.0e-8),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_CRU_India_divuqvq_slope, axs[0], n, np.where(pre_CRU_India_divuqvq_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
)
m = axs[0].quiver(
    pre_CRU_India_uq_slope.where(pre_CRU_India_uqvq_mask > 0.0)[::ski, ::ski],
    pre_CRU_India_vq_slope.where(pre_CRU_India_uqvq_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.10,
    pivot="mid",
    color="black",
)

qk = axs[0].quiverkey(
    m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
)

axs[0].format(
    rtitle="1950-2014", ltitle="CRU",
)
# ===================================================
con = axs[1].contourf(
    pre_his_India_divuqvq_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-3.0e-7, 3.5e-7, 5.0e-8),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_his_India_divuqvq_slope_ens, axs[1], n, np.where(pre_his_India_divuqvq_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)
m = axs[1].quiver(
    pre_his_India_uq_slope_ens.where(pre_his_India_uqvq_mask > 0.0)[::ski, ::ski],
    pre_his_India_vq_slope_ens.where(pre_his_India_uqvq_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.10,
    pivot="mid",
    color="black",
)

qk = axs[1].quiverkey(
    m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
)

axs[1].format(
    rtitle="1950-2014", ltitle="historical ensmean",
)
# ===================================================
con = axs[2].contourf(
    pre_ssp585_India_divuqvq_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-3.0e-7, 3.5e-7, 5.0e-8),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_ssp585_India_divuqvq_slope_ens, axs[2], n, np.where(pre_ssp585_India_divuqvq_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)
m = axs[2].quiver(
    pre_ssp585_India_uq_slope_ens.where(pre_ssp585_India_uqvq_mask > 0.0)[::ski, ::ski],
    pre_ssp585_India_vq_slope_ens.where(pre_ssp585_India_uqvq_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.10,
    pivot="mid",
    color="black",
)

qk = axs[2].quiverkey(
    m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
)

axs[2].format(
    rtitle="2015-2099", ltitle="ssp585 ensmean",
)
axs[2].colorbar(con, loc="b", width=0.13, length=0.7, label="")
# ===================================================
con = axs[3].contourf(
    pre_ssp585_India_divuqvq_slope_ens - pre_his_India_divuqvq_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-5.0e-8, 5.5e-8, 5.0e-9),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_ssp585_India_divuqvq_slope_ens - pre_his_India_divuqvq_slope_ens, axs[3], n, np.where(pre_diff_India_divuqvq_slope_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)
m = axs[3].quiver(
    (pre_ssp585_India_uq_slope_ens-pre_his_India_uq_slope_ens)[::ski, ::ski],
    (pre_ssp585_India_vq_slope_ens-pre_his_India_vq_slope_ens)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.015,
    pivot="mid",
    color="black",
)

qk = axs[3].quiverkey(
    m, X=1 - w / 2, Y=0.7 * h, U=0.1, label="0.1", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
)

axs[3].format(
    rtitle="diff", ltitle="ssp585 - historical",
)
# ===================================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="Uq & div reg IndR")
# %%
#   combined the three variables 200hPa hgt, u, v into one variables
#   first select the MV-EOF area
lat = IndR_diff_hgt_avalue.coords["lat"]

# %%
# #   output the lowlim and highlim for historical run and ssp585 run
# pre_his_India_pre_slope_lowlim.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/pre_his_India_pre_slope_lowlim.nc")
# pre_his_India_pre_slope_highlim.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/pre_his_India_pre_slope_highlim.nc")

# pre_ssp585_India_pre_slope_lowlim.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/pre_ssp585_India_pre_slope_lowlim.nc")
# pre_ssp585_India_pre_slope_highlim.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/pre_ssp585_India_pre_slope_highlim.nc")

# pre_his_India_divuqvq_slope_lowlim.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/pre_his_India_divuqvq_slope_lowlim.nc")
# pre_his_India_divuqvq_slope_highlim.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/pre_his_India_divuqvq_slope_highlim.nc")

# pre_ssp585_India_divuqvq_slope_lowlim.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/pre_ssp585_India_divuqvq_slope_lowlim.nc")
# pre_ssp585_India_divuqvq_slope_highlim.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/pre_ssp585_India_divuqvq_slope_highlim.nc")

# pre_his_India_uq_slope_lowlim.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/pre_his_India_uq_slope_lowlim.nc")
# pre_his_India_uq_slope_highlim.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/pre_his_India_uq_slope_highlim.nc")

# pre_ssp585_India_uq_slope_lowlim.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/pre_ssp585_India_uq_slope_lowlim.nc")
# pre_ssp585_India_uq_slope_highlim.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/pre_ssp585_India_uq_slope_highlim.nc")

# pre_his_India_vq_slope_lowlim.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/pre_his_India_vq_slope_lowlim.nc")
# pre_his_India_vq_slope_highlim.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/pre_his_India_vq_slope_highlim.nc")

# pre_ssp585_India_vq_slope_lowlim.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/pre_ssp585_India_vq_slope_lowlim.nc")
# pre_ssp585_India_vq_slope_highlim.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/pre_ssp585_India_vq_slope_highlim.nc")
# %%
#   calculate for the last 30yr(2070-2099) for ssp585
(
    pre_ssp585_p3_India_pre_slope,
    pre_ssp585_p3_India_pre_intercept,
    pre_ssp585_p3_India_pre_rvalue,
    pre_ssp585_p3_India_pre_pvalue,
    pre_ssp585_p3_India_pre_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2070), pressp585_JJA.sel(time=pressp585_JJA.time.dt.year>=2070))

gmodels = ["CNRM-CM6-1", "MIROC-ES2L", "NorESM2-LM", "HadGEM3-GC31-LL", "MRI-ESM2-0", "ACCESS-CM2", "MIROC6", "EC-Earth3", "CESM2-WACCM", "CAMS-CSM1-0"]
pre_ssp585_p3_India_pre_slope_gmodels = pre_ssp585_p3_India_pre_slope.sel(models=gmodels)
pre_ssp585_p3_India_pre_pvalue_gmodels = pre_ssp585_p3_India_pre_pvalue.sel(models=gmodels)
pre_ssp585_p3_India_pre_rvalue_gmodels = pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels)

pre_ssp585_p3_India_pre_slope_gmodels_ens = pre_ssp585_p3_India_pre_slope_gmodels.mean(dim="models", skipna=True)


pre_ssp585_p3_India_pre_slope_gmodels_ens_mask = ca.MME_reg_mask(pre_ssp585_p3_India_pre_slope_gmodels_ens, pre_ssp585_p3_India_pre_slope_gmodels.std(dim="models", skipna=True), len(gmodels), True)

#   bootstrap method test for the difference MME
B = 1000
alpha = 0.90
dim = "models"
pre_his_India_pre_slope_lowlim, pre_his_India_pre_slope_highlim = ca.cal_mean_bootstrap_confidence_intervals_pattern(pre_his_India_pre_rvalue_gmodels, B, alpha, dim)
pre_ssp585_p3_India_pre_slope_lowlim, pre_ssp585_p3_India_pre_slope_highlim = ca.cal_mean_bootstrap_confidence_intervals_pattern(pre_ssp585_p3_India_pre_rvalue_gmodels, B, alpha, dim)

pre_diff_p3_India_pre_slope_mask = ca.generate_bootstrap_mask(pre_his_India_pre_slope_lowlim, pre_his_India_pre_slope_highlim, pre_ssp585_p3_India_pre_slope_lowlim, pre_ssp585_p3_India_pre_slope_highlim)
# %%
#   plot the regression coefficients of CRU, historical ensmean, ssp585_p3 ensmean, diff:ssp585_p3-historical
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=4, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # Inida area
    x0 = 70
    y0 = 8.0
    # y0 = 30.0
    width = 16.0
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_CRU_India_pre_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_CRU_India_pre_slope, axs[0], n, np.where(pre_CRU_India_pre_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
)

# axs[0].format(
#     rtitle="1950-2014", ltitle="CRU",
# )
axs[0].format(
    rtitle="1979-2014", ltitle="GPCP",
)
# ===================================================
con = axs[1].contourf(
    pre_his_India_pre_slope_gmodels_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_his_India_pre_slope_gmodels_ens, axs[1], n, np.where(pre_his_India_pre_slope_gmodels_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

# axs[1].format(
#     rtitle="1950-2014", ltitle="historical ensmean",
# )
axs[1].format(
    rtitle="1979-2014", ltitle="historical ensmean",
)
# ===================================================
con = axs[2].contourf(
    pre_ssp585_p3_India_pre_slope_gmodels_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_ssp585_p3_India_pre_slope_gmodels_ens, axs[2], n, np.where(pre_ssp585_p3_India_pre_slope_gmodels_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[2].format(
    rtitle="2070-2099", ltitle="ssp585 p3 ensmean",
)
axs[2].colorbar(con, loc="b", width=0.13, length=0.7, label="")

# ===================================================
con = axs[3].contourf(
    pre_ssp585_p3_India_pre_slope_gmodels_ens - pre_his_India_pre_slope_gmodels_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-0.5, 0.55, 0.05),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_ssp585_p3_India_pre_slope_gmodels_ens - pre_his_India_pre_slope_gmodels_ens, axs[3], n, np.where(pre_diff_p3_India_pre_slope_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[3].format(
    rtitle="diff", ltitle="ssp585_p3 - historical",
)
# ===================================================
cb = fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
cb.set_ticks(np.arange(-0.5, 0.55, 0.1))
fig.format(abc="(a)", abcloc="l", suptitle="pre reg YZRR")
# %%
