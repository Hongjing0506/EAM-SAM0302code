'''
Author: ChenHJ
Date: 2022-04-14 16:32:41
LastEditors: ChenHJ
LastEditTime: 2022-04-19 11:29:53
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
    rtitle="2064-2099", ltitle="MME",
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
        rtitle="2064-2099", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
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
#   calculate the divuqvq/uq/vq in ERA5 reanalysis
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

uERA5_ver_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, :, :]
vERA5_ver_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, :, :]
qERA5_ver_JJA = ca.p_time(qERA5, 6, 9, True).loc[:, 100.0:, :, :]
spERA5_JJA = ca.p_time(spERA5, 6, 8, True)

uERA5_ver_JJA = ca.detrend_dim(uERA5_ver_JJA, "time", deg=1, demean=False)
vERA5_ver_JJA = ca.detrend_dim(vERA5_ver_JJA, "time", deg=1, demean=False)
qERA5_ver_JJA = ca.detrend_dim(qERA5_ver_JJA, "time", deg=1, demean=False)
spERA5_JJA = ca.detrend_dim(spERA5_JJA, "time", deg=1, demean=False)

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
