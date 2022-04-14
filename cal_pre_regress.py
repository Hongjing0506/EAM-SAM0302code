'''
Author: ChenHJ
Date: 2022-04-14 16:32:41
LastEditors: ChenHJ
LastEditTime: 2022-04-14 18:20:30
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
preCRU_JJA = ca.p_time(preCRU, 6, 8, True) / 30.67
preCRU_JJA = ca.detrend_dim(preCRU_JJA, "time", deg=1, demean=False)

fprehis = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/pr_historical_r144x72_195001-201412.nc")
prehis_JJA = fprehis["pr"] * 3600 * 24

fpressp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/pr_ssp585_r144x72_201501-209912.nc")
pressp585_JJA = fpressp585["pr"] * 3600 *24

# %%
#   calculate the precipitation regress onto India precipitation
preCRU_India_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.loc[:, 8:28, 70:86]).mean(dim="lon", skipna=True)

lat = prehis.coords["lat"]
lon = prehis.coords["lon"]

lat_India_range = lat[(lat >= 8.0) & (lat <= 28.0)]
lon_India_range = lon[(lon >= 70.0) & (lon <= 86.0)]

prehis_India_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)
pressp585_India_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)

# %%
(
    pre_CRU_India_pre_slope,
    pre_CRU_India_pre_intercept,
    pre_CRU_India_pre_rvalue,
    pre_CRU_India_pre_pvalue,
    pre_CRU_India_pre_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA, preCRU_JJA)

(
    pre_his_India_pre_slope,
    pre_his_India_pre_intercept,
    pre_his_India_pre_rvalue,
    pre_his_India_pre_pvalue,
    pre_his_India_pre_hypothesis,
) = ca.dim_linregress(prehis_India_JJA, prehis_JJA)
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


# %%
#   plot the regression coefficients
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
    pre_CRU_India_pre_slope, axs[0], n, np.where(pre_CRU_India_pre_pvalue[::n, ::n] <= 0.05), "denim", 3.0,
)

axs[0].format(
    rtitle="1950-2014", ltitle="CRU",
)
# ===================================================
con = axs[1].contourf(
    pre_his_India_pre_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_his_India_pre_slope_ens, axs[1], n, np.where(pre_his_India_pre_slope_ens_mask[::n, ::n] > 0.0), "denim", 3.0,
)

axs[1].format(
    rtitle="1950-2014", ltitle="historical ensmean",
)
# ===================================================
for num_models,mod in enumerate(pre_his_India_pre_slope.coords["models"].data):
    con = axs[num_models+2].contourf(
    pre_his_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
    sepl.plt_sig(
        pre_his_India_pre_slope.sel(models=mod), axs[num_models+2], n, np.where(pre_his_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "denim", 3.0,
    )

    axs[num_models+2].format(
        rtitle="1950-2014", ltitle="historical {}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the regression coefficients
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
    pre_his_India_pre_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_his_India_pre_slope_ens, axs[0], n, np.where(pre_his_India_pre_slope_ens_mask[::n, ::n] > 0.0), "denim", 3.0,
)

axs[0].format(
    rtitle="1950-2014", ltitle="historical ensmean",
)
# ===================================================
con = axs[1].contourf(
    pre_ssp585_India_pre_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_ssp585_India_pre_slope_ens, axs[1], n, np.where(pre_ssp585_India_pre_slope_ens_mask[::n, ::n] > 0.0), "denim", 3.0,
)

axs[1].format(
    rtitle="1950-2014", ltitle="ssp585 ensmean",
)
# ===================================================
for num_models,mod in enumerate(pre_ssp585_India_pre_slope.coords["models"].data):
    con = axs[num_models+2].contourf(
    pre_ssp585_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
    sepl.plt_sig(
        pre_ssp585_India_pre_slope.sel(models=mod), axs[num_models+2], n, np.where(pre_ssp585_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "denim", 3.0,
    )

    axs[num_models+2].format(
        rtitle="1950-2014", ltitle="ssp585 {}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
# pick_up the good models
gmodels = ["CNRM-CM6-1", "MIROC-ES2L", "NorESM2-LM", "HadGEM3-GC31-LL", "MRI-ESM2-0", "ACCESS-CM2", "MIROC6", "EC-Earth3", "CESM2-WACCM", "CAMS-CSM1-0"]
prehis_JJA_gmodels = prehis_JJA.sel(models=gmodels)
pressp_JJA_gmodels = pressp585_JJA.sel(models=gmodels)
pre_his_India_pre_slope_gmodels = pre_his_India_pre_slope.sel(models=gmodels)
pre_ssp585_India_pre_slope_gmodels = pre_ssp585_India_pre_slope.sel(models=gmodels)
pre_his_India_pre_pvalue_gmodels = pre_his_India_pre_pvalue.sel(models=gmodels)
pre_ssp585_India_pre_pvalue_gmodels = pre_ssp585_India_pre_pvalue.sel(models=gmodels)

pre_his_India_pre_slope_gmodels_ens = pre_his_India_pre_slope_gmodels.mean(dim="models", skipna=True)
pre_ssp585_India_pre_slope_gmodels_ens = pre_ssp585_India_pre_slope_gmodels.mean(dim="models", skipna=True)

pre_his_India_pre_slope_gmodels_ens_mask = ca.MME_reg_mask(pre_his_India_pre_slope_gmodels_ens, pre_his_India_pre_slope_gmodels.std(dim="models", skipna=True), len(gmodels), True)
pre_ssp585_India_pre_slope_gmodels_ens_mask = ca.MME_reg_mask(pre_ssp585_India_pre_slope_gmodels_ens, pre_ssp585_India_pre_slope_gmodels.std(dim="models", skipna=True), len(gmodels), True)
# %%
#   plot the regression coefficients of CRU, historical ensmean, ssp585 ensmean, diff:ssp585-historical
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
    pre_CRU_India_pre_slope, axs[0], n, np.where(pre_CRU_India_pre_pvalue[::n, ::n] <= 0.05), "denim", 3.0,
)

axs[0].format(
    rtitle="1950-2014", ltitle="CRU",
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
    pre_his_India_pre_slope_gmodels_ens, axs[1], n, np.where(pre_his_India_pre_slope_gmodels_ens_mask[::n, ::n] > 0.0), "denim", 3.0,
)

axs[1].format(
    rtitle="1950-2014", ltitle="historical ensmean",
)
# ===================================================
con = axs[2].contourf(
    pre_ssp585_India_pre_slope_gmodels_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_ssp585_India_pre_slope_gmodels_ens, axs[2], n, np.where(pre_ssp585_India_pre_slope_gmodels_ens_mask[::n, ::n] > 0.0), "denim", 3.0,
)

axs[2].format(
    rtitle="2015-2099", ltitle="ssp585 ensmean",
)
# ===================================================
con = axs[3].contourf(
    pre_ssp585_India_pre_slope_gmodels_ens - pre_his_India_pre_slope_gmodels_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-0.5, 0.6, 0.05),
    zorder=0.8,
    extend="both",
    )
# sepl.plt_sig(
#     pre_ssp585_India_pre_slope_gmodels_ens, axs[3], n, np.where(pre_ssp585_India_pre_slope_gmodels_ens_mask[::n, ::n] > 0.0), "denim", 3.0,
# )

axs[3].format(
    rtitle="diff", ltitle="ssp585 - historical",
)
# ===================================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
