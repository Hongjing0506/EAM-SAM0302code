"""
Author: ChenHJ
Date: 2022-04-09 19:19:37
LastEditors: ChenHJ
LastEditTime: 2022-04-09 19:39:41
FilePath: /chenhj/0302code/decadal_trend.py
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
#   read obs and historical data
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

fqERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/q_mon_r144x72_195001-201412.nc"
)
qERA5 = fqERA5["q"]

fhgthis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models/historical/zg/zg_Amon_ensemble_historical_gn_195001-201412.nc"
)
hgthis = fhgthis["zg"]
hgthis.coords["plev"] = hgthis.coords["plev"] / 100.0
hgthis = hgthis.rename({"plev": "level"})

fuhis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models/historical/ua/ua_Amon_ensemble_historical_gn_195001-201412.nc"
)
uhis = fuhis["ua"]
uhis.coords["plev"] = uhis.coords["plev"] / 100.0
uhis = uhis.rename({"plev": "level"})

fvhis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models/historical/va/va_Amon_ensemble_historical_gn_195001-201412.nc"
)
vhis = fvhis["va"]
vhis.coords["plev"] = vhis.coords["plev"] / 100.0
vhis = vhis.rename({"plev": "level"})

fsphis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models/historical/ps/ps_Amon_ensemble_historical_gn_195001-201412.nc"
)
sphis = fsphis["ps"]

fqhis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models/historical/hus/hus_Amon_ensemble_historical_gn_195001-201412.nc"
)
qhis = fqhis["hus"]
qhis.coords["plev"] = qhis.coords["plev"] / 100.0
qhis = qhis.rename({"plev": "level"})

# %%
#   read the precipitation data
fpreCRU = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/cru_ts4.01_r144x72_195001-201412.nc"
)
preCRU = fpreCRU["pre"]


fprehis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models/historical/pr/pr_Amon_ensemble_historical_gn_195001-201412.nc"
)
prehis = fprehis["pr"] * 3600 * 24


# GPCP data just have 1979-2014 year
fpreGPCP = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/GPCP_r144x72_197901-201412.nc"
)
preGPCP = fpreGPCP["precip"]

# %%
#   pick up the obs and historical JJA
hgtERA5_ver_JJA = ca.p_time(hgtERA5, 6, 8, True).loc[:, 100.0:, :, :]
uERA5_ver_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, :, :]
vERA5_ver_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, :, :]
qERA5_ver_JJA = ca.p_time(qERA5, 6, 9, True).loc[:, 100.0:, :, :]
preCRU_JJA = ca.p_time(preCRU, 6, 8, True) / 30.67
spERA5_JJA = ca.p_time(spERA5, 6, 8, True)

hgthis_ver_JJA = ca.p_time(hgthis, 6, 8, True).loc[:, :100, :, :]
uhis_ver_JJA = ca.p_time(uhis, 6, 8, True).loc[:, :100, :, :]
vhis_ver_JJA = ca.p_time(vhis, 6, 8, True).loc[:, :100, :, :]
qhis_ver_JJA = ca.p_time(qhis, 6, 8, True).loc[:, :100, :, :]
prehis_JJA = ca.p_time(prehis, 6, 8, True)
sphis_JJA = ca.p_time(sphis, 6, 8, True)

preGPCP_JJA = ca.p_time(preGPCP, 6, 8, True)

# %%
#   read the ssp585 data
fhgtssp585 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models/ssp585/zg/zg_Amon_ensemble_ssp585_gn_201501-209912.nc"
)
hgtssp585 = fhgtssp585["zg"]
hgtssp585.coords["plev"] = hgtssp585.coords["plev"] / 100.0
hgtssp585 = hgtssp585.rename({"plev": "level"})

fussp585 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models/ssp585/ua/ua_Amon_ensemble_ssp585_gn_201501-209912.nc"
)
ussp585 = fussp585["ua"]
ussp585.coords["plev"] = ussp585.coords["plev"] / 100.0
ussp585 = ussp585.rename({"plev": "level"})

fvssp585 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models/ssp585/va/va_Amon_ensemble_ssp585_gn_201501-209912.nc"
)
vssp585 = fvssp585["va"]
vssp585.coords["plev"] = vssp585.coords["plev"] / 100.0
vssp585 = vssp585.rename({"plev": "level"})

fspssp585 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models/ssp585/ps/ps_Amon_ensemble_ssp585_gn_201501-209912.nc"
)
spssp585 = fspssp585["ps"]

fqssp585 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models/ssp585/hus/hus_Amon_ensemble_ssp585_gn_201501-209912.nc"
)
qssp585 = fqssp585["hus"]
qssp585.coords["plev"] = qssp585.coords["plev"] / 100.0
qssp585 = qssp585.rename({"plev": "level"})

fpressp585 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models/ssp585/pr/pr_Amon_ensemble_ssp585_gn_201501-209912.nc"
)
pressp585 = fpressp585["pr"] * 3600 * 24
# %%
#   pick up the ssp585 data
hgtssp585_ver_JJA = ca.p_time(hgtssp585, 6, 8, True).loc[:, :100, :, :]

ussp585_ver_JJA = ca.p_time(ussp585, 6, 8, True).loc[:, :100, :, :]
vssp585_ver_JJA = ca.p_time(vssp585, 6, 8, True).loc[:, :100, :, :]
qssp585_ver_JJA = ca.p_time(qssp585, 6, 8, True).loc[:, :100, :, :]
pressp585_JJA = ca.p_time(pressp585, 6, 8, True)
spssp585_JJA = ca.p_time(spssp585, 6, 8, True)

# %%
#   calculate the decadal linear trend of circulations
#   first remove the zonal mean of hgt
hgtERA5_ver_JJA = ca.dezonal_mean(hgtERA5_ver_JJA)
hgthis_ver_JJA = ca.dezonal_mean(hgthis_ver_JJA)
hgtssp585_ver_JJA = ca.dezonal_mean(hgtssp585_ver_JJA)

uERA5_ver_JJA = ca.dezonal_mean(uERA5_ver_JJA)
uhis_ver_JJA = ca.dezonal_mean(uhis_ver_JJA)
ussp585_ver_JJA = ca.dezonal_mean(ussp585_ver_JJA)

vERA5_ver_JJA = ca.dezonal_mean(vERA5_ver_JJA)
vhis_ver_JJA = ca.dezonal_mean(vhis_ver_JJA)
vssp585_ver_JJA = ca.dezonal_mean(vssp585_ver_JJA)

#   calculate the linear trend of circulation
hgtERA5_ver_JJA_trend = ca.dim_linregress(
    np.arange(len(hgtERA5_ver_JJA.coords["time"])), hgtERA5_ver_JJA
)
hgthis_ver_JJA_trend = ca.dim_linregress(
    np.arange(len(hgthis_ver_JJA.coords["time"])), hgthis_ver_JJA
)
hgtssp585_ver_JJA_trend = ca.dim_linregress(
    np.arange(len(hgtssp585_ver_JJA.coords["time"])), hgtssp585_ver_JJA
)

uERA5_ver_JJA_trend = ca.dim_linregress(
    np.arange(len(uERA5_ver_JJA.coords["time"])), uERA5_ver_JJA
)
uhis_ver_JJA_trend = ca.dim_linregress(
    np.arange(len(uhis_ver_JJA.coords["time"])), uhis_ver_JJA
)
ussp585_ver_JJA_trend = ca.dim_linregress(
    np.arange(len(ussp585_ver_JJA.coords["time"])), ussp585_ver_JJA
)

vERA5_ver_JJA_trend = ca.dim_linregress(
    np.arange(len(vERA5_ver_JJA.coords["time"])), vERA5_ver_JJA
)
vhis_ver_JJA_trend = ca.dim_linregress(
    np.arange(len(vhis_ver_JJA.coords["time"])), vhis_ver_JJA
)
vssp585_ver_JJA_trend = ca.dim_linregress(
    np.arange(len(vssp585_ver_JJA.coords["time"])), vssp585_ver_JJA
)
# %%
#   wind check
uv_ERA5_JJA_mask = ca.wind_check(
    xr.where(uERA5_ver_JJA_trend[3] <= 0.05, 1.0, 0.0),
    xr.where(vERA5_ver_JJA_trend[3] <= 0.05, 1.0, 0.0),
    xr.where(uERA5_ver_JJA_trend[3] <= 0.05, 1.0, 0.0),
    xr.where(vERA5_ver_JJA_trend[3] <= 0.05, 1.0, 0.0),
)

uv_his_JJA_mask = ca.wind_check(
    xr.where(uhis_ver_JJA_trend[3] <= 0.05, 1.0, 0.0),
    xr.where(vhis_ver_JJA_trend[3] <= 0.05, 1.0, 0.0),
    xr.where(uhis_ver_JJA_trend[3] <= 0.05, 1.0, 0.0),
    xr.where(vhis_ver_JJA_trend[3] <= 0.05, 1.0, 0.0),
)

uv_ssp585_JJA_mask = ca.wind_check(
    xr.where(ussp585_ver_JJA_trend[3] <= 0.05, 1.0, 0.0),
    xr.where(vssp585_ver_JJA_trend[3] <= 0.05, 1.0, 0.0),
    xr.where(ussp585_ver_JJA_trend[3] <= 0.05, 1.0, 0.0),
    xr.where(vssp585_ver_JJA_trend[3] <= 0.05, 1.0, 0.0),
)

# %%
#   plot the linear trend of circulation
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig_rvalue = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig_rvalue.subplots(ncols=1, nrows=3, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
startlevel = [-0.70, -0.40, -0.30]
endlevel = [0.70, 0.40, 0.30]
spacinglevel = [0.1, 0.08, 0.05]
scales = [0.008, 0.004, 0.004]
# ======================================
for ax in axs:
    rect = Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1
    )
    ax.add_patch(rect)
    # SAM area
    x0 = 70
    y0 = 10.0
    width = 40
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
    # IWF area
    x0 = 90
    y0 = 5.0
    width = 50
    height = 27.5
    patches(ax, x0 - cl, y0, width, height, proj)
# ======================================
levels = [200.0, 500.0, 850.0]
for i, lev in enumerate(levels):
    # con = axs[i, 0].contourf(
    #     hgtERA5_ver_JJA_trend[0].sel(level=lev),
    #     cmap="ColdHot",
    #     cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    #     levels=np.arange(
    #         startlevel[i], endlevel[i] + spacinglevel[i] / 2, spacinglevel[i]
    #     ),
    #     zorder=0.8,
    #     extend="both",
    # )
    # sepl.plt_sig(
    #     hgtERA5_ver_JJA_trend[0].sel(level=lev),
    #     axs[i, 0],
    #     n,
    #     np.where(hgtERA5_ver_JJA_trend[3].sel(level=lev)[::n, ::n] <= 0.05),
    #     "denim",
    #     3.0,
    # )
    # axs[i, 0].quiver(
    #     uERA5_ver_JJA_trend[0].sel(level=lev)[::ski, ::ski],
    #     vERA5_ver_JJA_trend[0].sel(level=lev)[::ski, ::ski],
    #     zorder=1.1,
    #     headwidth=2.6,
    #     headlength=2.3,
    #     headaxislength=2.3,
    #     scale_units="xy",
    #     scale=0.05,
    #     pivot="mid",
    #     color="grey6",
    # )

    # m = axs[i, 0].quiver(
    #     uERA5_ver_JJA_trend[0]
    #     .sel(level=lev)
    #     .where(uv_ERA5_JJA_mask.sel(level=lev) > 0.0)[::ski, ::ski],
    #     vERA5_ver_JJA_trend[0]
    #     .sel(level=lev)
    #     .where(uv_ERA5_JJA_mask.sel(level=lev) > 0.0)[::ski, ::ski],
    #     zorder=1.1,
    #     headwidth=2.6,
    #     headlength=2.3,
    #     headaxislength=2.3,
    #     scale_units="xy",
    #     scale=0.05,
    #     pivot="mid",
    #     color="black",
    # )

    # qk = axs[i, 0].quiverkey(
    #     m,
    #     X=1 - w / 2,
    #     Y=0.7 * h,
    #     U=0.5,
    #     label="0.5",
    #     labelpos="S",
    #     labelsep=0.05,
    #     fontproperties={"size": 5},
    #     zorder=3.1,
    # )

    # axs[i, 0].format(
    #     rtitle="1950-2014 {:.0f}hPa".format(lev), ltitle="ERA5",
    # )
    # # ======================================
    # con = axs[i, 1].contourf(
    #     hgthis_ver_JJA_trend[0].sel(level=lev),
    #     cmap="ColdHot",
    #     cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    #     levels=np.arange(
    #         startlevel[i], endlevel[i] + spacinglevel[i] / 2, spacinglevel[i]
    #     ),
    #     zorder=0.8,
    #     extend="both",
    # )
    # sepl.plt_sig(
    #     hgthis_ver_JJA_trend[0].sel(level=lev),
    #     axs[i, 1],
    #     n,
    #     np.where(hgthis_ver_JJA_trend[3].sel(level=lev)[::n, ::n] <= 0.05),
    #     "denim",
    #     3.0,
    # )
    # axs[i, 1].quiver(
    #     uhis_ver_JJA_trend[0].sel(level=lev)[::ski, ::ski],
    #     vhis_ver_JJA_trend[0].sel(level=lev)[::ski, ::ski],
    #     zorder=1.1,
    #     headwidth=2.6,
    #     headlength=2.3,
    #     headaxislength=2.3,
    #     scale_units="xy",
    #     scale=0.05,
    #     pivot="mid",
    #     color="grey6",
    # )

    # m = axs[i, 1].quiver(
    #     uhis_ver_JJA_trend[0]
    #     .sel(level=lev)
    #     .where(uv_his_JJA_mask.sel(level=lev) > 0.0)[::ski, ::ski],
    #     vhis_ver_JJA_trend[0]
    #     .sel(level=lev)
    #     .where(uv_his_JJA_mask.sel(level=lev) > 0.0)[::ski, ::ski],
    #     zorder=1.1,
    #     headwidth=2.6,
    #     headlength=2.3,
    #     headaxislength=2.3,
    #     scale_units="xy",
    #     scale=0.05,
    #     pivot="mid",
    #     color="black",
    # )

    # qk = axs[i, 1].quiverkey(
    #     m,
    #     X=1 - w / 2,
    #     Y=0.7 * h,
    #     U=0.5,
    #     label="0.5",
    #     labelpos="S",
    #     labelsep=0.05,
    #     fontproperties={"size": 5},
    #     zorder=3.1,
    # )

    # axs[i, 1].format(
    #     rtitle="1950-2014 {:.0f}hPa".format(lev), ltitle="historical",
    # )
    # # ======================================
    con = axs[i, 0].contourf(
        hgtssp585_ver_JJA_trend[0].sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(
            startlevel[i], endlevel[i] + spacinglevel[i] / 2, spacinglevel[i]
        ),
        zorder=0.8,
        extend="both",
    )
    sepl.plt_sig(
        hgtssp585_ver_JJA_trend[0].sel(level=lev),
        axs[i, 0],
        n,
        np.where(hgtssp585_ver_JJA_trend[3].sel(level=lev)[::n, ::n] <= 0.05),
        "denim",
        3.0,
    )
    # axs[i, 0].quiver(
    #     ussp585_ver_JJA_trend[0].sel(level=lev)[::ski, ::ski],
    #     vssp585_ver_JJA_trend[0].sel(level=lev)[::ski, ::ski],
    #     zorder=1.1,
    #     headwidth=2.6,
    #     headlength=2.3,
    #     headaxislength=2.3,
    #     scale_units="xy",
    #     scale=0.05,
    #     pivot="mid",
    #     color="grey6",
    # )

    m = axs[i, 0].quiver(
        ussp585_ver_JJA_trend[0]
        .sel(level=lev)
        .where(uv_ssp585_JJA_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        vssp585_ver_JJA_trend[0]
        .sel(level=lev)
        .where(uv_ssp585_JJA_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=0.004,
        pivot="mid",
        color="black",
    )

    qk = axs[i, 0].quiverkey(
        m,
        X=1 - w / 2,
        Y=0.7 * h,
        U=0.01,
        label="0.01",
        labelpos="S",
        labelsep=0.05,
        fontproperties={"size": 5},
        zorder=3.1,
    )

    axs[i, 0].format(
        rtitle="2015-2099 {:.0f}hPa".format(lev), ltitle="ssp585",
    )
    axs[i, 0].colorbar(con, loc="r", width=0.13, length=0.90, label="")
    # ======================================
# fig_rvalue.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig_rvalue.format(abc="(a)", abcloc="l", suptitle="linear trends")
# %%
#   calculate the whole levels water vapor flux in ERA5 historical and ssp585
ptop = 100 * 100
g = 9.8
# #  ERA5 data
# ERA5level = qERA5_ver_JJA.coords["level"] * 100.0
# ERA5level.attrs["units"] = "Pa"
# ERA5dp = geocat.comp.dpres_plevel(ERA5level, spERA5_JJA, ptop)
# ERA5dpg = ERA5dp / g
# ERA5dpg.attrs["units"] = "kg/m2"
# uqERA5_ver_JJA = uERA5_ver_JJA * qERA5_ver_JJA.data * 1000.0
# vqERA5_ver_JJA = vERA5_ver_JJA * qERA5_ver_JJA.data * 1000.0
# uqERA5_ver_JJA.attrs["units"] = "[m/s][g/kg]"
# vqERA5_ver_JJA.attrs["units"] = "[m/s][g/kg]"
# uq_dpg_ERA5_JJA = (uqERA5_ver_JJA * ERA5dpg.data).sum(dim="level", skipna=True) / 1e05
# vq_dpg_ERA5_JJA = (vqERA5_ver_JJA * ERA5dpg.data).sum(dim="level", skipna=True) / 1e05
# uq_dpg_ERA5_JJA = ca.detrend_dim(uq_dpg_ERA5_JJA, "time", deg=1, demean=False)
# vq_dpg_ERA5_JJA = ca.detrend_dim(vq_dpg_ERA5_JJA, "time", deg=1, demean=False)
# uq_dpg_ERA5_JJA.attrs["units"] = "100kg/(m*s)"
# vq_dpg_ERA5_JJA.attrs["units"] = "100kg/(m*s)"

# hislevel = qhis_ver_JJA.coords["level"] * 100.0
# hislevel.attrs["units"] = "Pa"
# hisdp = geocat.comp.dpres_plevel(hislevel, sphis_JJA, ptop)
# hisdpg = hisdp / g
# hisdpg.attrs["units"] = "kg/m2"
# uqhis_ver_JJA = uhis_ver_JJA * qhis_ver_JJA.data * 1000.0
# vqhis_ver_JJA = vhis_ver_JJA * qhis_ver_JJA.data * 1000.0
# uqhis_ver_JJA.attrs["units"] = "[m/s][g/kg]"
# vqhis_ver_JJA.attrs["units"] = "[m/s][g/kg]"
# uq_dpg_his_JJA = (uqhis_ver_JJA * hisdpg.data).sum(dim="level", skipna=True) / 1e05
# vq_dpg_his_JJA = (vqhis_ver_JJA * hisdpg.data).sum(dim="level", skipna=True) / 1e05
# uq_dpg_his_JJA = ca.detrend_dim(uq_dpg_his_JJA, "time", deg=1, demean=False)
# vq_dpg_his_JJA = ca.detrend_dim(vq_dpg_his_JJA, "time", deg=1, demean=False)
# uq_dpg_his_JJA.attrs["units"] = "100kg/(m*s)"
# vq_dpg_his_JJA.attrs["units"] = "100kg/(m*s)"

ssp585level = qssp585_ver_JJA.coords["level"] * 100.0
ssp585level.attrs["units"] = "Pa"
ssp585dp = geocat.comp.dpres_plevel(ssp585level, spssp585_JJA, ptop)
ssp585dpg = ssp585dp / g
ssp585dpg.attrs["units"] = "kg/m2"
uqssp585_ver_JJA = ussp585_ver_JJA * qssp585_ver_JJA.data * 1000.0
vqssp585_ver_JJA = vssp585_ver_JJA * qssp585_ver_JJA.data * 1000.0
uqssp585_ver_JJA.attrs["units"] = "[m/s][g/kg]"
vqssp585_ver_JJA.attrs["units"] = "[m/s][g/kg]"
uq_dpg_ssp585_JJA = (uqssp585_ver_JJA * ssp585dpg.data).sum(
    dim="level", skipna=True
) / 1e05
vq_dpg_ssp585_JJA = (vqssp585_ver_JJA * ssp585dpg.data).sum(
    dim="level", skipna=True
) / 1e05
# uq_dpg_ssp585_JJA = ca.detrend_dim(uq_dpg_ssp585_JJA, "time", deg=1, demean=False)
# vq_dpg_ssp585_JJA = ca.detrend_dim(vq_dpg_ssp585_JJA, "time", deg=1, demean=False)
uq_dpg_ssp585_JJA.attrs["units"] = "100kg/(m*s)"
vq_dpg_ssp585_JJA.attrs["units"] = "100kg/(m*s)"