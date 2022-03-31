"""
Author: ChenHJ
Date: 2022-03-27 11:46:10
LastEditors: ChenHJ
LastEditTime: 2022-03-27 14:05:32
FilePath: /chenhj/0302code/mon_index.py
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
#   read obs data
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
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/zg/zg_Amon_ensemble_historical_gn_195001-201412.nc"
)
hgthis = fhgthis["zg"]
hgthis.coords["plev"] = hgthis.coords["plev"] / 100.0
hgthis = hgthis.rename({"plev": "level"})

fuhis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/ua/ua_Amon_ensemble_historical_gn_195001-201412.nc"
)
uhis = fuhis["ua"]
uhis.coords["plev"] = uhis.coords["plev"] / 100.0
uhis = uhis.rename({"plev": "level"})

fvhis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/va/va_Amon_ensemble_historical_gn_195001-201412.nc"
)
vhis = fvhis["va"]
vhis.coords["plev"] = vhis.coords["plev"] / 100.0
vhis = vhis.rename({"plev": "level"})

fsphis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/ps/ps_Amon_ensemble_historical_gn_195001-201412.nc"
)
sphis = fsphis["ps"]

fqhis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/hus/hus_Amon_ensemble_historical_gn_195001-201412.nc"
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
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/pr/pr_Amon_ensemble_historical_gn_195001-201412.nc"
)
prehis = fprehis["pr"]


# GPCP data just have 1979-2014 year
fpreGPCP = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/GPCP_r144x72_197901-201412.nc"
)
preGPCP = fpreGPCP["precip"]

# %%
#   pick up the JJA

hgtERA5_ver_JJA = ca.p_time(hgtERA5, 6, 8, True).loc[:, 100.0:, :, :]
uERA5_ver_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, :, :]
vERA5_ver_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, :, :]
qERA5_ver_JJA = ca.p_time(qERA5, 6, 9, True).loc[:, 100.0:, :, :]
preCRU_JJA = ca.p_time(preCRU, 6, 8, True)
spERA5_JJA = ca.p_time(spERA5, 6, 8, True)

hgtERA5_ver_JJA = ca.detrend_dim(hgtERA5_ver_JJA, "time", deg=1, demean=False)
uERA5_ver_JJA = ca.detrend_dim(uERA5_ver_JJA, "time", deg=1, demean=False)
vERA5_ver_JJA = ca.detrend_dim(vERA5_ver_JJA, "time", deg=1, demean=False)
qERA5_ver_JJA = ca.detrend_dim(qERA5_ver_JJA, "time", deg=1, demean=False)
preCRU_JJA = ca.detrend_dim(preCRU_JJA, "time", deg=1, demean=False)
spERA5_JJA = ca.detrend_dim(spERA5_JJA, "time", deg=1, demean=False)

hgthis_ver_JJA = ca.p_time(hgthis, 6, 8, True).loc[:, :100, :, :]

uhis_ver_JJA = ca.p_time(uhis, 6, 8, True).loc[:, :100, :, :]
vhis_ver_JJA = ca.p_time(vhis, 6, 8, True).loc[:, :100, :, :]
qhis_ver_JJA = ca.p_time(qhis, 6, 8, True).loc[:, :100, :, :]
prehis_JJA = ca.p_time(prehis, 6, 8, True)
sphis_JJA = ca.p_time(sphis, 6, 8, True)

hgthis_ver_JJA = ca.detrend_dim(hgthis_ver_JJA, "time", deg=1, demean=False)
uhis_ver_JJA = ca.detrend_dim(uhis_ver_JJA, "time", deg=1, demean=False)
vhis_ver_JJA = ca.detrend_dim(vhis_ver_JJA, "time", deg=1, demean=False)
qhis_ver_JJA = ca.detrend_dim(qhis_ver_JJA, "time", deg=1, demean=False)
prehis_JJA = ca.detrend_dim(prehis_JJA, "time", deg=1, demean=False)
sphis_JJA = ca.detrend_dim(sphis_JJA, "time", deg=1, demean=False)

preGPCP_JJA = ca.p_time(preGPCP, 6, 8, True)
preGPCP_JJA = ca.detrend_dim(preGPCP_JJA, "time", deg=1, demean=False)
# %%
#   calculate the whole levels water vapor flux
ptop = 100 * 100
g = 9.8
#   ERA5 data
ERA5level = qERA5_ver_JJA.coords["level"] * 100.0
ERA5level.attrs["units"] = "Pa"
ERA5dp = geocat.comp.dpres_plevel(ERA5level, spERA5_JJA, ptop)
ERA5dpg = ERA5dp / g
ERA5dpg.attrs["units"] = "kg/m2"
uqERA5_ver_JJA = uERA5_ver_JJA * qERA5_ver_JJA.data * 1000.0
vqERA5_ver_JJA = vERA5_ver_JJA * qERA5_ver_JJA.data * 1000.0
uqERA5_ver_JJA.attrs["units"] = "[m/s][g/kg]"
vqERA5_ver_JJA.attrs["units"] = "[m/s][g/kg]"
uq_dpg_ERA5_JJA = (uqERA5_ver_JJA * ERA5dpg.data).sum(dim="level", skipna=True)
vq_dpg_ERA5_JJA = (vqERA5_ver_JJA * ERA5dpg.data).sum(dim="level", skipna=True)
uq_dpg_ERA5_JJA = ca.detrend_dim(uq_dpg_ERA5_JJA, "time", deg=1, demean=False)
vq_dpg_ERA5_JJA = ca.detrend_dim(vq_dpg_ERA5_JJA, "time", deg=1, demean=False)

#   historical run data
hislevel = qhis_ver_JJA.coords["level"] * 100.0
hislevel.attrs["units"] = "Pa"
hisdp = geocat.comp.dpres_plevel(hislevel, sphis_JJA, ptop)
hisdpg = hisdp / g
hisdpg.attrs["units"] = "kg/m2"
uqhis_ver_JJA = uhis_ver_JJA * qhis_ver_JJA.data * 1000.0
vqhis_ver_JJA = vhis_ver_JJA * qhis_ver_JJA.data * 1000.0
uqhis_ver_JJA.attrs["units"] = "[m/s][g/kg]"
vqhis_ver_JJA.attrs["units"] = "[m/s][g/kg]"
uq_dpg_his_JJA = (uqhis_ver_JJA * hisdpg.data).sum(dim="level", skipna=True)
vq_dpg_his_JJA = (vqhis_ver_JJA * hisdpg.data).sum(dim="level", skipna=True)
uq_dpg_his_JJA = ca.detrend_dim(uq_dpg_his_JJA, "time", deg=1, demean=False)
vq_dpg_his_JJA = ca.detrend_dim(vq_dpg_his_JJA, "time", deg=1, demean=False)

# %%
#   calculate the monsoon index
ERA5_SAM_index = ca.SAM(vERA5_ver_JJA)
ERA5_SAM_index = ca.detrend_dim(ERA5_SAM_index, "time", deg=1, demean=False)
his_SAM_index = ca.SAM(vhis_ver_JJA)
his_SAM_index = ca.detrend_dim(his_SAM_index, "time", deg=1, demean=False)

ERA5_EAM_index = ca.EAM(uERA5_ver_JJA)
ERA5_EAM_index = ca.detrend_dim(ERA5_EAM_index, "time", deg=1, demean=False)
his_EAM_index = ca.EAM(uhis_ver_JJA)
his_EAM_index = ca.detrend_dim(his_EAM_index, "time", deg=1, demean=False)

ERA5_WY_index = ca.WY(uERA5_ver_JJA)
ERA5_WY_index = ca.detrend_dim(ERA5_WY_index, "time", deg=1, demean=False)
his_WY_index = ca.WY(uhis_ver_JJA)
his_WY_index = ca.detrend_dim(his_WY_index, "time", deg=1, demean=False)

ERA5_IWF_index = ca.IWF(uERA5_ver_JJA, vERA5_ver_JJA)
ERA5_IWF_index = ca.detrend_dim(ERA5_IWF_index, "time", deg=1, demean=False)
his_IWF_index = ca.IWF(uhis_ver_JJA, vhis_ver_JJA)
his_IWF_index = ca.detrend_dim(his_IWF_index, "time", deg=1, demean=False)
# %%
#   calculate the regression of two monsoon index
ERA5_regress = stats.linregress(ERA5_SAM_index, ERA5_EAM_index)
his_regress = stats.linregress(his_SAM_index, his_EAM_index)
ERA5_WY_EAM_regress = stats.linregress(ERA5_WY_index, ERA5_EAM_index)
his_WY_EAM_regress = stats.linregress(his_WY_index, his_EAM_index)
ERA5_IWF_EAM_regress = stats.linregress(ERA5_IWF_index, ERA5_EAM_index)
his_IWF_EAM_regress = stats.linregress(his_IWF_index, his_EAM_index)
ERA5_IWF_SAM_regress = stats.linregress(ERA5_IWF_index, ERA5_SAM_index)
his_IWF_SAM_regress = stats.linregress(his_IWF_index, his_SAM_index)
ERA5_IWF_WY_regress = stats.linregress(ERA5_IWF_index, ERA5_WY_index)
his_IWF_WY_regress = stats.linregress(his_IWF_index, his_WY_index)
ERA5_WY_SAM_regress = stats.linregress(ERA5_WY_index, ERA5_SAM_index)
his_WY_SAM_regress = stats.linregress(his_WY_index, his_SAM_index)
# %%
ERA5_his_EAM_regress = stats.linregress(ERA5_EAM_index, his_EAM_index)
ERA5_his_SAM_regress = stats.linregress(ERA5_SAM_index, his_SAM_index)
ERA5_his_WY_regress = stats.linregress(ERA5_WY_index, his_WY_index)
ERA5_his_IWF_regress = stats.linregress(ERA5_IWF_index, his_IWF_index)

# %%
# print(ERA5_his_IWF_regress)
print(
    ERA5_WY_SAM_regress, his_WY_SAM_regress
)
# %%
#   plot the monsoon index
fig = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
axs = fig.subplots(ncols=1, nrows=2)

lw = 1.0
# ========================================
m1 = axs[0].line(
    ERA5_IWF_index.time.dt.year, ca.standardize(ERA5_IWF_index), color="blue", lw=lw
)
m2 = axs[0].line(
    ERA5_SAM_index.time.dt.year, ca.standardize(ERA5_SAM_index), color="red", lw=lw
)

axs[0].legend(handles=[m1, m2], loc="ll", labels=["IWF_index", "SAM_index"], ncols=1)
axs[0].format(ltitle="ERA5", rtitle="r = {:.2f}".format(ERA5_IWF_SAM_regress[2]))
# ========================================
m1 = axs[1].line(
    his_IWF_index.time.dt.year, ca.standardize(his_IWF_index), color="blue", lw=lw
)
m2 = axs[1].line(
    his_SAM_index.time.dt.year, ca.standardize(his_SAM_index), color="red", lw=lw
)

axs[1].legend(handles=[m1, m2], loc="ll", labels=["IWF_index", "SAM_index"], ncols=1)
axs[1].format(ltitle="historical", rtitle="r = {:.2f}".format(his_IWF_SAM_regress[2]))
# ========================================
axs.format(ylim=(-3.0, 3.0), ylocator=1.0, yminorlocator=0.2, ylabel="", xlabel="")
fig.format(abc="(a)", abcloc="l")
# %%
#   calculate the hgt and u,v regress into the monsoon index
(
    hgt_ERA5_EAM_slope,
    hgt_ERA5_EAM_intercept,
    hgt_ERA5_EAM_rvalue,
    hgt_ERA5_EAM_pvalue,
    hgt_ERA5_EAM_hypothesis,
) = ca.dim_linregress(ERA5_EAM_index, hgtERA5_ver_JJA)
(
    u_ERA5_EAM_slope,
    u_ERA5_EAM_intercept,
    u_ERA5_EAM_rvalue,
    u_ERA5_EAM_pvalue,
    u_ERA5_EAM_hypothesis,
) = ca.dim_linregress(ERA5_EAM_index, uERA5_ver_JJA)
(
    v_ERA5_EAM_slope,
    v_ERA5_EAM_intercept,
    v_ERA5_EAM_rvalue,
    v_ERA5_EAM_pvalue,
    v_ERA5_EAM_hypothesis,
) = ca.dim_linregress(ERA5_EAM_index, vERA5_ver_JJA)

(
    hgt_his_EAM_slope,
    hgt_his_EAM_intercept,
    hgt_his_EAM_rvalue,
    hgt_his_EAM_pvalue,
    hgt_his_EAM_hypothesis,
) = ca.dim_linregress(his_EAM_index, hgthis_ver_JJA)
(
    u_his_EAM_slope,
    u_his_EAM_intercept,
    u_his_EAM_rvalue,
    u_his_EAM_pvalue,
    u_his_EAM_hypothesis,
) = ca.dim_linregress(his_EAM_index, uhis_ver_JJA)
(
    v_his_EAM_slope,
    v_his_EAM_intercept,
    v_his_EAM_rvalue,
    v_his_EAM_pvalue,
    v_his_EAM_hypothesis,
) = ca.dim_linregress(his_EAM_index, vhis_ver_JJA)
# %%
(
    hgt_ERA5_SAM_slope,
    hgt_ERA5_SAM_intercept,
    hgt_ERA5_SAM_rvalue,
    hgt_ERA5_SAM_pvalue,
    hgt_ERA5_SAM_hypothesis,
) = ca.dim_linregress(ERA5_SAM_index, hgtERA5_ver_JJA)
(
    u_ERA5_SAM_slope,
    u_ERA5_SAM_intercept,
    u_ERA5_SAM_rvalue,
    u_ERA5_SAM_pvalue,
    u_ERA5_SAM_hypothesis,
) = ca.dim_linregress(ERA5_SAM_index, uERA5_ver_JJA)
(
    v_ERA5_SAM_slope,
    v_ERA5_SAM_intercept,
    v_ERA5_SAM_rvalue,
    v_ERA5_SAM_pvalue,
    v_ERA5_SAM_hypothesis,
) = ca.dim_linregress(ERA5_SAM_index, vERA5_ver_JJA)

(
    hgt_his_SAM_slope,
    hgt_his_SAM_intercept,
    hgt_his_SAM_rvalue,
    hgt_his_SAM_pvalue,
    hgt_his_SAM_hypothesis,
) = ca.dim_linregress(his_SAM_index, hgthis_ver_JJA)
(
    u_his_SAM_slope,
    u_his_SAM_intercept,
    u_his_SAM_rvalue,
    u_his_SAM_pvalue,
    u_his_SAM_hypothesis,
) = ca.dim_linregress(his_SAM_index, uhis_ver_JJA)
(
    v_his_SAM_slope,
    v_his_SAM_intercept,
    v_his_SAM_rvalue,
    v_his_SAM_pvalue,
    v_his_SAM_hypothesis,
) = ca.dim_linregress(his_SAM_index, vhis_ver_JJA)

# %%
(
    hgt_ERA5_WY_slope,
    hgt_ERA5_WY_intercept,
    hgt_ERA5_WY_rvalue,
    hgt_ERA5_WY_pvalue,
    hgt_ERA5_WY_hypothesis,
) = ca.dim_linregress(ERA5_WY_index, hgtERA5_ver_JJA)
(
    u_ERA5_WY_slope,
    u_ERA5_WY_intercept,
    u_ERA5_WY_rvalue,
    u_ERA5_WY_pvalue,
    u_ERA5_WY_hypothesis,
) = ca.dim_linregress(ERA5_WY_index, uERA5_ver_JJA)
(
    v_ERA5_WY_slope,
    v_ERA5_WY_intercept,
    v_ERA5_WY_rvalue,
    v_ERA5_WY_pvalue,
    v_ERA5_WY_hypothesis,
) = ca.dim_linregress(ERA5_WY_index, vERA5_ver_JJA)

(
    hgt_his_WY_slope,
    hgt_his_WY_intercept,
    hgt_his_WY_rvalue,
    hgt_his_WY_pvalue,
    hgt_his_WY_hypothesis,
) = ca.dim_linregress(his_WY_index, hgthis_ver_JJA)
(
    u_his_WY_slope,
    u_his_WY_intercept,
    u_his_WY_rvalue,
    u_his_WY_pvalue,
    u_his_WY_hypothesis,
) = ca.dim_linregress(his_WY_index, uhis_ver_JJA)
(
    v_his_WY_slope,
    v_his_WY_intercept,
    v_his_WY_rvalue,
    v_his_WY_pvalue,
    v_his_WY_hypothesis,
) = ca.dim_linregress(his_WY_index, vhis_ver_JJA)
# %%
(
    hgt_ERA5_IWF_slope,
    hgt_ERA5_IWF_intercept,
    hgt_ERA5_IWF_rvalue,
    hgt_ERA5_IWF_pvalue,
    hgt_ERA5_IWF_hypothesis,
) = ca.dim_linregress(ERA5_IWF_index, hgtERA5_ver_JJA)
(
    u_ERA5_IWF_slope,
    u_ERA5_IWF_intercept,
    u_ERA5_IWF_rvalue,
    u_ERA5_IWF_pvalue,
    u_ERA5_IWF_hypothesis,
) = ca.dim_linregress(ERA5_IWF_index, uERA5_ver_JJA)
(
    v_ERA5_IWF_slope,
    v_ERA5_IWF_intercept,
    v_ERA5_IWF_rvalue,
    v_ERA5_IWF_pvalue,
    v_ERA5_IWF_hypothesis,
) = ca.dim_linregress(ERA5_IWF_index, vERA5_ver_JJA)

(
    hgt_his_IWF_slope,
    hgt_his_IWF_intercept,
    hgt_his_IWF_rvalue,
    hgt_his_IWF_pvalue,
    hgt_his_IWF_hypothesis,
) = ca.dim_linregress(his_IWF_index, hgthis_ver_JJA)
(
    u_his_IWF_slope,
    u_his_IWF_intercept,
    u_his_IWF_rvalue,
    u_his_IWF_pvalue,
    u_his_IWF_hypothesis,
) = ca.dim_linregress(his_IWF_index, uhis_ver_JJA)
(
    v_his_IWF_slope,
    v_his_IWF_intercept,
    v_his_IWF_rvalue,
    v_his_IWF_pvalue,
    v_his_IWF_hypothesis,
) = ca.dim_linregress(his_IWF_index, vhis_ver_JJA)
# %%
#  wind check
wind_ERA5_EAM_mask = ca.wind_check(
    xr.where(u_ERA5_EAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_EAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_ERA5_EAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_EAM_pvalue <= 0.05, 1.0, 0.0),
)

wind_ERA5_SAM_mask = ca.wind_check(
    xr.where(u_ERA5_SAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_SAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_ERA5_SAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_SAM_pvalue <= 0.05, 1.0, 0.0),
)

wind_ERA5_WY_mask = ca.wind_check(
    xr.where(u_ERA5_WY_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_WY_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_ERA5_WY_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_WY_pvalue <= 0.05, 1.0, 0.0),
)

wind_ERA5_IWF_mask = ca.wind_check(
    xr.where(u_ERA5_IWF_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_IWF_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_ERA5_IWF_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_IWF_pvalue <= 0.05, 1.0, 0.0),
)

wind_his_EAM_mask = ca.wind_check(
    xr.where(u_his_EAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_EAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_EAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_EAM_pvalue <= 0.05, 1.0, 0.0),
)

wind_his_SAM_mask = ca.wind_check(
    xr.where(u_his_SAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_SAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_SAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_SAM_pvalue <= 0.05, 1.0, 0.0),
)

wind_his_WY_mask = ca.wind_check(
    xr.where(u_his_WY_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_WY_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_WY_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_WY_pvalue <= 0.05, 1.0, 0.0),
)

wind_his_IWF_mask = ca.wind_check(
    xr.where(u_his_IWF_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_IWF_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_IWF_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_IWF_pvalue <= 0.05, 1.0, 0.0),
)
# %%
#   plot the linear regression between monsoon index and hgt, u, v
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=2, nrows=3, proj=proj)

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
    rect = Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1
    )
    ax.add_patch(rect)
    # region 1
    x0 = 110
    y0 = 40
    width = 40
    height = 10
    patches(ax, x0 - cl, y0, width, height, proj)
    # region 2
    x0 = 110
    y0 = 25
    width = 40
    height = 10
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0, 0].contourf(
    hgt_ERA5_EAM_rvalue.sel(level=200.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_EAM_pvalue.sel(level=200.0),
    axs[0, 0],
    n,
    np.where(hgt_ERA5_EAM_pvalue.sel(level=200.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 0].quiver(
    u_ERA5_EAM_rvalue.sel(level=200.0)[::ski, ::ski],
    v_ERA5_EAM_rvalue.sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 0].quiver(
    u_ERA5_EAM_rvalue.where(wind_ERA5_EAM_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    v_ERA5_EAM_rvalue.where(wind_ERA5_EAM_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 0].format(ltitle="EAM index", rtitle="ERA5 200hPa")
# ===========================================

con = axs[1, 0].contourf(
    hgt_ERA5_EAM_rvalue.sel(level=500.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_EAM_pvalue.sel(level=500.0),
    axs[1, 0],
    n,
    np.where(hgt_ERA5_EAM_pvalue.sel(level=500.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1, 0].quiver(
    u_ERA5_EAM_rvalue.sel(level=500.0)[::ski, ::ski],
    v_ERA5_EAM_rvalue.sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1, 0].quiver(
    u_ERA5_EAM_rvalue.where(wind_ERA5_EAM_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    v_ERA5_EAM_rvalue.where(wind_ERA5_EAM_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1, 0].format(ltitle="EAM index", rtitle="ERA5 500hPa")
# ===================================================
con = axs[2, 0].contourf(
    hgt_ERA5_EAM_rvalue.sel(level=850.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_EAM_pvalue.sel(level=850.0),
    axs[2, 0],
    n,
    np.where(hgt_ERA5_EAM_pvalue.sel(level=850.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[2, 0].quiver(
    u_ERA5_EAM_rvalue.sel(level=850.0)[::ski, ::ski],
    v_ERA5_EAM_rvalue.sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[2, 0].quiver(
    u_ERA5_EAM_rvalue.where(wind_ERA5_EAM_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    v_ERA5_EAM_rvalue.where(wind_ERA5_EAM_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[2, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[2, 0].format(ltitle="EAM index", rtitle="ERA5 850hPa")
# ===================================================
#   plot the historical run result
con = axs[0, 1].contourf(
    hgt_his_EAM_rvalue.sel(level=200.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_EAM_pvalue.sel(level=200.0),
    axs[0, 1],
    n,
    np.where(hgt_his_EAM_pvalue.sel(level=200.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 1].quiver(
    u_his_EAM_rvalue.sel(level=200.0)[::ski, ::ski],
    v_his_EAM_rvalue.sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 1].quiver(
    u_his_EAM_rvalue.where(wind_his_EAM_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    v_his_EAM_rvalue.where(wind_his_EAM_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 1].format(ltitle="EAM index", rtitle="historical 200hPa")
# ===========================================

con = axs[1, 1].contourf(
    hgt_his_EAM_rvalue.sel(level=500.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_EAM_pvalue.sel(level=500.0),
    axs[1, 1],
    n,
    np.where(hgt_his_EAM_pvalue.sel(level=500.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1, 1].quiver(
    u_his_EAM_rvalue.sel(level=500.0)[::ski, ::ski],
    v_his_EAM_rvalue.sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1, 1].quiver(
    u_his_EAM_rvalue.where(wind_his_EAM_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    v_his_EAM_rvalue.where(wind_his_EAM_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1, 1].format(ltitle="EAM index", rtitle="historical 500hPa")
# ===================================================
con = axs[2, 1].contourf(
    hgt_his_EAM_rvalue.sel(level=850.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_EAM_pvalue.sel(level=850.0),
    axs[2, 1],
    n,
    np.where(hgt_his_EAM_pvalue.sel(level=850.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[2, 1].quiver(
    u_his_EAM_rvalue.sel(level=850.0)[::ski, ::ski],
    v_his_EAM_rvalue.sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[2, 1].quiver(
    u_his_EAM_rvalue.where(wind_his_EAM_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    v_his_EAM_rvalue.where(wind_his_EAM_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[2, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[2, 1].format(ltitle="EAM index", rtitle="historical 850hPa")
# ===================================================


fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")

# %%
#   plot the linear regression between monsoon index and hgt, u, v
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=2, nrows=3, proj=proj)

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
    rect = Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1
    )
    ax.add_patch(rect)

    x0 = 70
    y0 = 10
    width = 40
    height = 20
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0, 0].contourf(
    hgt_ERA5_SAM_rvalue.sel(level=200.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_SAM_pvalue.sel(level=200.0),
    axs[0, 0],
    n,
    np.where(hgt_ERA5_SAM_pvalue.sel(level=200.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 0].quiver(
    u_ERA5_SAM_rvalue.sel(level=200.0)[::ski, ::ski],
    v_ERA5_SAM_rvalue.sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 0].quiver(
    u_ERA5_SAM_rvalue.where(wind_ERA5_SAM_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    v_ERA5_SAM_rvalue.where(wind_ERA5_SAM_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 0].format(ltitle="SAM index", rtitle="ERA5 200hPa")
# ===========================================

con = axs[1, 0].contourf(
    hgt_ERA5_SAM_rvalue.sel(level=500.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_SAM_pvalue.sel(level=500.0),
    axs[1, 0],
    n,
    np.where(hgt_ERA5_SAM_pvalue.sel(level=500.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1, 0].quiver(
    u_ERA5_SAM_rvalue.sel(level=500.0)[::ski, ::ski],
    v_ERA5_SAM_rvalue.sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1, 0].quiver(
    u_ERA5_SAM_rvalue.where(wind_ERA5_SAM_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    v_ERA5_SAM_rvalue.where(wind_ERA5_SAM_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1, 0].format(ltitle="SAM index", rtitle="ERA5 500hPa")
# ===================================================
con = axs[2, 0].contourf(
    hgt_ERA5_SAM_rvalue.sel(level=850.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_SAM_pvalue.sel(level=850.0),
    axs[2, 0],
    n,
    np.where(hgt_ERA5_SAM_pvalue.sel(level=850.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[2, 0].quiver(
    u_ERA5_SAM_rvalue.sel(level=850.0)[::ski, ::ski],
    v_ERA5_SAM_rvalue.sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[2, 0].quiver(
    u_ERA5_SAM_rvalue.where(wind_ERA5_SAM_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    v_ERA5_SAM_rvalue.where(wind_ERA5_SAM_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[2, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[2, 0].format(ltitle="SAM index", rtitle="ERA5 850hPa")
# ===================================================
#   plot the historical run result
con = axs[0, 1].contourf(
    hgt_his_SAM_rvalue.sel(level=200.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_SAM_pvalue.sel(level=200.0),
    axs[0, 1],
    n,
    np.where(hgt_his_SAM_pvalue.sel(level=200.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 1].quiver(
    u_his_SAM_rvalue.sel(level=200.0)[::ski, ::ski],
    v_his_SAM_rvalue.sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 1].quiver(
    u_his_SAM_rvalue.where(wind_his_SAM_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    v_his_SAM_rvalue.where(wind_his_SAM_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 1].format(ltitle="SAM index", rtitle="historical 200hPa")
# ===========================================

con = axs[1, 1].contourf(
    hgt_his_SAM_rvalue.sel(level=500.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_SAM_pvalue.sel(level=500.0),
    axs[1, 1],
    n,
    np.where(hgt_his_SAM_pvalue.sel(level=500.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1, 1].quiver(
    u_his_SAM_rvalue.sel(level=500.0)[::ski, ::ski],
    v_his_SAM_rvalue.sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1, 1].quiver(
    u_his_SAM_rvalue.where(wind_his_SAM_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    v_his_SAM_rvalue.where(wind_his_SAM_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1, 1].format(ltitle="SAM index", rtitle="historical 500hPa")
# ===================================================
con = axs[2, 1].contourf(
    hgt_his_SAM_rvalue.sel(level=850.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_SAM_pvalue.sel(level=850.0),
    axs[2, 1],
    n,
    np.where(hgt_his_SAM_pvalue.sel(level=850.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[2, 1].quiver(
    u_his_SAM_rvalue.sel(level=850.0)[::ski, ::ski],
    v_his_SAM_rvalue.sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[2, 1].quiver(
    u_his_SAM_rvalue.where(wind_his_SAM_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    v_his_SAM_rvalue.where(wind_his_SAM_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[2, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[2, 1].format(ltitle="SAM index", rtitle="historical 850hPa")
# ===================================================


fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
#   plot the linear regression between monsoon index and hgt, u, v
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=2, nrows=3, proj=proj)

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
    rect = Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1
    )
    ax.add_patch(rect)

    x0 = 90
    y0 = 5
    width = 50
    height = 27.5
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0, 0].contourf(
    hgt_ERA5_IWF_rvalue.sel(level=200.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_IWF_pvalue.sel(level=200.0),
    axs[0, 0],
    n,
    np.where(hgt_ERA5_IWF_pvalue.sel(level=200.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 0].quiver(
    u_ERA5_IWF_rvalue.sel(level=200.0)[::ski, ::ski],
    v_ERA5_IWF_rvalue.sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 0].quiver(
    u_ERA5_IWF_rvalue.where(wind_ERA5_IWF_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    v_ERA5_IWF_rvalue.where(wind_ERA5_IWF_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 0].format(ltitle="IWF index", rtitle="ERA5 200hPa")
# ===========================================

con = axs[1, 0].contourf(
    hgt_ERA5_IWF_rvalue.sel(level=500.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_IWF_pvalue.sel(level=500.0),
    axs[1, 0],
    n,
    np.where(hgt_ERA5_IWF_pvalue.sel(level=500.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1, 0].quiver(
    u_ERA5_IWF_rvalue.sel(level=500.0)[::ski, ::ski],
    v_ERA5_IWF_rvalue.sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1, 0].quiver(
    u_ERA5_IWF_rvalue.where(wind_ERA5_IWF_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    v_ERA5_IWF_rvalue.where(wind_ERA5_IWF_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1, 0].format(ltitle="IWF index", rtitle="ERA5 500hPa")
# ===================================================
con = axs[2, 0].contourf(
    hgt_ERA5_IWF_rvalue.sel(level=850.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_IWF_pvalue.sel(level=850.0),
    axs[2, 0],
    n,
    np.where(hgt_ERA5_IWF_pvalue.sel(level=850.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[2, 0].quiver(
    u_ERA5_IWF_rvalue.sel(level=850.0)[::ski, ::ski],
    v_ERA5_IWF_rvalue.sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[2, 0].quiver(
    u_ERA5_IWF_rvalue.where(wind_ERA5_IWF_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    v_ERA5_IWF_rvalue.where(wind_ERA5_IWF_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[2, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[2, 0].format(ltitle="IWF index", rtitle="ERA5 850hPa")
# ===================================================
#   plot the historical run result
con = axs[0, 1].contourf(
    hgt_his_IWF_rvalue.sel(level=200.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_IWF_pvalue.sel(level=200.0),
    axs[0, 1],
    n,
    np.where(hgt_his_IWF_pvalue.sel(level=200.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 1].quiver(
    u_his_IWF_rvalue.sel(level=200.0)[::ski, ::ski],
    v_his_IWF_rvalue.sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 1].quiver(
    u_his_IWF_rvalue.where(wind_his_IWF_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    v_his_IWF_rvalue.where(wind_his_IWF_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 1].format(ltitle="IWF index", rtitle="historical 200hPa")
# ===========================================

con = axs[1, 1].contourf(
    hgt_his_IWF_rvalue.sel(level=500.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_IWF_pvalue.sel(level=500.0),
    axs[1, 1],
    n,
    np.where(hgt_his_IWF_pvalue.sel(level=500.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1, 1].quiver(
    u_his_IWF_rvalue.sel(level=500.0)[::ski, ::ski],
    v_his_IWF_rvalue.sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1, 1].quiver(
    u_his_IWF_rvalue.where(wind_his_IWF_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    v_his_IWF_rvalue.where(wind_his_IWF_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1, 1].format(ltitle="IWF index", rtitle="historical 500hPa")
# ===================================================
con = axs[2, 1].contourf(
    hgt_his_IWF_rvalue.sel(level=850.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_IWF_pvalue.sel(level=850.0),
    axs[2, 1],
    n,
    np.where(hgt_his_IWF_pvalue.sel(level=850.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[2, 1].quiver(
    u_his_IWF_rvalue.sel(level=850.0)[::ski, ::ski],
    v_his_IWF_rvalue.sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[2, 1].quiver(
    u_his_IWF_rvalue.where(wind_his_IWF_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    v_his_IWF_rvalue.where(wind_his_IWF_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[2, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[2, 1].format(ltitle="IWF index", rtitle="historical 850hPa")
# ===================================================


fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
#   plot the linear regression between monsoon index and hgt, u, v
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=2, nrows=3, proj=proj)

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
    rect = Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1
    )
    ax.add_patch(rect)

    x0 = 40.0
    y0 = 5.0
    width = 70.0
    height = 15.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0, 0].contourf(
    hgt_ERA5_WY_rvalue.sel(level=200.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_WY_pvalue.sel(level=200.0),
    axs[0, 0],
    n,
    np.where(hgt_ERA5_WY_pvalue.sel(level=200.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 0].quiver(
    u_ERA5_WY_rvalue.sel(level=200.0)[::ski, ::ski],
    v_ERA5_WY_rvalue.sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 0].quiver(
    u_ERA5_WY_rvalue.where(wind_ERA5_WY_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    v_ERA5_WY_rvalue.where(wind_ERA5_WY_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 0].format(ltitle="WY index", rtitle="ERA5 200hPa")
# ===========================================

con = axs[1, 0].contourf(
    hgt_ERA5_WY_rvalue.sel(level=500.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_WY_pvalue.sel(level=500.0),
    axs[1, 0],
    n,
    np.where(hgt_ERA5_WY_pvalue.sel(level=500.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1, 0].quiver(
    u_ERA5_WY_rvalue.sel(level=500.0)[::ski, ::ski],
    v_ERA5_WY_rvalue.sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1, 0].quiver(
    u_ERA5_WY_rvalue.where(wind_ERA5_WY_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    v_ERA5_WY_rvalue.where(wind_ERA5_WY_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1, 0].format(ltitle="WY index", rtitle="ERA5 500hPa")
# ===================================================
con = axs[2, 0].contourf(
    hgt_ERA5_WY_rvalue.sel(level=850.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_WY_pvalue.sel(level=850.0),
    axs[2, 0],
    n,
    np.where(hgt_ERA5_WY_pvalue.sel(level=850.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[2, 0].quiver(
    u_ERA5_WY_rvalue.sel(level=850.0)[::ski, ::ski],
    v_ERA5_WY_rvalue.sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[2, 0].quiver(
    u_ERA5_WY_rvalue.where(wind_ERA5_WY_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    v_ERA5_WY_rvalue.where(wind_ERA5_WY_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[2, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[2, 0].format(ltitle="WY index", rtitle="ERA5 850hPa")
# ===================================================
#   plot the historical run result
con = axs[0, 1].contourf(
    hgt_his_WY_rvalue.sel(level=200.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_WY_pvalue.sel(level=200.0),
    axs[0, 1],
    n,
    np.where(hgt_his_WY_pvalue.sel(level=200.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 1].quiver(
    u_his_WY_rvalue.sel(level=200.0)[::ski, ::ski],
    v_his_WY_rvalue.sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 1].quiver(
    u_his_WY_rvalue.where(wind_his_WY_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    v_his_WY_rvalue.where(wind_his_WY_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 1].format(ltitle="WY index", rtitle="historical 200hPa")
# ===========================================

con = axs[1, 1].contourf(
    hgt_his_WY_rvalue.sel(level=500.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_WY_pvalue.sel(level=500.0),
    axs[1, 1],
    n,
    np.where(hgt_his_WY_pvalue.sel(level=500.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1, 1].quiver(
    u_his_WY_rvalue.sel(level=500.0)[::ski, ::ski],
    v_his_WY_rvalue.sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1, 1].quiver(
    u_his_WY_rvalue.where(wind_his_WY_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    v_his_WY_rvalue.where(wind_his_WY_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1, 1].format(ltitle="WY index", rtitle="historical 500hPa")
# ===================================================
con = axs[2, 1].contourf(
    hgt_his_WY_rvalue.sel(level=850.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_WY_pvalue.sel(level=850.0),
    axs[2, 1],
    n,
    np.where(hgt_his_WY_pvalue.sel(level=850.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[2, 1].quiver(
    u_his_WY_rvalue.sel(level=850.0)[::ski, ::ski],
    v_his_WY_rvalue.sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[2, 1].quiver(
    u_his_WY_rvalue.where(wind_his_WY_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    v_his_WY_rvalue.where(wind_his_WY_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[2, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[2, 1].format(ltitle="WY index", rtitle="historical 850hPa")
# ===================================================


fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
print(uq_dpg_ERA5_JJA)
# %%
#   calculate the precipitation regress into IWF index
#   ERA5
preCRU_JJA.coords["time"] = ERA5_IWF_index.coords["time"]
preGPCP_JJA.coords["time"] = ERA5_IWF_index.coords["time"].sel(
    time=ERA5_IWF_index.time.dt.year >= 1979
)
prehis_JJA.coords["time"] = his_IWF_index.coords["time"]
(
    pre_CRU_IWF_slope,
    pre_CRU_IWF_intercept,
    pre_CRU_IWF_rvalue,
    pre_CRU_IWF_pvalue,
    pre_CRU_IWF_hypothesis,
) = ca.dim_linregress(ERA5_IWF_index, preCRU_JJA)

(
    pre_GPCP_IWF_slope,
    pre_GPCP_IWF_intercept,
    pre_GPCP_IWF_rvalue,
    pre_GPCP_IWF_pvalue,
    pre_GPCP_IWF_hypothesis,
) = ca.dim_linregress(
    ERA5_IWF_index.sel(time=ERA5_IWF_index.time.dt.year >= 1979), preGPCP_JJA
)

(
    uq_dpg_ERA5_IWF_slope,
    uq_dpg_ERA5_IWF_intercept,
    uq_dpg_ERA5_IWF_rvalue,
    uq_dpg_ERA5_IWF_pvalue,
    uq_dpg_ERA5_IWF_hypothesis,
) = ca.dim_linregress(ERA5_IWF_index, uq_dpg_ERA5_JJA)

(
    vq_dpg_ERA5_IWF_slope,
    vq_dpg_ERA5_IWF_intercept,
    vq_dpg_ERA5_IWF_rvalue,
    vq_dpg_ERA5_IWF_pvalue,
    vq_dpg_ERA5_IWF_hypothesis,
) = ca.dim_linregress(ERA5_IWF_index, vq_dpg_ERA5_JJA)

#   historical run
(
    pre_his_IWF_slope,
    pre_his_IWF_intercept,
    pre_his_IWF_rvalue,
    pre_his_IWF_pvalue,
    pre_his_IWF_hypothesis,
) = ca.dim_linregress(his_IWF_index, prehis_JJA)

(
    uq_dpg_his_IWF_slope,
    uq_dpg_his_IWF_intercept,
    uq_dpg_his_IWF_rvalue,
    uq_dpg_his_IWF_pvalue,
    uq_dpg_his_IWF_hypothesis,
) = ca.dim_linregress(his_IWF_index, uq_dpg_his_JJA)

(
    vq_dpg_his_IWF_slope,
    vq_dpg_his_IWF_intercept,
    vq_dpg_his_IWF_rvalue,
    vq_dpg_his_IWF_pvalue,
    vq_dpg_his_IWF_hypothesis,
) = ca.dim_linregress(his_IWF_index, vq_dpg_his_JJA)
# %%
#   check the uq and vq
uqvq_ERA5_IWF_mask = ca.wind_check(
    xr.where(uq_dpg_ERA5_IWF_pvalue <= 0.05, 1.0, 0.0),
    xr.where(vq_dpg_ERA5_IWF_pvalue <= 0.05, 1.0, 0.0),
    xr.where(uq_dpg_ERA5_IWF_pvalue <= 0.05, 1.0, 0.0),
    xr.where(vq_dpg_ERA5_IWF_pvalue <= 0.05, 1.0, 0.0),
)

uqvq_his_IWF_mask = ca.wind_check(
    xr.where(uq_dpg_his_IWF_pvalue <= 0.05, 1.0, 0.0),
    xr.where(vq_dpg_his_IWF_pvalue <= 0.05, 1.0, 0.0),
    xr.where(uq_dpg_his_IWF_pvalue <= 0.05, 1.0, 0.0),
    xr.where(vq_dpg_his_IWF_pvalue <= 0.05, 1.0, 0.0),
)
# %%
#   plot the precipitation and uqvq
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=3, nrows=1, proj=proj)

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
    rect = Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1
    )
    ax.add_patch(rect)
    # region 1
    x0 = 90
    y0 = 5
    width = 50
    height = 27.5
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0, 0].contourf(
    pre_CRU_IWF_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    pre_CRU_IWF_pvalue,
    axs[0, 0],
    n,
    np.where(pre_CRU_IWF_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 0].quiver(
    uq_dpg_ERA5_IWF_rvalue[::ski, ::ski],
    vq_dpg_ERA5_IWF_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 0].quiver(
    uq_dpg_ERA5_IWF_rvalue.where(uqvq_ERA5_IWF_mask > 0.0)[::ski, ::ski],
    vq_dpg_ERA5_IWF_rvalue.where(uqvq_ERA5_IWF_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 0].format(ltitle="CRU & ERA5", rtitle="precip&Uq reg IWF")
# ===================================================
con = axs[0, 1].contourf(
    pre_GPCP_IWF_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    pre_GPCP_IWF_pvalue,
    axs[0, 1],
    n,
    np.where(pre_GPCP_IWF_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 1].quiver(
    uq_dpg_ERA5_IWF_rvalue[::ski, ::ski],
    vq_dpg_ERA5_IWF_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 1].quiver(
    uq_dpg_ERA5_IWF_rvalue.where(uqvq_ERA5_IWF_mask > 0.0)[::ski, ::ski],
    vq_dpg_ERA5_IWF_rvalue.where(uqvq_ERA5_IWF_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 1].format(ltitle="GPCP & ERA5", rtitle="precip&Uq reg IWF")
# ===================================================
con = axs[0, 2].contourf(
    pre_his_IWF_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    pre_his_IWF_pvalue,
    axs[0, 2],
    n,
    np.where(pre_his_IWF_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 2].quiver(
    uq_dpg_his_IWF_rvalue[::ski, ::ski],
    vq_dpg_his_IWF_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 2].quiver(
    uq_dpg_his_IWF_rvalue.where(uqvq_his_IWF_mask > 0.0)[::ski, ::ski],
    vq_dpg_his_IWF_rvalue.where(uqvq_his_IWF_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 2].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 2].format(ltitle="historical", rtitle="precip&Uq reg IWF")
# ===================================================
fig.colorbar(con, loc="b", width=0.13, length=0.5, label="")
fig.format(abc="(a)", abcloc="l")
# %%
#   calculate the wind shear
ushearERA5_JJA = uERA5_ver_JJA.sel(level=850.0) - uERA5_ver_JJA.sel(level=200.0)
vshearERA5_JJA = vERA5_ver_JJA.sel(level=850.0) - vERA5_ver_JJA.sel(level=200.0)
ushearERA5_JJA = ca.detrend_dim(ushearERA5_JJA, "time", deg=1, demean=False)
vshearERA5_JJA = ca.detrend_dim(vshearERA5_JJA, "time", deg=1, demean=False)

ushearhis_JJA = uhis_ver_JJA.sel(level=850.0) - uhis_ver_JJA.sel(level=200.0)
vshearhis_JJA = vhis_ver_JJA.sel(level=850.0) - vhis_ver_JJA.sel(level=200.0)
ushearhis_JJA = ca.detrend_dim(ushearhis_JJA, "time", deg=1, demean=False)
vshearhis_JJA = ca.detrend_dim(vshearhis_JJA, "time", deg=1, demean=False)

# %%
#   calculate the vorticity
windERA5_JJA = VectorWind(
    uERA5_ver_JJA.sel(level=850.0), vERA5_ver_JJA.sel(level=850.0)
)
vorERA5_JJA = windERA5_JJA.vorticity()
vorERA5_JJA = ca.detrend_dim(vorERA5_JJA, "time", deg=1, demean=False)

windhis_JJA = VectorWind(uhis_ver_JJA.sel(level=850.0), vhis_ver_JJA.sel(level=850.0))
vorhis_JJA = windhis_JJA.vorticity()
vorhis_JJA = ca.detrend_dim(vorhis_JJA, "time", deg=1, demean=False)
# %%
#   calculate the regression of windshear and windvorticity into the IWF
#   ERA5
(
    ushear_ERA5_IWF_slope,
    ushear_ERA5_IWF_intercept,
    ushear_ERA5_IWF_rvalue,
    ushear_ERA5_IWF_pvalue,
    ushear_ERA5_IWF_hypothesis,
) = ca.dim_linregress(ERA5_IWF_index, ushearERA5_JJA)

(
    vshear_ERA5_IWF_slope,
    vshear_ERA5_IWF_intercept,
    vshear_ERA5_IWF_rvalue,
    vshear_ERA5_IWF_pvalue,
    vshear_ERA5_IWF_hypothesis,
) = ca.dim_linregress(ERA5_IWF_index, vshearERA5_JJA)

(
    vor_ERA5_IWF_slope,
    vor_ERA5_IWF_intercept,
    vor_ERA5_IWF_rvalue,
    vor_ERA5_IWF_pvalue,
    vor_ERA5_IWF_hypothesis,
) = ca.dim_linregress(ERA5_IWF_index, vorERA5_JJA)
#   historical run
(
    ushear_his_IWF_slope,
    ushear_his_IWF_intercept,
    ushear_his_IWF_rvalue,
    ushear_his_IWF_pvalue,
    ushear_his_IWF_hypothesis,
) = ca.dim_linregress(his_IWF_index, ushearhis_JJA)

(
    vshear_his_IWF_slope,
    vshear_his_IWF_intercept,
    vshear_his_IWF_rvalue,
    vshear_his_IWF_pvalue,
    vshear_his_IWF_hypothesis,
) = ca.dim_linregress(his_IWF_index, vshearhis_JJA)

(
    vor_his_IWF_slope,
    vor_his_IWF_intercept,
    vor_his_IWF_rvalue,
    vor_his_IWF_pvalue,
    vor_his_IWF_hypothesis,
) = ca.dim_linregress(his_IWF_index, vorhis_JJA)

# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=2, nrows=3, proj=proj)

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
    x0 = 90.0
    y0 = 5.0
    width = 50.0
    height = 27.5
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0, 0].contourf(
    ushear_ERA5_IWF_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    ushear_ERA5_IWF_pvalue,
    axs[0, 0],
    n,
    np.where(ushear_ERA5_IWF_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[0, 0].format(ltitle="ERA5", rtitle="U850-U200 reg IWF")
# ===================================================
con = axs[1, 0].contourf(
    vshear_ERA5_IWF_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    vshear_ERA5_IWF_pvalue,
    axs[1, 0],
    n,
    np.where(vshear_ERA5_IWF_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[1, 0].format(ltitle="ERA5", rtitle="V850-V200 reg IWF")
# ===================================================
con = axs[2, 0].contourf(
    vor_ERA5_IWF_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    vor_ERA5_IWF_pvalue,
    axs[2, 0],
    n,
    np.where(vor_ERA5_IWF_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[2, 0].format(ltitle="ERA5", rtitle="vorticity reg IWF")
# ===================================================
#   historical run
con = axs[0, 1].contourf(
    ushear_his_IWF_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    ushear_his_IWF_pvalue,
    axs[0, 1],
    n,
    np.where(ushear_his_IWF_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[0, 1].format(ltitle="historical", rtitle="U850-U200 reg IWF")
# ===================================================
con = axs[1, 1].contourf(
    vshear_his_IWF_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    vshear_his_IWF_pvalue,
    axs[1, 1],
    n,
    np.where(vshear_his_IWF_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[1, 1].format(ltitle="historical", rtitle="V850-V200 reg IWF")
# ===================================================
con = axs[2, 1].contourf(
    vor_his_IWF_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    vor_his_IWF_pvalue,
    axs[2, 1],
    n,
    np.where(vor_his_IWF_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[2, 1].format(ltitle="historical", rtitle="vorticity reg IWF")
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
#   calculate the Indian JJA mean rainfall and its water vapor source mean
preCRU_India_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.loc[:, 8:28, 70:86]).mean(
    dim="lon", skipna=True
)
preCRU_India_JJA = ca.detrend_dim(preCRU_India_JJA, "time", deg=1, demean=False)
preGPCP_India_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.loc[:, 8:28, 70:86]).mean(
    dim="lon", skipna=True
)
preGPCP_India_JJA = ca.detrend_dim(preGPCP_India_JJA, "time", deg=1, demean=False)
uq_dpg_ERA5_India_JJA = ca.cal_lat_weighted_mean(
    uq_dpg_ERA5_JJA.loc[:, 5:25, 50:80]
).mean(dim="lon", skipna=True)
uq_dpg_ERA5_India_JJA = ca.detrend_dim(
    uq_dpg_ERA5_India_JJA, "time", deg=1, demean=False
)
prehis_India_JJA = ca.cal_lat_weighted_mean(prehis_JJA.loc[:, 8:28, 70:86]).mean(
    dim="lon", skipna=True
)
prehis_India_JJA = ca.detrend_dim(prehis_India_JJA, "time", deg=1, demean=False)
uq_dpg_his_India_JJA = ca.cal_lat_weighted_mean(
    uq_dpg_his_JJA.loc[:, 5:25, 50:80]
).mean(dim="lon", skipna=True)
uq_dpg_his_India_JJA = ca.detrend_dim(uq_dpg_his_India_JJA, "time", deg=1, demean=False)

# %%
#   calculate the correlation of windshear and vorticity to Indian precipitation and water vapor source
preCRU_India_JJA.coords["time"] = ushearERA5_JJA.coords["time"]
preGPCP_India_JJA.coords["time"] = ushearERA5_JJA.coords["time"].sel(
    time=ERA5_IWF_index.time.dt.year >= 1979
)
prehis_India_JJA.coords["time"] = ushearhis_JJA.coords["time"]
(
    ushear_CRU_India_slope,
    ushear_CRU_India_intercept,
    ushear_CRU_India_rvalue,
    ushear_CRU_India_pvalue,
    ushear_CRU_India_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA, ushearERA5_JJA)

(
    ushear_GPCP_India_slope,
    ushear_GPCP_India_intercept,
    ushear_GPCP_India_rvalue,
    ushear_GPCP_India_pvalue,
    ushear_GPCP_India_hypothesis,
) = ca.dim_linregress(
    preGPCP_India_JJA, ushearERA5_JJA.sel(time=ushearERA5_JJA.time.dt.year >= 1979)
)

(
    ushear_his_India_slope,
    ushear_his_India_intercept,
    ushear_his_India_rvalue,
    ushear_his_India_pvalue,
    ushear_his_India_hypothesis,
) = ca.dim_linregress(prehis_India_JJA, ushearhis_JJA)

(
    vshear_CRU_India_slope,
    vshear_CRU_India_intercept,
    vshear_CRU_India_rvalue,
    vshear_CRU_India_pvalue,
    vshear_CRU_India_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA, vshearERA5_JJA)

(
    vshear_GPCP_India_slope,
    vshear_GPCP_India_intercept,
    vshear_GPCP_India_rvalue,
    vshear_GPCP_India_pvalue,
    vshear_GPCP_India_hypothesis,
) = ca.dim_linregress(
    preGPCP_India_JJA, vshearERA5_JJA.sel(time=vshearERA5_JJA.time.dt.year >= 1979)
)

(
    vshear_his_India_slope,
    vshear_his_India_intercept,
    vshear_his_India_rvalue,
    vshear_his_India_pvalue,
    vshear_his_India_hypothesis,
) = ca.dim_linregress(prehis_India_JJA, vshearhis_JJA)

(
    vor_CRU_India_slope,
    vor_CRU_India_intercept,
    vor_CRU_India_rvalue,
    vor_CRU_India_pvalue,
    vor_CRU_India_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA, vorERA5_JJA)

(
    vor_GPCP_India_slope,
    vor_GPCP_India_intercept,
    vor_GPCP_India_rvalue,
    vor_GPCP_India_pvalue,
    vor_GPCP_India_hypothesis,
) = ca.dim_linregress(
    preGPCP_India_JJA, vorERA5_JJA.sel(time=vorERA5_JJA.time.dt.year >= 1979)
)

(
    vor_his_India_slope,
    vor_his_India_intercept,
    vor_his_India_rvalue,
    vor_his_India_pvalue,
    vor_his_India_hypothesis,
) = ca.dim_linregress(prehis_India_JJA, vorhis_JJA)

# %%
(
    ushear_ERA5_uqIndia_slope,
    ushear_ERA5_uqIndia_intercept,
    ushear_ERA5_uqIndia_rvalue,
    ushear_ERA5_uqIndia_pvalue,
    ushear_ERA5_uqIndia_hypothesis,
) = ca.dim_linregress(uq_dpg_ERA5_India_JJA, ushearERA5_JJA)

(
    ushear_his_uqIndia_slope,
    ushear_his_uqIndia_intercept,
    ushear_his_uqIndia_rvalue,
    ushear_his_uqIndia_pvalue,
    ushear_his_uqIndia_hypothesis,
) = ca.dim_linregress(uq_dpg_his_India_JJA, ushearhis_JJA)

(
    vshear_ERA5_uqIndia_slope,
    vshear_ERA5_uqIndia_intercept,
    vshear_ERA5_uqIndia_rvalue,
    vshear_ERA5_uqIndia_pvalue,
    vshear_ERA5_uqIndia_hypothesis,
) = ca.dim_linregress(uq_dpg_ERA5_India_JJA, vshearERA5_JJA)

(
    vshear_his_uqIndia_slope,
    vshear_his_uqIndia_intercept,
    vshear_his_uqIndia_rvalue,
    vshear_his_uqIndia_pvalue,
    vshear_his_uqIndia_hypothesis,
) = ca.dim_linregress(uq_dpg_his_India_JJA, vshearhis_JJA)

(
    vor_ERA5_uqIndia_slope,
    vor_ERA5_uqIndia_intercept,
    vor_ERA5_uqIndia_rvalue,
    vor_ERA5_uqIndia_pvalue,
    vor_ERA5_uqIndia_hypothesis,
) = ca.dim_linregress(uq_dpg_ERA5_India_JJA, vorERA5_JJA)

(
    vor_his_uqIndia_slope,
    vor_his_uqIndia_intercept,
    vor_his_uqIndia_rvalue,
    vor_his_uqIndia_pvalue,
    vor_his_uqIndia_hypothesis,
) = ca.dim_linregress(uq_dpg_his_India_JJA, vorhis_JJA)
# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=3, nrows=3, proj=proj)

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
    x0 = 70
    y0 = 8.0
    width = 16.0
    height = 20
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0, 0].contourf(
    ushear_CRU_India_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    ushear_CRU_India_pvalue,
    axs[0, 0],
    n,
    np.where(ushear_CRU_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[0, 0].format(ltitle="CRU TS4.01 & ERA5", rtitle="U850-U200 reg IndR")
# ===================================================
con = axs[1, 0].contourf(
    vshear_CRU_India_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    vshear_CRU_India_pvalue,
    axs[1, 0],
    n,
    np.where(vshear_CRU_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[1, 0].format(ltitle="CRU TS4.01 & ERA5", rtitle="V850-V200 reg IndR")
# ===================================================
con = axs[2, 0].contourf(
    vor_CRU_India_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    vor_CRU_India_pvalue,
    axs[2, 0],
    n,
    np.where(vor_CRU_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[2, 0].format(ltitle="CRU TS4.01 & ERA5", rtitle="vorticity reg IndR")
# ===================================================
con = axs[0, 1].contourf(
    ushear_GPCP_India_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    ushear_GPCP_India_pvalue,
    axs[0, 1],
    n,
    np.where(ushear_GPCP_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[0, 1].format(ltitle="GPCP & ERA5", rtitle="U850-U200 reg IndR")
# ===================================================
con = axs[1, 1].contourf(
    vshear_GPCP_India_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    vshear_GPCP_India_pvalue,
    axs[1, 1],
    n,
    np.where(vshear_GPCP_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[1, 1].format(ltitle="GPCP & ERA5", rtitle="V850-V200 reg IndR")
# ===================================================
con = axs[2, 1].contourf(
    vor_GPCP_India_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    vor_GPCP_India_pvalue,
    axs[2, 1],
    n,
    np.where(vor_GPCP_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[2, 1].format(ltitle="GPCP & ERA5", rtitle="vorticity reg IndR")
# ===================================================
con = axs[0, 2].contourf(
    ushear_his_India_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    ushear_his_India_pvalue,
    axs[0, 2],
    n,
    np.where(ushear_his_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[0, 2].format(ltitle="historical & ERA5", rtitle="U850-U200 reg IndR")
# ===================================================
con = axs[1, 2].contourf(
    vshear_his_India_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    vshear_his_India_pvalue,
    axs[1, 2],
    n,
    np.where(vshear_his_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[1, 2].format(ltitle="historical & ERA5", rtitle="V850-V200 reg IndR")
# ===================================================
con = axs[2, 2].contourf(
    vor_his_India_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    vor_his_India_pvalue,
    axs[2, 2],
    n,
    np.where(vor_his_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[2, 2].format(ltitle="historical & ERA5", rtitle="vorticity reg IndR")


fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=2, nrows=3, proj=proj)

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
    x0 = 50
    y0 = 5.0
    width = 30.0
    height = 15
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0, 0].contourf(
    ushear_ERA5_uqIndia_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    ushear_ERA5_uqIndia_pvalue,
    axs[0, 0],
    n,
    np.where(ushear_ERA5_uqIndia_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[0, 0].format(ltitle="ERA5", rtitle="U850-U200 reg uq")
# ===================================================
con = axs[1, 0].contourf(
    vshear_ERA5_uqIndia_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    vshear_ERA5_uqIndia_pvalue,
    axs[1, 0],
    n,
    np.where(vshear_ERA5_uqIndia_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[1, 0].format(ltitle="ERA5", rtitle="V850-V200 reg uq")

# ===================================================
con = axs[2, 0].contourf(
    vor_ERA5_uqIndia_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    vor_ERA5_uqIndia_pvalue,
    axs[2, 0],
    n,
    np.where(vor_ERA5_uqIndia_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[2, 0].format(ltitle="ERA5", rtitle="vorticity reg uq")
# ===================================================
con = axs[0, 1].contourf(
    ushear_his_uqIndia_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    ushear_his_uqIndia_pvalue,
    axs[0, 1],
    n,
    np.where(ushear_his_uqIndia_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[0, 1].format(ltitle="historical", rtitle="U850-U200 reg uq")
# ===================================================
con = axs[1, 1].contourf(
    vshear_his_uqIndia_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    vshear_his_uqIndia_pvalue,
    axs[1, 1],
    n,
    np.where(vshear_his_uqIndia_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[1, 1].format(ltitle="historical", rtitle="V850-V200 reg uq")

# ===================================================
con = axs[2, 1].contourf(
    vor_his_uqIndia_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    vor_his_uqIndia_pvalue,
    axs[2, 1],
    n,
    np.where(vor_his_uqIndia_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[2, 1].format(ltitle="historical", rtitle="vorticity reg uq")

fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
reload(ca)
#   calculate the NEWI index
ERA5_NEWI_index = ca.NEWI(uERA5_ver_JJA)
his_NEWI_index = ca.NEWI(uhis_ver_JJA)
ERA5_NEWI_index = ca.detrend_dim(ERA5_NEWI_index, "time", deg=1, demean=False)
his_NEWI_index = ca.detrend_dim(his_NEWI_index, "time", deg=1, demean=False)

ERA5_IndR_NEWI_regress = stats.linregress(uq_dpg_ERA5_India_JJA, ERA5_NEWI_index)
his_IndR_NEWI_regress = stats.linregress(uq_dpg_his_India_JJA, his_NEWI_index)

print(ERA5_IndR_NEWI_regress, his_IndR_NEWI_regress)
# %%
(
    hgt_ERA5_NEWI_slope,
    hgt_ERA5_NEWI_intercept,
    hgt_ERA5_NEWI_rvalue,
    hgt_ERA5_NEWI_pvalue,
    hgt_ERA5_NEWI_hypothesis,
) = ca.dim_linregress(ERA5_NEWI_index, hgtERA5_ver_JJA)
(
    u_ERA5_NEWI_slope,
    u_ERA5_NEWI_intercept,
    u_ERA5_NEWI_rvalue,
    u_ERA5_NEWI_pvalue,
    u_ERA5_NEWI_hypothesis,
) = ca.dim_linregress(ERA5_NEWI_index, uERA5_ver_JJA)
(
    v_ERA5_NEWI_slope,
    v_ERA5_NEWI_intercept,
    v_ERA5_NEWI_rvalue,
    v_ERA5_NEWI_pvalue,
    v_ERA5_NEWI_hypothesis,
) = ca.dim_linregress(ERA5_NEWI_index, vERA5_ver_JJA)

(
    hgt_his_NEWI_slope,
    hgt_his_NEWI_intercept,
    hgt_his_NEWI_rvalue,
    hgt_his_NEWI_pvalue,
    hgt_his_NEWI_hypothesis,
) = ca.dim_linregress(his_NEWI_index, hgthis_ver_JJA)
(
    u_his_NEWI_slope,
    u_his_NEWI_intercept,
    u_his_NEWI_rvalue,
    u_his_NEWI_pvalue,
    u_his_NEWI_hypothesis,
) = ca.dim_linregress(his_NEWI_index, uhis_ver_JJA)
(
    v_his_NEWI_slope,
    v_his_NEWI_intercept,
    v_his_NEWI_rvalue,
    v_his_NEWI_pvalue,
    v_his_NEWI_hypothesis,
) = ca.dim_linregress(his_NEWI_index, vhis_ver_JJA)
# %%
wind_ERA5_NEWI_mask = ca.wind_check(
    xr.where(u_ERA5_NEWI_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_NEWI_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_ERA5_NEWI_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_NEWI_pvalue <= 0.05, 1.0, 0.0),
)
wind_his_NEWI_mask = ca.wind_check(
    xr.where(u_his_NEWI_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_NEWI_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_NEWI_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_NEWI_pvalue <= 0.05, 1.0, 0.0),
)
# %%
#   plot the linear regression between monsoon index and hgt, u, v
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=2, nrows=3, proj=proj)

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
    rect = Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1
    )
    ax.add_patch(rect)
    # region 1
    x0 = 105
    y0 = 2.5
    width = 35
    height = 7.5
    patches(ax, x0 - cl, y0, width, height, proj)
    # region 2
    x0 = 105
    y0 = 17.5
    width = 35
    height = 5.0
    patches(ax, x0 - cl, y0, width, height, proj)
    # region 3
    x0 = 105
    y0 = 30
    width = 35
    height = 7.5
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0, 0].contourf(
    hgt_ERA5_NEWI_rvalue.sel(level=200.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_NEWI_pvalue.sel(level=200.0),
    axs[0, 0],
    n,
    np.where(hgt_ERA5_NEWI_pvalue.sel(level=200.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 0].quiver(
    u_ERA5_NEWI_rvalue.sel(level=200.0)[::ski, ::ski],
    v_ERA5_NEWI_rvalue.sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 0].quiver(
    u_ERA5_NEWI_rvalue.where(wind_ERA5_NEWI_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    v_ERA5_NEWI_rvalue.where(wind_ERA5_NEWI_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 0].format(ltitle="NEWI index", rtitle="ERA5 200hPa")
# ===========================================

con = axs[1, 0].contourf(
    hgt_ERA5_NEWI_rvalue.sel(level=500.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_NEWI_pvalue.sel(level=500.0),
    axs[1, 0],
    n,
    np.where(hgt_ERA5_NEWI_pvalue.sel(level=500.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1, 0].quiver(
    u_ERA5_NEWI_rvalue.sel(level=500.0)[::ski, ::ski],
    v_ERA5_NEWI_rvalue.sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1, 0].quiver(
    u_ERA5_NEWI_rvalue.where(wind_ERA5_NEWI_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    v_ERA5_NEWI_rvalue.where(wind_ERA5_NEWI_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1, 0].format(ltitle="NEWI index", rtitle="ERA5 500hPa")
# ===================================================
con = axs[2, 0].contourf(
    hgt_ERA5_NEWI_rvalue.sel(level=850.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_NEWI_pvalue.sel(level=850.0),
    axs[2, 0],
    n,
    np.where(hgt_ERA5_NEWI_pvalue.sel(level=850.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[2, 0].quiver(
    u_ERA5_NEWI_rvalue.sel(level=850.0)[::ski, ::ski],
    v_ERA5_NEWI_rvalue.sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[2, 0].quiver(
    u_ERA5_NEWI_rvalue.where(wind_ERA5_NEWI_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    v_ERA5_NEWI_rvalue.where(wind_ERA5_NEWI_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[2, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[2, 0].format(ltitle="NEWI index", rtitle="ERA5 850hPa")
# ===================================================
#   plot the historical run result
con = axs[0, 1].contourf(
    hgt_his_NEWI_rvalue.sel(level=200.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_NEWI_pvalue.sel(level=200.0),
    axs[0, 1],
    n,
    np.where(hgt_his_NEWI_pvalue.sel(level=200.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 1].quiver(
    u_his_NEWI_rvalue.sel(level=200.0)[::ski, ::ski],
    v_his_NEWI_rvalue.sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 1].quiver(
    u_his_NEWI_rvalue.where(wind_his_NEWI_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    v_his_NEWI_rvalue.where(wind_his_NEWI_mask > 0.0).sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 1].format(ltitle="NEWI index", rtitle="historical 200hPa")
# ===========================================

con = axs[1, 1].contourf(
    hgt_his_NEWI_rvalue.sel(level=500.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_NEWI_pvalue.sel(level=500.0),
    axs[1, 1],
    n,
    np.where(hgt_his_NEWI_pvalue.sel(level=500.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1, 1].quiver(
    u_his_NEWI_rvalue.sel(level=500.0)[::ski, ::ski],
    v_his_NEWI_rvalue.sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1, 1].quiver(
    u_his_NEWI_rvalue.where(wind_his_NEWI_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    v_his_NEWI_rvalue.where(wind_his_NEWI_mask > 0.0).sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1, 1].format(ltitle="NEWI index", rtitle="historical 500hPa")
# ===================================================
con = axs[2, 1].contourf(
    hgt_his_NEWI_rvalue.sel(level=850.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_NEWI_pvalue.sel(level=850.0),
    axs[2, 1],
    n,
    np.where(hgt_his_NEWI_pvalue.sel(level=850.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[2, 1].quiver(
    u_his_NEWI_rvalue.sel(level=850.0)[::ski, ::ski],
    v_his_NEWI_rvalue.sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[2, 1].quiver(
    u_his_NEWI_rvalue.where(wind_his_NEWI_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    v_his_NEWI_rvalue.where(wind_his_NEWI_mask > 0.0).sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[2, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[2, 1].format(ltitle="NEWI index", rtitle="historical 850hPa")
# ===================================================


fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
(
    hgt_ERA5_India_uq_slope,
    hgt_ERA5_India_uq_intercept,
    hgt_ERA5_India_uq_rvalue,
    hgt_ERA5_India_uq_pvalue,
    hgt_ERA5_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_ERA5_India_JJA, hgtERA5_ver_JJA)
(
    u_ERA5_India_uq_slope,
    u_ERA5_India_uq_intercept,
    u_ERA5_India_uq_rvalue,
    u_ERA5_India_uq_pvalue,
    u_ERA5_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_ERA5_India_JJA, uERA5_ver_JJA)
(
    v_ERA5_India_uq_slope,
    v_ERA5_India_uq_intercept,
    v_ERA5_India_uq_rvalue,
    v_ERA5_India_uq_pvalue,
    v_ERA5_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_ERA5_India_JJA, vERA5_ver_JJA)

(
    hgt_his_India_uq_slope,
    hgt_his_India_uq_intercept,
    hgt_his_India_uq_rvalue,
    hgt_his_India_uq_pvalue,
    hgt_his_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_his_India_JJA, hgthis_ver_JJA)
(
    u_his_India_uq_slope,
    u_his_India_uq_intercept,
    u_his_India_uq_rvalue,
    u_his_India_uq_pvalue,
    u_his_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_his_India_JJA, uhis_ver_JJA)
(
    v_his_India_uq_slope,
    v_his_India_uq_intercept,
    v_his_India_uq_rvalue,
    v_his_India_uq_pvalue,
    v_his_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_his_India_JJA, vhis_ver_JJA)
# %%
wind_ERA5_India_uq_mask = ca.wind_check(
    xr.where(u_ERA5_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_ERA5_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_India_uq_pvalue <= 0.05, 1.0, 0.0),
)
wind_his_India_uq_mask = ca.wind_check(
    xr.where(u_his_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_India_uq_pvalue <= 0.05, 1.0, 0.0),
)
# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=2, nrows=3, proj=proj)

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
    rect = Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1
    )
    ax.add_patch(rect)
    # region 1
    x0 = 50
    y0 = 5.0
    width = 30
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0, 0].contourf(
    hgt_ERA5_India_uq_rvalue.sel(level=200.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_India_uq_pvalue.sel(level=200.0),
    axs[0, 0],
    n,
    np.where(hgt_ERA5_India_uq_pvalue.sel(level=200.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 0].quiver(
    u_ERA5_India_uq_rvalue.sel(level=200.0)[::ski, ::ski],
    v_ERA5_India_uq_rvalue.sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 0].quiver(
    u_ERA5_India_uq_rvalue.where(wind_ERA5_India_uq_mask > 0.0).sel(level=200.0)[
        ::ski, ::ski
    ],
    v_ERA5_India_uq_rvalue.where(wind_ERA5_India_uq_mask > 0.0).sel(level=200.0)[
        ::ski, ::ski
    ],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 0].format(ltitle="India u index", rtitle="ERA5 200hPa")
# ===========================================

con = axs[1, 0].contourf(
    hgt_ERA5_India_uq_rvalue.sel(level=500.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_India_uq_pvalue.sel(level=500.0),
    axs[1, 0],
    n,
    np.where(hgt_ERA5_India_uq_pvalue.sel(level=500.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1, 0].quiver(
    u_ERA5_India_uq_rvalue.sel(level=500.0)[::ski, ::ski],
    v_ERA5_India_uq_rvalue.sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1, 0].quiver(
    u_ERA5_India_uq_rvalue.where(wind_ERA5_India_uq_mask > 0.0).sel(level=500.0)[
        ::ski, ::ski
    ],
    v_ERA5_India_uq_rvalue.where(wind_ERA5_India_uq_mask > 0.0).sel(level=500.0)[
        ::ski, ::ski
    ],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1, 0].format(ltitle="India uq index", rtitle="ERA5 500hPa")
# ===================================================
con = axs[2, 0].contourf(
    hgt_ERA5_India_uq_rvalue.sel(level=850.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_India_uq_pvalue.sel(level=850.0),
    axs[2, 0],
    n,
    np.where(hgt_ERA5_India_uq_pvalue.sel(level=850.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[2, 0].quiver(
    u_ERA5_India_uq_rvalue.sel(level=850.0)[::ski, ::ski],
    v_ERA5_India_uq_rvalue.sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[2, 0].quiver(
    u_ERA5_India_uq_rvalue.where(wind_ERA5_India_uq_mask > 0.0).sel(level=850.0)[
        ::ski, ::ski
    ],
    v_ERA5_India_uq_rvalue.where(wind_ERA5_India_uq_mask > 0.0).sel(level=850.0)[
        ::ski, ::ski
    ],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[2, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[2, 0].format(ltitle="India uq index", rtitle="ERA5 850hPa")
# ===================================================
#   plot the historical run result
con = axs[0, 1].contourf(
    hgt_his_India_uq_rvalue.sel(level=200.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_India_uq_pvalue.sel(level=200.0),
    axs[0, 1],
    n,
    np.where(hgt_his_India_uq_pvalue.sel(level=200.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 1].quiver(
    u_his_India_uq_rvalue.sel(level=200.0)[::ski, ::ski],
    v_his_India_uq_rvalue.sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 1].quiver(
    u_his_India_uq_rvalue.where(wind_his_India_uq_mask > 0.0).sel(level=200.0)[
        ::ski, ::ski
    ],
    v_his_India_uq_rvalue.where(wind_his_India_uq_mask > 0.0).sel(level=200.0)[
        ::ski, ::ski
    ],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 1].format(ltitle="India uq index", rtitle="historical 200hPa")
# ===========================================

con = axs[1, 1].contourf(
    hgt_his_India_uq_rvalue.sel(level=500.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_India_uq_pvalue.sel(level=500.0),
    axs[1, 1],
    n,
    np.where(hgt_his_India_uq_pvalue.sel(level=500.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1, 1].quiver(
    u_his_India_uq_rvalue.sel(level=500.0)[::ski, ::ski],
    v_his_India_uq_rvalue.sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1, 1].quiver(
    u_his_India_uq_rvalue.where(wind_his_India_uq_mask > 0.0).sel(level=500.0)[
        ::ski, ::ski
    ],
    v_his_India_uq_rvalue.where(wind_his_India_uq_mask > 0.0).sel(level=500.0)[
        ::ski, ::ski
    ],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1, 1].format(ltitle="India uq index", rtitle="historical 500hPa")
# ===================================================
con = axs[2, 1].contourf(
    hgt_his_India_uq_rvalue.sel(level=850.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_India_uq_pvalue.sel(level=850.0),
    axs[2, 1],
    n,
    np.where(hgt_his_India_uq_pvalue.sel(level=850.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[2, 1].quiver(
    u_his_India_uq_rvalue.sel(level=850.0)[::ski, ::ski],
    v_his_India_uq_rvalue.sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[2, 1].quiver(
    u_his_India_uq_rvalue.where(wind_his_India_uq_mask > 0.0).sel(level=850.0)[
        ::ski, ::ski
    ],
    v_his_India_uq_rvalue.where(wind_his_India_uq_mask > 0.0).sel(level=850.0)[
        ::ski, ::ski
    ],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[2, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[2, 1].format(ltitle="India uq index", rtitle="historical 850hPa")
# ===================================================


fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
uERA5_ver_India_JJA = ca.cal_lat_weighted_mean(
    uERA5_ver_JJA.sel(level=200.0).loc[:, 5:25, 50:80]
).mean(dim="lon", skipna=True)
uERA5_ver_India_JJA = ca.detrend_dim(uERA5_ver_India_JJA, "time", deg=1, demean=False)
uhis_ver_India_JJA = ca.cal_lat_weighted_mean(
    uhis_ver_JJA.sel(level=200.0).loc[:, 5:25, 50:80]
).mean(dim="lon", skipna=True)
uhis_ver_India_JJA = ca.detrend_dim(uhis_ver_India_JJA, "time", deg=1, demean=False)
# %%
(
    hgt_ERA5_India_u_slope,
    hgt_ERA5_India_u_intercept,
    hgt_ERA5_India_u_rvalue,
    hgt_ERA5_India_u_pvalue,
    hgt_ERA5_India_u_hypothesis,
) = ca.dim_linregress(uERA5_ver_India_JJA, hgtERA5_ver_JJA)
(
    u_ERA5_India_u_slope,
    u_ERA5_India_u_intercept,
    u_ERA5_India_u_rvalue,
    u_ERA5_India_u_pvalue,
    u_ERA5_India_u_hypothesis,
) = ca.dim_linregress(uERA5_ver_India_JJA, uERA5_ver_JJA)
(
    v_ERA5_India_u_slope,
    v_ERA5_India_u_intercept,
    v_ERA5_India_u_rvalue,
    v_ERA5_India_u_pvalue,
    v_ERA5_India_u_hypothesis,
) = ca.dim_linregress(uERA5_ver_India_JJA, vERA5_ver_JJA)

(
    hgt_his_India_u_slope,
    hgt_his_India_u_intercept,
    hgt_his_India_u_rvalue,
    hgt_his_India_u_pvalue,
    hgt_his_India_u_hypothesis,
) = ca.dim_linregress(uhis_ver_India_JJA, hgthis_ver_JJA)
(
    u_his_India_u_slope,
    u_his_India_u_intercept,
    u_his_India_u_rvalue,
    u_his_India_u_pvalue,
    u_his_India_u_hypothesis,
) = ca.dim_linregress(uhis_ver_India_JJA, uhis_ver_JJA)
(
    v_his_India_u_slope,
    v_his_India_u_intercept,
    v_his_India_u_rvalue,
    v_his_India_u_pvalue,
    v_his_India_u_hypothesis,
) = ca.dim_linregress(uhis_ver_India_JJA, vhis_ver_JJA)
# %%
#   calculate wind check
wind_ERA5_India_u_mask = ca.wind_check(
    xr.where(u_ERA5_India_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_India_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_ERA5_India_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_India_u_pvalue <= 0.05, 1.0, 0.0),
)
wind_his_India_u_mask = ca.wind_check(
    xr.where(u_his_India_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_India_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_India_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_India_u_pvalue <= 0.05, 1.0, 0.0),
)
# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=2, nrows=3, proj=proj)

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
    rect = Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1
    )
    ax.add_patch(rect)
    # region 1
    x0 = 50
    y0 = 5.0
    width = 30
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0, 0].contourf(
    hgt_ERA5_India_u_rvalue.sel(level=200.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_India_u_pvalue.sel(level=200.0),
    axs[0, 0],
    n,
    np.where(hgt_ERA5_India_u_pvalue.sel(level=200.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 0].quiver(
    u_ERA5_India_u_rvalue.sel(level=200.0)[::ski, ::ski],
    v_ERA5_India_u_rvalue.sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 0].quiver(
    u_ERA5_India_u_rvalue.where(wind_ERA5_India_u_mask > 0.0).sel(level=200.0)[
        ::ski, ::ski
    ],
    v_ERA5_India_u_rvalue.where(wind_ERA5_India_u_mask > 0.0).sel(level=200.0)[
        ::ski, ::ski
    ],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 0].format(ltitle="India u index", rtitle="ERA5 200hPa")
# ===========================================

con = axs[1, 0].contourf(
    hgt_ERA5_India_u_rvalue.sel(level=500.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_India_u_pvalue.sel(level=500.0),
    axs[1, 0],
    n,
    np.where(hgt_ERA5_India_u_pvalue.sel(level=500.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1, 0].quiver(
    u_ERA5_India_u_rvalue.sel(level=500.0)[::ski, ::ski],
    v_ERA5_India_u_rvalue.sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1, 0].quiver(
    u_ERA5_India_u_rvalue.where(wind_ERA5_India_u_mask > 0.0).sel(level=500.0)[
        ::ski, ::ski
    ],
    v_ERA5_India_u_rvalue.where(wind_ERA5_India_u_mask > 0.0).sel(level=500.0)[
        ::ski, ::ski
    ],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1, 0].format(ltitle="India u index", rtitle="ERA5 500hPa")
# ===================================================
con = axs[2, 0].contourf(
    hgt_ERA5_India_u_rvalue.sel(level=850.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_India_u_pvalue.sel(level=850.0),
    axs[2, 0],
    n,
    np.where(hgt_ERA5_India_u_pvalue.sel(level=850.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[2, 0].quiver(
    u_ERA5_India_u_rvalue.sel(level=850.0)[::ski, ::ski],
    v_ERA5_India_u_rvalue.sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[2, 0].quiver(
    u_ERA5_India_u_rvalue.where(wind_ERA5_India_u_mask > 0.0).sel(level=850.0)[
        ::ski, ::ski
    ],
    v_ERA5_India_u_rvalue.where(wind_ERA5_India_u_mask > 0.0).sel(level=850.0)[
        ::ski, ::ski
    ],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[2, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[2, 0].format(ltitle="India u index", rtitle="ERA5 850hPa")
# ===================================================
#   plot the historical run result
con = axs[0, 1].contourf(
    hgt_his_India_u_rvalue.sel(level=200.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_India_u_pvalue.sel(level=200.0),
    axs[0, 1],
    n,
    np.where(hgt_his_India_u_pvalue.sel(level=200.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 1].quiver(
    u_his_India_u_rvalue.sel(level=200.0)[::ski, ::ski],
    v_his_India_u_rvalue.sel(level=200.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 1].quiver(
    u_his_India_u_rvalue.where(wind_his_India_u_mask > 0.0).sel(level=200.0)[
        ::ski, ::ski
    ],
    v_his_India_u_rvalue.where(wind_his_India_u_mask > 0.0).sel(level=200.0)[
        ::ski, ::ski
    ],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 1].format(ltitle="India u index", rtitle="historical 200hPa")
# ===========================================

con = axs[1, 1].contourf(
    hgt_his_India_u_rvalue.sel(level=500.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_India_u_pvalue.sel(level=500.0),
    axs[1, 1],
    n,
    np.where(hgt_his_India_u_pvalue.sel(level=500.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1, 1].quiver(
    u_his_India_u_rvalue.sel(level=500.0)[::ski, ::ski],
    v_his_India_u_rvalue.sel(level=500.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1, 1].quiver(
    u_his_India_u_rvalue.where(wind_his_India_u_mask > 0.0).sel(level=500.0)[
        ::ski, ::ski
    ],
    v_his_India_u_rvalue.where(wind_his_India_u_mask > 0.0).sel(level=500.0)[
        ::ski, ::ski
    ],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[1, 1].format(ltitle="India u index", rtitle="historical 500hPa")
# ===================================================
con = axs[2, 1].contourf(
    hgt_his_India_u_rvalue.sel(level=850.0),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_India_u_pvalue.sel(level=850.0),
    axs[2, 1],
    n,
    np.where(hgt_his_India_u_pvalue.sel(level=850.0)[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[2, 1].quiver(
    u_his_India_u_rvalue.sel(level=850.0)[::ski, ::ski],
    v_his_India_u_rvalue.sel(level=850.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[2, 1].quiver(
    u_his_India_u_rvalue.where(wind_his_India_u_mask > 0.0).sel(level=850.0)[
        ::ski, ::ski
    ],
    v_his_India_u_rvalue.where(wind_his_India_u_mask > 0.0).sel(level=850.0)[
        ::ski, ::ski
    ],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[2, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[2, 1].format(ltitle="India u index", rtitle="historical 850hPa")
# ===================================================


fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
cli_qERA5 = qERA5_ver_JJA.sel(level=500.0).mean(dim="time", skipna=True)
cli_qhis = qhis_ver_JJA.sel(level=500.0).mean(dim="time", skipna=True)
print(cli_qERA5)
# %%
for ti in hgthis_ver_JJA.coords["time"]:
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(
        span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
    )
    axs = fig.subplots(ncols=1, nrows=1, proj=proj)

    #   set the geo_ticks and map projection to the plots
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

    # ===================================================
    llim_500 = 5600
    hlim_500 = 5920
    spacing_500 = 40
    # ===================================================
    con = axs[0, 0].contourf(
        hgthis_ver_JJA.sel(time=ti, level=500.0)
        - hgthis_ver_JJA.sel(level=500.0).mean(dim="time", skipna=True),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        zorder=0.8,
        extend="both",
        levels=np.arange(-10, 11, 1),
    )
    m = axs[0, 0].quiver(
        (
            uhis_ver_JJA.sel(time=ti, level=500.0)
            - uhis_ver_JJA.sel(level=500.0).mean(dim="time", skipna=True)
        )[::ski, ::ski],
        (
            vhis_ver_JJA.sel(time=ti, level=500.0)
            - vhis_ver_JJA.sel(level=500.0).mean(dim="time", skipna=True)
        )[::ski, ::ski],
        zorder=1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=0.3,
        pivot="mid",
        color="black",
    )
    axs[0, 0].colorbar(con, loc="r")
    axs[0, 0].format(ltitle="year = {}".format(ti.dt.year))

# %%
print(hgt_ERA5_India_uq_rvalue.sel(level=200.0).loc[13.0:15.0, :])

# %%
#   calculate the precipitation regress into IWF index
#   ERA5
preCRU_JJA.coords["time"] = ERA5_SAM_index.coords["time"]
preGPCP_JJA.coords["time"] = ERA5_SAM_index.coords["time"].sel(
    time=ERA5_SAM_index.time.dt.year >= 1979
)
prehis_JJA.coords["time"] = his_SAM_index.coords["time"]
(
    pre_CRU_SAM_slope,
    pre_CRU_SAM_intercept,
    pre_CRU_SAM_rvalue,
    pre_CRU_SAM_pvalue,
    pre_CRU_SAM_hypothesis,
) = ca.dim_linregress(ERA5_SAM_index, preCRU_JJA)

(
    pre_GPCP_SAM_slope,
    pre_GPCP_SAM_intercept,
    pre_GPCP_SAM_rvalue,
    pre_GPCP_SAM_pvalue,
    pre_GPCP_SAM_hypothesis,
) = ca.dim_linregress(
    ERA5_SAM_index.sel(time=ERA5_SAM_index.time.dt.year >= 1979), preGPCP_JJA
)

(
    uq_dpg_ERA5_SAM_slope,
    uq_dpg_ERA5_SAM_intercept,
    uq_dpg_ERA5_SAM_rvalue,
    uq_dpg_ERA5_SAM_pvalue,
    uq_dpg_ERA5_SAM_hypothesis,
) = ca.dim_linregress(ERA5_SAM_index, uq_dpg_ERA5_JJA)

(
    vq_dpg_ERA5_SAM_slope,
    vq_dpg_ERA5_SAM_intercept,
    vq_dpg_ERA5_SAM_rvalue,
    vq_dpg_ERA5_SAM_pvalue,
    vq_dpg_ERA5_SAM_hypothesis,
) = ca.dim_linregress(ERA5_SAM_index, vq_dpg_ERA5_JJA)

#   historical run
(
    pre_his_SAM_slope,
    pre_his_SAM_intercept,
    pre_his_SAM_rvalue,
    pre_his_SAM_pvalue,
    pre_his_SAM_hypothesis,
) = ca.dim_linregress(his_SAM_index, prehis_JJA)

(
    uq_dpg_his_SAM_slope,
    uq_dpg_his_SAM_intercept,
    uq_dpg_his_SAM_rvalue,
    uq_dpg_his_SAM_pvalue,
    uq_dpg_his_SAM_hypothesis,
) = ca.dim_linregress(his_SAM_index, uq_dpg_his_JJA)

(
    vq_dpg_his_SAM_slope,
    vq_dpg_his_SAM_intercept,
    vq_dpg_his_SAM_rvalue,
    vq_dpg_his_SAM_pvalue,
    vq_dpg_his_SAM_hypothesis,
) = ca.dim_linregress(his_SAM_index, vq_dpg_his_JJA)
# %%
#   check the uq and vq
uqvq_ERA5_SAM_mask = ca.wind_check(
    xr.where(uq_dpg_ERA5_SAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(vq_dpg_ERA5_SAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(uq_dpg_ERA5_SAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(vq_dpg_ERA5_SAM_pvalue <= 0.05, 1.0, 0.0),
)

uqvq_his_SAM_mask = ca.wind_check(
    xr.where(uq_dpg_his_SAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(vq_dpg_his_SAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(uq_dpg_his_SAM_pvalue <= 0.05, 1.0, 0.0),
    xr.where(vq_dpg_his_SAM_pvalue <= 0.05, 1.0, 0.0),
)
# %%
#   plot the precipitation and uqvq
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=3, nrows=1, proj=proj)

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
    rect = Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1
    )
    ax.add_patch(rect)
    # region 1
    x0 = 90
    y0 = 5
    width = 50
    height = 27.5
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0, 0].contourf(
    pre_CRU_SAM_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    pre_CRU_SAM_pvalue,
    axs[0, 0],
    n,
    np.where(pre_CRU_SAM_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 0].quiver(
    uq_dpg_ERA5_SAM_rvalue[::ski, ::ski],
    vq_dpg_ERA5_SAM_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 0].quiver(
    uq_dpg_ERA5_SAM_rvalue.where(uqvq_ERA5_SAM_mask > 0.0)[::ski, ::ski],
    vq_dpg_ERA5_SAM_rvalue.where(uqvq_ERA5_SAM_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 0].format(ltitle="CRU & ERA5", rtitle="precip&Uq reg SAM")
# ===================================================
con = axs[0, 1].contourf(
    pre_GPCP_SAM_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    pre_GPCP_SAM_pvalue,
    axs[0, 1],
    n,
    np.where(pre_GPCP_SAM_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 1].quiver(
    uq_dpg_ERA5_SAM_rvalue[::ski, ::ski],
    vq_dpg_ERA5_SAM_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 1].quiver(
    uq_dpg_ERA5_SAM_rvalue.where(uqvq_ERA5_SAM_mask > 0.0)[::ski, ::ski],
    vq_dpg_ERA5_SAM_rvalue.where(uqvq_ERA5_SAM_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 1].format(ltitle="GPCP & ERA5", rtitle="precip&Uq reg SAM")
# ===================================================
con = axs[0, 2].contourf(
    pre_his_SAM_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    pre_his_SAM_pvalue,
    axs[0, 2],
    n,
    np.where(pre_his_SAM_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0, 2].quiver(
    uq_dpg_his_SAM_rvalue[::ski, ::ski],
    vq_dpg_his_SAM_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0, 2].quiver(
    uq_dpg_his_SAM_rvalue.where(uqvq_his_SAM_mask > 0.0)[::ski, ::ski],
    vq_dpg_his_SAM_rvalue.where(uqvq_his_SAM_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0, 2].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=0.5,
    label="0.5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)
axs[0, 2].format(ltitle="historical", rtitle="precip&Uq reg SAM")
# ===================================================
fig.colorbar(con, loc="b", width=0.13, length=0.5, label="")
fig.format(abc="(a)", abcloc="l")
# %%
