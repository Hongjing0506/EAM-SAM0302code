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
hgtERA5 = ca.detrend_dim(hgtERA5, "time", deg=1, demean=False)

fuERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/uwind_mon_r144x72_195001-201412.nc"
)
uERA5 = fuERA5["u"]
uERA5 = ca.detrend_dim(uERA5, "time", deg=1, demean=False)

fvERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/vwind_mon_r144x72_195001-201412.nc"
)
vERA5 = fvERA5["v"]
vERA5 = ca.detrend_dim(vERA5, "time", deg=1, demean=False)

fspERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/sp_mon_r144x72_195001-201412.nc"
)
spERA5 = fspERA5["sp"]
spERA5 = ca.detrend_dim(spERA5, "time", deg=1, demean=False)

fqERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/q_mon_r144x72_195001-201412.nc"
)
qERA5 = fqERA5["q"]
qERA5 = ca.detrend_dim(qERA5, "time", deg=1, demean=False)

fhgthis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/zg/zg_Amon_ensemble_historical_gn_195001-201412.nc"
)
hgthis = fhgthis["zg"]
hgthis = ca.detrend_dim(hgthis, "time", deg=1, demean=False)
hgthis.coords["plev"] = hgthis.coords["plev"] / 100.0
hgthis = hgthis.rename({"plev": "level"})

fuhis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/ua/ua_Amon_ensemble_historical_gn_195001-201412.nc"
)
uhis = fuhis["ua"]
uhis = ca.detrend_dim(uhis, "time", deg=1, demean=False)
uhis.coords["plev"] = uhis.coords["plev"] / 100.0
uhis = uhis.rename({"plev": "level"})

fvhis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/va/va_Amon_ensemble_historical_gn_195001-201412.nc"
)
vhis = fvhis["va"]
vhis = ca.detrend_dim(vhis, "time", deg=1, demean=False)
vhis.coords["plev"] = vhis.coords["plev"] / 100.0
vhis = vhis.rename({"plev": "level"})

fsphis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/ps/ps_Amon_ensemble_historical_gn_195001-201412.nc"
)
sphis = fsphis["ps"]
sphis = ca.detrend_dim(sphis, "time", deg=1, demean=False)

fqhis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/hus/hus_Amon_ensemble_historical_gn_195001-201412.nc"
)
qhis = fqhis["hus"]
qhis = ca.detrend_dim(qhis, "time", deg=1, demean=False)
qhis.coords["plev"] = qhis.coords["plev"] / 100.0
qhis = qhis.rename({"plev": "level"})

# %%
#   read the precipitation data
fpreCRU = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/cru_ts4.01_r144x72_195001-201412.nc"
)
preCRU = fpreCRU["pre"]
preCRU = ca.detrend_dim(preCRU, "time", deg=1, demean=False)

fprehis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/pr/pr_Amon_ensemble_historical_gn_195001-201412.nc"
)
prehis = fprehis["pr"]
prehis = ca.detrend_dim(prehis, "time", deg=1, demean=False)

# GPCP data just have 1979-2014 year
fpreGPCP = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/GPCP_r144x72_197901-201412.nc"
)
preGPCP = fpreGPCP["precip"]
preGPCP = ca.detrend_dim(preGPCP, "time", deg=1, demean=False)

# %%
#   pick up the JJA

hgtERA5_ver_JJA = ca.p_time(hgtERA5, 6, 8, True).loc[:, 100.0:, :, :]

uERA5_ver_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, :, :]
vERA5_ver_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, :, :]
qERA5_ver_JJA = ca.p_time(qERA5, 6, 9, True).loc[:, 100.0:, :, :]
preCRU_JJA = ca.p_time(preCRU, 6, 8, True)
spERA5_JJA = ca.p_time(spERA5, 6, 8, True)

hgthis_ver_JJA = ca.p_time(hgthis, 6, 8, True).loc[:, :100, :, :]

uhis_ver_JJA = ca.p_time(uhis, 6, 8, True).loc[:, :100, :, :]
vhis_ver_JJA = ca.p_time(vhis, 6, 8, True).loc[:, :100, :, :]
qhis_ver_JJA = ca.p_time(qhis, 6, 8, True).loc[:, :100, :, :]
prehis_JJA = ca.p_time(prehis, 6, 8, True)
sphis_JJA = ca.p_time(sphis, 6, 8, True)

preGPCP_JJA = ca.p_time(preGPCP, 6, 8, True)
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

# %%
#   calculate the monsoon index
ERA5_SAM_index = ca.SAM(vERA5_ver_JJA)
his_SAM_index = ca.SAM(vhis_ver_JJA)

ERA5_EAM_index = ca.EAM(uERA5_ver_JJA)
his_EAM_index = ca.EAM(uhis_ver_JJA)

ERA5_WY_index = ca.WY(uERA5_ver_JJA)
his_WY_index = ca.WY(uhis_ver_JJA)

ERA5_IWF_index = ca.IWF(uERA5_ver_JJA, vERA5_ver_JJA)
his_IWF_index = ca.IWF(uhis_ver_JJA, vhis_ver_JJA)
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
    ERA5_his_EAM_regress,
    ERA5_his_SAM_regress,
    ERA5_his_WY_regress,
    ERA5_his_IWF_regress,
)
# %%
#   plot the monsoon index
fig = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
axs = fig.subplots(ncols=1, nrows=2)

lw = 1.0
# ========================================
m1 = axs[0].line(
    ERA5_EAM_index.time.dt.year, ca.standardize(ERA5_EAM_index), color="blue", lw=lw
)
m2 = axs[0].line(
    ERA5_SAM_index.time.dt.year, ca.standardize(ERA5_SAM_index), color="red", lw=lw
)

axs[0].legend(handles=[m1, m2], loc="ll", labels=["EAM_index", "SAM_index"], ncols=1)
axs[0].format(ltitle="ERA5", rtitle="r = {:.2f}".format(ERA5_regress[2]))
# ========================================
m1 = axs[1].line(
    his_EAM_index.time.dt.year, ca.standardize(his_EAM_index), color="blue", lw=lw
)
m2 = axs[1].line(
    his_SAM_index.time.dt.year, ca.standardize(his_SAM_index), color="red", lw=lw
)

axs[1].legend(handles=[m1, m2], loc="ll", labels=["EAM_index", "SAM_index"], ncols=1)
axs[1].format(ltitle="historical", rtitle="r = {:.2f}".format(his_regress[2]))
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

ushearhis_JJA = uhis_ver_JJA.sel(level=850.0) - uhis_ver_JJA.sel(level=200.0)
vshearhis_JJA = vhis_ver_JJA.sel(level=850.0) - vhis_ver_JJA.sel(level=200.0)

# %%
#   calculate the vorticity
windERA5_JJA = VectorWind(
    uERA5_ver_JJA.sel(level=850.0), vERA5_ver_JJA.sel(level=850.0)
)
vorERA5_JJA = windERA5_JJA.vorticity()

windhis_JJA = VectorWind(uhis_ver_JJA.sel(level=850.0), vhis_ver_JJA.sel(level=850.0))
vorhis_JJA = windhis_JJA.vorticity()
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
# %%
