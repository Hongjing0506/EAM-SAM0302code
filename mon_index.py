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

# %%
#   pick up the JJA

hgtERA5_ver_JJA = ca.p_time(hgtERA5, 6, 8, True).loc[:, 100.0:, :, :]

uERA5_ver_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, :, :]
vERA5_ver_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, :, :]

hgthis_ver_JJA = ca.p_time(hgthis, 6, 8, True).loc[:, :100, :, :]

uhis_ver_JJA = ca.p_time(uhis, 6, 8, True).loc[:, :100, :, :]
vhis_ver_JJA = ca.p_time(vhis, 6, 8, True).loc[:, :100, :, :]



# %%
#   calculate the monsoon index
ERA5_SAM_index = ca.SAM(vERA5_ver_JJA)
his_SAM_index = ca.SAM(vhis_ver_JJA)

ERA5_EAM_index = ca.EAM(uERA5_ver_JJA)
his_EAM_index = ca.EAM(uhis_ver_JJA)

# %%
#   calculate the regression of two monsoon index
ERA5_regress = stats.linregress(ERA5_SAM_index, ERA5_EAM_index)
his_regress = stats.linregress(his_SAM_index, his_EAM_index)
# %%
#   plot the monsoon index
fig = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
axs = fig.subplots(ncols=1, nrows=2)

lw = 1.0
# ========================================
m1 = axs[0].line(ERA5_EAM_index.time.dt.year, ca.standardize(ERA5_EAM_index), color="blue", lw=lw)
m2 = axs[0].line(ERA5_SAM_index.time.dt.year, ca.standardize(ERA5_SAM_index), color="red", lw=lw)

axs[0].legend(handles=[m1, m2], loc="ll", labels=["EAM_index", "SAM_index"], ncols=1)
axs[0].format(ltitle="ERA5", rtitle="r = {:.2f}".format(ERA5_regress[2]))
#========================================
m1 = axs[1].line(his_EAM_index.time.dt.year, ca.standardize(his_EAM_index), color="blue", lw=lw)
m2 = axs[1].line(his_SAM_index.time.dt.year, ca.standardize(his_SAM_index), color="red", lw=lw)

axs[1].legend(handles=[m1, m2], loc="ll", labels=["EAM_index", "SAM_index"], ncols=1)
axs[1].format(ltitle="historical", rtitle="r = {:.2f}".format(his_regress[2]))
#========================================
axs.format(
    ylim=(-3.0, 3.0),
    ylocator=1.0,
    yminorlocator=0.2,
    ylabel="",
    xlabel=""
)
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
