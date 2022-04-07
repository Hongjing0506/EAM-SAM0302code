"""
Author: ChenHJ
Date: 2022-03-31 10:40:07
LastEditors: ChenHJ
LastEditTime: 2022-03-31 10:40:07
FilePath: /chenhj/0302code/ssp585_mon_index.py
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
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models//historical/zg/zg_Amon_ensemble_historical_gn_195001-201412.nc"
)
hgthis = fhgthis["zg"]
hgthis.coords["plev"] = hgthis.coords["plev"] / 100.0
hgthis = hgthis.rename({"plev": "level"})

fuhis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models//historical/ua/ua_Amon_ensemble_historical_gn_195001-201412.nc"
)
uhis = fuhis["ua"]
uhis.coords["plev"] = uhis.coords["plev"] / 100.0
uhis = uhis.rename({"plev": "level"})

fvhis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models//historical/va/va_Amon_ensemble_historical_gn_195001-201412.nc"
)
vhis = fvhis["va"]
vhis.coords["plev"] = vhis.coords["plev"] / 100.0
vhis = vhis.rename({"plev": "level"})

fsphis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models//historical/ps/ps_Amon_ensemble_historical_gn_195001-201412.nc"
)
sphis = fsphis["ps"]

fqhis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/pick_models//historical/hus/hus_Amon_ensemble_historical_gn_195001-201412.nc"
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
prehis = fprehis["pr"]*3600*24


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
preCRU_JJA = ca.p_time(preCRU, 6, 8, True)/30.67
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
pressp585 = fpressp585["pr"]*3600*24
# %%
#   pick up the ssp585 data
hgtssp585_ver_JJA = ca.p_time(hgtssp585, 6, 8, True).loc[:, :100, :, :]

ussp585_ver_JJA = ca.p_time(ussp585, 6, 8, True).loc[:, :100, :, :]
vssp585_ver_JJA = ca.p_time(vssp585, 6, 8, True).loc[:, :100, :, :]
qssp585_ver_JJA = ca.p_time(qssp585, 6, 8, True).loc[:, :100, :, :]
pressp585_JJA = ca.p_time(pressp585, 6, 8, True)
spssp585_JJA = ca.p_time(spssp585, 6, 8, True)

hgtssp585_ver_JJA = ca.detrend_dim(hgtssp585_ver_JJA, "time", deg=1, demean=False)
ussp585_ver_JJA = ca.detrend_dim(ussp585_ver_JJA, "time", deg=1, demean=False)
vssp585_ver_JJA = ca.detrend_dim(vssp585_ver_JJA, "time", deg=1, demean=False)
qssp585_ver_JJA = ca.detrend_dim(qssp585_ver_JJA, "time", deg=1, demean=False)
pressp585_JJA = ca.detrend_dim(pressp585_JJA, "time", deg=1, demean=False)
spssp585_JJA = ca.detrend_dim(spssp585_JJA, "time", deg=1, demean=False)

# %%
#   calculate the SAM and IWF index in ERA5, historical, ssp585
ERA5_IWF_index = ca.IWF(uERA5_ver_JJA, vERA5_ver_JJA)
ERA5_IWF_index = ca.detrend_dim(ERA5_IWF_index, "time", deg=1, demean=False)
ERA5_SAM_index = ca.SAM(vERA5_ver_JJA)
ERA5_SAM_index = ca.detrend_dim(ERA5_SAM_index, "time", deg=1, demean=False)

his_IWF_index = ca.IWF(uhis_ver_JJA, vhis_ver_JJA)
his_IWF_index = ca.detrend_dim(his_IWF_index, "time", deg=1, demean=False)
his_SAM_index = ca.SAM(vhis_ver_JJA)
his_SAM_index = ca.detrend_dim(his_SAM_index, "time", deg=1, demean=False)

ssp585_IWF_index = ca.IWF(ussp585_ver_JJA, vssp585_ver_JJA)
ssp585_IWF_index = ca.detrend_dim(ssp585_IWF_index, "time", deg=1, demean=False)
ssp585_SAM_index = ca.SAM(vssp585_ver_JJA)
ssp585_SAM_index = ca.detrend_dim(ssp585_SAM_index, "time", deg=1, demean=False)
# %%
print(stats.linregress(ssp585_IWF_index, ssp585_SAM_index))
# %%
#   calculate the rolling correlation of SAM and IWF index in ssp585
freq = "AS-JUL"
window = 31

ssp585_IWF_SAM_rolling_9 = ca.rolling_reg_index(
    ssp585_IWF_index, ssp585_SAM_index, ssp585_IWF_index.time, window, freq, True
)
# %%
#   plot the rolling correlation coefficients of SAM and IWF index
fig = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
axs = fig.subplots(ncols=1, nrows=1)
lw = 0.8
# cycle = pplt.Cycle('Pastel1', 'Pastel2', 27, left=0.1)


m1 = axs[0].line(
    ssp585_IWF_index.time.dt.year,
    ca.standardize(ssp585_IWF_index),
    lw=lw,
    color="black",
)
m2 = axs[0].line(
    ssp585_SAM_index.time.dt.year,
    ca.standardize(ssp585_SAM_index),
    lw=lw,
    color="blue",
)
m3 = axs[0].line(
    ssp585_IWF_index.time.dt.year,
    np.array(ssp585_IWF_SAM_rolling_9["rvalue"]),
    lw=lw,
    color="red",
    linestyle="--",
)


axs[0].axhline(0, lw=0.8, color="grey5", linestyle="--")
axs[0].axhline(0.2133, lw=0.8, color="grey5", linestyle="--")
axs[0].axhline(-0.2133, lw=0.8, color="grey5", linestyle="--")
axs[0].format(
    ltitle="window={}".format(window),
    rtitle="2015-2099",
    title="IWF & SAM",
    xrotation=0,
    ymin=-3.0,
    ymax=3.0,
    ylocator=0.5,
    yminorlocator=0.25,
)
axs[0].legend(handles=[m1, m2, m3], loc="ll", labels=["IWF", "SAM", "r"], ncols=1)
# %%
#   calculate the whole levels water vapor flux in ERA5 historical and ssp585
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

hislevel = qhis_ver_JJA.coords["level"] * 100.0
hislevel.attrs["units"] = "Pa"
hisdp = geocat.comp.dpres_plevel(hislevel, sphis_JJA, ptop)
hisdpg = hisdp / g
hisdpg.attrs["units"] = "kg/m2"
uqhis_ver_JJA = uhis_ver_JJA * qhis_ver_JJA.data * 1000.0
vqhis_ver_JJA = vhis_ver_JJA * qhis_ver_JJA.data * 1000.0
uqhis_ver_JJA.attrs["units"] = "[m/s][g/kg]"
vqhis_ver_JJA.attrs["units"] = "[m/s][g/kg]"
uq_dpg_his_JJA = (uqhis_ver_JJA * hisdpg.data).sum(dim="level", skipna=True) / 1e05
vq_dpg_his_JJA = (vqhis_ver_JJA * hisdpg.data).sum(dim="level", skipna=True) / 1e05
uq_dpg_his_JJA = ca.detrend_dim(uq_dpg_his_JJA, "time", deg=1, demean=False)
vq_dpg_his_JJA = ca.detrend_dim(vq_dpg_his_JJA, "time", deg=1, demean=False)
uq_dpg_his_JJA.attrs["units"] = "100kg/(m*s)"
vq_dpg_his_JJA.attrs["units"] = "100kg/(m*s)"

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
uq_dpg_ssp585_JJA = ca.detrend_dim(uq_dpg_ssp585_JJA, "time", deg=1, demean=False)
vq_dpg_ssp585_JJA = ca.detrend_dim(vq_dpg_ssp585_JJA, "time", deg=1, demean=False)
uq_dpg_ssp585_JJA.attrs["units"] = "100kg/(m*s)"
vq_dpg_ssp585_JJA.attrs["units"] = "100kg/(m*s)"
# %%
#   calculate the India and NCR uq\vq area mean
uq_dpg_ERA5_India_JJA = ca.cal_lat_weighted_mean(
    uq_dpg_ERA5_JJA.loc[:, 5:25, 50:80]
).mean(dim="lon", skipna=True)
uq_dpg_ERA5_India_JJA = ca.detrend_dim(
    uq_dpg_ERA5_India_JJA, "time", deg=1, demean=False
)
uq_dpg_his_India_JJA = ca.cal_lat_weighted_mean(
    uq_dpg_his_JJA.loc[:, 5:25, 50:80]
).mean(dim="lon", skipna=True)
uq_dpg_his_India_JJA = ca.detrend_dim(uq_dpg_his_India_JJA, "time", deg=1, demean=False)

uq_dpg_ssp585_India_JJA = ca.cal_lat_weighted_mean(
    uq_dpg_ssp585_JJA.loc[:, 5:25, 50:80]
).mean(dim="lon", skipna=True)
uq_dpg_ssp585_India_JJA = ca.detrend_dim(
    uq_dpg_ssp585_India_JJA, "time", deg=1, demean=False
)

vq_dpg_ERA5_NCR_JJA = ca.cal_lat_weighted_mean(
    vq_dpg_ERA5_JJA.loc[:, 25.0:37.5, 110.0:125.0]
).mean(dim="lon", skipna=True)
vq_dpg_ERA5_NCR_JJA = ca.detrend_dim(vq_dpg_ERA5_NCR_JJA, "time", deg=1, demean=False)
vq_dpg_his_NCR_JJA = ca.cal_lat_weighted_mean(
    vq_dpg_his_JJA.loc[:, 25.0:37.5, 110.0:125.0]
).mean(dim="lon", skipna=True)
vq_dpg_his_NCR_JJA = ca.detrend_dim(vq_dpg_his_NCR_JJA, "time", deg=1, demean=False)

vq_dpg_ssp585_NCR_JJA = ca.cal_lat_weighted_mean(
    vq_dpg_ssp585_JJA.loc[:, 25.0:37.5, 110.0:125.0]
).mean(dim="lon", skipna=True)
vq_dpg_ssp585_NCR_JJA = ca.detrend_dim(
    vq_dpg_ssp585_NCR_JJA, "time", deg=1, demean=False
)
# %%
#   calculate the uq/vq correlation coefficients and rolling correlation coefficients
ERA5_uqIND_vqNCR_regress = stats.linregress(uq_dpg_ERA5_India_JJA, vq_dpg_ERA5_NCR_JJA)
his_uqIND_vqNCR_regress = stats.linregress(uq_dpg_his_India_JJA, vq_dpg_his_NCR_JJA)
ssp585_uqIND_vqNCR_regress = stats.linregress(
    uq_dpg_ssp585_India_JJA, vq_dpg_ssp585_NCR_JJA
)

freq = "AS-JUL"
window = 31
ERA5_uqIND_vqNCR_rolling_regress = ca.rolling_reg_index(
    uq_dpg_ERA5_India_JJA,
    vq_dpg_ERA5_NCR_JJA,
    uq_dpg_ERA5_India_JJA.time,
    window,
    freq,
    True,
)
his_uqIND_vqNCR_rolling_regress = ca.rolling_reg_index(
    uq_dpg_his_India_JJA,
    vq_dpg_his_NCR_JJA,
    uq_dpg_his_India_JJA.time,
    window,
    freq,
    True,
)
ssp585_uqIND_vqNCR_rolling_regress = ca.rolling_reg_index(
    uq_dpg_ssp585_India_JJA,
    vq_dpg_ssp585_NCR_JJA,
    uq_dpg_ssp585_India_JJA.time,
    window,
    freq,
    True,
)

# %%
#   plot the Ind uq/NCR vq and their rolling correlation coefficients
fig = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
axs = fig.subplots(ncols=1, nrows=3)
lw = 0.8
# cycle = pplt.Cycle('Pastel1', 'Pastel2', 27, left=0.1)

#=======================================================
m1 = axs[0].line(
    uq_dpg_ERA5_India_JJA.time.dt.year,
    ca.standardize(uq_dpg_ERA5_India_JJA),
    lw=lw,
    color="black",
)
m2 = axs[0].line(
    vq_dpg_ERA5_NCR_JJA.time.dt.year,
    ca.standardize(vq_dpg_ERA5_NCR_JJA),
    lw=lw,
    color="blue",
)
m3 = axs[0].line(
    vq_dpg_ERA5_NCR_JJA.time.dt.year,
    ERA5_uqIND_vqNCR_rolling_regress["rvalue"].data,
    lw=lw,
    color="red",
    linestyle="--",
)

rlim = ca.cal_rlim1(0.95, len(uq_dpg_ERA5_India_JJA.time.dt.year))

axs[0].axhline(0, lw=0.8, color="grey5", linestyle="--")
axs[0].axhline(rlim, lw=0.8, color="grey5", linestyle="--")
axs[0].axhline(-rlim, lw=0.8, color="grey5", linestyle="--")
axs[0].text(1952, 2.5, "r={:.2f}".format(ERA5_uqIND_vqNCR_regress[2]), size=10)
axs[0].legend(handles=[m1, m2, m3], loc="ll", labels=["uq Ind", "vq NCR", "r"], ncols=1)
axs[0].format(
    ltitle="window={}".format(window),
    rtitle="ERA5",
    title="uq Ind & vq NCR",
    xrotation=0,
    ymin=-3.0,
    ymax=3.0,
    ylocator=0.5,
    yminorlocator=0.25,
    ylabel=""
)
#=======================================================
m1 = axs[1].line(
    uq_dpg_his_India_JJA.time.dt.year,
    ca.standardize(uq_dpg_his_India_JJA),
    lw=lw,
    color="black",
)
m2 = axs[1].line(
    vq_dpg_his_NCR_JJA.time.dt.year,
    ca.standardize(vq_dpg_his_NCR_JJA),
    lw=lw,
    color="blue",
)
m3 = axs[1].line(
    vq_dpg_his_NCR_JJA.time.dt.year,
    his_uqIND_vqNCR_rolling_regress["rvalue"].data,
    lw=lw,
    color="red",
    linestyle="--",
)

rlim = ca.cal_rlim1(0.95, len(uq_dpg_his_India_JJA.time.dt.year))

axs[1].axhline(0, lw=0.8, color="grey5", linestyle="--")
axs[1].axhline(rlim, lw=0.8, color="grey5", linestyle="--")
axs[1].axhline(-rlim, lw=0.8, color="grey5", linestyle="--")
axs[1].text(1952, 2.5, "r={:.2f}".format(his_uqIND_vqNCR_regress[2]), size=10)
axs[1].legend(handles=[m1, m2, m3], loc="ll", labels=["uq Ind", "vq NCR", "r"], ncols=1)
axs[1].format(
    ltitle="window={}".format(window),
    rtitle="historical",
    title="uq Ind & vq NCR",
    xrotation=0,
    ymin=-3.0,
    ymax=3.0,
    ylocator=0.5,
    yminorlocator=0.25,
    ylabel=""
)
#=======================================================
m1 = axs[2].line(
    uq_dpg_ssp585_India_JJA.time.dt.year,
    ca.standardize(uq_dpg_ssp585_India_JJA),
    lw=lw,
    color="black",
)
m2 = axs[2].line(
    vq_dpg_ssp585_NCR_JJA.time.dt.year,
    ca.standardize(vq_dpg_ssp585_NCR_JJA),
    lw=lw,
    color="blue",
)
m3 = axs[2].line(
    vq_dpg_ssp585_NCR_JJA.time.dt.year,
    ssp585_uqIND_vqNCR_rolling_regress["rvalue"].data,
    lw=lw,
    color="red",
    linestyle="--",
)

rlim = ca.cal_rlim1(0.95, len(uq_dpg_ssp585_India_JJA.time.dt.year))

axs[2].axhline(0, lw=0.8, color="grey5", linestyle="--")
axs[2].axhline(rlim, lw=0.8, color="grey5", linestyle="--")
axs[2].axhline(-rlim, lw=0.8, color="grey5", linestyle="--")
axs[2].text(2017, 2.5, "r={:.2f}".format(ssp585_uqIND_vqNCR_regress[2]), size=10)
axs[2].legend(handles=[m1, m2, m3], loc="ll", labels=["uq Ind", "vq NCR", "r"], ncols=1)
axs[2].format(
    ltitle="window={}".format(window),
    rtitle="ssp585",
    title="uq Ind & vq NCR",
    xrotation=0,
    ymin=-3.0,
    ymax=3.0,
    ylocator=0.5,
    yminorlocator=0.25,
    ylabel=""
)

# %%
#   plot the SAM index and IWF index correlation coefficients and rolling correlation coefficients


# %%
#   calculate the precipitation area mean of India and NCR
preCRU_India_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.loc[:, 8:28, 70:86]).mean(dim="lon", skipna=True)
preCRU_NCR_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.loc[:, 36:42, 108:118]).mean(dim="lon", skipna=True)

preGPCP_India_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.loc[:, 8:28, 70:86]).mean(dim="lon", skipna=True)
preGPCP_NCR_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.loc[:, 36:42, 108:118]).mean(dim="lon", skipna=True)

prehis_India_JJA = ca.cal_lat_weighted_mean(prehis_JJA.loc[:, 8:28, 70:86]).mean(dim="lon", skipna=True)
prehis_NCR_JJA = ca.cal_lat_weighted_mean(prehis_JJA.loc[:, 36:42, 108:118]).mean(dim="lon", skipna=True)

pressp585_India_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.loc[:, 8:28, 70:86]).mean(dim="lon", skipna=True)
pressp585_NCR_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.loc[:, 36:42, 108:118]).mean(dim="lon", skipna=True)
# %%
#   calculate the precipitation regress onto preInd and preNCR
(
    pre_CRU_India_pre_slope,
    pre_CRU_India_pre_intercept,
    pre_CRU_India_pre_rvalue,
    pre_CRU_India_pre_pvalue,
    pre_CRU_India_pre_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA, preCRU_JJA)

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
) = ca.dim_linregress(prehis_India_JJA, prehis_JJA)

(
    pre_ssp585_India_pre_slope,
    pre_ssp585_India_pre_intercept,
    pre_ssp585_India_pre_rvalue,
    pre_ssp585_India_pre_pvalue,
    pre_ssp585_India_pre_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA, pressp585_JJA)

(
    pre_CRU_NCR_pre_slope,
    pre_CRU_NCR_pre_intercept,
    pre_CRU_NCR_pre_rvalue,
    pre_CRU_NCR_pre_pvalue,
    pre_CRU_NCR_pre_hypothesis,
) = ca.dim_linregress(preCRU_NCR_JJA, preCRU_JJA)

(
    pre_GPCP_NCR_pre_slope,
    pre_GPCP_NCR_pre_intercept,
    pre_GPCP_NCR_pre_rvalue,
    pre_GPCP_NCR_pre_pvalue,
    pre_GPCP_NCR_pre_hypothesis,
) = ca.dim_linregress(preGPCP_NCR_JJA, preGPCP_JJA)

(
    pre_his_NCR_pre_slope,
    pre_his_NCR_pre_intercept,
    pre_his_NCR_pre_rvalue,
    pre_his_NCR_pre_pvalue,
    pre_his_NCR_pre_hypothesis,
) = ca.dim_linregress(prehis_NCR_JJA, prehis_JJA)

(
    pre_ssp585_NCR_pre_slope,
    pre_ssp585_NCR_pre_intercept,
    pre_ssp585_NCR_pre_rvalue,
    pre_ssp585_NCR_pre_pvalue,
    pre_ssp585_NCR_pre_hypothesis,
) = ca.dim_linregress(pressp585_NCR_JJA, pressp585_JJA)
# %%
