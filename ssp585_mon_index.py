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
fhgt585 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/zg/zg_Amon_ensemble_ssp585_gn_201501-209912.nc"
)
hgt585 = fhgt585["zg"]
hgt585.coords["plev"] = hgt585.coords["plev"] / 100.0
hgt585 = hgt585.rename({"plev": "level"})

fu585 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/ua/ua_Amon_ensemble_ssp585_gn_201501-209912.nc"
)
u585 = fu585["ua"]
u585.coords["plev"] = u585.coords["plev"] / 100.0
u585 = u585.rename({"plev": "level"})

fv585 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/va/va_Amon_ensemble_ssp585_gn_201501-209912.nc"
)
v585 = fv585["va"]
v585.coords["plev"] = v585.coords["plev"] / 100.0
v585 = v585.rename({"plev": "level"})

fsp585 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/ps/ps_Amon_ensemble_ssp585_gn_201501-209912.nc"
)
sp585 = fsp585["ps"]

fq585 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/hus/hus_Amon_ensemble_ssp585_gn_201501-209912.nc"
)
q585 = fq585["hus"]
q585.coords["plev"] = q585.coords["plev"] / 100.0
q585 = q585.rename({"plev": "level"})
# %%
hgt585_ver_JJA = ca.p_time(hgt585, 6, 8, True).loc[:, :100, :, :]

u585_ver_JJA = ca.p_time(u585, 6, 8, True).loc[:, :100, :, :]
v585_ver_JJA = ca.p_time(v585, 6, 8, True).loc[:, :100, :, :]
q585_ver_JJA = ca.p_time(q585, 6, 8, True).loc[:, :100, :, :]
pre585_JJA = ca.p_time(pre585, 6, 8, True)
sp585_JJA = ca.p_time(sp585, 6, 8, True)

hgt585_ver_JJA = ca.detrend_dim(hgt585_ver_JJA, "time", deg=1, demean=False)
u585_ver_JJA = ca.detrend_dim(u585_ver_JJA, "time", deg=1, demean=False)
v585_ver_JJA = ca.detrend_dim(v585_ver_JJA, "time", deg=1, demean=False)
q585_ver_JJA = ca.detrend_dim(q585_ver_JJA, "time", deg=1, demean=False)
pre585_JJA = ca.detrend_dim(pre585_JJA, "time", deg=1, demean=False)
sp585_JJA = ca.detrend_dim(sp585_JJA, "time", deg=1, demean=False)

# %%
