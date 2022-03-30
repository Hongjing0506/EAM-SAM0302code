"""
Author: ChenHJ
Date: 2022-03-29 23:37:08
LastEditors: ChenHJ
LastEditTime: 2022-03-30 13:30:51
FilePath: /chenhj/0302code/check_hgt_corr.py
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
hgt_his_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/zg"
)
g = os.walk(hgt_his_path)
filepath = []
modelname_hgt = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_hgt.append(filename[loc[1] + 1 : loc[2]])
hgtds_his = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
hgthis_ds = xr.DataArray(hgtds_his["zg"])
hgthis_ds.coords["models"] = modelname_hgt
# %%
u_his_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/ua"
)
g = os.walk(u_his_path)
filepath = []
modelname_u = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_u.append(filename[loc[1] + 1 : loc[2]])
uds_his = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
uhis_ds = xr.DataArray(uds_his["ua"])
uhis_ds.coords["models"] = modelname_u
# %%
v_his_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/va"
)
g = os.walk(v_his_path)
filepath = []
modelname_v = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_v.append(filename[loc[1] + 1 : loc[2]])
vds_his = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
vhis_ds = xr.DataArray(vds_his["va"])
vhis_ds.coords["models"] = modelname_v
# %%
sp_his_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/ps"
)
g = os.walk(sp_his_path)
filepath = []
modelname_sp = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_sp.append(filename[loc[1] + 1 : loc[2]])
spds_his = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
sphis_ds = xr.DataArray(spds_his["ps"])
sphis_ds.coords["models"] = modelname_sp
# %%
q_his_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/hus"
)
g = os.walk(q_his_path)
filepath = []
modelname_q = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_q.append(filename[loc[1] + 1 : loc[2]])
qds_his = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
qhis_ds = xr.DataArray(qds_his["hus"])
qhis_ds.coords["models"] = modelname_q
# %%
hgthis_ds.coords["plev"] = hgthis_ds["plev"] / 100.0
hgthis_ds = hgthis_ds.rename({"plev": "level"})
uhis_ds.coords["plev"] = uhis_ds["plev"] / 100.0
uhis_ds = uhis_ds.rename({"plev": "level"})
vhis_ds.coords["plev"] = vhis_ds["plev"] / 100.0
vhis_ds = vhis_ds.rename({"plev": "level"})
qhis_ds.coords["plev"] = qhis_ds["plev"] / 100.0
qhis_ds = qhis_ds.rename({"plev": "level"})
# %%
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
hgtERA5_ver_JJA = ca.p_time(hgtERA5, 6, 8, True).loc[:, 100.0:, :, :]
uERA5_ver_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, :, :]
vERA5_ver_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, :, :]
qERA5_ver_JJA = ca.p_time(qERA5, 6, 9, True).loc[:, 100.0:, :, :]
spERA5_JJA = ca.p_time(spERA5, 6, 8, True)

hgtERA5_ver_JJA = ca.detrend_dim(hgtERA5_ver_JJA, "time", deg=1, demean=False)
uERA5_ver_JJA = ca.detrend_dim(uERA5_ver_JJA, "time", deg=1, demean=False)
vERA5_ver_JJA = ca.detrend_dim(vERA5_ver_JJA, "time", deg=1, demean=False)
qERA5_ver_JJA = ca.detrend_dim(qERA5_ver_JJA, "time", deg=1, demean=False)
spERA5_JJA = ca.detrend_dim(spERA5_JJA, "time", deg=1, demean=False)
# %%
hgthis_ver_JJA = ca.p_time(hgthis, 6, 8, True).loc[:, :100, :, :]

uhis_ver_JJA = ca.p_time(uhis, 6, 8, True).loc[:, :100, :, :]
vhis_ver_JJA = ca.p_time(vhis, 6, 8, True).loc[:, :100, :, :]
qhis_ver_JJA = ca.p_time(qhis, 6, 8, True).loc[:, :100, :, :]
sphis_JJA = ca.p_time(sphis, 6, 8, True)

hgthis_ver_JJA = ca.detrend_dim(hgthis_ver_JJA, "time", deg=1, demean=False)
uhis_ver_JJA = ca.detrend_dim(uhis_ver_JJA, "time", deg=1, demean=False)
vhis_ver_JJA = ca.detrend_dim(vhis_ver_JJA, "time", deg=1, demean=False)
qhis_ver_JJA = ca.detrend_dim(qhis_ver_JJA, "time", deg=1, demean=False)
sphis_JJA = ca.detrend_dim(sphis_JJA, "time", deg=1, demean=False)
# %%
hgthis_ds_ver_JJA = ca.p_time(hgthis_ds, 6, 8, True).loc[:, :, :100, :, :]

uhis_ds_ver_JJA = ca.p_time(uhis_ds, 6, 8, True).loc[:, :, :100, :, :]
vhis_ds_ver_JJA = ca.p_time(vhis_ds, 6, 8, True).loc[:, :, :100, :, :]
qhis_ds_ver_JJA = ca.p_time(qhis_ds, 6, 8, True).loc[:, :, :100, :, :]
sphis_ds_JJA = ca.p_time(sphis_ds, 6, 8, True)

hgthis_ds_ver_JJA = ca.detrend_dim(hgthis_ds_ver_JJA, "time", deg=1, demean=False)
uhis_ds_ver_JJA = ca.detrend_dim(uhis_ds_ver_JJA, "time", deg=1, demean=False)
vhis_ds_ver_JJA = ca.detrend_dim(vhis_ds_ver_JJA, "time", deg=1, demean=False)
qhis_ds_ver_JJA = ca.detrend_dim(qhis_ds_ver_JJA, "time", deg=1, demean=False)
sphis_ds_JJA = ca.detrend_dim(sphis_ds_JJA, "time", deg=1, demean=False)
# %%
#   calculate the whole levels water vapor flux
ptop = 100 * 100
g = 9.8
#  ERA5 data
ERA5level = qERA5_ver_JJA.coords["level"] * 100.0
ERA5level.attrs["units"] = "Pa"
ERA5dp = geocat.comp.dpres_plevel(ERA5level, spERA5_JJA, ptop)
ERA5dpg = ERA5dp / g
ERA5dpg.attrs["units"] = "kg/m2"
uqERA5_ver_JJA = uERA5_ver_JJA * qERA5_ver_JJA.data * 1000.0
# vqERA5_ver_JJA = vERA5_ver_JJA * qERA5_ver_JJA.data * 1000.0
uqERA5_ver_JJA.attrs["units"] = "[m/s][g/kg]"
# vqERA5_ver_JJA.attrs["units"] = "[m/s][g/kg]"
uq_dpg_ERA5_JJA = (uqERA5_ver_JJA * ERA5dpg.data).sum(dim="level", skipna=True)
# vq_dpg_ERA5_JJA = (vqERA5_ver_JJA * ERA5dpg.data).sum(dim="level", skipna=True)
uq_dpg_ERA5_JJA = ca.detrend_dim(uq_dpg_ERA5_JJA, "time", deg=1, demean=False)
# vq_dpg_ERA5_JJA = ca.detrend_dim(vq_dpg_ERA5_JJA, "time", deg=1, demean=False)

hislevel = qhis_ver_JJA.coords["level"] * 100.0
hislevel.attrs["units"] = "Pa"
hisdp = geocat.comp.dpres_plevel(hislevel, sphis_JJA, ptop)
hisdpg = hisdp / g
hisdpg.attrs["units"] = "kg/m2"
uqhis_ver_JJA = uhis_ver_JJA * qhis_ver_JJA.data * 1000.0
# vqhis_ver_JJA = vhis_ver_JJA * qhis_ver_JJA.data * 1000.0
uqhis_ver_JJA.attrs["units"] = "[m/s][g/kg]"
# vqhis_ver_JJA.attrs["units"] = "[m/s][g/kg]"
uq_dpg_his_JJA = (uqhis_ver_JJA * hisdpg.data).sum(dim="level", skipna=True)
# vq_dpg_his_JJA = (vqhis_ver_JJA * hisdpg.data).sum(dim="level", skipna=True)
uq_dpg_his_JJA = ca.detrend_dim(uq_dpg_his_JJA, "time", deg=1, demean=False)
# vq_dpg_his_JJA = ca.detrend_dim(vq_dpg_his_JJA, "time", deg=1, demean=False)
# %%
#   calculate the water vapor flux of multi-models
his_dslevel = qhis_ds_ver_JJA.coords["level"] * 100.0
his_dslevel.attrs["units"] = "Pa"
# his_dsdp = geocat.comp.dpres_plevel(his_dslevel, sphis_ds_JJA, ptop)
# print(sphis_ds_JJA)
his_dsdp = xr.apply_ufunc(
    geocat.comp.dpres_plevel,
    his_dslevel,
    sphis_ds_JJA,
    ptop,
    input_core_dims=[["level"], [], []],
    output_core_dims=[["level"]],
    vectorize=True,
    dask="parallelized",
)
# %%
# for i in np.arange(0,26):
#     print(his_dsdp[i, 0, 0, 0, :])
his_dsdp = his_dsdp.transpose("models", "time", "level", "lat", "lon")
his_dsdpg = his_dsdp / g
his_dsdpg.attrs["units"] = "kg/m2"
uqhis_ds_ver_JJA = uhis_ds_ver_JJA * qhis_ds_ver_JJA.data * 1000.0
# vqhis_ds_ver_JJA = vhis_ds_ver_JJA * qhis_ds_ver_JJA.data * 1000.0
uqhis_ds_ver_JJA.attrs["units"] = "[m/s][g/kg]"
# # vqhis_ds_ver_JJA.attrs["units"] = "[m/s][g/kg]"
uq_dpg_his_ds_JJA = (uqhis_ds_ver_JJA * his_dsdpg.data).sum(dim="level", skipna=True)
# vq_dpg_his_ds_JJA = (vqhis_ds_ver_JJA * his_dsdpg.data).sum(dim="level", skipna=True)
uq_dpg_his_ds_JJA = ca.detrend_dim(uq_dpg_his_ds_JJA, "time", deg=1, demean=False)
# vq_dpg_his_ds_JJA = ca.detrend_dim(vq_dpg_his_ds_JJA, "time", deg=1, demean=False)
# %%
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

uq_dpg_his_ds_India_JJA = ca.cal_lat_weighted_mean(
    uq_dpg_his_ds_JJA.loc[:, 5:25, 50:80]
).mean(dim="lon", skipna=True)
uq_dpg_his_ds_India_JJA = ca.detrend_dim(uq_dpg_his_ds_India_JJA, "time", deg=1, demean=False)
# %%
(
    hgt_ERA5_India_uq_slope,
    hgt_ERA5_India_uq_intercept,
    hgt_ERA5_India_uq_rvalue,
    hgt_ERA5_India_uq_pvalue,
    hgt_ERA5_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_ERA5_India_JJA, hgtERA5_ver_JJA.sel(level=200.0))
(
    u_ERA5_India_uq_slope,
    u_ERA5_India_uq_intercept,
    u_ERA5_India_uq_rvalue,
    u_ERA5_India_uq_pvalue,
    u_ERA5_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_ERA5_India_JJA, uERA5_ver_JJA.sel(level=200.0))
(
    v_ERA5_India_uq_slope,
    v_ERA5_India_uq_intercept,
    v_ERA5_India_uq_rvalue,
    v_ERA5_India_uq_pvalue,
    v_ERA5_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_ERA5_India_JJA, vERA5_ver_JJA.sel(level=200.0))

(
    hgt_his_India_uq_slope,
    hgt_his_India_uq_intercept,
    hgt_his_India_uq_rvalue,
    hgt_his_India_uq_pvalue,
    hgt_his_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_his_India_JJA, hgthis_ver_JJA.sel(level=200.0))
(
    u_his_India_uq_slope,
    u_his_India_uq_intercept,
    u_his_India_uq_rvalue,
    u_his_India_uq_pvalue,
    u_his_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_his_India_JJA, uhis_ver_JJA.sel(level=200.0))
(
    v_his_India_uq_slope,
    v_his_India_uq_intercept,
    v_his_India_uq_rvalue,
    v_his_India_uq_pvalue,
    v_his_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_his_India_JJA, vhis_ver_JJA.sel(level=200.0))

(
    hgt_his_ds_India_uq_slope,
    hgt_his_ds_India_uq_intercept,
    hgt_his_ds_India_uq_rvalue,
    hgt_his_ds_India_uq_pvalue,
    hgt_his_ds_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_his_ds_India_JJA, hgthis_ds_ver_JJA.sel(level=200.0))
(
    u_his_ds_India_uq_slope,
    u_his_ds_India_uq_intercept,
    u_his_ds_India_uq_rvalue,
    u_his_ds_India_uq_pvalue,
    u_his_ds_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_his_ds_India_JJA, uhis_ds_ver_JJA.sel(level=200.0))
(
    v_his_ds_India_uq_slope,
    v_his_ds_India_uq_intercept,
    v_his_ds_India_uq_rvalue,
    v_his_ds_India_uq_pvalue,
    v_his_ds_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_his_ds_India_JJA, vhis_ds_ver_JJA.sel(level=200.0))

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
wind_his_ds_India_uq_mask = ca.wind_check(
    xr.where(u_his_ds_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_ds_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_ds_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_ds_India_uq_pvalue <= 0.05, 1.0, 0.0),
)