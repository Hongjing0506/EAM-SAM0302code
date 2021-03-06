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
from importlib import reload
reload(sepl)

import pandas as pd
import metpy.calc as mpcalc
import metpy.constants as constants
import geocat.comp
from windspharm.xarray import VectorWind


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
from scipy.interpolate import interp2d
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
#   read the precipitation data
fpreCRU = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/cru_ts4.01_r144x72_195001-201412.nc"
)
preCRU = fpreCRU["pre"]

fpreGPCP = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/GPCP_r144x72_197901-201412.nc"
)
preGPCP = fpreGPCP["precip"]

fprehis = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/pr/pr_Amon_ensemble_historical_gn_195001-201412.nc"
)
prehis = fprehis["pr"]*3600*24

pr_his_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/pr"
)
g = os.walk(pr_his_path)
filepath = []
modelname_pr = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_pr.append(filename[loc[1] + 1 : loc[2]])
preds_his = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
prehis_ds = xr.DataArray(preds_his["pr"])*3600*24
prehis_ds.coords["models"] = modelname_pr
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
preCRU_JJA = ca.p_time(preCRU, 6, 8, True)/30.67
preGPCP_JJA = ca.p_time(preGPCP, 6, 8, True)

hgtERA5_ver_JJA = ca.detrend_dim(hgtERA5_ver_JJA, "time", deg=1, demean=False)
uERA5_ver_JJA = ca.detrend_dim(uERA5_ver_JJA, "time", deg=1, demean=False)
vERA5_ver_JJA = ca.detrend_dim(vERA5_ver_JJA, "time", deg=1, demean=False)
qERA5_ver_JJA = ca.detrend_dim(qERA5_ver_JJA, "time", deg=1, demean=False)
spERA5_JJA = ca.detrend_dim(spERA5_JJA, "time", deg=1, demean=False)
preCRU_JJA = ca.detrend_dim(preCRU_JJA, "time", deg=1, demean=False)
preGPCP_JJA = ca.detrend_dim(preGPCP_JJA, "time", deg=1, demean=False)
# %%
hgthis_ver_JJA = ca.p_time(hgthis, 6, 8, True).loc[:, :100, :, :]

uhis_ver_JJA = ca.p_time(uhis, 6, 8, True).loc[:, :100, :, :]
vhis_ver_JJA = ca.p_time(vhis, 6, 8, True).loc[:, :100, :, :]
qhis_ver_JJA = ca.p_time(qhis, 6, 8, True).loc[:, :100, :, :]
sphis_JJA = ca.p_time(sphis, 6, 8, True)
prehis_JJA = ca.p_time(prehis, 6, 8, True)

hgthis_ver_JJA = ca.detrend_dim(hgthis_ver_JJA, "time", deg=1, demean=False)
uhis_ver_JJA = ca.detrend_dim(uhis_ver_JJA, "time", deg=1, demean=False)
vhis_ver_JJA = ca.detrend_dim(vhis_ver_JJA, "time", deg=1, demean=False)
qhis_ver_JJA = ca.detrend_dim(qhis_ver_JJA, "time", deg=1, demean=False)
sphis_JJA = ca.detrend_dim(sphis_JJA, "time", deg=1, demean=False)
prehis_JJA = ca.detrend_dim(prehis_JJA, "time", deg=1, demean=False)
# %%
hgthis_ds_ver_JJA = ca.p_time(hgthis_ds, 6, 8, True).loc[:, :, :100, :, :]

uhis_ds_ver_JJA = ca.p_time(uhis_ds, 6, 8, True).loc[:, :, :100, :, :]
vhis_ds_ver_JJA = ca.p_time(vhis_ds, 6, 8, True).loc[:, :, :100, :, :]
qhis_ds_ver_JJA = ca.p_time(qhis_ds, 6, 8, True).loc[:, :, :100, :, :]
sphis_ds_JJA = ca.p_time(sphis_ds, 6, 8, True)
prehis_ds_JJA = ca.p_time(prehis_ds, 6, 8, True)

hgthis_ds_ver_JJA = ca.detrend_dim(hgthis_ds_ver_JJA, "time", deg=1, demean=False)
uhis_ds_ver_JJA = ca.detrend_dim(uhis_ds_ver_JJA, "time", deg=1, demean=False)
vhis_ds_ver_JJA = ca.detrend_dim(vhis_ds_ver_JJA, "time", deg=1, demean=False)
qhis_ds_ver_JJA = ca.detrend_dim(qhis_ds_ver_JJA, "time", deg=1, demean=False)
sphis_ds_JJA = ca.detrend_dim(sphis_ds_JJA, "time", deg=1, demean=False)
prehis_ds_JJA = ca.detrend_dim(prehis_ds_JJA, "time", deg=1, demean=False)
# %%
#   reorder the shape
# print(hgthis_ds_ver_JJA)
hgthis_ds_ver_JJA_copy = hgthis_ds_ver_JJA.copy()
uhis_ds_ver_JJA_copy = uhis_ds_ver_JJA.copy()
vhis_ds_ver_JJA_copy = vhis_ds_ver_JJA.copy()
qhis_ds_ver_JJA_copy = qhis_ds_ver_JJA.copy()
sphis_ds_JJA_copy = sphis_ds_JJA.copy()
prehis_ds_JJA_copy = prehis_ds_JJA.copy()
models = hgthis_ds_ver_JJA.coords["models"]

print(models)
for i, mod in enumerate(models):
    hgthis_ds_ver_JJA_copy[i, :, :, :, :] = np.array(hgthis_ds_ver_JJA.sel(models=mod))
    uhis_ds_ver_JJA_copy[i, :, :, :, :] = np.array(uhis_ds_ver_JJA.sel(models=mod))
    vhis_ds_ver_JJA_copy[i, :, :, :, :] = np.array(vhis_ds_ver_JJA.sel(models=mod))
    qhis_ds_ver_JJA_copy[i, :, :, :, :] = np.array(qhis_ds_ver_JJA.sel(models=mod))
    sphis_ds_JJA_copy[i, :, :, :] = np.array(sphis_ds_JJA.sel(models=mod))
    prehis_ds_JJA_copy[i, :, :, :] = np.array(prehis_ds_JJA.sel(models=mod))
hgthis_ds_ver_JJA = hgthis_ds_ver_JJA_copy.copy()
uhis_ds_ver_JJA = uhis_ds_ver_JJA_copy.copy()
vhis_ds_ver_JJA = vhis_ds_ver_JJA_copy.copy()
qhis_ds_ver_JJA = qhis_ds_ver_JJA_copy.copy()
sphis_ds_JJA = sphis_ds_JJA_copy.copy()
prehis_ds_JJA = prehis_ds_JJA_copy.copy()

hgthis_ds_ver_JJA.coords["models"] = models
uhis_ds_ver_JJA.coords["models"] = models
vhis_ds_ver_JJA.coords["models"] = models
qhis_ds_ver_JJA.coords["models"] = models
sphis_ds_JJA.coords["models"] = models
prehis_ds_JJA.coords["models"] = models
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
# %%
#   calculate the water vapor flux of multi-models
# his_dslevel = qhis_ds_ver_JJA.coords["level"] * 100.0
# his_dslevel.attrs["units"] = "Pa"
# # his_dsdp = geocat.comp.dpres_plevel(his_dslevel, sphis_ds_JJA, ptop)
# # print(sphis_ds_JJA)
# his_dsdp = xr.apply_ufunc(
#     geocat.comp.dpres_plevel,
#     his_dslevel,
#     sphis_ds_JJA,
#     ptop,
#     input_core_dims=[["level"], [], []],
#     output_core_dims=[["level"]],
#     vectorize=True,
#     dask="parallelized",
# )
# %%
# for i in np.arange(0,26):
#     print(his_dsdp[i, 0, 0, 0, :])
# his_dsdp = his_dsdp.transpose("models", "time", "level", "lat", "lon")
# his_dsdpg = his_dsdp / g
# his_dsdpg.attrs["units"] = "kg/m2"
his_dsdpg = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/his_dsdpg.nc")
uqhis_ds_ver_JJA = uhis_ds_ver_JJA * qhis_ds_ver_JJA * 1000.0
vqhis_ds_ver_JJA = vhis_ds_ver_JJA * qhis_ds_ver_JJA * 1000.0
uqhis_ds_ver_JJA.attrs["units"] = "[m/s][g/kg]"
vqhis_ds_ver_JJA.attrs["units"] = "[m/s][g/kg]"
uq_dpg_his_ds_JJA = (uqhis_ds_ver_JJA * his_dsdpg.data).sum(dim="level", skipna=True) / 1e05
vq_dpg_his_ds_JJA = (vqhis_ds_ver_JJA * his_dsdpg.data).sum(dim="level", skipna=True) / 1e05
uq_dpg_his_ds_JJA = ca.detrend_dim(uq_dpg_his_ds_JJA, "time", deg=1, demean=False)
vq_dpg_his_ds_JJA = ca.detrend_dim(vq_dpg_his_ds_JJA, "time", deg=1, demean=False)
uq_dpg_his_ds_JJA.attrs["units"] = "100kg/(m*s)"
vq_dpg_his_ds_JJA.attrs["units"] = "100kg/(m*s)"

# %%
# his_dsdpg.name = "his_dsdpg"
# his_dsdpg.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/his_dsdpg.nc")
# %%
#   calculate uq in India
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
    uq_dpg_his_ds_JJA.loc[:, :, 5:25, 50:80]
).mean(dim="lon", skipna=True)
uq_dpg_his_ds_India_JJA = ca.detrend_dim(
    uq_dpg_his_ds_India_JJA, "time", deg=1, demean=False
)
#   calculate vq in NCR


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
) = ca.dim_linregress(
    uq_dpg_his_ds_India_JJA, hgthis_ds_ver_JJA.sel(level=200.0)
)
(
    u_his_ds_India_uq_slope,
    u_his_ds_India_uq_intercept,
    u_his_ds_India_uq_rvalue,
    u_his_ds_India_uq_pvalue,
    u_his_ds_India_uq_hypothesis,
) = ca.dim_linregress(
    uq_dpg_his_ds_India_JJA, uhis_ds_ver_JJA.sel(level=200.0)
)
(
    v_his_ds_India_uq_slope,
    v_his_ds_India_uq_intercept,
    v_his_ds_India_uq_rvalue,
    v_his_ds_India_uq_pvalue,
    v_his_ds_India_uq_hypothesis,
) = ca.dim_linregress(
    uq_dpg_his_ds_India_JJA, vhis_ds_ver_JJA.sel(level=200.0)
)

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
# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=4, nrows=7, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # ??????????????????
yticks = np.arange(-30, 46, 15)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
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
con = axs[0].contourf(
    hgt_ERA5_India_uq_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_India_uq_pvalue,
    axs[0],
    n,
    np.where(hgt_ERA5_India_uq_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0].quiver(
    u_ERA5_India_uq_rvalue[::ski, ::ski],
    v_ERA5_India_uq_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0].quiver(
    u_ERA5_India_uq_rvalue.where(wind_ERA5_India_uq_mask > 0.0)[::ski, ::ski],
    v_ERA5_India_uq_rvalue.where(wind_ERA5_India_uq_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0].quiverkey(
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
axs[0].format(ltitle="India uq index", rtitle="ERA5 200hPa")
# ===================================================
con = axs[1].contourf(
    hgt_his_India_uq_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_India_uq_pvalue,
    axs[1],
    n,
    np.where(hgt_his_India_uq_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1].quiver(
    u_his_India_uq_rvalue[::ski, ::ski],
    v_his_India_uq_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1].quiver(
    u_his_India_uq_rvalue.where(wind_his_India_uq_mask > 0.0)[::ski, ::ski],
    v_his_India_uq_rvalue.where(wind_his_India_uq_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1].quiverkey(
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
axs[1].format(ltitle="India uq index", rtitle="ens 200hPa")
# ===================================================
for i, mod in enumerate(models):
    con = axs[i + 2].contourf(
        hgt_his_ds_India_uq_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-1.0, 1.1, 0.1),
        zorder=0.8,
    )
    sepl.plt_sig(
        hgt_his_ds_India_uq_pvalue.sel(models=mod),
        axs[i + 2],
        n,
        np.where(hgt_his_ds_India_uq_pvalue.sel(models=mod)[::n, ::n] <= 0.05),
        "denim",
        3.0,
    )

    axs[i + 2].quiver(
        u_his_ds_India_uq_rvalue.sel(models=mod)[::ski, ::ski],
        v_his_ds_India_uq_rvalue.sel(models=mod)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=0.17,
        pivot="mid",
        color="grey6",
    )

    m = axs[i + 2].quiver(
        u_his_ds_India_uq_rvalue.where(wind_his_ds_India_uq_mask > 0.0).sel(models=mod)[
            ::ski, ::ski
        ],
        v_his_ds_India_uq_rvalue.where(wind_his_ds_India_uq_mask > 0.0).sel(models=mod)[
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

    qk = axs[i + 2].quiverkey(
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
    axs[i + 2].format(
        ltitle="India uq index", rtitle="{} 200hPa".format(np.array(models[i]))
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
print(his_dsdpg)
# print(qhis_ds_ver_JJA)

# %%
print(uhis_ds_ver_JJA_copy.coords["models"],vhis_ds_ver_JJA_copy.coords["models"],qhis_ds_ver_JJA_copy.coords["models"],hgthis_ds_ver_JJA_copy.coords["models"],sphis_ds_JJA_copy.coords["models"])
# %%
# print(uhis_ds_ver_JJA[:,0,0,0,0])
# print(uhis_ds_ver_JJA_copy.sel(models="EC-Earth3"))
# print(uhis_ds_ver_JJA.coords["models"])
print(uhis_ds_ver_JJA[9,0,0,:,:])
# print(uhis_ds_ver_JJA_copy[0,0,0,:,:])


# %%
u_ERA5_India_JJA = ca.cal_lat_weighted_mean(
    uERA5_ver_JJA.sel(level=850.0).loc[:, 5:25, 50:80]
).mean(dim="lon", skipna=True)
u_ERA5_India_JJA = ca.detrend_dim(
    u_ERA5_India_JJA, "time", deg=1, demean=False
)
u_his_India_JJA = ca.cal_lat_weighted_mean(
    uhis_ver_JJA.sel(level=850.0).loc[:, 5:25, 50:80]
).mean(dim="lon", skipna=True)
u_his_India_JJA = ca.detrend_dim(u_his_India_JJA, "time", deg=1, demean=False)

u_his_ds_India_JJA = ca.cal_lat_weighted_mean(
    uhis_ds_ver_JJA.sel(level=850.0).loc[:, :, 5:25, 50:80]
).mean(dim="lon", skipna=True)
u_his_ds_India_JJA = ca.detrend_dim(
    u_his_ds_India_JJA, "time", deg=1, demean=False
)
# %%
q_ERA5_India_JJA = ca.cal_lat_weighted_mean(
    qERA5_ver_JJA.sel(level=850.0).loc[:, 5:25, 50:80]
).mean(dim="lon", skipna=True)
q_ERA5_India_JJA = ca.detrend_dim(
    q_ERA5_India_JJA, "time", deg=1, demean=False
)
q_his_India_JJA = ca.cal_lat_weighted_mean(
    qhis_ver_JJA.sel(level=850.0).loc[:, 5:25, 50:80]
).mean(dim="lon", skipna=True)
q_his_India_JJA = ca.detrend_dim(q_his_India_JJA, "time", deg=1, demean=False)

q_his_ds_India_JJA = ca.cal_lat_weighted_mean(
    qhis_ds_ver_JJA.sel(level=850.0).loc[:, :, 5:25, 50:80]
).mean(dim="lon", skipna=True)
q_his_ds_India_JJA = ca.detrend_dim(
    q_his_ds_India_JJA, "time", deg=1, demean=False
)
# %%
(
    hgt_ERA5_India_u_slope,
    hgt_ERA5_India_u_intercept,
    hgt_ERA5_India_u_rvalue,
    hgt_ERA5_India_u_pvalue,
    hgt_ERA5_India_u_hypothesis,
) = ca.dim_linregress(u_ERA5_India_JJA, hgtERA5_ver_JJA.sel(level=200.0))
(
    u_ERA5_India_u_slope,
    u_ERA5_India_u_intercept,
    u_ERA5_India_u_rvalue,
    u_ERA5_India_u_pvalue,
    u_ERA5_India_u_hypothesis,
) = ca.dim_linregress(u_ERA5_India_JJA, uERA5_ver_JJA.sel(level=200.0))
(
    v_ERA5_India_u_slope,
    v_ERA5_India_u_intercept,
    v_ERA5_India_u_rvalue,
    v_ERA5_India_u_pvalue,
    v_ERA5_India_u_hypothesis,
) = ca.dim_linregress(u_ERA5_India_JJA, vERA5_ver_JJA.sel(level=200.0))

(
    hgt_his_India_u_slope,
    hgt_his_India_u_intercept,
    hgt_his_India_u_rvalue,
    hgt_his_India_u_pvalue,
    hgt_his_India_u_hypothesis,
) = ca.dim_linregress(u_his_India_JJA, hgthis_ver_JJA.sel(level=200.0))
(
    u_his_India_u_slope,
    u_his_India_u_intercept,
    u_his_India_u_rvalue,
    u_his_India_u_pvalue,
    u_his_India_u_hypothesis,
) = ca.dim_linregress(u_his_India_JJA, uhis_ver_JJA.sel(level=200.0))
(
    v_his_India_u_slope,
    v_his_India_u_intercept,
    v_his_India_u_rvalue,
    v_his_India_u_pvalue,
    v_his_India_u_hypothesis,
) = ca.dim_linregress(u_his_India_JJA, vhis_ver_JJA.sel(level=200.0))

(
    hgt_his_ds_India_u_slope,
    hgt_his_ds_India_u_intercept,
    hgt_his_ds_India_u_rvalue,
    hgt_his_ds_India_u_pvalue,
    hgt_his_ds_India_u_hypothesis,
) = ca.dim_linregress(
    u_his_ds_India_JJA, hgthis_ds_ver_JJA.sel(level=200.0)
)
(
    u_his_ds_India_u_slope,
    u_his_ds_India_u_intercept,
    u_his_ds_India_u_rvalue,
    u_his_ds_India_u_pvalue,
    u_his_ds_India_u_hypothesis,
) = ca.dim_linregress(
    u_his_ds_India_JJA, uhis_ds_ver_JJA.sel(level=200.0)
)
(
    v_his_ds_India_u_slope,
    v_his_ds_India_u_intercept,
    v_his_ds_India_u_rvalue,
    v_his_ds_India_u_pvalue,
    v_his_ds_India_u_hypothesis,
) = ca.dim_linregress(
    u_his_ds_India_JJA, vhis_ds_ver_JJA.sel(level=200.0)
)
# %%
q_ERA5_India_JJA.coords["time"] = hgtERA5_ver_JJA.coords["time"]

(
    hgt_ERA5_India_q_slope,
    hgt_ERA5_India_q_intercept,
    hgt_ERA5_India_q_rvalue,
    hgt_ERA5_India_q_pvalue,
    hgt_ERA5_India_q_hypothesis,
) = ca.dim_linregress(q_ERA5_India_JJA, hgtERA5_ver_JJA.sel(level=200.0))
(
    u_ERA5_India_q_slope,
    u_ERA5_India_q_intercept,
    u_ERA5_India_q_rvalue,
    u_ERA5_India_q_pvalue,
    u_ERA5_India_q_hypothesis,
) = ca.dim_linregress(q_ERA5_India_JJA, uERA5_ver_JJA.sel(level=200.0))
(
    v_ERA5_India_q_slope,
    v_ERA5_India_q_intercept,
    v_ERA5_India_q_rvalue,
    v_ERA5_India_q_pvalue,
    v_ERA5_India_q_hypothesis,
) = ca.dim_linregress(q_ERA5_India_JJA, vERA5_ver_JJA.sel(level=200.0))

(
    hgt_his_India_q_slope,
    hgt_his_India_q_intercept,
    hgt_his_India_q_rvalue,
    hgt_his_India_q_pvalue,
    hgt_his_India_q_hypothesis,
) = ca.dim_linregress(q_his_India_JJA, hgthis_ver_JJA.sel(level=200.0))
(
    u_his_India_q_slope,
    u_his_India_q_intercept,
    u_his_India_q_rvalue,
    u_his_India_q_pvalue,
    u_his_India_q_hypothesis,
) = ca.dim_linregress(q_his_India_JJA, uhis_ver_JJA.sel(level=200.0))
(
    v_his_India_q_slope,
    v_his_India_q_intercept,
    v_his_India_q_rvalue,
    v_his_India_q_pvalue,
    v_his_India_q_hypothesis,
) = ca.dim_linregress(q_his_India_JJA, vhis_ver_JJA.sel(level=200.0))

(
    hgt_his_ds_India_q_slope,
    hgt_his_ds_India_q_intercept,
    hgt_his_ds_India_q_rvalue,
    hgt_his_ds_India_q_pvalue,
    hgt_his_ds_India_q_hypothesis,
) = ca.dim_linregress(
    q_his_ds_India_JJA, hgthis_ds_ver_JJA.sel(level=200.0)
)
(
    u_his_ds_India_q_slope,
    u_his_ds_India_q_intercept,
    u_his_ds_India_q_rvalue,
    u_his_ds_India_q_pvalue,
    u_his_ds_India_q_hypothesis,
) = ca.dim_linregress(
    q_his_ds_India_JJA, uhis_ds_ver_JJA.sel(level=200.0)
)
(
    v_his_ds_India_q_slope,
    v_his_ds_India_q_intercept,
    v_his_ds_India_q_rvalue,
    v_his_ds_India_q_pvalue,
    v_his_ds_India_q_hypothesis,
) = ca.dim_linregress(
    q_his_ds_India_JJA, vhis_ds_ver_JJA.sel(level=200.0)
)
# %%
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
wind_his_ds_India_u_mask = ca.wind_check(
    xr.where(u_his_ds_India_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_ds_India_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_ds_India_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_ds_India_u_pvalue <= 0.05, 1.0, 0.0),
)
wind_ERA5_India_q_mask = ca.wind_check(
    xr.where(u_ERA5_India_q_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_India_q_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_ERA5_India_q_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_India_q_pvalue <= 0.05, 1.0, 0.0),
)
wind_his_India_q_mask = ca.wind_check(
    xr.where(u_his_India_q_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_India_q_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_India_q_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_India_q_pvalue <= 0.05, 1.0, 0.0),
)
wind_his_ds_India_q_mask = ca.wind_check(
    xr.where(u_his_ds_India_q_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_ds_India_q_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_ds_India_q_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_ds_India_q_pvalue <= 0.05, 1.0, 0.0),
)
# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=4, nrows=7, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # ??????????????????
yticks = np.arange(-30, 46, 15)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
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
con = axs[0].contourf(
    hgt_ERA5_India_u_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_India_u_pvalue,
    axs[0],
    n,
    np.where(hgt_ERA5_India_u_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0].quiver(
    u_ERA5_India_u_rvalue[::ski, ::ski],
    v_ERA5_India_u_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0].quiver(
    u_ERA5_India_u_rvalue.where(wind_ERA5_India_u_mask > 0.0)[::ski, ::ski],
    v_ERA5_India_u_rvalue.where(wind_ERA5_India_u_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0].quiverkey(
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
axs[0].format(ltitle="India u index", rtitle="ERA5 200hPa")
# ===================================================
con = axs[1].contourf(
    hgt_his_India_u_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_India_u_pvalue,
    axs[1],
    n,
    np.where(hgt_his_India_u_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1].quiver(
    u_his_India_u_rvalue[::ski, ::ski],
    v_his_India_u_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1].quiver(
    u_his_India_u_rvalue.where(wind_his_India_u_mask > 0.0)[::ski, ::ski],
    v_his_India_u_rvalue.where(wind_his_India_u_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1].quiverkey(
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
axs[1].format(ltitle="India u index", rtitle="ens 200hPa")
# ===================================================
for i, mod in enumerate(models):
    con = axs[i + 2].contourf(
        hgt_his_ds_India_u_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-1.0, 1.1, 0.1),
        zorder=0.8,
    )
    sepl.plt_sig(
        hgt_his_ds_India_u_pvalue.sel(models=mod),
        axs[i + 2],
        n,
        np.where(hgt_his_ds_India_u_pvalue.sel(models=mod)[::n, ::n] <= 0.05),
        "denim",
        3.0,
    )

    axs[i + 2].quiver(
        u_his_ds_India_u_rvalue.sel(models=mod)[::ski, ::ski],
        v_his_ds_India_u_rvalue.sel(models=mod)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=0.17,
        pivot="mid",
        color="grey6",
    )

    m = axs[i + 2].quiver(
        u_his_ds_India_u_rvalue.where(wind_his_ds_India_u_mask > 0.0).sel(models=mod)[
            ::ski, ::ski
        ],
        v_his_ds_India_u_rvalue.where(wind_his_ds_India_u_mask > 0.0).sel(models=mod)[
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

    qk = axs[i + 2].quiverkey(
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
    axs[i + 2].format(
        ltitle="India u index", rtitle="{} 200hPa".format(np.array(models[i]))
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")

# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=4, nrows=7, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # ??????????????????
yticks = np.arange(-30, 46, 15)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
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
con = axs[0].contourf(
    hgt_ERA5_India_q_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_India_q_pvalue,
    axs[0],
    n,
    np.where(hgt_ERA5_India_q_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0].quiver(
    u_ERA5_India_q_rvalue[::ski, ::ski],
    v_ERA5_India_q_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0].quiver(
    u_ERA5_India_q_rvalue.where(wind_ERA5_India_q_mask > 0.0)[::ski, ::ski],
    v_ERA5_India_q_rvalue.where(wind_ERA5_India_q_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0].quiverkey(
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
axs[0].format(ltitle="India q index", rtitle="ERA5 200hPa")
# ===================================================
con = axs[1].contourf(
    hgt_his_India_q_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_India_q_pvalue,
    axs[1],
    n,
    np.where(hgt_his_India_q_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1].quiver(
    u_his_India_q_rvalue[::ski, ::ski],
    v_his_India_q_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1].quiver(
    u_his_India_q_rvalue.where(wind_his_India_q_mask > 0.0)[::ski, ::ski],
    v_his_India_q_rvalue.where(wind_his_India_q_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1].quiverkey(
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
axs[1].format(ltitle="India q index", rtitle="ens 200hPa")
# ===================================================
for i, mod in enumerate(models):
    con = axs[i + 2].contourf(
        hgt_his_ds_India_q_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-1.0, 1.1, 0.1),
        zorder=0.8,
    )
    sepl.plt_sig(
        hgt_his_ds_India_q_pvalue.sel(models=mod),
        axs[i + 2],
        n,
        np.where(hgt_his_ds_India_q_pvalue.sel(models=mod)[::n, ::n] <= 0.05),
        "denim",
        3.0,
    )

    axs[i + 2].quiver(
        u_his_ds_India_q_rvalue.sel(models=mod)[::ski, ::ski],
        v_his_ds_India_q_rvalue.sel(models=mod)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=0.17,
        pivot="mid",
        color="grey6",
    )

    m = axs[i + 2].quiver(
        u_his_ds_India_q_rvalue.where(wind_his_ds_India_q_mask > 0.0).sel(models=mod)[
            ::ski, ::ski
        ],
        v_his_ds_India_q_rvalue.where(wind_his_ds_India_q_mask > 0.0).sel(models=mod)[
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

    qk = axs[i + 2].quiverkey(
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
    axs[i + 2].format(
        ltitle="India q index", rtitle="{} 200hPa".format(np.array(models[i]))
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
#   calculate the pcc
reload(ca)
ulist = []
for i, mod in enumerate(models):
    ulist.append({"models": str(np.array(mod)),"pcc": ca.cal_pcc(hgt_ERA5_India_u_rvalue.loc[-30:55, 30:180], hgt_his_ds_India_u_rvalue.sel(models=mod).loc[-30:55, 30:180])})
# print(ulist)
print(sorted(ulist, key=lambda x : x["pcc"]))
# %%
uqlist = []
for i, mod in enumerate(models):
    uqlist.append({"models": str(np.array(mod)),"pcc": ca.cal_pcc(hgt_ERA5_India_uq_rvalue.loc[-30:55, 30:180], hgt_his_ds_India_uq_rvalue.sel(models=mod).loc[-30:55, 30:180])})
# print(uqlist)
print(sorted(uqlist, key=lambda x : x["pcc"]))
# %%
qlist = []
for i, mod in enumerate(models):
    qlist.append({"models": str(np.array(mod)),"pcc": ca.cal_pcc(hgt_ERA5_India_q_rvalue.loc[-30:55, 30:180], hgt_his_ds_India_q_rvalue.sel(models=mod).loc[-30:55, 30:180])})
# print(qlist)
print(sorted(qlist, key=lambda x : x["pcc"]))
# %%
print(models)

# %%
hgthis_ver_JJA_new =hgthis_ds_ver_JJA[0:6,:,:,:,:]
hgthis_ver_JJA_new = xr.concat([hgthis_ver_JJA_new, hgthis_ds_ver_JJA[7:11,:,:,:,:], hgthis_ds_ver_JJA[12:14,:,:,:,:], hgthis_ds_ver_JJA[15:16,:,:,:,:], hgthis_ds_ver_JJA[17:24,:,:,:,:], hgthis_ds_ver_JJA[25:26,:,:,:,:]], "models")
hgthis_ver_JJA_new = hgthis_ver_JJA_new.mean(dim="models", skipna=True)
hgthis_ver_JJA_new = ca.detrend_dim(hgthis_ver_JJA_new, "time", deg=1, demean=False)

uhis_ver_JJA_new =uhis_ds_ver_JJA[0:6,:,:,:,:]
uhis_ver_JJA_new = xr.concat([uhis_ver_JJA_new, uhis_ds_ver_JJA[7:11,:,:,:,:], uhis_ds_ver_JJA[12:14,:,:,:,:], uhis_ds_ver_JJA[15:16,:,:,:,:], uhis_ds_ver_JJA[17:24,:,:,:,:], uhis_ds_ver_JJA[25:26,:,:,:,:]], "models")
print(uhis_ver_JJA_new)
uhis_ver_JJA_new = uhis_ver_JJA_new.mean(dim="models", skipna=True)
uhis_ver_JJA_new = ca.detrend_dim(uhis_ver_JJA_new, "time", deg=1, demean=False)

vhis_ver_JJA_new =vhis_ds_ver_JJA[0:6,:,:,:,:]
vhis_ver_JJA_new = xr.concat([vhis_ver_JJA_new, vhis_ds_ver_JJA[7:11,:,:,:,:], vhis_ds_ver_JJA[12:14,:,:,:,:], vhis_ds_ver_JJA[15:16,:,:,:,:], vhis_ds_ver_JJA[17:24,:,:,:,:], vhis_ds_ver_JJA[25:26,:,:,:,:]], "models")
vhis_ver_JJA_new = vhis_ver_JJA_new.mean(dim="models", skipna=True)
vhis_ver_JJA_new = ca.detrend_dim(vhis_ver_JJA_new, "time", deg=1, demean=False)

qhis_ver_JJA_new =qhis_ds_ver_JJA[0:6,:,:,:,:]
qhis_ver_JJA_new = xr.concat([qhis_ver_JJA_new, qhis_ds_ver_JJA[7:11,:,:,:,:], qhis_ds_ver_JJA[12:14,:,:,:,:], qhis_ds_ver_JJA[15:16,:,:,:,:], qhis_ds_ver_JJA[17:24,:,:,:,:], qhis_ds_ver_JJA[25:26,:,:,:,:]], "models")
qhis_ver_JJA_new = qhis_ver_JJA_new.mean(dim="models", skipna=True)
qhis_ver_JJA_new = ca.detrend_dim(qhis_ver_JJA_new, "time", deg=1, demean=False)

sphis_JJA_new =sphis_ds_JJA[0:6,:,:,:]
sphis_JJA_new = xr.concat([sphis_JJA_new, sphis_ds_JJA[7:11,:,:,:], sphis_ds_JJA[12:14,:,:,:], sphis_ds_JJA[15:16,:,:,:], sphis_ds_JJA[17:24,:,:,:], sphis_ds_JJA[25:26,:,:,:]], "models")
sphis_JJA_new = sphis_JJA_new.mean(dim="models", skipna=True)
sphis_JJA_new = ca.detrend_dim(sphis_JJA_new, "time", deg=1, demean=False)

# %%
hislevel = qhis_ver_JJA_new.coords["level"] * 100.0
hislevel.attrs["units"] = "Pa"
hisdp = geocat.comp.dpres_plevel(hislevel, sphis_JJA_new, ptop)
hisdpg = hisdp / g
hisdpg.attrs["units"] = "kg/m2"
uqhis_ver_JJA_new = uhis_ver_JJA_new * qhis_ver_JJA_new.data * 1000.0
# vqhis_ver_JJA_new = vhis_ver_JJA_new * qhis_ver_JJA_new.data * 1000.0
uqhis_ver_JJA_new.attrs["units"] = "[m/s][g/kg]"
# vqhis_ver_JJA_new.attrs["units"] = "[m/s][g/kg]"
uq_dpg_his_JJA = (uqhis_ver_JJA_new * hisdpg.data).sum(dim="level", skipna=True)
# vq_dpg_his_JJA = (vqhis_ver_JJA_new * hisdpg.data).sum(dim="level", skipna=True)
uq_dpg_his_JJA = ca.detrend_dim(uq_dpg_his_JJA, "time", deg=1, demean=False)
# vq_dpg_his_JJA = ca.detrend_dim(vq_dpg_his_JJA, "time", deg=1, demean=False)
# %%
uq_dpg_his_India_JJA = ca.cal_lat_weighted_mean(
    uq_dpg_his_JJA.loc[:, 5:25, 50:80]
).mean(dim="lon", skipna=True)
uq_dpg_his_India_JJA = ca.detrend_dim(uq_dpg_his_India_JJA, "time", deg=1, demean=False)
u_his_India_JJA = ca.cal_lat_weighted_mean(
    uhis_ver_JJA_new.sel(level=850.0).loc[:, 5:25, 50:80]
).mean(dim="lon", skipna=True)
u_his_India_JJA = ca.detrend_dim(u_his_India_JJA, "time", deg=1, demean=False)
# %%
(
    hgt_his_India_u_slope,
    hgt_his_India_u_intercept,
    hgt_his_India_u_rvalue,
    hgt_his_India_u_pvalue,
    hgt_his_India_u_hypothesis,
) = ca.dim_linregress(u_his_India_JJA, hgthis_ver_JJA_new.sel(level=200.0))
(
    u_his_India_u_slope,
    u_his_India_u_intercept,
    u_his_India_u_rvalue,
    u_his_India_u_pvalue,
    u_his_India_u_hypothesis,
) = ca.dim_linregress(u_his_India_JJA, uhis_ver_JJA_new.sel(level=200.0))
(
    v_his_India_u_slope,
    v_his_India_u_intercept,
    v_his_India_u_rvalue,
    v_his_India_u_pvalue,
    v_his_India_u_hypothesis,
) = ca.dim_linregress(u_his_India_JJA, vhis_ver_JJA_new.sel(level=200.0))
# %%
(
    hgt_his_India_uq_slope,
    hgt_his_India_uq_intercept,
    hgt_his_India_uq_rvalue,
    hgt_his_India_uq_pvalue,
    hgt_his_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_his_India_JJA, hgthis_ver_JJA_new.sel(level=200.0))
(
    u_his_India_uq_slope,
    u_his_India_uq_intercept,
    u_his_India_uq_rvalue,
    u_his_India_uq_pvalue,
    u_his_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_his_India_JJA, uhis_ver_JJA_new.sel(level=200.0))
(
    v_his_India_uq_slope,
    v_his_India_uq_intercept,
    v_his_India_uq_rvalue,
    v_his_India_uq_pvalue,
    v_his_India_uq_hypothesis,
) = ca.dim_linregress(uq_dpg_his_India_JJA, vhis_ver_JJA_new.sel(level=200.0))
# %%
wind_his_India_uq_mask = ca.wind_check(
    xr.where(u_his_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_India_uq_pvalue <= 0.05, 1.0, 0.0),
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
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=4, nrows=7, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # ??????????????????
yticks = np.arange(-30, 46, 15)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
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
con = axs[0].contourf(
    hgt_ERA5_India_uq_intercept,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.1},
    levels=np.arange(11400, 12601, 40),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    hgt_ERA5_India_uq_pvalue,
    axs[0],
    n,
    np.where(hgt_ERA5_India_uq_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0].quiver(
    u_ERA5_India_uq_rvalue[::ski, ::ski],
    v_ERA5_India_uq_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0].quiver(
    u_ERA5_India_uq_rvalue.where(wind_ERA5_India_uq_mask > 0.0)[::ski, ::ski],
    v_ERA5_India_uq_rvalue.where(wind_ERA5_India_uq_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0].quiverkey(
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
axs[0].format(ltitle="India uq index", rtitle="ERA5 200hPa")
# ===================================================
con = axs[1].contourf(
    hgt_his_India_uq_intercept,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.1},
    levels=np.arange(11400, 12601, 40),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    hgt_his_India_uq_pvalue,
    axs[1],
    n,
    np.where(hgt_his_India_uq_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1].quiver(
    u_his_India_uq_rvalue[::ski, ::ski],
    v_his_India_uq_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1].quiver(
    u_his_India_uq_rvalue.where(wind_his_India_uq_mask > 0.0)[::ski, ::ski],
    v_his_India_uq_rvalue.where(wind_his_India_uq_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1].quiverkey(
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
axs[1].format(ltitle="India uq index", rtitle="ens 200hPa")
# ===================================================
for i, mod in enumerate(models):
    con = axs[i + 2].contourf(
        hgt_his_ds_India_uq_intercept.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.1},
        levels=np.arange(11400, 12601, 40),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        hgt_his_ds_India_uq_pvalue.sel(models=mod),
        axs[i + 2],
        n,
        np.where(hgt_his_ds_India_uq_pvalue.sel(models=mod)[::n, ::n] <= 0.05),
        "denim",
        3.0,
    )

    axs[i + 2].quiver(
        u_his_ds_India_uq_rvalue.sel(models=mod)[::ski, ::ski],
        v_his_ds_India_uq_rvalue.sel(models=mod)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=0.17,
        pivot="mid",
        color="grey6",
    )

    m = axs[i + 2].quiver(
        u_his_ds_India_uq_rvalue.where(wind_his_ds_India_uq_mask > 0.0).sel(models=mod)[
            ::ski, ::ski
        ],
        v_his_ds_India_uq_rvalue.where(wind_his_ds_India_uq_mask > 0.0).sel(models=mod)[
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

    qk = axs[i + 2].quiverkey(
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
    axs[i + 2].format(
        ltitle="India uq index", rtitle="{} 200hPa".format(np.array(models[i]))
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
#   interpolate the nan in wind fields
lat = uhis_ds_ver_JJA.coords["lat"]
lon = uhis_ds_ver_JJA.coords["lon"]
time = uhis_ds_ver_JJA.coords["time"]
uhis_ds_ver_JJA_filled = uhis_ds_ver_JJA.sel(level=850.0).copy()
vhis_ds_ver_JJA_filled = vhis_ds_ver_JJA.sel(level=850.0).copy()
for i,mod in enumerate(models):
    for ti in range(len(time)):
        u_filled_func = interp2d(lon, lat, uhis_ds_ver_JJA.sel(level=850.0)[i,ti,:,:], kind="linear", bounds_error=False)
        uhis_ds_ver_JJA_filled[i,ti,:,:] = u_filled_func(lon, lat)
        v_filled_func = interp2d(lon, lat, vhis_ds_ver_JJA.sel(level=850.0)[i,ti,:,:], kind="linear", bounds_error=False)
        vhis_ds_ver_JJA_filled[i,ti,:,:] = v_filled_func(lon, lat)
lenmodels = np.arange(len(models))
uhis_ds_ver_JJA_filled.coords["models"] = lenmodels
vhis_ds_ver_JJA_filled.coords["models"] = lenmodels

uhis_ds_ver_JJA_filled = uhis_ds_ver_JJA_filled.interpolate_na(dim="models",method="nearest", fill_value="extrapolate")
vhis_ds_ver_JJA_filled = vhis_ds_ver_JJA_filled.interpolate_na(dim="models",method="nearest", fill_value="extrapolate")
uhis_ds_ver_JJA_filled.coords["models"] = models
vhis_ds_ver_JJA_filled.coords["models"] = models
# print(uhis_ds_ver_JJA_filled)


# %%
#   plot the climatology of two indexes
windhis_ds_JJA = VectorWind(uhis_ds_ver_JJA_filled, vhis_ds_ver_JJA_filled)
vorhis_ds_JJA = windhis_ds_JJA.vorticity()
vorhis_ds_JJA = ca.detrend_dim(vorhis_ds_JJA, "time", deg=1, demean=False)

windhis_JJA = VectorWind(uhis_ver_JJA.sel(level=850.0), vhis_ver_JJA.sel(level=850.0))
vorhis_JJA = windhis_JJA.vorticity()
vorhis_JJA = ca.detrend_dim(vorhis_JJA, "time", deg=1, demean=False)

windERA5_JJA = VectorWind(uERA5_ver_JJA.sel(level=850.0), vERA5_ver_JJA.sel(level=850.0))
vorERA5_JJA = windERA5_JJA.vorticity()
vorERA5_JJA = ca.detrend_dim(vorERA5_JJA, "time", deg=1, demean=False)
# %%
#   calculate the vorticity climatological mean
vorhis_ds_JJA_mean = vorhis_ds_JJA.mean(dim="time", skipna=True)
vorhis_JJA_mean = vorhis_JJA.mean(dim="time", skipna=True)
vorERA5_JJA_mean = vorERA5_JJA.mean(dim="time", skipna=True)

# %%
print(vorERA5_JJA_mean.max())
# %%
#   plot the climatology vorticity
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=4, nrows=7, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # ??????????????????
yticks = np.arange(-30, 46, 15)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # region 1
    x0 = 90
    y0 = 5.0
    width = 50
    height = 27.5
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    vorERA5_JJA_mean,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.1},
    levels=np.arange(-3e-05, 3.01e-05, 3e-06),
    zorder=0.8,
    extend="both"
)
axs[0].format(ltitle="vorticity", rtitle="ERA5 850hPa")

# ===================================================
con = axs[1].contourf(
    vorhis_JJA_mean,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.1},
    levels=np.arange(-3e-05, 3.01e-05, 3e-06),
    zorder=0.8,
    extend="both"
)
axs[1].format(ltitle="vorticity", rtitle="ens 850hPa")

# ===================================================
for i, mod in enumerate(models):
    con = axs[i + 2].contourf(
        vorhis_ds_JJA_mean.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.1},
        levels=np.arange(-3e-05, 3.01e-05, 3e-06),
        zorder=0.8,
        extend="both"
    )
    axs[i + 2].format(ltitle="vorticity", rtitle="{} 850hPa".format(str(mod.data)))
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
#   calculate the meridional wind shear
vERA5_shear = vERA5_ver_JJA.sel(level=850.0) - vERA5_ver_JJA.sel(level=200.0)
vERA5_shear = ca.detrend_dim(vERA5_shear, "time", deg=1, demean=False)
vERA5_shear_mean = vERA5_shear.mean(dim="time", skipna=True)

vhis_shear = vhis_ver_JJA.sel(level=850.0) - vhis_ver_JJA.sel(level=200.0)
vhis_shear = ca.detrend_dim(vhis_shear, "time", deg=1, demean=False)
vhis_shear_mean = vhis_shear.mean(dim="time", skipna=True)


vhis_ds_shear = vhis_ds_ver_JJA.sel(level=850.0) - vhis_ds_ver_JJA.sel(level=200.0)
vhis_ds_shear = ca.detrend_dim(vhis_ds_shear, "time", deg=1, demean=False)
vhis_ds_shear_mean = vhis_ds_shear.mean(dim="time", skipna=True)
# %%
#   plot the meridional wind shear
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=4, nrows=7, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # ??????????????????
yticks = np.arange(-30, 46, 15)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # region 1
    x0 = 70
    y0 = 10
    width = 40
    height = 20
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    vERA5_shear_mean,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.1},
    levels=np.arange(-15, 16, 1),
    zorder=0.8,
    extend="both"
)
axs[0].format(ltitle="v850-v200", rtitle="ERA5 850hPa")

# ===================================================
con = axs[1].contourf(
    vhis_shear_mean,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.1},
    levels=np.arange(-15, 16, 1),
    zorder=0.8,
    extend="both"
)
axs[1].format(ltitle="v850-v200", rtitle="ens 850hPa")

# ===================================================
for i, mod in enumerate(models):
    con = axs[i + 2].contourf(
        vhis_ds_shear_mean.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.1},
        levels=np.arange(-15, 16, 1),
        zorder=0.8,
        extend="both"
    )
    axs[i + 2].format(ltitle="v850-v200", rtitle="{} 850hPa".format(str(mod.data)))
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
print(vERA5_shear_mean)
# %%
shearlist = []
for i, mod in enumerate(models):
    shearlist.append({"models": str(mod.data),"pcc": ca.cal_pcc(vERA5_shear_mean.loc[-30:55, 30:180], vhis_ds_shear_mean.sel(models=mod).loc[-30:55, 30:180])})
# print(shearlist)
print(sorted(shearlist, key=lambda x : x["pcc"]))

# %%
vorlist = []
for i, mod in enumerate(models):
    vorlist.append({"models": str(mod.data),"pcc": ca.cal_pcc(vorERA5_JJA_mean.loc[-30:55, 30:180], vorhis_ds_JJA_mean.sel(models=mod).loc[-30:55, 30:180])})
# print(vorlist)
print(sorted(vorlist, key=lambda x : x["pcc"]))
# %%
#   calculate the 850hPa hgt/u/v regress onto u_India
(
    hgt_ERA5_India_u_slope,
    hgt_ERA5_India_u_intercept,
    hgt_ERA5_India_u_rvalue,
    hgt_ERA5_India_u_pvalue,
    hgt_ERA5_India_u_hypothesis,
) = ca.dim_linregress(u_ERA5_India_JJA, hgtERA5_ver_JJA.sel(level=850.0))
(
    u_ERA5_India_u_slope,
    u_ERA5_India_u_intercept,
    u_ERA5_India_u_rvalue,
    u_ERA5_India_u_pvalue,
    u_ERA5_India_u_hypothesis,
) = ca.dim_linregress(u_ERA5_India_JJA, uERA5_ver_JJA.sel(level=850.0))
(
    v_ERA5_India_u_slope,
    v_ERA5_India_u_intercept,
    v_ERA5_India_u_rvalue,
    v_ERA5_India_u_pvalue,
    v_ERA5_India_u_hypothesis,
) = ca.dim_linregress(u_ERA5_India_JJA, vERA5_ver_JJA.sel(level=850.0))

(
    hgt_his_India_u_slope,
    hgt_his_India_u_intercept,
    hgt_his_India_u_rvalue,
    hgt_his_India_u_pvalue,
    hgt_his_India_u_hypothesis,
) = ca.dim_linregress(u_his_India_JJA, hgthis_ver_JJA.sel(level=850.0))
(
    u_his_India_u_slope,
    u_his_India_u_intercept,
    u_his_India_u_rvalue,
    u_his_India_u_pvalue,
    u_his_India_u_hypothesis,
) = ca.dim_linregress(u_his_India_JJA, uhis_ver_JJA.sel(level=850.0))
(
    v_his_India_u_slope,
    v_his_India_u_intercept,
    v_his_India_u_rvalue,
    v_his_India_u_pvalue,
    v_his_India_u_hypothesis,
) = ca.dim_linregress(u_his_India_JJA, vhis_ver_JJA.sel(level=850.0))

(
    hgt_his_ds_India_u_slope,
    hgt_his_ds_India_u_intercept,
    hgt_his_ds_India_u_rvalue,
    hgt_his_ds_India_u_pvalue,
    hgt_his_ds_India_u_hypothesis,
) = ca.dim_linregress(
    u_his_ds_India_JJA, hgthis_ds_ver_JJA.sel(level=850.0)
)
(
    u_his_ds_India_u_slope,
    u_his_ds_India_u_intercept,
    u_his_ds_India_u_rvalue,
    u_his_ds_India_u_pvalue,
    u_his_ds_India_u_hypothesis,
) = ca.dim_linregress(
    u_his_ds_India_JJA, uhis_ds_ver_JJA.sel(level=850.0)
)
(
    v_his_ds_India_u_slope,
    v_his_ds_India_u_intercept,
    v_his_ds_India_u_rvalue,
    v_his_ds_India_u_pvalue,
    v_his_ds_India_u_hypothesis,
) = ca.dim_linregress(
    u_his_ds_India_JJA, vhis_ds_ver_JJA.sel(level=850.0)
)
# %%
#   generate the wind mask of 850hgt/u/v/ regress onto u_India
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
wind_his_ds_India_u_mask = ca.wind_check(
    xr.where(u_his_ds_India_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_ds_India_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_ds_India_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_ds_India_u_pvalue <= 0.05, 1.0, 0.0),
)

# %%
#   plot the 850hPa hgt/u/v/ regress onto u_India
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=4, nrows=7, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # ??????????????????
yticks = np.arange(-30, 46, 15)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
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
con = axs[0].contourf(
    hgt_ERA5_India_u_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_India_u_pvalue,
    axs[0],
    n,
    np.where(hgt_ERA5_India_u_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0].quiver(
    u_ERA5_India_u_rvalue[::ski, ::ski],
    v_ERA5_India_u_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0].quiver(
    u_ERA5_India_u_rvalue.where(wind_ERA5_India_u_mask > 0.0)[::ski, ::ski],
    v_ERA5_India_u_rvalue.where(wind_ERA5_India_u_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0].quiverkey(
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
axs[0].format(ltitle="India u index", rtitle="ERA5 850hPa")
# ===================================================
con = axs[1].contourf(
    hgt_his_India_u_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_India_u_pvalue,
    axs[1],
    n,
    np.where(hgt_his_India_u_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1].quiver(
    u_his_India_u_rvalue[::ski, ::ski],
    v_his_India_u_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1].quiver(
    u_his_India_u_rvalue.where(wind_his_India_u_mask > 0.0)[::ski, ::ski],
    v_his_India_u_rvalue.where(wind_his_India_u_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1].quiverkey(
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
axs[1].format(ltitle="India u index", rtitle="ens 850hPa")
# ===================================================
for i, mod in enumerate(models):
    con = axs[i + 2].contourf(
        hgt_his_ds_India_u_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-1.0, 1.1, 0.1),
        zorder=0.8,
    )
    sepl.plt_sig(
        hgt_his_ds_India_u_pvalue.sel(models=mod),
        axs[i + 2],
        n,
        np.where(hgt_his_ds_India_u_pvalue.sel(models=mod)[::n, ::n] <= 0.05),
        "denim",
        3.0,
    )

    axs[i + 2].quiver(
        u_his_ds_India_u_rvalue.sel(models=mod)[::ski, ::ski],
        v_his_ds_India_u_rvalue.sel(models=mod)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=0.17,
        pivot="mid",
        color="grey6",
    )

    m = axs[i + 2].quiver(
        u_his_ds_India_u_rvalue.where(wind_his_ds_India_u_mask > 0.0).sel(models=mod)[
            ::ski, ::ski
        ],
        v_his_ds_India_u_rvalue.where(wind_his_ds_India_u_mask > 0.0).sel(models=mod)[
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

    qk = axs[i + 2].quiverkey(
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
    axs[i + 2].format(
        ltitle="India u index", rtitle="{} 850hPa".format(np.array(models[i]))
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%

(
    hgt_ERA5_India_u_slope,
    hgt_ERA5_India_u_intercept,
    hgt_ERA5_India_u_rvalue,
    hgt_ERA5_India_u_pvalue,
    hgt_ERA5_India_u_hypothesis,
) = ca.dim_linregress(u_ERA5_India_JJA, hgtERA5_ver_JJA.sel(level=500.0))
(
    u_ERA5_India_u_slope,
    u_ERA5_India_u_intercept,
    u_ERA5_India_u_rvalue,
    u_ERA5_India_u_pvalue,
    u_ERA5_India_u_hypothesis,
) = ca.dim_linregress(u_ERA5_India_JJA, uERA5_ver_JJA.sel(level=500.0))
(
    v_ERA5_India_u_slope,
    v_ERA5_India_u_intercept,
    v_ERA5_India_u_rvalue,
    v_ERA5_India_u_pvalue,
    v_ERA5_India_u_hypothesis,
) = ca.dim_linregress(u_ERA5_India_JJA, vERA5_ver_JJA.sel(level=500.0))

(
    hgt_his_India_u_slope,
    hgt_his_India_u_intercept,
    hgt_his_India_u_rvalue,
    hgt_his_India_u_pvalue,
    hgt_his_India_u_hypothesis,
) = ca.dim_linregress(u_his_India_JJA, hgthis_ver_JJA.sel(level=500.0))
(
    u_his_India_u_slope,
    u_his_India_u_intercept,
    u_his_India_u_rvalue,
    u_his_India_u_pvalue,
    u_his_India_u_hypothesis,
) = ca.dim_linregress(u_his_India_JJA, uhis_ver_JJA.sel(level=500.0))
(
    v_his_India_u_slope,
    v_his_India_u_intercept,
    v_his_India_u_rvalue,
    v_his_India_u_pvalue,
    v_his_India_u_hypothesis,
) = ca.dim_linregress(u_his_India_JJA, vhis_ver_JJA.sel(level=500.0))

(
    hgt_his_ds_India_u_slope,
    hgt_his_ds_India_u_intercept,
    hgt_his_ds_India_u_rvalue,
    hgt_his_ds_India_u_pvalue,
    hgt_his_ds_India_u_hypothesis,
) = ca.dim_linregress(
    u_his_ds_India_JJA, hgthis_ds_ver_JJA.sel(level=500.0)
)
(
    u_his_ds_India_u_slope,
    u_his_ds_India_u_intercept,
    u_his_ds_India_u_rvalue,
    u_his_ds_India_u_pvalue,
    u_his_ds_India_u_hypothesis,
) = ca.dim_linregress(
    u_his_ds_India_JJA, uhis_ds_ver_JJA.sel(level=500.0)
)
(
    v_his_ds_India_u_slope,
    v_his_ds_India_u_intercept,
    v_his_ds_India_u_rvalue,
    v_his_ds_India_u_pvalue,
    v_his_ds_India_u_hypothesis,
) = ca.dim_linregress(
    u_his_ds_India_JJA, vhis_ds_ver_JJA.sel(level=500.0)
)
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
wind_his_ds_India_u_mask = ca.wind_check(
    xr.where(u_his_ds_India_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_ds_India_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_ds_India_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_ds_India_u_pvalue <= 0.05, 1.0, 0.0),
)

# %%
#   plot the 850hPa hgt/u/v/ regress onto u_India
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=4, nrows=7, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # ??????????????????
yticks = np.arange(-30, 46, 15)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
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
con = axs[0].contourf(
    hgt_ERA5_India_u_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_India_u_pvalue,
    axs[0],
    n,
    np.where(hgt_ERA5_India_u_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0].quiver(
    u_ERA5_India_u_rvalue[::ski, ::ski],
    v_ERA5_India_u_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0].quiver(
    u_ERA5_India_u_rvalue.where(wind_ERA5_India_u_mask > 0.0)[::ski, ::ski],
    v_ERA5_India_u_rvalue.where(wind_ERA5_India_u_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0].quiverkey(
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
axs[0].format(ltitle="India u index", rtitle="ERA5 500hPa")
# ===================================================
con = axs[1].contourf(
    hgt_his_India_u_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_India_u_pvalue,
    axs[1],
    n,
    np.where(hgt_his_India_u_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1].quiver(
    u_his_India_u_rvalue[::ski, ::ski],
    v_his_India_u_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1].quiver(
    u_his_India_u_rvalue.where(wind_his_India_u_mask > 0.0)[::ski, ::ski],
    v_his_India_u_rvalue.where(wind_his_India_u_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1].quiverkey(
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
axs[1].format(ltitle="India u index", rtitle="ens 500hPa")
# ===================================================
for i, mod in enumerate(models):
    con = axs[i + 2].contourf(
        hgt_his_ds_India_u_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-1.0, 1.1, 0.1),
        zorder=0.8,
    )
    sepl.plt_sig(
        hgt_his_ds_India_u_pvalue.sel(models=mod),
        axs[i + 2],
        n,
        np.where(hgt_his_ds_India_u_pvalue.sel(models=mod)[::n, ::n] <= 0.05),
        "denim",
        3.0,
    )

    axs[i + 2].quiver(
        u_his_ds_India_u_rvalue.sel(models=mod)[::ski, ::ski],
        v_his_ds_India_u_rvalue.sel(models=mod)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=0.17,
        pivot="mid",
        color="grey6",
    )

    m = axs[i + 2].quiver(
        u_his_ds_India_u_rvalue.where(wind_his_ds_India_u_mask > 0.0).sel(models=mod)[
            ::ski, ::ski
        ],
        v_his_ds_India_u_rvalue.where(wind_his_ds_India_u_mask > 0.0).sel(models=mod)[
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

    qk = axs[i + 2].quiverkey(
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
    axs[i + 2].format(
        ltitle="India u index", rtitle="{} 500hPa".format(np.array(models[i]))
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")

# %%
#   calculate the uq/vq/pre climatology
uq_dpg_ERA5_JJA_cli = uq_dpg_ERA5_JJA.mean(dim="time", skipna=True)
vq_dpg_ERA5_JJA_cli = vq_dpg_ERA5_JJA.mean(dim="time", skipna=True)
uq_dpg_his_JJA_cli = uq_dpg_his_JJA.mean(dim="time", skipna=True)
vq_dpg_his_JJA_cli = vq_dpg_his_JJA.mean(dim="time", skipna=True)
uq_dpg_his_ds_JJA_cli = uq_dpg_his_ds_JJA.mean(dim="time", skipna=True)
vq_dpg_his_ds_JJA_cli = vq_dpg_his_ds_JJA.mean(dim="time", skipna=True)

preGPCP_JJA_cli = preGPCP_JJA.mean(dim="time", skipna=True)
prehis_JJA_cli = prehis_JJA.mean(dim="time", skipna=True)
prehis_ds_JJA_cli = prehis_ds_JJA.mean(dim="time", skipna=True)

# %%
#   calculate the divergence of uq and vq
div_uqvq_ERA5_cli = ca.cal_divergence(uq_dpg_ERA5_JJA_cli, vq_dpg_ERA5_JJA_cli)
div_uqvq_his_cli = ca.cal_divergence(uq_dpg_his_JJA_cli, vq_dpg_his_JJA_cli)
div_uqvq_his_ds_cli = ca.cal_divergence(uq_dpg_his_ds_JJA_cli, vq_dpg_his_ds_JJA_cli)

# %%
#   plot the climatology of uq/vq/pre
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=4, nrows=7, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # ??????????????????
yticks = np.arange(-30, 46, 15)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
cmap = pplt.Colormap("Greens", alpha=(0.0, 1.0))

for ax in axs:
    rect = Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1
    )
    ax.add_patch(rect)
# ===================================================
# axs[0].contour(
#     div_uqvq_ERA5_cli,
#     color="bright purple",
#     zorder=0.9,
#     lw=1.0,
#     # levels=np.array([-0.15,0]),
#     vmin=-0.10,
#     vmax=-0.10,
#     linestyle="--",
#     discrete=True,
    
# )
con = axs[0].contourf(
    preGPCP_JJA_cli,
    cmap=cmap,
    cmap_kw={"right": 0.94},
    levels=np.arange(0, 20, 2),
    zorder=0.8,
    extend="both"
)

m = axs[0].quiver(
    uq_dpg_ERA5_JJA_cli[::ski, ::ski],
    vq_dpg_ERA5_JJA_cli[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.8,
    pivot="mid",
    color="black",
)

qk = axs[0].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=5,
    label="5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)

axs[0].format(ltitle="100kg/(m*s)", rtitle="ERA5 & GPCP")

# # ===================================================
con = axs[1].contourf(
    prehis_JJA_cli,
    cmap=cmap,
    cmap_kw={"right": 0.94},
    levels=np.arange(0, 20, 2),
    zorder=0.8,
    extend="both"
)

m = axs[1].quiver(
    uq_dpg_his_JJA_cli[::ski, ::ski],
    vq_dpg_his_JJA_cli[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.8,
    pivot="mid",
    color="black",
)

qk = axs[1].quiverkey(
    m,
    X=1 - w / 2,
    Y=0.7 * h,
    U=5,
    label="5",
    labelpos="S",
    labelsep=0.05,
    fontproperties={"size": 5},
    zorder=3.1,
)

axs[1].format(ltitle="100kg/(m*s)", rtitle="ens")

# # ===================================================
for i, mod in enumerate(models):
    con = axs[i + 2].contourf(
        prehis_ds_JJA_cli.sel(models=mod),
        cmap=cmap,
        cmap_kw={"right": 0.94},
        levels=np.arange(0, 20, 2),
        zorder=0.8,
        extend="both"
    )
    m = axs[i + 2].quiver(
        uq_dpg_his_ds_JJA_cli.sel(models=mod)[::ski, ::ski],
        vq_dpg_his_ds_JJA_cli.sel(models=mod)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=0.8,
        pivot="mid",
        color="black",
    )

    qk = axs[i + 2].quiverkey(
        m,
        X=1 - w / 2,
        Y=0.7 * h,
        U=5,
        label="5",
        labelpos="S",
        labelsep=0.05,
        fontproperties={"size": 5},
        zorder=3.1,
    )
    axs[i + 2].format(ltitle="100kg/(m*s)", rtitle="{}".format(str(mod.data)))
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="precip & Uq climatology")
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
) = ca.dim_linregress(
    uq_dpg_his_ds_India_JJA, hgthis_ds_ver_JJA.sel(level=200.0)
)
(
    u_his_ds_India_uq_slope,
    u_his_ds_India_uq_intercept,
    u_his_ds_India_uq_rvalue,
    u_his_ds_India_uq_pvalue,
    u_his_ds_India_uq_hypothesis,
) = ca.dim_linregress(
    uq_dpg_his_ds_India_JJA, uhis_ds_ver_JJA.sel(level=200.0)
)
(
    v_his_ds_India_uq_slope,
    v_his_ds_India_uq_intercept,
    v_his_ds_India_uq_rvalue,
    v_his_ds_India_uq_pvalue,
    v_his_ds_India_uq_hypothesis,
) = ca.dim_linregress(
    uq_dpg_his_ds_India_JJA, vhis_ds_ver_JJA.sel(level=200.0)
)
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

# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=4, nrows=7, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # ??????????????????
yticks = np.arange(-30, 46, 15)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
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
con = axs[0].contourf(
    hgt_ERA5_India_uq_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_India_uq_pvalue,
    axs[0],
    n,
    np.where(hgt_ERA5_India_uq_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0].quiver(
    u_ERA5_India_uq_rvalue[::ski, ::ski],
    v_ERA5_India_uq_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0].quiver(
    u_ERA5_India_uq_rvalue.where(wind_ERA5_India_uq_mask > 0.0)[::ski, ::ski],
    v_ERA5_India_uq_rvalue.where(wind_ERA5_India_uq_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0].quiverkey(
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
axs[0].format(ltitle="India uq index", rtitle="ERA5 200hPa")
# ===================================================
con = axs[1].contourf(
    hgt_his_India_uq_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_India_uq_pvalue,
    axs[1],
    n,
    np.where(hgt_his_India_uq_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1].quiver(
    u_his_India_uq_rvalue[::ski, ::ski],
    v_his_India_uq_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1].quiver(
    u_his_India_uq_rvalue.where(wind_his_India_uq_mask > 0.0)[::ski, ::ski],
    v_his_India_uq_rvalue.where(wind_his_India_uq_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1].quiverkey(
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
axs[1].format(ltitle="India uq index", rtitle="ens 200hPa")
# ===================================================
for i, mod in enumerate(models):
    con = axs[i + 2].contourf(
        hgt_his_ds_India_uq_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-1.0, 1.1, 0.1),
        zorder=0.8,
    )
    sepl.plt_sig(
        hgt_his_ds_India_uq_pvalue.sel(models=mod),
        axs[i + 2],
        n,
        np.where(hgt_his_ds_India_uq_pvalue.sel(models=mod)[::n, ::n] <= 0.05),
        "denim",
        3.0,
    )

    axs[i + 2].quiver(
        u_his_ds_India_uq_rvalue.sel(models=mod)[::ski, ::ski],
        v_his_ds_India_uq_rvalue.sel(models=mod)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=0.17,
        pivot="mid",
        color="grey6",
    )

    m = axs[i + 2].quiver(
        u_his_ds_India_uq_rvalue.where(wind_his_ds_India_uq_mask > 0.0).sel(models=mod)[
            ::ski, ::ski
        ],
        v_his_ds_India_uq_rvalue.where(wind_his_ds_India_uq_mask > 0.0).sel(models=mod)[
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

    qk = axs[i + 2].quiverkey(
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
    axs[i + 2].format(
        ltitle="India uq index", rtitle="{} 200hPa".format(np.array(models[i]))
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
# %%
#   calculate uq in NCR
vq_dpg_ERA5_NCR_JJA = ca.cal_lat_weighted_mean(
    vq_dpg_ERA5_JJA.loc[:, 25.0:37.5, 110.0:125.0]
).mean(dim="lon", skipna=True)
vq_dpg_ERA5_NCR_JJA = ca.detrend_dim(
    vq_dpg_ERA5_NCR_JJA, "time", deg=1, demean=False
)
vq_dpg_his_NCR_JJA = ca.cal_lat_weighted_mean(
    vq_dpg_his_JJA.loc[:, 25.0:37.5, 110.0:125.0]
).mean(dim="lon", skipna=True)
vq_dpg_his_NCR_JJA = ca.detrend_dim(vq_dpg_his_NCR_JJA, "time", deg=1, demean=False)

vq_dpg_his_ds_NCR_JJA = ca.cal_lat_weighted_mean(
    vq_dpg_his_ds_JJA.loc[:, :, 25.0:37.5, 110.0:125.0]
).mean(dim="lon", skipna=True)
vq_dpg_his_ds_NCR_JJA = ca.detrend_dim(
    vq_dpg_his_ds_NCR_JJA, "time", deg=1, demean=False
)

v_ERA5_NCR_JJA = ca.cal_lat_weighted_mean(
    vERA5_ver_JJA.sel(level=850.0).loc[:, 25.0:37.5, 110.0:125.0]
).mean(dim="lon", skipna=True)
v_ERA5_NCR_JJA = ca.detrend_dim(
    v_ERA5_NCR_JJA, "time", deg=1, demean=False
)
v_his_NCR_JJA = ca.cal_lat_weighted_mean(
    vhis_ver_JJA.sel(level=850.0).loc[:, 25.0:37.5, 110.0:125.0]
).mean(dim="lon", skipna=True)
v_his_NCR_JJA = ca.detrend_dim(v_his_NCR_JJA, "time", deg=1, demean=False)

v_his_ds_NCR_JJA = ca.cal_lat_weighted_mean(
    vhis_ds_ver_JJA.sel(level=850.0).loc[:, :, 25.0:37.5, 110.0:125.0]
).mean(dim="lon", skipna=True)
v_his_ds_NCR_JJA = ca.detrend_dim(
    v_his_ds_NCR_JJA, "time", deg=1, demean=False
)
# %%
#   calculate the regression of vq
(
    hgt_ERA5_NCR_vq_slope,
    hgt_ERA5_NCR_vq_intercept,
    hgt_ERA5_NCR_vq_rvalue,
    hgt_ERA5_NCR_vq_pvalue,
    hgt_ERA5_NCR_vq_hypothesis,
) = ca.dim_linregress(vq_dpg_ERA5_NCR_JJA, hgtERA5_ver_JJA.sel(level=200.0))
(
    u_ERA5_NCR_vq_slope,
    u_ERA5_NCR_vq_intercept,
    u_ERA5_NCR_vq_rvalue,
    u_ERA5_NCR_vq_pvalue,
    u_ERA5_NCR_vq_hypothesis,
) = ca.dim_linregress(vq_dpg_ERA5_NCR_JJA, uERA5_ver_JJA.sel(level=200.0))
(
    v_ERA5_NCR_vq_slope,
    v_ERA5_NCR_vq_intercept,
    v_ERA5_NCR_vq_rvalue,
    v_ERA5_NCR_vq_pvalue,
    v_ERA5_NCR_vq_hypothesis,
) = ca.dim_linregress(vq_dpg_ERA5_NCR_JJA, vERA5_ver_JJA.sel(level=200.0))

(
    hgt_his_NCR_vq_slope,
    hgt_his_NCR_vq_intercept,
    hgt_his_NCR_vq_rvalue,
    hgt_his_NCR_vq_pvalue,
    hgt_his_NCR_vq_hypothesis,
) = ca.dim_linregress(vq_dpg_his_NCR_JJA, hgthis_ver_JJA.sel(level=200.0))
(
    u_his_NCR_vq_slope,
    u_his_NCR_vq_intercept,
    u_his_NCR_vq_rvalue,
    u_his_NCR_vq_pvalue,
    u_his_NCR_vq_hypothesis,
) = ca.dim_linregress(vq_dpg_his_NCR_JJA, uhis_ver_JJA.sel(level=200.0))
(
    v_his_NCR_vq_slope,
    v_his_NCR_vq_intercept,
    v_his_NCR_vq_rvalue,
    v_his_NCR_vq_pvalue,
    v_his_NCR_vq_hypothesis,
) = ca.dim_linregress(vq_dpg_his_NCR_JJA, vhis_ver_JJA.sel(level=200.0))

(
    hgt_his_ds_NCR_vq_slope,
    hgt_his_ds_NCR_vq_intercept,
    hgt_his_ds_NCR_vq_rvalue,
    hgt_his_ds_NCR_vq_pvalue,
    hgt_his_ds_NCR_vq_hypothesis,
) = ca.dim_linregress(
    vq_dpg_his_ds_NCR_JJA, hgthis_ds_ver_JJA.sel(level=200.0)
)
(
    u_his_ds_NCR_vq_slope,
    u_his_ds_NCR_vq_intercept,
    u_his_ds_NCR_vq_rvalue,
    u_his_ds_NCR_vq_pvalue,
    u_his_ds_NCR_vq_hypothesis,
) = ca.dim_linregress(
    vq_dpg_his_ds_NCR_JJA, uhis_ds_ver_JJA.sel(level=200.0)
)
(
    v_his_ds_NCR_vq_slope,
    v_his_ds_NCR_vq_intercept,
    v_his_ds_NCR_vq_rvalue,
    v_his_ds_NCR_vq_pvalue,
    v_his_ds_NCR_vq_hypothesis,
) = ca.dim_linregress(
    vq_dpg_his_ds_NCR_JJA, vhis_ds_ver_JJA.sel(level=200.0)
)
wind_ERA5_NCR_vq_mask = ca.wind_check(
    xr.where(u_ERA5_NCR_vq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_NCR_vq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_ERA5_NCR_vq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_NCR_vq_pvalue <= 0.05, 1.0, 0.0),
)
wind_his_NCR_vq_mask = ca.wind_check(
    xr.where(u_his_NCR_vq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_NCR_vq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_NCR_vq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_NCR_vq_pvalue <= 0.05, 1.0, 0.0),
)
wind_his_ds_NCR_vq_mask = ca.wind_check(
    xr.where(u_his_ds_NCR_vq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_ds_NCR_vq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_ds_NCR_vq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_ds_NCR_vq_pvalue <= 0.05, 1.0, 0.0),
)
# %%
#   plot the regression on NCR vq
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=4, nrows=7, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # ??????????????????
yticks = np.arange(-30, 46, 15)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
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
    x0 = 110.0
    y0 = 25.0
    width = 15
    height = 12.5
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    hgt_ERA5_NCR_vq_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_NCR_vq_pvalue,
    axs[0],
    n,
    np.where(hgt_ERA5_NCR_vq_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0].quiver(
    u_ERA5_NCR_vq_rvalue[::ski, ::ski],
    v_ERA5_NCR_vq_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0].quiver(
    u_ERA5_NCR_vq_rvalue.where(wind_ERA5_NCR_vq_mask > 0.0)[::ski, ::ski],
    v_ERA5_NCR_vq_rvalue.where(wind_ERA5_NCR_vq_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0].quiverkey(
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
axs[0].format(ltitle="NCR vq index", rtitle="ERA5 200hPa")
# ===================================================
con = axs[1].contourf(
    hgt_his_NCR_vq_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_NCR_vq_pvalue,
    axs[1],
    n,
    np.where(hgt_his_NCR_vq_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1].quiver(
    u_his_NCR_vq_rvalue[::ski, ::ski],
    v_his_NCR_vq_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1].quiver(
    u_his_NCR_vq_rvalue.where(wind_his_NCR_vq_mask > 0.0)[::ski, ::ski],
    v_his_NCR_vq_rvalue.where(wind_his_NCR_vq_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1].quiverkey(
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
axs[1].format(ltitle="NCR vq index", rtitle="ens 200hPa")
# ===================================================
for i, mod in enumerate(models):
    con = axs[i + 2].contourf(
        hgt_his_ds_NCR_vq_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-1.0, 1.1, 0.1),
        zorder=0.8,
    )
    sepl.plt_sig(
        hgt_his_ds_NCR_vq_pvalue.sel(models=mod),
        axs[i + 2],
        n,
        np.where(hgt_his_ds_NCR_vq_pvalue.sel(models=mod)[::n, ::n] <= 0.05),
        "denim",
        3.0,
    )

    axs[i + 2].quiver(
        u_his_ds_NCR_vq_rvalue.sel(models=mod)[::ski, ::ski],
        v_his_ds_NCR_vq_rvalue.sel(models=mod)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=0.17,
        pivot="mid",
        color="grey6",
    )

    m = axs[i + 2].quiver(
        u_his_ds_NCR_vq_rvalue.where(wind_his_ds_NCR_vq_mask > 0.0).sel(models=mod)[
            ::ski, ::ski
        ],
        v_his_ds_NCR_vq_rvalue.where(wind_his_ds_NCR_vq_mask > 0.0).sel(models=mod)[
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

    qk = axs[i + 2].quiverkey(
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
    axs[i + 2].format(
        ltitle="NCR vq index", rtitle="{} 200hPa".format(np.array(models[i]))
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
#   calculate the regression of v
(
    hgt_ERA5_NCR_v_slope,
    hgt_ERA5_NCR_v_intercept,
    hgt_ERA5_NCR_v_rvalue,
    hgt_ERA5_NCR_v_pvalue,
    hgt_ERA5_NCR_v_hypothesis,
) = ca.dim_linregress(v_ERA5_NCR_JJA, hgtERA5_ver_JJA.sel(level=850.0))
(
    u_ERA5_NCR_v_slope,
    u_ERA5_NCR_v_intercept,
    u_ERA5_NCR_v_rvalue,
    u_ERA5_NCR_v_pvalue,
    u_ERA5_NCR_v_hypothesis,
) = ca.dim_linregress(v_ERA5_NCR_JJA, uERA5_ver_JJA.sel(level=850.0))
(
    v_ERA5_NCR_v_slope,
    v_ERA5_NCR_v_intercept,
    v_ERA5_NCR_v_rvalue,
    v_ERA5_NCR_v_pvalue,
    v_ERA5_NCR_v_hypothesis,
) = ca.dim_linregress(v_ERA5_NCR_JJA, vERA5_ver_JJA.sel(level=850.0))

(
    hgt_his_NCR_v_slope,
    hgt_his_NCR_v_intercept,
    hgt_his_NCR_v_rvalue,
    hgt_his_NCR_v_pvalue,
    hgt_his_NCR_v_hypothesis,
) = ca.dim_linregress(v_his_NCR_JJA, hgthis_ver_JJA.sel(level=850.0))
(
    u_his_NCR_v_slope,
    u_his_NCR_v_intercept,
    u_his_NCR_v_rvalue,
    u_his_NCR_v_pvalue,
    u_his_NCR_v_hypothesis,
) = ca.dim_linregress(v_his_NCR_JJA, uhis_ver_JJA.sel(level=850.0))
(
    v_his_NCR_v_slope,
    v_his_NCR_v_intercept,
    v_his_NCR_v_rvalue,
    v_his_NCR_v_pvalue,
    v_his_NCR_v_hypothesis,
) = ca.dim_linregress(v_his_NCR_JJA, vhis_ver_JJA.sel(level=850.0))

(
    hgt_his_ds_NCR_v_slope,
    hgt_his_ds_NCR_v_intercept,
    hgt_his_ds_NCR_v_rvalue,
    hgt_his_ds_NCR_v_pvalue,
    hgt_his_ds_NCR_v_hypothesis,
) = ca.dim_linregress(
    v_his_ds_NCR_JJA, hgthis_ds_ver_JJA.sel(level=850.0)
)
(
    u_his_ds_NCR_v_slope,
    u_his_ds_NCR_v_intercept,
    u_his_ds_NCR_v_rvalue,
    u_his_ds_NCR_v_pvalue,
    u_his_ds_NCR_v_hypothesis,
) = ca.dim_linregress(
    v_his_ds_NCR_JJA, uhis_ds_ver_JJA.sel(level=850.0)
)
(
    v_his_ds_NCR_v_slope,
    v_his_ds_NCR_v_intercept,
    v_his_ds_NCR_v_rvalue,
    v_his_ds_NCR_v_pvalue,
    v_his_ds_NCR_v_hypothesis,
) = ca.dim_linregress(
    v_his_ds_NCR_JJA, vhis_ds_ver_JJA.sel(level=850.0)
)
wind_ERA5_NCR_v_mask = ca.wind_check(
    xr.where(u_ERA5_NCR_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_NCR_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_ERA5_NCR_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_ERA5_NCR_v_pvalue <= 0.05, 1.0, 0.0),
)
wind_his_NCR_v_mask = ca.wind_check(
    xr.where(u_his_NCR_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_NCR_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_NCR_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_NCR_v_pvalue <= 0.05, 1.0, 0.0),
)
wind_his_ds_NCR_v_mask = ca.wind_check(
    xr.where(u_his_ds_NCR_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_ds_NCR_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(u_his_ds_NCR_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(v_his_ds_NCR_v_pvalue <= 0.05, 1.0, 0.0),
)
# %%
#   plot the regression of NCR v
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=4, nrows=7, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # ??????????????????
yticks = np.arange(-30, 46, 15)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
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
    x0 = 110.0
    y0 = 25.0
    width = 15
    height = 12.5
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    hgt_ERA5_NCR_v_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_ERA5_NCR_v_pvalue,
    axs[0],
    n,
    np.where(hgt_ERA5_NCR_v_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0].quiver(
    u_ERA5_NCR_v_rvalue[::ski, ::ski],
    v_ERA5_NCR_v_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0].quiver(
    u_ERA5_NCR_v_rvalue.where(wind_ERA5_NCR_v_mask > 0.0)[::ski, ::ski],
    v_ERA5_NCR_v_rvalue.where(wind_ERA5_NCR_v_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0].quiverkey(
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
axs[0].format(ltitle="NCR v index", rtitle="ERA5 850hPa")
# ===================================================
con = axs[1].contourf(
    hgt_his_NCR_v_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    hgt_his_NCR_v_pvalue,
    axs[1],
    n,
    np.where(hgt_his_NCR_v_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1].quiver(
    u_his_NCR_v_rvalue[::ski, ::ski],
    v_his_NCR_v_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1].quiver(
    u_his_NCR_v_rvalue.where(wind_his_NCR_v_mask > 0.0)[::ski, ::ski],
    v_his_NCR_v_rvalue.where(wind_his_NCR_v_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1].quiverkey(
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
axs[1].format(ltitle="NCR v index", rtitle="ens 850hPa")
# ===================================================
for i, mod in enumerate(models):
    con = axs[i + 2].contourf(
        hgt_his_ds_NCR_v_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-1.0, 1.1, 0.1),
        zorder=0.8,
    )
    sepl.plt_sig(
        hgt_his_ds_NCR_v_pvalue.sel(models=mod),
        axs[i + 2],
        n,
        np.where(hgt_his_ds_NCR_v_pvalue.sel(models=mod)[::n, ::n] <= 0.05),
        "denim",
        3.0,
    )

    axs[i + 2].quiver(
        u_his_ds_NCR_v_rvalue.sel(models=mod)[::ski, ::ski],
        v_his_ds_NCR_v_rvalue.sel(models=mod)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=0.17,
        pivot="mid",
        color="grey6",
    )

    m = axs[i + 2].quiver(
        u_his_ds_NCR_v_rvalue.where(wind_his_ds_NCR_v_mask > 0.0).sel(models=mod)[
            ::ski, ::ski
        ],
        v_his_ds_NCR_v_rvalue.where(wind_his_ds_NCR_v_mask > 0.0).sel(models=mod)[
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

    qk = axs[i + 2].quiverkey(
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
    axs[i + 2].format(
        ltitle="NCR v index", rtitle="{} 850hPa".format(np.array(models[i]))
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
#   calculate model ranking for both India and NCR

levels = [850.0, 500.0, 200.0]
pcc_models = np.zeros(26)
for lev in levels:
    (
        hgt_ERA5_India_uq_slope,
        hgt_ERA5_India_uq_intercept,
        hgt_ERA5_India_uq_rvalue,
        hgt_ERA5_India_uq_pvalue,
        hgt_ERA5_India_uq_hypothesis,
    ) = ca.dim_linregress(uq_dpg_ERA5_India_JJA, hgtERA5_ver_JJA.sel(level=lev))
    (
        u_ERA5_India_uq_slope,
        u_ERA5_India_uq_intercept,
        u_ERA5_India_uq_rvalue,
        u_ERA5_India_uq_pvalue,
        u_ERA5_India_uq_hypothesis,
    ) = ca.dim_linregress(uq_dpg_ERA5_India_JJA, uERA5_ver_JJA.sel(level=lev))
    (
        v_ERA5_India_uq_slope,
        v_ERA5_India_uq_intercept,
        v_ERA5_India_uq_rvalue,
        v_ERA5_India_uq_pvalue,
        v_ERA5_India_uq_hypothesis,
    ) = ca.dim_linregress(uq_dpg_ERA5_India_JJA, vERA5_ver_JJA.sel(level=lev))

    (
        hgt_his_ds_India_uq_slope,
        hgt_his_ds_India_uq_intercept,
        hgt_his_ds_India_uq_rvalue,
        hgt_his_ds_India_uq_pvalue,
        hgt_his_ds_India_uq_hypothesis,
    ) = ca.dim_linregress(
        uq_dpg_his_ds_India_JJA, hgthis_ds_ver_JJA.sel(level=lev)
    )
    (
        u_his_ds_India_uq_slope,
        u_his_ds_India_uq_intercept,
        u_his_ds_India_uq_rvalue,
        u_his_ds_India_uq_pvalue,
        u_his_ds_India_uq_hypothesis,
    ) = ca.dim_linregress(
        uq_dpg_his_ds_India_JJA, uhis_ds_ver_JJA.sel(level=lev)
    )
    (
        v_his_ds_India_uq_slope,
        v_his_ds_India_uq_intercept,
        v_his_ds_India_uq_rvalue,
        v_his_ds_India_uq_pvalue,
        v_his_ds_India_uq_hypothesis,
    ) = ca.dim_linregress(
        uq_dpg_his_ds_India_JJA, vhis_ds_ver_JJA.sel(level=lev)
    )
    (
        hgt_ERA5_NCR_vq_slope,
        hgt_ERA5_NCR_vq_intercept,
        hgt_ERA5_NCR_vq_rvalue,
        hgt_ERA5_NCR_vq_pvalue,
        hgt_ERA5_NCR_vq_hypothesis,
    ) = ca.dim_linregress(vq_dpg_ERA5_NCR_JJA, hgtERA5_ver_JJA.sel(level=lev))
    (
        u_ERA5_NCR_vq_slope,
        u_ERA5_NCR_vq_intercept,
        u_ERA5_NCR_vq_rvalue,
        u_ERA5_NCR_vq_pvalue,
        u_ERA5_NCR_vq_hypothesis,
    ) = ca.dim_linregress(vq_dpg_ERA5_NCR_JJA, uERA5_ver_JJA.sel(level=lev))
    (
        v_ERA5_NCR_vq_slope,
        v_ERA5_NCR_vq_intercept,
        v_ERA5_NCR_vq_rvalue,
        v_ERA5_NCR_vq_pvalue,
        v_ERA5_NCR_vq_hypothesis,
    ) = ca.dim_linregress(vq_dpg_ERA5_NCR_JJA, vERA5_ver_JJA.sel(level=lev))

    (
        hgt_his_ds_NCR_vq_slope,
        hgt_his_ds_NCR_vq_intercept,
        hgt_his_ds_NCR_vq_rvalue,
        hgt_his_ds_NCR_vq_pvalue,
        hgt_his_ds_NCR_vq_hypothesis,
    ) = ca.dim_linregress(
        vq_dpg_his_ds_NCR_JJA, hgthis_ds_ver_JJA.sel(level=lev)
    )
    (
        u_his_ds_NCR_vq_slope,
        u_his_ds_NCR_vq_intercept,
        u_his_ds_NCR_vq_rvalue,
        u_his_ds_NCR_vq_pvalue,
        u_his_ds_NCR_vq_hypothesis,
    ) = ca.dim_linregress(
        vq_dpg_his_ds_NCR_JJA, uhis_ds_ver_JJA.sel(level=lev)
    )
    (
        v_his_ds_NCR_vq_slope,
        v_his_ds_NCR_vq_intercept,
        v_his_ds_NCR_vq_rvalue,
        v_his_ds_NCR_vq_pvalue,
        v_his_ds_NCR_vq_hypothesis,
    ) = ca.dim_linregress(
        vq_dpg_his_ds_NCR_JJA, vhis_ds_ver_JJA.sel(level=lev)
    )
    for i, mod in enumerate(models):
        pcc_models[i] += ca.cal_pcc(hgt_ERA5_India_uq_rvalue.loc[-30:55, 30:180], hgt_his_ds_India_uq_rvalue.sel(models=mod).loc[-30:55, 30:180])
        pcc_models[i] += ca.cal_pcc(u_ERA5_India_uq_rvalue.loc[-30:55, 30:180], u_his_ds_India_uq_rvalue.sel(models=mod).loc[-30:55, 30:180])
        pcc_models[i] += ca.cal_pcc(v_ERA5_India_uq_rvalue.loc[-30:55, 30:180], v_his_ds_India_uq_rvalue.sel(models=mod).loc[-30:55, 30:180])
        pcc_models[i] += ca.cal_pcc(hgt_ERA5_NCR_vq_rvalue.loc[-30:55, 30:180], hgt_his_ds_NCR_vq_rvalue.sel(models=mod).loc[-30:55, 30:180])
        pcc_models[i] += ca.cal_pcc(u_ERA5_NCR_vq_rvalue.loc[-30:55, 30:180], u_his_ds_NCR_vq_rvalue.sel(models=mod).loc[-30:55, 30:180])
        pcc_models[i] += ca.cal_pcc(v_ERA5_NCR_vq_rvalue.loc[-30:55, 30:180], v_his_ds_NCR_vq_rvalue.sel(models=mod).loc[-30:55, 30:180])
print(pcc_models)
# %%
pcc_list = []
for i,mod in enumerate(models):
    pcc_list.append({"models": str(mod.data), "pcc": pcc_models[i]})
print(sorted(pcc_list, key=lambda x : x["pcc"]))
# %%
#   calculate the uq and vq regress onto IndR
preCRU_India_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.loc[:, 8:28, 70:86]).mean(dim="lon", skipna=True)

prehis_India_JJA = ca.cal_lat_weighted_mean(prehis_JJA.loc[:, 8:28, 70:86]).mean(dim="lon", skipna=True)

prehis_ds_India_JJA = ca.cal_lat_weighted_mean(prehis_ds_JJA.loc[:, :, 8:28, 70:86]).mean(dim="lon", skipna=True)

preCRU_India_JJA.coords["time"] = uq_dpg_ERA5_JJA.coords["time"]

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
    pre_his_India_uq_slope,
    pre_his_India_uq_intercept,
    pre_his_India_uq_rvalue,
    pre_his_India_uq_pvalue,
    pre_his_India_uq_hypothesis,
) = ca.dim_linregress(prehis_India_JJA, uq_dpg_his_JJA)

(
    pre_his_India_vq_slope,
    pre_his_India_vq_intercept,
    pre_his_India_vq_rvalue,
    pre_his_India_vq_pvalue,
    pre_his_India_vq_hypothesis,
) = ca.dim_linregress(prehis_India_JJA, vq_dpg_his_JJA)

(
    pre_his_ds_India_uq_slope,
    pre_his_ds_India_uq_intercept,
    pre_his_ds_India_uq_rvalue,
    pre_his_ds_India_uq_pvalue,
    pre_his_ds_India_uq_hypothesis,
) = ca.dim_linregress(prehis_ds_India_JJA, uq_dpg_his_ds_JJA)

(
    pre_his_ds_India_vq_slope,
    pre_his_ds_India_vq_intercept,
    pre_his_ds_India_vq_rvalue,
    pre_his_ds_India_vq_pvalue,
    pre_his_ds_India_vq_hypothesis,
) = ca.dim_linregress(prehis_ds_India_JJA, vq_dpg_his_ds_JJA)
# %%
pre_CRU_India_uqvq_mask = ca.wind_check(
    xr.where(pre_CRU_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(pre_CRU_India_vq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(pre_CRU_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(pre_CRU_India_vq_pvalue <= 0.05, 1.0, 0.0),
)
pre_his_India_uqvq_mask = ca.wind_check(
    xr.where(pre_his_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(pre_his_India_vq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(pre_his_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(pre_his_India_vq_pvalue <= 0.05, 1.0, 0.0),
)
pre_his_ds_India_uqvq_mask = ca.wind_check(
    xr.where(pre_his_ds_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(pre_his_ds_India_vq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(pre_his_ds_India_uq_pvalue <= 0.05, 1.0, 0.0),
    xr.where(pre_his_ds_India_vq_pvalue <= 0.05, 1.0, 0.0),
)

# %%
#   calculate the divergence of uq and vq
div_uqvq_ERA5 = ca.cal_divergence(uq_dpg_ERA5_JJA, vq_dpg_ERA5_JJA)
div_uqvq_his = ca.cal_divergence(uq_dpg_his_JJA, vq_dpg_his_JJA)
div_uqvq_his_ds = ca.cal_divergence(uq_dpg_his_ds_JJA, vq_dpg_his_ds_JJA)
# %%
#   calculate the divergence of uq and vq regress onto IndR
(
    pre_CRU_India_divuqvq_slope,
    pre_CRU_India_divuqvq_intercept,
    pre_CRU_India_divuqvq_rvalue,
    pre_CRU_India_divuqvq_pvalue,
    pre_CRU_India_divuqvq_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA, div_uqvq_ERA5)

(
    pre_his_India_divuqvq_slope,
    pre_his_India_divuqvq_intercept,
    pre_his_India_divuqvq_rvalue,
    pre_his_India_divuqvq_pvalue,
    pre_his_India_divuqvq_hypothesis,
) = ca.dim_linregress(prehis_India_JJA, div_uqvq_his)

(
    pre_his_ds_India_divuqvq_slope,
    pre_his_ds_India_divuqvq_intercept,
    pre_his_ds_India_divuqvq_rvalue,
    pre_his_ds_India_divuqvq_pvalue,
    pre_his_ds_India_divuqvq_hypothesis,
) = ca.dim_linregress(prehis_ds_India_JJA, div_uqvq_his_ds)
# %%
#   plot the div uq and vq regress onto IndR
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # ?????????????????????????????????
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(ncols=4, nrows=7, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # ??????????????????
yticks = np.arange(-30, 46, 15)  # ??????????????????
# ??????????????????????????????extents?????????????????????????????????????????????????????????????????????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????extents???????????????????????????
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
    #   Indian area
    x0 = 70
    y0 = 8.0
    width = 16
    height = 20.0
    patches(ax, x0 - cl, y0, width, height, proj)
    #   NCR area
    x0 = 108
    y0 = 36.0
    width = 10.0
    height = 6.0
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_CRU_India_divuqvq_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    pre_CRU_India_divuqvq_pvalue,
    axs[0],
    n,
    np.where(pre_CRU_India_divuqvq_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[0].quiver(
    pre_CRU_India_uq_rvalue[::ski, ::ski],
    pre_CRU_India_vq_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[0].quiver(
    pre_CRU_India_uq_rvalue.where(pre_CRU_India_uqvq_mask > 0.0)[::ski, ::ski],
    pre_CRU_India_vq_rvalue.where(pre_CRU_India_uqvq_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[0].quiverkey(
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
axs[0].format(ltitle="div&uq/vq reg IndR", rtitle="CRU TS4.01")

# ===================================================
con = axs[1].contourf(
    pre_his_India_divuqvq_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
)
sepl.plt_sig(
    pre_his_India_divuqvq_pvalue,
    axs[1],
    n,
    np.where(pre_his_India_divuqvq_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)

axs[1].quiver(
    pre_his_India_uq_rvalue[::ski, ::ski],
    pre_his_India_vq_rvalue[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="grey6",
)

m = axs[1].quiver(
    pre_his_India_uq_rvalue.where(pre_his_India_uqvq_mask > 0.0)[::ski, ::ski],
    pre_his_India_vq_rvalue.where(pre_his_India_uqvq_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.17,
    pivot="mid",
    color="black",
)

qk = axs[1].quiverkey(
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
axs[1].format(ltitle="div&uq/vq reg IndR", rtitle="ensmean")
# ===================================================
for i, mod in enumerate(models):
    con = axs[i + 2].contourf(
        pre_his_ds_India_divuqvq_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-1.0, 1.1, 0.1),
        zorder=0.8,
    )
    sepl.plt_sig(
        pre_his_ds_India_divuqvq_pvalue.sel(models=mod),
        axs[i + 2],
        n,
        np.where(pre_his_ds_India_divuqvq_pvalue.sel(models=mod)[::n, ::n] <= 0.05),
        "denim",
        3.0,
    )

    axs[i + 2].quiver(
        pre_his_ds_India_uq_rvalue.sel(models=mod)[::ski, ::ski],
        pre_his_ds_India_vq_rvalue.sel(models=mod)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=0.17,
        pivot="mid",
        color="grey6",
    )

    m = axs[i + 2].quiver(
        pre_his_ds_India_uq_rvalue.where(pre_his_ds_India_uqvq_mask > 0.0).sel(models=mod)[
            ::ski, ::ski
        ],
        pre_his_ds_India_vq_rvalue.where(pre_his_ds_India_uqvq_mask > 0.0).sel(models=mod)[
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

    qk = axs[i + 2].quiverkey(
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
    axs[i + 2].format(
        ltitle="div&uq/vq reg IndR", rtitle="{}".format(np.array(models[i]))
    )
# %%
#   calculate model ranking by calculate the performance of models in modeling the divergence/uq/vq reg IndR

pcc_models = np.zeros(26)

levels = [850.0, 500.0, 200.0]
for lev in levels:
    (
        hgt_ERA5_India_uq_slope,
        hgt_ERA5_India_uq_intercept,
        hgt_ERA5_India_uq_rvalue,
        hgt_ERA5_India_uq_pvalue,
        hgt_ERA5_India_uq_hypothesis,
    ) = ca.dim_linregress(uq_dpg_ERA5_India_JJA, hgtERA5_ver_JJA.sel(level=lev))
    (
        u_ERA5_India_uq_slope,
        u_ERA5_India_uq_intercept,
        u_ERA5_India_uq_rvalue,
        u_ERA5_India_uq_pvalue,
        u_ERA5_India_uq_hypothesis,
    ) = ca.dim_linregress(uq_dpg_ERA5_India_JJA, uERA5_ver_JJA.sel(level=lev))
    (
        v_ERA5_India_uq_slope,
        v_ERA5_India_uq_intercept,
        v_ERA5_India_uq_rvalue,
        v_ERA5_India_uq_pvalue,
        v_ERA5_India_uq_hypothesis,
    ) = ca.dim_linregress(uq_dpg_ERA5_India_JJA, vERA5_ver_JJA.sel(level=lev))

    (
        hgt_his_ds_India_uq_slope,
        hgt_his_ds_India_uq_intercept,
        hgt_his_ds_India_uq_rvalue,
        hgt_his_ds_India_uq_pvalue,
        hgt_his_ds_India_uq_hypothesis,
    ) = ca.dim_linregress(
        uq_dpg_his_ds_India_JJA, hgthis_ds_ver_JJA.sel(level=lev)
    )
    (
        u_his_ds_India_uq_slope,
        u_his_ds_India_uq_intercept,
        u_his_ds_India_uq_rvalue,
        u_his_ds_India_uq_pvalue,
        u_his_ds_India_uq_hypothesis,
    ) = ca.dim_linregress(
        uq_dpg_his_ds_India_JJA, uhis_ds_ver_JJA.sel(level=lev)
    )
    (
        v_his_ds_India_uq_slope,
        v_his_ds_India_uq_intercept,
        v_his_ds_India_uq_rvalue,
        v_his_ds_India_uq_pvalue,
        v_his_ds_India_uq_hypothesis,
    ) = ca.dim_linregress(
        uq_dpg_his_ds_India_JJA, vhis_ds_ver_JJA.sel(level=lev)
    )
    for i, mod in enumerate(models):
        pcc_models[i] += ca.cal_pcc(hgt_ERA5_India_uq_rvalue.loc[0:55, 90:150], hgt_his_ds_India_uq_rvalue.sel(models=mod).loc[0:55, 90:150])
        pcc_models[i] += ca.cal_pcc(u_ERA5_India_uq_rvalue.loc[0:55, 90:150], u_his_ds_India_uq_rvalue.sel(models=mod).loc[0:55, 90:150])
        pcc_models[i] += ca.cal_pcc(v_ERA5_India_uq_rvalue.loc[0:55, 90:150], v_his_ds_India_uq_rvalue.sel(models=mod).loc[0:55, 90:150])
for i, mod in enumerate(models):
    pcc_models[i] += ca.cal_pcc(pre_CRU_India_divuqvq_rvalue.loc[0:55, 90:150], pre_his_ds_India_divuqvq_rvalue.sel(models=mod).loc[0:55, 90:150])
    pcc_models[i] += ca.cal_pcc(pre_CRU_India_uq_rvalue.loc[0:55, 90:150], pre_his_ds_India_uq_rvalue.sel(models=mod).loc[0:55, 90:150])
    pcc_models[i] += ca.cal_pcc(pre_CRU_India_vq_rvalue.loc[0:55, 90:150], pre_his_ds_India_vq_rvalue.sel(models=mod).loc[0:55, 90:150])
pcc_models = pcc_models/12.0

print(pcc_models)
# %%
pcc_list = []
for i,mod in enumerate(models):
    pcc_list.append({"models": str(mod.data), "pcc": pcc_models[i]})
print(sorted(pcc_list, key=lambda x : x["pcc"]))
# %%
pcc_models = np.zeros(26)
for i, mod in enumerate(models):
    pcc_models[i] += ca.cal_pcc(pre_CRU_India_divuqvq_rvalue.loc[0:55, 90:150], pre_his_ds_India_divuqvq_rvalue.sel(models=mod).loc[0:55, 90:150])
    pcc_models[i] += ca.cal_pcc(pre_CRU_India_uq_rvalue.loc[0:55, 90:150], pre_his_ds_India_uq_rvalue.sel(models=mod).loc[0:55, 90:150])
    pcc_models[i] += ca.cal_pcc(pre_CRU_India_vq_rvalue.loc[0:55, 90:150], pre_his_ds_India_vq_rvalue.sel(models=mod).loc[0:55, 90:150])
pcc_models = pcc_models/3.0
pcc_list = []
for i,mod in enumerate(models):
    pcc_list.append({"models": str(mod.data), "pcc": pcc_models[i]})
print(sorted(pcc_list, key=lambda x : x["pcc"]))
# %%
