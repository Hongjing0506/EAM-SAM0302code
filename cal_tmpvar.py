'''
Author: ChenHJ
Date: 2022-04-11 23:24:18
LastEditors: ChenHJ
LastEditTime: 2022-04-11 23:59:04
FilePath: /chenhj/0302code/cal_tmpvar.py
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
#   read multi-models data of historical
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
wap_his_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/wap"
)
g = os.walk(wap_his_path)
filepath = []
modelname_wap = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_wap.append(filename[loc[1] + 1 : loc[2]])
wapds_his = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
waphis_ds = xr.DataArray(wapds_his["wap"])
waphis_ds.coords["models"] = modelname_wap
# %%
ta_his_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/ta"
)
g = os.walk(ta_his_path)
filepath = []
modelname_ta = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_ta.append(filename[loc[1] + 1 : loc[2]])
tads_his = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
tahis_ds = xr.DataArray(tads_his["ta"])
tahis_ds.coords["models"] = modelname_ta
# %%
#   recalculate the plevel of the different variables in historical run
hgthis_ds.coords["plev"] = hgthis_ds["plev"] / 100.0
hgthis_ds = hgthis_ds.rename({"plev": "level"})
uhis_ds.coords["plev"] = uhis_ds["plev"] / 100.0
uhis_ds = uhis_ds.rename({"plev": "level"})
vhis_ds.coords["plev"] = vhis_ds["plev"] / 100.0
vhis_ds = vhis_ds.rename({"plev": "level"})
qhis_ds.coords["plev"] = qhis_ds["plev"] / 100.0
qhis_ds = qhis_ds.rename({"plev": "level"})
waphis_ds.coords["plev"] = waphis_ds["plev"] / 100.0
waphis_ds = waphis_ds.rename({"plev": "level"})
tahis_ds.coords["plev"] = tahis_ds["plev"] / 100.0
tahis_ds = tahis_ds.rename({"plev": "level"})
# %%
#   calculate the JJA mean in different variables of multi-models
hgthis_ds_ver_JJA = ca.p_time(hgthis_ds, 6, 8, True)
uhis_ds_ver_JJA = ca.p_time(uhis_ds, 6, 8, True)
vhis_ds_ver_JJA = ca.p_time(vhis_ds, 6, 8, True)
qhis_ds_ver_JJA = ca.p_time(qhis_ds, 6, 8, True)
waphis_ds_ver_JJA = ca.p_time(waphis_ds, 6, 8, True)
tahis_ds_ver_JJA = ca.p_time(tahis_ds, 6, 8, True)
sphis_ds_JJA = ca.p_time(sphis_ds, 6, 8, True)
prehis_ds_JJA = ca.p_time(prehis_ds, 6, 8, True)

# %%
#   reorder the multi-models in historical run
hgthis_ds_ver_JJA_copy = hgthis_ds_ver_JJA.copy()
uhis_ds_ver_JJA_copy = uhis_ds_ver_JJA.copy()
vhis_ds_ver_JJA_copy = vhis_ds_ver_JJA.copy()
qhis_ds_ver_JJA_copy = qhis_ds_ver_JJA.copy()
waphis_ds_ver_JJA_copy = waphis_ds_ver_JJA.copy()
tahis_ds_ver_JJA_copy = tahis_ds_ver_JJA.copy()
sphis_ds_JJA_copy = sphis_ds_JJA.copy()
prehis_ds_JJA_copy = prehis_ds_JJA.copy()
models = hgthis_ds_ver_JJA.coords["models"]

print(models)
for i, mod in enumerate(models):
    hgthis_ds_ver_JJA_copy[i, :, :, :, :] = np.array(hgthis_ds_ver_JJA.sel(models=mod))
    uhis_ds_ver_JJA_copy[i, :, :, :, :] = np.array(uhis_ds_ver_JJA.sel(models=mod))
    vhis_ds_ver_JJA_copy[i, :, :, :, :] = np.array(vhis_ds_ver_JJA.sel(models=mod))
    qhis_ds_ver_JJA_copy[i, :, :, :, :] = np.array(qhis_ds_ver_JJA.sel(models=mod))
    waphis_ds_ver_JJA_copy[i, :, :, :, :] = np.array(waphis_ds_ver_JJA.sel(models=mod))
    tahis_ds_ver_JJA_copy[i, :, :, :, :] = np.array(tahis_ds_ver_JJA.sel(models=mod))
    sphis_ds_JJA_copy[i, :, :, :] = np.array(sphis_ds_JJA.sel(models=mod))
    prehis_ds_JJA_copy[i, :, :, :] = np.array(prehis_ds_JJA.sel(models=mod))
hgthis_ds_ver_JJA = hgthis_ds_ver_JJA_copy.copy()
uhis_ds_ver_JJA = uhis_ds_ver_JJA_copy.copy()
vhis_ds_ver_JJA = vhis_ds_ver_JJA_copy.copy()
qhis_ds_ver_JJA = qhis_ds_ver_JJA_copy.copy()
waphis_ds_ver_JJA = waphis_ds_ver_JJA_copy.copy()
tahis_ds_ver_JJA = tahis_ds_ver_JJA_copy.copy()
sphis_ds_JJA = sphis_ds_JJA_copy.copy()
prehis_ds_JJA = prehis_ds_JJA_copy.copy()

hgthis_ds_ver_JJA.coords["models"] = models
uhis_ds_ver_JJA.coords["models"] = models
vhis_ds_ver_JJA.coords["models"] = models
qhis_ds_ver_JJA.coords["models"] = models
waphis_ds_ver_JJA.coords["models"] = models
tahis_ds_ver_JJA.coords["models"] = models
sphis_ds_JJA.coords["models"] = models
prehis_ds_JJA.coords["models"] = models

# %%
#   output the non-detrend variables of multi-models in historical run
hgthis_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/zg_historical_r144x72_195001-201412.nc")
uhis_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ua_historical_r144x72_195001-201412.nc")
vhis_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/va_historical_r144x72_195001-201412.nc")
qhis_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/hus_historical_r144x72_195001-201412.nc")
waphis_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/wap_historical_r144x72_195001-201412.nc")
tahis_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ta_historical_r144x72_195001-201412.nc")
prehis_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/pr_historical_r144x72_195001-201412.nc")
sphis_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/sp_historical_r144x72_195001-201412.nc")

# %%
#   calculate the detrend of different variables of multi-models
hgthis_ds_ver_JJA = ca.detrend_dim(hgthis_ds_ver_JJA, "time", deg=1, demean=False)
uhis_ds_ver_JJA = ca.detrend_dim(uhis_ds_ver_JJA, "time", deg=1, demean=False)
vhis_ds_ver_JJA = ca.detrend_dim(vhis_ds_ver_JJA, "time", deg=1, demean=False)
qhis_ds_ver_JJA = ca.detrend_dim(qhis_ds_ver_JJA, "time", deg=1, demean=False)
sphis_ds_JJA = ca.detrend_dim(sphis_ds_JJA, "time", deg=1, demean=False)
prehis_ds_JJA = ca.detrend_dim(prehis_ds_JJA, "time", deg=1, demean=False)