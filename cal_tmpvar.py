'''
Author: ChenHJ
Date: 2022-04-11 23:24:18
LastEditors: ChenHJ
LastEditTime: 2022-12-06 17:08:03
FilePath: /chenhj/0302code/cal_tmpvar.py
Aim: 
Mission: 
'''
# %%
from mailbox import _PartialFile
import numpy as np
import xarray as xr
import os
import re
from cdo import *
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
import skill_metrics as sm
from brokenaxes import brokenaxes

reload(ca)

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
import statsmodels.api as sm
import metpy.constants as constants

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
ts_his_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/ts"
)
g = os.walk(ts_his_path)
filepath = []
modelname_ts = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_ts.append(filename[loc[1] + 1 : loc[2]])
tsds_his = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
tshis_ds = xr.DataArray(tsds_his["ts"])
tshis_ds.coords["models"] = modelname_ts

tos_his_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tos"
)
g = os.walk(tos_his_path)
filepath = []
modelname_tos = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_tos.append(filename[loc[1] + 1 : loc[2]])
tosds_his = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
toshis_ds = xr.DataArray(tosds_his["tos"])
toshis_ds.coords["models"] = modelname_tos

hfls_his_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/hfls"
)
g = os.walk(hfls_his_path)
filepath = []
modelname_hfls = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_hfls.append(filename[loc[1] + 1 : loc[2]])
hflsds_his = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
hflshis_ds = xr.DataArray(hflsds_his["hfls"])
hflshis_ds.coords["models"] = modelname_hfls

hflshis_ds_ver_JJA = ca.p_time(hflshis_ds, 6, 8, True)


#   read multi-models data of ssp585
hgt_ssp585_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/zg"
)
g = os.walk(hgt_ssp585_path)
filepath = []
modelname_hgt = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_hgt.append(filename[loc[1] + 1 : loc[2]])
hgtds_ssp585 = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
hgtssp585_ds = xr.DataArray(hgtds_ssp585["zg"])
hgtssp585_ds.coords["models"] = modelname_hgt
# %%
u_ssp585_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/ua"
)
g = os.walk(u_ssp585_path)
filepath = []
modelname_u = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_u.append(filename[loc[1] + 1 : loc[2]])
uds_ssp585 = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
ussp585_ds = xr.DataArray(uds_ssp585["ua"])
ussp585_ds.coords["models"] = modelname_u
# %%
v_ssp585_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/va"
)
g = os.walk(v_ssp585_path)
filepath = []
modelname_v = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_v.append(filename[loc[1] + 1 : loc[2]])
vds_ssp585 = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
vssp585_ds = xr.DataArray(vds_ssp585["va"])
vssp585_ds.coords["models"] = modelname_v
# %%
sp_ssp585_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/ps"
)
g = os.walk(sp_ssp585_path)
filepath = []
modelname_sp = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_sp.append(filename[loc[1] + 1 : loc[2]])
spds_ssp585 = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
spssp585_ds = xr.DataArray(spds_ssp585["ps"])
spssp585_ds.coords["models"] = modelname_sp
# %%
q_ssp585_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/hus"
)
g = os.walk(q_ssp585_path)
filepath = []
modelname_q = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_q.append(filename[loc[1] + 1 : loc[2]])
qds_ssp585 = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
qssp585_ds = xr.DataArray(qds_ssp585["hus"])
qssp585_ds.coords["models"] = modelname_q


# %%
pr_ssp585_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/pr"
)
g = os.walk(pr_ssp585_path)
filepath = []
modelname_pr = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_pr.append(filename[loc[1] + 1 : loc[2]])
preds_ssp585 = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
pressp585_ds = xr.DataArray(preds_ssp585["pr"])*3600*24
pressp585_ds.coords["models"] = modelname_pr

# %%
wap_ssp585_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/wap"
)
g = os.walk(wap_ssp585_path)
filepath = []
modelname_wap = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_wap.append(filename[loc[1] + 1 : loc[2]])
wapds_ssp585 = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
wapssp585_ds = xr.DataArray(wapds_ssp585["wap"])
wapssp585_ds.coords["models"] = modelname_wap
# %%
ta_ssp585_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/ta"
)
g = os.walk(ta_ssp585_path)
filepath = []
modelname_ta = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_ta.append(filename[loc[1] + 1 : loc[2]])
tads_ssp585 = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
tassp585_ds = xr.DataArray(tads_ssp585["ta"])
tassp585_ds.coords["models"] = modelname_ta
# %%
ts_ssp585_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/ts"
)
g = os.walk(ts_ssp585_path)
filepath = []
modelname_ts = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_ts.append(filename[loc[1] + 1 : loc[2]])
tsds_ssp585 = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
tsssp585_ds = xr.DataArray(tsds_ssp585["ts"])
tsssp585_ds.coords["models"] = modelname_ts

tos_ssp585_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tos"
)
g = os.walk(tos_ssp585_path)
filepath = []
modelname_tos = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_tos.append(filename[loc[1] + 1 : loc[2]])
tosds_ssp585 = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
tosssp585_ds = xr.DataArray(tosds_ssp585["tos"])
tosssp585_ds.coords["models"] = modelname_tos


hfls_ssp585_path = (
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/hfls"
)
g = os.walk(hfls_ssp585_path)
filepath = []
modelname_hfls = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_hfls.append(filename[loc[1] + 1 : loc[2]])
hflsds_ssp585 = xr.open_mfdataset(filepath, concat_dim="models", combine="nested")
hflsssp585_ds = xr.DataArray(hflsds_ssp585["hfls"])
hflsssp585_ds.coords["models"] = modelname_hfls

hflsssp585_ds_ver_JJA = ca.p_time(hflsssp585_ds, 6, 8, True)
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

#   recalculate the plevel of the different variables in ssp585 run
hgtssp585_ds.coords["plev"] = hgtssp585_ds["plev"] / 100.0
hgtssp585_ds = hgtssp585_ds.rename({"plev": "level"})
ussp585_ds.coords["plev"] = ussp585_ds["plev"] / 100.0
ussp585_ds = ussp585_ds.rename({"plev": "level"})
vssp585_ds.coords["plev"] = vssp585_ds["plev"] / 100.0
vssp585_ds = vssp585_ds.rename({"plev": "level"})
qssp585_ds.coords["plev"] = qssp585_ds["plev"] / 100.0
qssp585_ds = qssp585_ds.rename({"plev": "level"})
wapssp585_ds.coords["plev"] = wapssp585_ds["plev"] / 100.0
wapssp585_ds = wapssp585_ds.rename({"plev": "level"})
tassp585_ds.coords["plev"] = tassp585_ds["plev"] / 100.0
tassp585_ds = tassp585_ds.rename({"plev": "level"})


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
tshis_ds_JJA = ca.p_time(tshis_ds, 6, 8, True)
toshis_ds_JJA = ca.p_time(toshis_ds, 6, 8, True)

#   calculate the JJA mean in different variables of multi-models
hgtssp585_ds_ver_JJA = ca.p_time(hgtssp585_ds, 6, 8, True)
ussp585_ds_ver_JJA = ca.p_time(ussp585_ds, 6, 8, True)
vssp585_ds_ver_JJA = ca.p_time(vssp585_ds, 6, 8, True)
qssp585_ds_ver_JJA = ca.p_time(qssp585_ds, 6, 8, True)
wapssp585_ds_ver_JJA = ca.p_time(wapssp585_ds, 6, 8, True)
tassp585_ds_ver_JJA = ca.p_time(tassp585_ds, 6, 8, True)
spssp585_ds_JJA = ca.p_time(spssp585_ds, 6, 8, True)
pressp585_ds_JJA = ca.p_time(pressp585_ds, 6, 8, True)
tsssp585_ds_JJA = ca.p_time(tsssp585_ds, 6, 8, True)
tosssp585_ds_JJA = ca.p_time(tosssp585_ds, 6, 8, True)
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
tshis_ds_JJA_copy = tshis_ds_JJA.copy()
toshis_ds_JJA_copy = toshis_ds_JJA.copy()
hflshis_ds_ver_JJA_copy = hflshis_ds_ver_JJA.copy()

#   reorder the multi-models in ssp585 run
hgtssp585_ds_ver_JJA_copy = hgtssp585_ds_ver_JJA.copy()
ussp585_ds_ver_JJA_copy = ussp585_ds_ver_JJA.copy()
vssp585_ds_ver_JJA_copy = vssp585_ds_ver_JJA.copy()
qssp585_ds_ver_JJA_copy = qssp585_ds_ver_JJA.copy()
wapssp585_ds_ver_JJA_copy = wapssp585_ds_ver_JJA.copy()
tassp585_ds_ver_JJA_copy = tassp585_ds_ver_JJA.copy()
spssp585_ds_JJA_copy = spssp585_ds_JJA.copy()
pressp585_ds_JJA_copy = pressp585_ds_JJA.copy()
tsssp585_ds_JJA_copy = tsssp585_ds_JJA.copy()
tosssp585_ds_JJA_copy = tosssp585_ds_JJA.copy()
hflsssp585_ds_ver_JJA_copy = hflsssp585_ds_ver_JJA.copy()

models = hgthis_ds_ver_JJA.coords["models"]
# print(models)
for i, mod in enumerate(models):
    hgthis_ds_ver_JJA_copy[i, :, :, :, :] = np.array(hgthis_ds_ver_JJA.sel(models=mod))
    uhis_ds_ver_JJA_copy[i, :, :, :, :] = np.array(uhis_ds_ver_JJA.sel(models=mod))
    vhis_ds_ver_JJA_copy[i, :, :, :, :] = np.array(vhis_ds_ver_JJA.sel(models=mod))
    qhis_ds_ver_JJA_copy[i, :, :, :, :] = np.array(qhis_ds_ver_JJA.sel(models=mod))
    waphis_ds_ver_JJA_copy[i, :, :, :, :] = np.array(waphis_ds_ver_JJA.sel(models=mod))
    tahis_ds_ver_JJA_copy[i, :, :, :, :] = np.array(tahis_ds_ver_JJA.sel(models=mod))
    sphis_ds_JJA_copy[i, :, :, :] = np.array(sphis_ds_JJA.sel(models=mod))
    prehis_ds_JJA_copy[i, :, :, :] = np.array(prehis_ds_JJA.sel(models=mod))
    tshis_ds_JJA_copy[i, :, :, :] = np.array(tshis_ds_JJA.sel(models=mod))
    toshis_ds_JJA_copy[i, :, :, :] = np.array(toshis_ds_JJA.sel(models=mod))
    
    hgtssp585_ds_ver_JJA_copy[i, :, :, :, :] = np.array(hgtssp585_ds_ver_JJA.sel(models=mod))
    ussp585_ds_ver_JJA_copy[i, :, :, :, :] = np.array(ussp585_ds_ver_JJA.sel(models=mod))
    vssp585_ds_ver_JJA_copy[i, :, :, :, :] = np.array(vssp585_ds_ver_JJA.sel(models=mod))
    qssp585_ds_ver_JJA_copy[i, :, :, :, :] = np.array(qssp585_ds_ver_JJA.sel(models=mod))
    wapssp585_ds_ver_JJA_copy[i, :, :, :, :] = np.array(wapssp585_ds_ver_JJA.sel(models=mod))
    tassp585_ds_ver_JJA_copy[i, :, :, :, :] = np.array(tassp585_ds_ver_JJA.sel(models=mod))
    spssp585_ds_JJA_copy[i, :, :, :] = np.array(spssp585_ds_JJA.sel(models=mod))
    pressp585_ds_JJA_copy[i, :, :, :] = np.array(pressp585_ds_JJA.sel(models=mod))
    tsssp585_ds_JJA_copy[i, :, :, :] = np.array(tsssp585_ds_JJA.sel(models=mod))
    tosssp585_ds_JJA_copy[i, :, :, :] = np.array(tosssp585_ds_JJA.sel(models=mod))
    
    hflshis_ds_ver_JJA_copy[i, :, :, :] = np.array(hflshis_ds_ver_JJA.sel(models=mod))
    hflsssp585_ds_ver_JJA_copy[i, :, :, :] = np.array(hflsssp585_ds_ver_JJA.sel(models=mod))
    
hgthis_ds_ver_JJA = hgthis_ds_ver_JJA_copy.copy()
uhis_ds_ver_JJA = uhis_ds_ver_JJA_copy.copy()
vhis_ds_ver_JJA = vhis_ds_ver_JJA_copy.copy()
qhis_ds_ver_JJA = qhis_ds_ver_JJA_copy.copy()
waphis_ds_ver_JJA = waphis_ds_ver_JJA_copy.copy()
tahis_ds_ver_JJA = tahis_ds_ver_JJA_copy.copy()
sphis_ds_JJA = sphis_ds_JJA_copy.copy()
prehis_ds_JJA = prehis_ds_JJA_copy.copy()
tshis_ds_JJA = tshis_ds_JJA_copy.copy()
toshis_ds_JJA = toshis_ds_JJA_copy.copy()

hgthis_ds_ver_JJA.coords["models"] = models
uhis_ds_ver_JJA.coords["models"] = models
vhis_ds_ver_JJA.coords["models"] = models
qhis_ds_ver_JJA.coords["models"] = models
waphis_ds_ver_JJA.coords["models"] = models
tahis_ds_ver_JJA.coords["models"] = models
sphis_ds_JJA.coords["models"] = models
prehis_ds_JJA.coords["models"] = models
tshis_ds_JJA.coords["models"] = models
toshis_ds_JJA.coords["models"] = models


hgtssp585_ds_ver_JJA = hgtssp585_ds_ver_JJA_copy.copy()
ussp585_ds_ver_JJA = ussp585_ds_ver_JJA_copy.copy()
vssp585_ds_ver_JJA = vssp585_ds_ver_JJA_copy.copy()
qssp585_ds_ver_JJA = qssp585_ds_ver_JJA_copy.copy()
wapssp585_ds_ver_JJA = wapssp585_ds_ver_JJA_copy.copy()
tassp585_ds_ver_JJA = tassp585_ds_ver_JJA_copy.copy()
spssp585_ds_JJA = spssp585_ds_JJA_copy.copy()
pressp585_ds_JJA = pressp585_ds_JJA_copy.copy()
tsssp585_ds_JJA = tsssp585_ds_JJA_copy.copy()
tosssp585_ds_JJA = tosssp585_ds_JJA_copy.copy()

hgtssp585_ds_ver_JJA.coords["models"] = models
ussp585_ds_ver_JJA.coords["models"] = models
vssp585_ds_ver_JJA.coords["models"] = models
qssp585_ds_ver_JJA.coords["models"] = models
wapssp585_ds_ver_JJA.coords["models"] = models
tassp585_ds_ver_JJA.coords["models"] = models
spssp585_ds_JJA.coords["models"] = models
pressp585_ds_JJA.coords["models"] = models
tsssp585_ds_JJA.coords["models"] = models
tosssp585_ds_JJA.coords["models"] = models

hflshis_ds_ver_JJA = hflshis_ds_ver_JJA_copy.copy()
hflshis_ds_ver_JJA.coords["models"] = models

hflsssp585_ds_ver_JJA = hflsssp585_ds_ver_JJA_copy.copy()
hflsssp585_ds_ver_JJA.coords["models"] = models
# %%
#   output the non-detrend variables of multi-models in historical run
hgthis_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/zg_historical_r144x72_195001-201412.nc")
uhis_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ua_historical_r144x72_195001-201412.nc")
vhis_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/va_historical_r144x72_195001-201412.nc")
qhis_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/hus_historical_r144x72_195001-201412.nc")
waphis_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/wap_historical_r144x72_195001-201412.nc")
tahis_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ta_historical_r144x72_195001-201412.nc")
prehis_ds_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/pr_historical_r144x72_195001-201412.nc")
sphis_ds_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ps_historical_r144x72_195001-201412.nc")
tshis_ds_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ts_historical_r144x72_195001-201412.nc")
toshis_ds_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/tos_historical_r144x72_195001-201412.nc")

#   output the non-detrend variables of multi-models in ssp585 run
hgtssp585_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/zg_ssp585_r144x72_201501-209912.nc")
ussp585_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ua_ssp585_r144x72_201501-209912.nc")
vssp585_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/va_ssp585_r144x72_201501-209912.nc")
qssp585_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/hus_ssp585_r144x72_201501-209912.nc")
wapssp585_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/wap_ssp585_r144x72_201501-209912.nc")
tassp585_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ta_ssp585_r144x72_201501-209912.nc")
pressp585_ds_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/pr_ssp585_r144x72_201501-209912.nc")
spssp585_ds_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ps_ssp585_r144x72_201501-209912.nc")
tsssp585_ds_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ts_ssp585_r144x72_201501-209912.nc")
tosssp585_ds_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/tos_ssp585_r144x72_201501-209912.nc")


hflshis_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/hfls_historical_r144x72_195001-201412.nc")

hflsssp585_ds_ver_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/hfls_ssp585_r144x72_201501-209912.nc")
# %%
# concatenate the variables in historical to ssp585
# First, read the non-detrend variables
hgthis_ds_ver_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/zg_historical_r144x72_195001-201412.nc")
uhis_ds_ver_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ua_historical_r144x72_195001-201412.nc")
vhis_ds_ver_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/va_historical_r144x72_195001-201412.nc")
qhis_ds_ver_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/hus_historical_r144x72_195001-201412.nc")
waphis_ds_ver_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/wap_historical_r144x72_195001-201412.nc")
tahis_ds_ver_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ta_historical_r144x72_195001-201412.nc")
prehis_ds_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/pr_historical_r144x72_195001-201412.nc")
sphis_ds_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ps_historical_r144x72_195001-201412.nc")
tshis_ds_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ts_historical_r144x72_195001-201412.nc")
toshis_ds_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/tos_historical_r144x72_195001-201412.nc")
hflshis_ds_ver_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/hfls_historical_r144x72_195001-201412.nc")
#   output the non-detrend variables of multi-models in ssp585 run
hgtssp585_ds_ver_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/zg_ssp585_r144x72_201501-209912.nc")
ussp585_ds_ver_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ua_ssp585_r144x72_201501-209912.nc")
vssp585_ds_ver_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/va_ssp585_r144x72_201501-209912.nc")
qssp585_ds_ver_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/hus_ssp585_r144x72_201501-209912.nc")
wapssp585_ds_ver_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/wap_ssp585_r144x72_201501-209912.nc")
tassp585_ds_ver_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ta_ssp585_r144x72_201501-209912.nc")
pressp585_ds_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/pr_ssp585_r144x72_201501-209912.nc")
spssp585_ds_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ps_ssp585_r144x72_201501-209912.nc")
tsssp585_ds_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ts_ssp585_r144x72_201501-209912.nc")
tosssp585_ds_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/tos_ssp585_r144x72_201501-209912.nc")
hflsssp585_ds_ver_JJA = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/hfls_ssp585_r144x72_201501-209912.nc")
# Second, concatenate the variables
hgtCMIP6_ds_ver_JJA = xr.concat([hgthis_ds_ver_JJA.sel(time=hgthis_ds_ver_JJA.time.dt.year>=1979), hgtssp585_ds_ver_JJA], "time")
uCMIP6_ds_ver_JJA = xr.concat([uhis_ds_ver_JJA.sel(time=uhis_ds_ver_JJA.time.dt.year>=1979), ussp585_ds_ver_JJA], "time")
vCMIP6_ds_ver_JJA = xr.concat([vhis_ds_ver_JJA.sel(time=vhis_ds_ver_JJA.time.dt.year>=1979), vssp585_ds_ver_JJA], "time")
qCMIP6_ds_ver_JJA = xr.concat([qhis_ds_ver_JJA.sel(time=qhis_ds_ver_JJA.time.dt.year>=1979), qssp585_ds_ver_JJA], "time")
wapCMIP6_ds_ver_JJA = xr.concat([waphis_ds_ver_JJA.sel(time=waphis_ds_ver_JJA.time.dt.year>=1979), wapssp585_ds_ver_JJA], "time")
taCMIP6_ds_ver_JJA = xr.concat([tahis_ds_ver_JJA.sel(time=tahis_ds_ver_JJA.time.dt.year>=1979), tassp585_ds_ver_JJA], "time")
preCMIP6_ds_JJA = xr.concat([prehis_ds_JJA.sel(time=prehis_ds_JJA.time.dt.year>=1979), pressp585_ds_JJA], "time")
spCMIP6_ds_JJA = xr.concat([sphis_ds_JJA.sel(time=sphis_ds_JJA.time.dt.year>=1979), spssp585_ds_JJA], "time")
tsCMIP6_ds_JJA = xr.concat([tshis_ds_JJA.sel(time=tshis_ds_JJA.time.dt.year>=1979), tsssp585_ds_JJA], "time")
tosCMIP6_ds_JJA = xr.concat([toshis_ds_JJA.sel(time=toshis_ds_JJA.time.dt.year>=1979), tosssp585_ds_JJA], "time")
hflsCMIP6_ds_ver_JJA = xr.concat([hflshis_ds_ver_JJA.sel(time=hflshis_ds_ver_JJA.time.dt.year>=1979), hflsssp585_ds_ver_JJA], "time")

# %%
# calculate the trend of different variables
hgtCMIP6_ds_ver_JJA_trend = hgtCMIP6_ds_ver_JJA.polyfit(dim="time",deg=1,skipna=True)["polyfit_coefficients"].sel(degree=1)
uCMIP6_ds_ver_JJA_trend = uCMIP6_ds_ver_JJA.polyfit(dim="time",deg=1,skipna=True)["polyfit_coefficients"].sel(degree=1)
vCMIP6_ds_ver_JJA_trend = vCMIP6_ds_ver_JJA.polyfit(dim="time",deg=1,skipna=True)["polyfit_coefficients"].sel(degree=1)
qCMIP6_ds_ver_JJA_trend = qCMIP6_ds_ver_JJA.polyfit(dim="time",deg=1,skipna=True)["polyfit_coefficients"].sel(degree=1)
wapCMIP6_ds_ver_JJA_trend = wapCMIP6_ds_ver_JJA.polyfit(dim="time",deg=1,skipna=True)["polyfit_coefficients"].sel(degree=1)
taCMIP6_ds_ver_JJA_trend = taCMIP6_ds_ver_JJA.polyfit(dim="time",deg=1,skipna=True)["polyfit_coefficients"].sel(degree=1)
preCMIP6_ds_JJA_trend = preCMIP6_ds_JJA.polyfit(dim="time",deg=1,skipna=True)["polyfit_coefficients"].sel(degree=1)
spCMIP6_ds_JJA_trend = spCMIP6_ds_JJA.polyfit(dim="time",deg=1,skipna=True)["polyfit_coefficients"].sel(degree=1)
tsCMIP6_ds_JJA_trend = tsCMIP6_ds_JJA.polyfit(dim="time",deg=1,skipna=True)["polyfit_coefficients"].sel(degree=1)
tosCMIP6_ds_JJA_trend = tosCMIP6_ds_JJA.polyfit(dim="time",deg=1,skipna=True)["polyfit_coefficients"].sel(degree=1)
hflsCMIP6_ds_ver_JJA_trend = hflsCMIP6_ds_ver_JJA.polyfit(dim="time",deg=1,skipna=True)["polyfit_coefficients"].sel(degree=1)


hgtCMIP6_ds_ver_JJA_trend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/hgtCMIP6_trend.nc")
uCMIP6_ds_ver_JJA_trend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/uCMIP6_trend.nc")
vCMIP6_ds_ver_JJA_trend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/vCMIP6_trend.nc")
qCMIP6_ds_ver_JJA_trend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/qCMIP6_trend.nc")
wapCMIP6_ds_ver_JJA_trend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/wapCMIP6_trend.nc")
taCMIP6_ds_ver_JJA_trend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/taCMIP6_trend.nc")
preCMIP6_ds_JJA_trend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/preCMIP6_trend.nc")
spCMIP6_ds_JJA_trend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/spCMIP6_trend.nc")
tsCMIP6_ds_JJA_trend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/tsCMIP6_trend.nc")
tosCMIP6_ds_JJA_trend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/tosCMIP6_trend.nc")
hflsCMIP6_ds_ver_JJA_trend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/hflsCMIP6_trend.nc")
# %%
#   calculate the detrend of different variables of multi-models
hgtCMIP6_ds_ver_JJA_detrend = ca.detrend_dim(hgtCMIP6_ds_ver_JJA, "time", deg=1, demean=True)
uCMIP6_ds_ver_JJA_detrend = ca.detrend_dim(uCMIP6_ds_ver_JJA, "time", deg=1, demean=True)
vCMIP6_ds_ver_JJA_detrend = ca.detrend_dim(vCMIP6_ds_ver_JJA, "time", deg=1, demean=True)
qCMIP6_ds_ver_JJA_detrend = ca.detrend_dim(qCMIP6_ds_ver_JJA, "time", deg=1, demean=True)
wapCMIP6_ds_ver_JJA_detrend = ca.detrend_dim(wapCMIP6_ds_ver_JJA, "time", deg=1, demean=True)
taCMIP6_ds_ver_JJA_detrend = ca.detrend_dim(taCMIP6_ds_ver_JJA, "time", deg=1, demean=True)
spCMIP6_ds_JJA_detrend = ca.detrend_dim(spCMIP6_ds_JJA, "time", deg=1, demean=True)
preCMIP6_ds_JJA_detrend = ca.detrend_dim(preCMIP6_ds_JJA, "time", deg=1, demean=True)
tsCMIP6_ds_JJA_detrend = ca.detrend_dim(tsCMIP6_ds_JJA, "time", deg=1, demean=True)
tosCMIP6_ds_JJA_detrend = ca.detrend_dim(tosCMIP6_ds_JJA, "time", deg=1, demean=True)
hflsCMIP6_ds_ver_JJA_detrend = ca.detrend_dim(hflsCMIP6_ds_ver_JJA, "time", deg=1, demean=True)
# %%
#   rename the variables of detrend data
hgtCMIP6_ds_ver_JJA_detrend.name = "zg"
uCMIP6_ds_ver_JJA_detrend.name = "ua"
vCMIP6_ds_ver_JJA_detrend.name = "va"
qCMIP6_ds_ver_JJA_detrend.name = "hus"
wapCMIP6_ds_ver_JJA_detrend.name = "wap"
taCMIP6_ds_ver_JJA_detrend.name = "ta"
spCMIP6_ds_JJA_detrend.name = "ps"
preCMIP6_ds_JJA_detrend.name = "pr"
tsCMIP6_ds_JJA_detrend.name = "ts"
tosCMIP6_ds_JJA_detrend.name = "sst"
hflsCMIP6_ds_ver_JJA_detrend.name = "hfls"
# %%
# cut the CMIP6 variables into two experiment
hgthis_ds_ver_JJA_detrend = ca.detrend_dim(hgtCMIP6_ds_ver_JJA_detrend.sel(time=hgtCMIP6_ds_ver_JJA_detrend.time.dt.year<=2014), "time", deg=1, demean=True)
hgtssp585_ds_ver_JJA_detrend = ca.detrend_dim(hgtCMIP6_ds_ver_JJA_detrend.sel(time=hgtCMIP6_ds_ver_JJA_detrend.time.dt.year>=2015), "time", deg=1, demean=True)
hgtssp585_p3_ds_ver_JJA_detrend = ca.detrend_dim(hgtCMIP6_ds_ver_JJA_detrend.sel(time=hgtCMIP6_ds_ver_JJA_detrend.time.dt.year>=2064), "time", deg=1, demean=True)

uhis_ds_ver_JJA_detrend = ca.detrend_dim(uCMIP6_ds_ver_JJA_detrend.sel(time=uCMIP6_ds_ver_JJA_detrend.time.dt.year<=2014), "time", deg=1, demean=True)
ussp585_ds_ver_JJA_detrend = ca.detrend_dim(uCMIP6_ds_ver_JJA_detrend.sel(time=uCMIP6_ds_ver_JJA_detrend.time.dt.year>=2015), "time", deg=1, demean=True)
ussp585_p3_ds_ver_JJA_detrend = ca.detrend_dim(uCMIP6_ds_ver_JJA_detrend.sel(time=uCMIP6_ds_ver_JJA_detrend.time.dt.year>=2064), "time", deg=1, demean=True)

vhis_ds_ver_JJA_detrend = ca.detrend_dim(vCMIP6_ds_ver_JJA_detrend.sel(time=vCMIP6_ds_ver_JJA_detrend.time.dt.year<=2014), "time", deg=1, demean=True)
vssp585_ds_ver_JJA_detrend = ca.detrend_dim(vCMIP6_ds_ver_JJA_detrend.sel(time=vCMIP6_ds_ver_JJA_detrend.time.dt.year>=2015), "time", deg=1, demean=True)
vssp585_p3_ds_ver_JJA_detrend = ca.detrend_dim(vCMIP6_ds_ver_JJA_detrend.sel(time=vCMIP6_ds_ver_JJA_detrend.time.dt.year>=2064), "time", deg=1, demean=True)

qhis_ds_ver_JJA_detrend = ca.detrend_dim(qCMIP6_ds_ver_JJA_detrend.sel(time=qCMIP6_ds_ver_JJA_detrend.time.dt.year<=2014), "time", deg=1, demean=True)
qssp585_ds_ver_JJA_detrend = ca.detrend_dim(qCMIP6_ds_ver_JJA_detrend.sel(time=qCMIP6_ds_ver_JJA_detrend.time.dt.year>=2015), "time", deg=1, demean=True)
qssp585_p3_ds_ver_JJA_detrend = ca.detrend_dim(qCMIP6_ds_ver_JJA_detrend.sel(time=qCMIP6_ds_ver_JJA_detrend.time.dt.year>=2064), "time", deg=1, demean=True)

waphis_ds_ver_JJA_detrend = ca.detrend_dim(wapCMIP6_ds_ver_JJA_detrend.sel(time=wapCMIP6_ds_ver_JJA_detrend.time.dt.year<=2014), "time", deg=1, demean=True)
wapssp585_ds_ver_JJA_detrend = ca.detrend_dim(wapCMIP6_ds_ver_JJA_detrend.sel(time=wapCMIP6_ds_ver_JJA_detrend.time.dt.year>=2015), "time", deg=1, demean=True)
wapssp585_p3_ds_ver_JJA_detrend = ca.detrend_dim(wapCMIP6_ds_ver_JJA_detrend.sel(time=wapCMIP6_ds_ver_JJA_detrend.time.dt.year>=2064), "time", deg=1, demean=True)

tahis_ds_ver_JJA_detrend = ca.detrend_dim(taCMIP6_ds_ver_JJA_detrend.sel(time=taCMIP6_ds_ver_JJA_detrend.time.dt.year<=2014), "time", deg=1, demean=True)
tassp585_ds_ver_JJA_detrend = ca.detrend_dim(taCMIP6_ds_ver_JJA_detrend.sel(time=taCMIP6_ds_ver_JJA_detrend.time.dt.year>=2015), "time", deg=1, demean=True)
tassp585_p3_ds_ver_JJA_detrend = ca.detrend_dim(taCMIP6_ds_ver_JJA_detrend.sel(time=taCMIP6_ds_ver_JJA_detrend.time.dt.year>=2064), "time", deg=1, demean=True)

sphis_ds_JJA_detrend = ca.detrend_dim(spCMIP6_ds_JJA_detrend.sel(time=spCMIP6_ds_JJA_detrend.time.dt.year<=2014), "time", deg=1, demean=True)
spssp585_ds_JJA_detrend = ca.detrend_dim(spCMIP6_ds_JJA_detrend.sel(time=spCMIP6_ds_JJA_detrend.time.dt.year>=2015), "time", deg=1, demean=True)
spssp585_p3_ds_JJA_detrend = ca.detrend_dim(spCMIP6_ds_JJA_detrend.sel(time=spCMIP6_ds_JJA_detrend.time.dt.year>=2064), "time", deg=1, demean=True)

prehis_ds_JJA_detrend = ca.detrend_dim(preCMIP6_ds_JJA_detrend.sel(time=preCMIP6_ds_JJA_detrend.time.dt.year<=2014), "time", deg=1, demean=True)
pressp585_ds_JJA_detrend = ca.detrend_dim(preCMIP6_ds_JJA_detrend.sel(time=preCMIP6_ds_JJA_detrend.time.dt.year>=2015), "time", deg=1, demean=True)
pressp585_p3_ds_JJA_detrend = ca.detrend_dim(preCMIP6_ds_JJA_detrend.sel(time=preCMIP6_ds_JJA_detrend.time.dt.year>=2064), "time", deg=1, demean=True)

tshis_ds_JJA_detrend = ca.detrend_dim(tsCMIP6_ds_JJA_detrend.sel(time=tsCMIP6_ds_JJA_detrend.time.dt.year<=2014), "time", deg=1, demean=True)
tsssp585_ds_JJA_detrend = ca.detrend_dim(tsCMIP6_ds_JJA_detrend.sel(time=tsCMIP6_ds_JJA_detrend.time.dt.year>=2015), "time", deg=1, demean=True)
tsssp585_p3_ds_JJA_detrend = ca.detrend_dim(tsCMIP6_ds_JJA_detrend.sel(time=tsCMIP6_ds_JJA_detrend.time.dt.year>=2064), "time", deg=1, demean=True)

toshis_ds_JJA_detrend = ca.detrend_dim(tosCMIP6_ds_JJA_detrend.sel(time=tosCMIP6_ds_JJA_detrend.time.dt.year<=2014), "time", deg=1, demean=True)
tosssp585_ds_JJA_detrend = ca.detrend_dim(tosCMIP6_ds_JJA_detrend.sel(time=tosCMIP6_ds_JJA_detrend.time.dt.year>=2015), "time", deg=1, demean=True)
tosssp585_p3_ds_JJA_detrend = ca.detrend_dim(tosCMIP6_ds_JJA_detrend.sel(time=tosCMIP6_ds_JJA_detrend.time.dt.year>=2064), "time", deg=1, demean=True)

hflshis_ds_ver_JJA_detrend = ca.detrend_dim(hflsCMIP6_ds_ver_JJA_detrend.sel(time=hflsCMIP6_ds_ver_JJA_detrend.time.dt.year<=2014), "time", deg=1, demean=True)
hflsssp585_ds_ver_JJA_detrend = ca.detrend_dim(hflsCMIP6_ds_ver_JJA_detrend.sel(time=hflsCMIP6_ds_ver_JJA_detrend.time.dt.year>=2015), "time", deg=1, demean=True)
hflsssp585_p3_ds_ver_JJA_detrend = ca.detrend_dim(hflsCMIP6_ds_ver_JJA_detrend.sel(time=hflsCMIP6_ds_ver_JJA_detrend.time.dt.year>=2064), "time", deg=1, demean=True)
# %%
#   rename the variables of detrend data
hgthis_ds_ver_JJA_detrend.name = "zg"
uhis_ds_ver_JJA_detrend.name = "ua"
vhis_ds_ver_JJA_detrend.name = "va"
qhis_ds_ver_JJA_detrend.name = "hus"
waphis_ds_ver_JJA_detrend.name = "wap"
tahis_ds_ver_JJA_detrend.name = "ta"
sphis_ds_JJA_detrend.name = "ps"
prehis_ds_JJA_detrend.name = "pr"
tshis_ds_JJA_detrend.name = "ts"
toshis_ds_JJA_detrend.name = "sst"

hgtssp585_ds_ver_JJA_detrend.name = "zg"
ussp585_ds_ver_JJA_detrend.name = "ua"
vssp585_ds_ver_JJA_detrend.name = "va"
qssp585_ds_ver_JJA_detrend.name = "hus"
wapssp585_ds_ver_JJA_detrend.name = "wap"
tassp585_ds_ver_JJA_detrend.name = "ta"
spssp585_ds_JJA_detrend.name = "ps"
pressp585_ds_JJA_detrend.name = "pr"
tsssp585_ds_JJA_detrend.name = "ts"
tosssp585_ds_JJA_detrend.name = "sst"

hgtssp585_p3_ds_ver_JJA_detrend.name = "zg"
ussp585_p3_ds_ver_JJA_detrend.name = "ua"
vssp585_p3_ds_ver_JJA_detrend.name = "va"
qssp585_p3_ds_ver_JJA_detrend.name = "hus"
wapssp585_p3_ds_ver_JJA_detrend.name = "wap"
tassp585_p3_ds_ver_JJA_detrend.name = "ta"
spssp585_p3_ds_JJA_detrend.name = "ps"
pressp585_p3_ds_JJA_detrend.name = "pr"
tsssp585_p3_ds_JJA_detrend.name = "ts"
tosssp585_p3_ds_JJA_detrend.name = "sst"

hflshis_ds_ver_JJA_detrend.name = "hfls"
hflsssp585_ds_ver_JJA_detrend.name = "hfls"
hflsssp585_p3_ds_ver_JJA_detrend.name = "hfls"
# %%
#   output the detrended variables of multi-models in historical run
hgthis_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/zg_historical_r144x72_197901-201412.nc")
uhis_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/ua_historical_r144x72_197901-201412.nc")
vhis_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/va_historical_r144x72_197901-201412.nc")
qhis_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/hus_historical_r144x72_197901-201412.nc")
waphis_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/wap_historical_r144x72_197901-201412.nc")
tahis_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/ta_historical_r144x72_197901-201412.nc")
prehis_ds_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/pr_historical_r144x72_197901-201412.nc")
sphis_ds_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/ps_historical_r144x72_197901-201412.nc")
tshis_ds_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/ts_historical_r144x72_197901-201412.nc")
toshis_ds_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/tos_historical_r144x72_197901-201412.nc")

hgtssp585_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/zg_ssp585_r144x72_201501-209912.nc")
ussp585_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ua_ssp585_r144x72_201501-209912.nc")
vssp585_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/va_ssp585_r144x72_201501-209912.nc")
qssp585_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/hus_ssp585_r144x72_201501-209912.nc")
wapssp585_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/wap_ssp585_r144x72_201501-209912.nc")
tassp585_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ta_ssp585_r144x72_201501-209912.nc")
pressp585_ds_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/pr_ssp585_r144x72_201501-209912.nc")
spssp585_ds_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ps_ssp585_r144x72_201501-209912.nc")
tsssp585_ds_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ts_ssp585_r144x72_201501-209912.nc")
tosssp585_ds_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/tos_ssp585_r144x72_201501-209912.nc")

hgtssp585_p3_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/zg_ssp585_p3_r144x72_206401-209912.nc")
ussp585_p3_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ua_ssp585_p3_r144x72_206401-209912.nc")
vssp585_p3_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/va_ssp585_p3_r144x72_206401-209912.nc")
qssp585_p3_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/hus_ssp585_p3_r144x72_206401-209912.nc")
wapssp585_p3_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/wap_ssp585_p3_r144x72_206401-209912.nc")
tassp585_p3_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ta_ssp585_p3_r144x72_206401-209912.nc")
pressp585_p3_ds_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/pr_ssp585_p3_r144x72_206401-209912.nc")
spssp585_p3_ds_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ps_ssp585_p3_r144x72_206401-209912.nc")
tsssp585_p3_ds_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ts_ssp585_p3_r144x72_206401-209912.nc")
tosssp585_p3_ds_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/tos_ssp585_p3_r144x72_206401-209912.nc")

hflshis_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/hfls_historical_r144x72_197901-201412.nc")
hflsssp585_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/hfls_ssp585_r144x72_201501-209912.nc")
hflsssp585_p3_ds_ver_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/hfls_ssp585_p3_r144x72_206401-209912.nc")
# %%
#   read non-detrend data
fuhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ua_historical_r144x72_195001-201412.nc")
uhis_ver_JJA = fuhis_ver_JJA["ua"]

fvhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/va_historical_r144x72_195001-201412.nc")
vhis_ver_JJA = fvhis_ver_JJA["va"]
# %%
# #   interpolate the nan in 850hPa wind fields
# from scipy.interpolate import interp2d
# lat = uhis_ver_JJA.coords["lat"]
# lon = uhis_ver_JJA.coords["lon"]
# time = uhis_ver_JJA.coords["time"]
# models = uhis_ver_JJA.coords["models"]
# uhis_ver_JJA_filled = uhis_ver_JJA.sel(level=850.0).copy()
# vhis_ver_JJA_filled = vhis_ver_JJA.sel(level=850.0).copy()
# for i,mod in enumerate(models):
#     for ti in range(len(time)):
#         u_filled_func = interp2d(lon, lat, uhis_ver_JJA.sel(level=850.0)[i,ti,:,:], kind="linear", bounds_error=False)
#         uhis_ver_JJA_filled[i,ti,:,:] = u_filled_func(lon, lat)
#         v_filled_func = interp2d(lon, lat, vhis_ver_JJA.sel(level=850.0)[i,ti,:,:], kind="linear", bounds_error=False)
#         vhis_ver_JJA_filled[i,ti,:,:] = v_filled_func(lon, lat)
# lenmodels = np.arange(len(models))
# uhis_ver_JJA_filled.coords["models"] = lenmodels
# vhis_ver_JJA_filled.coords["models"] = lenmodels

# uhis_ver_JJA_filled = uhis_ver_JJA_filled.interpolate_na(dim="models",method="nearest", fill_value="extrapolate")
# vhis_ver_JJA_filled = vhis_ver_JJA_filled.interpolate_na(dim="models",method="nearest", fill_value="extrapolate")
# uhis_ver_JJA_filled.coords["models"] = models
# vhis_ver_JJA_filled.coords["models"] = models

# # %%
# uhis_ver_JJA_filled = uhis_ver_JJA_filled.expand_dims("level")
# vhis_ver_JJA_filled = vhis_ver_JJA_filled.expand_dims("level")
# %%
#   calculate the non-detrend SAM/EAM/IWF
his_SAM_index = ca.SAM(uhis_ver_JJA)
his_EAM_index = ca.EAM(uhis_ver_JJA)
his_IWF_index = ca.IWF(uhis_ver_JJA, vhis_ver_JJA)

# %%
#   ouput the non-detrend SAM/EAM/IWF
his_SAM_index.name = "SAM"
his_EAM_index.name = "EAM"
his_IWF_index.name = "IWF"

his_SAM_index.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_SAM_index_1950-2014.nc")
his_EAM_index.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_EAM_index_1950-2014.nc")
his_IWF_index.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_IWF_index_1950-2014.nc")
# %%
#   calculate the uq and vq for non-detrend data in different models
fqhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/hus_historical_r144x72_195001-201412.nc")
qhis_ver_JJA = fqhis_ver_JJA["hus"]
fsphis_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ps_historical_r144x72_195001-201412.nc")
sphis_JJA = fsphis_JJA["ps"]


ptop = 1 * 100
g = 9.8
his_dslevel = qhis_ver_JJA.coords["level"] * 100.0
his_dslevel.attrs["units"] = "Pa"
# his_dsdp = geocat.comp.dpres_plevel(his_dslevel, sphis_JJA, ptop)
# print(sphis_ds_JJA)
his_dsdp = xr.apply_ufunc(
    geocat.comp.dpres_plevel,
    his_dslevel,
    sphis_JJA,
    ptop,
    input_core_dims=[["level"], [], []],
    output_core_dims=[["level"]],
    vectorize=True,
    dask="parallelized",
)
# for i in np.arange(0,26):
#     print(his_dsdp[i, 0, 0, 0, :])
his_dsdp = his_dsdp.transpose("models", "time", "level", "lat", "lon")
his_dsdpg = his_dsdp / g
his_dsdpg.attrs["units"] = "kg/m2"
# %%
his_dsdpg.name = "dsdpg"
his_dsdpg.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_dsdpg.nc")


# %%
# his_dsdpg = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/his_dsdpg.nc")
uqhis_ver_JJA = uhis_ver_JJA * qhis_ver_JJA * 1000.0
vqhis_ver_JJA = vhis_ver_JJA * qhis_ver_JJA * 1000.0
uqhis_ver_JJA.attrs["units"] = "[m/s][g/kg]"
vqhis_ver_JJA.attrs["units"] = "[m/s][g/kg]"
uq_dpg_his_JJA = (uqhis_ver_JJA * his_dsdpg.data).sum(dim="level", skipna=True) / 1e05
vq_dpg_his_JJA = (vqhis_ver_JJA * his_dsdpg.data).sum(dim="level", skipna=True) / 1e05
# uq_dpg_his_JJA = ca.detrend_dim(uq_dpg_his_JJA, "time", deg=1, demean=False)
# vq_dpg_his_JJA = ca.detrend_dim(vq_dpg_his_JJA, "time", deg=1, demean=False)
uq_dpg_his_JJA.attrs["units"] = "100kg/(m*s)"
vq_dpg_his_JJA.attrs["units"] = "100kg/(m*s)"
# %%
uq_dpg_his_JJA.name = "uq_dpg"
vq_dpg_his_JJA.name = "vq_dpg"

uq_dpg_his_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_uq_dpg.nc")
vq_dpg_his_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_vq_dpg.nc")
# %%
#   calculate detrended SAM/EAM/IWF
his_SAM_index_detrend = ca.detrend_dim(his_SAM_index, "time", deg=1, demean=False)
his_EAM_index_detrend = ca.detrend_dim(his_EAM_index, "time", deg=1, demean=False)
his_IWF_index_detrend = ca.detrend_dim(his_IWF_index, "time", deg=1, demean=False)

his_SAM_index_detrend.name = "SAM"
his_EAM_index_detrend.name = "EAM"
his_IWF_index_detrend.name = "IWF"

# %%
#   output the detrended SAM/EAM/IWF
his_SAM_index_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_SAM_index_1950-2014.nc")
his_EAM_index_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_EAM_index_1950-2014.nc")
his_IWF_index_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_IWF_index_1950-2014.nc")


# %%
#   calculate the detrended his_dsdpg/uq/vq
his_dsdpg_detrend = ca.detrend_dim(his_dsdpg, "time", deg=1, demean=False)
his_dsdpg_detrend.name = "dsdpg"
his_dsdpg_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_dsdpg.nc")
# %%
uq_dpg_his_JJA_detrend = ca.detrend_dim(uq_dpg_his_JJA, "time", deg=1, demean=False)
vq_dpg_his_JJA_detrend = ca.detrend_dim(vq_dpg_his_JJA, "time", deg=1, demean=False)
uq_dpg_his_JJA_detrend.name = "uq_dpg"
vq_dpg_his_JJA_detrend.name = "vq_dpg"
uq_dpg_his_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_uq_dpg.nc")
vq_dpg_his_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_vq_dpg.nc")

# %%
#   calculate the detrend of different variables of multi-models
hgtssp585_ds_ver_JJA = ca.detrend_dim(hgtssp585_ds_ver_JJA, "time", deg=1, demean=False)
ussp585_ds_ver_JJA = ca.detrend_dim(ussp585_ds_ver_JJA, "time", deg=1, demean=False)
vssp585_ds_ver_JJA = ca.detrend_dim(vssp585_ds_ver_JJA, "time", deg=1, demean=False)
qssp585_ds_ver_JJA = ca.detrend_dim(qssp585_ds_ver_JJA, "time", deg=1, demean=False)
wapssp585_ds_ver_JJA = ca.detrend_dim(wapssp585_ds_ver_JJA, "time", deg=1, demean=False)
tassp585_ds_ver_JJA = ca.detrend_dim(tassp585_ds_ver_JJA, "time", deg=1, demean=False)
spssp585_ds_JJA = ca.detrend_dim(spssp585_ds_JJA, "time", deg=1, demean=False)
pressp585_ds_JJA = ca.detrend_dim(pressp585_ds_JJA, "time", deg=1, demean=False)
# %%
tsssp585_ds_JJA = ca.detrend_dim(tsssp585_ds_JJA, "time", deg=1, demean=False)
tosssp585_ds_JJA = ca.detrend_dim(tosssp585_ds_JJA, "time", deg=1, demean=False)
# %%
#   rename the variables of detrend data
hgtssp585_ds_ver_JJA.name = "zg"
ussp585_ds_ver_JJA.name = "ua"
vssp585_ds_ver_JJA.name = "va"
qssp585_ds_ver_JJA.name = "hus"
wapssp585_ds_ver_JJA.name = "wap"
tassp585_ds_ver_JJA.name = "ta"
spssp585_ds_JJA.name = "ps"
pressp585_ds_JJA.name = "pr"
# %%
tsssp585_ds_JJA.name = "ts"
tosssp585_ds_JJA.name = "sst"
# %%
#   output the detrended variables of multi-models in ssp585 run

# %%
#   read non-detrend data
fussp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ua_ssp585_r144x72_201501-209912.nc")
ussp585_ver_JJA = fussp585_ver_JJA["ua"]

fvssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/va_ssp585_r144x72_201501-209912.nc")
vssp585_ver_JJA = fvssp585_ver_JJA["va"]

# %%
#   interpolate the nan in 850hPa wind fields
from scipy.interpolate import interp2d
lat = ussp585_ver_JJA.coords["lat"]
lon = ussp585_ver_JJA.coords["lon"]
time = ussp585_ver_JJA.coords["time"]
models = ussp585_ver_JJA.coords["models"]
ussp585_ver_JJA_filled = ussp585_ver_JJA.sel(level=850.0).copy()
vssp585_ver_JJA_filled = vssp585_ver_JJA.sel(level=850.0).copy()
for i,mod in enumerate(models):
    for ti in range(len(time)):
        u_filled_func = interp2d(lon, lat, ussp585_ver_JJA.sel(level=850.0)[i,ti,:,:], kind="linear", bounds_error=False)
        ussp585_ver_JJA_filled[i,ti,:,:] = u_filled_func(lon, lat)
        v_filled_func = interp2d(lon, lat, vssp585_ver_JJA.sel(level=850.0)[i,ti,:,:], kind="linear", bounds_error=False)
        vssp585_ver_JJA_filled[i,ti,:,:] = v_filled_func(lon, lat)
lenmodels = np.arange(len(models))
ussp585_ver_JJA_filled.coords["models"] = lenmodels
vssp585_ver_JJA_filled.coords["models"] = lenmodels

ussp585_ver_JJA_filled = ussp585_ver_JJA_filled.interpolate_na(dim="models",method="nearest", fill_value="extrapolate")
vssp585_ver_JJA_filled = vssp585_ver_JJA_filled.interpolate_na(dim="models",method="nearest", fill_value="extrapolate")
ussp585_ver_JJA_filled.coords["models"] = models
vssp585_ver_JJA_filled.coords["models"] = models
# %%
ussp585_ver_JJA_filled = ussp585_ver_JJA_filled.expand_dims("level")
vssp585_ver_JJA_filled = vssp585_ver_JJA_filled.expand_dims("level")
# %%
#   calculate the non-detrend SAM/EAM/IWF
ssp585_SAM_index = ca.SAM(ussp585_ver_JJA)
ssp585_EAM_index = ca.EAM(ussp585_ver_JJA)
ssp585_IWF_index = ca.IWF(ussp585_ver_JJA_filled, vssp585_ver_JJA_filled)
# %%
#   ouput the non-detrend SAM/EAM/IWF
ssp585_SAM_index.name = "SAM"
ssp585_EAM_index.name = "EAM"
ssp585_IWF_index.name = "IWF"

ssp585_SAM_index.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_SAM_index_2015-2099.nc")
ssp585_EAM_index.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_EAM_index_2015-2099.nc")
ssp585_IWF_index.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_IWF_index_2015-2099.nc")

# %%
#   calculate the uq and vq for non-detrend data in different models
fqssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/hus_ssp585_r144x72_201501-209912.nc")
qssp585_ver_JJA = fqssp585_ver_JJA["hus"]
fspssp585_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ps_ssp585_r144x72_201501-209912.nc")
spssp585_JJA = fspssp585_JJA["ps"]

# %%
ptop = 1 * 100
g = 9.8
ssp585_dslevel = qssp585_ver_JJA.coords["level"] * 100.0
ssp585_dslevel.attrs["units"] = "Pa"
# ssp585_dsdp = geocat.comp.dpres_plevel(ssp585_dslevel, spssp585_JJA, ptop)
# print(spssp585_ds_JJA)
ssp585_dsdp = xr.apply_ufunc(
    geocat.comp.dpres_plevel,
    ssp585_dslevel,
    spssp585_JJA,
    ptop,
    input_core_dims=[["level"], [], []],
    output_core_dims=[["level"]],
    vectorize=True,
    dask="parallelized",
)
# for i in np.arange(0,26):
#     print(ssp585_dsdp[i, 0, 0, 0, :])
ssp585_dsdp = ssp585_dsdp.transpose("models", "time", "level", "lat", "lon")
ssp585_dsdpg = ssp585_dsdp / g
ssp585_dsdpg.attrs["units"] = "kg/m2"
# %%
ssp585_dsdpg.name = "dsdpg"
ssp585_dsdpg.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_dsdpg.nc")
# %%
ssp585_dsdpg = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_dsdpg.nc")
uqssp585_ver_JJA = ussp585_ver_JJA * qssp585_ver_JJA * 1000.0
vqssp585_ver_JJA = vssp585_ver_JJA * qssp585_ver_JJA * 1000.0
uqssp585_ver_JJA.attrs["units"] = "[m/s][g/kg]"
vqssp585_ver_JJA.attrs["units"] = "[m/s][g/kg]"
uq_dpg_ssp585_JJA = (uqssp585_ver_JJA * ssp585_dsdpg.data).sum(dim="level", skipna=True) / 1e05
vq_dpg_ssp585_JJA = (vqssp585_ver_JJA * ssp585_dsdpg.data).sum(dim="level", skipna=True) / 1e05
# uq_dpg_ssp585_JJA = ca.detrend_dim(uq_dpg_ssp585_JJA, "time", deg=1, demean=False)
# vq_dpg_ssp585_JJA = ca.detrend_dim(vq_dpg_ssp585_JJA, "time", deg=1, demean=False)
uq_dpg_ssp585_JJA.attrs["units"] = "100kg/(m*s)"
vq_dpg_ssp585_JJA.attrs["units"] = "100kg/(m*s)"
# %%
uq_dpg_ssp585_JJA.name = "uq_dpg"
vq_dpg_ssp585_JJA.name = "vq_dpg"

uq_dpg_ssp585_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_uq_dpg.nc")
vq_dpg_ssp585_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_vq_dpg.nc")

# %%
#   calculate detrended SAM/EAM/IWF
ssp585_SAM_index_detrend = ca.detrend_dim(ssp585_SAM_index, "time", deg=1, demean=False)
ssp585_EAM_index_detrend = ca.detrend_dim(ssp585_EAM_index, "time", deg=1, demean=False)
ssp585_IWF_index_detrend = ca.detrend_dim(ssp585_IWF_index, "time", deg=1, demean=False)

ssp585_SAM_index_detrend.name = "SAM"
ssp585_EAM_index_detrend.name = "EAM"
ssp585_IWF_index_detrend.name = "IWF"

#   output the detrended SAM/EAM/IWF
ssp585_SAM_index_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_SAM_index_2015-2099.nc")
ssp585_EAM_index_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_EAM_index_2015-2099.nc")
ssp585_IWF_index_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_IWF_index_2015-2099.nc")
# %%

ssp585_dsdpg_detrend = ca.detrend_dim(ssp585_dsdpg, "time", deg=1, demean=False)

ssp585_dsdpg_detrend.name = "dsdpg"
ssp585_dsdpg_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_dsdpg.nc")
# %%
uq_dpg_ssp585_JJA_detrend = ca.detrend_dim(uq_dpg_ssp585_JJA, "time", deg=1, demean=False)
vq_dpg_ssp585_JJA_detrend = ca.detrend_dim(vq_dpg_ssp585_JJA, "time", deg=1, demean=False)

uq_dpg_ssp585_JJA_detrend.name = "uq_dpg"
vq_dpg_ssp585_JJA_detrend.name = "vq_dpg"

uq_dpg_ssp585_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_uq_dpg.nc")
vq_dpg_ssp585_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_vq_dpg.nc")
# %%

# %%

# %%
