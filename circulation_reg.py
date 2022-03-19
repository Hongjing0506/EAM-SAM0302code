'''
Author: ChenHJ
Date: 2022-03-16 17:42:02
LastEditors: ChenHJ
LastEditTime: 2022-03-19 18:03:39
FilePath: /chenhj/0302code/circulation_reg.py
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
import pandas as pd
from importlib import reload

import metpy.calc as mpcalc
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
from eofs.multivariate.standard import MultivariateEof
from eofs.standard import Eof
# %%
#   read the data file
fpreCRU = xr.open_dataset(
    "/home/ys17-23/chenhj/SAM_EAM_data/obs/cru_ts4.01_r144x72_195001-201412.nc"
)
preCRU = fpreCRU["pre"]

fprehis = xr.open_dataset(
    "/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/pr/pr_Amon_ensemble_historical_gn_195001-201412.nc"
)
prehis = fprehis["pr"]

pr_his_path = "/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/pr"
g = os.walk(pr_his_path)
filepath = []
modelname_pr = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_pr.append(filename[loc[1]+1:loc[2]])
preds_his = xr.open_mfdataset(filepath, concat_dim="models", combine='nested')
prehis_ds = xr.DataArray(preds_his['pr'])
prehis_ds.coords["models"] = modelname_pr

fvERA5 = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/obs/vwind_mon_r144x72_195001-201412.nc")
vERA5 = fvERA5["v"]

fvhis = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/va/va_Amon_ensemble_historical_gn_195001-201412.nc")
vhis = fvhis["va"]

fuERA5 = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/obs/uwind_mon_r144x72_195001-201412.nc")
uERA5 = fuERA5["u"]

fuhis = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/ua/ua_Amon_ensemble_historical_gn_195001-201412.nc")
uhis = fuhis["ua"]

fspERA5 = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/obs/sp_mon_r144x72_195001-201412.nc")
spERA5 = fspERA5["sp"]

fsphis = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/ps/ps_Amon_ensemble_historical_gn_195001-201412.nc")
sphis = fsphis["ps"]

fqERA5 = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/obs/q_mon_r144x72_195001-201412.nc")
qERA5 = fqERA5["q"]

fqhis = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/hus/hus_Amon_ensemble_historical_gn_195001-201412.nc")
qhis = fqhis["hus"]
# %%
#   calculate the meridional water vapor transport
#   select the level
uERA5_ver_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, :, :]
uERA5_ver_India_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, 8:28, 70:86]
uERA5_ver_EA_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, 36:42, 108:118]

vERA5_ver_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, :, :]
vERA5_ver_India_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, 8:28, 70:86]
vERA5_ver_EA_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, 36:42, 108:118]

qERA5_ver_JJA = ca.p_time(qERA5, 6, 8, True).loc[:, 100.0:, :, :]
qERA5_ver_India_JJA = ca.p_time(qERA5, 6, 8, True).loc[:, 100.0:, 8:28, 70:86]
qERA5_ver_EA_JJA = ca.p_time(qERA5, 6, 8, True).loc[:, 100.0:, 36:42, 108:118]

spERA5_ver_JJA = ca.p_time(spERA5, 6, 8, True).loc[:, :, :]
spERA5_ver_India_JJA = ca.p_time(spERA5, 6, 8, True).loc[:, 8:28, 70:86]
spERA5_ver_EA_JJA = ca.p_time(spERA5, 6, 8, True).loc[:, 36:42, 108:118]

uhis_ver_JJA = ca.p_time(uhis, 6, 8, True).loc[:, :10000.0, :, :]
uhis_ver_India_JJA = ca.p_time(uhis, 6, 8, True).loc[:, :10000.0, 8:28, 70:86]
uhis_ver_EA_JJA = ca.p_time(uhis, 6, 8, True).loc[:, :10000.0, 36:42, 108:118]

vhis_ver_JJA = ca.p_time(vhis, 6, 8, True).loc[:, :10000.0, :, :]
vhis_ver_India_JJA = ca.p_time(vhis, 6, 8, True).loc[:, :10000.0, 8:28, 70:86]
vhis_ver_EA_JJA = ca.p_time(vhis, 6, 8, True).loc[:, :10000.0, 36:42, 108:118]

qhis_ver_JJA = ca.p_time(qhis, 6, 8, True).loc[:, :10000.0, :, :]
qhis_ver_India_JJA = ca.p_time(qhis, 6, 8, True).loc[:, :10000.0, 8:28, 70:86]
qhis_ver_EA_JJA = ca.p_time(qhis, 6, 8, True).loc[:, :10000.0, 36:42, 108:118]

sphis_ver_JJA = ca.p_time(sphis, 6, 8, True).loc[:, :, :]
sphis_ver_India_JJA = ca.p_time(sphis, 6, 8, True).loc[:, 8:28, 70:86]
sphis_ver_EA_JJA = ca.p_time(sphis, 6, 8, True).loc[:, 36:42, 108:118]

preCRU_JJA = ca.p_time(preCRU, 6, 8, True)
preCRU_India_JJA = ca.p_time(preCRU.loc[:, 8:28, 70:86], 6, 8, True)
preCRU_EA_JJA = ca.p_time(preCRU.loc[:, 36:42, 108:118], 6, 8, True)

prehis_JJA = ca.p_time(prehis, 6, 8, True)
prehis_India_JJA = ca.p_time(prehis.loc[:, 8:28, 70:86], 6, 8, True)
prehis_EA_JJA = ca.p_time(prehis.loc[:, 36:42, 108:118], 6, 8, True)
# %%
#   calculate the area mean
uERA5_ver_India_JJA_mean = ca.cal_lat_weighted_mean(uERA5_ver_India_JJA).mean(dim="lon", skipna=True)
uERA5_ver_EA_JJA_mean = ca.cal_lat_weighted_mean(uERA5_ver_India_JJA).mean(dim="lon", skipna=True)

vERA5_ver_India_JJA_mean = ca.cal_lat_weighted_mean(vERA5_ver_India_JJA).mean(dim="lon", skipna=True)
vERA5_ver_EA_JJA_mean = ca.cal_lat_weighted_mean(vERA5_ver_India_JJA).mean(dim="lon", skipna=True)

qERA5_ver_India_JJA_mean = ca.cal_lat_weighted_mean(qERA5_ver_India_JJA).mean(dim="lon", skipna=True)
qERA5_ver_EA_JJA_mean = ca.cal_lat_weighted_mean(qERA5_ver_India_JJA).mean(dim="lon", skipna=True)

spERA5_ver_India_JJA_mean = ca.cal_lat_weighted_mean(spERA5_ver_India_JJA).mean(dim="lon", skipna=True)
spERA5_ver_EA_JJA_mean = ca.cal_lat_weighted_mean(spERA5_ver_India_JJA).mean(dim="lon", skipna=True)

uhis_ver_India_JJA_mean = ca.cal_lat_weighted_mean(uhis_ver_India_JJA).mean(dim="lon", skipna=True)
uhis_ver_EA_JJA_mean = ca.cal_lat_weighted_mean(uhis_ver_India_JJA).mean(dim="lon", skipna=True)

vhis_ver_India_JJA_mean = ca.cal_lat_weighted_mean(vhis_ver_India_JJA).mean(dim="lon", skipna=True)
vhis_ver_EA_JJA_mean = ca.cal_lat_weighted_mean(vhis_ver_India_JJA).mean(dim="lon", skipna=True)

qhis_ver_India_JJA_mean = ca.cal_lat_weighted_mean(qhis_ver_India_JJA).mean(dim="lon", skipna=True)
qhis_ver_EA_JJA_mean = ca.cal_lat_weighted_mean(qhis_ver_India_JJA).mean(dim="lon", skipna=True)

sphis_ver_India_JJA_mean = ca.cal_lat_weighted_mean(sphis_ver_India_JJA).mean(dim="lon", skipna=True)
sphis_ver_EA_JJA_mean = ca.cal_lat_weighted_mean(sphis_ver_India_JJA).mean(dim="lon", skipna=True)

preCRU_India_mean = ca.cal_lat_weighted_mean(preCRU_India_JJA).mean(
    dim="lon", skipna=True
)
preCRU_EA_mean = ca.cal_lat_weighted_mean(preCRU_EA_JJA).mean(dim="lon", skipna=True)
prehis_India_mean = ca.cal_lat_weighted_mean(prehis_India_JJA).mean(
    dim="lon", skipna=True
)
prehis_EA_mean = ca.cal_lat_weighted_mean(prehis_EA_JJA).mean(dim="lon", skipna=True)

# %%
#   calculate the waver vapor vertical intergration
ptop = 100*100
g = 9.8
ERA5level = qERA5_ver_JJA.coords["level"] * 100.0
ERA5level.attrs["units"] = "Pa"
ERA5dp = geocat.comp.dpres_plevel(ERA5level, spERA5_ver_JJA, ptop)
ERA5dpg = ERA5dp/g
ERA5dpg.attrs["units"] = "kg/m2"
# calculate the water vapor transport
uq_ERA5 = uERA5_ver_JJA * qERA5_ver_JJA * 1000.0
vq_ERA5 = vERA5_ver_JJA * qERA5_ver_JJA * 1000.0
uq_ERA5.attrs["units"] = "[m/s][g/kg]"
vq_ERA5.attrs["units"] = "[m/s][g/kg]"
# calculate the whole levels water vapor transport
uq_dpg_ERA5 = (uq_ERA5 * ERA5dpg.data).sum(dim="level")
vq_dpg_ERA5 = (vq_ERA5 * ERA5dpg.data).sum(dim="level")
uq_dpg_ERA5.attrs["units"] = "[m/s][g/kg]"
vq_dpg_ERA5.attrs["units"] = "[m/s][g/kg]"


hislevel = qhis_ver_JJA.coords["plev"] * 100.0
hislevel.attrs["units"] = "Pa"
hisdp = geocat.comp.dpres_plevel(hislevel, sphis_ver_JJA, ptop)
hisdpg = hisdp/g
hisdpg.attrs["units"] = "kg/m2"
# calculate the water vapor transport
uq_his = uhis_ver_JJA * qhis_ver_JJA * 1000.0
vq_his = vhis_ver_JJA * qhis_ver_JJA * 1000.0
uq_his.attrs["units"] = "[m/s][g/kg]"
vq_his.attrs["units"] = "[m/s][g/kg]"
# calculate the whole levels water vapor transport
uq_dpg_his = (uq_his * hisdpg.data).sum(dim="plev")
vq_dpg_his = (vq_his * hisdpg.data).sum(dim="plev")
uq_dpg_his.attrs["units"] = "[m/s][g/kg]"
vq_dpg_his.attrs["units"] = "[m/s][g/kg]"
# %%
#   calculate the correlation of India precipitation and meridional water vapor transport
#   calculate the area mean of meridional water vapor transport
uq_dpg_ERA5_India = uq_dpg_ERA5.loc[:, 8:28, 70:86]
vq_dpg_ERA5_India = vq_dpg_ERA5.loc[:, 8:28, 70:86]

uq_dpg_ERA5_EA_mean = ca.cal_lat_weighted_mean(uq_dpg_ERA5_India).mean(
    dim="lon", skipna=True
)
vq_dpg_ERA5_EA_mean = ca.cal_lat_weighted_mean(vq_dpg_ERA5_India).mean(
    dim="lon", skipna=True
)

uq_dpg_his_India = uq_dpg_his.loc[:, 8:28, 70:86]
vq_dpg_his_India = vq_dpg_his.loc[:, 8:28, 70:86]

uq_dpg_his_India_mean = ca.cal_lat_weighted_mean(uq_dpg_his_India).mean(
    dim="lon", skipna=True
)
vq_dpg_his_India_mean = ca.cal_lat_weighted_mean(vq_dpg_his_India).mean(
    dim="lon", skipna=True
)

# %%
preCRU_India_mean.coords['time'] = uq_dpg_ERA5.coords['time']
(
    uq_CRU_India_slope,
    uq_CRU_India_intercept,
    uq_CRU_India_rvalue,
    uq_CRU_India_pvalue,
    uq_CRU_India_hypothesis,
) = ca.dim_linregress(preCRU_India_mean, uq_dpg_ERA5)

(
    vq_CRU_India_slope,
    vq_CRU_India_intercept,
    vq_CRU_India_rvalue,
    vq_CRU_India_pvalue,
    vq_CRU_India_hypothesis,
) = ca.dim_linregress(preCRU_India_mean, vq_dpg_ERA5)

(
    uq_his_India_slope,
    uq_his_India_intercept,
    uq_his_India_rvalue,
    uq_his_India_pvalue,
    uq_his_India_hypothesis,
) = ca.dim_linregress(prehis_India_mean, uq_dpg_his)

(
    vq_his_India_slope,
    vq_his_India_intercept,
    vq_his_India_rvalue,
    vq_his_India_pvalue,
    vq_his_India_hypothesis,
) = ca.dim_linregress(prehis_India_mean, vq_dpg_his)

# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig_rvalue = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig_rvalue.subplots(ncols=2, nrows=2, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.arange(50, 151, 10)  # 设置纬度刻度
yticks = np.arange(10, 51, 10)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], 5, 55]
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
n = 1
# ==========================
con = axs[0, 0].contourf(
    uq_CRU_India_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    uq_CRU_India_pvalue,
    axs[0, 0],
    n,
    np.where(uq_CRU_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
# axs[0,0].contour(
#     CRU_India_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[0, 0].format(
    title="Uq reg IndR", rtitle="1950-2014", ltitle="CRU TS4.01",
)
sepl.patches(axs[0, 0], 70.0, 8.0, 16.0, 20.0, proj)
sepl.patches(axs[0, 0], 108, 36, 10.0, 6.0, proj)
# ==========================
con = axs[0, 1].contourf(
    vq_CRU_India_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    vq_CRU_India_pvalue,
    axs[0, 1],
    n,
    np.where(vq_CRU_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
# axs[0,0].contour(
#     CRU_India_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[0, 1].format(
    title="Vq reg IndR", rtitle="1950-2014", ltitle="CRU TS4.01",
)
sepl.patches(axs[0, 1], 70.0, 8.0, 16.0, 20.0, proj)
sepl.patches(axs[0, 1], 108, 36, 10.0, 6.0, proj)
# ==========================
con = axs[1, 0].contourf(
    uq_his_India_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    uq_his_India_pvalue,
    axs[1, 0],
    n,
    np.where(uq_his_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
# axs[0,0].contour(
#     his_India_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[1, 0].format(
    title="Uq reg IndR", rtitle="1950-2014", ltitle="historical",
)
sepl.patches(axs[1, 0], 70.0, 8.0, 16.0, 20.0, proj)
sepl.patches(axs[1, 0], 108, 36, 10.0, 6.0, proj)
# ==========================
con = axs[1, 1].contourf(
    vq_his_India_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    vq_his_India_pvalue,
    axs[1, 1],
    n,
    np.where(vq_his_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
# axs[0,0].contour(
#     his_India_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[1, 1].format(
    title="Vq reg IndR", rtitle="1950-2014", ltitle="historical",
)
sepl.patches(axs[1, 1], 70.0, 8.0, 16.0, 20.0, proj)
sepl.patches(axs[1, 1], 108, 36, 10.0, 6.0, proj)
fig_rvalue.colorbar(con, loc="b", width=0.13, length=0.7, label="")
# %%
preCRU_EA_mean.coords['time'] = uq_dpg_ERA5.coords['time']
(
    uq_CRU_EA_slope,
    uq_CRU_EA_intercept,
    uq_CRU_EA_rvalue,
    uq_CRU_EA_pvalue,
    uq_CRU_EA_hypothesis,
) = ca.dim_linregress(preCRU_EA_mean, uq_dpg_ERA5)

(
    vq_CRU_EA_slope,
    vq_CRU_EA_intercept,
    vq_CRU_EA_rvalue,
    vq_CRU_EA_pvalue,
    vq_CRU_EA_hypothesis,
) = ca.dim_linregress(preCRU_EA_mean, vq_dpg_ERA5)

(
    uq_his_EA_slope,
    uq_his_EA_intercept,
    uq_his_EA_rvalue,
    uq_his_EA_pvalue,
    uq_his_EA_hypothesis,
) = ca.dim_linregress(prehis_EA_mean, uq_dpg_his)

(
    vq_his_EA_slope,
    vq_his_EA_intercept,
    vq_his_EA_rvalue,
    vq_his_EA_pvalue,
    vq_his_EA_hypothesis,
) = ca.dim_linregress(prehis_EA_mean, vq_dpg_his)
# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig_rvalue = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig_rvalue.subplots(ncols=2, nrows=2, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.arange(50, 151, 10)  # 设置纬度刻度
yticks = np.arange(10, 51, 10)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], 5, 55]
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
n = 1
# ==========================
con = axs[0, 0].contourf(
    uq_CRU_EA_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    uq_CRU_EA_pvalue,
    axs[0, 0],
    n,
    np.where(uq_CRU_EA_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
# axs[0,0].contour(
#     CRU_EA_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[0, 0].format(
    title="Uq reg NCR", rtitle="1950-2014", ltitle="CRU TS4.01",
)
sepl.patches(axs[0, 0], 70.0, 8.0, 16.0, 20.0, proj)
sepl.patches(axs[0, 0], 108, 36, 10.0, 6.0, proj)
# ==========================
con = axs[0, 1].contourf(
    vq_CRU_EA_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    vq_CRU_EA_pvalue,
    axs[0, 1],
    n,
    np.where(vq_CRU_EA_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
# axs[0,0].contour(
#     CRU_EA_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[0, 1].format(
    title="Vq reg NCR", rtitle="1950-2014", ltitle="CRU TS4.01",
)
sepl.patches(axs[0, 1], 70.0, 8.0, 16.0, 20.0, proj)
sepl.patches(axs[0, 1], 108, 36, 10.0, 6.0, proj)
# ==========================
con = axs[1, 0].contourf(
    uq_his_EA_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    uq_his_EA_pvalue,
    axs[1, 0],
    n,
    np.where(uq_his_EA_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
# axs[0,0].contour(
#     his_EA_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[1, 0].format(
    title="Uq reg NCR", rtitle="1950-2014", ltitle="historical",
)
sepl.patches(axs[1, 0], 70.0, 8.0, 16.0, 20.0, proj)
sepl.patches(axs[1, 0], 108, 36, 10.0, 6.0, proj)
# ==========================
con = axs[1, 1].contourf(
    vq_his_EA_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    vq_his_EA_pvalue,
    axs[1, 1],
    n,
    np.where(vq_his_EA_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
# axs[0,0].contour(
#     his_India_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[1, 1].format(
    title="Vq reg NCR", rtitle="1950-2014", ltitle="historical",
)
sepl.patches(axs[1, 1], 70.0, 8.0, 16.0, 20.0, proj)
sepl.patches(axs[1, 1], 108, 36, 10.0, 6.0, proj)
fig_rvalue.colorbar(con, loc="b", width=0.13, length=0.7, label="")
# %%
#   calculate the rolling correlation coefficient
#   calculate the area mean

uq_dpg_ERA5_EA = uq_dpg_ERA5.loc[:, 32:38, 95:118]
vq_dpg_ERA5_EA = vq_dpg_ERA5.loc[:, 30:37.5, 112.5:120]

uq_dpg_ERA5_EA_mean = ca.cal_lat_weighted_mean(uq_dpg_ERA5_EA).mean(
    dim="lon", skipna=True
)
vq_dpg_ERA5_EA_mean = ca.cal_lat_weighted_mean(vq_dpg_ERA5_EA).mean(
    dim="lon", skipna=True
)

uq_dpg_his_EA = uq_dpg_his.loc[:, 32:38, 95:118]
vq_dpg_his_EA = vq_dpg_his.loc[:, 30:37.5, 112.5:120]

uq_dpg_his_EA_mean = ca.cal_lat_weighted_mean(uq_dpg_his_EA).mean(
    dim="lon", skipna=True
)
vq_dpg_his_EA_mean = ca.cal_lat_weighted_mean(vq_dpg_his_EA).mean(
    dim="lon", skipna=True
)
# %%
CRUtime = uq_dpg_ERA5.coords['time']
histime = uq_dpg_his.coords['time']
preCRU_India_mean.coords['time'] = CRUtime
freq = "AS-JUL"
window = 9

uq_CRU_India_rolling_9 = ca.rolling_reg_index(preCRU_India_mean, uq_dpg_ERA5_EA_mean, CRUtime, window, freq, True)

vq_CRU_India_rolling_9 = ca.rolling_reg_index(preCRU_India_mean, vq_dpg_ERA5_EA_mean, CRUtime, window, freq, True)

uq_his_India_rolling_9 = ca.rolling_reg_index(prehis_India_mean, uq_dpg_his_EA_mean, histime, window, freq, True)

vq_his_India_rolling_9 = ca.rolling_reg_index(prehis_India_mean, vq_dpg_his_EA_mean, histime, window, freq, True)

# %%
window = 9
CRU_India_EA_regress_9 = ca.rolling_reg_index(
    preCRU_India_mean, preCRU_EA_mean, CRUtime, window, freq, True
)
his_India_EA_regress_9 = ca.rolling_reg_index(
    prehis_India_mean, prehis_EA_mean, histime, window, freq, True
)
# %%
#   plot the rolling correlation coefficient
fig = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
axs = fig.subplots(ncols=1, nrows=2)
lw = 0.8
# cycle = pplt.Cycle('Pastel1', 'Pastel2', 27, left=0.1)

m1 = axs[0].line(
    CRUtime,
    np.array(uq_CRU_India_rolling_9["rvalue"]),
    lw=lw,
    color="blue",
)
m3 = axs[0].line(
    histime,
    np.array(uq_his_India_rolling_9["rvalue"]),
    lw=lw,
    color="red",
)
axs[0].line(
    CRUtime,
    np.array(CRU_India_EA_regress_9["rvalue"]),
    lw=lw,
    color="grey6",
    linestyle="--",
)

axs[0].axhline(0, lw = 0.8, color="grey5", linestyle="--")
axs[0].axhline(0.6021, lw = 0.8, color="grey5", linestyle="--")
axs[0].axhline(-0.6021, lw = 0.8, color="grey5", linestyle="--")
axs[0].format(ltitle="window=9", rtitle="1950-2014", title="Uq reg IndR", xrotation=0, ymin=-1.0, ymax=1.0, ylocator=0.2, yminorlocator=0.1)
axs[0].legend(handles=[m1,m3], loc="ll", labels=["CRU", "historical"], ncols=1)
# ======================
m1 = axs[1].line(
    CRUtime,
    np.array(vq_CRU_India_rolling_9["rvalue"]),
    lw=lw,
    color="blue",
)
m3 = axs[1].line(
    histime,
    np.array(vq_his_India_rolling_9["rvalue"]),
    lw=lw,
    color="red",
)
axs[1].line(
    CRUtime,
    np.array(CRU_India_EA_regress_9["rvalue"]),
    lw=lw,
    color="grey6",
    linestyle="--",
)

axs[1].axhline(0, lw = 0.8, color="grey5", linestyle="--")
axs[1].axhline(0.6021, lw = 0.8, color="grey5", linestyle="--")
axs[1].axhline(-0.6021, lw = 0.8, color="grey5", linestyle="--")
axs[1].format(ltitle="window=9", rtitle="1950-2014", title="Vq reg IndR", xrotation=0, ymin=-1.0, ymax=1.0, ylocator=0.2, yminorlocator=0.1)
axs[1].legend(handles=[m1,m3], loc="ll", labels=["CRU", "historical"], ncols=1)
# %%
#   pick up the different year to do the component analysis
ERA5time = uERA5_ver_JJA.coords["time"]
uERA5_ver_JJA_p1 = uERA5_ver_JJA.sel(time=(uERA5_ver_JJA.time.dt.year < 1970))
uERA5_ver_JJA_p2 = uERA5_ver_JJA.sel(time=((uERA5_ver_JJA.time.dt.year >= 1970) & (uERA5_ver_JJA.time.dt.year < 1984)))
uERA5_ver_JJA_p3 = uERA5_ver_JJA.sel(time=(uERA5_ver_JJA.time.dt.year >= 1984))

vERA5_ver_JJA_p1 = vERA5_ver_JJA.sel(time=(vERA5_ver_JJA.time.dt.year < 1970))
vERA5_ver_JJA_p2 = vERA5_ver_JJA.sel(time=((vERA5_ver_JJA.time.dt.year >= 1970) & (vERA5_ver_JJA.time.dt.year < 1984)))
vERA5_ver_JJA_p3 = vERA5_ver_JJA.sel(time=(vERA5_ver_JJA.time.dt.year >= 1984))



# %%
