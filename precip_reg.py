"""
Author: ChenHJ
Date: 2022-03-13 10:26:30
LastEditors: ChenHJ
LastEditTime: 2022-03-14 17:26:46
FilePath: /chenhj/0302code/precip_reg.py
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

# GPCP data just have 1979-2014 year
fpreGPCP = xr.open_dataset(
    "/home/ys17-23/chenhj/SAM_EAM_data/obs/GPCP_r144x72_197901-201412.nc"
)
preGPCP = fpreGPCP["precip"]

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


# %%
preCRU_JJA = ca.p_time(preCRU, 6, 8, True)

preCRU_India_JJA = ca.p_time(preCRU.loc[:, 8:28, 70:86], 6, 8, True)
preCRU_EA_JJA = ca.p_time(preCRU.loc[:, 36:42, 108:118], 6, 8, True)
preCRU_Japan_JJA = ca.p_time(preCRU.loc[:, 31:36, 130:140], 6, 8, True)
preCRU_SC_JJA = ca.p_time(preCRU.loc[:, 20:30, 100:120], 6, 8, True)

preGPCP_JJA = ca.p_time(preGPCP, 6, 8, True)

preGPCP_India_JJA = ca.p_time(preGPCP.loc[:, 8:28, 70:86], 6, 8, True)
preGPCP_EA_JJA = ca.p_time(preGPCP.loc[:, 36:42, 108:118], 6, 8, True)
preGPCP_Japan_JJA = ca.p_time(preGPCP.loc[:, 31:36, 130:140], 6, 8, True)
preGPCP_SC_JJA = ca.p_time(preGPCP.loc[:, 20:30, 100:120], 6, 8, True)


prehis_JJA = ca.p_time(prehis, 6, 8, True)

prehis_India_JJA = ca.p_time(prehis.loc[:, 8:28, 70:86], 6, 8, True)
prehis_EA_JJA = ca.p_time(prehis.loc[:, 36:42, 108:118], 6, 8, True)
prehis_Japan_JJA = ca.p_time(prehis.loc[:, 31:36, 130:140], 6, 8, True)
prehis_SC_JJA = ca.p_time(prehis.loc[:, 20:30, 100:120], 6, 8, True)

prehis_ds_JJA = ca.p_time(prehis_ds, 6, 8, True)
prehis_ds_India_JJA = ca.p_time(prehis_ds.loc[:, :, 8:28, 70:86], 6, 8, True)
prehis_ds_EA_JJA = ca.p_time(prehis_ds.loc[:, :, 36:42, 108:118], 6, 8, True)
prehis_ds_Japan_JJA = ca.p_time(prehis_ds.loc[:, :, 31:36, 130:140], 6, 8, True)
prehis_ds_SC_JJA = ca.p_time(prehis_ds.loc[:, :, 20:30, 100:120], 6, 8, True)
# %%
#   calculate area mean precipitation
preCRU_India_mean = ca.cal_lat_weighted_mean(preCRU_India_JJA).mean(
    dim="lon", skipna=True
)
preCRU_EA_mean = ca.cal_lat_weighted_mean(preCRU_EA_JJA).mean(dim="lon", skipna=True)
preCRU_Japan_mean = ca.cal_lat_weighted_mean(preCRU_Japan_JJA).mean(
    dim="lon", skipna=True
)
preCRU_SC_mean = ca.cal_lat_weighted_mean(preCRU_SC_JJA).mean(
    dim="lon", skipna=True
)

preGPCP_India_mean = ca.cal_lat_weighted_mean(preGPCP_India_JJA).mean(
    dim="lon", skipna=True
)
preGPCP_EA_mean = ca.cal_lat_weighted_mean(preGPCP_EA_JJA).mean(dim="lon", skipna=True)
preGPCP_Japan_mean = ca.cal_lat_weighted_mean(preGPCP_Japan_JJA).mean(
    dim="lon", skipna=True
)
preGPCP_SC_mean = ca.cal_lat_weighted_mean(preGPCP_SC_JJA).mean(
    dim="lon", skipna=True
)

prehis_India_mean = ca.cal_lat_weighted_mean(prehis_India_JJA).mean(
    dim="lon", skipna=True
)
prehis_EA_mean = ca.cal_lat_weighted_mean(prehis_EA_JJA).mean(dim="lon", skipna=True)
prehis_Japan_mean = ca.cal_lat_weighted_mean(prehis_Japan_JJA).mean(
    dim="lon", skipna=True
)
prehis_SC_mean = ca.cal_lat_weighted_mean(prehis_SC_JJA).mean(
    dim="lon", skipna=True
)

prehis_ds_India_mean = ca.cal_lat_weighted_mean(prehis_ds_India_JJA).mean(
    dim="lon", skipna=True
)
prehis_ds_EA_mean = ca.cal_lat_weighted_mean(prehis_ds_EA_JJA).mean(dim="lon", skipna=True)
prehis_ds_Japan_mean = ca.cal_lat_weighted_mean(prehis_ds_Japan_JJA).mean(
    dim="lon", skipna=True
)
prehis_ds_SC_mean = ca.cal_lat_weighted_mean(prehis_ds_SC_JJA).mean(
    dim="lon", skipna=True
)

# %%
#   calculate the regression distribution of the whole precipitation
(
    CRU_India_slope,
    CRU_India_intercept,
    CRU_India_rvalue,
    CRU_India_pvalue,
    CRU_India_hypothesis,
) = ca.dim_linregress(preCRU_India_mean, preCRU_JJA)

(
    CRU_EA_slope,
    CRU_EA_intercept,
    CRU_EA_rvalue,
    CRU_EA_pvalue,
    CRU_EA_hypothesis,
) = ca.dim_linregress(preCRU_EA_mean, preCRU_JJA)

(
    CRU_Japan_slope,
    CRU_Japan_intercept,
    CRU_Japan_rvalue,
    CRU_Japan_pvalue,
    CRU_Japan_hypothesis,
) = ca.dim_linregress(preCRU_Japan_mean, preCRU_JJA)

(
    GPCP_India_slope,
    GPCP_India_intercept,
    GPCP_India_rvalue,
    GPCP_India_pvalue,
    GPCP_India_hypothesis,
) = ca.dim_linregress(preGPCP_India_mean, preGPCP_JJA)

(
    GPCP_EA_slope,
    GPCP_EA_intercept,
    GPCP_EA_rvalue,
    GPCP_EA_pvalue,
    GPCP_EA_hypothesis,
) = ca.dim_linregress(preGPCP_EA_mean, preGPCP_JJA)

(
    GPCP_Japan_slope,
    GPCP_Japan_intercept,
    GPCP_Japan_rvalue,
    GPCP_Japan_pvalue,
    GPCP_Japan_hypothesis,
) = ca.dim_linregress(preGPCP_Japan_mean, preGPCP_JJA)

(
    his_India_slope,
    his_India_intercept,
    his_India_rvalue,
    his_India_pvalue,
    his_India_hypothesis,
) = ca.dim_linregress(prehis_India_mean, prehis_JJA)

(
    his_EA_slope,
    his_EA_intercept,
    his_EA_rvalue,
    his_EA_pvalue,
    his_EA_hypothesis,
) = ca.dim_linregress(prehis_EA_mean, prehis_JJA)

(
    his_Japan_slope,
    his_Japan_intercept,
    his_Japan_rvalue,
    his_Japan_pvalue,
    his_Japan_hypothesis,
) = ca.dim_linregress(prehis_Japan_mean, prehis_JJA)

# %%
#   plot the rvalue distribution for different area precipitation
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig_rvalue = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig_rvalue.subplots(ncols=3, nrows=3, proj=proj)

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
    CRU_India_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    CRU_India_pvalue,
    axs[0, 0],
    n,
    np.where(CRU_India_pvalue[::n, ::n] <= 0.05),
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
    title="Pr reg IndR", rtitle="1950-2014", ltitle="CRU TS4.01",
)
sepl.patches(axs[0, 0], 70.0, 8.0, 16.0, 20.0, proj)
# ==========================
axs[1, 0].contourf(
    CRU_EA_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    CRU_EA_pvalue, axs[1, 0], n, np.where(CRU_EA_pvalue[::n, ::n] <= 0.05), "denim", 3.0
)
# axs[0,0].contour(
#     CRU_India_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[1, 0].format(
    title="Pr reg NCR", rtitle="1950-2014", ltitle="CRU TS4.01",
)
sepl.patches(axs[1, 0], 108, 36, 10.0, 6.0, proj)
# ==========================
axs[2, 0].contourf(
    CRU_Japan_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
sepl.plt_sig(
    CRU_Japan_pvalue,
    axs[2, 0],
    n,
    np.where(CRU_Japan_pvalue[::n, ::n] <= 0.05),
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
axs[2, 0].format(
    title="Pr reg SJR", rtitle="1950-2014", ltitle="CRU TS4.01",
)
sepl.patches(axs[2, 0], 130, 31, 10.0, 5.0, proj)
# ==========================
axs[0, 1].contourf(
    GPCP_India_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
# axs[0,1].contour(
#     GPCP_India_pvalue,
#     color="black",
#     levels=np.array([0.01, 0.05])
# )
sepl.plt_sig(
    GPCP_India_pvalue,
    axs[0, 1],
    n,
    np.where(GPCP_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[0, 1].format(
    title="Pr reg IndR", rtitle="1979-2014", ltitle="GPCP",
)
sepl.patches(axs[0, 1], 70.0, 8.0, 16.0, 20.0, proj)
# ==========================
axs[1, 1].contourf(
    GPCP_EA_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
# axs[0,1].contour(
#     GPCP_India_pvalue,
#     color="black",
#     levels=np.array([0.01, 0.05])
# )
sepl.plt_sig(
    GPCP_EA_pvalue,
    axs[1, 1],
    n,
    np.where(GPCP_EA_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[1, 1].format(
    title="Pr reg NCR", rtitle="1979-2014", ltitle="GPCP",
)
sepl.patches(axs[1, 1], 108, 36, 10.0, 6.0, proj)
# ==========================
axs[2, 1].contourf(
    GPCP_Japan_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
# axs[0,1].contour(
#     GPCP_India_pvalue,
#     color="black",
#     levels=np.array([0.01, 0.05])
# )
sepl.plt_sig(
    GPCP_Japan_pvalue,
    axs[2, 1],
    n,
    np.where(GPCP_Japan_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[2, 1].format(
    title="Pr reg SJR", rtitle="1979-2014", ltitle="GPCP",
)
sepl.patches(axs[2, 1], 130, 31, 10.0, 5.0, proj)
# ==========================
axs[0, 2].contourf(
    his_India_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
# axs[0,2].contour(
#     his_India_pvalue,
#     color="black",
#     levels=np.array([0.01, 0.05])
# )
sepl.plt_sig(
    his_India_pvalue,
    axs[0, 2],
    n,
    np.where(his_India_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[0, 2].format(
    title="Pr reg IndR", rtitle="1950-2014", ltitle="historical",
)
sepl.patches(axs[0, 2], 70.0, 8.0, 16.0, 20.0, proj)
# ==========================
axs[1, 2].contourf(
    his_EA_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
# axs[0,2].contour(
#     his_India_pvalue,
#     color="black",
#     levels=np.array([0.01, 0.05])
# )
sepl.plt_sig(
    his_EA_pvalue, axs[1, 2], n, np.where(his_EA_pvalue[::n, ::n] <= 0.05), "denim", 3.0
)
axs[1, 2].format(
    title="Pr reg NCR", rtitle="1950-2014", ltitle="historical",
)
sepl.patches(axs[1, 2], 108, 36, 10.0, 6.0, proj)
# ==========================
axs[2, 2].contourf(
    his_Japan_rvalue, cmap="ColdHot", levels=np.arange(-1.0, 1.1, 0.1),
)
# axs[0,2].contour(
#     his_India_pvalue,
#     color="black",
#     levels=np.array([0.01, 0.05])
# )
sepl.plt_sig(
    his_Japan_pvalue,
    axs[2, 2],
    n,
    np.where(his_Japan_pvalue[::n, ::n] <= 0.05),
    "denim",
    3.0,
)
axs[2, 2].format(
    title="Pr reg SJR", rtitle="1950-2014", ltitle="historical",
)
sepl.patches(axs[2, 2], 130, 31, 10.0, 5.0, proj)

fig_rvalue.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig_rvalue.format(abc="(a)", abcloc="l")
# %%
#   calculate the rolling correlation
#   India precipitation & Northern China precipitation area mean
reload(ca)
freq = "AS-JUL"
window = 7
CRUtime = preCRU_India_mean.coords["time"]
GPCPtime = preGPCP_India_mean.coords["time"]
histime = prehis_India_mean.coords["time"]
his_dstime = prehis_ds_India_mean.coords["time"]


CRU_India_EA_regress_7 = ca.rolling_reg_index(
    preCRU_India_mean, preCRU_EA_mean, CRUtime, window, freq, True
)
GPCP_India_EA_regress_7 = ca.rolling_reg_index(
    preGPCP_India_mean, preGPCP_EA_mean, GPCPtime, window, freq, True
)
his_India_EA_regress_7 = ca.rolling_reg_index(
    prehis_India_mean, prehis_EA_mean, histime, window, freq, True
)

his_ds_India_EA_avalue_7, his_ds_India_EA_bvalue_7, his_ds_India_EA_rvalue_7, his_ds_India_EA_pvalue_7, his_ds_India_EA_hyvalue_7 = ca.rolling_regression_pattern(
    prehis_ds_India_mean, prehis_ds_EA_mean, his_dstime, window, freq
)

window = 9
CRU_India_EA_regress_9 = ca.rolling_reg_index(
    preCRU_India_mean, preCRU_EA_mean, CRUtime, window, freq, True
)
GPCP_India_EA_regress_9 = ca.rolling_reg_index(
    preGPCP_India_mean, preGPCP_EA_mean, GPCPtime, window, freq, True
)
his_India_EA_regress_9 = ca.rolling_reg_index(
    prehis_India_mean, prehis_EA_mean, histime, window, freq, True
)
his_ds_India_EA_avalue_9, his_ds_India_EA_bvalue_9, his_ds_India_EA_rvalue_9, his_ds_India_EA_pvalue_9, his_ds_India_EA_hyvalue_9 = ca.rolling_regression_pattern(
    prehis_ds_India_mean, prehis_ds_EA_mean, his_dstime, window, freq
)

window = 11
CRU_India_EA_regress_11 = ca.rolling_reg_index(
    preCRU_India_mean, preCRU_EA_mean, CRUtime, window, freq, True
)
GPCP_India_EA_regress_11 = ca.rolling_reg_index(
    preGPCP_India_mean, preGPCP_EA_mean, GPCPtime, window, freq, True
)
his_India_EA_regress_11 = ca.rolling_reg_index(
    prehis_India_mean, prehis_EA_mean, histime, window, freq, True
)
his_ds_India_EA_avalue_11, his_ds_India_EA_bvalue_11, his_ds_India_EA_rvalue_11, his_ds_India_EA_pvalue_11, his_ds_India_EA_hyvalue_11 = ca.rolling_regression_pattern(
    prehis_ds_India_mean, prehis_ds_EA_mean, his_dstime, window, freq
)


# %%


# %%
#   plot the rolling_reg_index
fig = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
axs = fig.subplots(ncols=1, nrows=3)
lw = 0.8
# cycle = pplt.Cycle('Pastel1', 'Pastel2', 27, left=0.1)
cycle = "Pastel1"

m1 = axs[0].line(
    CRU_India_EA_regress_7.time.dt.year,
    np.array(CRU_India_EA_regress_7["rvalue"]),
    lw=lw,
    color="blue",
)

# m2 = axs[0].line(
#     GPCP_India_EA_regress_7.time.dt.year,
#     np.array(GPCP_India_EA_regress_7["rvalue"]),
#     lw=lw,
#     color="black",
# )

m3 = axs[0].line(
    his_India_EA_regress_7.time.dt.year,
    np.array(his_India_EA_regress_7["rvalue"]),
    lw=lw,
    color="red",
)


axs[0].axhline(0, lw = 0.8, color="grey5", linestyle="--")
axs[0].axhline(0.6664, lw = 0.8, color="grey5", linestyle="--")
axs[0].axhline(-0.6664, lw = 0.8, color="grey5", linestyle="--")
axs[0].format(ltitle="window=7", rtitle="1950-2014")
axs[0].legend(handles=[m1,m3], loc="ll", labels=["CRU", "historical"], ncols=1)

m1 = axs[1].line(
    CRU_India_EA_regress_9.time.dt.year,
    np.array(CRU_India_EA_regress_9["rvalue"]),
    lw=lw,
    color="blue",
)
# m2 = axs[1].line(
#     GPCP_India_EA_regress_9.time.dt.year,
#     np.array(GPCP_India_EA_regress_9["rvalue"]),
#     lw=lw,
#     color="black",
# )
m3 = axs[1].line(
    his_India_EA_regress_9.time.dt.year,
    np.array(his_India_EA_regress_9["rvalue"]),
    lw=lw,
    color="red",
)
axs[1].format(ltitle="window=9", rtitle="1979-2014")
axs[1].legend(handles=[m1,m3], loc="ll", labels=["CRU", "historical"], ncols=1)
axs[1].axhline(0, lw = 0.8, color="grey5", linestyle="--")
axs[1].axhline(0.6021, lw = 0.8, color="grey5", linestyle="--")
axs[1].axhline(-0.6021, lw = 0.8, color="grey5", linestyle="--")

m1 = axs[2].line(
    CRU_India_EA_regress_11.time.dt.year,
    np.array(CRU_India_EA_regress_11["rvalue"]),
    lw=lw,
    color="blue",
)
# m2 = axs[2].line(
#     GPCP_India_EA_regress_11.time.dt.year,
#     np.array(GPCP_India_EA_regress_11["rvalue"]),
#     lw=lw,
#     color="black",
# )
m3 = axs[2].line(
    his_India_EA_regress_11.time.dt.year,
    np.array(his_India_EA_regress_11["rvalue"]),
    lw=lw,
    color="red",
)
axs[2].format(ltitle="window=11", rtitle="1950-2014")
axs[2].legend(handles=[m1,m3], loc="ll", labels=["CRU", "historical"], ncols=1)
axs[2].axhline(0, lw = 0.8, color="grey5", linestyle="--")
axs[2].axhline(0.5529, lw = 0.8, color="grey5", linestyle="--")
axs[2].axhline(-0.5529, lw = 0.8, color="grey5", linestyle="--")
axs.format(
    ylim=(-1.0, 1.0), ylocator=0.2, yminorlocator=0.1, xrotation=0, xlim=(1950, 2014),
)
fig.format(abc="(a)", abcloc="l")
# %%
#   calculate the rolling correlation pattern
reload(ca)
window = 11
freq = "AS-JUL"
(
    CRUreg_Ind_avalue,
    CRUreg_Ind_bvalue,
    CRUreg_Ind_rvalue,
    CRUreg_Ind_pvalue,
    CRUreg_Ind_hyvalue,
) = ca.rolling_regression_pattern(
    preCRU_India_mean,
    preCRU_JJA,
    np.array(preCRU_India_mean.coords["time"]),
    window,
    freq,
)

(
    GPCPreg_Ind_avalue,
    GPCPreg_Ind_bvalue,
    GPCPreg_Ind_rvalue,
    GPCPreg_Ind_pvalue,
    GPCPreg_Ind_hyvalue,
) = ca.rolling_regression_pattern(
    preGPCP_India_mean,
    preGPCP_JJA,
    np.array(preGPCP_India_mean.coords["time"]),
    window,
    freq,
)

(
    hisreg_Ind_avalue,
    hisreg_Ind_bvalue,
    hisreg_Ind_rvalue,
    hisreg_Ind_pvalue,
    hisreg_Ind_hyvalue,
) = ca.rolling_regression_pattern(
    prehis_India_mean,
    prehis_JJA,
    np.array(prehis_India_mean.coords["time"]),
    window,
    freq,
)

# %%
#   calculate the rolling correlation coefficient std distribution
CRUreg_Ind_rvalue_std = CRUreg_Ind_rvalue.std(dim="time", skipna=True)
GPCPreg_Ind_rvalue_std = GPCPreg_Ind_rvalue.std(dim="time", skipna=True)
hisreg_Ind_rvalue_std = hisreg_Ind_rvalue.std(dim="time", skipna=True)

# %%
#   plot the rolling correlation coefficient std distribution
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0	#设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig_std = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig_std.subplots(ncols=3, nrows=3, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.arange(0, 181, 30)	#设置纬度刻度
yticks = np.arange(-30, 61, 30)						#设置经度刻度
#设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
#当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)

con = axs[0,0].contourf(
    CRUreg_Ind_rvalue_std,
    levels=np.arange(0.3,0.71,0.05),
    cmap="Blues",
    extend="both"
    )
sepl.patches(axs[0,0], 108, 36, 10.0, 6.0, proj)

axs[0,1].contourf(
    GPCPreg_Ind_rvalue_std,
    levels=np.arange(0.3,0.71,0.05),
    cmap="Blues",
    extend="both"
    )
sepl.patches(axs[0,1], 108, 36, 10.0, 6.0, proj)

axs[0,2].contourf(
    hisreg_Ind_rvalue_std,
    levels=np.arange(0.3,0.71,0.05),
    cmap="Blues",
    extend="both"
    )
sepl.patches(axs[0,2], 108, 36, 10.0, 6.0, proj)

fig_std.colorbar(con, loc="b", width=0.13, length=0.7)
fig_std.format(abc="(a)", abcloc="l")
# %%
#   plot the rolling_reg_index
fig = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
array = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25],[26,27,28,29,0]]
axs = fig.subplots(array)
lw = 0.8
# cycle = pplt.Cycle('Pastel1', 'Pastel2', 27, left=0.1)
cycle = "Pastel1"

axs[0].line(
    CRU_India_EA_regress_7.time.dt.year,
    np.array(CRU_India_EA_regress_7["rvalue"]),
    lw=lw,
    color="black",
)
axs[0].format(title = "CRU TS4.01")

axs[1].line(
    GPCP_India_EA_regress_7.time.dt.year,
    np.array(GPCP_India_EA_regress_7["rvalue"]),
    lw=lw,
    color="black",
)
axs[1].format(title = "GPCP")

axs[2].line(
    his_India_EA_regress_7.time.dt.year,
    np.array(his_India_EA_regress_7["rvalue"]),
    lw=lw,
    color="black",
)
axs[2].format(title = "historical")

for i, ax in enumerate(axs[3:]):
    ax.line(his_ds_India_EA_rvalue_7.time.dt.year, his_ds_India_EA_rvalue_7[:,i], color="black", lw=lw)
    ax.format(title=np.array(his_ds_India_EA_rvalue_7.coords["models"][i]))

for ax in axs:
    ax.axhline(0, lw = 0.8, color="grey5", linestyle="--")
    ax.axhline(0.6664, lw = 0.8, color="grey5", linestyle="--")
    ax.axhline(-0.6664, lw = 0.8, color="grey5", linestyle="--")
    ax.axvline(1972, lw=0.8, color="grey5", linestyle="--")
    ax.format(ltitle="window=7", rtitle="1950-2014")

axs.format(
    ylim=(-1.0, 1.0), ylocator=0.2, yminorlocator=0.1, xrotation=0, xlim=(1950, 2014),
)
fig.format(abc="(a)", abcloc="l")
# %%

fig = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
array = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25],[26,27,28,29,0]]
axs = fig.subplots(array)
lw = 0.8
# cycle = pplt.Cycle('Pastel1', 'Pastel2', 27, left=0.1)
cycle = "Pastel1"

axs[0].line(
    CRU_India_EA_regress_9.time.dt.year,
    np.array(CRU_India_EA_regress_9["rvalue"]),
    lw=lw,
    color="black",
)
axs[0].format(title = "CRU TS4.01")

axs[1].line(
    GPCP_India_EA_regress_9.time.dt.year,
    np.array(GPCP_India_EA_regress_9["rvalue"]),
    lw=lw,
    color="black",
)
axs[1].format(title = "GPCP")

axs[2].line(
    his_India_EA_regress_9.time.dt.year,
    np.array(his_India_EA_regress_9["rvalue"]),
    lw=lw,
    color="black",
)
axs[2].format(title = "historical")

for i, ax in enumerate(axs[3:]):
    ax.line(his_ds_India_EA_rvalue_9.time.dt.year, his_ds_India_EA_rvalue_9[:,i], color="black", lw=lw)
    ax.format(title=np.array(his_ds_India_EA_rvalue_9.coords["models"][i]))

for ax in axs:
    ax.axhline(0, lw = 0.8, color="grey5", linestyle="--")
    ax.axhline(0.6021, lw = 0.8, color="grey5", linestyle="--")
    ax.axhline(-0.6021, lw = 0.8, color="grey5", linestyle="--")
    ax.axvline(1972, lw=0.8, color="grey5", linestyle="--")
    ax.format(ltitle="window=9", rtitle="1950-2014")

axs.format(
    ylim=(-1.0, 1.0), ylocator=0.2, yminorlocator=0.1, xrotation=0, xlim=(1950, 2014),
)
fig.format(abc="(a)", abcloc="l")


# %%
fig = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
array = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25],[26,27,28,29,0]]
axs = fig.subplots(array)
lw = 0.8
# cycle = pplt.Cycle('Pastel1', 'Pastel2', 27, left=0.1)
cycle = "Pastel1"

axs[0].line(
    CRU_India_EA_regress_11.time.dt.year,
    np.array(CRU_India_EA_regress_11["rvalue"]),
    lw=lw,
    color="black",
)
axs[0].format(title = "CRU TS4.01")

axs[1].line(
    GPCP_India_EA_regress_11.time.dt.year,
    np.array(GPCP_India_EA_regress_11["rvalue"]),
    lw=lw,
    color="black",
)
axs[1].format(title = "GPCP")

axs[2].line(
    his_India_EA_regress_11.time.dt.year,
    np.array(his_India_EA_regress_11["rvalue"]),
    lw=lw,
    color="black",
)
axs[2].format(title = "historical")

for i, ax in enumerate(axs[3:]):
    ax.line(his_ds_India_EA_rvalue_11.time.dt.year, his_ds_India_EA_rvalue_11[:,i], color="black", lw=lw)
    ax.format(title=np.array(his_ds_India_EA_rvalue_11.coords["models"][i]))

for ax in axs:
    ax.axhline(0, lw = 0.8, color="grey5", linestyle="--")
    ax.axhline(0.5529, lw = 0.8, color="grey5", linestyle="--")
    ax.axhline(-0.5529, lw = 0.8, color="grey5", linestyle="--")
    ax.axvline(1972, lw=0.8, color="grey5", linestyle="--")
    ax.format(ltitle="window=11", rtitle="1950-2014")

axs.format(
    ylim=(-1.0, 1.0), ylocator=0.2, yminorlocator=0.1, xrotation=0, xlim=(1950, 2014),
)
fig.format(abc="(a)", abcloc="l")
# %%
