'''
Author: ChenHJ
Date: 2022-03-13 10:26:30
LastEditors: ChenHJ
LastEditTime: 2022-03-14 16:18:38
FilePath: /chenhj/0302code/precip_reg.py
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
fpreCRU = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/obs/cru_ts4.01_r144x72_195001-201412.nc")
preCRU = fpreCRU["pre"]

# GPCP data just have 1979-2014 year
fpreGPCP = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/obs/GPCP_r144x72_197901-201412.nc")
preGPCP = fpreGPCP["precip"]

# %%
preCRU_JJA = ca.p_time(preCRU, 6, 8, True)

preCRU_India_JJA = ca.p_time(preCRU.loc[:, 8:28, 70:86], 6, 8, True)
preCRU_EA_JJA = ca.p_time(preCRU.loc[:, 36:42, 108:118], 6, 8, True)
preCRU_Japan_JJA = ca.p_time(preCRU.loc[:, 31:36, 130:140], 6, 8, True)

preGPCP_JJA = ca.p_time(preGPCP, 6, 8, True)

preGPCP_India_JJA = ca.p_time(preGPCP.loc[:, 8:28, 70:86], 6, 8, True)
preGPCP_EA_JJA = ca.p_time(preGPCP.loc[:, 36:42, 108:118], 6, 8, True)
preGPCP_Japan_JJA = ca.p_time(preGPCP.loc[:, 31:36, 130:140], 6, 8, True)
# %%
#   calculate area mean precipitation
preCRU_India_mean = ca.cal_lat_weighted_mean(preCRU_India_JJA).mean(dim="lon", skipna=True)
preCRU_EA_mean = ca.cal_lat_weighted_mean(preCRU_EA_JJA).mean(dim="lon", skipna=True)
preCRU_Japan_mean = ca.cal_lat_weighted_mean(preCRU_Japan_JJA).mean(dim="lon", skipna=True)

preGPCP_India_mean = ca.cal_lat_weighted_mean(preGPCP_India_JJA).mean(dim="lon", skipna=True)
preGPCP_EA_mean = ca.cal_lat_weighted_mean(preGPCP_EA_JJA).mean(dim="lon", skipna=True)
preGPCP_Japan_mean = ca.cal_lat_weighted_mean(preGPCP_Japan_JJA).mean(dim="lon", skipna=True)



# %%
#   calculate the regression distribution of the whole precipitation
CRU_India_slope, CRU_India_intercept, CRU_India_rvalue, CRU_India_pvalue, CRU_India_hypothesis = ca.dim_linregress(preCRU_India_mean, preCRU_JJA)

GPCP_India_slope, GPCP_India_intercept, GPCP_India_rvalue, GPCP_India_pvalue, GPCP_India_hypothesis = ca.dim_linregress(preGPCP_India_mean, preGPCP_JJA)

# %%
#   plot the rvalue distribution for different area precipitation
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0	#设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig_rvalue = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig_rvalue.subplots(ncols=2, nrows=3, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.arange(50, 151, 10)	#设置纬度刻度
yticks = np.arange(10, 51, 10)						#设置经度刻度
#设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
#当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], 5, 55]
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
n = 1

con = axs[0,0].contourf(
    CRU_India_rvalue,
    cmap = "ColdHot",
    levels=np.arange(-1.0, 1.1, 0.1),
    )
sepl.plt_sig(CRU_India_pvalue, axs[0,0], n, np.where(CRU_India_pvalue[::n, ::n] <= 0.05), "denim", 3.0)
# axs[0,0].contour(
#     CRU_India_pvalue,
#     color="black",
#     vmin=0.05,
#     vmax=0.05,
#     lw=0.8
# )
axs[0,0].format(
    title="Pr reg IndR",
    rtitle="1950-2014"
)
sepl.patches(axs[0,0], 70.0, 8.0, 16.0, 20.0, proj)

axs[0,1].contourf(
    GPCP_India_rvalue,
    cmap = "ColdHot",
    levels=np.arange(-1.0, 1.1, 0.1),
    )
# axs[0,1].contour(
#     GPCP_India_pvalue,
#     color="black",
#     levels=np.array([0.01, 0.05])
# )
sepl.plt_sig(GPCP_India_pvalue, axs[0,1], n, np.where(GPCP_India_pvalue[::n, ::n] <= 0.05), "denim", 3.0)
axs[0,1].format(
    title="Pr reg IndR",
    rtitle="1979-2014"
)
sepl.patches(axs[0,1], 70.0, 8.0, 16.0, 20.0, proj)

fig_rvalue.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig_rvalue.format(abc="(a)", abcloc="l")
# %%