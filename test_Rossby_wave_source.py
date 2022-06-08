'''
Author: ChenHJ
Date: 2022-06-05 17:39:45
LastEditors: ChenHJ
LastEditTime: 2022-06-07 23:42:22
FilePath: /chenhj/0302code/test_Rossby_wave_source.py
Aim: 
Mission: 
'''
#%%
from mailbox import _PartialFile
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
import skill_metrics as sm
from brokenaxes import brokenaxes

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
#%%
#   read the data from ERA5 and calculate bandpass filter
fvERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/vwind_mon_r144x72_195001-201412.nc")
vERA5 = fvERA5["v"]
vERA5_fil = ca.butterworth_filter(vERA5, 8, 2*12, 9*12, "bandpass")
vERA5_MAM = ca.p_time(vERA5_fil, 3, 5, True).sel(level=250.0)

fuERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/uwind_mon_r144x72_195001-201412.nc")
uERA5 = fuERA5["u"]
uERA5_fil = ca.butterworth_filter(uERA5, 8, 2*12, 9*12, "bandpass")
uERA5_MAM = ca.p_time(uERA5_fil, 3, 5, True).sel(level=250.0)
# %%
#   rearange the coordinate of vERA5_MAM into -180°-180°
uERA5_MAM = ca.filplonlat(uERA5_MAM)
vERA5_MAM = ca.filplonlat(vERA5_MAM)
EOF_area_N = 75.0
EOF_area_S = 35.0
EOF_area_E = 120.0
EOF_area_W = -80.0

lat = vERA5_MAM.coords["lat"]
lon = vERA5_MAM.coords["lon"]
lat_EOF_range = lat[(lat >= EOF_area_S) & (lat <= EOF_area_N)]
lon_EOF_range = lon[(lon >= EOF_area_W) & (lon <= EOF_area_E)]
vERA5_MAM_EOF_area = vERA5_MAM.sel(lat=lat_EOF_range, lon=lon_EOF_range)
vERA5_MAM_EOF, vERA5_MAM_pc1, vERA5_MAM_pcC = ca.eof_analyse(vERA5_MAM_EOF_area.data, lat_EOF_range.data, 1)
vERA5_MAM_EOF1 = vERA5_MAM_EOF[0,:,:]
vERA5_MAM_pc1 = np.squeeze(vERA5_MAM_pc1,axis=1)
# %%
#   calculate the absolute vorticity and irrotational(divergent) wind
#   absolute vorticity
abvor = mpcalc.absolute_vorticity(uERA5_MAM, vERA5_MAM).metpy.dequantify()
revor = mpcalc.vorticity(uERA5_MAM, vERA5_MAM).metpy.dequantify()
#   irrotational(divergence) wind
w = VectorWind(uERA5_MAM, vERA5_MAM)
udiv, vdiv = w.irrotationalcomponent()

udiv_bar = udiv.mean(dim="time", skipna=True)
vdiv_bar = vdiv.mean(dim="time", skipna=True)
abvor_bar = abvor.mean(dim="time", skipna=True)

pc1_udiv_slope, pc1_udiv_intercept, pc1_udiv_rvalue, pc1_udiv_pvalue, pc1_udiv_hypothesis = ca.dim_linregress(vERA5_MAM_pc1, udiv)
pc1_vdiv_slope, pc1_vdiv_intercept, pc1_vdiv_rvalue, pc1_vdiv_pvalue, pc1_vdiv_hypothesis = ca.dim_linregress(vERA5_MAM_pc1, vdiv)
pc1_revor_slope, pc1_revor_intercept, pc1_revor_rvalue, pc1_revor_pvalue, pc1_revor_hypothesis = ca.dim_linregress(vERA5_MAM_pc1, revor)

#   udiv and vdiv prime
udiv_prime = pc1_udiv_slope.copy()
vdiv_prime = pc1_vdiv_slope.copy()

#   vorticity prime
revor_prime = pc1_revor_slope.copy()

#   calculate the lineared Rossby wave source
# S = -1.0*mpcalc.divergence(udiv_prime*abvor_bar, vdiv_prime*abvor_bar) - 1.0*mpcalc.divergence(udiv_bar*revor_prime, vdiv_bar*revor_prime)

wS1 = VectorWind(udiv_prime*abvor_bar, vdiv_prime*abvor_bar)

wS2 = VectorWind(udiv_bar*revor_prime, vdiv_bar*revor_prime)

S = -wS1.divergence()-wS2.divergence()
# %%
#   plot the Rossby wave source
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
# plot_array = np.reshape(range(1, 9), (2, 4))
# plot_array[-1,-3:] = 0
axs = fig.subplots(ncols=1,nrows=1, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([-60, 0, 60, 120])  # 设置纬度刻度
yticks = np.array([30.0, 60.0])  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [-80.0, 150.0, 10.0, 80.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    S,
    cmap="ColdHot",
    levels=np.array([-3e-11,-2e-11,-1e-11,1e-11,2e-11,3e-11]),
    zorder=0.8,
    extend="both",
    nozero=True
)
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(suptitle="Rossby Wave Source")
# %%
