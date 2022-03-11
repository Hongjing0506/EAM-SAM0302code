"""
Author: ChenHJ
Date: 2022-03-09 17:17:59
LastEditors: ChenHJ
LastEditTime: 2022-03-10 20:51:12
FilePath: /chenhj/0302code/center_cal.py
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

# %%
#   read obs data
fhgt_ERA5 = xr.open_dataset(
    "/home/ys17-23/chenhj/SAM_EAM_data/obs/hgt_mon_r144x72_195001-201412.nc"
)
hgt_ERA5 = fhgt_ERA5["z"]

fu_ERA5 = xr.open_dataset(
    "/home/ys17-23/chenhj/SAM_EAM_data/obs/uwind_mon_r144x72_195001-201412.nc"
)
u_ERA5 = fu_ERA5["u"]

fhgt_his = xr.open_dataset(
    "/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/zg/zg_Amon_ensemble_historical_gn_195001-201412.nc"
)
hgt_his = fhgt_his["zg"]

fu_his = xr.open_dataset(
    "/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/ua/ua_Amon_ensemble_historical_gn_195001-201412.nc"
)
u_his = fu_his["ua"]

# %%
#   calculate JJA
hgt_ERA5_JJA = ca.p_time(hgt_ERA5, 6, 8, True)
u_ERA5_JJA = ca.p_time(u_ERA5, 6, 8, True)

hgt_his_JJA = ca.p_time(hgt_his, 6, 8, True)
u_his_JJA = ca.p_time(u_his, 6, 8, True)
print(hgt_his_JJA)

# %%
#   select the 200hPa data and calculate the center of SAH and detrend
reload(ca)
hgt_ERA5_JJA_200 = hgt_ERA5_JJA.sel(level=200.0)
u_ERA5_JJA_200 = u_ERA5_JJA.sel(level=200.0)

hgt_his_JJA_200 = hgt_his_JJA[:, 9, :, :]
u_his_JJA_200 = u_his_JJA[:, 9, :, :]
# hgt_his_JJA_200 = hgt_his_JJA.sel(plev=2e+04)
# u_his_JJA_200 = u_his_JJA.sel(plev=2e+04)

startlon = 20.0
endlon = 130.0
startlat = 0.0
endlat = 45.0

hgt_ERA5_SAH_area = hgt_ERA5_JJA_200.loc[:, startlat:endlat, startlon:endlon]
u_ERA5_SAH_area = u_ERA5_JJA_200.loc[:, startlat:endlat, startlon:endlon]

hgt_his_SAH_area = hgt_his_JJA_200.loc[:, startlat:endlat, startlon:endlon]
u_his_SAH_area = u_his_JJA_200.loc[:, startlat:endlat, startlon:endlon]

time = hgt_ERA5_SAH_area.coords["time"]
lon = hgt_ERA5_SAH_area.coords["lon"]
center_loc_ERA5 = np.zeros(len(lon))
center_loc_his = np.zeros(len(lon))

# calculate the ridge line of SAH
for t_ERA5, t_his in zip(hgt_ERA5_SAH_area.time, hgt_his_SAH_area.time):
    ridgelat_ERA5, ridgelon_ERA5 = ca.cal_ridge_line(u_ERA5_SAH_area.sel(time=t_ERA5))
    ridgelat_his, ridgelon_his = ca.cal_ridge_line(u_his_SAH_area.sel(time=t_his))
    center_loc_ERA5[
        hgt_ERA5_SAH_area.sel(time=t_ERA5, lat=ridgelat_ERA5, lon=ridgelon_ERA5).argmax(
            dim=["lat", "lon"]
        )["lon"]
    ] += 1
    center_loc_his[
        hgt_his_SAH_area.sel(time=t_his, lat=ridgelat_his, lon=ridgelon_his).argmax(
            dim=["lat", "lon"]
        )["lon"]
    ] += 1


frequency_ERA5 = xr.DataArray(center_loc_ERA5, coords=[lon], dims=["lon"])
frequency_his = xr.DataArray(center_loc_his, coords=[lon], dims=["lon"])

# %%
fig = pplt.figure(refwidth=4.0, refheight=2.5, span=False, share=False)
axs = fig.subplots(ncols=1, nrows=2)
axs[0].bar(frequency_ERA5, width=0.6, color="black")
axs[1].bar(frequency_his, width=0.6, color="black")

axs[0].format(
    xformatter="deglon",
    ylim=(0, 20),
    xlim=(startlon, endlon),
    ylocator=5,
    xminorlocator=2.5,
    xlabel="Longitude",
    ylabel="Frequency",
    title="ERA5",
)
axs[1].format(
    xformatter="deglon",
    ylim=(0, 65),
    xlim=(startlon, endlon),
    ylocator=5,
    xminorlocator=2.5,
    xlabel="Longitude",
    ylabel="Frequency",
    title="ens",
)

# %%
#   calculate the SAHI
def cal_SAHI(da):
    areaA = da.loc[:, 22.5:32.5, 55:75]
    areaB = da.loc[:, 22.5:32.5, 85:105]
    weights = np.cos(np.deg2rad(areaA.lat))
    weights.name = "weights"
    indA = areaA.weighted(weights).mean(("lon", "lat"), skipna=True)
    indB = areaB.weighted(weights).mean(("lon", "lat"), skipna=True)
    return ca.standardize(indB - indA)


SAHI_ERA5 = cal_SAHI(hgt_ERA5_JJA_200)
SAHI_his = cal_SAHI(hgt_his_JJA_200)

# %%
fig_SAHI = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
axs = fig_SAHI.subplots(ncols=1, nrows=1)
lw = 0.8
axs[0].line(SAHI_ERA5, lw=lw, color="black")
axs[0].line(SAHI_his, lw=lw, color="blue")

axs[0].format(
    ylim=(-3, 3),
    ylocator=1.0,
    yminorlocator=0.5,
    xlabel="time",
    ylabel="SAHI",
    xrotation=0,
)
axs[0].legend(loc="ll", ncols=1, labels=["ERA5", "historical"])
# %%
#   choose the eastern-type and western-type
eastern_year_ERA5 = SAHI_ERA5.time.dt.year.where(SAHI_ERA5 >= 1.0, drop=True)
western_year_ERA5 = SAHI_ERA5.time.dt.year.where(SAHI_ERA5 <= -1.0, drop=True)

eastern_year_his = SAHI_his.time.dt.year.where(SAHI_his >= 1.0, drop=True)
western_year_his = SAHI_his.time.dt.year.where(SAHI_his <= -1.0, drop=True)

# %%
#   calculate the climatology SAH and eastern and western-type SAH and their ridge line
#   calculate the climatology SAH
hgt_cli_ERA5 = hgt_ERA5_SAH_area.mean(dim="time", skipna=True)
hgt_cli_his = hgt_his_SAH_area.mean(dim="time", skipna=True)
u_cli_ERA5 = u_ERA5_SAH_area.mean(dim="time", skipna=True)
u_cli_his = u_his_SAH_area.mean(dim="time", skipna=True)

cli_ERA5_ridgelat, cli_ERA5_ridgelon = ca.cal_ridge_line(u_cli_ERA5)
cli_his_ridgelat, cli_his_ridgelon = ca.cal_ridge_line(u_cli_his)

hgt_eastern_ERA5 = ca.year_choose(np.array(eastern_year_ERA5), hgt_ERA5_SAH_area)
hgt_eastern_his = ca.year_choose(np.array(eastern_year_his), hgt_his_SAH_area)
u_eastern_ERA5 = ca.year_choose(np.array(eastern_year_ERA5), u_ERA5_SAH_area)
u_eastern_his = ca.year_choose(np.array(eastern_year_his), u_his_SAH_area)

hgt_western_ERA5 = ca.year_choose(np.array(western_year_ERA5), hgt_ERA5_SAH_area)
hgt_western_his = ca.year_choose(np.array(western_year_his), hgt_his_SAH_area)
u_western_ERA5 = ca.year_choose(np.array(western_year_ERA5), u_ERA5_SAH_area)
u_western_his = ca.year_choose(np.array(western_year_his), u_his_SAH_area)

hgt_eastern_ERA5_mean = hgt_eastern_ERA5.mean(dim="time", skipna=True)
hgt_western_ERA5_mean = hgt_western_ERA5.mean(dim="time", skipna=True)
u_eastern_ERA5_mean = u_eastern_ERA5.mean(dim="time", skipna=True)
u_western_ERA5_mean = u_western_ERA5.mean(dim="time", skipna=True)

hgt_eastern_his_mean = hgt_eastern_his.mean(dim="time", skipna=True)
hgt_western_his_mean = hgt_western_his.mean(dim="time", skipna=True)
u_eastern_his_mean = u_eastern_his.mean(dim="time", skipna=True)
u_western_his_mean = u_western_his.mean(dim="time", skipna=True)

eastern_ERA5_ridgelat, eastern_ERA5_ridgelon = ca.cal_ridge_line(u_eastern_ERA5_mean)
western_ERA5_ridgelat, western_ERA5_ridgelon = ca.cal_ridge_line(u_western_ERA5_mean)
eastern_his_ridgelat, eastern_his_ridgelon = ca.cal_ridge_line(u_eastern_his_mean)
western_his_ridgelat, western_his_ridgelon = ca.cal_ridge_line(u_western_his_mean)

# %%

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0
proj = pplt.PlateCarree(central_longitude=cl)
fig_SAH = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig_SAH.subplots(ncols=2, nrows=3, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.arange(startlon, endlon + 1, 10)
yticks = np.arange(startlat, endlat + 1, 5)
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 2.5)

startrange = 12100
endrange = 12580
spacing = 40
con = axs[0, 0].contour(
    cli_ERA5,
    values=np.arange(startrange, endrange + 0.5 * spacing, spacing),
    extend="both",
    color="black",
)
axs[0, 0].scatter(cli_ERA5_ridgelon, cli_ERA5_ridgelat)

# w, h = 0.12, 0.14
# for i, ax in enumerate(axs):
#     add_patches_for_index(ind, ax, cl)
#     rect = Rectangle(
#         (1 - w, 0),
#         w,
#         h,
#         transform=ax.transAxes,
#         fc="white",
#         ec="k",
#         lw=0.5,
#         zorder=1.1,
#     )
#     con = ax.contourf(
#         r_mon_hgt850[0, i, :, :],
#         cmap="ColdHot",
#         extend="both",
#         vmin=-1.0,
#         vmax=1.0,
#     )
#     n = 4
#     sepl.plt_sig(
#         r_mon_hgt850[0, i, :, :],
#         ax,
#         n,
#         np.where(
#             abs(r_mon_hgt850[0, i, ::n, ::n]) >= rlim_mon_hgt850[0, i, ::n, ::n]
#         ),
#         "red",
#         1.0,
#     )
#     ski = 6
#     ax.quiver(
#         r_mon_u850[0, i, ::ski, ::ski],
#         r_mon_v850[0, i, ::ski, ::ski],
#         zorder=1,
#         headwidth=2.6,
#         headlength=2.3,
#         headaxislength=2.3,
#         scale_units="xy",
#         scale=0.12,
#         pivot="mid",
#         color="grey5",
#     )
#     tmp_check = ca.wind_check(r_mon_u850, r_mon_v850, rlim_mon_u850, rlim_mon_v850)
#     m = ax.quiver(
#         r_mon_u850.where(tmp_check != 0)[0, i, ::ski, ::ski],
#         r_mon_v850.where(tmp_check != 0)[0, i, ::ski, ::ski],
#         zorder=1,
#         headwidth=2.6,
#         headlength=2.3,
#         headaxislength=2.3,
#         scale_units="xy",
#         scale=0.12,
#         pivot="mid",
#         color="black",
#     )
#     del tmp_check
#     ax.add_patch(rect)
#     qk = ax.quiverkey(
#         m,
#         X=1 - w / 2,
#         Y=0.7 * h,
#         U=0.5,
#         label="0.5",
#         labelpos="S",
#         labelsep=0.02,
#         fontproperties={"size": 5},
#         zorder=3.1,
#     )
#     ax.format(
#         ltitle=f"{ind} " + np.array(r_mon_hgt850.coords["x"])[0],
#         rtitle="hgt850 & U850 " + np.array(r_mon_hgt850.coords["y"])[i],
#         fontsize=8,
#     )

fig_SAH.format(abcloc="l", abc="(a)")


# %%


# %%

