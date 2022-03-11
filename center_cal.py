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

hgt_ERA5_SAH_area = hgt_ERA5_JJA_200.loc[:, startlat-1.25:endlat+1.25, startlon:endlon]
u_ERA5_SAH_area = u_ERA5_JJA_200.loc[:, startlat-1.25:endlat+1.25, startlon:endlon]

hgt_his_SAH_area = hgt_his_JJA_200.loc[:, startlat-1.25:endlat+1.25, startlon:endlon]
u_his_SAH_area = u_his_JJA_200.loc[:, startlat-1.25:endlat+1.25, startlon:endlon]

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
    return indB - indA


SAHI_ERA5 = cal_SAHI(hgt_ERA5_JJA_200)
SAHI_his = cal_SAHI(hgt_his_JJA_200)

SAHI_ERA5_std = SAHI_ERA5.std(dim="time", skipna=True)
SAHI_his_std = SAHI_his.std(dim="time", skipna=True)

# %%
fig_SAHI = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
axs = fig_SAHI.subplots(ncols=1, nrows=1)
lw = 0.8
axs[0].line(SAHI_ERA5.time.dt.year, ca.standardize(SAHI_ERA5), lw=lw, color="black")
axs[0].line(SAHI_ERA5.time.dt.year, ca.standardize(SAHI_his), lw=lw, color="blue")

axs[0].format(
    ylim=(-3, 3),
    ylocator=1.0,
    yminorlocator=0.5,
    xminorlocator=2.0,
    xlabel="time",
    ylabel="SAHI",
    xrotation=0,
)
axs[0].legend(loc="ll", ncols=1, labels=["ERA5", "historical"])
axs[0].text(1952, 2.5, "SAHI ERA5 std : {:.2f}".format(np.array(SAHI_ERA5_std)), size=7)
axs[0].text(1952, 2.2, "SAHI his std : {:.2f}".format(np.array(SAHI_his_std)), size=7)
# %%
#   choose the eastern-type and western-type
eastern_year_ERA5 = SAHI_ERA5.time.dt.year.where(ca.standardize(SAHI_ERA5) >= 1.0, drop=True)
western_year_ERA5 = SAHI_ERA5.time.dt.year.where(ca.standardize(SAHI_ERA5) <= -1.0, drop=True)

eastern_year_his = SAHI_his.time.dt.year.where(ca.standardize(SAHI_his) >= 1.0, drop=True)
western_year_his = SAHI_his.time.dt.year.where(ca.standardize(SAHI_his) <= -1.0, drop=True)

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

# %%
reload(sepl)
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
sepl.add_shape(r"/home/ys17-23/chenhj/TP_shape/Qinghai-Tibet_Plateau.shp", axs, proj)

startrange = 12100
endrange = 12500
spacing = 80
levels = np.array([12100, 12180, 12260, 12340, 12420, 12500, 12540])
con_labels = levels
axs[0, 0].contour(
    hgt_cli_ERA5,
    levels=levels,
    extend="both",
    color="black",
    labels=True,
    labels_kw={"fontsize": 7, "levels": con_labels, "inline_spacing": 3},
    lw=0.8,
)
axs[0, 0].line(
    cli_ERA5_ridgelon, cli_ERA5_ridgelat, lw=0.8, color="grey7", linestyle="--"
)
axs[0, 0].format(ltitle="climatology", rtitle="ERA5")

axs[1, 0].contour(
    hgt_eastern_ERA5_mean,
    levels=levels,
    extend="both",
    color="black",
    labels=True,
    labels_kw={"fontsize": 7, "levels": con_labels, "inline_spacing": 3},
    lw=0.8,
)
axs[1, 0].line(
    eastern_ERA5_ridgelon, eastern_ERA5_ridgelat, lw=0.8, color="grey7", linestyle="--"
)
axs[1, 0].format(ltitle="eastern-type", rtitle="ERA5")

axs[2, 0].contour(
    hgt_western_ERA5_mean,
    levels=levels,
    extend="both",
    color="black",
    labels=True,
    labels_kw={"fontsize": 7, "levels": con_labels, "inline_spacing": 3},
    lw=0.8,
)
axs[2, 0].line(
    western_ERA5_ridgelon, western_ERA5_ridgelat, lw=0.8, color="grey7", linestyle="--"
)
axs[2, 0].format(ltitle="western-type", rtitle="ERA5")

levels = np.array(
    [12100, 12180, 12260, 12340, 12420, 12440, 12460, 12480, 12500, 12540]
)
con_labels = levels
axs[0, 1].contour(
    hgt_cli_his,
    levels=levels,
    extend="both",
    color="black",
    labels=True,
    labels_kw={"fontsize": 7, "levels": con_labels, "inline_spacing": 3},
    lw=0.8,
)
axs[0, 1].line(
    cli_his_ridgelon, cli_his_ridgelat, lw=0.8, color="grey7", linestyle="--"
)
axs[0, 1].format(ltitle="climatology", rtitle="historical")

axs[1, 1].contour(
    hgt_eastern_his_mean,
    levels=levels,
    extend="both",
    color="black",
    labels=True,
    labels_kw={"fontsize": 7, "levels": con_labels, "inline_spacing": 3},
    lw=0.8,
)
axs[1, 1].line(
    eastern_his_ridgelon, eastern_his_ridgelat, lw=0.8, color="grey7", linestyle="--"
)
axs[1, 1].format(ltitle="eastern-type", rtitle="historical")

axs[2, 1].contour(
    hgt_western_his_mean,
    levels=levels,
    extend="both",
    color="black",
    labels=True,
    labels_kw={"fontsize": 7, "levels": con_labels, "inline_spacing": 3},
    lw=0.8,
)
axs[2, 1].line(
    western_his_ridgelon, western_his_ridgelat, lw=0.8, color="grey7", linestyle="--"
)
axs[2, 1].format(ltitle="western-type", rtitle="historical")


fig_SAH.format(abcloc="l", abc="(a)")


# %%
#   calculate the distribution of std
hgt_ERA5_SAH_std = hgt_ERA5_SAH_area.std(dim = "time", skipna=True)
hgt_his_SAH_std = hgt_his_SAH_area.std(dim = "time", skipna=True)
print(hgt_ERA5_SAH_std)

# %%
#   plot the distribution of std field
reload(sepl)
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0
proj = pplt.PlateCarree(central_longitude=cl)
fig_std = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig_std.subplots(ncols=1, nrows=2, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.arange(startlon, endlon + 1, 10)
yticks = np.arange(startlat, endlat + 1, 5)
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 2.5)
std_levels = np.arange(4,41,2)
coord = 0.05
con = axs[0].contourf(hgt_ERA5_SAH_std, extend="both", cmap="Oranges", levels=std_levels, cmap_kw={'left':coord})
axs[1].contourf(hgt_his_SAH_std, extend="both", cmap="Oranges", levels=std_levels, cmap_kw={'left':coord})
axs[0].contour(hgt_cli_ERA5, levels=np.array([12500, 12540]), color="black", linestyle="--", lw=0.8, labels=False, extend="both")
axs[0].contour(
    hgt_cli_ERA5,
    levels=np.arange(12500, 12541, 20),
    extend="both",
    color="grey7",
    labels=False,
    lw=0.8,
    linestyle="--",
)
axs[1].contour(
    hgt_cli_his,
    levels=np.arange(12440, 12481, 20),
    extend="both",
    color="grey7",
    labels=False,
    lw=0.8,
    linestyle="--",
)
axs[0].format(ltitle="std", rtitle="ERA5")
axs[1].format(ltitle="std", rtitle="historical")

fig_std.colorbar(con, loc='b', width=0.13, length=0.7, label="")

# %%
#   calculate the I_NS index
def cal_SAHI_NS(da):
    areaA = da.loc[:, 27.5:32.5, 50:100]
    areaB = da.loc[:, 22.5:27.5, 50:100]
    weights = np.cos(np.deg2rad(areaA.lat))
    weights.name = "weights"
    indA = areaA.weighted(weights).mean(("lon", "lat"), skipna=True)
    indB = areaB.weighted(weights).mean(("lon", "lat"), skipna=True)
    return indA - indB

