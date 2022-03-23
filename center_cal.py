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
#   read obs data
fhgt_ERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/hgt_mon_r144x72_195001-201412.nc"
)
hgt_ERA5 = fhgt_ERA5["z"]
hgt_ERA5 = ca.detrend_dim(hgt_ERA5, "time", deg=1, demean=False)

fu_ERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/uwind_mon_r144x72_195001-201412.nc"
)
u_ERA5 = fu_ERA5["u"]
u_ERA5 = ca.detrend_dim(u_ERA5, "time", deg=1, demean=False)

fhgt_his = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/zg/zg_Amon_ensemble_historical_gn_195001-201412.nc"
)
hgt_his = fhgt_his["zg"]
hgt_his = ca.detrend_dim(hgt_his, "time", deg=1, demean=False)

fu_his = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/ua/ua_Amon_ensemble_historical_gn_195001-201412.nc"
)
u_his = fu_his["ua"]
u_his = ca.detrend_dim(u_his, "time", deg=1, demean=False)

# %%
#   calculate JJA and detrend
hgt_ERA5_JJA = ca.p_time(hgt_ERA5, 6, 8, True)
u_ERA5_JJA = ca.p_time(u_ERA5, 6, 8, True)

hgt_his_JJA = ca.p_time(hgt_his, 6, 8, True)
u_his_JJA = ca.p_time(u_his, 6, 8, True)

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

hgt_ERA5_SAH_area = hgt_ERA5_JJA_200.loc[
    :, startlat - 1.25 : endlat + 1.25, startlon:endlon
]
u_ERA5_SAH_area = u_ERA5_JJA_200.loc[
    :, startlat - 1.25 : endlat + 1.25, startlon:endlon
]

hgt_his_SAH_area = hgt_his_JJA_200.loc[
    :, startlat - 1.25 : endlat + 1.25, startlon:endlon
]
u_his_SAH_area = u_his_JJA_200.loc[:, startlat - 1.25 : endlat + 1.25, startlon:endlon]

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
    ylabel="SAHI EW",
    xrotation=0,
)
axs[0].legend(loc="ll", ncols=1, labels=["ERA5", "historical"])
axs[0].text(1952, 2.5, "SAHI ERA5 std : {:.2f}".format(np.array(SAHI_ERA5_std)), size=7)
axs[0].text(1952, 2.2, "SAHI his std : {:.2f}".format(np.array(SAHI_his_std)), size=7)
fig_SAHI.format(title="SAHI-EW")
# %%
#   choose the eastern-type and western-type
eastern_year_ERA5 = SAHI_ERA5.time.dt.year.where(
    ca.standardize(SAHI_ERA5) >= 1.0, drop=True
)
western_year_ERA5 = SAHI_ERA5.time.dt.year.where(
    ca.standardize(SAHI_ERA5) <= -1.0, drop=True
)

eastern_year_his = SAHI_his.time.dt.year.where(
    ca.standardize(SAHI_his) >= 1.0, drop=True
)
western_year_his = SAHI_his.time.dt.year.where(
    ca.standardize(SAHI_his) <= -1.0, drop=True
)

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
def patches(x0, y0, width, height, proj):
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x0, y0),
        width,
        height,
        fc="none",
        ec="grey5",
        linewidth=1.0,
        zorder=1.1,
        transform=proj,
        linestyle="--",
    )
    ax.add_patch(rect)


def add_patches_for_EW(ax):
    proj = pplt.PlateCarree(central_longitude=0)
    x0 = 55.0
    width = 20.0
    y0 = 22.5
    height = 10.0
    patches(x0, y0, width, height, proj)
    x0 = 85.0
    width = 20.0
    y0 = 22.5
    height = 10.0
    patches(x0, y0, width, height, proj)


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
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 2.5, extents)
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

for ax in axs:
    add_patches_for_EW(ax)

fig_SAH.format(abcloc="l", abc="(a)")


# %%
#   calculate the distribution of std
hgt_ERA5_SAH_std = hgt_ERA5_SAH_area.std(dim="time", skipna=True)
hgt_his_SAH_std = hgt_his_SAH_area.std(dim="time", skipna=True)
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
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 2.5, extents)
std_levels = np.arange(0, 41, 2)
coord = 0
con = axs[0].contourf(
    hgt_ERA5_SAH_std,
    extend="both",
    cmap="Oranges",
    levels=std_levels,
    cmap_kw={"left": coord},
)
axs[1].contourf(
    hgt_his_SAH_std,
    extend="both",
    cmap="Oranges",
    levels=std_levels,
    cmap_kw={"left": coord},
)

axs[0].contour(
    hgt_cli_ERA5,
    levels=np.arange(12500, 12541, 20),
    extend="both",
    color="grey7",
    labels=True,
    lw=0.8,
    linestyle="--",
)
axs[1].contour(
    hgt_cli_his,
    levels=np.arange(12440, 12481, 20),
    extend="both",
    color="grey7",
    labels=True,
    lw=0.8,
    linestyle="--",
)
axs[0].format(ltitle="std", rtitle="ERA5")
axs[1].format(ltitle="std", rtitle="historical")

fig_std.colorbar(con, loc="b", width=0.13, length=0.7, label="")

# %%
#   calculate the I_NS index
def cal_SAHI_NS(da):
    areaA = da.loc[:, 27.5:32.5, 50:100]
    areaB = da.loc[:, 22.5:27.5, 50:100]
    weightsA = np.cos(np.deg2rad(areaA.lat))
    weightsA.name = "weights"
    weightsB = np.cos(np.deg2rad(areaB.lat))
    weightsB.name = "weights"
    indA = areaA.weighted(weightsA).mean(("lon", "lat"), skipna=True)
    indB = areaB.weighted(weightsB).mean(("lon", "lat"), skipna=True)
    return indA - indB


SAHI_NS_ERA5 = cal_SAHI_NS(hgt_ERA5_JJA_200)
SAHI_NS_his = cal_SAHI_NS(hgt_his_JJA_200)


SAHI_NS_ERA5_std = SAHI_NS_ERA5.std(dim="time", skipna=True)
SAHI_NS_his_std = SAHI_NS_his.std(dim="time", skipna=True)

# %%
def add_patches_for_NS(ax):
    proj = pplt.PlateCarree(central_longitude=0)
    x0 = 50.0
    width = 50.0
    y0 = 27.5
    height = 5.0
    patches(x0, y0, width, height, proj)
    x0 = 50.0
    width = 50.0
    y0 = 22.5
    height = 5.0
    patches(x0, y0, width, height, proj)


# %%
fig_SAHI_NS = pplt.figure(refwidth=5.0, refheight=2.5, span=False, share=False)
axs = fig_SAHI_NS.subplots(ncols=1, nrows=1)
lw = 0.8
axs[0].line(
    SAHI_NS_ERA5.time.dt.year, ca.standardize(SAHI_NS_ERA5), lw=lw, color="black"
)
axs[0].line(SAHI_NS_ERA5.time.dt.year, ca.standardize(SAHI_NS_his), lw=lw, color="blue")

axs[0].format(
    ylim=(-3, 3),
    ylocator=1.0,
    yminorlocator=0.5,
    xminorlocator=2.0,
    xlabel="time",
    ylabel="SAHI NS",
    xrotation=0,
)
axs[0].legend(loc="ll", ncols=1, labels=["ERA5", "historical"])
axs[0].text(
    2000, 2.5, "SAHI ERA5 std : {:.2f}".format(np.array(SAHI_NS_ERA5_std)), size=7
)
axs[0].text(
    2000, 2.2, "SAHI his std : {:.2f}".format(np.array(SAHI_NS_his_std)), size=7
)
fig_SAHI_NS.format(title="SAHI-NS")
# %%
#   calculate the EOF of two hgt data

# hgt_ERA5_EOF_area = ca.standardize(
#     ca.detrend_dim(
#         hgt_ERA5_JJA_200.loc[:, 13.75:41.25, 25:130], "time", deg=1, demean=False
#     )
# )
# hgt_his_EOF_area = ca.standardize(
#     ca.detrend_dim(
#         hgt_his_JJA_200.loc[:, 13.75:41.25, 25:130], "time", deg=1, demean=False
#     )
# )

hgt_ERA5_EOF_area = ca.standardize(
        hgt_ERA5_JJA_200.loc[:, 13.75:41.25, 25:130]
)
hgt_his_EOF_area = ca.standardize(
        hgt_his_JJA_200.loc[:, 13.75:41.25, 25:130]
)

hgt_ERA5_EOFs, hgt_ERA5_PCs, hgt_ERA5_percentContrib = ca.eof_analys(
    np.array(hgt_ERA5_EOF_area), np.array(hgt_ERA5_EOF_area.coords["lat"]), 3
)
hgt_ERA5_pattern = xr.DataArray(
    hgt_ERA5_EOFs[:3, :, :],
    coords=[
        hgt_ERA5_percentContrib[:3],
        hgt_ERA5_EOF_area.coords["lat"],
        hgt_ERA5_EOF_area.coords["lon"],
    ],
    dims=["pers", "lat", "lon"],
)


hgt_his_EOFs, hgt_his_PCs, hgt_his_percentContrib = ca.eof_analys(
    np.array(hgt_his_EOF_area), np.array(hgt_his_EOF_area.coords["lat"]), 3
)
hgt_his_pattern = xr.DataArray(
    hgt_his_EOFs[:3, :, :],
    coords=[
        hgt_his_percentContrib[:3],
        hgt_his_EOF_area.coords["lat"],
        hgt_his_EOF_area.coords["lon"],
    ],
    dims=["pers", "lat", "lon"],
)
print(hgt_ERA5_PCs[:, 0])

# %%
#   plot the EOFs result
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0
proj = pplt.PlateCarree(central_longitude=cl)
fig_ERA5_eof = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig_ERA5_eof.subplots(ncols=2, nrows=3, proj=[proj, None, proj, None, proj, None])

#   set the geo_ticks and map projection to the plots
xticks = np.arange(30, 120 + 1, 30)
yticks = np.arange(20, 40 + 1, 10)
extents = np.array([25, 130, 15, 40])
sepl.geo_ticks(axs[:, 0], xticks, yticks, cl, 10, 2, extents)

levels = np.arange(-1.0, 1.1, 0.2)

for i in np.arange(0, 3):
    con = axs[i, 0].contourf(
        hgt_ERA5_pattern[i, :, :], extend="both", cmap="ColdHot", levels=levels
    )
    axs[i, 0].format(
        ltitle="Pattern {}".format(i + 1),
        rtitle="{:.2f}%".format(hgt_ERA5_percentContrib[i]),
    )
    axs[i, 1].line(
        hgt_ERA5_EOF_area.time.dt.year, hgt_ERA5_PCs[:, i], lw=0.8, color="black"
    )
    axs[i, 1].format(ltitle="PC {}".format(i + 1))
    axs[i, 0].contour(
        hgt_cli_ERA5,
        levels=np.array([12500]),
        color="red",
        linestyle="--",
        lw=0.8,
        labels=True,
        extend="both",
    )


fig_ERA5_eof.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig_ERA5_eof.format(abcloc="l", abc="(a)")
# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0
proj = pplt.PlateCarree(central_longitude=cl)
fig_his_eof = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig_his_eof.subplots(ncols=2, nrows=3, proj=[proj, None, proj, None, proj, None])

#   set the geo_ticks and map projection to the plots
xticks = np.arange(30, 120 + 1, 30)
yticks = np.arange(20, 40 + 1, 10)
extents = np.array([25, 130, 15, 40])
sepl.geo_ticks(axs[:, 0], xticks, yticks, cl, 10, 2, extents)

levels = np.arange(-1.0, 1.1, 0.2)

for i in np.arange(0, 3):
    con = axs[i, 0].contourf(
        hgt_his_pattern[i, :, :], extend="both", cmap="ColdHot", levels=levels
    )
    axs[i, 0].format(
        ltitle="Pattern {}".format(i + 1),
        rtitle="{:.2f}%".format(hgt_his_percentContrib[i]),
    )
    axs[i, 1].line(
        hgt_his_EOF_area.time.dt.year, hgt_his_PCs[:, i], lw=0.8, color="black"
    )
    axs[i, 1].format(ltitle="PC {}".format(i + 1))
    axs[i, 0].contour(
        hgt_cli_his,
        levels=np.array([12460]),
        color="red",
        linestyle="--",
        lw=0.8,
        labels=True,
        extend="both",
    )


fig_his_eof.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig_his_eof.format(abcloc="l", abc="(a)")
# %%
#   read different models data
hgt_his_path = "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/zg"
g = os.walk(hgt_his_path)
filepath = []
modelname_hgt = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_hgt.append(filename[loc[1]+1:loc[2]])
hgt_ds_his = xr.open_mfdataset(filepath, concat_dim="models", combine='nested')
hgt_his_ds = xr.DataArray(hgt_ds_his['zg'])
hgt_his_ds.coords["models"] = modelname_hgt

u_his_path = "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/ua"
g = os.walk(u_his_path)
filepath = []
modelname_u = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_u.append(filename[loc[1]+1:loc[2]])
u_ds_his = xr.open_mfdataset(filepath, concat_dim="models", combine='nested')
u_his_ds = xr.DataArray(u_ds_his['ua'])
u_his_ds.coords["models"] = modelname_u

# %%
#   calculate JJA
hgt_his_ds_JJA = ca.p_time(hgt_his_ds, 6, 8, True)
u_his_ds_JJA = ca.p_time(u_his_ds, 6, 8, True)

# %%
hgt_his_ds_JJA_200 = hgt_his_ds_JJA.sel(plev=20000)
u_his_ds_JJA_200 = u_his_ds_JJA.sel(plev=20000)

# %%
hgt_his_ds_SAH_area = hgt_his_ds_JJA_200.loc[:, 
    :, startlat - 1.25 : endlat + 1.25, startlon:endlon
]
u_his_ds_SAH_area = u_his_ds_JJA_200.loc[:, 
    :, startlat - 1.25 : endlat + 1.25, startlon:endlon
]

# %%
center_loc_his_ds = np.zeros((26, len(lon)))
for num_model, imodel in enumerate(modelname_hgt):
    for t_his_ds in hgt_his_ds_SAH_area.time:
        ridgelat_his_ds, ridgelon_his_ds = ca.cal_ridge_line(u_his_ds_SAH_area.sel(time=t_his_ds, models=imodel))
        center_loc_his_ds[num_model, hgt_his_ds_SAH_area.sel(models=imodel, time=t_his_ds, lat=ridgelat_his_ds, lon=ridgelon_his_ds).argmax(dim=["lat", "lon"])["lon"]] += 1
    print(u_his_ds_SAH_area.sel(models=imodel), hgt_his_ds_SAH_area.sel(models=imodel))
    print(center_loc_his_ds[num_model,:])
# %%
#   plot the different models SAH center location distribution
fig_models = pplt.figure(refwidth=4.0, refheight=2.5, span=False, share=False)
array = np.arange(1,31)
array = array.reshape([5,6])
array[4,5] = 0
array[4,4] = 0
axs = fig_models.subplots(array)
axs.format(
    xformatter="deglon",
    ylim=(0, 65),
    xlim=(startlon, endlon),
    ylocator=5,
    xminorlocator=2.5,
    xlabel="Longitude",
    ylabel="Frequency",
)
axs[0].bar(frequency_ERA5, width=0.6, color="black")
axs[1].bar(frequency_his, width=0.6, color="black")

axs[0].format(
    ltitle="ERA5",
)
axs[1].format(
    ltitle="ens",
)
for num_model, imodel in enumerate(modelname_hgt):
    axs[num_model+2].bar(lon, center_loc_his_ds[num_model,:], width=0.6, color="black")
    axs[num_model+2].format(ltitle=imodel)
fig_models.format(abcloc="l", abc="(a)")
# %%
