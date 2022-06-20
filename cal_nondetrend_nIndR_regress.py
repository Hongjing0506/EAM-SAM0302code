'''
Author: ChenHJ
Date: 2022-05-25 16:39:12
LastEditors: ChenHJ
LastEditTime: 2022-06-20 13:24:04
FilePath: /chenhj/0302code/cal_nondetrend_nIndR_regress.py
Aim: 
Mission: 
'''
# %%
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
# %%
# read the data in AIR/GPCP/ERA5/historical/ssp585
fpreGPCP = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/GPCP_r144x72_197901-201412.nc"
)
preGPCP = fpreGPCP["precip"]
preGPCP_JJA = ca.p_time(preGPCP, 6, 8, True)

preAIR = xr.open_dataarray("/home/ys17-23/Extension/All_India_Rainfall_index/AIR_mmperday.nc")
preAIR_JJA = ca.p_time(preAIR, 6, 8, True)
preAIR_JJA = preAIR_JJA.sel(time=(preAIR_JJA.time.dt.year>=1979) & (preAIR_JJA.time.dt.year <=2014))

fhgtERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/hgt_mon_r144x72_195001-201412.nc")
hgtERA5 = fhgtERA5["z"].sel(time=fhgtERA5["time"].dt.year>=1979)

fuERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/uwind_mon_r144x72_195001-201412.nc"
)
uERA5 = fuERA5["u"].sel(time=fuERA5["time"].dt.year>=1979)

fvERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/vwind_mon_r144x72_195001-201412.nc"
)
vERA5 = fvERA5["v"].sel(time=fvERA5["time"].dt.year>=1979)

fspERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/sp_mon_r144x72_195001-201412.nc"
)
spERA5 = fspERA5["sp"].sel(time=fspERA5["time"].dt.year>=1979)

fqERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/q_mon_r144x72_195001-201412.nc")
qERA5 = fqERA5["q"].sel(time=fqERA5["time"].dt.year>=1979)

fwERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/omega_mon_r144x72_195001-201412.nc")
wERA5 = fwERA5["w"].sel(time=fwERA5["time"].dt.year>=1979)

fsstHad = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/HadISST_r144x72_1870-2020.nc")
sstHad = fsstHad["sst"].sel(time=(fsstHad["time"].dt.year>=1979) & (fsstHad["time"].dt.year<=2014))

hgtERA5_ver_JJA = ca.p_time(hgtERA5, 6, 8, True)
hgtERA5_ver_JJA = hgtERA5_ver_JJA-hgtERA5_ver_JJA.mean(dim="lon", skipna=True)
uERA5_ver_JJA = ca.p_time(uERA5, 6, 8, True)
vERA5_ver_JJA = ca.p_time(vERA5, 6, 8, True)
qERA5_ver_JJA = ca.p_time(qERA5, 6, 8, True)
spERA5_JJA = ca.p_time(spERA5, 6, 8, True)
wERA5_JJA = ca.p_time(wERA5, 6, 8, True)
sstHad_JJA = ca.p_time(sstHad, 6, 8, True)

#   read the data in CMIP6
fprehis = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/pr_historical_r144x72_195001-201412.nc")
prehis_JJA = fprehis["pr"].sel(time=fprehis["time"].dt.year>=1979)
prehis_JJA.attrs["units"] = "mm/day"
prehis_JJA.attrs["standard_name"] = "precipitation"


fpressp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/pr_ssp585_r144x72_201501-209912.nc")
pressp585_JJA = fpressp585["pr"]
pressp585_JJA.attrs["units"] = "mm/day"
pressp585_JJA.attrs["standard_name"] = "precipitation"
pressp585_p3_JJA = pressp585_JJA.sel(time=pressp585_JJA.time.dt.year>=2064)


fhgthis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/zg_historical_r144x72_195001-201412.nc")
hgthis_ver_JJA = fhgthis_ver_JJA["zg"].sel(time=fhgthis_ver_JJA["time"].dt.year>=1979)
hgthis_ver_JJA = hgthis_ver_JJA - hgthis_ver_JJA.mean(dim="lon", skipna=True)

fuhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ua_historical_r144x72_195001-201412.nc")
uhis_ver_JJA = fuhis_ver_JJA["ua"].sel(time=fuhis_ver_JJA["time"].dt.year>=1979)


fvhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/va_historical_r144x72_195001-201412.nc")
vhis_ver_JJA = fvhis_ver_JJA["va"].sel(time=fvhis_ver_JJA["time"].dt.year>=1979)


fwhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/wap_historical_r144x72_195001-201412.nc") 
whis_ver_JJA = fwhis_ver_JJA["wap"].sel(time=fwhis_ver_JJA["time"].dt.year>=1979)


#   read the SST data in observation and CMIP6
fssthis_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/tos_historical_r144x72_195001-201412.nc")
ssthis_JJA = fssthis_JJA["tos"].sel(time=fssthis_JJA["time"].dt.year>=1979)



fhgtssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/zg_ssp585_r144x72_201501-209912.nc")
hgtssp585_ver_JJA = fhgtssp585_ver_JJA["zg"]
hgtssp585_p3_ver_JJA = hgtssp585_ver_JJA.sel(time=hgtssp585_ver_JJA.time.dt.year>=2064)


hgtssp585_ver_JJA = hgtssp585_ver_JJA - hgtssp585_ver_JJA.mean(dim="lon", skipna=True)
hgtssp585_p3_ver_JJA = hgtssp585_p3_ver_JJA - hgtssp585_p3_ver_JJA.mean(dim="lon", skipna=True)


fussp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ua_ssp585_r144x72_201501-209912.nc")
ussp585_ver_JJA = fussp585_ver_JJA["ua"]
ussp585_p3_ver_JJA = ussp585_ver_JJA.sel(time=ussp585_ver_JJA.time.dt.year>=2064)

fvssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/va_ssp585_r144x72_201501-209912.nc")
vssp585_ver_JJA = fvssp585_ver_JJA["va"]
vssp585_p3_ver_JJA = vssp585_ver_JJA.sel(time=vssp585_ver_JJA.time.dt.year>=2064)

fwssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/wap_ssp585_r144x72_201501-209912.nc")
wssp585_ver_JJA = fwssp585_ver_JJA["wap"]
wssp585_p3_ver_JJA = wssp585_ver_JJA.sel(time=wssp585_ver_JJA.time.dt.year>=2064)

fsstssp585_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/tos_ssp585_r144x72_201501-209912.nc")
sstssp585_JJA = fsstssp585_JJA["tos"]
sstssp585_p3_JJA = sstssp585_JJA.sel(time=sstssp585_JJA.time.dt.year>=2064)


#   read the temperature data in ERA5/historical/ssp585
ftERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/temp_mon_r144x72_195001-201412.nc")
tERA5 = ftERA5["t"].sel(time=ftERA5["time"].dt.year>=1979)
tERA5_ver_JJA = ca.p_time(tERA5, 6, 8, True)


fthis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ta_historical_r144x72_195001-201412.nc")
this_ver_JJA = fthis_ver_JJA["ta"].sel(time=fthis_ver_JJA["time"].dt.year>=1979)


ftssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ta_ssp585_r144x72_201501-209912.nc")
tssp585_ver_JJA = ftssp585_ver_JJA["ta"]
tssp585_p3_ver_JJA = tssp585_ver_JJA.sel(time=tssp585_ver_JJA.time.dt.year>=2064)

#   calculate the monsoon index
ERA5_EAM = ca.EAM(uERA5_ver_JJA)
ERA5_IWF = ca.IWF(uERA5_ver_JJA, vERA5_ver_JJA)
ERA5_LKY = ca.LKY(uERA5_ver_JJA, vERA5_ver_JJA)


fhis_EAM = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_EAM_index_1950-2014.nc")
his_EAM = fhis_EAM["EAM"].sel(time=fhis_EAM["time"].dt.year>=1979)

fssp585_EAM = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_EAM_index_2015-2099.nc")
ssp585_EAM = fssp585_EAM["EAM"]
ssp585_p3_EAM = fssp585_EAM["EAM"].sel(time=fssp585_EAM["time"].dt.year>=2064)

fhis_IWF = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_IWF_index_1950-2014.nc")
his_IWF = fhis_IWF["IWF"].sel(time=fhis_IWF["time"].dt.year>=1979)

fssp585_IWF = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_IWF_index_2015-2099.nc")
ssp585_IWF = fssp585_IWF["IWF"]
ssp585_p3_IWF = fssp585_IWF["IWF"].sel(time=fssp585_IWF["time"].dt.year>=2064)

# %%
#   change the coordinate of variables
preGPCP_JJA.coords["time"] = prehis_JJA.coords["time"]
preAIR_JJA.coords["time"] = prehis_JJA.coords["time"]
models = uhis_ver_JJA.coords["models"]
models_array = models.data
# %%
#   pick up the area data
#   calculate the vorticity in ERA5, historical and ssp585
vorERA5_ver_JJA = mpcalc.vorticity(uERA5_ver_JJA.sel(level=200.0), vERA5_ver_JJA.sel(level=200.0))
vorERA5_ver_JJA = vorERA5_ver_JJA.metpy.dequantify()

vorhis_ver_JJA = mpcalc.vorticity(uhis_ver_JJA.sel(level=200.0), vhis_ver_JJA.sel(level=200.0))
vorhis_ver_JJA = vorhis_ver_JJA.metpy.dequantify()

vorssp585_ver_JJA = mpcalc.vorticity(ussp585_ver_JJA.sel(level=200.0), vssp585_ver_JJA.sel(level=200.0))
vorssp585_ver_JJA = vorssp585_ver_JJA.metpy.dequantify()

vorssp585_p3_ver_JJA = mpcalc.vorticity(ussp585_p3_ver_JJA.sel(level=200.0), vssp585_p3_ver_JJA.sel(level=200.0))
vorssp585_p3_ver_JJA = vorssp585_p3_ver_JJA.metpy.dequantify()

#   absolute vorticity in ERA5, historical and ssp585
abvorERA5_ver_JJA = mpcalc.absolute_vorticity(uERA5_ver_JJA.sel(level=200.0), vERA5_ver_JJA.sel(level=200.0)).metpy.dequantify()

abvorhis_ver_JJA = mpcalc.absolute_vorticity(uhis_ver_JJA.sel(level=200.0), vhis_ver_JJA.sel(level=200.0)).metpy.dequantify()

abvorssp585_p3_ver_JJA = mpcalc.absolute_vorticity(ussp585_p3_ver_JJA.sel(level=200.0), vssp585_p3_ver_JJA.sel(level=200.0)).metpy.dequantify()

#   absolute vorticity climatology mean
abvorERA5_ver_JJA_cli = abvorERA5_ver_JJA.mean(dim="time", skipna=True)
abvorhis_ver_JJA_cli = abvorhis_ver_JJA.mean(dim="time", skipna=True)
abvorssp585_p3_ver_JJA_cli = abvorssp585_p3_ver_JJA.mean(dim="time", skipna=True)

#   irrotational(divergent) wind
wERA5 = VectorWind(uERA5_ver_JJA.sel(level=200.0), vERA5_ver_JJA.sel(level=200.0))
whis = VectorWind(uhis_ver_JJA.sel(level=200.0), vhis_ver_JJA.sel(level=200.0))
wssp585_p3 = VectorWind(ussp585_p3_ver_JJA.sel(level=200.0), vssp585_p3_ver_JJA.sel(level=200.0))

udivERA5_ver_JJA, vdivERA5_ver_JJA = wERA5.irrotationalcomponent()
udivhis_ver_JJA, vdivhis_ver_JJA = whis.irrotationalcomponent()
udivssp585_p3_ver_JJA, vdivssp585_p3_ver_JJA = wssp585_p3.irrotationalcomponent()

#   irrotational(divergent) wind climatology
udivERA5_bar = udivERA5_ver_JJA.mean(dim="time", skipna=True)
vdivERA5_bar = vdivERA5_ver_JJA.mean(dim="time", skipna=True)

udivhis_bar = udivhis_ver_JJA.mean(dim="time", skipna=True)
vdivhis_bar = vdivhis_ver_JJA.mean(dim="time", skipna=True)

udivssp585_p3_bar = udivssp585_p3_ver_JJA.mean(dim="time", skipna=True)
vdivssp585_p3_bar = vdivssp585_p3_ver_JJA.mean(dim="time", skipna=True)

# ==========================
his_LKY = ca.LKY(uhis_ver_JJA, vhis_ver_JJA)
ssp585_p3_LKY = ca.LKY(ussp585_p3_ver_JJA, vssp585_p3_ver_JJA)

#   calculate the precipitation in India
lat = preGPCP_JJA.coords["lat"]
lon = preGPCP_JJA.coords["lon"]

India_N = 32.5
# India_N = 30.0
India_S = 7.5
India_W = 70.0
India_E = 85.0
lat_India_range = lat[(lat >= India_S) & (lat <= India_N)]
lon_India_range = lon[(lon >= India_W) & (lon <= India_E)]

preGPCP_India_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)
prehis_India_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)
pressp585_India_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)
pressp585_p3_India_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)

prehis_India_JJA_sum = prehis_JJA.sel(lat=lat_India_range, lon=lon_India_range).sum(dim=["lat", "lon"], skipna=True)
pressp585_p3_India_JJA_sum = pressp585_p3_JJA.sel(lat=lat_India_range, lon=lon_India_range).sum(dim=["lat", "lon"], skipna=True)

#   calculate the north India peninsula precipitation
nIndia_N = 32.5
nIndia_S = 17.5
nIndia_W = 70.0
nIndia_E = 85.0
lat_nIndia_range = lat[(lat >= nIndia_S) & (lat <= nIndia_N)]
lon_nIndia_range = lon[(lon >= nIndia_W) & (lon <= nIndia_E)]

preGPCP_nIndia_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_nIndia_range, lon=lon_nIndia_range)).mean(dim="lon", skipna=True)
prehis_nIndia_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_nIndia_range, lon=lon_nIndia_range)).mean(dim="lon", skipna=True)
pressp585_nIndia_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_nIndia_range, lon=lon_nIndia_range)).mean(dim="lon", skipna=True)
pressp585_p3_nIndia_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_nIndia_range, lon=lon_nIndia_range)).mean(dim="lon", skipna=True)

prehis_nIndia_JJA_sum = prehis_JJA.sel(lat=lat_nIndia_range, lon=lon_nIndia_range).sum(dim=["lat", "lon"], skipna=True)
pressp585_p3_nIndia_JJA_sum = pressp585_p3_JJA.sel(lat=lat_nIndia_range, lon=lon_nIndia_range).sum(dim=["lat", "lon"], skipna=True)

#   calculate the west India peninsula precipitation
wIndia_N = 32.5
# wIndia_N = 30.0
wIndia_S = 7.5
wIndia_W = 70.0
wIndia_E = 77.5
lat_wIndia_range = lat[(lat >= wIndia_S) & (lat <= wIndia_N)]
lon_wIndia_range = lon[(lon >= wIndia_W) & (lon <= wIndia_E)]

prehis_wIndia_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_wIndia_range, lon=lon_wIndia_range)).mean(dim="lon", skipna=True)
pressp585_wIndia_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_wIndia_range, lon=lon_wIndia_range)).mean(dim="lon", skipna=True)
pressp585_p3_wIndia_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_wIndia_range, lon=lon_wIndia_range)).mean(dim="lon", skipna=True)

prehis_wIndia_JJA_sum = prehis_JJA.sel(lat=lat_wIndia_range, lon=lon_wIndia_range).sum(dim=["lat", "lon"], skipna=True)
pressp585_p3_wIndia_JJA_sum = pressp585_p3_JJA.sel(lat=lat_wIndia_range, lon=lon_wIndia_range).sum(dim=["lat", "lon"], skipna=True)

#   calculate the precipitation in Northern China
NC_N = 40.0
NC_S = 32.5
NC_W = 105.0
NC_E = 120.0
lat_NC_range = lat[(lat >= NC_S) & (lat <= NC_N)]
lon_NC_range = lon[(lon >= NC_W) & (lon <= NC_E)]
preGPCP_NC_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_NC_range, lon=lon_NC_range)).mean(dim="lon", skipna=True)
prehis_NC_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_NC_range, lon=lon_NC_range)).mean(dim="lon", skipna=True)
pressp585_NC_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_NC_range, lon=lon_NC_range)).mean(dim="lon", skipna=True)
pressp585_p3_NC_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_NC_range, lon=lon_NC_range)).mean(dim="lon", skipna=True)

#   calculate the precipitation in Southern China
lat_SC_range = lat[(lat>=20.0) & (lat<=27.5)]
lon_SC_range = lon[(lon>=105.0) & (lon<=125.0)]
preGPCP_SC_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_SC_range, lon=lon_SC_range)).mean(dim="lon", skipna=True)
prehis_SC_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_SC_range, lon=lon_SC_range)).mean(dim="lon", skipna=True)
pressp585_SC_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_SC_range, lon=lon_SC_range)).mean(dim="lon", skipna=True)
pressp585_p3_SC_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_SC_range, lon=lon_SC_range)).mean(dim="lon", skipna=True)

#   calculate the precipitation in Korean Peninsula
KP_N = 37.5
KP_S = 32.5
KP_W = 125.0
KP_E = 132.5
lat_KP_range = lat[(lat >= KP_S) & (lat <= KP_N)]
lon_KP_range = lon[(lon >= KP_W) & (lon <= KP_E)]
# lat_KP_range = lat[(lat>=27.5) & (lat<=37.5)]
# lon_KP_range = lon[(lon>=105.0) & (lon<=125.0)]
# preCRU_KP_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_KP_range, lon=lon_KP_range)).mean(dim="lon", skipna=True)
preGPCP_KP_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_KP_range, lon=lon_KP_range)).mean(dim="lon", skipna=True)
prehis_KP_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_KP_range, lon=lon_KP_range)).mean(dim="lon", skipna=True)
pressp585_KP_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_KP_range, lon=lon_KP_range)).mean(dim="lon", skipna=True)
pressp585_p3_KP_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_KP_range, lon=lon_KP_range)).mean(dim="lon", skipna=True)

#   calculate the precipitation in Korean Peninsula-Southern Japan
SJ_N = 30.0
SJ_S = 25.0
SJ_W = 125.0
SJ_E = 135.0
lat_SJ_range = lat[(lat >= SJ_S) & (lat <= SJ_N)]
lon_SJ_range = lon[(lon >= SJ_W) & (lon <= SJ_E)]
# lat_SJ_range = lat[(lat>=27.5) & (lat<=37.5)]
# lon_SJ_range = lon[(lon>=105.0) & (lon<=125.0)]
# preCRU_SJ_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_SJ_range, lon=lon_SJ_range)).mean(dim="lon", skipna=True)
preGPCP_SJ_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_SJ_range, lon=lon_SJ_range)).mean(dim="lon", skipna=True)
prehis_SJ_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_SJ_range, lon=lon_SJ_range)).mean(dim="lon", skipna=True)
pressp585_SJ_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_SJ_range, lon=lon_SJ_range)).mean(dim="lon", skipna=True)
pressp585_p3_SJ_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_SJ_range, lon=lon_SJ_range)).mean(dim="lon", skipna=True)

#   calculate the 200hPa u-wind over the East Asia
lat_EA_range = lat[(lat>=30.0) & (lat<=40.0)]
lon_EA_range = lon[(lon>=100.0) & (lon<=120.0)]

uERA5_EA_JJA = ca.cal_lat_weighted_mean(uERA5_ver_JJA.sel(lat=lat_EA_range, lon=lon_EA_range, level=200.0)).mean(dim="lon", skipna=True)
uhis_EA_JJA = ca.cal_lat_weighted_mean(uhis_ver_JJA.sel(lat=lat_EA_range, lon=lon_EA_range, level=200.0)).mean(dim="lon", skipna=True)
ussp585_EA_JJA = ca.cal_lat_weighted_mean(ussp585_ver_JJA.sel(lat=lat_EA_range, lon=lon_EA_range, level=200.0)).mean(dim="lon", skipna=True)
ussp585_p3_EA_JJA = ca.cal_lat_weighted_mean(ussp585_p3_ver_JJA.sel(lat=lat_EA_range, lon=lon_EA_range, level=200.0)).mean(dim="lon", skipna=True)

#   calculate the 200hPa vorticity over the East Asia
EAhigh_N = 50.0
EAhigh_S = 27.5
EAhigh_W = 105.0
EAhigh_E = 140.0
lat_EAhigh_range = lat[(lat>=EAhigh_S) & (lat<=EAhigh_N)]
lon_EAhigh_range = lon[(lon>=EAhigh_W) & (lon<=EAhigh_E)]
uERA5_EAhigh_JJA = uERA5_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
uhis_EAhigh_JJA = uhis_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
ussp585_EAhigh_JJA = ussp585_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
ussp585_p3_EAhigh_JJA = ussp585_p3_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)

vERA5_EAhigh_JJA = vERA5_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
vhis_EAhigh_JJA = vhis_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
vssp585_EAhigh_JJA = vssp585_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
vssp585_p3_EAhigh_JJA = vssp585_p3_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)

vorERA5_EAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uERA5_EAhigh_JJA, vERA5_EAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorhis_EAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uhis_EAhigh_JJA, vhis_EAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorssp585_EAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(ussp585_EAhigh_JJA, vssp585_EAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorssp585_p3_EAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(ussp585_p3_EAhigh_JJA, vssp585_p3_EAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()

#   calculate the vorticity over the West Asia in 200hPa
WAhigh_N = 50.0
WAhigh_S = 25.0
WAhigh_W = 50.0
WAhigh_E = 80.0
lat_WAhigh_range = lat[(lat>=WAhigh_S) & (lat<=WAhigh_N)]
lon_WAhigh_range = lon[(lon>=WAhigh_W) & (lon<=WAhigh_E)]
uERA5_WAhigh_JJA = uERA5_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
uhis_WAhigh_JJA = uhis_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
ussp585_WAhigh_JJA = ussp585_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
ussp585_p3_WAhigh_JJA = ussp585_p3_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)

vERA5_WAhigh_JJA = vERA5_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
vhis_WAhigh_JJA = vhis_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
vssp585_WAhigh_JJA = vssp585_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
vssp585_p3_WAhigh_JJA = vssp585_p3_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)

vorERA5_WAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uERA5_WAhigh_JJA, vERA5_WAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorhis_WAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uhis_WAhigh_JJA, vhis_WAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorssp585_WAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(ussp585_WAhigh_JJA, vssp585_WAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorssp585_p3_WAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(ussp585_p3_WAhigh_JJA, vssp585_p3_WAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()

#   calculate the vorticity over the West Asia in 200hPa
WNPhigh_N = 37.5
WNPhigh_S = 15.0
WNPhigh_W = 110.0
WNPhigh_E = 140.0
lat_WNPhigh_range = lat[(lat>=WNPhigh_S) & (lat<=WNPhigh_N)]
lon_WNPhigh_range = lon[(lon>=WNPhigh_W) & (lon<=WNPhigh_E)]
uERA5_WNPhigh_JJA = uERA5_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)
uhis_WNPhigh_JJA = uhis_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)
ussp585_WNPhigh_JJA = ussp585_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)
ussp585_p3_WNPhigh_JJA = ussp585_p3_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)

vERA5_WNPhigh_JJA = vERA5_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)
vhis_WNPhigh_JJA = vhis_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)
vssp585_WNPhigh_JJA = vssp585_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)
vssp585_p3_WNPhigh_JJA = vssp585_p3_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)

vorERA5_WNPhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uERA5_WNPhigh_JJA, vERA5_WNPhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorhis_WNPhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uhis_WNPhigh_JJA, vhis_WNPhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorssp585_WNPhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(ussp585_WNPhigh_JJA, vssp585_WNPhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorssp585_p3_WNPhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(ussp585_p3_WNPhigh_JJA, vssp585_p3_WNPhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
# %%
#   calculate the precipitation fields regression onto IndR
(
    pre_GPCP_India_pre_slope,
    pre_GPCP_India_pre_intercept,
    pre_GPCP_India_pre_rvalue,
    pre_GPCP_India_pre_pvalue,
    pre_GPCP_India_pre_hypothesis,
) = ca.dim_linregress(preGPCP_India_JJA, preGPCP_JJA)

preAIR_JJA.coords["time"] = preGPCP_JJA.coords["time"]
(
    pre_AIR_India_pre_slope,
    pre_AIR_India_pre_intercept,
    pre_AIR_India_pre_rvalue,
    pre_AIR_India_pre_pvalue,
    pre_AIR_India_pre_hypothesis,
) = ca.dim_linregress(preAIR_JJA, preGPCP_JJA)

(
    pre_his_India_pre_slope,
    pre_his_India_pre_intercept,
    pre_his_India_pre_rvalue,
    pre_his_India_pre_pvalue,
    pre_his_India_pre_hypothesis,
) = ca.dim_linregress(prehis_India_JJA, prehis_JJA)

(
    pre_ssp585_p3_India_pre_slope,
    pre_ssp585_p3_India_pre_intercept,
    pre_ssp585_p3_India_pre_rvalue,
    pre_ssp585_p3_India_pre_pvalue,
    pre_ssp585_p3_India_pre_hypothesis,
) = ca.dim_linregress(pressp585_p3_India_JJA, pressp585_p3_JJA)
# %%
#   calculate the MME for historical and ssp585_p3
pre_his_India_pre_slope_ens = pre_his_India_pre_slope.mean(dim="models", skipna=True)

pre_his_India_pre_slope_ens_mask = xr.where((ca.MME_reg_mask(pre_his_India_pre_slope_ens, pre_his_India_pre_slope.std(dim="models", skipna=True), len(pre_his_India_pre_slope.coords["models"]), True) + ca.cal_mmemask(pre_his_India_pre_slope)) >= 2.0, 1.0, 0.0)

pre_his_India_pre_rvalue_ens = ca.cal_rMME(pre_his_India_pre_rvalue,"models")

pre_his_India_pre_rvalue_ens_mask = xr.where((ca.MME_reg_mask(pre_his_India_pre_rvalue_ens, pre_his_India_pre_rvalue.std(dim="models", skipna=True), len(pre_his_India_pre_rvalue.coords["models"]), True) + ca.cal_mmemask(pre_his_India_pre_slope)) >= 2.0, 1.0, 0.0)

pre_ssp585_p3_India_pre_slope_ens = pre_ssp585_p3_India_pre_slope.mean(dim="models", skipna=True)

pre_ssp585_p3_India_pre_slope_ens_mask = xr.where((ca.MME_reg_mask(pre_ssp585_p3_India_pre_slope_ens, pre_ssp585_p3_India_pre_slope.std(dim="models", skipna=True), len(pre_ssp585_p3_India_pre_slope.coords["models"]), True) + ca.cal_mmemask(pre_ssp585_p3_India_pre_slope)) >= 2.0, 1.0, 0.0)

pre_ssp585_p3_India_pre_rvalue_ens = ca.cal_rMME(pre_ssp585_p3_India_pre_rvalue,"models")

pre_ssp585_p3_India_pre_rvalue_ens_mask = xr.where((ca.MME_reg_mask(pre_ssp585_p3_India_pre_rvalue_ens, pre_ssp585_p3_India_pre_rvalue.std(dim="models", skipna=True), len(pre_ssp585_p3_India_pre_rvalue.coords["models"]), True) + ca.cal_mmemask(pre_ssp585_p3_India_pre_slope)) >= 2.0, 1.0, 0.0)
# %%
# #   calculate the relative vorticity prime and irrotational winds prime
# preGPCP_India_JJA.coords["time"] = hgtERA5_ver_JJA.coords["time"]
# preAIR_JJA.coords["time"] = hgtERA5_ver_JJA.coords["time"]

# (
#     IndR_ERA5_vor_slope,
#     IndR_ERA5_vor_intercept,
#     IndR_ERA5_vor_rvalue,
#     IndR_ERA5_vor_pvalue,
#     IndR_ERA5_vor_hypothesis,
# ) = ca.dim_linregress(preAIR_JJA, vorERA5_ver_JJA)

# (
#     IndR_his_vor_slope,
#     IndR_his_vor_intercept,
#     IndR_his_vor_rvalue,
#     IndR_his_vor_pvalue,
#     IndR_his_vor_hypothesis,
# ) = ca.dim_linregress(prehis_India_JJA, vorhis_ver_JJA)

# (
#     IndR_ssp585_p3_vor_slope,
#     IndR_ssp585_p3_vor_intercept,
#     IndR_ssp585_p3_vor_rvalue,
#     IndR_ssp585_p3_vor_pvalue,
#     IndR_ssp585_p3_vor_hypothesis,
# ) = ca.dim_linregress(pressp585_p3_India_JJA, vorssp585_p3_ver_JJA)

# (
#     IndR_ERA5_udiv_slope,
#     IndR_ERA5_udiv_intercept,
#     IndR_ERA5_udiv_rvalue,
#     IndR_ERA5_udiv_pvalue,
#     IndR_ERA5_udiv_hypothesis,
# ) = ca.dim_linregress(preAIR_JJA, udivERA5_ver_JJA)

# (
#     IndR_his_udiv_slope,
#     IndR_his_udiv_intercept,
#     IndR_his_udiv_rvalue,
#     IndR_his_udiv_pvalue,
#     IndR_his_udiv_hypothesis,
# ) = ca.dim_linregress(prehis_India_JJA, udivhis_ver_JJA)

# (
#     IndR_ssp585_p3_udiv_slope,
#     IndR_ssp585_p3_udiv_intercept,
#     IndR_ssp585_p3_udiv_rvalue,
#     IndR_ssp585_p3_udiv_pvalue,
#     IndR_ssp585_p3_udiv_hypothesis,
# ) = ca.dim_linregress(pressp585_p3_India_JJA, udivssp585_p3_ver_JJA)

# (
#     IndR_ERA5_vdiv_slope,
#     IndR_ERA5_vdiv_intercept,
#     IndR_ERA5_vdiv_rvalue,
#     IndR_ERA5_vdiv_pvalue,
#     IndR_ERA5_vdiv_hypothesis,
# ) = ca.dim_linregress(preAIR_JJA, vdivERA5_ver_JJA)

# (
#     IndR_his_vdiv_slope,
#     IndR_his_vdiv_intercept,
#     IndR_his_vdiv_rvalue,
#     IndR_his_vdiv_pvalue,
#     IndR_his_vdiv_hypothesis,
# ) = ca.dim_linregress(prehis_India_JJA, vdivhis_ver_JJA)

# (
#     IndR_ssp585_p3_vdiv_slope,
#     IndR_ssp585_p3_vdiv_intercept,
#     IndR_ssp585_p3_vdiv_rvalue,
#     IndR_ssp585_p3_vdiv_pvalue,
#     IndR_ssp585_p3_vdiv_hypothesis,
# ) = ca.dim_linregress(pressp585_p3_India_JJA, vdivssp585_p3_ver_JJA)
# # %%
# #   save the regression results
# level=IndR_his_vor_slope.coords["level"]
# lat=IndR_his_vor_slope.coords["lat"]
# lon=IndR_his_vor_slope.coords["lon"]

# IndR_ERA5_vor_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["lat", "lon"], IndR_ERA5_vor_slope.data),
#         intercept=(["lat", "lon"], IndR_ERA5_vor_intercept.data),
#         rvalue=(["lat", "lon"], IndR_ERA5_vor_rvalue.data),
#         pvalue=(["lat", "lon"], IndR_ERA5_vor_pvalue.data),
#         hypothesis=(["lat", "lon"], IndR_ERA5_vor_hypothesis.data),
#     ),
#     coords=dict(
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="relative vorticity fields of ERA5 regress onto 1979-2014 AIR"),
# )

# IndR_his_vor_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "lat", "lon"], IndR_his_vor_slope.data),
#         intercept=(["models", "lat", "lon"], IndR_his_vor_intercept.data),
#         rvalue=(["models", "lat", "lon"], IndR_his_vor_rvalue.data),
#         pvalue=(["models", "lat", "lon"], IndR_his_vor_pvalue.data),
#         hypothesis=(["models", "lat", "lon"], IndR_his_vor_hypothesis.data),
#     ),
#     coords=dict(
#         models=models_array,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="relative vorticity fields of historical regress onto 1979-2014 India Rainfall"),
# )
# IndR_ssp585_p3_vor_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "lat", "lon"], IndR_ssp585_p3_vor_slope.data),
#         intercept=(["models", "lat", "lon"], IndR_ssp585_p3_vor_intercept.data),
#         rvalue=(["models", "lat", "lon"], IndR_ssp585_p3_vor_rvalue.data),
#         pvalue=(["models", "lat", "lon"], IndR_ssp585_p3_vor_pvalue.data),
#         hypothesis=(["models", "lat", "lon"], IndR_ssp585_p3_vor_hypothesis.data),
#     ),
#     coords=dict(
#         models=models_array,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="relative vorticity fields of ssp585_p3 regress onto 2064-2099 India Rainfall"),
# )

# IndR_ERA5_vor_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_ERA5_vor_regress.nc")
# IndR_his_vor_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_vor_regress.nc")
# IndR_ssp585_p3_vor_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_vor_regress.nc")

# IndR_ERA5_udiv_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["lat", "lon"], IndR_ERA5_udiv_slope.data),
#         intercept=(["lat", "lon"], IndR_ERA5_udiv_intercept.data),
#         rvalue=(["lat", "lon"], IndR_ERA5_udiv_rvalue.data),
#         pvalue=(["lat", "lon"], IndR_ERA5_udiv_pvalue.data),
#         hypothesis=(["lat", "lon"], IndR_ERA5_udiv_hypothesis.data),
#     ),
#     coords=dict(
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="relative divergent wind u fields of ERA5 regress onto 1979-2014 AIR"),
# )

# IndR_his_udiv_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "lat", "lon"], IndR_his_udiv_slope.data),
#         intercept=(["models", "lat", "lon"], IndR_his_udiv_intercept.data),
#         rvalue=(["models", "lat", "lon"], IndR_his_udiv_rvalue.data),
#         pvalue=(["models", "lat", "lon"], IndR_his_udiv_pvalue.data),
#         hypothesis=(["models", "lat", "lon"], IndR_his_udiv_hypothesis.data),
#     ),
#     coords=dict(
#         models=models_array,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="relative divergent wind u fields of historical regress onto 1979-2014 India Rainfall"),
# )
# IndR_ssp585_p3_udiv_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "lat", "lon"], IndR_ssp585_p3_udiv_slope.data),
#         intercept=(["models", "lat", "lon"], IndR_ssp585_p3_udiv_intercept.data),
#         rvalue=(["models", "lat", "lon"], IndR_ssp585_p3_udiv_rvalue.data),
#         pvalue=(["models", "lat", "lon"], IndR_ssp585_p3_udiv_pvalue.data),
#         hypothesis=(["models", "lat", "lon"], IndR_ssp585_p3_udiv_hypothesis.data),
#     ),
#     coords=dict(
#         models=models_array,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="relative divergent wind u fields of ssp585_p3 regress onto 2064-2099 India Rainfall"),
# )

# IndR_ERA5_udiv_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_ERA5_udiv_regress.nc")
# IndR_his_udiv_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_udiv_regress.nc")
# IndR_ssp585_p3_udiv_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_udiv_regress.nc")

# IndR_ERA5_vdiv_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["lat", "lon"], IndR_ERA5_vdiv_slope.data),
#         intercept=(["lat", "lon"], IndR_ERA5_vdiv_intercept.data),
#         rvalue=(["lat", "lon"], IndR_ERA5_vdiv_rvalue.data),
#         pvalue=(["lat", "lon"], IndR_ERA5_vdiv_pvalue.data),
#         hypothesis=(["lat", "lon"], IndR_ERA5_vdiv_hypothesis.data),
#     ),
#     coords=dict(
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="relative divergent wind v fields of ERA5 regress onto 1979-2014 AIR"),
# )

# IndR_his_vdiv_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "lat", "lon"], IndR_his_vdiv_slope.data),
#         intercept=(["models", "lat", "lon"], IndR_his_vdiv_intercept.data),
#         rvalue=(["models", "lat", "lon"], IndR_his_vdiv_rvalue.data),
#         pvalue=(["models", "lat", "lon"], IndR_his_vdiv_pvalue.data),
#         hypothesis=(["models", "lat", "lon"], IndR_his_vdiv_hypothesis.data),
#     ),
#     coords=dict(
#         models=models_array,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="relative divergent wind v fields of historical regress onto 1979-2014 India Rainfall"),
# )
# IndR_ssp585_p3_vdiv_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "lat", "lon"], IndR_ssp585_p3_vdiv_slope.data),
#         intercept=(["models", "lat", "lon"], IndR_ssp585_p3_vdiv_intercept.data),
#         rvalue=(["models", "lat", "lon"], IndR_ssp585_p3_vdiv_rvalue.data),
#         pvalue=(["models", "lat", "lon"], IndR_ssp585_p3_vdiv_pvalue.data),
#         hypothesis=(["models", "lat", "lon"], IndR_ssp585_p3_vdiv_hypothesis.data),
#     ),
#     coords=dict(
#         models=models_array,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="relative divergent wind v fields of ssp585_p3 regress onto 2064-2099 India Rainfall"),
# )

# IndR_ERA5_vdiv_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_ERA5_vdiv_regress.nc")
# IndR_his_vdiv_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_vdiv_regress.nc")
# IndR_ssp585_p3_vdiv_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_vdiv_regress.nc")
#%%
#   read the regression data
IndR_ERA5_vor_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_ERA5_vor_regress.nc")
IndR_his_vor_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_vor_regress.nc")
IndR_ssp585_p3_vor_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_vor_regress.nc")

IndR_ERA5_udiv_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_ERA5_udiv_regress.nc")
IndR_his_udiv_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_udiv_regress.nc")
IndR_ssp585_p3_udiv_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_udiv_regress.nc")

IndR_ERA5_vdiv_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_ERA5_vdiv_regress.nc")
IndR_his_vdiv_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_vdiv_regress.nc")
IndR_ssp585_p3_vdiv_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_vdiv_regress.nc")

vorERA5_ver_JJA_prime = IndR_ERA5_vor_regress["slope"]
vorhis_ver_JJA_prime = IndR_his_vor_regress["slope"]
vorssp585_p3_ver_JJA_prime = IndR_ssp585_p3_vor_regress["slope"]

udivERA5_prime = IndR_ERA5_udiv_regress["slope"]
udivhis_prime = IndR_his_udiv_regress["slope"]
udivssp585_p3_prime = IndR_ssp585_p3_udiv_regress["slope"]

vdivERA5_prime = IndR_ERA5_vdiv_regress["slope"]
vdivhis_prime = IndR_his_vdiv_regress["slope"]
vdivssp585_p3_prime = IndR_ssp585_p3_vdiv_regress["slope"]
#%%
#   calculate the Rossby Wave Source
SERA51 = -VectorWind(udivERA5_prime*abvorERA5_ver_JJA_cli, vdivERA5_prime*abvorERA5_ver_JJA_cli).divergence()
SERA52 = -VectorWind(udivERA5_bar*vorERA5_ver_JJA_prime, vdivERA5_bar*vorERA5_ver_JJA_prime).divergence()

#   term1:-wSERA51.divergence()
#   term2:-wSERA52.divergence()
SERA5 = SERA51 + SERA52

Shis1 = -VectorWind(udivhis_prime*abvorhis_ver_JJA_cli, vdivhis_prime*abvorhis_ver_JJA_cli).divergence()
Shis1_ens = Shis1.mean(dim="models", skipna=True)
Shis1_ens_mask = xr.where((ca.MME_reg_mask(Shis1_ens, Shis1.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(Shis1)) >= 2.0, 1.0, 0.0)

Shis2 = -VectorWind(udivhis_bar*vorhis_ver_JJA_prime, vdivhis_bar*vorhis_ver_JJA_prime).divergence()
Shis2_ens = Shis2.mean(dim="models", skipna=True)
Shis2_ens_mask = xr.where((ca.MME_reg_mask(Shis2_ens, Shis2.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(Shis2)) >= 2.0, 1.0, 0.0)

Shis = Shis1 + Shis2
Shis_ens = Shis.mean(dim="models", skipna=True)
Shis_ens_mask = xr.where((ca.MME_reg_mask(Shis_ens, Shis.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(Shis)) >= 2.0, 1.0, 0.0)

Sssp585_p31 = -VectorWind(udivssp585_p3_prime*abvorssp585_p3_ver_JJA_cli, vdivssp585_p3_prime*abvorssp585_p3_ver_JJA_cli).divergence()
Sssp585_p31_ens = Sssp585_p31.mean(dim="models", skipna=True)
Sssp585_p31_ens_mask = xr.where((ca.MME_reg_mask(Sssp585_p31_ens, Sssp585_p31.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(Sssp585_p31)) >= 2.0, 1.0, 0.0)
Sssp585_p32 = -VectorWind(udivssp585_p3_bar*vorssp585_p3_ver_JJA_prime, vdivssp585_p3_bar*vorssp585_p3_ver_JJA_prime).divergence()
Sssp585_p32_ens = Sssp585_p32.mean(dim="models", skipna=True)
Sssp585_p32_ens_mask = xr.where((ca.MME_reg_mask(Sssp585_p32_ens, Sssp585_p32.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(Sssp585_p32)) >= 2.0, 1.0, 0.0)

Sssp585_p3 = Sssp585_p31 + Sssp585_p32
Sssp585_p3_ens = Sssp585_p3.mean(dim="models", skipna=True)
Sssp585_p3_ens_mask = xr.where((ca.MME_reg_mask(Sssp585_p3_ens, Sssp585_p3.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(Sssp585_p3)) >= 2.0, 1.0, 0.0)

Sdiff1 = Sssp585_p31 - Shis1
Sdiff1_ens = Sdiff1.mean(dim="models", skipna=True)
Sdiff1_ens_mask = ca.cal_mmemask(Sdiff1)

Sdiff2 = Sssp585_p32 - Shis2
Sdiff2_ens = Sdiff2.mean(dim="models", skipna=True)
Sdiff2_ens_mask = ca.cal_mmemask(Sdiff2)

Sdiff = Sssp585_p3 - Shis
Sdiff_ens = Sdiff.mean(dim="models", skipna=True)
Sdiff_ens_mask = ca.cal_mmemask(Sdiff)

# %%
#   calculate the area mean of the Rossby wave source

WARWS_N = 42.5
WARWS_S = 32.5
WARWS_E = 90
WARWS_W = 62.5

lat_WARWS_range = lat[(lat>=WARWS_S) & (lat<=WARWS_N)]
lon_WARWS_range = lon[(lon>=WARWS_W) & (lon<=WARWS_E)]

SERA5_WARWS = ca.cal_lat_weighted_mean(SERA5.sel(lat=lat_WARWS_range, lon=lon_WARWS_range)).mean(dim="lon", skipna=True)
Shis_WARWS = ca.cal_lat_weighted_mean(Shis.sel(lat=lat_WARWS_range, lon=lon_WARWS_range)).mean(dim="lon", skipna=True)
Sssp585_p3_WARWS = ca.cal_lat_weighted_mean(Sssp585_p3.sel(lat=lat_WARWS_range, lon=lon_WARWS_range)).mean(dim="lon", skipna=True)
Sdiff_WARWS = Sssp585_p3_WARWS - Shis_WARWS

SERA51_WARWS = ca.cal_lat_weighted_mean(SERA51.sel(lat=lat_WARWS_range, lon=lon_WARWS_range)).mean(dim="lon", skipna=True)
Shis1_WARWS = ca.cal_lat_weighted_mean(Shis1.sel(lat=lat_WARWS_range, lon=lon_WARWS_range)).mean(dim="lon", skipna=True)
Sssp585_p31_WARWS = ca.cal_lat_weighted_mean(Sssp585_p31.sel(lat=lat_WARWS_range, lon=lon_WARWS_range)).mean(dim="lon", skipna=True)
Sdiff1_WARWS = Sssp585_p31_WARWS - Shis1_WARWS

SERA52_WARWS = ca.cal_lat_weighted_mean(SERA52.sel(lat=lat_WARWS_range, lon=lon_WARWS_range)).mean(dim="lon", skipna=True)
Shis2_WARWS = ca.cal_lat_weighted_mean(Shis2.sel(lat=lat_WARWS_range, lon=lon_WARWS_range)).mean(dim="lon", skipna=True)
Sssp585_p32_WARWS = ca.cal_lat_weighted_mean(Sssp585_p32.sel(lat=lat_WARWS_range, lon=lon_WARWS_range)).mean(dim="lon", skipna=True)
Sdiff2_WARWS = Sssp585_p32_WARWS - Shis2_WARWS

EARWS_N = 40.0
EARWS_S = 30.0
EARWS_E = 130.0
EARWS_W = 117.5

lat_EARWS_range = lat[(lat>=EARWS_S) & (lat<=EARWS_N)]
lon_EARWS_range = lon[(lon>=EARWS_W) & (lon<=EARWS_E)]

SERA5_EARWS = ca.cal_lat_weighted_mean(SERA5.sel(lat=lat_EARWS_range, lon=lon_EARWS_range)).mean(dim="lon", skipna=True)
Shis_EARWS = ca.cal_lat_weighted_mean(Shis.sel(lat=lat_EARWS_range, lon=lon_EARWS_range)).mean(dim="lon", skipna=True)
Sssp585_p3_EARWS = ca.cal_lat_weighted_mean(Sssp585_p3.sel(lat=lat_EARWS_range, lon=lon_EARWS_range)).mean(dim="lon", skipna=True)
Sdiff_EARWS = Sssp585_p3_EARWS - Shis_EARWS

SERA51_EARWS = ca.cal_lat_weighted_mean(SERA51.sel(lat=lat_EARWS_range, lon=lon_EARWS_range)).mean(dim="lon", skipna=True)
Shis1_EARWS = ca.cal_lat_weighted_mean(Shis1.sel(lat=lat_EARWS_range, lon=lon_EARWS_range)).mean(dim="lon", skipna=True)
Sssp585_p31_EARWS = ca.cal_lat_weighted_mean(Sssp585_p31.sel(lat=lat_EARWS_range, lon=lon_EARWS_range)).mean(dim="lon", skipna=True)
Sdiff1_EARWS = Sssp585_p31_EARWS - Shis1_EARWS

SERA52_EARWS = ca.cal_lat_weighted_mean(SERA52.sel(lat=lat_EARWS_range, lon=lon_EARWS_range)).mean(dim="lon", skipna=True)
Shis2_EARWS = ca.cal_lat_weighted_mean(Shis2.sel(lat=lat_EARWS_range, lon=lon_EARWS_range)).mean(dim="lon", skipna=True)
Sssp585_p32_EARWS = ca.cal_lat_weighted_mean(Sssp585_p32.sel(lat=lat_EARWS_range, lon=lon_EARWS_range)).mean(dim="lon", skipna=True)
Sdiff2_EARWS = Sssp585_p32_EARWS - Shis2_EARWS

#   calculate the ens in WARWS and EARWS
Shis_WARWS_ens = Shis_WARWS.mean(dim="models", skipna=True)
Sssp585_p3_WARWS_ens = Sssp585_p3_WARWS.mean(dim="models", skipna=True)
Sdiff_WARWS_ens = Sdiff_WARWS.mean(dim="models", skipna=True)
Shis1_WARWS_ens = Shis1_WARWS.mean(dim="models", skipna=True)
Sssp585_p31_WARWS_ens = Sssp585_p31_WARWS.mean(dim="models", skipna=True)
Sdiff1_WARWS_ens = Sdiff1_WARWS.mean(dim="models", skipna=True)
Shis2_WARWS_ens = Shis2_WARWS.mean(dim="models", skipna=True)
Sssp585_p32_WARWS_ens = Sssp585_p32_WARWS.mean(dim="models", skipna=True)
Sdiff2_WARWS_ens = Sdiff2_WARWS.mean(dim="models", skipna=True)

Shis_EARWS_ens = Shis_EARWS.mean(dim="models", skipna=True)
Sssp585_p3_EARWS_ens = Sssp585_p3_EARWS.mean(dim="models", skipna=True)
Sdiff_EARWS_ens = Sdiff_EARWS.mean(dim="models", skipna=True)
Shis1_EARWS_ens = Shis1_EARWS.mean(dim="models", skipna=True)
Sssp585_p31_EARWS_ens = Sssp585_p31_EARWS.mean(dim="models", skipna=True)
Sdiff1_EARWS_ens = Sdiff1_EARWS.mean(dim="models", skipna=True)
Shis2_EARWS_ens = Shis2_EARWS.mean(dim="models", skipna=True)
Sssp585_p32_EARWS_ens = Sssp585_p32_EARWS.mean(dim="models", skipna=True)
Sdiff2_EARWS_ens = Sdiff2_EARWS.mean(dim="models", skipna=True)

#   calculate the ens mask
Shis_WARWS_ens_mask = xr.where((ca.MME_reg_mask(Shis_WARWS_ens, Shis_WARWS.std(dim="models", skipna=True), len(Shis_WARWS.coords["models"]), True) + ca.cal_mmemask(Shis_WARWS)) >= 2.0, 1.0, 0.0)
Shis1_WARWS_ens_mask = xr.where((ca.MME_reg_mask(Shis1_WARWS_ens, Shis1_WARWS.std(dim="models", skipna=True), len(Shis1_WARWS.coords["models"]), True) + ca.cal_mmemask(Shis1_WARWS)) >= 2.0, 1.0, 0.0)
Shis2_WARWS_ens_mask = xr.where((ca.MME_reg_mask(Shis2_WARWS_ens, Shis2_WARWS.std(dim="models", skipna=True), len(Shis2_WARWS.coords["models"]), True) + ca.cal_mmemask(Shis2_WARWS)) >= 2.0, 1.0, 0.0)
Sssp585_p3_WARWS_ens_mask = xr.where((ca.MME_reg_mask(Sssp585_p3_WARWS_ens, Sssp585_p3_WARWS.std(dim="models", skipna=True), len(Sssp585_p3_WARWS.coords["models"]), True) + ca.cal_mmemask(Sssp585_p3_WARWS)) >= 2.0, 1.0, 0.0)
Sssp585_p31_WARWS_ens_mask = xr.where((ca.MME_reg_mask(Sssp585_p31_WARWS_ens, Sssp585_p31_WARWS.std(dim="models", skipna=True), len(Sssp585_p31_WARWS.coords["models"]), True) + ca.cal_mmemask(Sssp585_p31_WARWS)) >= 2.0, 1.0, 0.0)
Sssp585_p32_WARWS_ens_mask = xr.where((ca.MME_reg_mask(Sssp585_p32_WARWS_ens, Sssp585_p32_WARWS.std(dim="models", skipna=True), len(Sssp585_p32_WARWS.coords["models"]), True) + ca.cal_mmemask(Sssp585_p32_WARWS)) >= 2.0, 1.0, 0.0)
Sdiff_WARWS_ens_mask = xr.where((ca.MME_reg_mask(Sdiff_WARWS_ens, Sdiff_WARWS.std(dim="models", skipna=True), len(Sdiff_WARWS.coords["models"]), True) + ca.cal_mmemask(Sdiff_WARWS)) >= 2.0, 1.0, 0.0)
Sdiff1_WARWS_ens_mask = xr.where((ca.MME_reg_mask(Sdiff1_WARWS_ens, Sdiff1_WARWS.std(dim="models", skipna=True), len(Sdiff1_WARWS.coords["models"]), True) + ca.cal_mmemask(Sdiff1_WARWS)) >= 2.0, 1.0, 0.0)
Sdiff2_WARWS_ens_mask = xr.where((ca.MME_reg_mask(Sdiff2_WARWS_ens, Sdiff2_WARWS.std(dim="models", skipna=True), len(Sdiff2_WARWS.coords["models"]), True) + ca.cal_mmemask(Sdiff2_WARWS)) >= 2.0, 1.0, 0.0)
Shis_EARWS_ens_mask = xr.where((ca.MME_reg_mask(Shis_EARWS_ens, Shis_EARWS.std(dim="models", skipna=True), len(Shis_EARWS.coords["models"]), True) + ca.cal_mmemask(Shis_EARWS)) >= 2.0, 1.0, 0.0)
Shis1_EARWS_ens_mask = xr.where((ca.MME_reg_mask(Shis1_EARWS_ens, Shis1_EARWS.std(dim="models", skipna=True), len(Shis1_EARWS.coords["models"]), True) + ca.cal_mmemask(Shis1_EARWS)) >= 2.0, 1.0, 0.0)
Shis2_EARWS_ens_mask = xr.where((ca.MME_reg_mask(Shis2_EARWS_ens, Shis2_EARWS.std(dim="models", skipna=True), len(Shis2_EARWS.coords["models"]), True) + ca.cal_mmemask(Shis2_EARWS)) >= 2.0, 1.0, 0.0)
Sssp585_p3_EARWS_ens_mask = xr.where((ca.MME_reg_mask(Sssp585_p3_EARWS_ens, Sssp585_p3_EARWS.std(dim="models", skipna=True), len(Sssp585_p3_EARWS.coords["models"]), True) + ca.cal_mmemask(Sssp585_p3_EARWS)) >= 2.0, 1.0, 0.0)
Sssp585_p31_EARWS_ens_mask = xr.where((ca.MME_reg_mask(Sssp585_p31_EARWS_ens, Sssp585_p31_EARWS.std(dim="models", skipna=True), len(Sssp585_p31_EARWS.coords["models"]), True) + ca.cal_mmemask(Sssp585_p31_EARWS)) >= 2.0, 1.0, 0.0)
Sssp585_p32_EARWS_ens_mask = xr.where((ca.MME_reg_mask(Sssp585_p32_EARWS_ens, Sssp585_p32_EARWS.std(dim="models", skipna=True), len(Sssp585_p32_EARWS.coords["models"]), True) + ca.cal_mmemask(Sssp585_p32_EARWS)) >= 2.0, 1.0, 0.0)
Sdiff_EARWS_ens_mask = xr.where((ca.MME_reg_mask(Sdiff_EARWS_ens, Sdiff_EARWS.std(dim="models", skipna=True), len(Sdiff_EARWS.coords["models"]), True) + ca.cal_mmemask(Sdiff_EARWS)) >= 2.0, 1.0, 0.0)
Sdiff1_EARWS_ens_mask = xr.where((ca.MME_reg_mask(Sdiff1_EARWS_ens, Sdiff1_EARWS.std(dim="models", skipna=True), len(Sdiff1_EARWS.coords["models"]), True) + ca.cal_mmemask(Sdiff1_EARWS)) >= 2.0, 1.0, 0.0)
Sdiff2_EARWS_ens_mask = xr.where((ca.MME_reg_mask(Sdiff2_EARWS_ens, Sdiff2_EARWS.std(dim="models", skipna=True), len(Sdiff2_EARWS.coords["models"]), True) + ca.cal_mmemask(Sdiff2_EARWS)) >= 2.0, 1.0, 0.0)

# %%
# #   calculate the hgt/u/v regression onto IndR in ERA5, historical, ssp585, ssp585_p3
# preGPCP_India_JJA.coords["time"] = hgtERA5_ver_JJA.coords["time"]
# preAIR_JJA.coords["time"] = hgtERA5_ver_JJA.coords["time"]


# (
#     IndR_his_hgt_slope,
#     IndR_his_hgt_intercept,
#     IndR_his_hgt_rvalue,
#     IndR_his_hgt_pvalue,
#     IndR_his_hgt_hypothesis,
# ) = ca.dim_linregress(prehis_India_JJA, hgthis_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndR_his_u_slope,
#     IndR_his_u_intercept,
#     IndR_his_u_rvalue,
#     IndR_his_u_pvalue,
#     IndR_his_u_hypothesis,
# ) = ca.dim_linregress(prehis_India_JJA, uhis_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndR_his_v_slope,
#     IndR_his_v_intercept,
#     IndR_his_v_rvalue,
#     IndR_his_v_pvalue,
#     IndR_his_v_hypothesis,
# ) = ca.dim_linregress(prehis_India_JJA, vhis_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndR_ssp585_p3_hgt_slope,
#     IndR_ssp585_p3_hgt_intercept,
#     IndR_ssp585_p3_hgt_rvalue,
#     IndR_ssp585_p3_hgt_pvalue,
#     IndR_ssp585_p3_hgt_hypothesis,
# ) = ca.dim_linregress(pressp585_p3_India_JJA, hgtssp585_p3_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndR_ssp585_p3_u_slope,
#     IndR_ssp585_p3_u_intercept,
#     IndR_ssp585_p3_u_rvalue,
#     IndR_ssp585_p3_u_pvalue,
#     IndR_ssp585_p3_u_hypothesis,
# ) = ca.dim_linregress(pressp585_p3_India_JJA, ussp585_p3_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndR_ssp585_p3_v_slope,
#     IndR_ssp585_p3_v_intercept,
#     IndR_ssp585_p3_v_rvalue,
#     IndR_ssp585_p3_v_pvalue,
#     IndR_ssp585_p3_v_hypothesis,
# ) = ca.dim_linregress(pressp585_p3_India_JJA, vssp585_p3_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndRAIR_ERA5_hgt_slope,
#     IndRAIR_ERA5_hgt_intercept,
#     IndRAIR_ERA5_hgt_rvalue,
#     IndRAIR_ERA5_hgt_pvalue,
#     IndRAIR_ERA5_hgt_hypothesis,
# ) = ca.dim_linregress(preAIR_JJA, hgtERA5_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndRAIR_ERA5_u_slope,
#     IndRAIR_ERA5_u_intercept,
#     IndRAIR_ERA5_u_rvalue,
#     IndRAIR_ERA5_u_pvalue,
#     IndRAIR_ERA5_u_hypothesis,
# ) = ca.dim_linregress(preAIR_JJA, uERA5_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndRAIR_ERA5_v_slope,
#     IndRAIR_ERA5_v_intercept,
#     IndRAIR_ERA5_v_rvalue,
#     IndRAIR_ERA5_v_pvalue,
#     IndRAIR_ERA5_v_hypothesis,
# ) = ca.dim_linregress(preAIR_JJA, vERA5_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndRAIR_ERA5_vor_slope,
#     IndRAIR_ERA5_vor_intercept,
#     IndRAIR_ERA5_vor_rvalue,
#     IndRAIR_ERA5_vor_pvalue,
#     IndRAIR_ERA5_vor_hypothesis,
# ) = ca.dim_linregress(preAIR_JJA, vorERA5_ver_JJA)

# (
#     IndR_his_vor_slope,
#     IndR_his_vor_intercept,
#     IndR_his_vor_rvalue,
#     IndR_his_vor_pvalue,
#     IndR_his_vor_hypothesis,
# ) = ca.dim_linregress(prehis_India_JJA, vorhis_ver_JJA)

# (
#     IndR_ssp585_p3_vor_slope,
#     IndR_ssp585_p3_vor_intercept,
#     IndR_ssp585_p3_vor_rvalue,
#     IndR_ssp585_p3_vor_pvalue,
#     IndR_ssp585_p3_vor_hypothesis,
# ) = ca.dim_linregress(pressp585_p3_India_JJA, vorssp585_p3_ver_JJA)
# # %%
# #   save the regression results
# level=IndR_his_hgt_slope.coords["level"]
# lat=IndR_his_hgt_slope.coords["lat"]
# lon=IndR_his_hgt_slope.coords["lon"]

# IndRAIR_ERA5_hgt_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["level", "lat", "lon"], IndRAIR_ERA5_hgt_slope.data),
#         intercept=(["level", "lat", "lon"], IndRAIR_ERA5_hgt_intercept.data),
#         rvalue=(["level", "lat", "lon"], IndRAIR_ERA5_hgt_rvalue.data),
#         pvalue=(["level", "lat", "lon"], IndRAIR_ERA5_hgt_pvalue.data),
#         hypothesis=(["level", "lat", "lon"], IndRAIR_ERA5_hgt_hypothesis.data),
#     ),
#     coords=dict(
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="hgt fields of ERA5 regress onto 1979-2014 AIR"),
# )

# IndRAIR_ERA5_u_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["level", "lat", "lon"], IndRAIR_ERA5_u_slope.data),
#         intercept=(["level", "lat", "lon"], IndRAIR_ERA5_u_intercept.data),
#         rvalue=(["level", "lat", "lon"], IndRAIR_ERA5_u_rvalue.data),
#         pvalue=(["level", "lat", "lon"], IndRAIR_ERA5_u_pvalue.data),
#         hypothesis=(["level", "lat", "lon"], IndRAIR_ERA5_u_hypothesis.data),
#     ),
#     coords=dict(
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="u fields of ERA5 regress onto 1979-2014 AIR"),
# )

# IndRAIR_ERA5_v_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["level", "lat", "lon"], IndRAIR_ERA5_v_slope.data),
#         intercept=(["level", "lat", "lon"], IndRAIR_ERA5_v_intercept.data),
#         rvalue=(["level", "lat", "lon"], IndRAIR_ERA5_v_rvalue.data),
#         pvalue=(["level", "lat", "lon"], IndRAIR_ERA5_v_pvalue.data),
#         hypothesis=(["level", "lat", "lon"], IndRAIR_ERA5_v_hypothesis.data),
#     ),
#     coords=dict(
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="v fields of ERA5 regress onto 1979-2014 AIR"),
# )

# IndR_his_hgt_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "level", "lat", "lon"], IndR_his_hgt_slope.data),
#         intercept=(["models", "level", "lat", "lon"], IndR_his_hgt_intercept.data),
#         rvalue=(["models", "level", "lat", "lon"], IndR_his_hgt_rvalue.data),
#         pvalue=(["models", "level", "lat", "lon"], IndR_his_hgt_pvalue.data),
#         hypothesis=(["models", "level", "lat", "lon"], IndR_his_hgt_hypothesis.data),
#     ),
#     coords=dict(
#         models=models.data,
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="hgt fields of historical regress onto 1979-2014 IndR"),
# )

# IndR_his_u_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "level", "lat", "lon"], IndR_his_u_slope.data),
#         intercept=(["models", "level", "lat", "lon"], IndR_his_u_intercept.data),
#         rvalue=(["models", "level", "lat", "lon"], IndR_his_u_rvalue.data),
#         pvalue=(["models", "level", "lat", "lon"], IndR_his_u_pvalue.data),
#         hypothesis=(["models", "level", "lat", "lon"], IndR_his_u_hypothesis.data),
#     ),
#     coords=dict(
#         models=models.data,
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="u fields of historical regress onto 1979-2014 IndR"),
# )

# IndR_his_v_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "level", "lat", "lon"], IndR_his_v_slope.data),
#         intercept=(["models", "level", "lat", "lon"], IndR_his_v_intercept.data),
#         rvalue=(["models", "level", "lat", "lon"], IndR_his_v_rvalue.data),
#         pvalue=(["models", "level", "lat", "lon"], IndR_his_v_pvalue.data),
#         hypothesis=(["models", "level", "lat", "lon"], IndR_his_v_hypothesis.data),
#     ),
#     coords=dict(
#         models=models.data,
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="v fields of historical regress onto 1979-2014 IndR"),
# )

# IndR_ssp585_p3_hgt_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_slope.data),
#         intercept=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_intercept.data),
#         rvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_rvalue.data),
#         pvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_pvalue.data),
#         hypothesis=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_hypothesis.data),
#     ),
#     coords=dict(
#         models=models.data,
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="hgt fields of ssp585_p3 regress onto 2064-2099 IndR"),
# )

# IndR_ssp585_p3_u_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_slope.data),
#         intercept=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_intercept.data),
#         rvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_rvalue.data),
#         pvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_pvalue.data),
#         hypothesis=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_hypothesis.data),
#     ),
#     coords=dict(
#         models=models.data,
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="u fields of ssp585_p3 regress onto 2064-2099 IndR"),
# )

# IndR_ssp585_p3_v_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_slope.data),
#         intercept=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_intercept.data),
#         rvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_rvalue.data),
#         pvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_pvalue.data),
#         hypothesis=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_hypothesis.data),
#     ),
#     coords=dict(
#         models=models.data,
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="v fields of ssp585_p3 regress onto 2064-2099 IndR"),
# )

# IndRAIR_ERA5_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndRAIR_ERA5_hgt_regress.nc")
# IndRAIR_ERA5_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndRAIR_ERA5_u_regress.nc")
# IndRAIR_ERA5_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndRAIR_ERA5_v_regress.nc")

# IndR_his_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_hgt_regress.nc")
# IndR_his_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_u_regress.nc")
# IndR_his_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_v_regress.nc")

# IndR_ssp585_p3_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_hgt_regress.nc")
# IndR_ssp585_p3_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_u_regress.nc")
# IndR_ssp585_p3_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_v_regress.nc")
# %%
#   read the regression data
IndRAIR_ERA5_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndRAIR_ERA5_hgt_regress.nc")
IndRAIR_ERA5_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndRAIR_ERA5_u_regress.nc")
IndRAIR_ERA5_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndRAIR_ERA5_v_regress.nc")

IndR_his_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_hgt_regress.nc")
IndR_his_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_u_regress.nc")
IndR_his_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_v_regress.nc")

IndR_ssp585_p3_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_hgt_regress.nc")
IndR_ssp585_p3_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_u_regress.nc")
IndR_ssp585_p3_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_v_regress.nc")

IndRAIR_ERA5_hgt_slope = IndRAIR_ERA5_hgt_regress["slope"]
IndRAIR_ERA5_u_slope = IndRAIR_ERA5_u_regress["slope"]
IndRAIR_ERA5_v_slope = IndRAIR_ERA5_v_regress["slope"]
IndRAIR_ERA5_hgt_rvalue = IndRAIR_ERA5_hgt_regress["rvalue"]
IndRAIR_ERA5_u_rvalue = IndRAIR_ERA5_u_regress["rvalue"]
IndRAIR_ERA5_v_rvalue = IndRAIR_ERA5_v_regress["rvalue"]
IndRAIR_ERA5_hgt_pvalue = IndRAIR_ERA5_hgt_regress["pvalue"]
IndRAIR_ERA5_u_pvalue = IndRAIR_ERA5_u_regress["pvalue"]
IndRAIR_ERA5_v_pvalue = IndRAIR_ERA5_v_regress["pvalue"]

IndR_his_hgt_slope = IndR_his_hgt_regress["slope"]
IndR_his_u_slope = IndR_his_u_regress["slope"]
IndR_his_v_slope = IndR_his_v_regress["slope"]
IndR_his_hgt_rvalue = IndR_his_hgt_regress["rvalue"]
IndR_his_u_rvalue = IndR_his_u_regress["rvalue"]
IndR_his_v_rvalue = IndR_his_v_regress["rvalue"]
IndR_his_hgt_pvalue = IndR_his_hgt_regress["pvalue"]
IndR_his_u_pvalue = IndR_his_u_regress["pvalue"]
IndR_his_v_pvalue = IndR_his_v_regress["pvalue"]

IndR_ssp585_p3_hgt_slope = IndR_ssp585_p3_hgt_regress["slope"]
IndR_ssp585_p3_u_slope = IndR_ssp585_p3_u_regress["slope"]
IndR_ssp585_p3_v_slope = IndR_ssp585_p3_v_regress["slope"]
IndR_ssp585_p3_hgt_rvalue = IndR_ssp585_p3_hgt_regress["rvalue"]
IndR_ssp585_p3_u_rvalue = IndR_ssp585_p3_u_regress["rvalue"]
IndR_ssp585_p3_v_rvalue = IndR_ssp585_p3_v_regress["rvalue"]
IndR_ssp585_p3_hgt_pvalue = IndR_ssp585_p3_hgt_regress["pvalue"]
IndR_ssp585_p3_u_pvalue = IndR_ssp585_p3_u_regress["pvalue"]
IndR_ssp585_p3_v_pvalue = IndR_ssp585_p3_v_regress["pvalue"]
# %%
#   calculate the windcheck and ensmean
IndRAIR_ERA5_wind_mask = ca.wind_check(
    xr.where(IndRAIR_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRAIR_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRAIR_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRAIR_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
)

IndR_his_wind_mask = ca.wind_check(
    xr.where(IndR_his_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_his_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_his_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_his_v_pvalue <= 0.05, 1.0, 0.0),
)

IndR_ssp585_p3_wind_mask = ca.wind_check(
    xr.where(IndR_ssp585_p3_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_pvalue <= 0.05, 1.0, 0.0),
)

IndR_his_hgt_slope_ens = IndR_his_hgt_slope.mean(dim="models", skipna=True)
IndR_his_hgt_slope_ens_mask = xr.where((ca.MME_reg_mask(IndR_his_hgt_slope_ens, IndR_his_hgt_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_his_hgt_slope)) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_hgt_slope_ens = IndR_ssp585_p3_hgt_slope.mean(dim="models", skipna=True)
IndR_ssp585_p3_hgt_slope_ens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_hgt_slope_ens, IndR_ssp585_p3_hgt_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_ssp585_p3_hgt_slope)) >= 2.0, 1.0, 0.0)

IndR_his_u_slope_ens = IndR_his_u_slope.mean(dim="models", skipna=True)
IndR_his_u_slope_ens_mask = xr.where((ca.MME_reg_mask(IndR_his_u_slope_ens, IndR_his_u_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_his_u_slope)) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_u_slope_ens = IndR_ssp585_p3_u_slope.mean(dim="models", skipna=True)
IndR_ssp585_p3_u_slope_ens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_u_slope_ens, IndR_ssp585_p3_u_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_ssp585_p3_u_slope)) >= 2.0, 1.0, 0.0)

IndR_his_v_slope_ens = IndR_his_v_slope.mean(dim="models", skipna=True)
IndR_his_v_slope_ens_mask = xr.where((ca.MME_reg_mask(IndR_his_v_slope_ens, IndR_his_v_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_his_v_slope)) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_v_slope_ens = IndR_ssp585_p3_v_slope.mean(dim="models", skipna=True)
IndR_ssp585_p3_v_slope_ens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_v_slope_ens, IndR_ssp585_p3_v_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_ssp585_p3_v_slope)) >= 2.0, 1.0, 0.0)

IndR_his_hgt_rvalue_ens = ca.cal_rMME(IndR_his_hgt_rvalue, "models")
IndR_his_hgt_rvalue_ens_mask = xr.where((ca.MME_reg_mask(IndR_his_hgt_rvalue_ens, IndR_his_hgt_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_his_hgt_slope)) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_hgt_rvalue_ens = ca.cal_rMME(IndR_ssp585_p3_hgt_rvalue, "models")
IndR_ssp585_p3_hgt_rvalue_ens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_hgt_rvalue_ens, IndR_ssp585_p3_hgt_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_ssp585_p3_hgt_slope)) >= 2.0, 1.0, 0.0)

IndR_his_u_rvalue_ens = ca.cal_rMME(IndR_his_u_rvalue, "models")
IndR_his_u_rvalue_ens_mask = xr.where((ca.MME_reg_mask(IndR_his_u_rvalue_ens, IndR_his_u_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_his_u_slope)) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_u_rvalue_ens = ca.cal_rMME(IndR_ssp585_p3_u_rvalue, "models")
IndR_ssp585_p3_u_rvalue_ens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_u_rvalue_ens, IndR_ssp585_p3_u_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_ssp585_p3_u_slope)) >= 2.0, 1.0, 0.0)

IndR_his_v_rvalue_ens = ca.cal_rMME(IndR_his_v_rvalue, "models")
IndR_his_v_rvalue_ens_mask = xr.where((ca.MME_reg_mask(IndR_his_v_rvalue_ens, IndR_his_v_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_his_v_slope)) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_v_rvalue_ens = ca.cal_rMME(IndR_ssp585_p3_v_rvalue, "models")
IndR_ssp585_p3_v_rvalue_ens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_v_rvalue_ens, IndR_ssp585_p3_v_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_ssp585_p3_v_slope)) >= 2.0, 1.0, 0.0)

IndR_his_wind_ens_mask = ca.wind_check(
    xr.where(IndR_his_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_v_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_v_slope_ens_mask > 0.0, 1.0, 0.0),
)
IndR_ssp585_p3_wind_ens_mask = ca.wind_check(
    xr.where(IndR_ssp585_p3_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_slope_ens_mask > 0.0, 1.0, 0.0),
)
# %%
#   calculate the sst regress onto IndR in ERA5, historical and ssp585
# sstHad_JJA.coords["time"] = preAIR_JJA.coords["time"]
# (
#     IndR_Had_sst_slope,
#     IndR_Had_sst_intercept,
#     IndR_Had_sst_rvalue,
#     IndR_Had_sst_pvalue,
#     IndR_Had_sst_hypothesis,
# ) = ca.dim_linregress(preAIR_JJA, sstHad_JJA)

# (
#     IndR_his_sst_slope,
#     IndR_his_sst_intercept,
#     IndR_his_sst_rvalue,
#     IndR_his_sst_pvalue,
#     IndR_his_sst_hypothesis,
# ) = ca.dim_linregress(prehis_India_JJA, ssthis_JJA)

# (
#     IndR_ssp585_p3_sst_slope,
#     IndR_ssp585_p3_sst_intercept,
#     IndR_ssp585_p3_sst_rvalue,
#     IndR_ssp585_p3_sst_pvalue,
#     IndR_ssp585_p3_sst_hypothesis,
# ) = ca.dim_linregress(pressp585_p3_India_JJA, sstssp585_p3_JJA)

# #   save the result of the sst regression
# lat=IndR_Had_sst_slope.coords["lat"]
# lon=IndR_Had_sst_slope.coords["lon"]

# IndR_Had_sst_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["lat", "lon"], IndR_Had_sst_slope.data),
#         intercept=(["lat", "lon"], IndR_Had_sst_intercept.data),
#         rvalue=(["lat", "lon"], IndR_Had_sst_rvalue.data),
#         pvalue=(["lat", "lon"], IndR_Had_sst_pvalue.data),
#         hypothesis=(["lat", "lon"], IndR_Had_sst_hypothesis.data),
#     ),
#     coords=dict(
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="sst fields of HadISST regress onto 1979-2014 AIR"),
# )

# IndR_his_sst_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "lat", "lon"], IndR_his_sst_slope.data),
#         intercept=(["models", "lat", "lon"], IndR_his_sst_intercept.data),
#         rvalue=(["models", "lat", "lon"], IndR_his_sst_rvalue.data),
#         pvalue=(["models", "lat", "lon"], IndR_his_sst_pvalue.data),
#         hypothesis=(["models", "lat", "lon"], IndR_his_sst_hypothesis.data),
#     ),
#     coords=dict(
#         models=models_array,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="sst fields of historical regress onto 1979-2014 IndR"),
# )

# IndR_ssp585_p3_sst_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "lat", "lon"], IndR_ssp585_p3_sst_slope.data),
#         intercept=(["models", "lat", "lon"], IndR_ssp585_p3_sst_intercept.data),
#         rvalue=(["models", "lat", "lon"], IndR_ssp585_p3_sst_rvalue.data),
#         pvalue=(["models", "lat", "lon"], IndR_ssp585_p3_sst_pvalue.data),
#         hypothesis=(["models", "lat", "lon"], IndR_ssp585_p3_sst_hypothesis.data),
#     ),
#     coords=dict(
#         models=models_array,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="sst fields of ssp585_p3 regress onto 2064-2099 IndR"),
# )

# IndR_Had_sst_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_Had_sst_regress.nc")
# IndR_his_sst_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_sst_regress.nc")
# IndR_ssp585_p3_sst_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_sst_regress.nc")
# %%
#   read the sst regression data
IndR_Had_sst_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_Had_sst_regress.nc")
IndR_his_sst_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_sst_regress.nc")
IndR_ssp585_p3_sst_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_sst_regress.nc")

IndR_Had_sst_slope = IndR_Had_sst_regress["slope"]

IndR_Had_sst_rvalue = IndR_Had_sst_regress["rvalue"]

IndR_Had_sst_pvalue = IndR_Had_sst_regress["pvalue"]

IndR_his_sst_slope = IndR_his_sst_regress["slope"]

IndR_his_sst_rvalue = IndR_his_sst_regress["rvalue"]

IndR_his_sst_pvalue = IndR_his_sst_regress["pvalue"]

IndR_ssp585_p3_sst_slope = IndR_ssp585_p3_sst_regress["slope"]

IndR_ssp585_p3_sst_rvalue = IndR_ssp585_p3_sst_regress["rvalue"]

IndR_ssp585_p3_sst_pvalue = IndR_ssp585_p3_sst_regress["pvalue"]

IndR_his_sst_slope_ens = IndR_his_sst_slope.mean(dim="models", skipna=True)
IndR_his_sst_slope_ens_mask = xr.where((ca.MME_reg_mask(IndR_his_sst_slope_ens, IndR_his_sst_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_his_sst_slope)) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_sst_slope_ens = IndR_ssp585_p3_sst_slope.mean(dim="models", skipna=True)
IndR_ssp585_p3_sst_slope_ens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_sst_slope_ens, IndR_ssp585_p3_sst_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_ssp585_p3_sst_slope)) >= 2.0, 1.0, 0.0)

IndR_his_sst_rvalue_ens = ca.cal_rMME(IndR_his_sst_rvalue, "models")
IndR_his_sst_rvalue_ens_mask = xr.where((ca.MME_reg_mask(IndR_his_sst_rvalue_ens, IndR_his_sst_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_his_sst_slope)) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_sst_rvalue_ens = ca.cal_rMME(IndR_ssp585_p3_sst_rvalue, "models")
IndR_ssp585_p3_sst_rvalue_ens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_sst_rvalue_ens, IndR_ssp585_p3_sst_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_ssp585_p3_sst_slope)) >= 2.0, 1.0, 0.0)
# %%
#   calculate the ranking of different models but use the AIR data as observation
lat = prehis_JJA.coords["lat"]
lon = prehis_JJA.coords["lon"]
#   for 200hPa
lat_ranking_range1 = lat[(lat>=15) & (lat<=47.5)]
lon_ranking_range1 = lon[(lon>=50) & (lon<=140.0)]

#   for 850hPa
lat_ranking_range2 = lat[(lat>=15) & (lat<=37.5)]
lon_ranking_range2 = lon[(lon>=110) & (lon<=137.5)]

IndR_ranking_list = []
IndR_200hgt_pcc = []

IndR_850hgt_pcc = []

IndR_200hgt_RMSE = []
IndR_200hgt_std = []

IndR_850hgt_RMSE = []
IndR_850hgt_std = []


for num_mod, mod in enumerate(models):
    
    hgt200_pcc = ca.cal_pcc(IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0), IndR_his_hgt_slope.sel(models=mod, lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0))
    
    hgt850_pcc = ca.cal_pcc(IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0), IndR_his_hgt_slope.sel(models=mod, lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0))
    
    # IndR_ranking_list.append({"models": mod.data, "pcc": hgt_pcc+u_pcc+v_pcc})
    IndR_ranking_list.append({"models":mod.data, "pcc":hgt200_pcc + hgt850_pcc})
    
    IndR_200hgt_pcc.append(hgt200_pcc)
    IndR_850hgt_pcc.append(hgt850_pcc)
    
    IndR_200hgt_RMSE.append(np.sqrt(np.power((IndR_his_hgt_slope.sel(models=mod,lat=lat_ranking_range1,lon=lon_ranking_range1, level=200.0)-IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0)),2).mean(dim=["lat","lon"],skipna=True).data))
    
    IndR_850hgt_RMSE.append(np.sqrt(np.power((IndR_his_hgt_slope.sel(models=mod,lat=lat_ranking_range2,lon=lon_ranking_range2, level=850.0)-IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0)),2).mean(dim=["lat","lon"],skipna=True).data))
    
    IndR_200hgt_std.append(float((IndR_his_hgt_slope.sel(models=mod,lat=lat_ranking_range1,lon=lon_ranking_range1, level=200.0).std(dim=["lat","lon"],skipna=True)/IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0).std(dim=["lat","lon"],skipna=True)).data))
    
    IndR_850hgt_std.append(float((IndR_his_hgt_slope.sel(models=mod,lat=lat_ranking_range2,lon=lon_ranking_range2, level=850.0).std(dim=["lat","lon"],skipna=True)/IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0).std(dim=["lat","lon"],skipna=True)).data))


#   for MME
IndR_200hgt_pcc.append(ca.cal_pcc(IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0), IndR_his_hgt_slope_ens.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0)))
IndR_850hgt_pcc.append(ca.cal_pcc(IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0), IndR_his_hgt_slope_ens.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0)))


IndR_200hgt_RMSE.append(np.sqrt(np.power((IndR_his_hgt_slope_ens.sel(lat=lat_ranking_range1,lon=lon_ranking_range1, level=200.0)-IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0)),2).mean(dim=["lat","lon"],skipna=True).data))
IndR_850hgt_RMSE.append(np.sqrt(np.power((IndR_his_hgt_slope_ens.sel(lat=lat_ranking_range2,lon=lon_ranking_range2, level=850.0)-IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0)),2).mean(dim=["lat","lon"],skipna=True).data))



IndR_200hgt_std.append(float((IndR_his_hgt_slope_ens.sel(lat=lat_ranking_range1,lon=lon_ranking_range1, level=200.0).std(dim=["lat","lon"],skipna=True)/IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0).std(dim=["lat","lon"],skipna=True)).data))
IndR_850hgt_std.append(float((IndR_his_hgt_slope_ens.sel(lat=lat_ranking_range2,lon=lon_ranking_range2, level=850.0).std(dim=["lat","lon"],skipna=True)/IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0).std(dim=["lat","lon"],skipna=True)).data))


#   pick up the good models and calculate the gMME for hgt, u, v, precip
#   these gmodels are different from the ranking list calculated by the GPCP data
gmodels = ["CAMS-CSM1-0", "CESM2-WACCM", "CMCC-ESM2", "INM-CM4-8", "MRI-ESM2-0", "UKESM1-0-LL"]

Shis1_gens = Shis1.sel(models=gmodels).mean(dim="models", skipna=True)
Shis2_gens = Shis2.sel(models=gmodels).mean(dim="models", skipna=True)
Sdiff1_gens = Sdiff1.sel(models=gmodels).mean(dim="models", skipna=True)

Sssp585_p31_gens = Sssp585_p31.sel(models=gmodels).mean(dim="models", skipna=True)
Sssp585_p32_gens = Sssp585_p32.sel(models=gmodels).mean(dim="models", skipna=True)
Sdiff2_gens = Sdiff2.sel(models=gmodels).mean(dim="models", skipna=True)

Shis_gens = Shis.sel(models=gmodels).mean(dim="models", skipna=True)
Sssp585_p3_gens = Sssp585_p3.sel(models=gmodels).mean(dim="models", skipna=True)
Sdiff_gens = Sdiff.sel(models=gmodels).mean(dim="models", skipna=True)


Shis_gens_mask = xr.where((ca.MME_reg_mask(Shis_gens, Shis.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Shis.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sssp585_p3_gens_mask = xr.where((ca.MME_reg_mask(Sssp585_p3_gens, Sssp585_p3.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Sssp585_p3.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sdiff_gens_mask = ca.cal_mmemask(Sdiff.sel(models=gmodels))

Shis1_gens_mask = xr.where((ca.MME_reg_mask(Shis1_gens, Shis1.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Shis1.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sssp585_p31_gens_mask = xr.where((ca.MME_reg_mask(Sssp585_p31_gens, Sssp585_p31.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Sssp585_p31.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sdiff1_gens_mask = ca.cal_mmemask(Sdiff1.sel(models=gmodels))

Shis2_gens_mask = xr.where((ca.MME_reg_mask(Shis2_gens, Shis2.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Shis2.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sssp585_p32_gens_mask = xr.where((ca.MME_reg_mask(Sssp585_p32_gens, Sssp585_p32.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Sssp585_p32.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sdiff2_gens_mask = ca.cal_mmemask(Sdiff2.sel(models=gmodels))

#   calculate the area mean gens and gens mask
Shis1_WARWS_gens = Shis1_WARWS.sel(models=gmodels).mean(dim="models", skipna=True)
Shis2_WARWS_gens = Shis2_WARWS.sel(models=gmodels).mean(dim="models", skipna=True)

Sssp585_p31_WARWS_gens = Sssp585_p31_WARWS.sel(models=gmodels).mean(dim="models", skipna=True)
Sssp585_p32_WARWS_gens = Sssp585_p32_WARWS.sel(models=gmodels).mean(dim="models", skipna=True)

Sdiff1_WARWS_gens = Sdiff1_WARWS.sel(models=gmodels).mean(dim="models", skipna=True)
Sdiff2_WARWS_gens = Sdiff2_WARWS.sel(models=gmodels).mean(dim="models", skipna=True)

Shis_WARWS_gens = Shis_WARWS.sel(models=gmodels).mean(dim="models", skipna=True)
Sssp585_p3_WARWS_gens = Sssp585_p3_WARWS.sel(models=gmodels).mean(dim="models", skipna=True)
Sdiff_WARWS_gens = Sdiff_WARWS.sel(models=gmodels).mean(dim="models", skipna=True)

Shis1_EARWS_gens = Shis1_EARWS.sel(models=gmodels).mean(dim="models", skipna=True)
Shis2_EARWS_gens = Shis2_EARWS.sel(models=gmodels).mean(dim="models", skipna=True)

Sssp585_p31_EARWS_gens = Sssp585_p31_EARWS.sel(models=gmodels).mean(dim="models", skipna=True)
Sssp585_p32_EARWS_gens = Sssp585_p32_EARWS.sel(models=gmodels).mean(dim="models", skipna=True)

Sdiff1_EARWS_gens = Sdiff1_EARWS.sel(models=gmodels).mean(dim="models", skipna=True)
Sdiff2_EARWS_gens = Sdiff2_EARWS.sel(models=gmodels).mean(dim="models", skipna=True)

Shis_EARWS_gens = Shis_EARWS.sel(models=gmodels).mean(dim="models", skipna=True)
Sssp585_p3_EARWS_gens = Sssp585_p3_EARWS.sel(models=gmodels).mean(dim="models", skipna=True)
Sdiff_EARWS_gens = Sdiff_EARWS.sel(models=gmodels).mean(dim="models", skipna=True)

Shis_WARWS_gens_mask = xr.where((ca.MME_reg_mask(Shis_WARWS_gens, Shis_WARWS.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Shis_WARWS.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sssp585_p3_WARWS_gens_mask = xr.where((ca.MME_reg_mask(Sssp585_p3_WARWS_gens, Sssp585_p3_WARWS.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Sssp585_p3_WARWS.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sdiff_WARWS_gens_mask = ca.cal_mmemask(Sdiff_WARWS.sel(models=gmodels))

Shis1_WARWS_gens_mask = xr.where((ca.MME_reg_mask(Shis1_WARWS_gens, Shis1_WARWS.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Shis1_WARWS.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sssp585_p31_WARWS_gens_mask = xr.where((ca.MME_reg_mask(Sssp585_p31_WARWS_gens, Sssp585_p31_WARWS.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Sssp585_p31_WARWS.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sdiff1_WARWS_gens_mask = ca.cal_mmemask(Sdiff1_WARWS.sel(models=gmodels))

Shis2_WARWS_gens_mask = xr.where((ca.MME_reg_mask(Shis2_WARWS_gens, Shis2_WARWS.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Shis2_WARWS.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sssp585_p32_WARWS_gens_mask = xr.where((ca.MME_reg_mask(Sssp585_p32_WARWS_gens, Sssp585_p32_WARWS.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Sssp585_p32_WARWS.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sdiff2_WARWS_gens_mask = ca.cal_mmemask(Sdiff2_WARWS.sel(models=gmodels))

Shis_EARWS_gens_mask = xr.where((ca.MME_reg_mask(Shis_EARWS_gens, Shis_EARWS.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Shis_EARWS.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sssp585_p3_EARWS_gens_mask = xr.where((ca.MME_reg_mask(Sssp585_p3_EARWS_gens, Sssp585_p3_EARWS.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Sssp585_p3_EARWS.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sdiff_EARWS_gens_mask = ca.cal_mmemask(Sdiff_EARWS.sel(models=gmodels))

Shis1_EARWS_gens_mask = xr.where((ca.MME_reg_mask(Shis1_EARWS_gens, Shis1_EARWS.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Shis1_EARWS.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sssp585_p31_EARWS_gens_mask = xr.where((ca.MME_reg_mask(Sssp585_p31_EARWS_gens, Sssp585_p31_EARWS.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Sssp585_p31_EARWS.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sdiff1_EARWS_gens_mask = ca.cal_mmemask(Sdiff1_EARWS.sel(models=gmodels))

Shis2_EARWS_gens_mask = xr.where((ca.MME_reg_mask(Shis2_EARWS_gens, Shis2_EARWS.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Shis2_EARWS.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sssp585_p32_EARWS_gens_mask = xr.where((ca.MME_reg_mask(Sssp585_p32_EARWS_gens, Sssp585_p32_EARWS.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(Sssp585_p32_EARWS.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
Sdiff2_EARWS_gens_mask = ca.cal_mmemask(Sdiff2_EARWS.sel(models=gmodels))

pre_his_India_pre_slope_gens = pre_his_India_pre_slope.sel(models=gmodels).mean(dim="models", skipna=True)
pre_ssp585_p3_India_pre_slope_gens = pre_ssp585_p3_India_pre_slope.sel(models=gmodels).mean(dim="models", skipna=True)

pre_his_India_pre_slope_gens_mask = xr.where((ca.MME_reg_mask(pre_his_India_pre_slope_gens, pre_his_India_pre_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(pre_his_India_pre_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
pre_ssp585_p3_India_pre_slope_gens_mask = xr.where((ca.MME_reg_mask(pre_ssp585_p3_India_pre_slope_gens, pre_ssp585_p3_India_pre_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(pre_ssp585_p3_India_pre_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)


IndR_his_hgt_slope_gens = IndR_his_hgt_slope.sel(models=gmodels).mean(dim="models", skipna=True)
IndR_his_u_slope_gens = IndR_his_u_slope.sel(models=gmodels).mean(dim="models", skipna=True)
IndR_his_v_slope_gens = IndR_his_v_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_ssp585_p3_hgt_slope_gens = IndR_ssp585_p3_hgt_slope.sel(models=gmodels).mean(dim="models", skipna=True)
IndR_ssp585_p3_u_slope_gens = IndR_ssp585_p3_u_slope.sel(models=gmodels).mean(dim="models", skipna=True)
IndR_ssp585_p3_v_slope_gens = IndR_ssp585_p3_v_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_his_sst_slope_gens = IndR_his_sst_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_ssp585_p3_sst_slope_gens = IndR_ssp585_p3_sst_slope.sel(models=gmodels).mean(dim="models", skipna=True)


pre_his_India_pre_rvalue_gens = ca.cal_rMME(pre_his_India_pre_rvalue.sel(models=gmodels), "models")
pre_ssp585_p3_India_pre_rvalue_gens = ca.cal_rMME(pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels), "models")

IndR_his_hgt_rvalue_gens = ca.cal_rMME(IndR_his_hgt_rvalue.sel(models=gmodels), "models")
IndR_his_u_rvalue_gens = ca.cal_rMME(IndR_his_u_rvalue.sel(models=gmodels), "models")
IndR_his_v_rvalue_gens = ca.cal_rMME(IndR_his_v_rvalue.sel(models=gmodels), "models")

IndR_ssp585_p3_hgt_rvalue_gens = ca.cal_rMME(IndR_ssp585_p3_hgt_rvalue.sel(models=gmodels), "models")
IndR_ssp585_p3_u_rvalue_gens = ca.cal_rMME(IndR_ssp585_p3_u_rvalue.sel(models=gmodels), "models")
IndR_ssp585_p3_v_rvalue_gens = ca.cal_rMME(IndR_ssp585_p3_v_rvalue.sel(models=gmodels), "models")

IndR_his_sst_rvalue_gens = ca.cal_rMME(IndR_his_sst_rvalue.sel(models=gmodels), "models")
IndR_ssp585_p3_sst_rvalue_gens = ca.cal_rMME(IndR_ssp585_p3_sst_rvalue.sel(models=gmodels), "models")

IndR_his_hgt_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_hgt_slope_gens, IndR_his_hgt_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_hgt_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_his_u_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_u_slope_gens, IndR_his_u_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_u_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_his_v_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_v_slope_gens, IndR_his_v_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_v_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_his_sst_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_sst_slope_gens, IndR_his_sst_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_sst_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_hgt_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_hgt_slope_gens, IndR_ssp585_p3_hgt_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_hgt_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_u_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_u_slope_gens, IndR_ssp585_p3_u_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_u_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_v_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_v_slope_gens, IndR_ssp585_p3_v_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_v_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_sst_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_sst_slope_gens, IndR_ssp585_p3_sst_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_sst_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

pre_his_India_pre_rvalue_gens_mask = xr.where((ca.MME_reg_mask(pre_his_India_pre_rvalue_gens, pre_his_India_pre_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(pre_his_India_pre_rvalue.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
pre_ssp585_p3_India_pre_rvalue_gens_mask = xr.where((ca.MME_reg_mask(pre_ssp585_p3_India_pre_rvalue_gens, pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

IndR_his_hgt_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_hgt_rvalue_gens, IndR_his_hgt_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_hgt_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_his_u_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_u_rvalue_gens, IndR_his_u_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_u_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_his_v_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_v_rvalue_gens, IndR_his_v_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_v_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_his_sst_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_sst_rvalue_gens, IndR_his_sst_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_sst_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_hgt_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_hgt_rvalue_gens, IndR_ssp585_p3_hgt_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_hgt_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_u_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_u_rvalue_gens, IndR_ssp585_p3_u_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_u_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_v_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_v_rvalue_gens, IndR_ssp585_p3_v_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_v_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_sst_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_sst_rvalue_gens, IndR_ssp585_p3_sst_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_sst_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

IndR_his_wind_gens_mask = ca.wind_check(
    xr.where(IndR_his_u_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_v_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_u_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_v_slope_gens_mask > 0.0, 1.0, 0.0),
)

IndR_ssp585_p3_wind_gens_mask = ca.wind_check(
    xr.where(IndR_ssp585_p3_u_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_u_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_slope_gens_mask > 0.0, 1.0, 0.0),
)
#   for good models MME

IndR_200hgt_pcc.append(ca.cal_pcc(IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0), IndR_his_hgt_slope_gens.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0)))
IndR_850hgt_pcc.append(ca.cal_pcc(IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0), IndR_his_hgt_slope_gens.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0)))


IndR_200hgt_RMSE.append(np.sqrt(np.power((IndR_his_hgt_slope_gens.sel(lat=lat_ranking_range1,lon=lon_ranking_range1, level=200.0)-IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0)),2).mean(dim=["lat","lon"],skipna=True).data))
IndR_850hgt_RMSE.append(np.sqrt(np.power((IndR_his_hgt_slope_gens.sel(lat=lat_ranking_range2,lon=lon_ranking_range2, level=850.0)-IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0)),2).mean(dim=["lat","lon"],skipna=True).data))


IndR_200hgt_std.append(float((IndR_his_hgt_slope_gens.sel(lat=lat_ranking_range1,lon=lon_ranking_range1, level=200.0).std(dim=["lat","lon"],skipna=True)/IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0).std(dim=["lat","lon"],skipna=True)).data))
IndR_850hgt_std.append(float((IndR_his_hgt_slope_gens.sel(lat=lat_ranking_range2,lon=lon_ranking_range2, level=850.0).std(dim=["lat","lon"],skipna=True)/IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0).std(dim=["lat","lon"],skipna=True)).data))


print(sorted(IndR_ranking_list, key=lambda x : x["pcc"]))
# %%
#   calculate the East Asia westerly jet axis in ERA5, historical, ssp585_p3
wj_area_N = 50.0
wj_area_S = 20.0
wj_area_E = 140.0
wj_area_W = 40.0
lat_wj_range = lat[(lat>=wj_area_S) & (lat<=wj_area_N)]
lon_wj_range = lon[(lon>=wj_area_W) & (lon<=wj_area_E)]
ERA5_wj_axis = ca.cal_ridge_line(uERA5_ver_JJA.sel(level=200.0, lat=lat_wj_range, lon=lon_wj_range).mean(dim="time", skipna=True), ridge_trough="max")
his_wj_axis_lat = np.zeros((26, 41))
his_wj_axis_lon = np.zeros((26, 41))
ssp585_p3_wj_axis_lat = np.zeros((26, 41))
ssp585_p3_wj_axis_lon = np.zeros((26, 41))
for num_mod, mod in enumerate(models_array):
    his_wj_axis_lat[num_mod,:], his_wj_axis_lon[num_mod,:] = ca.cal_ridge_line(uhis_ver_JJA.sel(level=200.0, lat=lat_wj_range, lon=lon_wj_range, models=mod).mean(dim="time", skipna=True), ridge_trough="max")
    ssp585_p3_wj_axis_lat[num_mod,:], ssp585_p3_wj_axis_lon[num_mod,:] = ca.cal_ridge_line(ussp585_p3_ver_JJA.sel(level=200.0, lat=lat_wj_range, lon=lon_wj_range, models=mod).mean(dim="time", skipna=True), ridge_trough="max")
his_wj_axis = xr.DataArray(
    data=his_wj_axis_lat,
    dims=["models", "lon"],
    coords=dict(
        models=(["models"], models_array),
        lon=(["lon"], lon_wj_range.data)
    )
)
ssp585_p3_wj_axis = xr.DataArray(
    data=ssp585_p3_wj_axis_lat,
    dims=["models", "lon"],
    coords=dict(
        models=(["models"], models_array),
        lon=(["lon"], lon_wj_range.data)
    )
)
his_wj_axis_ens = his_wj_axis.mean(dim="models", skipna=True)
ssp585_p3_wj_axis_ens = ssp585_p3_wj_axis.mean(dim="models", skipna=True)

his_wj_axis_lat_gens = his_wj_axis.sel(models=gmodels).mean(dim="models", skipna=True)
ssp585_p3_wj_axis_lat_gens = ssp585_p3_wj_axis.sel(models=gmodels).mean(dim="models", skipna=True)
# %%
#   calculate the good models difference between historical run and ssp585_p3 run

pre_diff_India_pre_slope = pre_ssp585_p3_India_pre_slope - pre_his_India_pre_slope

pre_diff_India_pre_mask = ca.cal_mmemask(pre_diff_India_pre_slope)

pre_diff_India_pre_slope_gens = pre_diff_India_pre_slope.sel(models=gmodels).mean(dim="models", skipna=True)

pre_diff_India_pre_gens_mask = ca.cal_mmemask(pre_diff_India_pre_slope.sel(models=gmodels))

pre_diff_India_pre_rvalue = ca.cal_rdiff(pre_ssp585_p3_India_pre_rvalue, pre_his_India_pre_rvalue)
pre_diff_India_pre_rvalue_gens = ca.cal_rMME(pre_diff_India_pre_rvalue.sel(models=gmodels), "models")

IndR_diff_hgt_slope = IndR_ssp585_p3_hgt_slope - IndR_his_hgt_slope
IndR_diff_hgt_slope_ens = IndR_diff_hgt_slope.mean(dim="models", skipna=True)
IndR_diff_hgt_slope_gens = IndR_diff_hgt_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_diff_hgt_ens_mask = ca.cal_mmemask(IndR_diff_hgt_slope)
IndR_diff_hgt_gens_mask = ca.cal_mmemask(IndR_diff_hgt_slope.sel(models=gmodels))

IndR_diff_sst_slope = IndR_ssp585_p3_sst_slope - IndR_his_sst_slope
IndR_diff_sst_slope_ens = IndR_diff_sst_slope.mean(dim="models", skipna=True)
IndR_diff_sst_slope_gens = IndR_diff_sst_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_diff_sst_ens_mask = ca.cal_mmemask(IndR_diff_sst_slope)
IndR_diff_sst_gens_mask = ca.cal_mmemask(IndR_diff_sst_slope.sel(models=gmodels))

IndR_diff_u_slope = IndR_ssp585_p3_u_slope - IndR_his_u_slope
IndR_diff_u_slope_ens = IndR_diff_u_slope.mean(dim="models", skipna=True)
IndR_diff_u_slope_gens = IndR_diff_u_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_diff_u_ens_mask = ca.cal_mmemask(IndR_diff_u_slope)
IndR_diff_u_gens_mask = ca.cal_mmemask(IndR_diff_u_slope.sel(models=gmodels))

IndR_diff_v_slope = IndR_ssp585_p3_v_slope - IndR_his_v_slope
IndR_diff_v_slope_ens = IndR_diff_v_slope.mean(dim="models", skipna=True)
IndR_diff_v_slope_gens = IndR_diff_v_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_diff_v_ens_mask = ca.cal_mmemask(IndR_diff_v_slope)
IndR_diff_v_gens_mask = ca.cal_mmemask(IndR_diff_v_slope.sel(models=gmodels))

IndR_diff_wind_ens_mask = ca.wind_check(
    xr.where(IndR_diff_u_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_v_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_u_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_v_ens_mask > 0.0, 1.0, 0.0),
)
IndR_diff_wind_gens_mask = ca.wind_check(
    xr.where(IndR_diff_u_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_v_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_u_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_v_gens_mask > 0.0, 1.0, 0.0),
)

IndR_diff_hgt_rvalue = ca.cal_rdiff(IndR_ssp585_p3_hgt_rvalue, IndR_his_hgt_rvalue)
IndR_diff_hgt_rvalue_gens = ca.cal_rMME(IndR_diff_hgt_rvalue.sel(models=gmodels), "models")

IndR_diff_sst_rvalue = ca.cal_rdiff(IndR_ssp585_p3_sst_rvalue, IndR_his_sst_rvalue)
IndR_diff_sst_rvalue_gens = ca.cal_rMME(IndR_diff_sst_rvalue.sel(models=gmodels), "models")
IndR_diff_sst_rvalue_ens = ca.cal_rMME(IndR_diff_sst_rvalue, "models")


IndR_diff_u_rvalue = ca.cal_rdiff(IndR_ssp585_p3_u_rvalue, IndR_his_u_rvalue)
IndR_diff_u_rvalue_gens = ca.cal_rMME(IndR_diff_u_rvalue.sel(models=gmodels), "models")

IndR_diff_v_rvalue = ca.cal_rdiff(IndR_ssp585_p3_v_rvalue, IndR_his_v_rvalue)
IndR_diff_v_rvalue_gens = ca.cal_rMME(IndR_diff_v_rvalue.sel(models=gmodels), "models")
# %%
#   calculate the corr analysis between India and different variables
#   NCR
IndR_GPCP_NC_regress = stats.linregress(preAIR_JJA, preGPCP_NC_JJA)

IndR_his_NC_regress = ca.dim_linregress(prehis_India_JJA, prehis_NC_JJA)

IndR_ssp585_p3_NC_regress = ca.dim_linregress(pressp585_p3_India_JJA, pressp585_p3_NC_JJA)

IndR_diff_NC_slope = IndR_ssp585_p3_NC_regress[0] - IndR_his_NC_regress[0]
IndR_diff_NC_rvalue = ca.cal_rdiff(IndR_ssp585_p3_NC_regress[2], IndR_his_NC_regress[2])

#   KP
IndR_GPCP_KP_regress = stats.linregress(preAIR_JJA, preGPCP_KP_JJA)

IndR_his_KP_regress = ca.dim_linregress(prehis_India_JJA, prehis_KP_JJA)

IndR_ssp585_p3_KP_regress = ca.dim_linregress(pressp585_p3_India_JJA, pressp585_p3_KP_JJA)

IndR_diff_KP_slope = IndR_ssp585_p3_KP_regress[0] - IndR_his_KP_regress[0]
IndR_diff_KP_rvalue = ca.cal_rdiff(IndR_ssp585_p3_KP_regress[2], IndR_his_KP_regress[2])

#   SJ (actually Southern Japan and Korean peninsula)
IndR_GPCP_SJ_regress = stats.linregress(preAIR_JJA, preGPCP_SJ_JJA)

IndR_his_SJ_regress = ca.dim_linregress(prehis_India_JJA, prehis_SJ_JJA)

IndR_ssp585_p3_SJ_regress = ca.dim_linregress(pressp585_p3_India_JJA, pressp585_p3_SJ_JJA)

IndR_diff_SJ_slope = IndR_ssp585_p3_SJ_regress[0] - IndR_his_SJ_regress[0]
IndR_diff_SJ_rvalue = ca.cal_rdiff(IndR_ssp585_p3_SJ_regress[2], IndR_his_SJ_regress[2])

#   EAhigh
IndR_GPCP_EAhigh_regress = stats.linregress(preAIR_JJA, vorERA5_EAhigh_JJA)
IndR_his_EAhigh_regress = ca.dim_linregress(prehis_India_JJA, vorhis_EAhigh_JJA)
IndR_ssp585_p3_EAhigh_regress = ca.dim_linregress(pressp585_p3_India_JJA, vorssp585_p3_EAhigh_JJA)

IndR_diff_EAhigh_slope = IndR_ssp585_p3_EAhigh_regress[0] - IndR_his_EAhigh_regress[0]
IndR_diff_EAhigh_rvalue = ca.cal_rdiff(IndR_ssp585_p3_EAhigh_regress[2], IndR_his_EAhigh_regress[2])

#   WNPhigh
IndR_GPCP_WNPhigh_regress = stats.linregress(preAIR_JJA, vorERA5_WNPhigh_JJA)
IndR_his_WNPhigh_regress = ca.dim_linregress(prehis_India_JJA, vorhis_WNPhigh_JJA)
IndR_ssp585_p3_WNPhigh_regress = ca.dim_linregress(pressp585_p3_India_JJA, vorssp585_p3_WNPhigh_JJA)

IndR_diff_WNPhigh_slope = IndR_ssp585_p3_WNPhigh_regress[0] - IndR_his_WNPhigh_regress[0]
IndR_diff_WNPhigh_rvalue = ca.cal_rdiff(IndR_ssp585_p3_WNPhigh_regress[2], IndR_his_WNPhigh_regress[2])

#   EAhigh
IndR_GPCP_WAhigh_regress = stats.linregress(preAIR_JJA, vorERA5_WAhigh_JJA)
IndR_his_WAhigh_regress = ca.dim_linregress(prehis_India_JJA, vorhis_WAhigh_JJA)
IndR_ssp585_p3_WAhigh_regress = ca.dim_linregress(pressp585_p3_India_JJA, vorssp585_p3_WAhigh_JJA)

IndR_diff_WAhigh_slope = IndR_ssp585_p3_WAhigh_regress[0] - IndR_his_WAhigh_regress[0]
IndR_diff_WAhigh_rvalue = ca.cal_rdiff(IndR_ssp585_p3_WAhigh_regress[2], IndR_his_WAhigh_regress[2])

#   EAM index
IndR_GPCP_EAM_regress = stats.linregress(preAIR_JJA, ERA5_EAM)
IndR_his_EAM_regress = ca.dim_linregress(prehis_India_JJA, his_EAM)
IndR_ssp585_p3_EAM_regress = ca.dim_linregress(pressp585_p3_India_JJA, ssp585_p3_EAM)
IndR_diff_EAM_slope = IndR_ssp585_p3_EAM_regress[0] - IndR_his_EAM_regress[0]
IndR_diff_EAM_rvalue = ca.cal_rdiff(IndR_ssp585_p3_EAM_regress[2], IndR_his_EAM_regress[2])

#   IWF index
IndR_GPCP_IWF_regress = stats.linregress(preAIR_JJA, ERA5_IWF)
IndR_his_IWF_regress = ca.dim_linregress(prehis_India_JJA, his_IWF)
IndR_ssp585_p3_IWF_regress = ca.dim_linregress(pressp585_p3_India_JJA, ssp585_p3_IWF)
IndR_diff_IWF_slope = IndR_ssp585_p3_IWF_regress[0] - IndR_his_IWF_regress[0]
IndR_diff_IWF_rvalue = ca.cal_rdiff(IndR_ssp585_p3_IWF_regress[2], IndR_his_IWF_regress[2])

#   LKY index
IndR_GPCP_LKY_regress = stats.linregress(preAIR_JJA, ERA5_LKY)
IndR_his_LKY_regress = ca.dim_linregress(prehis_India_JJA, his_LKY)
IndR_ssp585_p3_LKY_regress = ca.dim_linregress(pressp585_p3_India_JJA, ssp585_p3_LKY)
IndR_diff_LKY_slope = IndR_ssp585_p3_LKY_regress[0] - IndR_his_LKY_regress[0]
IndR_diff_LKY_rvalue = ca.cal_rdiff(IndR_ssp585_p3_LKY_regress[2], IndR_his_LKY_regress[2])
# %%
#   plot the circulation regress onto IndR in good-models for corr coeff.
startlevel=[-1.0, -1.0, -1.0]
spacinglevel=[0.1, 0.1, 0.1]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 9), (2, 4))
    # plot_array[-1,-3:] = 0
    axs = fig.subplots(plot_array, proj=proj)

    #   set the geo_ticks and map projection to the plots
    # xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 1
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
        # India area
        x0 = India_W
        y0 = India_S
        width = India_E-India_W
        height = India_N-India_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # NC area
        x0 = NC_W
        y0 = NC_S
        width = NC_E-NC_W
        height = NC_N-NC_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndRAIR_ERA5_hgt_rvalue.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndRAIR_ERA5_hgt_rvalue.sel(level=lev), axs[0], n, np.where(IndRAIR_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndRAIR_ERA5_u_rvalue.sel(level=lev)[::ski, ::ski],
        IndRAIR_ERA5_v_rvalue.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IndRAIR_ERA5_u_rvalue.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRAIR_ERA5_v_rvalue.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="AIR & ERA5",
    )
    # ======================================
    con = axs[1].contourf(
        IndR_his_hgt_rvalue_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_his_hgt_rvalue_gens.sel(level=lev), axs[1], n, np.where(IndR_his_hgt_rvalue_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[1].quiver(
        IndR_his_u_rvalue_gens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_rvalue_gens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[1].quiver(
        IndR_his_u_rvalue_gens.sel(level=lev).where(IndR_his_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_rvalue_gens.sel(level=lev).where(IndR_his_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="gMME",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels):
        con = axs[num_mod+2].contourf(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
        )
        sepl.plt_sig(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod+2], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+2].quiver(
            IndR_his_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_his_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+2].quiver(
            IndR_his_u_rvalue.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_his_v_rvalue.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+2].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+2].format(
            rtitle="1979-2014", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))

#   plot the circulation regress onto IndR in good-models for reg coeff.
startlevel=[-15, -8, -6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 9), (2, 4))
    # plot_array[-1,-3:] = 0
    axs = fig.subplots(plot_array, proj=proj)

    #   set the geo_ticks and map projection to the plots
    # xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 1
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
        # India area
        x0 = India_W
        y0 = India_S
        width = India_E-India_W
        height = India_N-India_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # NC area
        x0 = NC_W
        y0 = NC_S
        width = NC_E-NC_W
        height = NC_N-NC_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndRAIR_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndRAIR_ERA5_hgt_slope.sel(level=lev), axs[0], n, np.where(IndRAIR_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndRAIR_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
        IndRAIR_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IndRAIR_ERA5_u_slope.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRAIR_ERA5_v_slope.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    if lev == 200.0:
        axs[0].line(ERA5_wj_axis[1], ERA5_wj_axis[0], lw=1.3, color="green6", ls="--")
        axs[0].contour(
            uERA5_ver_JJA.sel(level=200.0).mean(dim="time").loc[0.0:,:],
            color="green6",
            values=[27.5, 32.5],
            vmin=25.0,
            vmax=30.0,
            zorder=0.8
        )
    axs[0].format(
        rtitle="1979-2014", ltitle="AIR & ERA5",
    )
    # ======================================
    con = axs[1].contourf(
        IndR_his_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_gens.sel(level=lev), axs[1], n, np.where(IndR_his_hgt_slope_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    
    axs[1].quiver(
        IndR_his_u_slope_gens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_slope_gens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[1].quiver(
        IndR_his_u_slope_gens.sel(level=lev).where(IndR_his_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_slope_gens.sel(level=lev).where(IndR_his_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    if lev == 200.0:
        axs[1].line(his_wj_axis.coords["lon"], his_wj_axis_lat_gens, lw=1.3, color="green6", ls="--")
        axs[1].contour(
            uhis_ver_JJA.sel(level=200.0, models=gmodels).mean(dim=["time", "models"]).loc[0.0:,:],
            color="green6",
            values=[27.5, 32.5],
            vmin=25.0,
            vmax=30.0,
            zorder=0.8,
        )
    axs[1].format(
        rtitle="1979-2014", ltitle="gMME",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels):
        con = axs[num_mod+2].contourf(
            IndR_his_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IndR_his_hgt_slope.sel(models=mod,level=lev), axs[num_mod+2], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+2].quiver(
            IndR_his_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_his_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+2].quiver(
            IndR_his_u_slope.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_his_v_slope.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+2].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        if lev == 200.0:
            axs[num_mod+2].line(his_wj_axis.coords["lon"], his_wj_axis.sel(models=mod), lw=1.3, color="green6", ls="--")
            axs[num_mod+2].contour(
            uhis_ver_JJA.sel(level=200.0, models=mod).mean(dim="time").loc[0.0:,:],
            color="green6",
            values=[27.5, 32.5],
            vmin=25.0,
            vmax=30.0,
            zorder=0.8,
        )
        axs[num_mod+2].format(
            rtitle="1979-2014", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))

# %%
#   plot the circulation regress on IndR in good-models for corr coeff. in ssp585_p3
startlevel=[-1.0, -1.0, -1.0]
spacinglevel=[0.1, 0.1, 0.1]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 10), (3, 3))
    plot_array[-1,-2:] = 0
    axs = fig.subplots(plot_array, proj=proj)

    #   set the geo_ticks and map projection to the plots
    # xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 1
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
        # India area
        x0 = India_W
        y0 = India_S
        width = India_E-India_W
        height = India_N-India_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # NC area
        x0 = NC_W
        y0 = NC_S
        width = NC_E-NC_W
        height = NC_N-NC_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndR_ssp585_p3_hgt_rvalue_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_ssp585_p3_hgt_rvalue_gens.sel(level=lev), axs[0], n, np.where(IndR_ssp585_p3_hgt_rvalue_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndR_ssp585_p3_u_rvalue_gens.sel(level=lev)[::ski, ::ski],
        IndR_ssp585_p3_v_rvalue_gens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IndR_ssp585_p3_u_rvalue_gens.sel(level=lev).where(IndR_ssp585_p3_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_ssp585_p3_v_rvalue_gens.sel(level=lev).where(IndR_ssp585_p3_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="2064-2099", ltitle="gMME",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels):
        con = axs[num_mod+1].contourf(
            IndR_ssp585_p3_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
        )
        sepl.plt_sig(
            IndR_ssp585_p3_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod+1], n, np.where(IndR_ssp585_p3_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+1].quiver(
            IndR_ssp585_p3_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_ssp585_p3_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+1].quiver(
            IndR_ssp585_p3_u_rvalue.sel(models=mod,level=lev).where(IndR_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_ssp585_p3_v_rvalue.sel(models=mod,level=lev).where(IndR_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+1].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="2064-2099", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))

#   plot the circulation regress on IndR in good-models for reg coeff. in ssp585_p3
startlevel=[-15, -8, -6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 10), (3, 3))
    plot_array[-1,-2:] = 0
    axs = fig.subplots(plot_array, proj=proj)

    #   set the geo_ticks and map projection to the plots
    # xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 1
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
        # India area
        x0 = India_W
        y0 = India_S
        width = India_E-India_W
        height = India_N-India_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # NC area
        x0 = NC_W
        y0 = NC_S
        width = NC_E-NC_W
        height = NC_N-NC_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndR_ssp585_p3_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_ssp585_p3_hgt_slope_gens.sel(level=lev), axs[0], n, np.where(IndR_ssp585_p3_hgt_slope_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndR_ssp585_p3_u_slope_gens.sel(level=lev)[::ski, ::ski],
        IndR_ssp585_p3_v_slope_gens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IndR_ssp585_p3_u_slope_gens.sel(level=lev).where(IndR_ssp585_p3_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_ssp585_p3_v_slope_gens.sel(level=lev).where(IndR_ssp585_p3_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    if lev == 200.0:
        axs[0].line(ssp585_p3_wj_axis.coords["lon"], ssp585_p3_wj_axis_lat_gens, lw=1.3, color="green6", ls="--")
        axs[0].contour(
            ussp585_p3_ver_JJA.sel(level=200.0, models=gmodels).mean(dim=["time", "models"]).loc[0.0:,:],
            color="green6",
            values=[27.5, 32.5],
            vmin=25.0,
            vmax=30.0,
            zorder=0.8,
        )
    axs[0].format(
        rtitle="2064-2099", ltitle="gMME",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels):
        con = axs[num_mod+1].contourf(
            IndR_ssp585_p3_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IndR_ssp585_p3_hgt_slope.sel(models=mod,level=lev), axs[num_mod+1], n, np.where(IndR_ssp585_p3_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+1].quiver(
            IndR_ssp585_p3_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_ssp585_p3_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+1].quiver(
            IndR_ssp585_p3_u_slope.sel(models=mod,level=lev).where(IndR_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_ssp585_p3_v_slope.sel(models=mod,level=lev).where(IndR_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+1].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        if lev == 200.0:
            axs[num_mod+1].line(ssp585_p3_wj_axis.coords["lon"], ssp585_p3_wj_axis.sel(models=mod), lw=1.3, color="green6", ls="--")
            axs[num_mod+1].contour(
            ussp585_p3_ver_JJA.sel(level=200.0, models=mod).mean(dim="time").loc[0.0:,:],
            color="green6",
            values=[27.5, 32.5],
            vmin=25.0,
            vmax=30.0,
            zorder=0.8,
        )
        axs[num_mod+1].format(
            rtitle="2064-2099", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the precipitation fields of good-models for corr coeff. in ssp585_p3
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 10), (3, 3))
plot_array[-1,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # India area
    x0 = India_W
    y0 = India_S
    width = India_E-India_W
    height = India_N-India_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_ssp585_p3_India_pre_rvalue_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    
    )
sepl.plt_sig(
    pre_ssp585_p3_India_pre_rvalue_gens, axs[0], n, np.where(pre_ssp585_p3_India_pre_rvalue_gens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[0].format(
    rtitle="2064-2099", ltitle="gMME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+1].contourf(
    pre_ssp585_p3_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
    sepl.plt_sig(
        pre_ssp585_p3_India_pre_rvalue.sel(models=mod), axs[num_models+1], n, np.where(pre_ssp585_p3_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.10), "bright purple", 4.0,
    )

    axs[num_models+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")

#   plot the precipitation fields of good-models for reg coeff. in ssp585_p3
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 10), (3, 3))
plot_array[-1,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # India area
    x0 = India_W
    y0 = India_S
    width = India_E-India_W
    height = India_N-India_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_ssp585_p3_India_pre_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0,2.1, 0.1),
    zorder=0.8,
    extend="both"
    )
sepl.plt_sig(
    pre_ssp585_p3_India_pre_slope_gens, axs[0], n, np.where(pre_ssp585_p3_India_pre_slope_gens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[0].format(
    rtitle="2064-2099", ltitle="gMME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+1].contourf(
    pre_ssp585_p3_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0,2.1, 0.1),
    zorder=0.8,
    extend="both"
    )
    sepl.plt_sig(
        pre_ssp585_p3_India_pre_slope.sel(models=mod), axs[num_models+1], n, np.where(pre_ssp585_p3_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.10), "bright purple", 4.0,
    )

    axs[num_models+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the corr coeff. precipitation difference between historical and ssp585
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 10), (3, 3))
plot_array[-1,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # India area
    x0 = India_W
    y0 = India_S
    width = India_E-India_W
    height = India_N-India_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_diff_India_pre_rvalue_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    
    )

axs[0].format(
    rtitle="diff", ltitle="gMME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+1].contourf(
    pre_diff_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )

    axs[num_models+1].format(
        rtitle="diff", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")

#   plot the reg coeff. precipitation difference between historical and ssp585
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 10), (3, 3))
plot_array[-1,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # India area
    x0 = India_W
    y0 = India_S
    width = India_E-India_W
    height = India_N-India_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_diff_India_pre_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both"
    )

axs[0].format(
    rtitle="diff", ltitle="gMME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+1].contourf(
    pre_diff_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both"
    )

    axs[num_models+1].format(
        rtitle="diff", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the circulation regress onto IndR in good-models for corr coeff. in diff
startlevel=[-1.0, -1.0, -1.0]
spacinglevel=[0.1, 0.1, 0.1]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 10), (3, 3))
    plot_array[-1,-2:] = 0
    axs = fig.subplots(plot_array, proj=proj)

    #   set the geo_ticks and map projection to the plots
    # xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 1
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
        # India area
        x0 = India_W
        y0 = India_S
        width = India_E-India_W
        height = India_N-India_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # NC area
        x0 = NC_W
        y0 = NC_S
        width = NC_E-NC_W
        height = NC_N-NC_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndR_diff_hgt_rvalue_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_diff_hgt_rvalue_gens.sel(level=lev), axs[0], n, np.where(IndR_diff_hgt_gens_mask.sel(level=lev)[::n, ::n] > 0), "bright purple", 4.0,
    )
    
    axs[0].quiver(
        IndR_diff_u_rvalue_gens.sel(level=lev)[::ski, ::ski],
        IndR_diff_v_rvalue_gens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )
    
    m = axs[0].quiver(
        IndR_diff_u_rvalue_gens.sel(level=lev).where(IndR_diff_wind_gens_mask.sel(level=lev)>0)[::ski, ::ski],
        IndR_diff_v_rvalue_gens.sel(level=lev).where(IndR_diff_wind_gens_mask.sel(level=lev)>0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )
    
    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="diff", ltitle="gMME",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels):
        con = axs[num_mod+1].contourf(
            IndR_diff_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
        )

        m = axs[num_mod+1].quiver(
            IndR_diff_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_diff_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+1].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="diff", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))

#   plot the circulation regress onto IndR in good-models for reg coeff. in diff
startlevel=[-22, -15, -10]
spacinglevel=[1.1, 0.75, 0.5]
scalelevel=[0.14, 0.13, 0.13]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 10), (3, 3))
    plot_array[-1,-2:] = 0
    axs = fig.subplots(plot_array, proj=proj)

    #   set the geo_ticks and map projection to the plots
    # xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 1
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
        # India area
        x0 = India_W
        y0 = India_S
        width = India_E-India_W
        height = India_N-India_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # NC area
        x0 = NC_W
        y0 = NC_S
        width = NC_E-NC_W
        height = NC_N-NC_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndR_diff_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.0},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_diff_hgt_rvalue_gens.sel(level=lev), axs[0], n, np.where(IndR_diff_hgt_gens_mask.sel(level=lev)[::n, ::n] > 0), "bright purple", 4.0,
    )

    axs[0].quiver(
        IndR_diff_u_slope_gens.sel(level=lev)[::ski, ::ski],
        IndR_diff_v_slope_gens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )
    
    m = axs[0].quiver(
        IndR_diff_u_slope_gens.sel(level=lev).where(IndR_diff_wind_gens_mask.sel(level=lev)>0)[::ski, ::ski],
        IndR_diff_v_slope_gens.sel(level=lev).where(IndR_diff_wind_gens_mask.sel(level=lev)>0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    if lev == 200.0:
        axs[0].line(his_wj_axis.coords["lon"], his_wj_axis_lat_gens, lw=1.3, color="grape6", ls="--")
        axs[0].line(ssp585_p3_wj_axis.coords["lon"], ssp585_p3_wj_axis_lat_gens, lw=1.3, color="green6", ls="--")
        axs[0].contour(
            uhis_ver_JJA.sel(level=200.0, models=gmodels).mean(dim=["time", "models"]).loc[0.0:,:],
            color="grape6",
            values=[27.5, 32.5],
            vmin=25.0,
            vmax=30.0,
            zorder=0.8,
        )
        axs[0].contour(
            ussp585_p3_ver_JJA.sel(level=200.0, models=gmodels).mean(dim=["time", "models"]).loc[0.0:,:],
            color="green6",
            values=[27.5, 32.5],
            vmin=25.0,
            vmax=30.0,
            zorder=0.8,
        )
    axs[0].format(
        rtitle="diff", ltitle="gMME",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels):
        con = axs[num_mod+1].contourf(
            IndR_diff_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.0},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )

        m = axs[num_mod+1].quiver(
            IndR_diff_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_diff_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+1].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        if lev == 200.0:
            axs[num_mod+1].line(his_wj_axis.coords["lon"], his_wj_axis.sel(models=mod), lw=1.3, color="grape6")
            axs[num_mod+1].line(ssp585_p3_wj_axis.coords["lon"], ssp585_p3_wj_axis.sel(models=mod), lw=1.3, color="green6")
            axs[num_mod+1].contour(
            uhis_ver_JJA.sel(level=200.0, models=mod).mean(dim="time").loc[0.0:,:],
            color="grape6",
            values=[27.5, 32.5],
            vmin=25.0,
            vmax=30.0,
            zorder=0.8,
            )
            axs[num_mod+1].contour(
            ussp585_p3_ver_JJA.sel(level=200.0, models=mod).mean(dim="time").loc[0.0:,:],
            color="green6",
            values=[27.5, 32.5],
            vmin=25.0,
            vmax=30.0,
            zorder=0.8,
            )
        axs[num_mod+1].format(
            rtitle="diff", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the singular scatter-plot for good models for reg coeff.
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
# cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 8, left=0.5)
cycle = pplt.Cycle("Qual1", 7)

#   hist-GPCP
axs[0].scatter(0.75, IndR_GPCP_NC_regress[0],marker="s", labels="GPCP", legend_kw={"ncols":4}, color="blue5", legend="b")
#   hist-gMME
axs[0].scatter(0.75, IndR_his_NC_regress[0].sel(models=gmodels).mean(dim="models", skipna=True), marker="*", labels="gMME", legend_kw={"ncols":4}, cycle=cycle, legend="b", markersize=100)
#   hist-gmodels
for num_models, mod in enumerate(gmodels):
    axs[0].scatter(0.75, IndR_his_NC_regress[0].sel(models=mod), cycle=cycle, legend="b", legend_kw={"ncols":4}, labels=mod, marker=".", markersize=100)
    
#   ssp585_p3-gMME
axs[0].scatter(1.50, IndR_ssp585_p3_NC_regress[0].sel(models=gmodels).mean(dim="models", skipna=True), marker="*", cycle=cycle, markersize=100)

#   ssp585_p3-gmodels
for num_models, mod in enumerate(gmodels):
    axs[0].scatter(1.50, IndR_ssp585_p3_NC_regress[0].sel(models=mod), cycle=cycle, marker=".", markersize=100)

axs[0].format(xlocator=[0.75, 1.5], ylim=(-0.2,0.5), xlim=(0,2.25), xformatter=["historical", "ssp585_p3"], tickminor=False)
fig.format(suptitle="NC reg IndR")
# %%
#   plot the singular scatter-plot for good models for reg coeff.
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
# cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 8, left=0.5)
cycle = pplt.Cycle("Qual1", 7)

#   hist-GPCP
axs[0].scatter(0.75, IndR_GPCP_NC_regress[2],marker="s", labels="GPCP", legend_kw={"ncols":4}, color="blue5", legend="b")
#   hist-gMME
axs[0].scatter(0.75, IndR_his_NC_regress[2].sel(models=gmodels).mean(dim="models", skipna=True), marker="*", labels="gMME", legend_kw={"ncols":4}, cycle=cycle, legend="b", markersize=100)
#   hist-gmodels
for num_models, mod in enumerate(gmodels):
    axs[0].scatter(0.75, IndR_his_NC_regress[2].sel(models=mod), cycle=cycle, legend="b", legend_kw={"ncols":4}, labels=mod, marker=".", markersize=100)
    
#   ssp585_p3-gMME
axs[0].scatter(1.50, IndR_ssp585_p3_NC_regress[2].sel(models=gmodels).mean(dim="models", skipna=True), marker="*", cycle=cycle, markersize=100)

#   ssp585_p3-gmodels
for num_models, mod in enumerate(gmodels):
    axs[0].scatter(1.50, IndR_ssp585_p3_NC_regress[2].sel(models=mod), cycle=cycle, marker=".", markersize=100)

axs[0].format(xlocator=[0.75, 1.5], ylim=(-0.2,0.6), xlim=(0,2.25), xformatter=["historical", "ssp585_p3"], tickminor=False)
fig.format(suptitle="corr(IndR, NCR)")
# %%
#   plot the bar-plot of the IndR related NC precipitation for historical and ssp585_p3 (reg coeff.)
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_NC_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_NC_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_NC_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_NC_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_NC_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_NC_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-0.7,0.7),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=11, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and NCR")
# %%
#   plot the bar-plot of the IndR related NC precipitation for historical and ssp585_p3 (corr coeff.)
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_NC_regress[2].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_NC_regress[2].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_NC_rvalue.sel(models=gmodels).data
plot_data[-1,0] = ca.cal_rMME(IndR_his_NC_regress[2].sel(models=gmodels), "models").data
plot_data[-1,1] = ca.cal_rMME(IndR_ssp585_p3_NC_regress[2].sel(models=gmodels), "models").data
plot_data[-1,2] = ca.cal_rMME(IndR_diff_NC_rvalue.sel(models=gmodels), "models").data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
axs[0].axhline(ca.cal_rlim1(0.90, 36),lw=1.5,color="grey7",ls='--')
axs[0].axhline(-ca.cal_rlim1(0.90, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-0.7,0.7),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=11, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Corr. Coeff. IndR and NCR")
# %%
#   plot the rvalue of hgt&u&v regress onto IndR in ERA5 and historical, but for AIR data
startlevel=[-1.0, -1.0, -1.0]
spacinglevel=[0.1, 0.1, 0.1]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 31), (6, 5))
    plot_array[-1,-2:] = 0
    axs = fig.subplots(plot_array, proj=proj)

    #   set the geo_ticks and map projection to the plots
    # xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 1
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
        # India area
        x0 = India_W
        y0 = India_S
        width = India_E-India_W
        height = India_N-India_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # NC area
        x0 = NC_W
        y0 = NC_S
        width = NC_E-NC_W
        height = NC_N-NC_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndRAIR_ERA5_hgt_rvalue.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndRAIR_ERA5_hgt_rvalue.sel(level=lev), axs[0], n, np.where(IndRAIR_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndRAIR_ERA5_u_rvalue.sel(level=lev)[::ski, ::ski],
        IndRAIR_ERA5_v_rvalue.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IndRAIR_ERA5_u_rvalue.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRAIR_ERA5_v_rvalue.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="AIR & ERA5",
    )
    # ======================================
    con = axs[1].contourf(
        IndR_his_hgt_rvalue_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_his_hgt_rvalue_ens.sel(level=lev), axs[1], n, np.where(IndR_his_hgt_rvalue_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[1].quiver(
        IndR_his_u_rvalue_ens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_rvalue_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[1].quiver(
        IndR_his_u_rvalue_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_rvalue_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+2].contourf(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
        )
        sepl.plt_sig(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod+2], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+2].quiver(
            IndR_his_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_his_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+2].quiver(
            IndR_his_u_rvalue.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_his_v_rvalue.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+2].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+2].format(
            rtitle="1979-2014", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the correlation coefficients rvalue in AIR, GPCP, historical run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (6, 5))
plot_array[5,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # India area
    x0 = India_W
    y0 = India_S
    width = India_E-India_W
    height = India_N-India_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_AIR_India_pre_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_AIR_India_pre_rvalue, axs[0], n, np.where(pre_AIR_India_pre_pvalue[::n, ::n] <= 0.10), "bright purple", 4.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="AIR",
)

# ===================================================
con = axs[1].contourf(
    pre_his_India_pre_rvalue_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    
    )
sepl.plt_sig(
    pre_his_India_pre_rvalue_ens, axs[1], n, np.where(pre_his_India_pre_rvalue_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(pre_his_India_pre_slope.coords["models"].data):
    con = axs[num_models+2].contourf(
    pre_his_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
    sepl.plt_sig(
        pre_his_India_pre_rvalue.sel(models=mod), axs[num_models+2], n, np.where(pre_his_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.10), "bright purple", 4.0,
    )

    axs[num_models+2].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   fig. 3 plot the taylor-diagram
labels = list(models.data)
labels.append("MME")
labels.append("gMME")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#绘图
fig=plt.figure(figsize=(12,8),dpi=300)
plt.rc('font',family='Arial',size=13)
plt.rcParams["axes.facecolor"] = "white"

#调用函数
ax1=fig.add_subplot(111,projection='polar')
box = ax1.get_position()
ax1.set_position([0, box.y0, box.width*1.2, box.height])
# ax1.text(0.6,0.1,'(a)',fontsize=15)
# tar(ax1,np.array(IndR_EAM_pcc),np.array(IndR_EAM_std),labels)
sepl.taylor_diagram(ax1,np.array(IndR_200hgt_pcc),np.array(IndR_200hgt_std), dotlables=labels, lables=True, color="r")
sepl.taylor_diagram(ax1,np.array(IndR_850hgt_pcc),np.array(IndR_850hgt_std), color="b")
plt.legend(loc="center left", bbox_to_anchor=(1.1,0.5), ncol=2, frameon=True, numpoints=1, handlelength=0)
# %%
#   calculate the climatology relative vorticity in the historical and ssp585_p3
vorERA5_cli_ver_JJA = vorERA5_ver_JJA.sel(time=vorERA5_ver_JJA.time.dt.year>=1979).mean(dim="time", skipna=True)
vorhis_cli_ver_JJA = vorhis_ver_JJA.sel(time=vorhis_ver_JJA.time.dt.year>=1979).mean(dim="time", skipna=True)
vorssp585_p3_cli_ver_JJA = vorssp585_ver_JJA.sel(time=vorssp585_ver_JJA.time.dt.year>=2064).mean(dim="time", skipna=True)
# %%
#   plot the climatology relative vorticity in ERA5 and historical
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (6, 5))
plot_array[-1,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # India area
    x0 = India_W
    y0 = India_S
    width = India_E-India_W
    height = India_N-India_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    vorERA5_cli_ver_JJA,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-3.5e-5, 3.5e-5+3.5e-6, 3.5e-6),
    zorder=0.8,
    extend="both"
    )

axs[0].format(
    rtitle="1979-2014", ltitle="ERA5",
)
# ===================================================
con = axs[1].contourf(
    vorhis_cli_ver_JJA.mean(dim="models", skipna=True),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-3.5e-5, 3.5e-5+3.5e-6, 3.5e-6),
    zorder=0.8,
    extend="both"
    )

axs[1].format(
    rtitle="1979-2014", ltitle="MME",
)
# ===================================================
for num_mod, mod in enumerate(models):
    con = axs[num_mod+2].contourf(
        vorhis_cli_ver_JJA.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-3.5e-5, 3.5e-5+3.5e-6, 3.5e-6),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+2].format(
        rtitle="1979-2014", ltitle="{}".format(mod.data),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="200hPa vor reg IndR")
# %%
#   plot the climatology relative vorticity in ssp585_p3
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (6, 5))
plot_array[-1,-3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # India area
    x0 = India_W
    y0 = India_S
    width = India_E-India_W
    height = India_N-India_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    vorssp585_p3_cli_ver_JJA.mean(dim="models", skipna=True),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-3.5e-5, 3.5e-5+3.5e-6, 3.5e-6),
    zorder=0.8,
    extend="both"
    )

axs[0].format(
    rtitle="2064-2099", ltitle="MME",
)
# ===================================================
for num_mod, mod in enumerate(models):
    con = axs[num_mod+1].contourf(
        vorssp585_p3_cli_ver_JJA.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-3.5e-5, 3.5e-5+3.5e-6, 3.5e-6),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod.data),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="200hPa vor reg IndR")
# %%
#   calculate the diff of climatology relative vorticity in diff
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (6, 5))
plot_array[-1,-3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # India area
    x0 = India_W
    y0 = India_S
    width = India_E-India_W
    height = India_N-India_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    vorssp585_p3_cli_ver_JJA.mean(dim="models", skipna=True)-vorhis_cli_ver_JJA.mean(dim="models", skipna=True),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0e-5, 1.0e-5+1.0e-6, 1.0e-6),
    zorder=0.8,
    extend="both"
    )

axs[0].format(
    rtitle="diff", ltitle="MME",
)
# ===================================================
for num_mod, mod in enumerate(models):
    con = axs[num_mod+1].contourf(
        vorssp585_p3_cli_ver_JJA.sel(models=mod) - vorhis_cli_ver_JJA.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-1.0e-5, 1.0e-5+1.0e-6, 1.0e-6),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+1].format(
        rtitle="diff", ltitle="{}".format(mod.data),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="200hPa vor reg IndR")
# %%
#   only plot the precipitation regress onto AIR and IndR in MME
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
# plot_array = np.reshape(range(1, 31), (6, 5))
# plot_array[5,-1] = 0
axs = fig.subplots(ncols=1, nrows=2, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # India area
    x0 = India_W
    y0 = India_S
    width = India_E-India_W
    height = India_N-India_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # SJ-KP area
    x0 = SJ_W
    y0 = SJ_S
    width = SJ_E-SJ_W
    height = SJ_N-SJ_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_AIR_India_pre_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_AIR_India_pre_rvalue, axs[0], n, np.where(pre_AIR_India_pre_pvalue[::n, ::n] <= 0.10), "bright purple", 4.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="AIR and GPCP",
)
# ===================================================
con = axs[1].contourf(
    pre_his_India_pre_rvalue_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    
    )
sepl.plt_sig(
    pre_his_India_pre_rvalue_ens, axs[1], n, np.where(pre_his_India_pre_rvalue_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="MME",
)
# ===================================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   fig. 2 only plot the circulation regress onto AIR and IndR in MME
startlevel=[-15, -8, -6]
spacinglevel=[1.5, 0.8, 0.6]
scalelevel=[0.23, 0.17, 0.14]

pplt.rc.grid = False
pplt.rc.reso = "lo"
pplt.rc["figure.facecolor"] = "white"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=6.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 7), (3, 2))
# plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(0, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ======================================
for ax in axs:
    rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
    ax.add_patch(rect)
    # # India area
    # x0 = India_W
    # y0 = India_S
    # width = India_E-India_W
    # height = India_N-India_S
    # sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # # NC area
    # x0 = NC_W
    # y0 = NC_S
    # width = NC_E-NC_W
    # height = NC_N-NC_S
    # sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # # SJ-KP area
    # x0 = SJ_W
    # y0 = SJ_S
    # width = SJ_E-SJ_W
    # height = SJ_N-SJ_S
    # sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ======================================
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    if lev == 200.0:
        for ax in axs[num_lev, :]:
            x0 = 50
            y0 = 15
            width = 90
            height = 32.5
            sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="bright purple", linestyle="-")
            # #   EAM index
            # x0 = 110.0
            # y0 = 40.0
            # width = 40.0
            # height = 10.0
            # sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="green8", linestyle="-")
            # x0 = 110.0
            # y0 = 25.0
            # width = 40.0
            # height = 10.0
            # sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="green8", linestyle="-")
    elif lev == 850.0:
        for ax in axs[num_lev, :]:
            x0 = 110
            y0 = 15
            width = 27.5
            height = 22.5
            sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="bright purple", linestyle="-")
    con = axs[num_lev, 0].contourf(
        IndRAIR_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndRAIR_ERA5_hgt_slope.sel(level=lev), axs[num_lev, 0], n, np.where(IndRAIR_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_lev, 0].quiver(
        IndRAIR_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
        IndRAIR_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[num_lev, 0].quiver(
        IndRAIR_ERA5_u_slope.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRAIR_ERA5_v_slope.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[num_lev, 0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev, 0].format(
        ltitle="1979-2014 {:.0f}hPa".format(lev), rtitle="AIR & ERA5",
    )
    # ======================================
    con = axs[num_lev, 1].contourf(
        IndR_his_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_ens.sel(level=lev), axs[num_lev, 1], n, np.where(IndR_his_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[num_lev, 1].quiver(
        IndR_his_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_slope_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[num_lev, 1].quiver(
        IndR_his_u_slope_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_slope_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[num_lev, 1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev, 1].format(
        ltitle="1979-2014 {:.0f}hPa".format(lev), rtitle="MME",
    )
    cb = axs[num_lev, 1].colorbar(con, loc="r", width=0.13, length=0.8, label="", ticklabelsize=7)
    cb.set_ticks(np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]*2))
# ======================================
fig.format(abc="(a)", abcloc="l")
# %%
#   plot the avalue of hgt&u&v regress onto IndR in ERA5 and historical
startlevel=[-15, -8, -6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 31), (6, 5))
    plot_array[-1,-2:] = 0
    axs = fig.subplots(plot_array, proj=proj)

    #   set the geo_ticks and map projection to the plots
    # xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 1
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
        # India area
        x0 = India_W
        y0 = India_S
        width = India_E-India_W
        height = India_N-India_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # NC area
        x0 = NC_W
        y0 = NC_S
        width = NC_E-NC_W
        height = NC_N-NC_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # SJ-KP area
        x0 = SJ_W
        y0 = SJ_S
        width = SJ_E-SJ_W
        height = SJ_N-SJ_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndRAIR_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndRAIR_ERA5_hgt_slope.sel(level=lev), axs[0], n, np.where(IndRAIR_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndRAIR_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
        IndRAIR_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IndRAIR_ERA5_u_slope.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRAIR_ERA5_v_slope.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="AIR & ERA5",
    )
    # ======================================
    con = axs[1].contourf(
        IndR_his_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_ens.sel(level=lev), axs[1], n, np.where(IndR_his_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[1].quiver(
        IndR_his_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_slope_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[1].quiver(
        IndR_his_u_slope_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_slope_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+2].contourf(
            IndR_his_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IndR_his_hgt_slope.sel(models=mod,level=lev), axs[num_mod+2], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+2].quiver(
            IndR_his_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_his_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+2].quiver(
            IndR_his_u_slope.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_his_v_slope.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+2].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+2].format(
            rtitle="1979-2014", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the correlation scatter-plot, x:pcc, y:corr(IndR, NCR)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
m = axs[0].scatter(1.0, IndR_GPCP_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="GPCP", marker="s", color="blue5")
for num_models, mod in enumerate(models_array):
    m = axs[0].scatter((np.array(IndR_200hgt_pcc)+np.array(IndR_850hgt_pcc))[num_models]/2.0, IndR_his_NC_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod)
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter((np.array(IndR_200hgt_pcc)+np.array(IndR_850hgt_pcc))[26]/2.0, ca.cal_rMME(IndR_his_NC_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^")
m = axs[0].scatter((np.array(IndR_200hgt_pcc)+np.array(IndR_850hgt_pcc))[27]/2.0, ca.cal_rMME(IndR_his_NC_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*")

#   x-axis title
axs[0].text(-0.90,0.03,s='pcc_mean')
#   y-axis title
axs[0].text(0.03,-0.55,s='corr(IndR, NCR)')

axs[0].hlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")

axs[0].hlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].format(xlim=(-1.0,1.2), ylim=(-0.6,0.6), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="his Corr Coeff. with IndR")
# %%
#   only plot the precipitation regress onto AIR and IndR in gMME
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
# plot_array = np.reshape(range(1, 31), (6, 5))
# plot_array[5,-1] = 0
axs = fig.subplots(ncols=1, nrows=2, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # India area
    x0 = India_W
    y0 = India_S
    width = India_E-India_W
    height = India_N-India_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # SJ-KP area
    x0 = SJ_W
    y0 = SJ_S
    width = SJ_E-SJ_W
    height = SJ_N-SJ_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_AIR_India_pre_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_AIR_India_pre_rvalue, axs[0], n, np.where(pre_AIR_India_pre_pvalue[::n, ::n] <= 0.10), "bright purple", 4.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="AIR and GPCP",
)
# ===================================================
con = axs[1].contourf(
    pre_his_India_pre_rvalue_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    
    )
sepl.plt_sig(
    pre_his_India_pre_rvalue_gens, axs[1], n, np.where(pre_his_India_pre_rvalue_gens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="gMME",
)
# ===================================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   only plot the circulation regress onto AIR and IndR in MME
startlevel=[-15, -8, -6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 13), (3, 4))
# plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ======================================
for ax in axs:
    rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
    ax.add_patch(rect)
    # India area
    x0 = India_W
    y0 = India_S
    width = India_E-India_W
    height = India_N-India_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # SJ-KP area
    x0 = SJ_W
    y0 = SJ_S
    width = SJ_E-SJ_W
    height = SJ_N-SJ_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ======================================
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    # if lev == 200.0:
    #     for ax in axs[num_lev, :]:
    #         x0 = 50
    #         y0 = 15
    #         width = 90
    #         height = 32.5
    #         sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="bright purple", linestyle="-")
    # elif lev == 850.0:
    #     for ax in axs[num_lev, :]:
    #         x0 = 110
    #         y0 = 15
    #         width = 27
    #         height = 22.5
    #         sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="bright purple", linestyle="-")
    con = axs[num_lev, 0].contourf(
        IndRAIR_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndRAIR_ERA5_hgt_slope.sel(level=lev), axs[num_lev, 0], n, np.where(IndRAIR_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_lev, 0].quiver(
        IndRAIR_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
        IndRAIR_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[num_lev, 0].quiver(
        IndRAIR_ERA5_u_slope.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRAIR_ERA5_v_slope.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[num_lev, 0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev, 0].format(
        rtitle="1979-2014 {:.0f}hPa".format(lev), ltitle="AIR & ERA5",
    )
    # ======================================
    con = axs[num_lev, 1].contourf(
        IndR_his_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_gens.sel(level=lev), axs[num_lev, 1], n, np.where(IndR_his_hgt_slope_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[num_lev, 1].quiver(
        IndR_his_u_slope_gens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_slope_gens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[num_lev, 1].quiver(
        IndR_his_u_slope_gens.sel(level=lev).where(IndR_his_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_slope_gens.sel(level=lev).where(IndR_his_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[num_lev, 1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev, 1].format(
        rtitle="1979-2014 {:.0f}hPa".format(lev), ltitle="gMME",
    )
# ======================================
    con = axs[num_lev, 2].contourf(
        IndR_ssp585_p3_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_ssp585_p3_hgt_slope_gens.sel(level=lev), axs[num_lev, 2], n, np.where(IndR_ssp585_p3_hgt_slope_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[num_lev, 2].quiver(
        IndR_ssp585_p3_u_slope_gens.sel(level=lev)[::ski, ::ski],
        IndR_ssp585_p3_v_slope_gens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[num_lev, 2].quiver(
        IndR_ssp585_p3_u_slope_gens.sel(level=lev).where(IndR_ssp585_p3_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_ssp585_p3_v_slope_gens.sel(level=lev).where(IndR_ssp585_p3_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[num_lev, 2].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev, 2].format(
        rtitle="2064-2099 {:.0f}hPa".format(lev), ltitle="gMME",
    )
# ======================================
    con = axs[num_lev, 3].contourf(
        IndR_diff_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_diff_hgt_slope_gens.sel(level=lev), axs[num_lev, 3], n, np.where(IndR_diff_hgt_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[num_lev, 3].quiver(
        IndR_diff_u_slope_gens.sel(level=lev)[::ski, ::ski],
        IndR_diff_v_slope_gens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[num_lev, 3].quiver(
        IndR_diff_u_slope_gens.sel(level=lev).where(IndR_diff_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_diff_v_slope_gens.sel(level=lev).where(IndR_diff_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[num_lev, 3].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev, 3].format(
        rtitle="diff {:.0f}hPa".format(lev), ltitle="gMME",
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="hgt&U reg IndR".format(lev))
# %%
#   plot the bar-plot of the IndR related NC precipitation for historical and ssp585_p3 (reg coeff.)
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_NC_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_NC_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_NC_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_NC_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_NC_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_NC_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-0.7,0.7),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and NC")
# %%
#   plot the correlation scatter-plot, x:corr(IndR, EAhigh), y:corr(IndR, NCR)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
pmodels=['ACCESS-CM2','BCC-CSM2-MR','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','FGOALS-g3','GFDL-CM4','HadGEM3-GC31-LL','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC-ES2L','MIROC6','MPI-ESM1-2-HR','NESM3','NorESM2-LM','TaiESM1']
m = axs[0].scatter(IndR_GPCP_EAhigh_regress[2], IndR_GPCP_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="GPCP", marker="s", color="blue5", ec="black")

for num_models, mod in enumerate(pmodels):
    m = axs[0].scatter(IndR_his_EAhigh_regress[2].sel(models=mod), IndR_his_NC_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, ec="black")
    
for num_models, mod in enumerate(gmodels):
    m = axs[0].scatter(IndR_his_EAhigh_regress[2].sel(models=mod), IndR_his_NC_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, marker="h", ec="black")
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(ca.cal_rMME(IndR_his_EAhigh_regress[2], "models"), ca.cal_rMME(IndR_his_NC_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^", ec="black")
m = axs[0].scatter(ca.cal_rMME(IndR_his_EAhigh_regress[2].sel(models=gmodels), "models"), ca.cal_rMME(IndR_his_NC_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*", ec="black")
xyregress = stats.linregress(IndR_his_EAhigh_regress[2].sel(models=gmodels).data,IndR_his_NC_regress[2].sel(models=gmodels).data)
axs[0].line(np.linspace(-0.70,0), xyregress[0]*np.linspace(-0.70,0)+xyregress[1],zorder=0.8,color="sky blue")
axs[0].text(0.5,0.1,s='{:.3f}'.format(xyregress[0]))
#   x-axis title
axs[0].text(-0.90,0.03,s='corr(IndR, EAhigh)')
#   y-axis title
axs[0].text(0.03,-0.55,s='corr(IndR, NCR)')

axs[0].hlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")

axs[0].hlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].format(xlim=(-1.0,1.0), ylim=(-0.6,0.6), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="his Corr Coeff. with IndR")
# %%
#   plot the correlation scatter-plot, x:corr(IndR, WNPhigh), y:corr(IndR, NCR)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
pmodels=['ACCESS-CM2','BCC-CSM2-MR','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','FGOALS-g3','GFDL-CM4','HadGEM3-GC31-LL','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC-ES2L','MIROC6','MPI-ESM1-2-HR','NESM3','NorESM2-LM','TaiESM1']
m = axs[0].scatter(IndR_GPCP_WNPhigh_regress[2], IndR_GPCP_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="GPCP", marker="s", color="blue5", ec="black")

for num_models, mod in enumerate(pmodels):
    m = axs[0].scatter(IndR_his_WNPhigh_regress[2].sel(models=mod), IndR_his_NC_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, ec="black")
    
for num_models, mod in enumerate(gmodels):
    m = axs[0].scatter(IndR_his_WNPhigh_regress[2].sel(models=mod), IndR_his_NC_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, marker="h", ec="black")
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(ca.cal_rMME(IndR_his_WNPhigh_regress[2], "models"), ca.cal_rMME(IndR_his_NC_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^", ec="black")
m = axs[0].scatter(ca.cal_rMME(IndR_his_WNPhigh_regress[2].sel(models=gmodels), "models"), ca.cal_rMME(IndR_his_NC_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*", ec="black")
# xyregress = stats.linregress(IndR_his_WNPhigh_regress[2].sel(models=gmodels).data,IndR_his_NC_regress[2].sel(models=gmodels).data)
# axs[0].line(np.linspace(-0.70,0), xyregress[0]*np.linspace(-0.70,0)+xyregress[1],zorder=0.8,color="sky blue")
# axs[0].text(0.5,0.1,s='{:.3f}'.format(xyregress[0]))
#   x-axis title
axs[0].text(-0.95,0.03,s='corr(IndR, WNPhigh)')
#   y-axis title
axs[0].text(0.03,-0.55,s='corr(IndR, NCR)')

axs[0].hlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")

axs[0].hlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].format(xlim=(-1.0,1.0), ylim=(-0.6,0.6), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="his Corr Coeff. with IndR")
# %%
#   plot the India rainfall succession
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=7.0, wspace=4.0, hspace=3.5, outerpad=2.0)
# plot_array = np.reshape(range(1, 4), (1, 3))
# plot_array[-1,-3:] = 0
axs = fig.subplots(ncols=1, nrows=3)
# for gMME
scale = pplt.CutoffScale(2016,np.inf,2062)
axs.format(xlim=(1979,2099),xscale=scale,xlocator=np.append(np.arange(1979,2015,5),np.arange(2064, 2100,5)),ylim=(-3.0,3.0))
# axs = pplt.GridSpec(6,5)
# ax = fig.subplot(axs[0])
# bax = brokenaxes(xlims=((1979, 2014), (2064, 2099)), despine=False, subplot_spec=ax)
m1 = axs[0].plot(np.arange(1979,2015,1), ca.standardize(prehis_India_JJA.sel(time=prehis_India_JJA.time.dt.year>=1979, models=gmodels).mean(dim="models")), color="black", lw=1.2)
axs[0].plot(np.arange(2064,2100,1), ca.standardize(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2064, models=gmodels).mean(dim="models")), color="black", lw=1.2)
# m2 = axs[0].plot(np.arange(1979,2015,1), ca.standardize(uhis_EA_JJA.sel(time=uhis_EA_JJA.time.dt.year>=1979).mean(dim="models")), color="blue", lw=1.2)
# axs[0].plot(np.arange(2064,2100,1), ca.standardize(ussp585_EA_JJA.sel(time=ussp585_EA_JJA.time.dt.year>=2064).mean(dim="models")), color="blue", lw=1.2)
axs[0].plot(np.arange(1979,2015,1),np.zeros(36),lw=1.2,color="grey7")
axs[0].plot(np.arange(2064,2100,1),np.zeros(36),lw=1.2,color="grey7")
axs[0].fill_between([2014,2064],-3,3,color="grey7", alpha=0.1)
axs[0].format(ltitle="gMME", rtitle="Total India", xticklabelsize=11)
# ====================================
m1 = axs[1].plot(np.arange(1979,2015,1), ca.standardize(prehis_nIndia_JJA.sel(time=prehis_nIndia_JJA.time.dt.year>=1979, models=gmodels).mean(dim="models")), color="black", lw=1.2)
axs[1].plot(np.arange(2064,2100,1), ca.standardize(pressp585_nIndia_JJA.sel(time=pressp585_nIndia_JJA.time.dt.year>=2064, models=gmodels).mean(dim="models")), color="black", lw=1.2)
ox = axs[1].alty(color="blue6", label="rate", linewidth=1)
ox.plot(np.arange(1979,2015,1), (prehis_nIndia_JJA_sum/prehis_India_JJA_sum).sel(models=gmodels).mean(dim="models"), color="blue6", lw=1.2)
ox.plot(np.arange(2064,2100,1), (pressp585_p3_nIndia_JJA_sum/pressp585_p3_India_JJA_sum).sel(models=gmodels).mean(dim="models"), color="blue6", lw=1.2)
ox.format(ylim=(0.2,0.7))
# m2 = axs[1].plot(np.arange(1979,2015,1), ca.standardize(uhis_EA_JJA.sel(time=uhis_EA_JJA.time.dt.year>=1979).mean(dim="models")), color="blue", lw=1.2)
# axs[1].plot(np.arange(2064,2100,1), ca.standardize(ussp585_EA_JJA.sel(time=ussp585_EA_JJA.time.dt.year>=2064).mean(dim="models")), color="blue", lw=1.2)
axs[1].plot(np.arange(1979,2015,1),np.zeros(36),lw=1.2,color="grey7")
axs[1].plot(np.arange(2064,2100,1),np.zeros(36),lw=1.2,color="grey7")
axs[1].fill_between([2014,2064],-3,3,color="grey7", alpha=0.1)
axs[1].format(ltitle="gMME", rtitle="North India", xticklabelsize=11)
# =================================
m1 = axs[2].plot(np.arange(1979,2015,1), ca.standardize(prehis_wIndia_JJA.sel(time=prehis_wIndia_JJA.time.dt.year>=1979, models=gmodels).mean(dim="models")), color="black", lw=1.2)
axs[2].plot(np.arange(2064,2100,1), ca.standardize(pressp585_wIndia_JJA.sel(time=pressp585_wIndia_JJA.time.dt.year>=2064, models=gmodels).mean(dim="models")), color="black", lw=1.2)
ox = axs[2].alty(color="blue6", label="rate", linewidth=1)
ox.plot(np.arange(1979,2015,1), (prehis_wIndia_JJA_sum/prehis_India_JJA_sum).sel(models=gmodels).mean(dim="models"), color="blue6", lw=1.2)
ox.plot(np.arange(2064,2100,1), (pressp585_p3_wIndia_JJA_sum/pressp585_p3_India_JJA_sum).sel(models=gmodels).mean(dim="models"), color="blue6", lw=1.2)
ox.format(ylim=(0.0,0.6))
# m2 = axs[2].plot(np.arange(1979,2015,1), ca.standardize(uhis_EA_JJA.sel(time=uhis_EA_JJA.time.dt.year>=1979).mean(dim="models")), color="blue", lw=1.2)
# axs[2].plot(np.arange(2064,2100,1), ca.standardize(ussp585_EA_JJA.sel(time=ussp585_EA_JJA.time.dt.year>=2064).mean(dim="models")), color="blue", lw=1.2)
axs[2].plot(np.arange(1979,2015,1),np.zeros(36),lw=1.0,color="grey7")
axs[2].plot(np.arange(2064,2100,1),np.zeros(36),lw=1.0,color="grey7")
axs[2].fill_between([2014,2064],-3,3,color="grey7", alpha=0.1)
axs[2].format(ltitle="gMME", rtitle="West India", xticklabelsize=11)

fig.format(abc="(a)", abcloc="l", suptitle="IndR")
# %%
#   plot the 200hPa circulation mme result in 1979-2014, 2064-2099, diff
startlevel=[-15, -8, -6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]
for num_lev,lev in enumerate([200.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    # plot_array = np.reshape(range(1, 31), (6, 5))
    # plot_array[-1,-1] = 0
    axs = fig.subplots(ncols=3, nrows=1, proj=proj)

    #   set the geo_ticks and map projection to the plots
    # xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 1
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
        # India area
        x0 = India_W
        y0 = India_S
        width = India_E-India_W
        height = India_N-India_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # NC area
        x0 = NC_W
        y0 = NC_S
        width = NC_E-NC_W
        height = NC_N-NC_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
        # SJ-KP area
        x0 = SJ_W
        y0 = SJ_S
        width = SJ_E-SJ_W
        height = SJ_N-SJ_S
        sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndR_his_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_ens.sel(level=lev), axs[0], n, np.where(IndR_his_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndR_his_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_slope_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        IndR_his_u_slope_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_slope_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="MME",
    )
    # ======================================
    con = axs[1].contourf(
        IndR_ssp585_p3_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_ssp585_p3_hgt_slope_ens.sel(level=lev), axs[1], n, np.where(IndR_ssp585_p3_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[1].quiver(
        IndR_ssp585_p3_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IndR_ssp585_p3_v_slope_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[1].quiver(
        IndR_ssp585_p3_u_slope_ens.sel(level=lev).where(IndR_ssp585_p3_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_ssp585_p3_v_slope_ens.sel(level=lev).where(IndR_ssp585_p3_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="2064-2099", ltitle="MME",
    )
    # ======================================
    con = axs[2].contourf(
        IndR_diff_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_diff_hgt_slope_ens.sel(level=lev), axs[2], n, np.where(IndR_diff_hgt_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[2].quiver(
        IndR_diff_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IndR_diff_v_slope_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[2].quiver(
        IndR_diff_u_slope_ens.sel(level=lev).where(IndR_diff_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_diff_v_slope_ens.sel(level=lev).where(IndR_diff_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[2].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[2].format(
        rtitle="diff", ltitle="MME",
    )
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the bar-plot of the EA high
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_EAhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_EAhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_EAhigh_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_EAhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_EAhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_EAhigh_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-2.0e-6,2.0e-6),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and EAhigh")
# %%
#   plot the bar-plot of the WA high
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_WAhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_WAhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_WAhigh_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_WAhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_WAhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_WAhigh_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-3.0e-6,3.0e-6),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and WAhigh")
# %%
#   plot the bar-plot of the WNP high
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_WNPhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_WNPhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_WNPhigh_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_WNPhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_WNPhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_WNPhigh_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-1.0e-6,1.0e-6),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and WNPhigh")
# %%
#   plot the correlation scatter-plot, x:corr(IndR, WAhigh), y:corr(IndR, EAhigh)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
pmodels=['ACCESS-CM2','BCC-CSM2-MR','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','FGOALS-g3','GFDL-CM4','HadGEM3-GC31-LL','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC-ES2L','MIROC6','MPI-ESM1-2-HR','NESM3','NorESM2-LM','TaiESM1']
m = axs[0].scatter(IndR_GPCP_WAhigh_regress[2], IndR_GPCP_EAhigh_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="obs", marker="s", color="blue5", ec="black")

for num_models, mod in enumerate(pmodels):
    m = axs[0].scatter(IndR_his_WAhigh_regress[2].sel(models=mod), IndR_his_EAhigh_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, ec="black")
    
for num_models, mod in enumerate(gmodels):
    m = axs[0].scatter(IndR_his_WAhigh_regress[2].sel(models=mod), IndR_his_EAhigh_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, marker="h", ec="black")
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(ca.cal_rMME(IndR_his_WAhigh_regress[2], "models"), ca.cal_rMME(IndR_his_EAhigh_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^", ec="black")
m = axs[0].scatter(ca.cal_rMME(IndR_his_WAhigh_regress[2].sel(models=gmodels), "models"), ca.cal_rMME(IndR_his_EAhigh_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*", ec="black")
# xyregress = stats.linregress(IndR_his_WAhigh_regress[2].sel(models=gmodels).data,IndR_his_EAhigh_regress[2].sel(models=gmodels).data)
# axs[0].line(np.linspace(-0.70,0), xyregress[0]*np.linspace(-0.70,0)+xyregress[1],zorder=0.8,color="sky blue")
# axs[0].text(0.5,0.1,s='{:.3f}'.format(xyregress[0]))
#   x-axis title
axs[0].text(-0.95,0.03,s='corr(IndR, WAhigh)')
#   y-axis title
axs[0].text(0.03,-0.55,s='corr(IndR, EAhigh)')

axs[0].hlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")

axs[0].hlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].format(xlim=(-1.0,1.0), ylim=(-1.0,1.0), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="his Corr Coeff. with IndR")
# %%
#   plot the correlation scatter-plot, x:corr(IndR, WAhigh), y:corr(IndR, EAhigh)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
pmodels=['ACCESS-CM2','BCC-CSM2-MR','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','FGOALS-g3','GFDL-CM4','HadGEM3-GC31-LL','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC-ES2L','MIROC6','MPI-ESM1-2-HR','NESM3','NorESM2-LM','TaiESM1']
# m = axs[0].scatter(IndR_GPCP_WAhigh_regress[2], IndR_GPCP_EAhigh_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="obs", marker="s", color="blue5", ec="black")

for num_models, mod in enumerate(pmodels):
    m = axs[0].scatter(IndR_ssp585_p3_WAhigh_regress[2].sel(models=mod), IndR_ssp585_p3_EAhigh_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, ec="black")
    
for num_models, mod in enumerate(gmodels):
    m = axs[0].scatter(IndR_ssp585_p3_WAhigh_regress[2].sel(models=mod), IndR_ssp585_p3_EAhigh_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, marker="h", ec="black")
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(ca.cal_rMME(IndR_ssp585_p3_WAhigh_regress[2], "models"), ca.cal_rMME(IndR_ssp585_p3_EAhigh_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^", ec="black")
m = axs[0].scatter(ca.cal_rMME(IndR_ssp585_p3_WAhigh_regress[2].sel(models=gmodels), "models"), ca.cal_rMME(IndR_ssp585_p3_EAhigh_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*", ec="black")
# xyregress = stats.linregress(IndR_ssp585_p3_WAhigh_regress[2].sel(models=gmodels).data,IndR_ssp585_p3_EAhigh_regress[2].sel(models=gmodels).data)
# axs[0].line(np.linspace(-0.70,0), xyregress[0]*np.linspace(-0.70,0)+xyregress[1],zorder=0.8,color="sky blue")
# axs[0].text(0.5,0.1,s='{:.3f}'.format(xyregress[0]))
#   x-axis title
axs[0].text(-0.95,0.03,s='corr(IndR, WAhigh)')
#   y-axis title
axs[0].text(0.03,-0.55,s='corr(IndR, EAhigh)')

axs[0].hlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")

axs[0].hlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].format(xlim=(-1.0,1.0), ylim=(-1.0,1.0), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="ssp585_p3 Corr Coeff. with IndR")
# %%
#   only plot the circulation regress onto AIR and IndR in MME
startlevel=[-15, -8, -6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 4), (3, 1))
# plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ======================================
for ax in axs:
    rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
    ax.add_patch(rect)
    # India area
    x0 = India_W
    y0 = India_S
    width = India_E-India_W
    height = India_N-India_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # SJ-KP area
    x0 = SJ_W
    y0 = SJ_S
    width = SJ_E-SJ_W
    height = SJ_N-SJ_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ======================================
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    if lev == 200.0:
        for ax in axs[num_lev, :]:
            x0 = EAhigh_W
            y0 = EAhigh_S
            width = EAhigh_E-EAhigh_W
            height = EAhigh_N-EAhigh_S
            sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="bright purple", linestyle="-")
            x0 = WAhigh_W
            y0 = WAhigh_S
            width = WAhigh_E-WAhigh_W
            height = WAhigh_N-WAhigh_S
            sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="bright purple", linestyle="-")
    elif lev == 850.0:
        for ax in axs[num_lev, :]:
            x0 = WNPhigh_W
            y0 = WNPhigh_S
            width = WNPhigh_E-WNPhigh_W
            height = WNPhigh_N-WNPhigh_S
            sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="bright purple", linestyle="-")
    # ======================================
    con = axs[num_lev, 0].contourf(
        IndR_his_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_gens.sel(level=lev), axs[num_lev, 0], n, np.where(IndR_his_hgt_slope_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[num_lev, 0].quiver(
        IndR_his_u_slope_gens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_slope_gens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[num_lev, 0].quiver(
        IndR_his_u_slope_gens.sel(level=lev).where(IndR_his_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_slope_gens.sel(level=lev).where(IndR_his_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[num_lev, 0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev, 0].format(
        rtitle="1979-2014 {:.0f}hPa".format(lev), ltitle="gMME",
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="hgt&U reg IndR".format(lev))
# %%
#   plot the correlation scatter-plot, x:corr(IndR, WAhigh), y:corr(IndR, EAhigh)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
pmodels=['ACCESS-CM2','BCC-CSM2-MR','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','FGOALS-g3','GFDL-CM4','HadGEM3-GC31-LL','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC-ES2L','MIROC6','MPI-ESM1-2-HR','NESM3','NorESM2-LM','TaiESM1']
# m = axs[0].scatter(IndR_GPCP_WAhigh_regress[2], IndR_GPCP_EAhigh_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="obs", marker="s", color="blue5", ec="black")

for num_models, mod in enumerate(pmodels):
    m = axs[0].scatter(IndR_his_WNPhigh_regress[2].sel(models=mod), IndR_his_EAhigh_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, ec="black")
    
for num_models, mod in enumerate(gmodels):
    m = axs[0].scatter(IndR_his_WNPhigh_regress[2].sel(models=mod), IndR_his_EAhigh_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, marker="h", ec="black")
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(ca.cal_rMME(IndR_his_WNPhigh_regress[2], "models"), ca.cal_rMME(IndR_his_EAhigh_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^", ec="black")
m = axs[0].scatter(ca.cal_rMME(IndR_his_WNPhigh_regress[2].sel(models=gmodels), "models"), ca.cal_rMME(IndR_his_EAhigh_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*", ec="black")
# xyregress = stats.linregress(IndR_his_WNPhigh_regress[2].sel(models=gmodels).data,IndR_his_EAhigh_regress[2].sel(models=gmodels).data)
# axs[0].line(np.linspace(-0.70,0), xyregress[0]*np.linspace(-0.70,0)+xyregress[1],zorder=0.8,color="sky blue")
# axs[0].text(0.5,0.1,s='{:.3f}'.format(xyregress[0]))
#   x-axis title
axs[0].text(-0.95,0.03,s='corr(IndR, WNPhigh)')
#   y-axis title
axs[0].text(0.03,-0.55,s='corr(IndR, EAhigh)')

axs[0].hlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")

axs[0].hlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].format(xlim=(-1.0,1.0), ylim=(-1.0,1.0), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="his Corr Coeff. with IndR")
# %%
#   plot the correlation scatter-plot, x:corr(IndR, WAhigh), y:corr(IndR, EAhigh)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
pmodels=['ACCESS-CM2','BCC-CSM2-MR','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','FGOALS-g3','GFDL-CM4','HadGEM3-GC31-LL','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC-ES2L','MIROC6','MPI-ESM1-2-HR','NESM3','NorESM2-LM','TaiESM1']
# m = axs[0].scatter(IndR_GPCP_WAhigh_regress[2], IndR_GPCP_EAhigh_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="obs", marker="s", color="blue5", ec="black")

for num_models, mod in enumerate(pmodels):
    m = axs[0].scatter(IndR_ssp585_p3_WNPhigh_regress[2].sel(models=mod), IndR_ssp585_p3_EAhigh_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, ec="black")
    
for num_models, mod in enumerate(gmodels):
    m = axs[0].scatter(IndR_ssp585_p3_WNPhigh_regress[2].sel(models=mod), IndR_ssp585_p3_EAhigh_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, marker="h", ec="black")
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(ca.cal_rMME(IndR_ssp585_p3_WNPhigh_regress[2], "models"), ca.cal_rMME(IndR_ssp585_p3_EAhigh_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^", ec="black")
m = axs[0].scatter(ca.cal_rMME(IndR_ssp585_p3_WNPhigh_regress[2].sel(models=gmodels), "models"), ca.cal_rMME(IndR_ssp585_p3_EAhigh_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*", ec="black")
# xyregress = stats.linregress(IndR_ssp585_p3_WNPhigh_regress[2].sel(models=gmodels).data,IndR_ssp585_p3_EAhigh_regress[2].sel(models=gmodels).data)
# axs[0].line(np.linspace(-0.70,0), xyregress[0]*np.linspace(-0.70,0)+xyregress[1],zorder=0.8,color="sky blue")
# axs[0].text(0.5,0.1,s='{:.3f}'.format(xyregress[0]))
#   x-axis title
axs[0].text(-0.95,0.03,s='corr(IndR, WNPhigh)')
#   y-axis title
axs[0].text(0.03,-0.55,s='corr(IndR, EAhigh)')

axs[0].hlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")

axs[0].hlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].format(xlim=(-1.0,1.0), ylim=(-1.0,1.0), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="ssp585_p3 Corr Coeff. with IndR")
# %%
#   plot the bar-plot of the IndR related EAM index for historical and ssp585_p3 (reg coeff.)
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_EAM_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_EAM_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_EAM_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_EAM_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_EAM_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_EAM_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-3.0,3.0),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and EAM")
# %%
#   plot the bar-plot of the IndR related IWF index for historical and ssp585_p3 (reg coeff.)
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_IWF_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_IWF_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_IWF_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_IWF_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_IWF_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_IWF_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-1.0e-6,1.0e-6),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and IWF")
# %%
# plot the bar plot about the corr(IndR, LKY)
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_LKY_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_LKY_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_LKY_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_LKY_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_LKY_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_LKY_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-1e-6,1e-6),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and LKY")

# plot the bar plot about the corr(IndR, LKY)
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_LKY_regress[2].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_LKY_regress[2].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_LKY_rvalue.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_LKY_regress[2].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_LKY_regress[2].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_LKY_rvalue.sel(models=gmodels).mean(dim="models", skipna=True).data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-1,1),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Corr. Coeff. IndR and LKY")
# %%
#   plot the all-models bar-plots for corr(IndR, LKY)
plot_data = np.zeros((27,3))
plot_data[:-1,0] = IndR_his_LKY_regress[0].data
plot_data[:-1,1] = IndR_ssp585_p3_LKY_regress[0].data
plot_data[:-1,2] = IndR_diff_LKY_slope.data
plot_data[-1,0] = IndR_his_LKY_regress[0].mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_LKY_regress[0].mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_LKY_slope.mean(dim="models", skipna=True).data

label_models = list(models.data)
label_models.append("MME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-1e-6,1e-6),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and LKY")

# plot the bar plot about the corr(IndR, LKY)
plot_data = np.zeros((27,3))
plot_data[:-1,0] = IndR_his_LKY_regress[2].data
plot_data[:-1,1] = IndR_ssp585_p3_LKY_regress[2].data
plot_data[:-1,2] = IndR_diff_LKY_rvalue.data
plot_data[-1,0] = IndR_his_LKY_regress[2].mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_LKY_regress[2].mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_LKY_rvalue.mean(dim="models", skipna=True).data

label_models = list(models.data)
label_models.append("MME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-1,1),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Corr. Coeff. IndR and LKY")
# %%
#   plot the SST regress onto the IndR in historical
#   corr. coeffs.
startlevel=-1.0
spacinglevel=0.1

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (5, 6))
plot_array[-1,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    IndR_Had_sst_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
)
sepl.plt_sig(
    IndR_Had_sst_rvalue, axs[0], n, np.where(IndR_Had_sst_pvalue[::n, ::n] <= 0.05), "bright purple", 3.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="HadISST & AIR",
)
# ======================================
con = axs[1].contourf(
    IndR_his_sst_rvalue_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
)
sepl.plt_sig(
    IndR_his_sst_rvalue_ens, axs[1], n, np.where(IndR_his_sst_rvalue_ens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[1].format(
    rtitle="1979-2014", ltitle="MME",
)
# ======================================
for num_mod, mod in enumerate(models_array):
    con = axs[num_mod+2].contourf(
        IndR_his_sst_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_his_sst_rvalue.sel(models=mod), axs[num_mod+2], n, np.where(IndR_his_sst_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod+2].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")

# ======================================
#   reg. coeffs.
startlevel=-6e-1
spacinglevel=0.06

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (5, 6))
plot_array[-1,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    IndR_Had_sst_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_Had_sst_slope, axs[0], n, np.where(IndR_Had_sst_pvalue[::n, ::n] <= 0.05), "bright purple", 3.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="HadISST & AIR",
)
# ======================================
con = axs[1].contourf(
    IndR_his_sst_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_his_sst_slope_ens, axs[1], n, np.where(IndR_his_sst_slope_ens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[1].format(
    rtitle="1979-2014", ltitle="MME",
)
# ======================================
for num_mod, mod in enumerate(models_array):
    con = axs[num_mod+2].contourf(
        IndR_his_sst_slope.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_sst_slope.sel(models=mod), axs[num_mod+2], n, np.where(IndR_his_sst_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod+2].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")
# %%
#   plot the SST regress onto the IndR in ssp585_p3
#   corr. coeffs.
startlevel=-1.0
spacinglevel=0.1

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (5, 6))
plot_array[-1,-3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14

# ======================================
con = axs[0].contourf(
    IndR_ssp585_p3_sst_rvalue_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
)
sepl.plt_sig(
    IndR_ssp585_p3_sst_rvalue_ens, axs[0], n, np.where(IndR_ssp585_p3_sst_rvalue_ens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[0].format(
    rtitle="2064-2099", ltitle="MME",
)
# ======================================
for num_mod, mod in enumerate(models_array):
    con = axs[num_mod+1].contourf(
        IndR_ssp585_p3_sst_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_ssp585_p3_sst_rvalue.sel(models=mod), axs[num_mod+1], n, np.where(IndR_ssp585_p3_sst_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")

# ======================================
#   reg. coeffs.
startlevel=-6.0e-1
spacinglevel=0.06

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (5, 6))
plot_array[-1,-3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    IndR_ssp585_p3_sst_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_ssp585_p3_sst_slope_ens, axs[0], n, np.where(IndR_ssp585_p3_sst_slope_ens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[0].format(
    rtitle="2064-2099", ltitle="MME",
)
# ======================================
for num_mod, mod in enumerate(models_array):
    con = axs[num_mod+1].contourf(
        IndR_ssp585_p3_sst_slope.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_ssp585_p3_sst_slope.sel(models=mod), axs[num_mod+1], n, np.where(IndR_ssp585_p3_sst_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")
# %%
#   plot the SST regress onto the IndR in diff
#   corr. coeffs.
startlevel=-1.0
spacinglevel=0.1

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (5, 6))
plot_array[-1,-3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14

# ======================================
con = axs[0].contourf(
    IndR_diff_sst_rvalue_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
)
sepl.plt_sig(
    IndR_diff_sst_rvalue_ens, axs[0], n, np.where(IndR_diff_sst_ens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[0].format(
    rtitle="diff", ltitle="MME",
)
# ======================================
for num_mod, mod in enumerate(models_array):
    con = axs[num_mod+1].contourf(
        IndR_diff_sst_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
    )
    axs[num_mod+1].format(
        rtitle="diff", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")

# ======================================
#   reg. coeffs.
startlevel=-6.0e-1
spacinglevel=0.06

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (5, 6))
plot_array[-1,-3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    IndR_diff_sst_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_diff_sst_slope_ens, axs[0], n, np.where(IndR_diff_sst_ens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[0].format(
    rtitle="diff", ltitle="MME",
)
# ======================================
for num_mod, mod in enumerate(models_array):
    con = axs[num_mod+1].contourf(
        IndR_diff_sst_slope.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+1].format(
        rtitle="diff", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")
# %%
#   plot the good models SST regression result
#   plot the SST regress onto the IndR in historical
#   corr. coeffs.
startlevel=-1.0
spacinglevel=0.1

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 9), (2, 4))
# plot_array[-1,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    IndR_Had_sst_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
)
sepl.plt_sig(
    IndR_Had_sst_rvalue, axs[0], n, np.where(IndR_Had_sst_pvalue[::n, ::n] <= 0.05), "bright purple", 3.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="HadISST & AIR",
)
# ======================================
con = axs[1].contourf(
    IndR_his_sst_rvalue_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
)
sepl.plt_sig(
    IndR_his_sst_rvalue_gens, axs[1], n, np.where(IndR_his_sst_rvalue_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[1].format(
    rtitle="1979-2014", ltitle="gMME",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+2].contourf(
        IndR_his_sst_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_his_sst_rvalue.sel(models=mod), axs[num_mod+2], n, np.where(IndR_his_sst_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod+2].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")

# ======================================
#   reg. coeffs.
startlevel=-6e-1
spacinglevel=0.06

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 9), (2, 4))
# plot_array[-1,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    IndR_Had_sst_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_Had_sst_slope, axs[0], n, np.where(IndR_Had_sst_pvalue[::n, ::n] <= 0.05), "bright purple", 3.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="HadISST & AIR",
)
# ======================================
con = axs[1].contourf(
    IndR_his_sst_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_his_sst_slope_gens, axs[1], n, np.where(IndR_his_sst_slope_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[1].format(
    rtitle="1979-2014", ltitle="gMME",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+2].contourf(
        IndR_his_sst_slope.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_sst_slope.sel(models=mod), axs[num_mod+2], n, np.where(IndR_his_sst_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod+2].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")
# %%
#   plot the SST regress onto the IndR in ssp585_p3 in good models
#   corr. coeffs.
startlevel=-1.0
spacinglevel=0.1

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 9), (2, 4))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14

# ======================================
con = axs[0].contourf(
    IndR_ssp585_p3_sst_rvalue_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
)
sepl.plt_sig(
    IndR_ssp585_p3_sst_rvalue_gens, axs[0], n, np.where(IndR_ssp585_p3_sst_rvalue_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[0].format(
    rtitle="2064-2099", ltitle="gMME",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+1].contourf(
        IndR_ssp585_p3_sst_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_ssp585_p3_sst_rvalue.sel(models=mod), axs[num_mod+1], n, np.where(IndR_ssp585_p3_sst_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")

# ======================================
#   reg. coeffs.
startlevel=-6.0e-1
spacinglevel=0.06

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 9), (2, 4))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    IndR_ssp585_p3_sst_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_ssp585_p3_sst_slope_gens, axs[0], n, np.where(IndR_ssp585_p3_sst_slope_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[0].format(
    rtitle="2064-2099", ltitle="gMME",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+1].contourf(
        IndR_ssp585_p3_sst_slope.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_ssp585_p3_sst_slope.sel(models=mod), axs[num_mod+1], n, np.where(IndR_ssp585_p3_sst_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")
# %%
#   plot the SST regress onto the IndR in diff in good models
#   corr. coeffs.
startlevel=-1.0
spacinglevel=0.1

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 9), (2, 4))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14

# ======================================
con = axs[0].contourf(
    IndR_diff_sst_rvalue_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
)
sepl.plt_sig(
    IndR_diff_sst_rvalue_gens, axs[0], n, np.where(IndR_diff_sst_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[0].format(
    rtitle="diff", ltitle="gMME",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+1].contourf(
        IndR_diff_sst_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
    )
    axs[num_mod+1].format(
        rtitle="diff", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")

# ======================================
#   reg. coeffs.
startlevel=-6.0e-1
spacinglevel=0.06

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 9), (2, 4))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    IndR_diff_sst_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_diff_sst_slope_gens, axs[0], n, np.where(IndR_diff_sst_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[0].format(
    rtitle="diff", ltitle="gMME",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+1].contourf(
        IndR_diff_sst_slope.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+1].format(
        rtitle="diff", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")
# %%
#   plot the climatology-u wind in 200hPa in historical
startlevel=-30
spacinglevel=3
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 9), (2, 4))
# plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-15, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    uERA5_ver_JJA.sel(level=200.0).mean(dim=["time"]),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
axs[0].line(ERA5_wj_axis[1], ERA5_wj_axis[0], lw=1.3, color="black")
axs[0].format(
    rtitle="1979-2014", ltitle="ERA5",
)
# ======================================
con = axs[1].contourf(
    uhis_ver_JJA.sel(models=gmodels, level=200.0).mean(dim=["time", "models"]),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
axs[1].line(his_wj_axis.coords["lon"], his_wj_axis_lat_gens, lw=1.3, color="black")
axs[1].format(
    rtitle="1979-2014", ltitle="gMME",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+2].contourf(
        uhis_ver_JJA.sel(models=mod, level=200.0).mean(dim="time"),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+2].line(his_wj_axis.coords["lon"], his_wj_axis.sel(models=mod), lw=1.3, color="black")
    axs[num_mod+2].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="u-climatology")
# %%
#   plot the climatology-u wind in 200hPa in ssp585_p3
startlevel=-30
spacinglevel=3
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 9), (2, 4))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-15, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    ussp585_p3_ver_JJA.sel(models=gmodels, level=200.0).mean(dim=["time", "models"]),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
axs[0].line(ssp585_p3_wj_axis.coords["lon"], ssp585_p3_wj_axis_lat_gens, lw=1.3, color="black")
axs[0].format(
    rtitle="2064-2099", ltitle="gMME",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+1].contourf(
        ussp585_p3_ver_JJA.sel(models=mod, level=200.0).mean(dim="time"),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+1].line(ssp585_p3_wj_axis.coords["lon"], ssp585_p3_wj_axis.sel(models=mod), lw=1.3, color="black")
    axs[num_mod+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="u-climatology")
# %%
#   plot the climatology-u wind in 200hPa in diff
startlevel=-5
spacinglevel=0.5
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 9), (2, 4))
plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-15, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    ussp585_p3_ver_JJA.sel(models=gmodels, level=200.0).mean(dim=["time", "models"])-uhis_ver_JJA.sel(models=gmodels, level=200.0).mean(dim=["time", "models"]),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
axs[0].line(his_wj_axis.coords["lon"], his_wj_axis_lat_gens, lw=1.3, color="black")
axs[0].line(ssp585_p3_wj_axis.coords["lon"], ssp585_p3_wj_axis_lat_gens, lw=1.3, color="red6")
axs[0].format(
    rtitle="diff", ltitle="gMME",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+1].contourf(
        ussp585_p3_ver_JJA.sel(models=mod, level=200.0).mean(dim="time")-uhis_ver_JJA.sel(models=mod, level=200.0).mean(dim="time"),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+1].line(his_wj_axis.coords["lon"], his_wj_axis.sel(models=mod), lw=1.3, color="black")
    axs[num_mod+1].line(ssp585_p3_wj_axis.coords["lon"], ssp585_p3_wj_axis.sel(models=mod), lw=1.3, color="red6")
    axs[num_mod+1].format(
        rtitle="diff", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="u-climatology")
# %%
#   plot the RWS, term1 and term2 in Rossby Wave Source
startlevel=-3e-11
spacinglevel=5e-12
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 49), (8, 6))
plot_array[0,3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-15, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)
# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ======================================
con = axs[0,0].contourf(
    SERA5,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
    zorder=0.8,
    extend="both"
)

axs[0,0].format(
    rtitle="1979-2014", ltitle="ERA5 RWS",
)
# ======================================
con = axs[1,0].contourf(
    Shis_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
    zorder=0.8,
    extend="both"
)

sepl.plt_sig(
    Shis_gens, axs[1,0], n, np.where(Shis_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)

axs[1,0].format(
    rtitle="1979-2014", ltitle="gMME RWS",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+2,0].contourf(
        Shis.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+2,0].format(
        rtitle="1979-2014", ltitle="{} RWS".format(mod),
    )
# ======================================
con = axs[1,3].contourf(
    Sssp585_p3_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
    zorder=0.8,
    extend="both"
)

sepl.plt_sig(
    Sssp585_p3_gens, axs[1,3], n, np.where(Sssp585_p3_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)

axs[1,3].format(
    rtitle="2064-2099", ltitle="gMME RWS",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+2,3].contourf(
        Sssp585_p3.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+2,3].format(
        rtitle="2064-2099", ltitle="{} RWS".format(mod),
    )
# ======================================
con = axs[0,1].contourf(
    SERA51,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
    zorder=0.8,
    extend="both"
)

axs[0,1].format(
    rtitle="1979-2014", ltitle="ERA5 RWS1",
)
# ======================================
con = axs[1,1].contourf(
    Shis1_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
    zorder=0.8,
    extend="both"
)

sepl.plt_sig(
    Shis1_gens, axs[1,1], n, np.where(Shis1_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)

axs[1,1].format(
    rtitle="1979-2014", ltitle="gMME RWS1",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+2,1].contourf(
        Shis1.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+2,1].format(
        rtitle="1979-2014", ltitle="{} RWS1".format(mod),
    )
# ======================================
con = axs[1,4].contourf(
    Sssp585_p31_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
    zorder=0.8,
    extend="both"
)

sepl.plt_sig(
    Sssp585_p31_gens, axs[1,4], n, np.where(Sssp585_p31_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)

axs[1,4].format(
    rtitle="2064-2099", ltitle="gMME RWS1",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+2,4].contourf(
        Sssp585_p31.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+2,4].format(
        rtitle="2064-2099", ltitle="{} RWS1".format(mod),
    )
# ======================================
con = axs[0,2].contourf(
    SERA52,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
    zorder=0.8,
    extend="both"
)

axs[0,2].format(
    rtitle="1979-2014", ltitle="ERA5 RWS2",
)
# ======================================
con = axs[1,2].contourf(
    Shis2_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
    zorder=0.8,
    extend="both"
)

sepl.plt_sig(
    Shis2_gens, axs[1,2], n, np.where(Shis2_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)

axs[1,2].format(
    rtitle="1979-2014", ltitle="gMME RWS2",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+2,2].contourf(
        Shis2.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+2,2].format(
        rtitle="1979-2014", ltitle="{} RWS2".format(mod),
    )
# ======================================
con = axs[1,5].contourf(
    Sssp585_p32_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
    zorder=0.8,
    extend="both"
)

sepl.plt_sig(
    Sssp585_p32_gens, axs[1,5], n, np.where(Sssp585_p32_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)

axs[1,5].format(
    rtitle="2064-2099", ltitle="gMME RWS2",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+2,5].contourf(
        Sssp585_p32.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+2,5].format(
        rtitle="2064-2099", ltitle="{} RWS2".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
#   plot the difference in the RWS and term1 and term2
startlevel=-3e-11
spacinglevel=5e-12
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 22), (7, 3))
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-15, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)
# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ======================================
con = axs[0,0].contourf(
    Sdiff_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
    zorder=0.8,
    extend="both"
)

sepl.plt_sig(
    Sdiff_gens, axs[0,0], n, np.where(Sdiff_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)

axs[0,0].format(
    rtitle="diff", ltitle="gMME RWS",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+1,0].contourf(
        Sdiff.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+1,0].format(
        rtitle="diff", ltitle="{} RWS".format(mod),
    )
# ======================================
con = axs[0,1].contourf(
    Sdiff1_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
    zorder=0.8,
    extend="both"
)

sepl.plt_sig(
    Sdiff1_gens, axs[0,1], n, np.where(Sdiff1_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)

axs[0,1].format(
    rtitle="diff", ltitle="gMME RWS1",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+1,1].contourf(
        Sdiff1.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+1,1].format(
        rtitle="diff", ltitle="{} RWS1".format(mod),
    )
# ======================================
con = axs[0,2].contourf(
    Sdiff2_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
    zorder=0.8,
    extend="both"
)

sepl.plt_sig(
    Sdiff_gens, axs[0,2], n, np.where(Sdiff2_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)

axs[0,2].format(
    rtitle="diff", ltitle="gMME RWS2",
)
# ======================================
for num_mod, mod in enumerate(gmodels):
    con = axs[num_mod+1,2].contourf(
        Sdiff2.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+1,2].format(
        rtitle="diff", ltitle="{} RWS2".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
#   plot the Bar-plot of the West-Asia RWS
plot_data = np.zeros((7,3))
plot_data[:-1,0] = Shis_WARWS.sel(models=gmodels).data
plot_data[:-1,1] = Sssp585_p3_WARWS.sel(models=gmodels).data
plot_data[:-1,2] = Sdiff_WARWS.sel(models=gmodels).data
plot_data[-1,0] = Shis_WARWS_gens.data
plot_data[-1,1] = Sssp585_p3_WARWS_gens.data
plot_data[-1,2] = Sdiff_WARWS_gens.data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-3e-11,3e-11),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=11, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="West Asia RWS")
# %%
#   plot the Bar-plot of the East-Asia RWS
plot_data = np.zeros((7,3))
plot_data[:-1,0] = Shis_EARWS.sel(models=gmodels).data
plot_data[:-1,1] = Sssp585_p3_EARWS.sel(models=gmodels).data
plot_data[:-1,2] = Sdiff_EARWS.sel(models=gmodels).data
plot_data[-1,0] = Shis_EARWS_gens.data
plot_data[-1,1] = Sssp585_p3_EARWS_gens.data
plot_data[-1,2] = Sdiff_EARWS_gens.data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-3e-11,3e-11),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=11, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="East Asia RWS")
# %%
#   plot the Bar-plot of the West-Asia RWS1
plot_data = np.zeros((7,3))
plot_data[:-1,0] = Shis1_WARWS.sel(models=gmodels).data
plot_data[:-1,1] = Sssp585_p31_WARWS.sel(models=gmodels).data
plot_data[:-1,2] = Sdiff1_WARWS.sel(models=gmodels).data
plot_data[-1,0] = Shis1_WARWS_gens.data
plot_data[-1,1] = Sssp585_p31_WARWS_gens.data
plot_data[-1,2] = Sdiff1_WARWS_gens.data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-3e-11,3e-11),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=11, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="West Asia RWS1")
# %%
#   plot the Bar-plot of the West-Asia RWS2
plot_data = np.zeros((7,3))
plot_data[:-1,0] = Shis2_WARWS.sel(models=gmodels).data
plot_data[:-1,1] = Sssp585_p32_WARWS.sel(models=gmodels).data
plot_data[:-1,2] = Sdiff2_WARWS.sel(models=gmodels).data
plot_data[-1,0] = Shis2_WARWS_gens.data
plot_data[-1,1] = Sssp585_p32_WARWS_gens.data
plot_data[-1,2] = Sdiff2_WARWS_gens.data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-3e-11,3e-11),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=11, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="West Asia RWS2")
# %%
#   plot the Bar-plot of the East-Asia RWS1
plot_data = np.zeros((7,3))
plot_data[:-1,0] = Shis1_EARWS.sel(models=gmodels).data
plot_data[:-1,1] = Sssp585_p31_EARWS.sel(models=gmodels).data
plot_data[:-1,2] = Sdiff1_EARWS.sel(models=gmodels).data
plot_data[-1,0] = Shis1_EARWS_gens.data
plot_data[-1,1] = Sssp585_p31_EARWS_gens.data
plot_data[-1,2] = Sdiff1_EARWS_gens.data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-3e-11,3e-11),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=11, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="East Asia RWS1")
# %%
#   plot the Bar-plot of the East-Asia RWS2
plot_data = np.zeros((7,3))
plot_data[:-1,0] = Shis2_EARWS.sel(models=gmodels).data
plot_data[:-1,1] = Sssp585_p32_EARWS.sel(models=gmodels).data
plot_data[:-1,2] = Sdiff2_EARWS.sel(models=gmodels).data
plot_data[-1,0] = Shis2_EARWS_gens.data
plot_data[-1,1] = Sssp585_p32_EARWS_gens.data
plot_data[-1,2] = Sdiff2_EARWS_gens.data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-3e-11,3e-11),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=11, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="East Asia RWS2")
# %%
#   plot the scatter plot x:WARWS y:EARWS
pplt.rc["figure.facecolor"] = "white"
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
m = axs[0].scatter(SERA5_WARWS, SERA5_EARWS, cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="ERA5", marker="s", color="blue5",ec="grey7")
for num_models, mod in enumerate(models_array):
    m = axs[0].scatter(Shis_WARWS.sel(models=mod), Shis_EARWS.sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod,ec="grey7")
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(Shis_WARWS_ens, Shis_EARWS_ens, cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^",ec="grey7")
m = axs[0].scatter(Shis_WARWS_gens, Shis_EARWS_gens, cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*",ec="grey7")

# #   x-axis title
# axs[0].text(-0.90,0.03,s='West Asia RWS')
# #   y-axis title
# axs[0].text(0.03,-0.55,s='East Asia RWS')
axs[0].axhline(0,lw=1.0,color="grey7",zorder=0.9)
axs[0].axvline(0,lw=1.0,color="grey7",zorder=0.9)

xyregress = stats.linregress(Shis_WARWS.data,Shis_EARWS.data)
axs[0].line(np.linspace(-1e-11,2.5e-11), xyregress[0]*np.linspace(-1e-11,2.5e-11)+xyregress[1],zorder=0.8,color="grey7",ls="--")

axs[0].format(xlim=(-4e-11,4e-11), ylim=(-4e-11,4e-11), grid=False, xlabel="West Asia RWS", ylabel="East Asia RWS", ytickloc="both", xtickloc="both",rtitle="1979-2014")
# %%
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
# m = axs[0].scatter(SERA5_WARWS, SERA5_EARWS, cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="ERA5", marker="s", color="blue5",ec="grey7")
for num_models, mod in enumerate(models_array):
    m = axs[0].scatter(Sssp585_p3_WARWS.sel(models=mod), Sssp585_p3_EARWS.sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod,ec="grey7")
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(Sssp585_p3_WARWS_ens, Sssp585_p3_EARWS_ens, cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^",ec="grey7")
m = axs[0].scatter(Sssp585_p3_WARWS_gens, Sssp585_p3_EARWS_gens, cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*",ec="grey7")

# #   x-axis title
# axs[0].text(-0.90,0.03,s='West Asia RWS')
# #   y-axis title
# axs[0].text(0.03,-0.55,s='East Asia RWS')
axs[0].axhline(0,lw=1.0,color="grey7",zorder=0.9)
axs[0].axvline(0,lw=1.0,color="grey7",zorder=0.9)

xyregress = stats.linregress(Sssp585_p3_WARWS.data,Sssp585_p3_EARWS.data)
axs[0].line(np.linspace(-1e-11,2.5e-11), xyregress[0]*np.linspace(-1e-11,2.5e-11)+xyregress[1],zorder=0.8,color="grey7",ls="--")

axs[0].format(xlim=(-4e-11,4e-11), ylim=(-4e-11,4e-11), grid=False, xlabel="West Asia RWS", ylabel="East Asia RWS", ytickloc="both", xtickloc="both",rtitle="2064-2099")
# %%
#   plot the bar-plot of all models Corr(IndR, NCR)
plot_data = np.zeros((28,3))
plot_data[:-2,0] = IndR_his_NC_regress[0].data
plot_data[:-2,1] = IndR_ssp585_p3_NC_regress[0].data
plot_data[:-2,2] = IndR_diff_NC_slope.data
plot_data[-2,0] = IndR_his_NC_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-2,1] = IndR_ssp585_p3_NC_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-2,2] = IndR_diff_NC_slope.sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,0] = IndR_his_NC_regress[0].mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_NC_regress[0].mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_NC_slope.mean(dim="models", skipna=True).data

label_models = list(models_array)
label_models.append("gMME")
label_models.append("MME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-0.7,0.7),xlocator=np.arange(0,28), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and EAhigh")
# %%
#   plot the scatter plot EARWS and corr(IndR, NCR)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
m = axs[0].scatter(IndR_GPCP_NC_regress[2], SERA5_EARWS, cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="AIR&GPCP", marker="s", color="blue5",ec="grey7")

pmodels=['ACCESS-CM2','BCC-CSM2-MR','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','FGOALS-g3','GFDL-CM4','HadGEM3-GC31-LL','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC-ES2L','MIROC6','MPI-ESM1-2-HR','NESM3','NorESM2-LM','TaiESM1']

for num_models, mod in enumerate(pmodels):
    m = axs[0].scatter(IndR_his_NC_regress[2].sel(models=mod), Shis_EARWS.sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod,ec="grey7")
for num_models, mod in enumerate(gmodels):
    m = axs[0].scatter(IndR_his_NC_regress[2].sel(models=mod), Shis_EARWS.sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod,ec="grey7", marker="h")
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(IndR_his_NC_regress[2].mean(dim="models", skipna=True), Shis_EARWS_ens, cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^",ec="grey7")
m = axs[0].scatter(IndR_his_NC_regress[2].sel(models=gmodels).mean(dim="models", skipna=True), Shis_EARWS_gens, cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*",ec="grey7")

# #   x-axis title
# axs[0].text(-0.90,0.03,s='West Asia RWS')
# #   y-axis title
# axs[0].text(0.03,-0.55,s='East Asia RWS')
axs[0].axhline(0,lw=1.0,color="grey7",zorder=0.9)
axs[0].axvline(0,lw=1.0,color="grey7",zorder=0.9)

# xyregress = stats.linregress(IndR_his_NC_regress[2].sel(models=gmodels).data,Shis_EARWS.sel(models=gmodels).data)
# axs[0].line(np.linspace(-0.8,0.8), xyregress[0]*np.linspace(-0.8,0.8)+xyregress[1],zorder=0.8,color="grey7",ls="--")

axs[0].format(xlim=(-1,1), ylim=(-4e-11,4e-11), grid=False, xlabel="corr(IndR, NCR)", ylabel="East Asia RWS", ytickloc="both", xtickloc="both",rtitle="1979-2014")
# %%
#   scatter plot forr the WNPhigh and EARWS
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
m = axs[0].scatter(IndR_GPCP_WNPhigh_regress[2], SERA5_EARWS, cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="AIR&GPCP", marker="s", color="blue5",ec="grey7")

pmodels=['ACCESS-CM2','BCC-CSM2-MR','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','FGOALS-g3','GFDL-CM4','HadGEM3-GC31-LL','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC-ES2L','MIROC6','MPI-ESM1-2-HR','NESM3','NorESM2-LM','TaiESM1']

for num_models, mod in enumerate(pmodels):
    m = axs[0].scatter(IndR_his_WNPhigh_regress[2].sel(models=mod), Shis_EARWS.sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod,ec="grey7")
for num_models, mod in enumerate(gmodels):
    m = axs[0].scatter(IndR_his_WNPhigh_regress[2].sel(models=mod), Shis_EARWS.sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod,ec="grey7", marker="h")
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(IndR_his_WNPhigh_regress[2].mean(dim="models", skipna=True), Shis_EARWS_ens, cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^",ec="grey7")
m = axs[0].scatter(IndR_his_WNPhigh_regress[2].sel(models=gmodels).mean(dim="models", skipna=True), Shis_EARWS_gens, cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*",ec="grey7")

# #   x-axis title
# axs[0].text(-0.90,0.03,s='West Asia RWS')
# #   y-axis title
# axs[0].text(0.03,-0.55,s='East Asia RWS')
axs[0].axhline(0,lw=1.0,color="grey7",zorder=0.9)
axs[0].axvline(0,lw=1.0,color="grey7",zorder=0.9)

# xyregress = stats.linregress(IndR_his_NC_regress[2].sel(models=gmodels).data,Shis_EARWS.sel(models=gmodels).data)
# axs[0].line(np.linspace(-0.8,0.8), xyregress[0]*np.linspace(-0.8,0.8)+xyregress[1],zorder=0.8,color="grey7",ls="--")

axs[0].format(xlim=(-1,1), ylim=(-4e-11,4e-11), grid=False, xlabel="corr(IndR, WNPhigh)", ylabel="East Asia RWS", ytickloc="both", xtickloc="both",rtitle="1979-2014")
# %%
#   for word figure
#   fig. 1 plot the precipitation regress onto IndR(AIRI)
pplt.rc.grid = False
pplt.rc.reso = "lo"
pplt.rc["figure.facecolor"] = "white"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 3), (2, 1))
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(0, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [40.0, xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # India area
    x0 = India_W
    y0 = India_S
    width = India_E-India_W
    height = India_N-India_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj, linestyle="-")
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_AIR_India_pre_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.6,1.7,0.2),
    zorder=0.8,
    extend="both"
    )
sepl.plt_sig(
    pre_AIR_India_pre_slope, axs[0], n, np.where(pre_AIR_India_pre_pvalue[::n, ::n] < 0.10), "bright purple", 4.0,
)

axs[0].format(
    ltitle="1979-2014", rtitle="GPCP & AIR",
)
# ===================================================
con = axs[1].contourf(
    pre_his_India_pre_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.6,1.7,0.2),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_his_India_pre_slope_ens, axs[1], n, np.where(pre_his_India_pre_slope_ens_mask[::n, ::n] > 0.00), "bright purple", 4.0,
)

axs[1].format(
    ltitle="1979-2014", rtitle="MME",
)
# ===================================================
cb = fig.colorbar(con, loc="b", width=0.13, length=0.7, label="", ticklabelsize=7)
cb.set_ticks(np.arange(-2.0,2.1, 0.4))
fig.format(abc="(a)", abcloc="l")
# %%
#  fig. 4 gMME precipitation regress onto IndR
pplt.rc.grid = False
pplt.rc.reso = "lo"
pplt.rc["figure.facecolor"] = "white"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 3), (2, 1))
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([60, 90, 120, 150])  # 设置经度刻度
yticks = np.arange(0, 46, 15)  # 设置纬度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [40.0, xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
    # India area
    x0 = India_W
    y0 = India_S
    width = India_E-India_W
    height = India_N-India_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj, linestyle="-")
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_his_India_pre_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.6,1.7,0.2),
    zorder=0.8,
    extend="both"
    )
sepl.plt_sig(
    pre_his_India_pre_slope_gens, axs[0], n, np.where(pre_his_India_pre_slope_gens_mask[::n, ::n] > 0.00), "bright purple", 4.0,
)

axs[0].format(
    ltitle="1979-2014", rtitle="gMME",
)
# ===================================================
con = axs[1].contourf(
    pre_ssp585_p3_India_pre_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.6,1.7,0.2),
    zorder=0.8,
    extend="both"
    )
sepl.plt_sig(
    pre_ssp585_p3_India_pre_slope_gens, axs[1], n, np.where(pre_ssp585_p3_India_pre_slope_gens_mask[::n, ::n] > 0.00), "bright purple", 4.0,
)

axs[1].format(
    ltitle="2064-2099", rtitle="gMME",
)
# ===================================================
# con = axs[2].contourf(
#     pre_diff_India_pre_slope_gens,
#     cmap="ColdHot",
#     cmap_kw={"left": 0.06, "right": 0.94},
#     levels=np.arange(-1.6,1.7,0.2),
#     zorder=0.8,
#     extend="both"
#     )
# sepl.plt_sig(
#     pre_diff_India_pre_slope_gens, axs[2], n, np.where(pre_diff_India_pre_gens_mask[::n, ::n] > 0.00), "bright purple", 4.0,
# )

# axs[2].format(
#     ltitle="diff", rtitle="gMME",
# )
# ===================================================
cb = fig.colorbar(con, loc="b", width=0.13, length=0.7, label="", ticklabelsize=7)
cb.set_ticks(np.arange(-2.0,2.1, 0.4))
fig.format(abc="(a)", abcloc="l")
# %%
#   fig. 5 only plot the circulation regress onto IndR in ssp585_p3 and difference
startlevel=[-15, -8, -6]
spacinglevel=[1.5, 0.8, 0.6]
scalelevel=[0.23, 0.17, 0.14]

pplt.rc.grid = False
pplt.rc.reso = "lo"
pplt.rc["figure.facecolor"] = "white"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=6.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 10), (3, 3))
# plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(0, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ======================================
for ax in axs:
    rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
    ax.add_patch(rect)
    # # India area
    # x0 = India_W
    # y0 = India_S
    # width = India_E-India_W
    # height = India_N-India_S
    # sepl.patches(ax, x0 - cl, y0, width, height, proj, linestyle="-")
    # # NC area
    # x0 = NC_W
    # y0 = NC_S
    # width = NC_E-NC_W
    # height = NC_N-NC_S
    # sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # # SJ-KP area
    # x0 = SJ_W
    # y0 = SJ_S
    # width = SJ_E-SJ_W
    # height = SJ_N-SJ_S
    # sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ======================================
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    con = axs[num_lev, 0].contourf(
        IndR_his_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_gens.sel(level=lev), axs[num_lev, 0], n, np.where(IndR_his_hgt_slope_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[num_lev, 0].quiver(
        IndR_his_u_slope_gens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_slope_gens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[num_lev, 0].quiver(
        IndR_his_u_slope_gens.sel(level=lev).where(IndR_his_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_slope_gens.sel(level=lev).where(IndR_his_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[num_lev, 0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev, 0].format(
        ltitle="1979-2014 {:.0f}hPa".format(lev), rtitle="gMME",
    )
	# ======================================
    con = axs[num_lev, 1].contourf(
        IndR_ssp585_p3_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_ssp585_p3_hgt_slope_gens.sel(level=lev), axs[num_lev, 1], n, np.where(IndR_ssp585_p3_hgt_slope_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[num_lev, 1].quiver(
        IndR_ssp585_p3_u_slope_gens.sel(level=lev)[::ski, ::ski],
        IndR_ssp585_p3_v_slope_gens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[num_lev, 1].quiver(
        IndR_ssp585_p3_u_slope_gens.sel(level=lev).where(IndR_ssp585_p3_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_ssp585_p3_v_slope_gens.sel(level=lev).where(IndR_ssp585_p3_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[num_lev, 1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev, 1].format(
        ltitle="2064-2099 {:.0f}hPa".format(lev), rtitle="gMME",
    )
# ======================================
    con = axs[num_lev, 2].contourf(
        IndR_diff_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_diff_hgt_slope_gens.sel(level=lev), axs[num_lev, 2], n, np.where(IndR_diff_hgt_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[num_lev, 2].quiver(
        IndR_diff_u_slope_gens.sel(level=lev)[::ski, ::ski],
        IndR_diff_v_slope_gens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[num_lev, 2].quiver(
        IndR_diff_u_slope_gens.sel(level=lev).where(IndR_diff_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_diff_v_slope_gens.sel(level=lev).where(IndR_diff_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[num_lev, 2].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev, 2].format(
        ltitle="diff {:.0f}hPa".format(lev), rtitle="gMME",
    )
    cb = axs[num_lev, 2].colorbar(con, loc="r", width=0.13, length=0.8, label="", ticklabelsize=7)
    cb.set_ticks(np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]*2))
# ======================================
fig.format(abc="(a)", abcloc="l")
# %%
#   fig. 6
pplt.rc["figure.facecolor"] = "white"
fig = pplt.figure(span=False, share=False, refheight=3.0, refwidth=10.0, wspace=4.0, hspace=4.0, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=3)

#   plot the bar-plot of the WA high
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_WAhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_WAhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_WAhigh_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_WAhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_WAhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_WAhigh_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

label_models = list(gmodels)
label_models.append("gMME")

m = axs[0].bar(label_models,plot_data*1e6,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["1979-2014", "2064-2099", "diff"])
axs[0].format(ylim=(-3.0,3.0),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8", rtitle="WA")

#   plot the bar-plot of the EA high
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_EAhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_EAhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_EAhigh_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_EAhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_EAhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_EAhigh_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

label_models = list(gmodels)
label_models.append("gMME")

m = axs[1].bar(label_models,plot_data*1e6,width=0.4,cycle="tab10",edgecolor="grey7")
axs[1].axhline(0,lw=1.5,color="grey7")

# axs[1].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[1].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[1].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[1].legend(handles=m, loc='ur', labels=["1979-2014", "2064-2099", "diff"])
axs[1].format(ylim=(-2.0,2.0),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8", rtitle="EA")

#   plot the bar-plot of the WNP high
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_WNPhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_WNPhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_WNPhigh_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_WNPhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_WNPhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_WNPhigh_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

label_models = list(gmodels)
label_models.append("gMME")

m = axs[2].bar(label_models,plot_data*1e6,width=0.4,cycle="tab10",edgecolor="grey7")
axs[2].axhline(0,lw=1.5,color="grey7")
# axs[2].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[2].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[2].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[2].legend(handles=m, loc='ur', labels=["1979-2014", "2064-2099", "diff"])
axs[2].format(ylim=(-1.0,1.0),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8", rtitle="WNP")

# ax.outline_patch.set_linewidth(1.0)
fig.format(abc="(a)", abcloc="l")
# %%
#   fig. 7 Only plot the RWS for reanalysis historical and ssp585_p3
startlevel=-3e-11
spacinglevel=5e-12
pplt.rc.grid = False
pplt.rc.reso = "lo"
pplt.rc["figure.facecolor"] = "white"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=6.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=4, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(0, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)
# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    SERA5,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
    zorder=0.8,
    extend="both"
)

axs[0].format(
    rtitle="1979-2014", ltitle="ERA5",
)
# ======================================
con = axs[1].contourf(
    Shis_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
    zorder=0.8,
    extend="both"
)

sepl.plt_sig(
    Shis_gens, axs[1], n, np.where(Shis_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="gMME",
)
# ======================================
con = axs[2].contourf(
    Sssp585_p3_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
    zorder=0.8,
    extend="both"
)

sepl.plt_sig(
    Sssp585_p3_gens, axs[2], n, np.where(Sssp585_p3_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)

axs[2].format(
    rtitle="2064-2099", ltitle="gMME",
)
# ======================================
con = axs[3].contourf(
    Sdiff_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel/2, spacinglevel),
    zorder=0.8,
    extend="both"
)

sepl.plt_sig(
    Sdiff_gens, axs[3], n, np.where(Sdiff_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)

axs[3].format(
    rtitle="diff", ltitle="gMME",
)
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l")
# %%
#   fig. 8 plot the scatter plot x:WARWS y:EARWS
legendlist = []
pplt.rc["figure.facecolor"] = "white"
fig = pplt.figure(span=False, share=False, refwidth=6.0, refheight=4.0, outerpad=1.0)
# , wspace=4.0, hspace=6.0, outerpad=2.0
array = np.array([[1,1,2,2,5,5],[1,1,2,2,5,5],[3,3,3,4,4,4],[3,3,3,4,4,4]])
axs = fig.subplots(array, wspace=7.0)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
m = axs[0].scatter(SERA5_WARWS*1e11, SERA5_EARWS*1e11, cycle=cycle, labels="ERA5", marker="s", color="blue5",ec="grey7")
legendlist.append(m)

pmodels=['ACCESS-CM2','BCC-CSM2-MR','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','FGOALS-g3','GFDL-CM4','HadGEM3-GC31-LL','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC-ES2L','MIROC6','MPI-ESM1-2-HR','NESM3','NorESM2-LM','TaiESM1']

for num_models, mod in enumerate(pmodels):
    m = axs[0].scatter(Shis_WARWS.sel(models=mod)*1e11, Shis_EARWS.sel(models=mod)*1e11, cycle=cycle, labels=mod,ec="grey7")
    legendlist.append(m)
    
for num_models, mod in enumerate(gmodels):
    m = axs[0].scatter(Shis_WARWS.sel(models=mod)*1e11, Shis_EARWS.sel(models=mod)*1e11, cycle=cycle, labels=mod,ec="grey7", marker="h")
    legendlist.append(m)
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(Shis_WARWS_ens*1e11, Shis_EARWS_ens*1e11, cycle=cycle, labels="MME", marker="^",ec="grey7")
legendlist.append(m)
m = axs[0].scatter(Shis_WARWS_gens*1e11, Shis_EARWS_gens*1e11, cycle=cycle, labels="gMME", marker="*",ec="grey7")
legendlist.append(m)

# #   x-axis title
# axs[0].text(-0.90,0.03,s='West Asia RWS')
# #   y-axis title
# axs[0].text(0.03,-0.55,s='East Asia RWS')
axs[0].axhline(0,lw=1.0,color="grey7",zorder=0.9)
axs[0].axvline(0,lw=1.0,color="grey7",zorder=0.9)

xyregress = stats.linregress(Shis_WARWS.data*1e11,Shis_EARWS.data*1e11)
axs[0].line(np.linspace(-1,2.5), xyregress[0]*np.linspace(-1,2.5)+xyregress[1],zorder=0.8,color="grey7",ls="--")

axs[0].format(xlim=(-3.5,3.5), ylim=(-3.5,3.5), grid=False, xlabel="West Asia RWS", ylabel="East Asia RWS", ytickloc="both", xtickloc="both",ltitle="1979-2014", abc="(a)", abcloc="l")

# cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)

for num_models, mod in enumerate(pmodels):
    m = axs[1].scatter(Sssp585_p3_WARWS.sel(models=mod)*1e11, Sssp585_p3_EARWS.sel(models=mod)*1e11, cycle=cycle, labels=mod,ec="grey7")
for num_models, mod in enumerate(gmodels):
    m = axs[1].scatter(Sssp585_p3_WARWS.sel(models=mod)*1e11, Sssp585_p3_EARWS.sel(models=mod)*1e11, cycle=cycle, labels=mod,ec="grey7",marker="h")
# fig.legend(loc="bottom", labels=models)
# axs[1].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[1].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[1].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[1].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[1].scatter(Sssp585_p3_WARWS_ens*1e11, Sssp585_p3_EARWS_ens*1e11, cycle=cycle, labels="MME", marker="^",ec="grey7")
m = axs[1].scatter(Sssp585_p3_WARWS_gens*1e11, Sssp585_p3_EARWS_gens*1e11, cycle=cycle, labels="gMME", marker="*",ec="grey7")

# #   x-axis title
# axs[1].text(-0.90,0.03,s='West Asia RWS')
# #   y-axis title
# axs[1].text(0.03,-0.55,s='East Asia RWS')
axs[1].axhline(0,lw=1.0,color="grey7",zorder=0.9)
axs[1].axvline(0,lw=1.0,color="grey7",zorder=0.9)

xyregress = stats.linregress(Sssp585_p3_WARWS.data*1e11,Sssp585_p3_EARWS.data*1e11)
axs[1].line(np.linspace(-1,2.5), xyregress[0]*np.linspace(-1,2.5)+xyregress[1],zorder=0.8,color="grey7",ls="--")

axs[1].format(xlim=(-3.5,3.5), ylim=(-3.5,3.5), grid=False, xlabel="West Asia RWS", ylabel="East Asia RWS", ytickloc="both", xtickloc="both",ltitle="2064-2099", abc="(a)", abcloc="l")

axs[4].legend(handles=legendlist,loc="cl", ncols=2, frameon=False, space=3.0)
axs[4].axis("off")

#   plot the Bar-plot of the West-Asia RWS
plot_data = np.zeros((7,3))
plot_data[:-1,0] = Shis_WARWS.sel(models=gmodels).data
plot_data[:-1,1] = Sssp585_p3_WARWS.sel(models=gmodels).data
plot_data[:-1,2] = Sdiff_WARWS.sel(models=gmodels).data
plot_data[-1,0] = Shis_WARWS_gens.data
plot_data[-1,1] = Sssp585_p3_WARWS_gens.data
plot_data[-1,2] = Sdiff_WARWS_gens.data

label_models = list(gmodels)
label_models.append("gMME")

m = axs[2].bar(label_models,plot_data*1e11,width=0.4,cycle="tab10",edgecolor="grey7")
axs[2].axhline(0,lw=1.5,color="grey7")
# axs[2].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[2].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[2].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[2].legend(handles=m, loc='ur', labels=["1979-2014", "2064-2099", "diff"])
axs[2].format(ylim=(-3,3),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8", rtitle="West Asia RWS", abc="(a)", abcloc="l")

#   plot the Bar-plot of the East-Asia RWS
plot_data = np.zeros((7,3))
plot_data[:-1,0] = Shis_EARWS.sel(models=gmodels).data
plot_data[:-1,1] = Sssp585_p3_EARWS.sel(models=gmodels).data
plot_data[:-1,2] = Sdiff_EARWS.sel(models=gmodels).data
plot_data[-1,0] = Shis_EARWS_gens.data
plot_data[-1,1] = Sssp585_p3_EARWS_gens.data
plot_data[-1,2] = Sdiff_EARWS_gens.data

label_models = list(gmodels)
label_models.append("gMME")

m = axs[3].bar(label_models,plot_data*1e11,width=0.4,cycle="tab10",edgecolor="grey7")
axs[3].axhline(0,lw=1.5,color="grey7")
# axs[3].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[3].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[3].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[3].legend(handles=m, loc='ur', labels=["1979-2014", "2064-2099", "diff"])
axs[3].format(ylim=(-3,3),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8", rtitle="East Asia RWS", abc="(a)", abcloc="l")
# ax.outline_patch.set_linewidth(1.0)
# fig.format(abc="(a)", abcloc="l")
# %%
#   fig. 9 plot the Rossby wave Source change reason: term1 and term2
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=10.0, wspace=4.0, hspace=5.0, outerpad=2.0)
axs = fig.subplots(ncols=2, nrows=2)
#   plot the Bar-plot of the West-Asia RWS1
plot_data = np.zeros((7,3))
plot_data[:-1,0] = Shis1_WARWS.sel(models=gmodels).data
plot_data[:-1,1] = Sssp585_p31_WARWS.sel(models=gmodels).data
plot_data[:-1,2] = Sdiff1_WARWS.sel(models=gmodels).data
plot_data[-1,0] = Shis1_WARWS_gens.data
plot_data[-1,1] = Sssp585_p31_WARWS_gens.data
plot_data[-1,2] = Sdiff1_WARWS_gens.data

label_models = list(gmodels)
label_models.append("gMME")

m = axs[0,0].bar(label_models,plot_data*1e11,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0,0].axhline(0,lw=1.5,color="grey7")
# axs[0,0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0,0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0,0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0,0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0,0].format(ylim=(-3,3),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8", rtitle="West Asia RWS1")
# ax.outline_patch.set_linewidth(1.0)
# fig.format(suptitle="West Asia RWS1")
#   plot the Bar-plot of the West-Asia RWS2
plot_data = np.zeros((7,3))
plot_data[:-1,0] = Shis2_WARWS.sel(models=gmodels).data
plot_data[:-1,1] = Sssp585_p32_WARWS.sel(models=gmodels).data
plot_data[:-1,2] = Sdiff2_WARWS.sel(models=gmodels).data
plot_data[-1,0] = Shis2_WARWS_gens.data
plot_data[-1,1] = Sssp585_p32_WARWS_gens.data
plot_data[-1,2] = Sdiff2_WARWS_gens.data

label_models = list(gmodels)
label_models.append("gMME")

m = axs[0,1].bar(label_models,plot_data*1e11,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0,1].axhline(0,lw=1.5,color="grey7")
# axs[0,1].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0,1].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0,1].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0,1].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0,1].format(ylim=(-3,3),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8", rtitle="West Asia RWS2")
# ax.outline_patch.set_linewidth(1.0)
# fig.format(suptitle="West Asia RWS2")
#   plot the Bar-plot of the East-Asia RWS1
plot_data = np.zeros((7,3))
plot_data[:-1,0] = Shis1_EARWS.sel(models=gmodels).data
plot_data[:-1,1] = Sssp585_p31_EARWS.sel(models=gmodels).data
plot_data[:-1,2] = Sdiff1_EARWS.sel(models=gmodels).data
plot_data[-1,0] = Shis1_EARWS_gens.data
plot_data[-1,1] = Sssp585_p31_EARWS_gens.data
plot_data[-1,2] = Sdiff1_EARWS_gens.data

label_models = list(gmodels)
label_models.append("gMME")

m = axs[1,0].bar(label_models,plot_data*1e11,width=0.4,cycle="tab10",edgecolor="grey7")
axs[1,0].axhline(0,lw=1.5,color="grey7")
# axs[1,0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[1,0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[1,0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[1,0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[1,0].format(ylim=(-3,3),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8", rtitle="East Asia RWS1")
# ax.outline_patch.set_linewidth(1.0)
# fig.format(suptitle="East Asia RWS1")

#   plot the Bar-plot of the East-Asia RWS2
plot_data = np.zeros((7,3))
plot_data[:-1,0] = Shis2_EARWS.sel(models=gmodels).data
plot_data[:-1,1] = Sssp585_p32_EARWS.sel(models=gmodels).data
plot_data[:-1,2] = Sdiff2_EARWS.sel(models=gmodels).data
plot_data[-1,0] = Shis2_EARWS_gens.data
plot_data[-1,1] = Sssp585_p32_EARWS_gens.data
plot_data[-1,2] = Sdiff2_EARWS_gens.data

label_models = list(gmodels)
label_models.append("gMME")

m = axs[1,1].bar(label_models,plot_data*1e11,width=0.4,cycle="tab10",edgecolor="grey7")
axs[1,1].axhline(0,lw=1.5,color="grey7")
# axs[1,1].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[1,1].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[1,1].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[1,1].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[1,1].format(ylim=(-3,3),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8", rtitle="East Asia RWS2")
# ax.outline_patch.set_linewidth(1.0)
fig.format(abc="(a)", abcloc="l")
# %%
#   fig. 10 plot the SST reg SASMR in ERA5 HadISST, historical, ssp585_p3 and diff
startlevel=-6e-1
spacinglevel=0.12

pplt.rc.grid = False
pplt.rc.reso = "lo"
pplt.rc["figure.facecolor"] = "white"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
# plot_array = np.reshape(range(1, 31), (5, 6))
# plot_array[-1,-2:] = 0
axs = fig.subplots(ncols=1, nrows=4, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    IndR_Had_sst_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_Had_sst_slope, axs[0], n, np.where(IndR_Had_sst_pvalue[::n, ::n] <= 0.05), "bright purple", 3.0,
)

axs[0].format(
    ltitle="1979-2014", rtitle="HadISST & AIR",
)
# ======================================
con = axs[1].contourf(
    IndR_his_sst_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_his_sst_slope_gens, axs[1], n, np.where(IndR_his_sst_slope_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[1].format(
    ltitle="1979-2014", rtitle="gMME",
)
# ======================================
con = axs[2].contourf(
    IndR_ssp585_p3_sst_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_ssp585_p3_sst_slope_gens, axs[2], n, np.where(IndR_ssp585_p3_sst_slope_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[2].format(
    ltitle="2064-2099", rtitle="gMME",
)
# ======================================
con = axs[3].contourf(
    IndR_diff_sst_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_diff_sst_slope_gens, axs[3], n, np.where(IndR_diff_sst_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[3].format(
    ltitle="diff", rtitle="gMME",
)
# ======================================

fig.colorbar(con, loc="b", width=0.13, length=0.85, label="")
fig.format(abc="(a)", abcloc="l")
# %%
TIO_W = 50.0
TIO_E = 100.0
TIO_S = -20
TIO_N = 20

WP_W = 120
WP_E = 160
WP_S = -20
WP_N = 20

EP_W = 120
EP_E = 160
EP_S = -20
EP_N = 20

lat_TIO_range = lat[(lat>=TIO_S) & (lat<=TIO_N)]
lon_TIO_range = lon[(lon>=TIO_W) & (lon<=TIO_E)]

lat_WP_range = lat[(lat>=WP_S) & (lat<=WP_N)]
lon_WP_range = lon[(lon>=WP_W) & (lon<=WP_E)]

SSThis_TIO_JJA = ca.cal_lat_weighted_mean(IndR_his_sst_slope.sel(lat=lat_TIO_range, lon=lon_TIO_range)).mean(dim="lon", skipna=True)
SSTssp585_p3_TIO_JJA = ca.cal_lat_weighted_mean(IndR_ssp585_p3_sst_slope.sel(lat=lat_TIO_range, lon=lon_TIO_range)).mean(dim="lon", skipna=True)

SSThis_WP_JJA = ca.cal_lat_weighted_mean(IndR_his_sst_slope.sel(lat=lat_WP_range, lon=lon_WP_range)).mean(dim="lon", skipna=True)
SSTssp585_p3_WP_JJA = ca.cal_lat_weighted_mean(IndR_ssp585_p3_sst_slope.sel(lat=lat_WP_range, lon=lon_WP_range)).mean(dim="lon", skipna=True)

SSTa_his_WPTIO_JJA = SSThis_WP_JJA-SSThis_TIO_JJA
SSTa_ssp585_p3_WPTIO_JJA = SSTssp585_p3_WP_JJA-SSTssp585_p3_TIO_JJA

SSTa_diff_WPTIO_JJA = SSTa_ssp585_p3_WPTIO_JJA - SSTa_his_WPTIO_JJA
# %%
#	plot the bar plot
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=5.0, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
plot_data = np.zeros((7,3))
plot_data[:-1,0] = SSTa_his_WPTIO_JJA.sel(models=gmodels).data
plot_data[:-1,1] = SSTa_ssp585_p3_WPTIO_JJA.sel(models=gmodels).data
plot_data[:-1,2] = SSTa_diff_WPTIO_JJA.sel(models=gmodels).data
plot_data[-1,0] = SSTa_his_WPTIO_JJA.sel(models=gmodels).mean(dim="models",skipna=True).data
plot_data[-1,1] = SSTa_ssp585_p3_WPTIO_JJA.sel(models=gmodels).mean(dim="models",skipna=True).data
plot_data[-1,2] = SSTa_diff_WPTIO_JJA.sel(models=gmodels).mean(dim="models",skipna=True).data

label_models = list(gmodels)
label_models.append("gMME")

m = axs[0].bar(label_models,plot_data*1e11,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-2e10,2e10),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8", rtitle="SSTa WP-TIO")
# ax.outline_patch.set_linewidth(1.0)
fig.format(abc="(a)", abcloc="l")
# %%
# plot the change of the 200hPa level circulation
# calculate the hgt and wind difference in the two periods
udiff_cli = (ussp585_p3_ver_JJA.sel(level=200.0).mean(dim="time",skipna=True) - uhis_ver_JJA.sel(level=200.0).mean(dim="time",skipna=True))
udiff_cli_gens = udiff_cli.sel(models=gmodels).mean(dim="models",skipna=True)
udiff_cli_gens_mask = ca.cal_mmemask(udiff_cli.sel(models=gmodels))

vdiff_cli = (vssp585_p3_ver_JJA.sel(level=200.0).mean(dim="time",skipna=True) - vhis_ver_JJA.sel(level=200.0).mean(dim="time",skipna=True))
vdiff_cli_gens = vdiff_cli.sel(models=gmodels).mean(dim="models",skipna=True)
vdiff_cli_gens_mask = ca.cal_mmemask(vdiff_cli.sel(models=gmodels))

hgtdiff_cli = (hgtssp585_p3_ver_JJA.sel(level=200.0).mean(dim="time",skipna=True) - hgthis_ver_JJA.sel(level=200.0).mean(dim="time",skipna=True))
hgtdiff_cli_gens = hgtdiff_cli.sel(models=gmodels).mean(dim="models",skipna=True)
hgtdiff_cli_gens_mask = ca.cal_mmemask(hgtdiff_cli.sel(models=gmodels))

winddiff_cli_gens_mask = ca.wind_check(
    xr.where(udiff_cli_gens_mask > 0.0, 1.0, 0.0),
    xr.where(vdiff_cli_gens_mask > 0.0, 1.0, 0.0),
    xr.where(udiff_cli_gens_mask > 0.0, 1.0, 0.0),
    xr.where(vdiff_cli_gens_mask > 0.0, 1.0, 0.0),
)

#   plot the diff climatology-u,v,hgt wind in 200hPa 
startlevel=-80
spacinglevel=10
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
# plot_array = np.reshape(range(1, 9), (2, 4))
# plot_array[-1,-1] = 0
axs = fig.subplots(ncols=1, nrows=3, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-15, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)
# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
# ======================================
con = axs[0].contourf(
    hgthis_ver_JJA.sel(level=200.0, models=gmodels).mean(dim=["time", "models"]),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
m = axs[0].quiver(
    uhis_ver_JJA.sel(level=200.0, models=gmodels).mean(dim=["time", "models"])[::ski, ::ski],
    vhis_ver_JJA.sel(level=200.0, models=gmodels).mean(dim=["time", "models"])[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
axs[0].format(
    rtitle="1979-2014", ltitle="gMME",
)
# ======================================
con = axs[1].contourf(
    hgtssp585_p3_ver_JJA.sel(level=200.0, models=gmodels).mean(dim=["time", "models"]),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
m = axs[1].quiver(
    ussp585_p3_ver_JJA.sel(level=200.0, models=gmodels).mean(dim=["time", "models"])[::ski, ::ski],
    vssp585_p3_ver_JJA.sel(level=200.0, models=gmodels).mean(dim=["time", "models"])[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
axs[1].format(
    rtitle="2064-2099", ltitle="gMME",
)
axs[1].colorbar(con, loc="b")
# ======================================
startlevel=-30
spacinglevel=3
# ======================================
con = axs[2].contourf(
    hgtdiff_cli_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
axs[2].quiver(
    udiff_cli_gens[::ski, ::ski],
    vdiff_cli_gens[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.23,
    pivot="mid",
    color="grey6",
    )
axs[2].format(
    rtitle="diff", ltitle="gMME",
)
m = axs[2].quiver(
        udiff_cli_gens.where(winddiff_cli_gens_mask > 0.0)[::ski, ::ski],
        vdiff_cli_gens.where(winddiff_cli_gens_mask > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=0.23,
        pivot="mid",
        color="black",
    )

qk = axs[2].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
# ======================================
axs[2].colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="200hPa climatology")
# %%
# plot the climatology divergent wind and climatology vorticity

udivhis_bar_gens = udivhis_bar.sel(models=gmodels).mean(dim="models",skipna=True)
udivhis_bar_gens_mask = xr.where((ca.MME_reg_mask(udivhis_bar_gens, udivhis_bar.sel(models=gmodels).std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(udivhis_bar.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

vdivhis_bar_gens = vdivhis_bar.sel(models=gmodels).mean(dim="models",skipna=True)
vdivhis_bar_gens_mask = xr.where((ca.MME_reg_mask(vdivhis_bar_gens, vdivhis_bar.sel(models=gmodels).std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(vdivhis_bar.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

udivssp585_p3_bar_gens = udivssp585_p3_bar.sel(models=gmodels).mean(dim="models",skipna=True)
udivssp585_p3_bar_gens_mask = xr.where((ca.MME_reg_mask(udivssp585_p3_bar_gens, udivssp585_p3_bar.sel(models=gmodels).std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(udivssp585_p3_bar.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

vdivssp585_p3_bar_gens = vdivssp585_p3_bar.sel(models=gmodels).mean(dim="models",skipna=True)
vdivssp585_p3_bar_gens_mask = xr.where((ca.MME_reg_mask(vdivssp585_p3_bar_gens, vdivssp585_p3_bar.sel(models=gmodels).std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(vdivssp585_p3_bar.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

udivdiff_bar_gens = udivssp585_p3_bar_gens-udivhis_bar_gens

vdivdiff_bar_gens = vdivssp585_p3_bar_gens-vdivhis_bar_gens

udivdiff_bar_gens_mask = ca.cal_mmemask((udivssp585_p3_bar-udivhis_bar).sel(models=gmodels))
vdivdiff_bar_gens_mask = ca.cal_mmemask((vdivssp585_p3_bar-vdivhis_bar).sel(models=gmodels))

winddivhis_bar_gens_mask = ca.wind_check(
    xr.where(udivhis_bar_gens_mask > 0.0, 1.0, 0.0),
    xr.where(vdivhis_bar_gens_mask > 0.0, 1.0, 0.0),
    xr.where(udivhis_bar_gens_mask > 0.0, 1.0, 0.0),
    xr.where(vdivhis_bar_gens_mask > 0.0, 1.0, 0.0),
)
winddivssp585_p3_bar_gens_mask = ca.wind_check(
    xr.where(udivssp585_p3_bar_gens_mask > 0.0, 1.0, 0.0),
    xr.where(vdivssp585_p3_bar_gens_mask > 0.0, 1.0, 0.0),
    xr.where(udivssp585_p3_bar_gens_mask > 0.0, 1.0, 0.0),
    xr.where(vdivssp585_p3_bar_gens_mask > 0.0, 1.0, 0.0),
)
winddivdiff_bar_gens_mask = ca.wind_check(
    xr.where(udivdiff_bar_gens_mask > 0.0, 1.0, 0.0),
    xr.where(vdivdiff_bar_gens_mask > 0.0, 1.0, 0.0),
    xr.where(udivdiff_bar_gens_mask > 0.0, 1.0, 0.0),
    xr.where(vdivdiff_bar_gens_mask > 0.0, 1.0, 0.0),
)

startlevel=-80
spacinglevel=10
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
# plot_array = np.reshape(range(1, 9), (2, 4))
# plot_array[-1,-1] = 0
axs = fig.subplots(ncols=1, nrows=4, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-15, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)
# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
# ======================================
m = axs[0].quiver(
    udivERA5_bar[::ski, ::ski],
    vdivERA5_bar[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=1.0, label="1.0", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
axs[0].format(
    ltitle="1979-2014", rtitle="ERA5",
)
# ======================================
axs[1].quiver(
    udivhis_bar_gens[::ski, ::ski],
    vdivhis_bar_gens[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.8,
    pivot="mid",
    color="grey7",
    )
m = axs[1].quiver(
    udivhis_bar_gens.where(winddivhis_bar_gens_mask > 0.0)[::ski, ::ski],
    vdivhis_bar_gens.where(winddivhis_bar_gens_mask > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=1.0, label="1.0", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
axs[1].format(
    ltitle="1979-2014", rtitle="gMME",
)
# ======================================
axs[2].quiver(
    udivssp585_p3_bar_gens[::ski, ::ski],
    vdivssp585_p3_bar_gens[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.8,
    pivot="mid",
    color="grey7",
    )
m = axs[2].quiver(
    udivssp585_p3_bar_gens.where(winddivssp585_p3_bar_gens_mask > 0.0)[::ski, ::ski],
    vdivssp585_p3_bar_gens.where(winddivssp585_p3_bar_gens_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.8,
    pivot="mid",
    color="black",
    )
qk = axs[2].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=1.0, label="1.0", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
axs[2].format(
    ltitle="2064-2099", rtitle="gMME",
)
# ======================================
axs[3].quiver(
    udivdiff_bar_gens[::ski, ::ski],
    vdivdiff_bar_gens[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.4,
    pivot="mid",
    color="grey7",
    )
m = axs[3].quiver(
    udivdiff_bar_gens.where(winddivdiff_bar_gens_mask > 0.0)[::ski, ::ski],
    vdivdiff_bar_gens.where(winddivdiff_bar_gens_mask > 0.0)[::ski, ::ski],
    zorder=1.1,
    headwidth=2.6,
    headlength=2.3,
    headaxislength=2.3,
    scale_units="xy",
    scale=0.4,
    pivot="mid",
    color="black",
    )
qk = axs[3].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=1.0, label="1.0", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
axs[3].format(
    ltitle="diff", rtitle="gMME",
)
fig.format(abc="(a)", abcloc="l", suptitle="200hPa  divergent wind climatology")
# %%
