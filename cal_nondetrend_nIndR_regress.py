'''
Author: ChenHJ
Date: 2022-05-25 16:39:12
LastEditors: ChenHJ
LastEditTime: 2022-05-25 19:26:49
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
qERA5_ver_JJA = ca.p_time(qERA5, 6, 9, True)
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
sstHad_JJA.coords["time"] = preAIR_JJA.coords["time"]
(
    IndR_Had_sst_slope,
    IndR_Had_sst_intercept,
    IndR_Had_sst_rvalue,
    IndR_Had_sst_pvalue,
    IndR_Had_sst_hypothesis,
) = ca.dim_linregress(preAIR_JJA, sstHad_JJA)

(
    IndR_his_sst_slope,
    IndR_his_sst_intercept,
    IndR_his_sst_rvalue,
    IndR_his_sst_pvalue,
    IndR_his_sst_hypothesis,
) = ca.dim_linregress(prehis_India_JJA, ssthis_JJA)

(
    IndR_ssp585_p3_sst_slope,
    IndR_ssp585_p3_sst_intercept,
    IndR_ssp585_p3_sst_rvalue,
    IndR_ssp585_p3_sst_pvalue,
    IndR_ssp585_p3_sst_hypothesis,
) = ca.dim_linregress(pressp585_p3_India_JJA, sstssp585_p3_JJA)

#   save the result of the sst regression
lat=IndR_Had_sst_slope.coords["lat"]
lon=IndR_Had_sst_slope.coords["lon"]

IndR_Had_sst_regress = xr.Dataset(
    data_vars=dict(
        slope=(["lat", "lon"], IndR_Had_sst_slope.data),
        intercept=(["lat", "lon"], IndR_Had_sst_intercept.data),
        rvalue=(["lat", "lon"], IndR_Had_sst_rvalue.data),
        pvalue=(["lat", "lon"], IndR_Had_sst_pvalue.data),
        hypothesis=(["lat", "lon"], IndR_Had_sst_hypothesis.data),
    ),
    coords=dict(
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="sst fields of HadISST regress onto 1979-2014 AIR"),
)

IndR_his_sst_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "lat", "lon"], IndR_his_sst_slope.data),
        intercept=(["models", "lat", "lon"], IndR_his_sst_intercept.data),
        rvalue=(["models", "lat", "lon"], IndR_his_sst_rvalue.data),
        pvalue=(["models", "lat", "lon"], IndR_his_sst_pvalue.data),
        hypothesis=(["models", "lat", "lon"], IndR_his_sst_hypothesis.data),
    ),
    coords=dict(
        models=models_array,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="sst fields of historical regress onto 1979-2014 IndR"),
)

IndR_ssp585_p3_sst_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "lat", "lon"], IndR_ssp585_p3_sst_slope.data),
        intercept=(["models", "lat", "lon"], IndR_ssp585_p3_sst_intercept.data),
        rvalue=(["models", "lat", "lon"], IndR_ssp585_p3_sst_rvalue.data),
        pvalue=(["models", "lat", "lon"], IndR_ssp585_p3_sst_pvalue.data),
        hypothesis=(["models", "lat", "lon"], IndR_ssp585_p3_sst_hypothesis.data),
    ),
    coords=dict(
        models=models_array,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="sst fields of ssp585_p3 regress onto 2064-2099 IndR"),
)

IndR_Had_sst_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_Had_sst_regress.nc")
IndR_his_sst_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_sst_regress.nc")
IndR_ssp585_p3_sst_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_sst_regress.nc")
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
lon_ranking_range2 = lon[(lon>=110) & (lon<=137.0)]

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
