'''
Author: ChenHJ
Date: 2022-03-02 16:45:05
LastEditors: ChenHJ
LastEditTime: 2022-05-21 00:12:57
FilePath: /chenhj/0302code/data_predealing.py
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
modelname = ["ACCESS-CM2", "BCC-CSM2-MR", "CAMS-CSM1-0", "CanESM5", "CESM2", "CESM2-WACCM", "CMCC-ESM2", "CNRM-CM6-1", "CNRM-ESM2-1", "EC-Earth3-Veg", "EC-Earth3", "FGOALS-g3", "GFDL-CM4", "HadGEM3-GC31-LL", "IITM-ESM", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G", "MIROC-ES2L", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0", "NESM3", "NorESM2-LM", "TaiESM1", "UKESM1-0-LL"]
rlzn = ["r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r11i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f2", "r1i1p1f2", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f3", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f2", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r3i1p1f2"]
for model,rl in zip(modelname, rlzn):
    print(model)
    srcPath = "/home/ys17-23/Extension/CMIP6/CMIP6/"+ model +"/historical/" + rl + "/Amon"
    tmpPath = "/home/ys17-23/Extension/CMIP6/CMIP6/tmpPath"
    dstPath = "/home/ys17-23/Extension/personal-data/chenhj/CMIP6/historical"
    variable = ["ts"]
    freq = "Amon"
    ca.CMIP6_predealing_1(srcPath, tmpPath, dstPath, variable, freq, rl)
# %%
modelname = ["ACCESS-CM2", "BCC-CSM2-MR", "CAMS-CSM1-0", "CanESM5", "CESM2", "CESM2-WACCM", "CMCC-ESM2", "CNRM-CM6-1", "CNRM-ESM2-1", "EC-Earth3-Veg", "EC-Earth3", "FGOALS-g3", "GFDL-CM4", "HadGEM3-GC31-LL", "IITM-ESM", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G", "MIROC-ES2L", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0", "NESM3", "NorESM2-LM", "TaiESM1", "UKESM1-0-LL"]
yearstart = "185001"
yearend = ["201412", "201412", "201412", "201412", "201412", "201412", "201412", "201412", "201412", "201412", "201412", "201612", "201412", "201412", "201412", "201412", "201412", "201412", "201412", "201412", "201412", "201412", "201412", "201412", "201412", "201412", "201412"]
dstPath = "/home/ys17-23/Extension/personal-data/chenhj/CMIP6/historical"
variable = ["ts", "tos"]

for var in variable:
    dstpath = os.path.join(dstPath, var)
    ca.CMIP6_check(dstpath, modelname, yearstart, yearend)

# %%
modelname = ["ACCESS-CM2", "BCC-CSM2-MR", "CAMS-CSM1-0", "CanESM5", "CESM2", "CESM2-WACCM", "CMCC-ESM2", "CNRM-CM6-1", "CNRM-ESM2-1", "EC-Earth3-Veg", "EC-Earth3", "FGOALS-g3", "GFDL-CM4", "HadGEM3-GC31-LL", "IITM-ESM", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G", "MIROC-ES2L", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0", "NESM3", "NorESM2-LM", "TaiESM1", "UKESM1-0-LL"]
yearstart = "201501"
yearend = ["230012", "210012", "209912", "230012", "210012", "229912", "210012", "210012", "210012", "210012", "210012", "210012", "210012", "210012", "209912", "210012", "210012", "230012", "210012", "210012", "210012", "210012", "230012", "210012", "210012", "210012", "210012"]
dstPath = "/home/ys17-23/Extension/personal-data/chenhj/CMIP6/ssp585"
variable = ["ts", "tos"]

for var in variable:
    dstpath = os.path.join(dstPath, var)
    ca.CMIP6_check(dstpath, modelname, yearstart, yearend)
# %%

srcPath = "/home/ys17-23/chenhj/Nino/ERA5"
# %%
f1 = xr.open_dataset("/home/ys17-23/chenhj/Nino/ERA5/tp_mon_1x1_1979_2021.nc")
tmp1 = f1["tp"].sel(expver=1)[:-2].drop('expver')
tmp2 = f1["tp"].sel(expver=5)[-2:].drop('expver')
tp = xr.concat([tmp1, tmp2], dim="time")
tp.to_netcdf("/home/ys17-23/chenhj/Nino/ERA5/tp_mon_1x1_197901_202111.nc")

# %%
f2 = xr.open_dataset("/home/ys17-23/chenhj/Nino/ERA5/hgt_mon_1x1_1979_2021.nc")
print(f2["z"])
tmp1 = f2["z"].sel(expver=1)[:-2].drop('expver') / 9.8
tmp2 = f2["z"].sel(expver=5)[-2:].drop('expver') / 9.8
hgt = xr.concat([tmp1, tmp2], dim="time")
print(hgt)
hgt.to_netcdf("/home/ys17-23/chenhj/Nino/ERA5/hgt_mon_1x1_197901_202111.nc")
# %%
f3 = xr.open_dataset("/home/ys17-23/chenhj/Nino/ERA5/hgt_mon_1x1_1950_1978.nc")
hgt2 = f3["z"] / 9.8
hgt2.to_netcdf("/home/ys17-23/chenhj/Nino/ERA5/hgt_mon_1x1_195001_197812.nc")
# %%
f4 = xr.open_dataset("/home/ys17-23/chenhj/Nino/ERA5/tp_mon_1x1_1950_1978.nc")
tp2 = f4["tp"]
tpnew = xr.concat([tp, tp2], "time")
tpnew.to_netcdf("/home/ys17-23/chenhj/Nino/ERA5/tp_mon_1x1_195001_202111.nc")
# %%
variable = ["ua", "va", "ta", "wap", "hus", "zg", "pr"]
for var in variable:
    srcPath = "/home/ys17-23/chenhj/CMIP6/historical/" + var
    dstPath = "/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/" + var
    start = 1950
    end = 2014
    ca.p_year(srcPath, dstPath, start, end)
# %%
variable = ["ua", "va", "ta", "wap", "hus", "zg", "pr"]
for var in variable:
    srcPath = "/home/ys17-23/chenhj/CMIP6/ssp585/" + var
    dstPath = "/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/ssp585/" + var
    start = 2015
    end = 2099
    ca.p_year(srcPath, dstPath, start, end)
# %%
#   uniform the time of different models
reload(ca)
path1 = "/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/ssp585"
variable = ["pr"]
for var in variable:
    varpath = os.path.join(path1, var)
    g = os.walk(varpath)
    for path, dir_list, file_list in g:
        for filename in file_list:
            print(filename)
            ca.uniform_timestamp(os.path.join(path, filename), os.path.join("/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/ssp5852", var, filename), var, "20150101", "20991201", "MS")
# %%
def uniform_plev(filepath, dstpath, var):
    plev = np.array([100000.0, 92500.0, 85000.0, 70000.0, 60000.0, 50000.0, 40000.0, 30000.0, 25000.0, 20000.0, 15000.0, 10000.0, 7000.0, 5000.0, 3000.0, 2000.0, 1000.0, 500.0, 100.0])
    f = xr.open_dataset(filepath)
    fvar = f[var]
    fvar.coords['plev'] = plev
    fvar.to_netcdf(dstpath)

path1 = "/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/ssp585"
variable = ["zg", "ua", "va", "hus", "ta", "wap"]
for var in variable:
    varpath = os.path.join(path1, var)
    g = os.walk(varpath)
    for path, dir_list, file_list in g:
        for filename in file_list:
            print(filename)
            uniform_plev(os.path.join(path, filename), os.path.join("/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/ssp5852", var, filename), var)
# %%
#   calculate the ensmean of different variables
    
def cdo_ensmean(srcPath, dstPath):
    g = os.walk(srcPath)
    inputString = ""
    mergelist = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if re.search("IITM-ESM", file_name) == None and re.search("ensemble", file_name) == None:
                mergelist.append(os.path.join(path, file_name))
    for i in range(len(mergelist)):
        inputString += mergelist[i] + " "
    print(inputString)
    cdo.ensmean(input=inputString, output=os.path.join(dstPath, var + "_Amon_ensemble_ssp585_gn_201501-209912.nc"))
    
variables = ["ua", "va", "hus", "ta", "wap", "ps", "pr", "zg"]
for var in variables:
    srcPath = os.path.join("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585", var)
    dstPath = os.path.join("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585", var)
    cdo_ensmean(srcPath, dstPath)
# %%
modelname = ["ACCESS-CM2", "BCC-CSM2-MR", "CAMS-CSM1-0", "CanESM5", "CESM2", "CESM2-WACCM", "CMCC-ESM2", "CNRM-CM6-1", "CNRM-ESM2-1", "EC-Earth3-Veg", "EC-Earth3", "FGOALS-g3", "GFDL-CM4", "HadGEM3-GC31-LL", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G", "MIROC-ES2L", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0", "NESM3", "NorESM2-LM", "TaiESM1", "UKESM1-0-LL"]
rlzn = ["r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r11i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f2", "r1i1p1f2", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f3", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f2", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r3i1p1f2"]
for model,rl in zip(modelname, rlzn):
    print(model)
    srcPath = "/home/ys17-23/Extension/CMIP6/CMIP6/"+ model +"/historical/" + rl + "/Amon"
    tmpPath = "/home/ys17-23/chenhj/CMIP6/tmpPath"
    dstPath = "/home/ys17-23/chenhj/CMIP6/historical"
    variable = ["ps"]
    freq = "Amon"
    ca.CMIP6_predealing_1(srcPath, tmpPath, dstPath, variable, freq, rl)
# %%
#   select time
variable = ["ts", "tos"]
for var in variable:
    srcPath = "/home/ys17-23/Extension/personal-data/chenhj/CMIP6/historical/" + var
    dstPath = "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/" + var
    start = 1950
    end = 2014
    ca.p_year(srcPath, dstPath, start, end)
# %%
#   uniform the time of different models
reload(ca)
path1 = "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/"
variable = ["ts", "tos"]
for var in variable:
    varpath = os.path.join(path1, var)
    g = os.walk(varpath)
    for path, dir_list, file_list in g:
        for filename in file_list:
            print(filename)
            ca.uniform_timestamp(os.path.join(path, filename), os.path.join("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/", var + "2", filename), var, "20150101", "20991201", "MS")
# %%
#   Dealing the AIR index
f = open("/home/ys17-23/Extension/All_India_Rainfall_index/iitm-regionrf.txt")
time = pd.date_range(str(18710101), str(20161201), freq="MS")
data = np.zeros(np.size(time), dtype=np.float32)
for i in np.arange(1, 153):
    if i <= 6:
        f.readline()
    else:
        a = f.readline()
        for m in np.arange(1,13,1):
            # print(a[5+6*(m-1):11+6*(m-1)])
            # data[(i-????????????????????????)*12??????+???m-1?????????] = float(a[????????????????????????????????????+???????????????????????????????????????*???m-1????????????????????????????????????????????????+???????????????????????????????????????*m??????])
            data[(i-7)*12+m-1] = float(a[5+6*(m-1):5+6*m])/10.0
f.close()
print(data[-12:])
foo = xr.DataArray(data, coords=[time], dims=["time"])
foo.name = "precip"
foo.attrs["Name of the region"] = "ALL-INDIA RAINFALL"
foo.attrs["Data Period"] = "1871-2016"
foo.attrs["Area of the region in sq. km"] = "30 SUBDIVISIONS AREA 2880324 SQ.KM."
foo.attrs["units"] = "mm/month"
foo.to_netcdf("/home/ys17-23/Extension/All_India_Rainfall_index/AIR.nc")

# %%
#   calculate the mm/day
f = open("/home/ys17-23/Extension/All_India_Rainfall_index/iitm-regionrf.txt")
time = pd.date_range(str(18710101), str(20161201), freq="MS")
data = np.zeros(np.size(time), dtype=np.float32)
monthday1 = [31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.]
monthday2 = [31., 29., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.]
for i in np.arange(1, 153):
    if i <= 6:
        f.readline()
    else:
        a = f.readline()
        for m in np.arange(1,13,1):
            # print(a[5+6*(m-1):11+6*(m-1)])
            # data[(i-????????????????????????)*12??????+???m-1?????????] = float(a[????????????????????????????????????+???????????????????????????????????????*???m-1????????????????????????????????????????????????+???????????????????????????????????????*m??????])
            if (i-7+1871)%4==0 and (i-7+1871)%100!=0:
                data[(i-7)*12+m-1] = float(a[5+6*(m-1):5+6*m])/10.0/monthday2[m-1]
            elif (i-7+1871)%400==0:
                data[(i-7)*12+m-1] = float(a[5+6*(m-1):5+6*m])/10.0/monthday2[m-1]
            else:
                data[(i-7)*12+m-1] = float(a[5+6*(m-1):5+6*m])/10.0/monthday1[m-1]
f.close()
print(data[-12:])
foo = xr.DataArray(data, coords=[time], dims=["time"])
foo.name = "precip"
foo.attrs["Name of the region"] = "ALL-INDIA RAINFALL"
foo.attrs["Data Period"] = "1871-2016"
foo.attrs["Area of the region in sq. km"] = "30 SUBDIVISIONS AREA 2880324 SQ.KM."
foo.attrs["units"] = "mm/day"
foo.to_netcdf("/home/ys17-23/Extension/All_India_Rainfall_index/AIR_mmperday.nc")

# %%
