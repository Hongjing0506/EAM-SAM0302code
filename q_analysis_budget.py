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
import statsmodels.api as sm
import metpy.constants as constants

# %%

# %% [markdown]
# ## 读取数据

# %%
models = ['ACCESS-CM2', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CanESM5', 'CESM2',
       'CESM2-WACCM', 'CMCC-ESM2', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'EC-Earth3',
       'EC-Earth3-Veg', 'FGOALS-g3', 'GFDL-CM4', 'HadGEM3-GC31-LL',
       'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'MIROC6',
       'MIROC-ES2L', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM',
       'TaiESM1', 'UKESM1-0-LL']
pr_models = []
hfls_models = []
term1_models = []
term2_models = []
term3_models = []
term4_models = []
res_models = []
for mod in models:
  fhfls_ssp585 = xr.open_dataset("/home/ys17-23/Extension2/LuoHL/EmerCon/ssp585/mergetime_remap/hfls_{}_ssp585_remap.nc".format(mod))
  hfls_ssp585 = fhfls_ssp585["hfls"].sel(time=(fhfls_ssp585.time.dt.year>=2050)&(fhfls_ssp585.time.dt.year<=2099))*0.408*1e-6
  hfls_ssp585.attrs["units"] = "mm/day"

  fpr_ssp585 = xr.open_dataset("/home/ys17-23/Extension2/LuoHL/EmerCon/ssp585/mergetime_remap/pr_{}_ssp585_remap.nc".format(mod))
  pr_ssp585 = fpr_ssp585["pr"].sel(time=(fpr_ssp585.time.dt.year>=2050)&(fpr_ssp585.time.dt.year<=2099))
  pr_ssp585.attrs["units"] = "mm/day"

  fhus_ssp585 = xr.open_dataset("/home/ys17-23/Extension2/LuoHL/EmerCon/ssp585/mergetime_remap/hus_{}_ssp585_remap.nc".format(mod))
  hus_ssp585 = fhus_ssp585["hus"].sel(time=(fhus_ssp585.time.dt.year>=2050)&(fhus_ssp585.time.dt.year<=2099))

  fua_ssp585 = xr.open_dataset("/home/ys17-23/Extension2/LuoHL/EmerCon/ssp585/mergetime_remap/ua_{}_ssp585_remap.nc".format(mod))
  ua_ssp585 = fua_ssp585["ua"].sel(time=(fua_ssp585.time.dt.year>=2050)&(fua_ssp585.time.dt.year<=2099))

  fva_ssp585 = xr.open_dataset("/home/ys17-23/Extension2/LuoHL/EmerCon/ssp585/mergetime_remap/va_{}_ssp585_remap.nc".format(mod))
  va_ssp585 = fva_ssp585["va"].sel(time=(fva_ssp585.time.dt.year>=2050)&(fva_ssp585.time.dt.year<=2099))

  fps_ssp585 = xr.open_dataset("/home/ys17-23/Extension2/LuoHL/EmerCon/ssp585/mergetime_remap/ps_{}_ssp585_remap.nc".format(mod))
  ps_ssp585 = fps_ssp585["ps"].sel(time=(fps_ssp585.time.dt.year>=2050)&(fps_ssp585.time.dt.year<=2099))

  fwap_ssp585 = xr.open_dataset("/home/ys17-23/Extension2/LuoHL/EmerCon/ssp585/mergetime_remap/wap_{}_ssp585_remap.nc".format(mod))
  wap_ssp585 = fwap_ssp585["wap"].sel(time=(fwap_ssp585.time.dt.year>=2050)&(fwap_ssp585.time.dt.year<=2099))

  fhfls_hist = xr.open_dataset("/home/ys17-23/Extension2/LuoHL/EmerCon/hist/mergetime_remap/hfls_{}_hist_remap.nc".format(mod))
  hfls_hist = fhfls_hist["hfls"].sel(time=(fhfls_hist.time.dt.year>=1965)&(fhfls_hist.time.dt.year<=2014))*0.408*1e-6
  hfls_hist.attrs["units"] = "mm/day"

  fpr_hist = xr.open_dataset("/home/ys17-23/Extension2/LuoHL/EmerCon/hist/mergetime_remap/pr_{}_hist_remap.nc".format(mod))
  pr_hist = fpr_hist["pr"].sel(time=(fpr_hist.time.dt.year>=1965)&(fpr_hist.time.dt.year<=2014))
  pr_hist.attrs["units"] = "mm/day"

  fhus_hist = xr.open_dataset("/home/ys17-23/Extension2/LuoHL/EmerCon/hist/mergetime_remap/hus_{}_hist_remap.nc".format(mod))
  hus_hist = fhus_hist["hus"].sel(time=(fhus_hist.time.dt.year>=1965)&(fhus_hist.time.dt.year<=2014))

  fua_hist = xr.open_dataset("/home/ys17-23/Extension2/LuoHL/EmerCon/hist/mergetime_remap/ua_{}_hist_remap.nc".format(mod))
  ua_hist = fua_hist["ua"].sel(time=(fua_hist.time.dt.year>=1965)&(fua_hist.time.dt.year<=2014))

  fva_hist = xr.open_dataset("/home/ys17-23/Extension2/LuoHL/EmerCon/hist/mergetime_remap/va_{}_hist_remap.nc".format(mod))
  va_hist = fva_hist["va"].sel(time=(fva_hist.time.dt.year>=1965)&(fva_hist.time.dt.year<=2014))

  fps_hist = xr.open_dataset("/home/ys17-23/Extension2/LuoHL/EmerCon/hist/mergetime_remap/ps_{}_hist_remap.nc".format(mod))
  ps_hist = fps_hist["ps"].sel(time=(fps_hist.time.dt.year>=1965)&(fps_hist.time.dt.year<=2014))

  fwap_hist = xr.open_dataset("/home/ys17-23/Extension2/LuoHL/EmerCon/hist/mergetime_remap/wap_{}_hist_remap.nc".format(mod))
  wap_hist = fwap_hist["wap"].sel(time=(fwap_hist.time.dt.year>=1965)&(fwap_hist.time.dt.year<=2014))

  ua_ssp585_JJA = ca.p_time(ua_ssp585, 6, 8)
  va_ssp585_JJA = ca.p_time(va_ssp585, 6, 8)
  wap_ssp585_JJA = ca.p_time(wap_ssp585, 6, 8)
  hus_ssp585_JJA = ca.p_time(hus_ssp585, 6, 8)

  hfls_ssp585_JJA = ca.p_time(hfls_ssp585, 6, 8)
  pr_ssp585_JJA = ca.p_time(pr_ssp585, 6, 8)
  ps_ssp585_JJA = ca.p_time(ps_ssp585, 6, 8)

  ua_hist_JJA = ca.p_time(ua_hist, 6, 8)
  va_hist_JJA = ca.p_time(va_hist, 6, 8)
  wap_hist_JJA = ca.p_time(wap_hist, 6, 8)
  hus_hist_JJA = ca.p_time(hus_hist, 6, 8)

  hfls_hist_JJA = ca.p_time(hfls_hist, 6, 8)
  pr_hist_JJA = ca.p_time(pr_hist, 6, 8)
  ps_hist_JJA = ca.p_time(ps_hist, 6, 8)

  pr_prime_JJA = pr_ssp585_JJA.mean(dim="time")-pr_hist_JJA.mean(dim="time")
  hfls_prime_JJA = hfls_ssp585_JJA.mean(dim="time")-hfls_hist_JJA.mean(dim="time")

  hus_prime_JJA = hus_ssp585_JJA.mean(dim="time")-hus_hist_JJA.mean(dim="time")
  ua_prime_JJA = ua_ssp585_JJA.mean(dim="time")-ua_hist_JJA.mean(dim="time")
  va_prime_JJA = va_ssp585_JJA.mean(dim="time")-va_hist_JJA.mean(dim="time")
  wap_prime_JJA = wap_ssp585_JJA.mean(dim="time")-wap_hist_JJA.mean(dim="time")

  ua_bar_JJA = ua_hist_JJA.mean(dim="time")
  va_bar_JJA = va_hist_JJA.mean(dim="time")
  hus_bar_JJA = hus_hist_JJA.mean(dim="time")
  wap_bar_JJA = wap_hist_JJA.mean(dim="time")

  ps_bar_JJA = ps_hist_JJA.mean(dim="time")

  husprime_dp, husprime_dy, husprime_dx = mpcalc.gradient(hus_prime_JJA)
  husprime_dp = husprime_dp.metpy.dequantify()
  husprime_dy = husprime_dy.metpy.dequantify()
  husprime_dx = husprime_dx.metpy.dequantify()

  # husprime_dx.loc[85000, 25, 60]

  husbar_dp, husbar_dy, husbar_dx = mpcalc.gradient(hus_bar_JJA)
  husbar_dp = husbar_dp.metpy.dequantify()
  husbar_dy = husbar_dy.metpy.dequantify()
  husbar_dx = husbar_dx.metpy.dequantify()

  ptop = 1
  g = 9.8
  cal_level = hus_hist_JJA.coords["plev"].where(hus_hist_JJA.coords["plev"] >= ptop)
  cal_level.attrs["units"] = "Pa"
  CMIPdp = geocat.comp.dpres_plevel(cal_level, ps_bar_JJA, ptop)
  CMIPdpg = CMIPdp / g
  CMIPdpg.attrs["units"] = "kg/m2"

  integration_top = 1


  def vert_intergrate(da_sl, CMIP6dpg=CMIPdpg, integration_top=integration_top, cal_level=cal_level):
    res_vert = (da_sl * CMIP6dpg[:, :, :].data).sum(dim="plev")
    return(res_vert)


  term1_1 = vert_intergrate(ua_bar_JJA.loc[:, :, :]*husprime_dx.loc[:, :, :].data)
  term1_2 = vert_intergrate(va_bar_JJA.loc[:, :, :]*husprime_dy.loc[:, :, :].data)
  term1 = term1_1+term1_2

  term2_1 = vert_intergrate(ua_prime_JJA.loc[:, :, :]*husbar_dx.loc[:, :, :].data)
  term2_2 = vert_intergrate(va_prime_JJA.loc[:, :, :]*husbar_dy.loc[:, :, :].data)
  term2 = term2_1+term2_2

  term3 = vert_intergrate(wap_bar_JJA.loc[:, :, :]*husprime_dp.loc[:, :, :].data)

  term4 = vert_intergrate(wap_prime_JJA.loc[:, :, :]*husbar_dp.loc[:, :, :].data)

  Res = pr_prime_JJA-hfls_prime_JJA-term1-term2-term3-term4

  def cal_area_weighted_mean(da):
      """计算数据的区域（纬度加权）平均

      Args:
          da (dataarray): 原数据

      Returns:
          dataarray: 区域平均结果
      """    
      weights = np.cos(np.deg2rad(da.latitude))
      weights.name = "weights"
      da_mean = da.weighted(weights).mean(("latitude"), skipna=True)
      da_mean = da_mean.mean(dim="longitude", skipna=True)
      return da_mean

  lat = hus_bar_JJA.coords["latitude"]
  lon = hus_bar_JJA.coords["longitude"]

  India_N = 24.0
  India_S = 10.0
  India_W = 60.0
  India_E = 105.0
  lat_India_range = lat[(lat >= India_S) & (lat <= India_N)]
  lon_India_range = lon[(lon >= India_W) & (lon <= India_E)]


  pr_prime_JJA_India_JJA = cal_area_weighted_mean(pr_prime_JJA.sel(latitude=lat_India_range, longitude=lon_India_range))
  hfls_prime_JJA_India_JJA = cal_area_weighted_mean(hfls_prime_JJA.sel(latitude=lat_India_range, longitude=lon_India_range))
  term1_India_JJA = cal_area_weighted_mean(term1.sel(latitude=lat_India_range, longitude=lon_India_range))
  term2_India_JJA = cal_area_weighted_mean(term2.sel(latitude=lat_India_range, longitude=lon_India_range))
  term3_India_JJA = cal_area_weighted_mean(term3.sel(latitude=lat_India_range, longitude=lon_India_range))
  term4_India_JJA = cal_area_weighted_mean(term4.sel(latitude=lat_India_range, longitude=lon_India_range))
  Res_India_JJA = cal_area_weighted_mean(Res.sel(latitude=lat_India_range, longitude=lon_India_range))

  res_models.append((pr_prime_JJA_India_JJA-hfls_prime_JJA_India_JJA-term1_India_JJA-term2_India_JJA-term3_India_JJA-term4_India_JJA)*86400)
  # term2_India_JJA

  pr_models.append(pr_prime_JJA_India_JJA.data*86400)
  hfls_models.append(hfls_prime_JJA_India_JJA.data*86400)
  term1_models.append(term1_India_JJA.data*86400)
  term2_models.append(term2_India_JJA.data*86400)
  term3_models.append(term3_India_JJA.data*86400)
  term4_models.append(term4_India_JJA.data*86400)


# %%
print(np.array(pr_models).mean(axis=0))
print(np.array(hfls_models).mean(axis=0))
print(np.array(term1_models).mean(axis=0))
print(np.array(term2_models).mean(axis=0))
print(np.array(term3_models).mean(axis=0))
print(np.array(term4_models).mean(axis=0))
res_models = np.array(pr_models)-np.array(hfls_models)+np.array(term1_models)+np.array(term2_models)+np.array(term3_models)+np.array(term4_models)
print(np.array(res_models).mean(axis=0))
# %%