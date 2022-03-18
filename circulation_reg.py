'''
Author: ChenHJ
Date: 2022-03-16 17:42:02
LastEditors: ChenHJ
LastEditTime: 2022-03-18 21:08:20
FilePath: /chenhj/0302code/circulation_reg.py
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

import metpy.calc as mpcalc
import geocat.comp

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

fvERA5 = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/obs/vwind_mon_r144x72_195001-201412.nc")
vERA5 = fvERA5["v"]

fvhis = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/va/va_Amon_ensemble_historical_gn_195001-201412.nc")
vhis = fvhis["va"]

fuERA5 = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/obs/uwind_mon_r144x72_195001-201412.nc")
uERA5 = fuERA5["u"]

fuhis = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/ua/ua_Amon_ensemble_historical_gn_195001-201412.nc")
uhis = fuhis["ua"]

fspERA5 = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/obs/sp_mon_r144x72_195001-201412.nc")
spERA5 = fspERA5["sp"]

fsphis = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/ps/ps_Amon_ensemble_historical_gn_195001-201412.nc")
sphis = fsphis["ps"]

fqERA5 = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/obs/q_mon_r144x72_195001-201412.nc")
qERA5 = fqERA5["q"]

fqhis = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/hus/hus_Amon_ensemble_historical_gn_195001-201412.nc")
qhis = fqhis["hus"]
# %%
#   calculate the meridional water vapor transport
#   select the level
uERA5_ver_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, :, :]
uERA5_ver_India_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, 8:28, 70:86]
uERA5_ver_EA_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, 36:42, 108:118]

vERA5_ver_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, :, :]
vERA5_ver_India_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, 8:28, 70:86]
vERA5_ver_EA_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, 36:42, 108:118]

qERA5_ver_JJA = ca.p_time(qERA5, 6, 8, True).loc[:, 100.0:, :, :]
qERA5_ver_India_JJA = ca.p_time(qERA5, 6, 8, True).loc[:, 100.0:, 8:28, 70:86]
qERA5_ver_EA_JJA = ca.p_time(qERA5, 6, 8, True).loc[:, 100.0:, 36:42, 108:118]

spERA5_ver_JJA = ca.p_time(spERA5, 6, 8, True).loc[:, :, :]
spERA5_ver_India_JJA = ca.p_time(spERA5, 6, 8, True).loc[:, 8:28, 70:86]
spERA5_ver_EA_JJA = ca.p_time(spERA5, 6, 8, True).loc[:, 36:42, 108:118]

uhis_ver_JJA = ca.p_time(uhis, 6, 8, True).loc[:, :10000.0, :, :]
uhis_ver_India_JJA = ca.p_time(uhis, 6, 8, True).loc[:, :10000.0, 8:28, 70:86]
uhis_ver_EA_JJA = ca.p_time(uhis, 6, 8, True).loc[:, :10000.0, 36:42, 108:118]

vhis_ver_JJA = ca.p_time(vhis, 6, 8, True).loc[:, :10000.0, :, :]
vhis_ver_India_JJA = ca.p_time(vhis, 6, 8, True).loc[:, :10000.0, 8:28, 70:86]
vhis_ver_EA_JJA = ca.p_time(vhis, 6, 8, True).loc[:, :10000.0, 36:42, 108:118]

qhis_ver_JJA = ca.p_time(qhis, 6, 8, True).loc[:, :10000.0, :, :]
qhis_ver_India_JJA = ca.p_time(qhis, 6, 8, True).loc[:, :10000.0, 8:28, 70:86]
qhis_ver_EA_JJA = ca.p_time(qhis, 6, 8, True).loc[:, :10000.0, 36:42, 108:118]

sphis_ver_JJA = ca.p_time(sphis, 6, 8, True).loc[:, :, :]
sphis_ver_India_JJA = ca.p_time(sphis, 6, 8, True).loc[:, 8:28, 70:86]
sphis_ver_EA_JJA = ca.p_time(sphis, 6, 8, True).loc[:, 36:42, 108:118]

preCRU_JJA = ca.p_time(preCRU, 6, 8, True)
preCRU_India_JJA = ca.p_time(preCRU.loc[:, 8:28, 70:86], 6, 8, True)
preCRU_EA_JJA = ca.p_time(preCRU.loc[:, 36:42, 108:118], 6, 8, True)

prehis_JJA = ca.p_time(prehis, 6, 8, True)
prehis_India_JJA = ca.p_time(prehis.loc[:, 8:28, 70:86], 6, 8, True)
prehis_EA_JJA = ca.p_time(prehis.loc[:, 36:42, 108:118], 6, 8, True)
# %%
#   calculate the area mean
uERA5_ver_India_JJA_mean = ca.cal_lat_weighted_mean(uERA5_ver_India_JJA).mean(dim="lon", skipna=True)
uERA5_ver_EA_JJA_mean = ca.cal_lat_weighted_mean(uERA5_ver_India_JJA).mean(dim="lon", skipna=True)

vERA5_ver_India_JJA_mean = ca.cal_lat_weighted_mean(vERA5_ver_India_JJA).mean(dim="lon", skipna=True)
vERA5_ver_EA_JJA_mean = ca.cal_lat_weighted_mean(vERA5_ver_India_JJA).mean(dim="lon", skipna=True)

qERA5_ver_India_JJA_mean = ca.cal_lat_weighted_mean(qERA5_ver_India_JJA).mean(dim="lon", skipna=True)
qERA5_ver_EA_JJA_mean = ca.cal_lat_weighted_mean(qERA5_ver_India_JJA).mean(dim="lon", skipna=True)

spERA5_ver_India_JJA_mean = ca.cal_lat_weighted_mean(spERA5_ver_India_JJA).mean(dim="lon", skipna=True)
spERA5_ver_EA_JJA_mean = ca.cal_lat_weighted_mean(spERA5_ver_India_JJA).mean(dim="lon", skipna=True)

uhis_ver_India_JJA_mean = ca.cal_lat_weighted_mean(uhis_ver_India_JJA).mean(dim="lon", skipna=True)
uhis_ver_EA_JJA_mean = ca.cal_lat_weighted_mean(uhis_ver_India_JJA).mean(dim="lon", skipna=True)

vhis_ver_India_JJA_mean = ca.cal_lat_weighted_mean(vhis_ver_India_JJA).mean(dim="lon", skipna=True)
vhis_ver_EA_JJA_mean = ca.cal_lat_weighted_mean(vhis_ver_India_JJA).mean(dim="lon", skipna=True)

qhis_ver_India_JJA_mean = ca.cal_lat_weighted_mean(qhis_ver_India_JJA).mean(dim="lon", skipna=True)
qhis_ver_EA_JJA_mean = ca.cal_lat_weighted_mean(qhis_ver_India_JJA).mean(dim="lon", skipna=True)

sphis_ver_India_JJA_mean = ca.cal_lat_weighted_mean(sphis_ver_India_JJA).mean(dim="lon", skipna=True)
sphis_ver_EA_JJA_mean = ca.cal_lat_weighted_mean(sphis_ver_India_JJA).mean(dim="lon", skipna=True)

# %%
#   calculate the waver vapor vertical intergration
ptop = 100*100
g = 9.8
ERA5level = qERA5_ver_JJA.coords["level"] * 100.0
ERA5level.attrs["units"] = "Pa"
ERA5dp = geocat.comp.dpres_plevel(ERA5level, spERA5_ver_JJA, ptop)
ERA5dpg = dp/g
ERA5dpg.attrs["units"] = "kg/m2"
# calculate the water vapor transport
uq_ERA5 = uERA5_ver_JJA * qERA5_ver_JJA * 1000.0
vq_ERA5 = vERA5_ver_JJA * qERA5_ver_JJA * 1000.0
uq_ERA5.attrs["units"] = "[m/s][g/kg]"
vq_ERA5.attrs["units"] = "[m/s][g/kg]"
# calculate the whole levels water vapor transport
uq_dpg_ERA5 = (uq_ERA5 * ERA5dpg.data).sum(dim="level")
vq_dpg_ERA5 = (vq_ERA5 * ERA5dpg.data).sum(dim="level")
uq_dpg_ERA5.attrs["units"] = "[m/s][g/kg]"
vq_dpg_ERA5.attrs["units"] = "[m/s][g/kg]"


hislevel = qhis_ver_JJA.coords["plev"] * 100.0
hislevel.attrs["units"] = "Pa"
hisdp = geocat.comp.dpres_plevel(hislevel, sphis_ver_JJA, ptop)
hisdpg = dp/g
hisdpg.attrs["units"] = "kg/m2"
# calculate the water vapor transport
uq_his = uhis_ver_JJA * qhis_ver_JJA * 1000.0
vq_his = vhis_ver_JJA * qhis_ver_JJA * 1000.0
uq_his.attrs["units"] = "[m/s][g/kg]"
vq_his.attrs["units"] = "[m/s][g/kg]"
# calculate the whole levels water vapor transport
uq_dpg_his = (uq_his * hisdpg.data).sum(dim="plev")
vq_dpg_his = (vq_his * hisdpg.data).sum(dim="plev")
uq_dpg_his.attrs["units"] = "[m/s][g/kg]"
vq_dpg_his.attrs["units"] = "[m/s][g/kg]"
# %%
#   calculate the correlation of India precipitation and meridional water vapor transport
