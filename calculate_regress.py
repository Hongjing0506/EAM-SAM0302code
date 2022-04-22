'''
Author: ChenHJ
Date: 2022-04-13 16:04:45
LastEditors: ChenHJ
LastEditTime: 2022-04-16 23:48:41
FilePath: /chenhj/0302code/calculate_regress.py
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
from importlib import reload
reload(sepl)

import pandas as pd
import metpy.calc as mpcalc
import metpy.constants as constants
import geocat.comp
from windspharm.xarray import VectorWind


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
from scipy.interpolate import interp2d
from eofs.multivariate.standard import MultivariateEof
from eofs.standard import Eof


def patches(ax, x0, y0, width, height, proj):
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x0, y0),
        width,
        height,
        fc="none",
        ec="grey7",
        linewidth=0.8,
        zorder=1.1,
        transform=proj,
        linestyle="--",
    )
    ax.add_patch(rect)
# %%
#   calculate the regression on detrend SAM of different models in historical run
fSAM_his = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_SAM_index_1950-2014.nc")
his_SAM_index_detrend = fSAM_his["SAM"]

fhgt_his = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/zg_historical_r144x72_195001-201412.nc")
hgthis_ver_JJA = fhgt_his["zg"]

fu_his = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/ua_historical_r144x72_195001-201412.nc")
uhis_ver_JJA = fu_his["ua"]

fv_his = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/va_historical_r144x72_195001-201412.nc")
vhis_ver_JJA = fv_his["va"]

hgthis_ver_JJA_3lev = hgthis_ver_JJA.sel(level=[200.0, 500.0, 850.0])
uhis_ver_JJA_3lev = uhis_ver_JJA.sel(level=[200.0, 500.0, 850.0])
vhis_ver_JJA_3lev = vhis_ver_JJA.sel(level=[200.0, 500.0, 850.0])
(
    hgt_his_SAM_slope,
    hgt_his_SAM_intercept,
    hgt_his_SAM_rvalue,
    hgt_his_SAM_pvalue,
    hgt_his_SAM_hypothesis,
) = ca.dim_linregress(his_SAM_index_detrend, hgthis_ver_JJA_3lev)

(
    u_his_SAM_slope,
    u_his_SAM_intercept,
    u_his_SAM_rvalue,
    u_his_SAM_pvalue,
    u_his_SAM_hypothesis,
) = ca.dim_linregress(his_SAM_index_detrend, uhis_ver_JJA_3lev)

(
    v_his_SAM_slope,
    v_his_SAM_intercept,
    v_his_SAM_rvalue,
    v_his_SAM_pvalue,
    v_his_SAM_hypothesis,
) = ca.dim_linregress(his_SAM_index_detrend, vhis_ver_JJA_3lev)

# %%
models = hgt_his_SAM_rvalue.coords["models"]
lon = hgt_his_SAM_rvalue.coords["lon"]
lat = hgt_his_SAM_rvalue.coords["lat"]
level = hgt_his_SAM_rvalue.coords["level"]

# %%
#   create the dataset of hgt/u/v regress onto SAM index in historical run
hgt_his_SAM_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], hgt_his_SAM_slope.data),
        intercept=(["models", "level", "lat", "lon"], hgt_his_SAM_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], hgt_his_SAM_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], hgt_his_SAM_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], hgt_his_SAM_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of multi-models in historical run regress onto his_SAM_index"),
)

u_his_SAM_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], u_his_SAM_slope.data),
        intercept=(["models", "level", "lat", "lon"], u_his_SAM_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], u_his_SAM_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], u_his_SAM_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], u_his_SAM_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of multi-models in historical run regress onto his_SAM_index"),
)

v_his_SAM_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], v_his_SAM_slope.data),
        intercept=(["models", "level", "lat", "lon"], v_his_SAM_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], v_his_SAM_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], v_his_SAM_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], v_his_SAM_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of multi-models in historical run regress onto his_SAM_index"),
)
# %%
#   output the regress result
hgt_his_SAM_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/hgt_his_SAM_regress.nc")
u_his_SAM_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/u_his_SAM_regress.nc")
v_his_SAM_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/v_his_SAM_regress.nc")
# %%
#   calculate the hgt/u/v of multi-models in historical run regress onto IWF index
fIWF_his = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_IWF_index_1950-2014.nc")
his_IWF_index_detrend = fIWF_his["IWF"]

(
    hgt_his_IWF_slope,
    hgt_his_IWF_intercept,
    hgt_his_IWF_rvalue,
    hgt_his_IWF_pvalue,
    hgt_his_IWF_hypothesis,
) = ca.dim_linregress(his_IWF_index_detrend, hgthis_ver_JJA_3lev)

(
    u_his_IWF_slope,
    u_his_IWF_intercept,
    u_his_IWF_rvalue,
    u_his_IWF_pvalue,
    u_his_IWF_hypothesis,
) = ca.dim_linregress(his_IWF_index_detrend, uhis_ver_JJA_3lev)

(
    v_his_IWF_slope,
    v_his_IWF_intercept,
    v_his_IWF_rvalue,
    v_his_IWF_pvalue,
    v_his_IWF_hypothesis,
) = ca.dim_linregress(his_IWF_index_detrend, vhis_ver_JJA_3lev)

# %%
models = hgt_his_IWF_rvalue.coords["models"]
lon = hgt_his_IWF_rvalue.coords["lon"]
lat = hgt_his_IWF_rvalue.coords["lat"]
level = hgt_his_IWF_rvalue.coords["level"]

# %%
#   create the dataset of hgt/u/v regress onto IWF index in historical run
hgt_his_IWF_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], hgt_his_IWF_slope.data),
        intercept=(["models", "level", "lat", "lon"], hgt_his_IWF_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], hgt_his_IWF_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], hgt_his_IWF_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], hgt_his_IWF_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of multi-models in historical run regress onto his_IWF_index"),
)

u_his_IWF_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], u_his_IWF_slope.data),
        intercept=(["models", "level", "lat", "lon"], u_his_IWF_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], u_his_IWF_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], u_his_IWF_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], u_his_IWF_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of multi-models in historical run regress onto his_IWF_index"),
)

v_his_IWF_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], v_his_IWF_slope.data),
        intercept=(["models", "level", "lat", "lon"], v_his_IWF_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], v_his_IWF_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], v_his_IWF_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], v_his_IWF_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of multi-models in historical run regress onto his_IWF_index"),
)
# %%
#   output the regress result
hgt_his_IWF_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/hgt_his_IWF_regress.nc")
u_his_IWF_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/u_his_IWF_regress.nc")
v_his_IWF_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/v_his_IWF_regress.nc")
# %%
#   calculate the uq and vq of multi-models in historical run regress onto SAM index
fuq_his = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_uq_dpg.nc")
uqhis_JJA = fuq_his["uq_dpg"]

fvq_his = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_vq_dpg.nc")
vqhis_JJA = fvq_his["vq_dpg"]

# %%
(
    uq_his_SAM_slope,
    uq_his_SAM_intercept,
    uq_his_SAM_rvalue,
    uq_his_SAM_pvalue,
    uq_his_SAM_hypothesis,
) = ca.dim_linregress(his_SAM_index_detrend, uqhis_JJA)

(
    vq_his_SAM_slope,
    vq_his_SAM_intercept,
    vq_his_SAM_rvalue,
    vq_his_SAM_pvalue,
    vq_his_SAM_hypothesis,
) = ca.dim_linregress(his_SAM_index_detrend, vqhis_JJA)


# %%
models = uq_his_SAM_rvalue.coords["models"]
lon = uq_his_SAM_rvalue.coords["lon"]
lat = uq_his_SAM_rvalue.coords["lat"]

# %%
#   create the dataset of uq/vq regress onto SAM index in historical run
uq_his_SAM_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "lat", "lon"], uq_his_SAM_slope.data),
        intercept=(["models", "lat", "lon"], uq_his_SAM_intercept.data),
        rvalue=(["models", "lat", "lon"], uq_his_SAM_rvalue.data),
        pvalue=(["models", "lat", "lon"], uq_his_SAM_pvalue.data),
        hypothesis=(["models", "lat", "lon"], uq_his_SAM_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="uq fields of multi-models in historical run regress onto his_SAM_index"),
)

vq_his_SAM_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "lat", "lon"], vq_his_SAM_slope.data),
        intercept=(["models", "lat", "lon"], vq_his_SAM_intercept.data),
        rvalue=(["models", "lat", "lon"], vq_his_SAM_rvalue.data),
        pvalue=(["models", "lat", "lon"], vq_his_SAM_pvalue.data),
        hypothesis=(["models", "lat", "lon"], vq_his_SAM_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="vq fields of multi-models in historical run regress onto his_SAM_index"),
)
# %%
#   output the regress result
uq_his_SAM_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/uq_his_SAM_regress.nc")

vq_his_SAM_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/vq_his_SAM_regress.nc")
# %%
#   calculate the uq and vq of multi-models in historical run regress onto IWF index
(
    uq_his_IWF_slope,
    uq_his_IWF_intercept,
    uq_his_IWF_rvalue,
    uq_his_IWF_pvalue,
    uq_his_IWF_hypothesis,
) = ca.dim_linregress(his_IWF_index_detrend, uqhis_JJA)

(
    vq_his_IWF_slope,
    vq_his_IWF_intercept,
    vq_his_IWF_rvalue,
    vq_his_IWF_pvalue,
    vq_his_IWF_hypothesis,
) = ca.dim_linregress(his_IWF_index_detrend, vqhis_JJA)


# %%
models = uq_his_IWF_rvalue.coords["models"]
lon = uq_his_IWF_rvalue.coords["lon"]
lat = uq_his_IWF_rvalue.coords["lat"]

# %%
#   create the dataset of uq/vq regress onto IWF index in historical run
uq_his_IWF_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "lat", "lon"], uq_his_IWF_slope.data),
        intercept=(["models", "lat", "lon"], uq_his_IWF_intercept.data),
        rvalue=(["models", "lat", "lon"], uq_his_IWF_rvalue.data),
        pvalue=(["models", "lat", "lon"], uq_his_IWF_pvalue.data),
        hypothesis=(["models", "lat", "lon"], uq_his_IWF_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="uq fields of multi-models in historical run regress onto his_IWF_index"),
)

vq_his_IWF_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "lat", "lon"], vq_his_IWF_slope.data),
        intercept=(["models", "lat", "lon"], vq_his_IWF_intercept.data),
        rvalue=(["models", "lat", "lon"], vq_his_IWF_rvalue.data),
        pvalue=(["models", "lat", "lon"], vq_his_IWF_pvalue.data),
        hypothesis=(["models", "lat", "lon"], vq_his_IWF_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="vq fields of multi-models in historical run regress onto his_IWF_index"),
)
# %%
#   output the regress result
uq_his_IWF_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/uq_his_IWF_regress.nc")

vq_his_IWF_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/vq_his_IWF_regress.nc")
# %%
# ===============================================
# %%
#   calculate the hgt/u/v of multi-models in ssp585 run regress onto SAM index
fSAM_ssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_SAM_index_2015-2099.nc")
ssp585_SAM_index_detrend = fSAM_ssp585["SAM"]

fhgt_ssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/zg_ssp585_r144x72_201501-209912.nc")
hgtssp585_ver_JJA = fhgt_ssp585["zg"]

fu_ssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ua_ssp585_r144x72_201501-209912.nc")
ussp585_ver_JJA = fu_ssp585["ua"]

fv_ssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/va_ssp585_r144x72_201501-209912.nc")
vssp585_ver_JJA = fv_ssp585["va"]

hgtssp585_ver_JJA_3lev = hgtssp585_ver_JJA.sel(level=[200.0, 500.0, 850.0])
ussp585_ver_JJA_3lev = ussp585_ver_JJA.sel(level=[200.0, 500.0, 850.0])
vssp585_ver_JJA_3lev = vssp585_ver_JJA.sel(level=[200.0, 500.0, 850.0])
(
    hgt_ssp585_SAM_slope,
    hgt_ssp585_SAM_intercept,
    hgt_ssp585_SAM_rvalue,
    hgt_ssp585_SAM_pvalue,
    hgt_ssp585_SAM_hypothesis,
) = ca.dim_linregress(ssp585_SAM_index_detrend, hgtssp585_ver_JJA_3lev)

(
    u_ssp585_SAM_slope,
    u_ssp585_SAM_intercept,
    u_ssp585_SAM_rvalue,
    u_ssp585_SAM_pvalue,
    u_ssp585_SAM_hypothesis,
) = ca.dim_linregress(ssp585_SAM_index_detrend, ussp585_ver_JJA_3lev)

(
    v_ssp585_SAM_slope,
    v_ssp585_SAM_intercept,
    v_ssp585_SAM_rvalue,
    v_ssp585_SAM_pvalue,
    v_ssp585_SAM_hypothesis,
) = ca.dim_linregress(ssp585_SAM_index_detrend, vssp585_ver_JJA_3lev)

# %%
models = hgt_ssp585_SAM_rvalue.coords["models"]
lon = hgt_ssp585_SAM_rvalue.coords["lon"]
lat = hgt_ssp585_SAM_rvalue.coords["lat"]
level = hgt_ssp585_SAM_rvalue.coords["level"]

# %%
#   create the dataset of hgt/u/v regress onto SAM index in ssp585 run
hgt_ssp585_SAM_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], hgt_ssp585_SAM_slope.data),
        intercept=(["models", "level", "lat", "lon"], hgt_ssp585_SAM_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], hgt_ssp585_SAM_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], hgt_ssp585_SAM_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], hgt_ssp585_SAM_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of multi-models in ssp585 run regress onto ssp585_SAM_index"),
)

u_ssp585_SAM_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], u_ssp585_SAM_slope.data),
        intercept=(["models", "level", "lat", "lon"], u_ssp585_SAM_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], u_ssp585_SAM_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], u_ssp585_SAM_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], u_ssp585_SAM_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of multi-models in ssp585 run regress onto ssp585_SAM_index"),
)

v_ssp585_SAM_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], v_ssp585_SAM_slope.data),
        intercept=(["models", "level", "lat", "lon"], v_ssp585_SAM_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], v_ssp585_SAM_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], v_ssp585_SAM_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], v_ssp585_SAM_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of multi-models in ssp585 run regress onto ssp585_SAM_index"),
)
# %%
#   output the regress result
hgt_ssp585_SAM_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/hgt_ssp585_SAM_regress.nc")
u_ssp585_SAM_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/u_ssp585_SAM_regress.nc")
v_ssp585_SAM_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/v_ssp585_SAM_regress.nc")
# %%
#   calculate the hgt/u/v of multi-models in ssp585 run regress onto IWF index
fIWF_ssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_IWF_index_2015-2099.nc")
ssp585_IWF_index_detrend = fIWF_ssp585["IWF"]

(
    hgt_ssp585_IWF_slope,
    hgt_ssp585_IWF_intercept,
    hgt_ssp585_IWF_rvalue,
    hgt_ssp585_IWF_pvalue,
    hgt_ssp585_IWF_hypothesis,
) = ca.dim_linregress(ssp585_IWF_index_detrend, hgtssp585_ver_JJA_3lev)

(
    u_ssp585_IWF_slope,
    u_ssp585_IWF_intercept,
    u_ssp585_IWF_rvalue,
    u_ssp585_IWF_pvalue,
    u_ssp585_IWF_hypothesis,
) = ca.dim_linregress(ssp585_IWF_index_detrend, ussp585_ver_JJA_3lev)

(
    v_ssp585_IWF_slope,
    v_ssp585_IWF_intercept,
    v_ssp585_IWF_rvalue,
    v_ssp585_IWF_pvalue,
    v_ssp585_IWF_hypothesis,
) = ca.dim_linregress(ssp585_IWF_index_detrend, vssp585_ver_JJA_3lev)

# %%
models = hgt_ssp585_IWF_rvalue.coords["models"]
lon = hgt_ssp585_IWF_rvalue.coords["lon"]
lat = hgt_ssp585_IWF_rvalue.coords["lat"]
level = hgt_ssp585_IWF_rvalue.coords["level"]

# %%
#   create the dataset of hgt/u/v regress onto IWF index in ssp585 run
hgt_ssp585_IWF_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], hgt_ssp585_IWF_slope.data),
        intercept=(["models", "level", "lat", "lon"], hgt_ssp585_IWF_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], hgt_ssp585_IWF_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], hgt_ssp585_IWF_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], hgt_ssp585_IWF_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of multi-models in ssp585 run regress onto ssp585_IWF_index"),
)

u_ssp585_IWF_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], u_ssp585_IWF_slope.data),
        intercept=(["models", "level", "lat", "lon"], u_ssp585_IWF_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], u_ssp585_IWF_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], u_ssp585_IWF_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], u_ssp585_IWF_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of multi-models in ssp585 run regress onto ssp585_IWF_index"),
)

v_ssp585_IWF_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], v_ssp585_IWF_slope.data),
        intercept=(["models", "level", "lat", "lon"], v_ssp585_IWF_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], v_ssp585_IWF_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], v_ssp585_IWF_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], v_ssp585_IWF_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of multi-models in ssp585 run regress onto ssp585_IWF_index"),
)
# %%
#   output the regress result
hgt_ssp585_IWF_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/hgt_ssp585_IWF_regress.nc")
u_ssp585_IWF_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/u_ssp585_IWF_regress.nc")
v_ssp585_IWF_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/v_ssp585_IWF_regress.nc")
# %%
#   calculate the uq and vq of multi-models in ssp585 run regress onto SAM index
fuq_ssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_uq_dpg.nc")
uqssp585_JJA = fuq_ssp585["uq_dpg"]

fvq_ssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_vq_dpg.nc")
vqssp585_JJA = fvq_ssp585["vq_dpg"]

# %%
(
    uq_ssp585_SAM_slope,
    uq_ssp585_SAM_intercept,
    uq_ssp585_SAM_rvalue,
    uq_ssp585_SAM_pvalue,
    uq_ssp585_SAM_hypothesis,
) = ca.dim_linregress(ssp585_SAM_index_detrend, uqssp585_JJA)

(
    vq_ssp585_SAM_slope,
    vq_ssp585_SAM_intercept,
    vq_ssp585_SAM_rvalue,
    vq_ssp585_SAM_pvalue,
    vq_ssp585_SAM_hypothesis,
) = ca.dim_linregress(ssp585_SAM_index_detrend, vqssp585_JJA)


# %%
models = uq_ssp585_SAM_rvalue.coords["models"]
lon = uq_ssp585_SAM_rvalue.coords["lon"]
lat = uq_ssp585_SAM_rvalue.coords["lat"]

# %%
#   create the dataset of uq/vq regress onto SAM index in ssp585 run
uq_ssp585_SAM_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "lat", "lon"], uq_ssp585_SAM_slope.data),
        intercept=(["models", "lat", "lon"], uq_ssp585_SAM_intercept.data),
        rvalue=(["models", "lat", "lon"], uq_ssp585_SAM_rvalue.data),
        pvalue=(["models", "lat", "lon"], uq_ssp585_SAM_pvalue.data),
        hypothesis=(["models", "lat", "lon"], uq_ssp585_SAM_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="uq fields of multi-models in ssp585 run regress onto ssp585_SAM_index"),
)

vq_ssp585_SAM_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "lat", "lon"], vq_ssp585_SAM_slope.data),
        intercept=(["models", "lat", "lon"], vq_ssp585_SAM_intercept.data),
        rvalue=(["models", "lat", "lon"], vq_ssp585_SAM_rvalue.data),
        pvalue=(["models", "lat", "lon"], vq_ssp585_SAM_pvalue.data),
        hypothesis=(["models", "lat", "lon"], vq_ssp585_SAM_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="vq fields of multi-models in ssp585 run regress onto ssp585_SAM_index"),
)
# %%
#   output the regress result
uq_ssp585_SAM_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/uq_ssp585_SAM_regress.nc")

vq_ssp585_SAM_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/vq_ssp585_SAM_regress.nc")
# %%
#   calculate the uq and vq of multi-models in ssp585 run regress onto IWF index
(
    uq_ssp585_IWF_slope,
    uq_ssp585_IWF_intercept,
    uq_ssp585_IWF_rvalue,
    uq_ssp585_IWF_pvalue,
    uq_ssp585_IWF_hypothesis,
) = ca.dim_linregress(ssp585_IWF_index_detrend, uqssp585_JJA)

(
    vq_ssp585_IWF_slope,
    vq_ssp585_IWF_intercept,
    vq_ssp585_IWF_rvalue,
    vq_ssp585_IWF_pvalue,
    vq_ssp585_IWF_hypothesis,
) = ca.dim_linregress(ssp585_IWF_index_detrend, vqssp585_JJA)


# %%
models = uq_ssp585_IWF_rvalue.coords["models"]
lon = uq_ssp585_IWF_rvalue.coords["lon"]
lat = uq_ssp585_IWF_rvalue.coords["lat"]

# %%
#   create the dataset of uq/vq regress onto IWF index in ssp585 run
uq_ssp585_IWF_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "lat", "lon"], uq_ssp585_IWF_slope.data),
        intercept=(["models", "lat", "lon"], uq_ssp585_IWF_intercept.data),
        rvalue=(["models", "lat", "lon"], uq_ssp585_IWF_rvalue.data),
        pvalue=(["models", "lat", "lon"], uq_ssp585_IWF_pvalue.data),
        hypothesis=(["models", "lat", "lon"], uq_ssp585_IWF_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="uq fields of multi-models in ssp585 run regress onto ssp585_IWF_index"),
)

vq_ssp585_IWF_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "lat", "lon"], vq_ssp585_IWF_slope.data),
        intercept=(["models", "lat", "lon"], vq_ssp585_IWF_intercept.data),
        rvalue=(["models", "lat", "lon"], vq_ssp585_IWF_rvalue.data),
        pvalue=(["models", "lat", "lon"], vq_ssp585_IWF_pvalue.data),
        hypothesis=(["models", "lat", "lon"], vq_ssp585_IWF_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="vq fields of multi-models in ssp585 run regress onto ssp585_IWF_index"),
)
# %%
#   output the regress result
uq_ssp585_IWF_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/uq_ssp585_IWF_regress.nc")

vq_ssp585_IWF_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/vq_ssp585_IWF_regress.nc")
# %%
#   calculate the divergence of uq and vq
fuq_his = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_uq_dpg.nc")
uqhis_ver_JJA = fuq_his["uq_dpg"]

fvq_his = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_vq_dpg.nc")
vqhis_ver_JJA = fvq_his["vq_dpg"]

divuqvqhis_JJA = ca.cal_divergence(uqhis_ver_JJA, vqhis_ver_JJA)
divuqvqhis_JJA.name = "div_uqvq"
divuqvqhis_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_div_uqvq.nc")

divuqvqhis_JJA_detrend = ca.detrend_dim(divuqvqhis_JJA, "time", deg=1, demean=False)
divuqvqhis_JJA_detrend.name = "div_uqvq"
divuqvqhis_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_div_uqvq.nc")

# %%
fuq_ssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_uq_dpg.nc")
uqssp585_ver_JJA = fuq_ssp585["uq_dpg"]

fvq_ssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_vq_dpg.nc")
vqssp585_ver_JJA = fvq_ssp585["vq_dpg"]

divuqvqssp585_JJA = ca.cal_divergence(uqssp585_ver_JJA, vqssp585_ver_JJA)
divuqvqssp585_JJA.name = "div_uqvq"
divuqvqssp585_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_div_uqvq.nc")

divuqvqssp585_JJA_detrend = ca.detrend_dim(divuqvqssp585_JJA, "time", deg=1, demean=False)
divuqvqssp585_JJA_detrend.name = "div_uqvq"
divuqvqssp585_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_div_uqvq.nc")

# %%
#   calculate the linear trends of different variables in ssp585 run

fhgt_ssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/zg_ssp585_r144x72_201501-209912.nc")
hgtssp585_ver_JJA = fhgt_ssp585["zg"]

fu_ssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ua_ssp585_r144x72_201501-209912.nc")
ussp585_ver_JJA = fu_ssp585["ua"]

fv_ssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/va_ssp585_r144x72_201501-209912.nc")
vssp585_ver_JJA = fv_ssp585["va"]

fuq_ssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_uq_dpg.nc")
uqssp585_ver_JJA = fuq_ssp585["uq_dpg"]

fvq_ssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_vq_dpg.nc")
vqssp585_ver_JJA = fvq_ssp585["vq_dpg"]

fpr_ssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/pr_ssp585_r144x72_201501-209912.nc")
prssp585_ver_JJA = fpr_ssp585["pr"]

fdivuqvq_ssp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_div_uqvq.nc")
divuqvqssp585_ver_JJA = fdivuqvq_ssp585["div_uqvq"]

# %%
hgtssp585_ver_JJA_3lev = hgtssp585_ver_JJA.sel(level=[200.0, 500.0, 850.0])
ussp585_ver_JJA_3lev = ussp585_ver_JJA.sel(level=[200.0, 500.0, 850.0])
vssp585_ver_JJA_3lev = vssp585_ver_JJA.sel(level=[200.0, 500.0, 850.0])

tmp_time = np.arange(len(hgtssp585_ver_JJA_3lev.coords["time"]))
hgtssp585_ver_JJA_3lev.coords["time"] = tmp_time
ussp585_ver_JJA_3lev.coords["time"] = tmp_time
vssp585_ver_JJA_3lev.coords["time"] = tmp_time
divuqvqssp585_ver_JJA.coords["time"] = tmp_time
uqssp585_ver_JJA.coords["time"] = tmp_time
vqssp585_ver_JJA.coords["time"] = tmp_time
prssp585_ver_JJA.coords["time"] = tmp_time

(
    hgt_ssp585_time_slope,
    hgt_ssp585_time_intercept,
    hgt_ssp585_time_rvalue,
    hgt_ssp585_time_pvalue,
    hgt_ssp585_time_hypothesis,
) = ca.dim_linregress(hgtssp585_ver_JJA_3lev.coords["time"], hgtssp585_ver_JJA_3lev)

(
    u_ssp585_time_slope,
    u_ssp585_time_intercept,
    u_ssp585_time_rvalue,
    u_ssp585_time_pvalue,
    u_ssp585_time_hypothesis,
) = ca.dim_linregress(ussp585_ver_JJA_3lev.coords["time"], ussp585_ver_JJA_3lev)

(
    v_ssp585_time_slope,
    v_ssp585_time_intercept,
    v_ssp585_time_rvalue,
    v_ssp585_time_pvalue,
    v_ssp585_time_hypothesis,
) = ca.dim_linregress(vssp585_ver_JJA_3lev.coords["time"], vssp585_ver_JJA_3lev)

(
    divuqvq_ssp585_time_slope,
    divuqvq_ssp585_time_intercept,
    divuqvq_ssp585_time_rvalue,
    divuqvq_ssp585_time_pvalue,
    divuqvq_ssp585_time_hypothesis,
) = ca.dim_linregress(divuqvqssp585_ver_JJA.coords["time"], divuqvqssp585_ver_JJA)

(
    uq_ssp585_time_slope,
    uq_ssp585_time_intercept,
    uq_ssp585_time_rvalue,
    uq_ssp585_time_pvalue,
    uq_ssp585_time_hypothesis,
) = ca.dim_linregress(uqssp585_ver_JJA.coords["time"], uqssp585_ver_JJA)

(
    vq_ssp585_time_slope,
    vq_ssp585_time_intercept,
    vq_ssp585_time_rvalue,
    vq_ssp585_time_pvalue,
    vq_ssp585_time_hypothesis,
) = ca.dim_linregress(vqssp585_ver_JJA.coords["time"], vqssp585_ver_JJA)

(
    pr_ssp585_time_slope,
    pr_ssp585_time_intercept,
    pr_ssp585_time_rvalue,
    pr_ssp585_time_pvalue,
    pr_ssp585_time_hypothesis,
) = ca.dim_linregress(prssp585_ver_JJA.coords["time"], prssp585_ver_JJA)
# %%
#   generate the dataset and output the linear trend result of ssp585 run
models = hgt_ssp585_time_rvalue.coords["models"]
level = hgt_ssp585_time_rvalue.coords["level"]
lon = hgt_ssp585_time_rvalue.coords["lon"]
lat = hgt_ssp585_time_rvalue.coords["lat"]

hgt_ssp585_time_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], hgt_ssp585_time_slope.data),
        intercept=(["models", "level", "lat", "lon"], hgt_ssp585_time_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], hgt_ssp585_time_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], hgt_ssp585_time_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], hgt_ssp585_time_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt linear trends of multi-models in ssp585 run"),
)
hgt_ssp585_time_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/hgt_ssp585_time_regress.nc")

u_ssp585_time_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], u_ssp585_time_slope.data),
        intercept=(["models", "level", "lat", "lon"], u_ssp585_time_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], u_ssp585_time_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], u_ssp585_time_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], u_ssp585_time_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u linear trends of multi-models in ssp585 run"),
)
u_ssp585_time_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/u_ssp585_time_regress.nc")

v_ssp585_time_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], v_ssp585_time_slope.data),
        intercept=(["models", "level", "lat", "lon"], v_ssp585_time_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], v_ssp585_time_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], v_ssp585_time_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], v_ssp585_time_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v linear trends of multi-models in ssp585 run"),
)
v_ssp585_time_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/v_ssp585_time_regress.nc")

divuqvq_ssp585_time_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "lat", "lon"], divuqvq_ssp585_time_slope.data),
        intercept=(["models", "lat", "lon"], divuqvq_ssp585_time_intercept.data),
        rvalue=(["models", "lat", "lon"], divuqvq_ssp585_time_rvalue.data),
        pvalue=(["models", "lat", "lon"], divuqvq_ssp585_time_pvalue.data),
        hypothesis=(["models", "lat", "lon"], divuqvq_ssp585_time_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="divuqvq linear trends of multi-models in ssp585 run"),
)
divuqvq_ssp585_time_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/divuqvq_ssp585_time_regress.nc")

uq_ssp585_time_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "lat", "lon"], uq_ssp585_time_slope.data),
        intercept=(["models", "lat", "lon"], uq_ssp585_time_intercept.data),
        rvalue=(["models", "lat", "lon"], uq_ssp585_time_rvalue.data),
        pvalue=(["models", "lat", "lon"], uq_ssp585_time_pvalue.data),
        hypothesis=(["models", "lat", "lon"], uq_ssp585_time_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="uq linear trends of multi-models in ssp585 run"),
)
uq_ssp585_time_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/uq_ssp585_time_regress.nc")

vq_ssp585_time_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "lat", "lon"], vq_ssp585_time_slope.data),
        intercept=(["models", "lat", "lon"], vq_ssp585_time_intercept.data),
        rvalue=(["models", "lat", "lon"], vq_ssp585_time_rvalue.data),
        pvalue=(["models", "lat", "lon"], vq_ssp585_time_pvalue.data),
        hypothesis=(["models", "lat", "lon"], vq_ssp585_time_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="vq linear trends of multi-models in ssp585 run"),
)
vq_ssp585_time_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/vq_ssp585_time_regress.nc")

pr_ssp585_time_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "lat", "lon"], pr_ssp585_time_slope.data),
        intercept=(["models", "lat", "lon"], pr_ssp585_time_intercept.data),
        rvalue=(["models", "lat", "lon"], pr_ssp585_time_rvalue.data),
        pvalue=(["models", "lat", "lon"], pr_ssp585_time_pvalue.data),
        hypothesis=(["models", "lat", "lon"], pr_ssp585_time_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="pr linear trends of multi-models in ssp585 run"),
)
pr_ssp585_time_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/pr_ssp585_time_regress.nc")
# %%
