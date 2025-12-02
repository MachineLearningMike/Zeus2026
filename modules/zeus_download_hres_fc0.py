#!/usr/bin/env python3
"""
download_hres_fc0.py

Download HRES fc0 (analysis) data for selected variables using ecmwf‑opendata, save GRIB,
and (optionally) convert to NetCDF / pass to preprocessing.

Usage:
    python download_hres_fc0.py [output_filename] [date (YYYY‑MM‑DD)] [time (0/6/12/18)]
If date/time omitted: fetch latest available.
"""

import os
from ecmwf.opendata import Client

def download_fc0(target_file="hres_fc0.grib2", date=None, time=None,
                 params=None, levtype="pl", levels=None):
    """
    Download HRES fc0 data via ECMWF open data API.
    params: list of parameter short‑names (e.g. ["2t","msl","u","v","z","q"])
    levtype: "pl" (pressure levels) or "sfc" (surface)
    levels: list of pressure levels if levtype == "pl"
    """
    client = Client(source="ecmwf")
    request = {
        "type": "fc",
        "step": 0,       # fc0
    }
    if date:
        request["date"] = date
    if time is not None:
        request["time"] = time
    if params:
        request["param"] = params
    if levtype:
        request["levtype"] = levtype
    if levels is not None:
        request["levelist"] = levels

    client.retrieve(request, target_file)
    print("Downloaded:", target_file)

if __name__ == "__main__":
    import sys
    # default parameters: a few broad vars, you can expand
    default_params = ["2t","msl","u","v","z","q","10u","10v","tp","ssr","ssrd","tcc","lcc","mcc","hcc"]
    # for pressure-level fields
    pressure_levels = [50,100,150,200,250,300,400,500,600,700,850,925,1000]

    out_file = sys.argv[1] if len(sys.argv) > 1 else "hres_fc0.grib2"
    date = sys.argv[2] if len(sys.argv) > 2 else None
    time = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # Download surface + pressure‑level data
    download_fc0(target_file=out_file,
                 date=date,
                 time=time,
                 params=default_params,
                 levtype="pl",
                 levels=pressure_levels)

    download_fc0(target_file=out_file.replace(".grib2", "_sfc.grib2"),
                 date=date,
                 time=time,
                 params=default_params,
                 levtype="sfc",
                 levels=None)
