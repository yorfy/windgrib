# Test Data

This directory contains static GRIB files used for unit tests.

## Structure

```
20260116/
├── 12/
│   └── wind_ecmwf_ifs.oper.grib2
└── 18/
    └── wind_gfswave.global.0p25.grib2
```

## Available Files

- `20260116/18/wind_gfswave.global.0p25.grib2` - GFS Wave wind data (run 18Z, 10 steps)
- `20260116/12/wind_ecmwf_ifs.oper.grib2` - ECMWF IFS wind data (run 12Z, 10 steps)

## Usage

These files are used by unit tests to validate behavior without depending on dynamic data that could change or be deleted.