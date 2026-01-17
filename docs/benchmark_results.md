# WindGrib vs Herbie Benchmark

Comparison of WindGrib and Herbie performance for downloading and processing GFS Wave wind data (209 forecast steps).

## Test Configuration

- **Model**: GFS Wave (global.0p25)
- **Data**: Wind components (UGRD, VGRD)
- **Forecast**: 209 steps (418 GRIB messages)
- **Date**: 2026-01-16 18:00 UTC
- **Data Size**: ~226 MB

## Benchmark Results

### Detailed Phase Timings (seconds)

```
                        WindGrib (no cache)  WindGrib (cache)  Herbie (no cache)  Herbie (cache)
init_and_find_forecast                 3.87              2.30              25.55           24.05
download                              11.69              0.39              26.85            1.10
load                                  20.45              2.89              52.80           52.23
total                                 36.01              5.58             105.21           77.38
```

### Phase-by-Phase Analysis

- **Init And Find Forecast**: WindGrib **6.6x faster** (3.87s vs 25.55s)
- **Download**: WindGrib **2.3x faster** (11.69s vs 26.85s)
- **Load**: WindGrib **2.6x faster** (20.45s vs 52.80s)

### Cache Performance

- WindGrib cache speedup: **6.5x** (36.01s → 5.58s)
- Herbie cache speedup: **1.4x** (105.21s → 77.38s)

### Overall Comparison

**WindGrib is 2.9x faster overall** (36.01s vs 105.21s)

## Key Advantages

1. **Faster Initialization**: WindGrib's direct S3 index parsing is significantly faster than Herbie's approach
2. **Asyncio-Based Downloads**: Uses asyncio for concurrent downloads (vs FastHerbie's multi-threading approach), providing 2.3x faster download times
3. **Parallel GRIB Decoding**: Concurrent process-based GRIB message decoding (cfgrib doesn't support parallel reading on Windows), resulting in 2.6x faster loading
4. **Incremental NetCDF Caching**: Smart incremental saving to NetCDF format provides 6.5x speedup on subsequent runs (vs 1.4x for Herbie)

---
*Benchmark date: 2026-01-17*
