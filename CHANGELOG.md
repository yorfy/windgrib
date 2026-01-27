# Changelog

## [1.0.1] - 2025-01-27

### Improved
- Enhanced subsetting capabilities with better indexing methods documentation
- Expanded documentation with multi-level data processing examples
- Added comprehensive examples for different indexing techniques (boolean, slice, array)
- Improved code examples with `if __name__ == '__main__':` guards to prevent multiprocessing issues

### Documentation
- Updated `usage_examples.md` with detailed subsetting examples for GFS and ECMWF models
- Added `custom_models.md` section on multi-level data processing with xarray techniques
- Improved comments and explanations in example scripts

## [1.0.0] - 2025-01-XX

### Added
- Initial release of WindGrib
- Support for GFS Wave and ECMWF IFS models
- Subset-based downloads using GRIB index files
- Automatic latest data retrieval
- Smart caching with incremental download completion
- Asyncio-based downloads and parallel GRIB decoding
- Comprehensive documentation and examples
