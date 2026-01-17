"""Test comparing grib_to_xarray vs grib_bytes_to_xarray engines"""
import sys
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from windgrib.grib_to_dataset import grib_to_dataset

# Add windgrib to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGribEngine(unittest.TestCase):
    """Test comparison between grid_to_dataset and cfgrib engine of xarray open_dataset"""

    @classmethod
    def setUpClass(cls):
        """Set up test data - load GRIB files and datasets once"""
        test_data_path = Path(__file__).parent / "data"

        cls.grib_files = [
            ("GFS", test_data_path / "20260116/18/wind_gfswave.global.0p25.grib2"),
            ("ECMWF", test_data_path / "20260116/12/wind_ecmwf_ifs.oper.grib2"),
        ]

        # Check if files exist
        missing_files = [name for name, file in cls.grib_files if not file.exists()]
        if missing_files:
            raise unittest.SkipTest(f"Missing test files: {missing_files}")

        print(f"Using test files: {[f'{name}: {file.name}' for name, file in cls.grib_files]}")

        # Load all GRIB data and datasets once
        cls.test_data = {}
        for model_name, grib_file in cls.grib_files:
            # Load GRIB bytes
            with open(grib_file, 'rb') as f:
                grib_bytes = f.read()

            # Generate datasets
            ds_custom = grib_to_dataset(grib_bytes)
            ds_cfgrib = xr.open_dataset(grib_file, engine='cfgrib')

            cls.test_data[model_name] = {
                'grib_bytes': grib_bytes,
                'grib_file': grib_file,
                'ds_custom': ds_custom,
                'ds_cfgrib': ds_cfgrib,
            }

    def test_cfgrib_comparison(self):
        """Compare outputs of grib_to_dataset with cfgrib"""
        if not self.test_data:
            self.skipTest("No GRIB data available")

        for model_name, data in self.test_data.items():
            with self.subTest(model=model_name):
                print(f"\n=== Testing {model_name} file: {data['grib_file'].name} ===")

                ds = data['ds_custom']
                ds_cfgrib = data['ds_cfgrib']

                # Basic structure comparison
                self.assertEqual(set(ds.data_vars), set(ds_cfgrib.data_vars),
                                 f"{model_name}: Variables should be the same")

                # Compare each variable
                for var in ds.data_vars:
                    self._compare_variable(model_name, var, ds, ds_cfgrib)

    def _compare_variable(self, model_name, var, ds, ds_cfgrib):
        """Helper method to compare a single variable between datasets."""
        if var not in ds_cfgrib.data_vars:
            return

        custom_data = ds[var].values
        cfgrib_data = ds_cfgrib[var].values

        # Check shapes
        self.assertEqual(custom_data.shape, cfgrib_data.shape,
                         f"{model_name} {var}: Shape mismatch")

        # Compare NaN handling
        custom_nans = np.isnan(custom_data).sum()
        cfgrib_nans = np.isnan(cfgrib_data).sum()

        print(f"  {var} comparison:")
        print(f"    Custom engine - NaN: {custom_nans}, Shape: {custom_data.shape}")
        print(f"    cfgrib engine - NaN: {cfgrib_nans}, Shape: {cfgrib_data.shape}")

        # Check if NaN counts are similar (within 1% tolerance)
        total_points = custom_data.size
        nan_diff_pct = abs(custom_nans - cfgrib_nans) / total_points * 100

        self.assertLess(nan_diff_pct, 1.0,
                        f"{model_name} {var}: NaN count difference too large: {nan_diff_pct:.2f}%")

        # Compare valid data values
        self._compare_valid_data(model_name, var, custom_data, cfgrib_data)

    def _compare_valid_data(self, model_name, var, custom_data, cfgrib_data):
        """Helper method to compare valid data values."""
        valid_mask = ~(np.isnan(custom_data) | np.isnan(cfgrib_data))
        if not valid_mask.any():
            return

        custom_valid = custom_data[valid_mask]
        cfgrib_valid = cfgrib_data[valid_mask]

        # Check if values are close
        max_diff = np.max(np.abs(custom_valid - cfgrib_valid))
        mean_diff = np.mean(np.abs(custom_valid - cfgrib_valid))

        print(f"    Max difference in valid data: {max_diff:.6f}")
        print(f"    Mean difference in valid data: {mean_diff:.6f}")

        # Values should be very close (within 0.01 tolerance)
        self.assertLess(max_diff, 0.01,
                        f"{model_name} {var}: Data values too different")

    def test_attributes_comparison(self):
        """Compare attributes between engines"""
        if not self.test_data:
            self.skipTest("No GRIB data available")

        # Test only the first dataset for detailed attribute comparison
        model_name = list(self.test_data.keys())[0]
        data = self.test_data[model_name]
        print(f"\n=== Detailed Attributes Comparison for {model_name} ===")

        ds_custom = data['ds_custom']
        ds_cfgrib = data['ds_cfgrib']

        print("\n=== Attributes Comparison ===")

        # Compare global attributes
        custom_global = set(ds_custom.attrs.keys())
        cfgrib_global = set(ds_cfgrib.attrs.keys())

        print("\nGlobal attributes:")
        print(f"  Custom only: {custom_global - cfgrib_global}")
        print(f"  cfgrib only: {cfgrib_global - custom_global}")
        print(f"  Common: {custom_global & cfgrib_global}")

        # Compare variable attributes
        for var in ds_custom.data_vars:
            if var in ds_cfgrib.data_vars:
                custom_attrs = set(ds_custom[var].attrs.keys())
                cfgrib_attrs = set(ds_cfgrib[var].attrs.keys())

                print(f"\n{var} attributes:")
                print(f"  Custom only: {custom_attrs - cfgrib_attrs}")
                print(f"  cfgrib only: {cfgrib_attrs - custom_attrs}")
                print(f"  Common: {custom_attrs & cfgrib_attrs}")

                # Compare common attribute values
                common_attrs = custom_attrs & cfgrib_attrs
                for attr in common_attrs:
                    custom_val = ds_custom[var].attrs[attr]
                    cfgrib_val = ds_cfgrib[var].attrs[attr]

                    if custom_val != cfgrib_val:
                        print(f"    {attr}: Custom='{custom_val}' vs cfgrib='{cfgrib_val}'")

        # Compare coordinate attributes
        for coord in ds_custom.coords:
            if coord in ds_cfgrib.coords:
                custom_attrs = set(ds_custom[coord].attrs.keys())
                cfgrib_attrs = set(ds_cfgrib[coord].attrs.keys())

                if custom_attrs != cfgrib_attrs:
                    print(f"\n{coord} coordinate attributes:")
                    print(f"  Custom only: {custom_attrs - cfgrib_attrs}")
                    print(f"  cfgrib only: {cfgrib_attrs - custom_attrs}")

    def test_nan_preservation(self):
        """Test that both engines preserve NaN values correctly"""
        if not self.test_data:
            self.skipTest("No GRIB data available")

        for model_name, data in self.test_data.items():
            with self.subTest(model=model_name):
                print(f"\n=== NaN Preservation Test for {model_name} ===")

                ds_custom = data['ds_custom']
                ds_cfgrib = data['ds_cfgrib']

                for var in ds_cfgrib.data_vars:
                    if var in ds_custom.data_vars and var in ds_cfgrib.data_vars:
                        custom_data = ds_custom[var].values
                        cfgrib_data = ds_cfgrib[var].values

                        custom_nans = np.isnan(custom_data).sum()
                        cfgrib_nans = np.isnan(cfgrib_data).sum()

                        print(f"  {var} NaN counts:")
                        print(f"    Custom: {custom_nans}")
                        print(f"    cfgrib: {cfgrib_nans}")

                        # Both engines should have similar NaN patterns
                        total_points = cfgrib_data.size
                        custom_diff_pct = abs(cfgrib_nans - custom_nans) / total_points * 100

                        self.assertLess(custom_diff_pct, 1.0,
                                        f"{model_name} {var}: Custom NaN difference too large: {custom_diff_pct:.2f}%")


if __name__ == '__main__':
    unittest.main()
