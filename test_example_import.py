#!/usr/bin/env python3
"""Test script to verify the example can import correctly."""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_example_import():
    """Test that the example can import the corrected module."""
    print("Testing example import...")
    
    try:
        # Test the corrected import
        from windgrib import Grib
        print("SUCCESS: windgrib.Grib imported successfully")
        
        # Test that the old incorrect import fails
        try:
            from windgrid import Grib
            print("PROBLEM: windgrid.Grib should not work!")
            return False
        except ModuleNotFoundError:
            print("SUCCESS: windgrid.Grib correctly fails (as expected)")
        
        # Test basic functionality
        print("Testing basic Grib functionality...")
        grib = Grib()
        print(f"SUCCESS: Grib instance created successfully")
        print(f"Model: {grib.model['name']}")
        print(f"Date: {grib.date}")
        print(f"Hour: {grib.h}")
        
        return True
        
    except ImportError as e:
        print(f"FAILED: Import error - {e}")
        return False
    except Exception as e:
        print(f"FAILED: Unexpected error - {e}")
        return False

if __name__ == "__main__":
    success = test_example_import()
    if success:
        print("\nAll tests passed! The example should work correctly.")
        sys.exit(0)
    else:
        print("\nSome tests failed. Please check the errors above.")
        sys.exit(1)