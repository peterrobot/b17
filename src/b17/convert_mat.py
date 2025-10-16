import os

import numpy as np
import scipy.io
import xarray as xr


def convert_mat_to_nc(mat_filename, nc_filename, variable_name):
    """
    Loads a .mat file, extracts the specified variable, and saves it as a .nc file.

    Args:
        mat_filename (str): The input MATLAB file name.
        nc_filename (str): The output NetCDF file name.
        variable_name (str): The name of the data variable within the .mat file.
    """
    if not os.path.exists(mat_filename):
        print(f"Error: Input file not found: {mat_filename}")
        return

    try:
        # Load the .mat file
        mat_data = scipy.io.loadmat(mat_filename)

        # Extract the numpy array from the dictionary
        data_array = mat_data[variable_name]

        # Create an xarray Dataset
        # The variable name in the Dataset must match what the b17.py script expects
        ds = xr.Dataset(
            {variable_name: (("dim_0", "dim_1"), data_array)},
            coords={
                "dim_0": np.arange(data_array.shape[0]),
                "dim_1": np.arange(data_array.shape[1]),
            },
        )

        # Save to NetCDF format
        ds.to_netcdf(nc_filename)

        print(f"Successfully converted '{mat_filename}' to '{nc_filename}'")

    except Exception as e:
        print(f"An error occurred while converting {mat_filename}: {e}")
