"""Simple GRIB decoder function for converting GRIB bytes to xarray datasets."""

from concurrent.futures import ProcessPoolExecutor
import eccodes
import xarray as xr
import numpy as np
from cfgrib.cfmessage import CfMessage
from cfgrib.dataset import (
    GRID_TYPES_2D_NON_DIMENSION_COORDS,
    GLOBAL_ATTRIBUTES_KEYS,
    DATA_ATTRIBUTES_KEYS,
    EXTRA_DATA_ATTRIBUTES_KEYS,
    COORD_ATTRS
)
from tqdm import tqdm


def cfmessage(msg_bytes: bytes):
    """Create cfgrib CfMessage from message bytes."""
    mid = eccodes.codes_new_from_message(msg_bytes)
    return CfMessage(mid) if mid is not None else None


def get_var_name(msg: CfMessage):
    """Extract variable name from GRIB message."""
    return msg["cfVarName"] if "cfVarName" in msg else msg["shortName"]


def get_global_attributes(msg: CfMessage):
    """Extract global attributes from GRIB message."""
    attrs = {}
    for key in GLOBAL_ATTRIBUTES_KEYS:
        if key in msg:
            attrs[f"GRIB_{key}"] = msg[key]
    return attrs


def get_var_attributes(msg: CfMessage):
    """Extract variable attributes from GRIB message."""
    attrs = {f"GRIB_{k}": msg[k] for k in DATA_ATTRIBUTES_KEYS + EXTRA_DATA_ATTRIBUTES_KEYS
             if k in msg}
    # Add CF-compliant attributes
    for cf_attr in ["cfName", "name", "units"]:
        if cf_attr in msg:
            attr_name = {"cfName": "standard_name", "name": "long_name"}.get(cf_attr, cf_attr)
            attrs[attr_name] = msg[cf_attr]
    return attrs


def get_scalars_coords(msg: CfMessage):
    """Extract scalar coordinates from GRIB message."""
    coords = {}

    for coord in COORD_ATTRS:
        if coord in ['latitude', 'longitude']:
            continue
        if coord == 'step':
            x = msg.get('step:int', None)
        else:
            x = msg.get(coord, None)
        if x is None:
            continue

        if np.isscalar(x):
            x = np.array(x)[()]
            coord_dims = []
        else:
            x = np.array([x])
            coord_dims = [coord]

        coord_attrs = COORD_ATTRS.get(coord, {})
        if coord_attrs:
            coords[coord] = (coord_dims, x, coord_attrs)
        else:
            coords[coord] = (coord_dims, x)

    # Handle level coordinates
    if "typeOfLevel" in msg and "level" in msg:
        name = msg["typeOfLevel"]
        data = np.array(msg["level"], dtype=float)[()]
        if name == "surface":
            coords[name] = ([], data, {"units": "1", "long_name": "surface"})
        else:
            coords[name] = ([], data)

    return coords


def get_geo_coords(msg: CfMessage):
    """Extract geographical coordinates from GRIB message."""
    if not isinstance(msg, CfMessage):
        msg = cfmessage(msg)
    shape = msg["Ny"], msg["Nx"]
    data_size = eccodes.codes_get_size(msg.codes_id, "values")
    coords = {}
    for coord in ["latitude", "longitude"]:
        coord2 = {"latitude": "latitudes", "longitude": "longitudes"}.get(coord, coord)
        x = msg.get(coord2, None)
        if x is None:
            continue

        if isinstance(x, np.ndarray) and x.size == data_size:
            if msg["gridType"] in GRID_TYPES_2D_NON_DIMENSION_COORDS:
                coord_dims = ["y", "x"]
                x = x.reshape(shape)
            else:
                coord_dims = [coord]
                if coord == "latitude":
                    x = x.reshape(shape)[:, 0].copy()
                elif coord == "longitude":
                    x = x.reshape(shape)[0].copy()
        else:
            coord_dims = [coord]

        coord_attrs = COORD_ATTRS.get(coord, {})
        if coord_attrs:
            coords[coord] = (coord_dims, x, coord_attrs)
        else:
            coords[coord] = (coord_dims, x)

    return coords


def message_to_data_array(msg_bytes):
    """Convert message bytes to DataArray without geographical coordinates."""
    msg = cfmessage(msg_bytes)
    if msg is None:
        return None

    var_name = get_var_name(msg)
    vals = msg["values"].reshape(msg["Ny"], msg["Nx"]).astype(np.float32)

    # Handle missing values properly
    missing_value = msg.get("missingValue", None)
    if missing_value is not None:
        vals = np.where(vals == missing_value, np.nan, vals)

    scalar_coords = get_scalars_coords(msg)

    # Create DataArray without geographical coordinates
    da = xr.DataArray(
        data=vals,
        name=var_name,
        coords=scalar_coords,
        dims=('y', 'x')
    )
    return da


def split_messages(grib_bytes: bytes):
    """Split GRIB bytes into individual message bytes."""
    messages = []
    offset = 0

    while offset < len(grib_bytes):
        grib_pos = grib_bytes.find(b"GRIB", offset)
        if grib_pos == -1:
            break

        part_size = int.from_bytes(grib_bytes[grib_pos + 12:grib_pos + 16], "big")
        message = grib_bytes[grib_pos:grib_pos + part_size]
        messages.append(message)
        offset = grib_pos + part_size

    return messages


def grib_steps(grib_bytes: bytes):
    """Get a set of available steps from GRIB bytes."""
    messages = split_messages(grib_bytes)
    steps = set()
    for msg in messages:
        step = cfmessage(msg).get('step:int', None)
        if step is not None:
            steps.add(step)
    return steps


def grib_to_dataset(
        grib_bytes: bytes,
        steps=None,
        parallel: int = 20,
        progress_bar: bool = True,
        decode_timedelta: bool = False,
        desc: str = '📊 Decoding GRIB messages'
):  # pylint: disable=too-many-locals, too-many-branches
    """Convert GRIB bytes to xarray dataset."""
    # Split grib bytes in messages bytes
    messages = split_messages(grib_bytes)

    if steps:
        messages = [msg for msg in messages if cfmessage(msg)['step'] in steps]

    if parallel and len(messages) > parallel:
        with ProcessPoolExecutor() as executor:
            if progress_bar:
                data_arrays = list(tqdm(
                    executor.map(message_to_data_array, messages),
                    desc=desc,
                    total=len(messages)
                ))
            else:
                data_arrays = executor.map(message_to_data_array, messages)
    else:
        print(desc + f' ({len(messages)})')
        data_arrays = [message_to_data_array(msg) for msg in messages]

    if len(data_arrays) == 1:
        ds = data_arrays[0].to_dataset()
    else:
        # Sort data arrays by time and variable name
        data_arrays = sorted(data_arrays, key=lambda x: x.step)
        steps = np.unique([da.step for da in data_arrays if 'step' in da.coords])
        vars_data = {}
        for da in data_arrays:
            if da.name not in vars_data:
                vars_data[da.name] = []
            vars_data[da.name].append(da)

        # Create dataset
        if len(steps) > 1:
            datasets = [xr.concat(items, dim='step', coords='different', compat='no_conflicts')
                        for items in vars_data.values()]
        else:
            datasets = [da.to_dataset() for da in data_arrays]
        ds = xr.merge(datasets, compat='no_conflicts')

    # Add variables attrs
    var_attrs = {}
    for var_name in ds.data_vars:
        for msg_bytes in messages:
            msg = cfmessage(msg_bytes)
            if msg and var_name == get_var_name(msg):
                var_attrs[var_name] = get_var_attributes(msg)
                break
        if var_name in var_attrs:
            ds[var_name].attrs.update(var_attrs[var_name])

    # Get geo coords and global attributes from first message
    msg = cfmessage(messages[0])
    geo_coords = get_geo_coords(msg)
    global_attrs = get_global_attributes(msg)

    # Add geographical coordinates to dataset (shared by all variables and grib messages)
    ds = ds.rename_dims(y='latitude', x='longitude')
    ds = ds.assign_coords(**geo_coords)
    # add coords attributes
    for coord in ds.coords:
        if coord in COORD_ATTRS:
            ds[coord].attrs.update(COORD_ATTRS[coord])

    # Add generic attributes
    ds.attrs["Conventions"] = "CF-1.7"
    ds.attrs["history"] = "Created by windgrib grib_to_dataset"
    ds.attrs.update(global_attrs)
    if 'GRIB_centreDescription' in ds.attrs:
        ds.attrs['institution'] = ds.attrs['GRIB_centreDescription']

    ds = xr.decode_cf(ds, decode_timedelta=decode_timedelta)

    return ds
