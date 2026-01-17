"""GRIB data download and processing module for weather data."""

import asyncio
import json
import time
from pathlib import Path

import aiohttp
import s3fs
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from windgrib.grib_to_dataset import grib_to_dataset, grib_steps

MODELS = {
    'gfswave': {
        'product': 'global.0p25',
        'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
        'key': 'gfs.{date}/{h:02d}/wave/gridded/',
        'subsets': {
            'wind': ['UGRD', 'VGRD']
        }
    },
    'ecmwf_ifs': {
        'product': 'oper',
        'url': 'https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/',
        'key': '{date}/{h:02d}z/ifs/0p25/oper/',
        'idx': '.index',
        'subsets': {
            'wind': ['10u', '10v'],
            'land': ('lsm', 0)
        }
    }
}


async def download_file(session, url, headers=None):
    """Download a single grib file asynchronously
    with optional headers defining subset parts to download"""
    async with session.get(url, headers=headers) as response:
        if response.status in [200, 206]:
            return await response.read()
        print(f"Failed to download {url}: {response.status}")
        return None


async def download_messages(session, url, headers, idx):
    """Download messages from a single grib file asynchronously
        and return the number of downloaded messages (defined as input) """
    return idx, await download_file(session, url, headers)


def get_headers(start_byte, end_byte, **kwargs):
    """Return headers to download part of a file"""
    end = end_byte if end_byte != -1 else ''
    return {'Range': f'bytes={start_byte}-{end}'}


def get_partitions(df):
    """Compute partitions to be downloaded (from an idx DataFrame)"""
    partition_id = df.groupby('idx_file').message_id.diff().ne(1).cumsum()
    groups = df.groupby(partition_id)
    groups.agg({'start_byte': 'min'})
    partitions = pd.DataFrame([
        groups['step'].first().astype(int),
        groups['start_byte'].min(),
        groups['end_byte'].max().fillna(-1).astype(int),
        groups['url'].first(),
        groups['idx_file'].first(),
        groups['message_id'].count().rename('nb_messages'),
        groups['variable' if 'variable' in df else 'param'].agg(list)
    ]).T
    return partitions


def parse_subset_config(config):
    """
    Parse subset configuration into standardized format.

    Args:
        config: Can be:
            - list: ['var1', 'var2'] -> variables only
            - tuple: (['var1', 'var2'],) -> variables only
            - tuple: (['var1', 'var2'], step) -> variables + step
            - tuple: (['var1', 'var2'], step, filter_keys) -> variables + step + filters
            - tuple: (['var1', 'var2'], filter_keys) -> variables +  filters

    Returns:
        dict: {'variables': list, 'step': list/int/None, 'filter_keys': dict}
    """
    if isinstance(config, list):
        return {'var': config, 'step': None, 'filter_keys': {}}

    if isinstance(config, tuple) and len(config) in [2, 3]:
        filter_keys = {}
        if isinstance(config[-1], dict):
            filter_keys = config[-1]
            config = config[:-1]
        step = config[1] if len(config) == 2 else None

        return {
            'var': config[0],
            'step': step,
            'filter_keys': filter_keys
        }

    raise ValueError(f"Invalid subset configuration: {config}")


class GribSubset:
    """Handles individual subset operations for GRIB data with async downloads."""

    @staticmethod
    def from_model(name, grib, items):
        """
        Create a GribSubset from model subset definition
        items can be:
        - str: 'u' -> var='u'
        - list: ['var1', 'var2'] -> var
        - tuple: (['var1', 'var2'],) -> var only
        - tuple: (['var1', 'var2'], step) -> variables + step
        - tuple: (['var1', 'var2'], step, filter_keys) -> variables + step + filter_keys
        - tuple: (['var1', 'var2'], filter_keys) -> variables + filter_keys
        """
        if isinstance(items, str):
            items = [items]
        if isinstance(items, list):
            return GribSubset(name, grib, items)
        if isinstance(items, tuple):
            _args = [arg for arg in items if not isinstance(arg, dict)]
            filter_keys = {}
            for arg in items:
                if isinstance(arg, dict):
                    filter_keys.update(arg)
            return GribSubset(name, grib, *_args, **filter_keys)
        raise ValueError(f"Invalid subset configuration: {items}")

    def __init__(self, name, grib_instance, var=None, step=None, **filter_keys):
        """Initialize subset with name, configuration and parent grib instance."""
        self.name = name
        self.var = var
        self.filter_keys = filter_keys
        self.grib = grib_instance
        self._new_idx_files = None
        self._grib_data = None
        self._ds = None
        self._step = self.grib.step
        if step is not None:
            self._step = list(np.unique(step))

        print(f"\n🧩 [{self.grib.model['name']}] ({name})"
              f" Initializing subset: {self}")

    def __str__(self):
        return f"({self.name}) {self.var} len(step)={len(self.step)}"

    def __getitem__(self, key):
        key = np.atleast_1d(key)
        if key.dtype == np.bool_:
            key = np.array(self.step)[key]
        if not np.issubdtype(key.dtype, np.integer):
            raise KeyError('GriSubset indexing support only integer dtype')
        invalid_keys = set(key) - set(self.step)
        if invalid_keys:
            raise KeyError(f'{invalid_keys} not found in step')
        return GribSubset(self.name, self.grib, self.var, step=key, **self.filter_keys)

    @property
    def grib_file(self):
        """Get grib file path for this subset."""
        return (self.grib.folder_path /
                f"{self.name}_{self.grib.model['name']}."
                f"{self.grib.model['product']}.grib2")

    @property
    def netcdf_file(self):
        """Get NetCDF file path for this subset."""
        return self.grib_file.with_suffix('.nc')

    @property
    def idx(self):
        """Return a dataframe with extracted GRIB messages index for this subset"""

        # get grib message dataframe from grib instance
        df = self.grib.idx

        if df.empty:
            return df

        # filter with var
        var_name = 'variable' if 'variable' in df else 'param'
        df = df.loc[df[var_name].isin(self.var)]

        # filter with step
        df = df.loc[df['step'].isin(self.step)]

        # filter with subset filter keys
        for key, value in self.filter_keys.items():
            if isinstance(value, list):
                df = df.loc[df[key].isin(value)]
            else:
                df = df.loc[df[key] == value]

        return df.sort_values(['step', 'message_id']).reset_index(drop=True)

    @property
    def step(self):
        """Get available steps for this subset."""
        return self._step

    def current_step(self):
        """Get current step for this subset."""
        return self.grib.current_step

    @property
    def ds(self):
        """dataset accessor"""
        if self._ds is None:
            self.load_dataset()
        return self._ds

    def load_dataset(self):
        """loading dataset from cache files"""
        start_time = time.time()
        ds = None
        nc_steps = set()

        # Load from NetCDF if exists
        if self.grib.use_cache and self.netcdf_file.exists():
            with xr.open_dataset(self.netcdf_file, chunks={'step': 1}, decode_timedelta=False) as nc_ds:
                ds = nc_ds.sel(step=self.step).load()
                nc_steps = set(ds.step.values)
                print(f"📂 [{self.grib.model['name']}] ({self.name})"
                      f" Found {len(nc_steps)} steps from NetCDF file")

        grib_data = self._grib_data
        if self.grib.use_cache and self.grib_file.exists():
            with open(self.grib_file, 'rb') as f:
                grib_data = f.read()

        new_steps = (grib_steps(grib_data) - nc_steps).intersection(self.step)

        if new_steps:
            print(f"🎯 [{self.grib.model['name']}] ({self.name})"
                  f" Found {len(new_steps)} steps to load in GRIB")

            # Check GRIB data for additional steps
            desc = f"📊 [{self.grib.model['name']}] ({self.name}) Decoding GRIB messages"

            if ds is None:
                ds = grib_to_dataset(grib_data, new_steps, progress_bar=self.grib.progress_bar, desc=desc)
            else:
                new_ds = grib_to_dataset(grib_data, new_steps, progress_bar=self.grib.progress_bar, desc=desc)
                ds = xr.concat([ds, new_ds], dim='step', join='outer')
                ds = ds.sortby('step')

        print(f"✅ [{self.grib.model['name']}] ({self.name})"
              f" Dataset loaded in {time.time() - start_time:.2f}s")

        self._ds = ds

    def download(self):
        """Download GRIB files for this subset asynchronously."""
        start_time = time.time()

        # Get idx dataframe for this subset
        df = self.idx

        if self.grib.use_cache:

            available_steps = set()

            # get existing steps from NetCDF
            if self.netcdf_file.exists():
                with xr.open_dataset(self.netcdf_file, chunks={'step': 1}, decode_timedelta=False) as ds:
                    ds = ds.sel(step=self.step)
                    available_steps = set(ds.step.values)

            # get existing steps from GRIB file
            if self.grib_file.exists():
                with open(self.grib_file, 'rb') as f:
                    grib_data = f.read()
                    if grib_data:
                        available_steps.update(grib_steps(grib_data))

            # filter available steps from idx dataframe
            if available_steps:
                df = df[~df['step'].isin(available_steps)]

            if df.empty:
                print(f'✔️ [{self.grib.model["name"]}] ({self.name})'
                      f' All available GRIB messages have already been downloaded')
                return 0

        print(f"🎯 [{self.grib.model['name']}] ({self.name})"
              f" Found {len(df)} new message{'s' if len(df) > 1 else ''}")
        # Wait for a short period to avoid console prints interleaving
        time.sleep(0.01)

        desc = (f'📥 [{self.grib.model["name"]}] ({self.name})'
                f' Downloading from GRIB files')

        grib_data = self.grib.download_data(df, desc=desc)

        print(f"✅ [{self.grib.model['name']}] ({self.name})"
              f" Downloaded in {time.time() - start_time:.2f}s")

        if self.grib.use_cache:
            with open(self.grib_file, 'ab' if self.grib_file.exists() else 'wb') as f:
                f.write(grib_data)
        else:
            self._grib_data = grib_data

        return len(df)

    def clear_cache(self):
        """Clean cache files."""
        self.grib_file.unlink(missing_ok=True)
        self.netcdf_file.unlink(missing_ok=True)

    def to_netcdf(self, encoding=None, zlib=False, complevel=1):
        """Save dataset to NetCDF file with uint16 encoding."""
        print(f"💾 [{self.grib.model['name']}] ({self.name})"
              f" Saving NetCDF file: {self.netcdf_file.as_uri()}")

        # Check existing NetCDF file
        if self.netcdf_file.exists():
            with xr.open_dataset(self.netcdf_file, chunks={'step': 1}, decode_timedelta=False) as ds:
                steps = set(ds.step.values)
                if not steps.difference(self.step):
                    print(f"✅ {self.grib.model['name']}] ({self.name})"
                          f" NetCDF file already exists and up to date")
                    return

        ds = self.ds

        start_time = time.time()

        if encoding is None:
            # Calculate uint16 encoding for each variable
            encoding = {}
            uint16_max = 65534  # 2^16 - 2 (exclude 0 for _FillValue)

            for var in ds.data_vars:
                data = ds[var].values
                valid_data = data[~np.isnan(data)]

                if len(valid_data) > 0:
                    data_min = float(valid_data.min())
                    data_max = float(valid_data.max())

                    # Calculate scale and offset for uint16 (1-65535 range)
                    scale = (data_max - data_min) / uint16_max if data_max != data_min else 1.0
                    offset = data_min - scale  # Adjust so min maps to 1, not 0

                    encoding[var] = {
                        'dtype': 'uint16',
                        'scale_factor': scale,
                        'add_offset': offset,
                        '_FillValue': 0,
                        'zlib': zlib,
                        'complevel': complevel
                    }
                else:
                    # Fallback for variables with all NaN
                    encoding[var] = {'dtype': 'float32', '_FillValue': np.nan,
                                     'zlib': zlib, 'complevel': complevel}

        print(f"✅ [{self.grib.model['name']}] ({self.name})"
              f" NetCDF encoding computed in {time.time() - start_time:.2f}s")

        ds.to_netcdf(self.netcdf_file, encoding=encoding)

        print(f"✅ [{self.grib.model['name']}] ({self.name})"
              f" Netcdf file saved in {time.time() - start_time:.2f}s")


class Grib:
    """Main class for GRIB data operations."""

    def __init__(self, timestamp=None, date=None, model='gfswave', data_path=None,
                 max_concurrent=100, use_cache=True, progress_bar=True):
        """Initialize Grib instance."""
        self.model_name = model
        self.model = MODELS[model].copy()
        self.model['name'] = model
        self.max_concurrent = max_concurrent
        self.use_cache = use_cache
        self.progress_bar = progress_bar
        self._idx_files = None
        self._idx = None
        self._step = None
        self._current_step = None
        self._subsets = {}

        # Initialize data_path
        if data_path is None:
            self.data_path = Path(__file__).parent.parent / 'data' / 'grib'
        else:
            self.data_path = Path(data_path)

        # Initialize timestamp
        timestamp = (pd.Timestamp(timestamp) if timestamp is not None
                     else pd.Timestamp.utcnow().as_unit('s'))
        self.timestamp = timestamp

        print(f"\n🚀 [{model}]"
              f" Initializing Grib for product"
              f" {self.model['product']} at {self.timestamp.strftime('%Y%m%d-%Hh')} UTC")

        if date is None:
            self.date = timestamp.floor('6h')
            self._idx_files = self.latest
        else:
            self.date = pd.Timestamp(date).floor('6h')

        # Initialize forecast data folder
        if self.use_cache:
            self.folder_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 [{model}] Data folder: {self.folder_path.as_uri()}")

        # Initialize grib subsets from model
        for name, subset_args in self.model['subsets'].items():
            if isinstance(subset_args, list):
                self._subsets[name] = GribSubset(name, self, var=subset_args)
            elif not isinstance(subset_args, tuple):
                raise ValueError(f"Invalid subset definition: {subset_args}")
            elif isinstance(subset_args[-1], dict):
                self._subsets[name] = GribSubset(name, self, *subset_args[:-1], **subset_args[-1])
            else:
                self._subsets[name] = GribSubset(name, self, *subset_args)

    def __str__(self):
        return f"Grib['{self.model['name']}']"

    def __getitem__(self, key):
        """Get an existing subset and create a new one with additional args from keys."""
        if not isinstance(key, str):
            raise KeyError(f"'{key}' should be a string")
        if hasattr(self, key):
            return getattr(self, key)
        if key in self._subsets:
            return self._subsets[key]
        if key in self.model['subsets']:
            return GribSubset.from_model(key, self, self.model['subsets'][key])
        raise KeyError(f"'{key}' not found in {self}")

    def __iter__(self):
        # First iterate on model subsets keys
        yield from self.model['subsets']
        # Then iterate on existing subsets keys
        for key in self._subsets:
            if key not in self.model['subsets']:
                yield key

    @property
    def folder_path(self):
        """Get local folder path for storing data."""
        return (self.data_path / self.date.strftime("%Y%m%d") /
                str(self.date.hour))

    @property
    def latest(self):
        """Find latest available forecast for this model
        before timestamp attribute defined at init"""
        idx_files = []
        retry_count = 0
        max_retry = 10
        self.date = self.timestamp.floor('6h')
        while not idx_files and retry_count <= max_retry:
            idx_files = self.idx_files
            if not idx_files:
                print(f"❌ [{self.model['name']}]"
                      f" Grib files not found: {self.date.strftime('%Y%m%d-%Hh')}")
                self.date -= pd.Timedelta(6, 'h')
                self._idx_files = None
            retry_count += 1
        if idx_files:
            print(f"✅ [{self.model['name']}]"
                  f" Grib files found ({len(idx_files)}): {self.date.strftime('%Y%m%d-%Hh')}")
        else:
            print(f"⚠️ [{self.model['name']}]"
                  f" No files found after {max_retry} attempts for {self.model['name']}")
            return []
        return self.idx_files

    @property
    def url(self):
        """Get URL for the model data."""
        return self.model['url'] + self.model['key'].format(
            date=self.date.strftime("%Y%m%d"),
            h=self.date.hour
        )

    @property
    def s3(self):
        """Get S3 path for the model data."""
        return (self.url.replace('https', 's3')
                .replace('.s3', '')
                .replace('.amazonaws.com', '')
                .replace('.eu-central-1', ''))

    @property
    def idx_files(self):
        """Get list of available index files."""
        if self._idx_files is None:
            files_pattern = self.s3 + '*'
            if 'product' in self.model:
                product = self.model['product']
                files_pattern += f'{product}*'
            files_pattern += self.model.get('idx', '.idx')
            idx_files = s3fs.S3FileSystem(anon=True).glob(files_pattern)
            self._idx_files = [f.split('/')[-1] for f in idx_files]
        return self._idx_files

    @property
    def idx(self):
        """Parse idx files and return a DataFrame of all messages."""
        if self._idx is None:
            self._idx = asyncio.run(self._download_idx_files(self.idx_files))
        return self._idx

    @property
    def step(self):
        """Get sorted array of available steps."""
        if self._step is None:
            self._step = np.unique(self.idx['step'].values)
        return self._step

    @property
    def current_step(self):
        """Get the last step before timestamp"""
        if self._current_step is None:
            h = int((self.timestamp - self.date).round('1h') / pd.Timedelta('1h'))
            self._current_step = self.step[np.searchsorted(self.step, h)]
        return self._current_step

    def download(self):
        """Download all subsets."""
        for subset in self:
            self[subset].download()

    def download_data(self, df, desc=None):
        """Apply async download of subsets data defined in df"""
        return asyncio.run(self._download_data(df, desc))

    def load(self):
        for subset in self:
            self[subset].load_dataset()

    def to_netcdf(self):
        """Save each subset dataset to separate NetCDF files."""
        for subset in self:
            self[subset].to_netcdf()

    def clear_cache(self):
        """Clean cache files."""
        for subset in self.model['subsets'].keys():
            self[subset].clear_cache()

    def sel(self, var=None, step=None, name=None, **kwargs):
        """Create a GribSubset with specified filters (xarray-like interface)."""
        if name is None:
            name = f"({var if var else ''},{step if step else ''},{kwargs})"
        return GribSubset(name, self, var=var, step=step, **kwargs)

    def get_file_url(self, idx_file):
        """Get full URL for file."""
        file = idx_file.replace(self.model.get('idx', '.idx'), '')
        url = self.url + file
        ext = self.model.get('ext', '.grib2')
        if ext and not url.endswith(ext):
            url += ext
        return url

    @property
    def _connector(self):
        return aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=self.max_concurrent // 2,
            ttl_dns_cache=600,
            use_dns_cache=True,
        )

    async def _download_idx_files(self, idx_files):
        """Execute all downloads asynchronously."""

        async with aiohttp.ClientSession(connector=self._connector) as session:
            tasks = [self._download_idx_file(session, idx_file) for idx_file in idx_files]
            dfs = []
            for coro in asyncio.as_completed(tasks):
                df = await coro
                if df is not None:
                    dfs.append(df)
            return pd.concat(dfs, ignore_index=True)

    async def _download_idx_file(self, session, idx_file):
        """Download index file and return DataFrame of messages."""

        timeout = aiohttp.ClientTimeout(total=30)
        async with session.get(self.url + idx_file, timeout=timeout) as response:
            idx_txt = await response.text()

        if idx_file.endswith('.index'):
            idx_txt = idx_txt.rstrip('\n ').replace('\n', ',')
            idx_txt = '[' + idx_txt + ']'
            df = pd.DataFrame(json.loads(idx_txt))
            df['message_id'] = df.index
            df['start_byte'] = df['_offset']
            df['end_byte'] = df['start_byte'] + df['_length'] - 1
        else:
            idx_txt = idx_txt.split('\n')
            df = pd.DataFrame(
                [row.split(':') for row in idx_txt if row],
                columns=[
                    'message_id', 'start_byte', 'reference_time',
                    'variable', 'layer', 'step', '?'
                ])
            # df = df.drop(columns='?')
            df['message_id'] = df['message_id'].astype(int)
            df['start_byte'] = df['start_byte'].astype(int)
            df['end_byte'] = df['start_byte'].shift(-1) - 1
            df['_length'] = df['end_byte'] - df['start_byte'] + 1
            df['end_byte'] = df['end_byte'].fillna(-1).astype(int)
            df['_length'] = df['_length'].fillna(-1).astype(int)
            # replace bad step definition (anl, whitespace and step range)
            df.loc[df['step'] == 'anl', 'step'] = '0'
            df['step'] = df['step'].str.split(' ').str[0]
            df['step'] = df['step'].str.split('-').str[0]


        df['step'] = df['step'].astype(int)
        df['url'] = self.get_file_url(idx_file)
        df['idx_file'] = idx_file

        return df

    async def _download_data(self, df, desc=None):
        """Download GRIB data asynchronously."""

        downloaded_data = []
        partitions = get_partitions(df)

        async with aiohttp.ClientSession(connector=self._connector) as session:
            tasks = []
            for partitions_id, idx_data in partitions.iterrows():
                headers = get_headers(**idx_data.to_dict())
                tasks.append(download_messages(session, idx_data['url'], headers, partitions_id))

            if self.progress_bar:
                if desc is None:
                    desc = f'📥 [{self.model["name"]}] Downloading GRIB files'
                with tqdm(total=len(df), desc=desc) as pbar:
                    for coro in asyncio.as_completed(tasks):
                        data = await coro
                        if data:
                            downloaded_data.append(data)
                        pbar.update(partitions.loc[data[0], 'nb_messages'])
            else:
                downloaded_data = await asyncio.gather(*tasks)

        downloaded_data = sorted(downloaded_data)

        return b''.join([data[1] for data in downloaded_data])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download and process GRIB data')

    parser.add_argument('--model', default='gfswave',
                        help='Model to use (default: gfswave)')
    parser.add_argument('--timestamp',
                        help='Timestamp for forecast (default: UTC now)')
    parser.add_argument('--no-cache', default=False, action='store_true',
                        help='Disable cache')
    parser.add_argument('--save-nc', default=True, action='store_true',
                        help='Save to NetCDF')
    parser.add_argument('--progress-bar', default=True, action='store_true',
                        help='Show progress bar while downloading or decoding GRIB data')
    parser.add_argument('--clear-cache', default=False, action='store_true',
                        help='Clear cache from GRIB files')

    args = parser.parse_args()

    g = Grib(
        model=args.model,
        timestamp=args.timestamp,
        use_cache=not args.no_cache,
        progress_bar=args.progress_bar
    )

    if args.clear_cache:
        g.clear_cache()

    g.download()

    if args.save_nc:
        g.to_netcdf()
