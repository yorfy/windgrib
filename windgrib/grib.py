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

from windgrib.grib_to_dataset import grib_to_dataset

MODELS = {
    'gfswave': {
        'product': 'global.0p25',
        'url': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/',
        'key': 'gfs.{date}/{h:02d}/wave/gridded/',
        'subsets': {
            'wind': {'variable': ['UGRD', 'VGRD']}
        }
    },
    'ecmwf_ifs': {
        'product': 'oper',
        'url': 'https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/',
        'key': '{date}/{h:02d}z/ifs/0p25/oper/',
        'idx': '.index',
        'subsets': {
            'wind': {'param': ['10u', '10v'], 'rename': {'u10': 'u', 'v10': 'v'}},
            'land': {'param': ['lsm'], 'step': 0}
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
    return {'Range': f'bytes={start_byte}-{end_byte if end_byte != -1 else ''}'}


def get_partitions(df):
    """Compute partitions to be downloaded (from an idx DataFrame)"""
    df['partition_id'] = df.groupby('idx_file').message_id.diff().ne(1).cumsum()
    groups = df.groupby('partition_id')
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


class GribSubset:
    """Handles individual subset operations for GRIB data with async downloads."""

    def __init__(self, name, config, grib_instance):
        """Initialize subset with name, configuration and parent grib instance."""
        self.name = name
        self.config = config
        self.grib = grib_instance
        self._new_idx_files = None
        self._grib_data = None
        self._ds = None

        print(f"🧩 [{self.grib.model['name']}] Initializing {name} subset:"
              f" {str(config).replace('{', '').replace('}', '')}")

    @property
    def grib_file(self):
        """Get grib file path for this subset."""
        return (self.grib.folder_path /
                f"{self.name}_{self.grib.model['name']}."
                f"{self.grib.model['product']}.grib2")

    @property
    def grib_ls(self):
        """Get GRIB list file path for this subset."""
        return self.grib_file.with_suffix('.grib2.ls')

    @property
    def nc_file(self):
        """Get NetCDF file path for this subset."""
        return self.grib_file.with_suffix('.nc')

    @property
    def nc_ls(self):
        """Get GRIB list file path for this subset."""
        return self.nc_file.with_suffix('.nc.ls')

    @property
    def idx(self):
        """Return a dataframe with extracted GRIB messages index for this subset"""

        # get grib message dataframe from grib instance
        df = self.grib.idx

        # filter with subset params
        for key, value in self.config.items():
            if key == 'rename':
                continue
            if isinstance(value, list):
                df = df.loc[df[key].isin(value)]
            else:
                df = df.loc[df[key] == value]

        return df.sort_values(['step', 'message_id']).reset_index(drop=True)

    def download(self, clear_cache=False, progress_bar=True):
        """Download GRIB files for this subset asynchronously."""
        start_time = time.time()

        if clear_cache:
            self.clear_cache()

        # Evaluate new_idx_files
        new_files = self.new_idx_files
        # get grib messages for new idx files
        if not new_files:
            print(f'✔️ [{self.grib.model["name"]}] All available GRIB files '
                  f'for {self.name} have already been downloaded')
            return None

        df = self.idx
        df = df[df['idx_file'].isin(new_files)]

        print(f"🎯 [{self.grib.model['name']}] Found {len(df)} "
              f"new message{'s' if len(df) > 1 else ''} for {self.name} subset")
        # wait for a short time to prevent logs interleaving
        time.sleep(0.01)

        # download all messages
        desc = f'📥 [{self.grib.model["name"]}] Downloading {self.name} subset from GRIB files'
        self._grib_data = self.grib.download_data(df, progress_bar=progress_bar, desc=desc)

        download_time = time.time() - start_time
        print(f"✅ [{self.grib.model['name']}] {self.name} subset "
              f"downloaded in {download_time:.2f}s")

        if self.grib.use_cache:
            self.to_grib_file()

        return len(df)

    @property
    def new_idx_files(self):
        """Get list of files that need to be downloaded."""
        if self._new_idx_files is None:
            idx_files = self.grib.idx_files

            if self.grib_ls.exists():
                with open(self.grib_ls, 'r', encoding='utf-8') as f:
                    ls = f.read().splitlines()
                idx_files = [idx_file for idx_file in idx_files if idx_file not in ls]

            self._new_idx_files = idx_files

        return self._new_idx_files

    @property
    def ds(self):
        """dataset accessor"""
        if self._ds is None:
            self._ds = self.load_dataset()
        return self._ds

    def load_dataset(self):
        """loading dataset from cache files"""
        ds = None
        # if self.grib_ls.exists():
        #     with open(self.grib_file, 'rb') as f:
        #         grib_data = f.read()
        #     ds = grib_to_dataset(grib_data)

        if not self.new_idx_files:
            return ds

        start_time = time.time()
        desc = f"📊 [{self.grib.model['name']}] Decoding GRIB messages for {self.name} subset"
        if ds is None:
            ds = grib_to_dataset(self._grib_data, desc=desc)
            if 'rename' in self.config:
                ds = ds.rename(self.config['rename'])
        else:
            new_steps = grib_to_dataset(self._grib_data, desc=desc)
            ds = xr.concat([ds, new_steps], dim='time', join='outer')

        ds = ds.sortby('step')

        print(f"✅ [{self.grib.model['name']}] {self.name} subset loaded in "
              f"{time.time() - start_time:.2f}s")

        return ds

    def clear_cache(self):
        """Clean cache files."""
        self.grib_file.unlink(missing_ok=True)
        self.grib_ls.unlink(missing_ok=True)

    def to_nc(self, zlib=False, complevel=1):
        """Save dataset to NetCDF file with uint16 encoding."""

        start_time = time.time()
        print(f"💾 [{self.grib.model['name']}] Saving {self.name} Netcdf file to: "
              f"{self.nc_file.as_uri()}")

        if self.nc_file.exists() and not self.new_idx_files:
            print(f"NetCDF file already exists and up to date: {self.nc_file}")
            return

        with open(self.nc_ls, 'a', encoding='utf-8') as f:
            f.write('\n'.join(self.new_idx_files))

        ds = self.ds

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

        print(f"✅ [{self.grib.model['name']}] {self.name} Netcdf file encoding computed in "
              f"{time.time() - start_time:.2f}s")
        ds.to_netcdf(self.nc_file, encoding=encoding)

        print(f"✅ [{self.grib.model['name']}] {self.name} Netcdf file saved in "
              f"{time.time() - start_time:.2f}s")

    def to_grib_file(self):
        """Save GRIB file."""
        if not self._grib_data:
            print(f"❌  [{self.grib.model['name']}] No GRIB data to save for {self.name} subset")
            return

        with open(self.grib_file, 'ab') as f:
            f.write(self._grib_data)

        with open(self.grib_ls, 'a', encoding='utf-8') as f:
            f.write('\n'.join(self.new_idx_files))

        print(f"💾 [{self.grib.model['name']}] {self.name} GRIB file saved: "
              f"{self.grib_file.as_uri()}")


class Grib:
    """Main class for GRIB data operations."""

    def __init__(self, timestamp=None, date=None, model='gfswave', data_path=None,
                 max_concurrent=100, use_cache=True):
        """Initialize Grib instance."""
        self.model_name = model
        self.model = MODELS[model].copy()
        self.model['name'] = model
        self.max_concurrent = max_concurrent
        self.use_cache = use_cache
        self._idx_files = None
        self._idx = None
        self._steps = None
        self._step = None

        # Initialize data_path
        if data_path is None:
            self.data_path = Path(__file__).parent.parent / 'data' / 'grib'
        else:
            self.data_path = Path(data_path)

        # Initialize time and find forecast
        timestamp = (pd.Timestamp(timestamp) if timestamp is not None
                     else pd.Timestamp.utcnow())
        self.timestamp = timestamp

        print(f"🚀 [{model}] Initializing Grib for product "
              f"{self.model['product']} at {self.timestamp.strftime('%Y%m%d-%Hh')} UTC")

        if date is None:
            self.date = timestamp.floor('6h')
            self._idx_files = self.latest
        else:
            self.date = pd.Timestamp(date).floor('6h')

        # Initialize forecast data folder
        self.folder_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 [{model}] Data folder: {self.folder_path.as_uri()}")

        self._subsets = {}
        if 'subsets' in self.model:
            for name, config in self.model['subsets'].items():
                self._subsets[name] = GribSubset(name, config, self)

    def __getitem__(self, key):
        """Access subset."""
        if key in self._subsets:
            return self._subsets[key]
        raise KeyError(f"Subset '{key}' not found")

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
                print(f"❌ [{self.model['name']}] Grib files not found: "
                      f"{self.date.strftime('%Y%m%d-%Hh')}")
                self.date -= pd.Timedelta(6, 'h')
                self._idx_files = None
            retry_count += 1
        if idx_files:
            print(f"✅ [{self.model['name']}] Grib files found "
                  f"({len(idx_files)}): {self.date.strftime('%Y%m%d-%Hh')}")
        else:
            print(f"⚠️ [{self.model['name']}] No files found after "
                  f"{max_retry} attempts for {self.model['name']}")
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
    def steps(self):
        """Get sorted array of available steps."""
        if self._steps is None:
            self._steps = np.unique(self.idx['step'].values)
        return self._steps

    @property
    def step(self):
        """Get the step at timestamp"""
        if self._step is None:
            h = int((self.timestamp - self.date).round('1h') / pd.Timedelta('1h'))
            self._step = self.steps[np.abs((self.steps - h)).argmin()]
        return self._step

    def download(self, clear_cache=False):
        """Download all subsets."""
        for subset in self._subsets.values():
            subset.download(clear_cache)

    def download_data(self, df, progress_bar=True, desc=None):
        return asyncio.run(self._download_data(df, progress_bar, desc))

    def to_nc(self):
        """Save each subset dataset to separate NetCDF files."""
        for subset in self._subsets.values():
            subset.to_nc()

    def to_grib_file(self):
        """Save each subset dataset to separate GRIB files."""
        for subset in self._subsets.values():
            subset.to_grib_file()

    def clear_cache(self):
        """Clean cache files."""
        for subset in self._subsets.values():
            subset.clear_cache()

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
            df['message_id'] = df['message_id'].astype(int)
            df['start_byte'] = df['start_byte'].astype(int)
            df['end_byte'] = df['start_byte'].shift(-1) - 1
            df.loc[df['step'] == 'anl', 'step'] = '0'
            df['step'] = df['step'].str.split(' ').str[0]

        df['step'] = df['step'].astype(int)
        df['url'] = self.get_file_url(idx_file)
        df['idx_file'] = idx_file

        return df

    async def _download_data(self, df, progress_bar=True, desc=None):
        """Download GRIB data asynchronously."""

        downloaded_data = []
        partitions = get_partitions(df)

        async with aiohttp.ClientSession(connector=self._connector) as session:
            tasks = []
            for partitions_id, idx_data in partitions.iterrows():
                headers = get_headers(**idx_data.to_dict())
                tasks.append(download_messages(session, idx_data['url'], headers, partitions_id))

            if progress_bar:
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
    parser.add_argument('--model', default='gfswave', help='Model to use (default: gfswave)')
    parser.add_argument('--timestamp', help='Timestamp for forecast (default: latest)')
    parser.add_argument('--no-cache', default=False, action='store_true', help='Disable cache')
    parser.add_argument('--save-nc', action='store_true', help='Save to NetCDF')

    args = parser.parse_args()

    g = Grib(model=args.model, timestamp=args.timestamp)
    g.download(clear_cache=args.no_cache)

    if args.save_nc:
        g.to_nc()

    print(g['wind'].ds)
