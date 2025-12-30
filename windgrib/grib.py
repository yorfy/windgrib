"""GRIB data download and processing module for weather data."""

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from requests import get
from tqdm import tqdm
from tqdm.dask import TqdmCallback

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
            'wind': {'param': ['10u', '10v']},
            'land': {'param': ['lsm'], 'step': 0}
        },
        'var_mapping': {'u10': 'u', 'v10': 'v'},
        'filter_key': 'shortName'
    }
}


class GribSubset:
    """Handles individual subset operations for GRIB data."""

    def __init__(self, name, config, grib_instance):
        """Initialize subset with name, configuration and parent grib instance."""
        self.name = name
        self.config = config
        self.grib = grib_instance
        self._ds = None
        self._has_new_steps = False

    @property
    def grib_file(self):
        """Get GRIB file path for this subset."""
        return (self.grib.folder_path /
                f"{self.name}_{self.grib.model['name']}.{self.grib.model['product']}.grib2")

    @property
    def grib_ls(self):
        """Get GRIB list file path for this subset."""
        return self.grib_file.with_suffix('.grib2.ls')

    @property
    def grib_idx(self):
        """Get GRIB index file path for this subset."""
        return self.grib_file.with_suffix('.grib2.idx')

    @property
    def nc_file(self):
        """Get NetCDF file path for this subset."""
        return self.grib_file.with_suffix('.nc')

    def messages_df(self, idx_file):
        """Parse index file and return DataFrame of messages."""
        url = self.grib.url
        if idx_file.endswith('.index'):
            idx_txt = get(url + idx_file, timeout=30).text.rstrip('\n ').replace('\n', ',')
            idx_txt = '[' + idx_txt + ']'
            df = pd.DataFrame(json.loads(idx_txt))
            df['message_id'] = df.index
            df['start_byte'] = df['_offset']
            df['end_byte'] = df['start_byte'] + df['_length'] - 1
        else:
            idx_txt = get(url + idx_file, timeout=30).text.split('\n')
            df = pd.DataFrame(
                [row.split(':') for row in idx_txt if row],
                columns=[
                    'message_id', 'start_byte', 'reference_time',
                    'variable', 'layer', 'forecast_time', '?'
                ])
            df['message_id'] = df['message_id'].astype(int)
            df['start_byte'] = df['start_byte'].astype(int)
            df['end_byte'] = df['start_byte'].shift(-1) - 1
        return df

    def download_file(self, idx_file):
        """Download GRIB file based on index file for this subset."""
        df = self.messages_df(idx_file)

        # filter df with subset variables
        var = list(self.config.keys())[0]
        values = self.config[var]
        df = df[df[var].isin(values)]

        if df.empty:
            return

        df['download_groups'] = df['message_id'].diff().ne(1).cumsum()
        for _, group in df.groupby('download_groups'):
            start_byte = group['start_byte'].min()
            end_byte = group['end_byte'].max()
            end_byte = "" if group['end_byte'].isna().any() else int(end_byte)
            headers = {'Range': f"bytes={start_byte}-{end_byte}"}
            url = self.grib.url + idx_file.replace(self.grib.model.get('idx', '.idx'), '')
            if '.grib2' not in url:
                url += '.grib2'
            r = get(url, headers=headers, timeout=30)
            if r.content:
                with open(self.grib_file, 'ab', encoding=None) as f:
                    f.write(r.content)

        with open(self.grib_ls, 'a', encoding='utf-8') as f:
            f.write(idx_file + '\n')

    def download(self, use_cache=True):
        """Download GRIB files for this subset."""
        print(f"Downloading subset: {self.name} - {self.config}")

        if not use_cache:
            self.grib_file.unlink(missing_ok=True)
            self.grib_ls.unlink(missing_ok=True)

        ls = self._get_existing_files()
        idx_files = self._get_files_to_download(ls)
        downloaded_count = self._execute_downloads(idx_files)

        return downloaded_count

    def _get_existing_files(self):
        """Get list of existing files."""
        if self.grib_ls.exists():
            with open(self.grib_ls, 'r', encoding='utf-8') as f:
                return f.read().splitlines()
        return []

    def _get_files_to_download(self, ls):
        """Get list of files that need to be downloaded."""
        idx_files = self.grib.idx_files

        # Handle step filtering
        if 'step' in self.config and self.config['step'] == 0:
            idx_files = [sorted(idx_files)[0]] if idx_files else []

        idx_files = [idx_file for idx_file in idx_files if idx_file not in ls]

        if len(idx_files) == 0:
            print(f'all available grib files for {self.name} are already downloaded')

        return idx_files

    def _execute_downloads(self, idx_files):
        """Execute the actual downloads."""
        executor = ThreadPoolExecutor(max_workers=100)
        download_tasks = [executor.submit(self.download_file, idx_file)
                          for idx_file in idx_files]

        desc = f'Downloading {self.name} grib files'
        with tqdm(total=len(download_tasks), desc=desc) as progress_bar:
            for _ in as_completed(download_tasks):
                progress_bar.update(1)

        executor.shutdown(wait=True)
        return len(idx_files)

    @property
    def ds(self):
        """dataset accessor"""
        if self._ds is None:
            self._ds = self.load_dataset()
        return self._ds

    def load_dataset(self):
        """loading dataset from cache files"""
        ds = None
        if self.nc_file.exists():
            ds = xr.open_dataset(self.nc_file, decode_timedelta=True)
            with open(self.grib_ls, "r", encoding='utf-8') as f:
                n_grids = len(f.read().splitlines())
            if ds is not None and 'step' in ds.dims and n_grids > len(ds.step):
                gribs = self.load_grib_file()
                new_steps = gribs.step.to_index().difference(ds.step.to_index())
                self._has_new_steps = not new_steps.empty
                if self._has_new_steps:
                    ds = xr.combine_nested([ds, gribs.sel(step=new_steps)],
                                           concat_dim='step', join='outer')
        if ds is None:
            self._has_new_steps = True
            ds = self.load_grib_file()

        if ds is None:
            print('fuck')
        if 'step' in ds.dims:
            ds = ds.sortby('step')

        ds = ds.sortby('latitude')
        return ds

    def load_grib_file(self):
        """Load dataset for this subset."""

        if not self.grib_file.exists():
            print(f"Warning: {self.name} file not found: {self.grib_file}")
            return None

        if 'filter_key' in self.grib.model:
            ds = self._load_grib_with_filter_key()
        else:
            ds = self._load_grib()

        if 'var_mapping' in self.grib.model:
            # Only rename variables that exist in the dataset
            rename_dict = {old: new for old, new in self.grib.model['var_mapping'].items()
                           if old in ds.data_vars}
            if rename_dict:
                ds = ds.rename(rename_dict)
                print(f"Renamed variables : {rename_dict}")

        if ds is not None:
            print(f"Loaded {self.name} from GRIB: {list(ds.data_vars)}")
        else:
            print(f"Failed to load {self.name}")

        return ds

    def _load_grib_with_filter_key(self):
        """Load dataset with filtering."""
        filter_key = self.grib.model['filter_key']
        datasets = []

        for key, value in self.config.items():
            if key == 'step':
                continue

            for var_name in value:
                print(f"  Loading variable: {var_name}")
                ds_var = xr.open_dataset(
                    self.grib_file,
                    engine='cfgrib',
                    decode_timedelta=True,
                    backend_kwargs={
                        'errors': 'ignore',
                        'filter_by_keys': {filter_key: var_name}
                    }, chunks={'step': 1, 'latitude': -1, 'longitude': -1}
                )

                if not ds_var.data_vars:
                    print(f"    No variables found for {var_name}")
                    continue

                print(f"    Successfully loaded {var_name}: {list(ds_var.data_vars)}")
                datasets.append(ds_var)

        if len(datasets) > 1:
            return xr.merge(datasets, compat='override', join='outer')
        if len(datasets) == 1:
            return datasets[0]
        return None

    def _load_grib(self):
        """Load dataset without filtering."""
        return xr.open_dataset(
            self.grib_file,
            engine='cfgrib',
            decode_timedelta=True,
            backend_kwargs={'errors': 'ignore'},
            chunks={'step': 1, 'latitude': -1, 'longitude': -1}
        )

    def to_nc(self):
        """Save dataset to NetCDF file."""

        if self.nc_file.exists() and not self._has_new_steps:
            print(f"NetCDF file already exists and up to date: {self.nc_file}")
            return

        ds = self.ds

        with TqdmCallback(desc=f'loading {self.name} dataset'):
            ds.load()

        encoding = {var: {'dtype': 'int16', 'scale_factor': 0.01,
                          '_FillValue': np.iinfo('int16').max, 'zlib': False}
                    for var in ds.data_vars}

        with TqdmCallback(desc=f'saving {self.name} as netcdf file'):
            ds.to_netcdf(self.nc_file, encoding=encoding)

        print(f"Saved {self.name} to: {self.nc_file}")


class Grib:
    """GRIB data downloader and processor for weather models."""

    def __init__(self, time=None, model='gfswave',
                 data_path=None, max_retry=10):
        """Initialize GRIB downloader."""
        self.max_retry = max_retry

        if isinstance(model, str):
            model = model.lower()
            self.model = MODELS[model]
            self.model['name'] = model
        else:
            self.model = model

        # Initialize data_path
        self.data_path = data_path
        if self.data_path is None:
            self.data_path = Path(__file__).parent.parent / 'data' / 'grib'
        else:
            self.data_path = Path(data_path)

        time = pd.Timestamp(time) if time is not None else pd.Timestamp.utcnow()
        h = time.hour - time.hour % 6
        self.date = time.strftime("%Y%m%d")
        self.h = str(h)

        self._retry_count = 0
        self.find_forecast_time(time)

        # Initialize subsets dictionary
        self._subsets = {}
        for name, config in self.model['subsets'].items():
            self._subsets[name] = GribSubset(name, config, self)

    @property
    def url(self):
        """Get URL for the model data."""
        h = int(self.h)
        return self.model['url'] + self.model['key'].format(date=self.date, h=h)

    @property
    def s3(self):
        """Get S3 path for the model data."""
        return (self.url.replace('https', 's3')
                .replace('.s3', '')
                .replace('.amazonaws.com', '')
                .replace('.eu-central-1', ''))

    @property
    def folder_path(self):
        """Get local folder path for storing data."""
        return self.data_path / self.date / self.h

    @property
    def subset_names(self):
        """Get list of subset names to process."""
        return list(self.model['subsets'].keys())

    def find_forecast_time(self, time):
        """Initialize files and handle retry logic."""
        idx_files = self.idx_files
        if idx_files:
            print(f"Found grib files to download at: {time}")
            print(f"Using data path: {self.data_path}")
            self.folder_path.mkdir(parents=True, exist_ok=True)
            return

        self._retry_count += 1
        if self._retry_count > self.max_retry:
            raise ValueError(f"No files found after 10 attempts for {self.model['name']}")

        print(f"Grib files not found at: {time}. Looking 6h before.")
        time -= pd.Timedelta(6, 'h')
        new_instance = Grib(time, model=self.model, data_path=self.data_path)

        self.__dict__.update(new_instance.__dict__)

    @property
    def idx_files(self):
        """Get list of available index files."""
        files_pattern = self.s3 + '*'
        if 'product' in self.model:
            product = self.model['product']
            files_pattern += f'{product}*'
        files_pattern += self.model.get('idx', '.idx')
        idx_files = s3fs.S3FileSystem(anon=True).glob(files_pattern)
        return [f.split('/')[-1] for f in idx_files]

    def download(self, use_cache=True):
        """Download GRIB files for subsets."""
        total_downloaded = 0

        for subset in self._subsets.values():
            total_downloaded += subset.download(use_cache)

        return total_downloaded

    def __getitem__(self, subset_name):
        """Get dataset for a specific subset using bracket notation."""
        return self._subsets[subset_name].ds

    def to_nc(self):
        """Save each subset dataset to separate NetCDF files."""
        for name in self.subset_names:
            self._subsets[name].to_nc()
