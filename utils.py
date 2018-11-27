import io
import os
import tarfile
import zipfile

import requests
from tqdm import tqdm


def download(url, save_filepath, extract_dir=None, file_type=''):
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length'))
    block_size = 256*1024
    pbar = tqdm(r.iter_content(chunk_size=block_size),
                total=total_size,
                unit='B', unit_scale=True)
    with open(save_filepath, 'wb') as f:
        for chunk in pbar:
            f.write(chunk)
            f.flush()
            pbar.update(block_size)
            pbar.refresh()

        f.seek(0, 0)
        if file_type == '':
            z = None
        elif file_type == 'zip':
            z = zipfile.ZipFile(f)

        elif file_type == 'tar':
            z = tarfile.open(fileobj=f)

        else:
            raise ValueError("file type not supported")

        if z and extract_dir:
            z.extractall(extract_dir)
            z.close()
