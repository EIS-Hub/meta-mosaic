import os
import urllib.request
import gzip, shutil
import hashlib
import h5py
import jax.random as random
from six.moves.urllib.error import HTTPError 
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlretrieve
import jax

import numpy as np
import jax.numpy as jnp
from functools import partial

# Code to retrieve and pre-process the Spiking Heidelberg Digits dataset
"""
Taken from 
    - https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
    - https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py
"""

def get_audio_dataset(cache_dir, cache_subdir):
    base_url = "https://zenkelab.org/datasets"
    response = urllib.request.urlopen("%s/md5sums.txt"%base_url)
    data = response.read() 
    lines = data.decode('utf-8').split("\n")
    file_hashes = { line.split()[1]:line.split()[0] \
                    for line in lines if len(line.split())==2 }
    files = [ "shd_train.h5.gz", "shd_test.h5.gz"]
        
    for fn in files:
        origin = "%s/%s"%(base_url,fn)
        hdf5_file_path = get_and_gunzip(origin, 
                                        fn, 
                                        md5hash=file_hashes[fn],
                                        cache_dir=cache_dir,
                                        cache_subdir=cache_subdir)
        print("Available at: %s"%hdf5_file_path)

def get_and_gunzip(origin, filename, md5hash=None, cache_dir=None, 
                   cache_subdir=None):
    gz_file_path = get_file(filename, origin, md5_hash=md5hash,
                            cache_dir=cache_dir, cache_subdir=cache_subdir)
    hdf5_file_path = gz_file_path[:-3]
    if not os.path.isfile(hdf5_file_path) or os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path):
        print("Decompressing %s"%gz_file_path)
        with gzip.open(gz_file_path, 'r') as f_in, open(hdf5_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return hdf5_file_path

def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(file_hash) == 64):
        hasher = 'sha256'
    else:
        hasher = 'md5'
    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False

def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()

def get_file(fname,
             origin,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.data-cache')
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.data-cache')
    datadir = os.path.join(datadir_base, cache_subdir)

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(datadir, exist_ok=True)

    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated because the ' + hash_algorithm +
                      ' file hash does not match the original value of ' + file_hash +
                      ' so we will re-download the data.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)

    return fpath

def get_h5py_files():
    cache_dir = os.path.expanduser("./content/")
    cache_subdir = "audiospikes"
    get_audio_dataset(cache_dir, cache_subdir)
    train_shd_file = h5py.File(os.path.join(cache_dir, cache_subdir, 'shd_train.h5'), 'r') #r
    test_shd_file  = h5py.File(os.path.join(cache_dir, cache_subdir, 'shd_test.h5'), 'r')
    return train_shd_file, test_shd_file

def preprocess_h5py_files(h5py_file):
    nb_steps = 100
    nb_units = 700
    max_time = 1.4 
    num_samples = h5py_file['spikes']['times'].shape[0]

    firing_times = h5py_file['spikes']['times']
    units_fired  = h5py_file['spikes']['units']    
    labels       = h5py_file['labels']

    time_bins = np.linspace(0, max_time, num=nb_steps)
    input  = np.zeros((num_samples, nb_steps, nb_units), dtype=np.uint8)
    output = np.array(labels, dtype=np.uint8)

    for idx in range(num_samples):
        times = np.digitize(firing_times[idx], time_bins)
        units = units_fired[idx] 
        input[idx, times, units] = 1

    return input, output

'''Code to generate the mask for the mosaic architecture'''

#Code to prune the weights
@partial(jax.jit, static_argnums=(1,2))
def pruner(train_step, prune_start_step, prune_thr, weight):
    weight = jax.lax.cond(train_step > prune_start_step, 
                          lambda weight: jnp.where(jnp.abs(weight) < prune_thr, 0., weight),
                          lambda weight: weight, 
                          weight)

    weight = jnp.clip(weight, -2.5*jnp.std(weight), 2.5*jnp.std(weight))
    return weight

def sparser(W_in, W_mask):
    L = jnp.mean(W_mask * W_in**2)
    return L

#Code to generate the mask for the mosaic architecture
def generate_mosaic_mask(n_rec=256, n_core=16, beta=0.4):
    num_cols = jnp.sqrt(n_core)
    num_hops = jnp.zeros((n_core, n_core))

    for i in range(n_core): #0,1,2 -> 1, 5, 7
        for j in range(i+1, n_core):
            i_col_id = i % num_cols
            i_row_id = i // num_cols
            j_col_id = j % num_cols
            j_row_id = j // num_cols

            x_hop = 0
            y_hop = 0
            # COLUMNS
            if i_col_id == j_col_id and jnp.abs(i_row_id - j_row_id) == 1:
                x_hop = 0
                y_hop = 1
            if i_col_id == j_col_id and jnp.abs(i_row_id - j_row_id) > 1:
                x_hop = 1
                y_hop = 2 * jnp.abs(i_row_id - j_row_id)
            if i_col_id != j_col_id and jnp.abs(i_col_id - j_col_id) == 1:
                x_hop = 1
                y_hop = 2 * jnp.abs(i_row_id - j_row_id)
            #Far away 
            if i_col_id != j_col_id and jnp.abs(i_col_id - j_col_id) > 1:
                x_hop = 2 * jnp.abs(i_col_id - j_col_id) - 1
                y_hop = 2 * jnp.abs(i_row_id - j_row_id)

            # ROWS
            if i_row_id == j_row_id and jnp.abs(i_col_id - j_col_id) == 1:
                x_hop = 1
                y_hop = 0
            if i_row_id == j_row_id and jnp.abs(i_col_id - j_col_id) > 1:
                x_hop = 1
                y_hop = 2 * jnp.abs(i_col_id - j_col_id)
            if i_row_id != j_row_id  and jnp.abs(i_row_id - j_row_id) == 1:
                x_hop = 2 * jnp.abs(i_col_id - j_col_id)
                y_hop = 1

            num_hops = num_hops.at[i,j].set(x_hop + y_hop)
            num_hops = num_hops.at[j,i].set(x_hop + y_hop)


    weight_mask = jnp.clip(jnp.exp(beta * jnp.kron(num_hops, jnp.ones((n_rec//n_core, n_rec//n_core))))-1, 0, jnp.exp(15))
    return weight_mask

def calc_mosaic_stats(W_example, mask, beta):
    MAX_NUM_HOPS = int(jnp.max(jnp.log(mask)/beta))
    num_total_memristors_per_hop, _ = jnp.histogram(mask, jnp.exp(beta*jnp.arange(MAX_NUM_HOPS+2)))
    num_non_zero_weights_per_hop = jnp.array([jnp.sum(jnp.abs(W_example[jnp.where(mask==jnp.exp(beta*i))])>0) for i in range(MAX_NUM_HOPS+1)])
    percent_occupancy = 100 * num_non_zero_weights_per_hop / num_total_memristors_per_hop
    return jnp.nan_to_num(percent_occupancy)

#Code to add noise to the weights
@jax.custom_jvp
def add_noise(weights, key, noise_std):
    ''' Adds noise only for inference
    '''
    noise_added_weights = []
    for weight in weights:
        weight = jnp.where(weight != 0.0, 
                          weight + random.normal(key, weight.shape) * jnp.max(weight) * noise_std, 
                          weight)

        weight = jnp.clip(weight, -1, 1)
        noise_added_weights.append(weight)
    return noise_added_weights
 
@add_noise.defjvp
def add_noise_jvp(primals, tangents):
    weight, key, noise_std = primals
    x_dot, y_dot, z_dot = tangents
    primal_out = add_noise(weight, key, noise_std)
    tangent_out = x_dot
    return primal_out, tangent_out
