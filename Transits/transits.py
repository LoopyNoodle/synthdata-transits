import numpy as np
import pandas as pd
import random
import os
from pytransit import QuadraticModel
from glob import glob
from tqdm import tqdm
import multiprocessing

def calc_mean(t, T_range, flux):
    return np.mean(flux[np.abs(T_range - t) < 15/1440])

def gen_transits(file):
    df = pd.read_csv(file)
    index = os.path.splitext(os.path.basename(file))[0].split('_')[-1] # assuming files are named final_lcs_<ind> for ex
    time = df['time']
    s_flux = df['final_flux']

    # high-resolution time array
    # extending time range on both ends by 15 days and creating 100000 equally spaced points in this range
    T_range = np.linspace(time.iloc[0] - 15/1440, time.iloc[-1] + 15/1440, 100000)
    '''note: thispart caused error as time is a pandas series but we're treating it like a numpy 
    array. So add iloc for position based index with pandas'''

    transit_model = QuadraticModel() # object that creates transit curve mased on quadratic limb-darkening law
    transit_model.set_data(T_range)

    for j in range(10):
            # transit params: info from https://pytransit.readthedocs.io/en/latest/index.html
            
            k = np.exp(np.random.uniform(np.log(0.1), np.log(0.3)))  # planet-star radius ratio
            p = 1/np.exp(np.random.uniform(np.log(1/15), np.log(14)))  # period
            t0 = np.random.uniform(0, p)  # the zero epoch
            a = np.random.uniform(1.6, 41)  # orbital semi-major axis divided by the stellar radius
            i = np.pi/2 # orbital inclination (rad)
            e = np.random.uniform(0, 0.6) # orbital eccentricity
            w = np.random.uniform(0, 2*np.pi) # argument of periastron (rad)
            ldc = np.array([0.3, 0.1]) # limb darkening model coefficients

            flux = transit_model.evaluate(k, ldc, t0, p, a, i, e, w)
            downsamp_f = np.array([calc_mean(t, T_range, flux) for t in time])
            in_transit_count = np.sum(downsamp_f < 1) # how many times we see transit

            # logging metadata
            with open('transit_params.csv', 'a') as f:
                 f.write(f'{index},{j},{k},{t0},{p},{a},{i},{e},{w},{in_transit_count}\n')

            out_path_transit = os.path.join(out_dir, f'transit_{index}_{j}.npz')

            np.savez(out_path_transit,
                    filename = str(file),
                    k = k,
                    ldc = ldc, 
                    t0 = t0, 
                    p = p, 
                    a = a,
                    inclination = i,
                    eccentricity = e, 
                    omega = w, 
                    times = time,
                    model_flux = downsamp_f * s_flux,
                    in_transit_count = in_transit_count)
            print(f'\nSaved {out_path_transit}')

    # also saving the non-transit versions
    out_path_notransit = os.path.join(out_dir, f'notransit_{index}.npz')
    np.savez(out_path_notransit,
            filename = str(file),
            times = time, 
            model_flux = s_flux)
    print(f'\nSaved {out_path_notransit}')

    return f'\nProcessed {file}'

in_dir = 'final_lcs'
out_dir = 'valdata_deeptransit_nooutliers_lowamp'
os.makedirs(out_dir, exist_ok = True)

files = glob(f'{in_dir}/*.csv')
print(f'Total Files Found: {len(files)}')
random.shuffle(files)

# limiting the files
files = files[:min(300000, len(files))]

if __name__ == '__main__':
    num_cores = 3

    # log header
    with open('transit_params.csv', 'w') as f:
         f.write('index, j, k, t0, p, a, i, e, w, in_transit_count\n')

    # parallel processing
    with multiprocessing.Pool(num_cores) as pool:
        results = list(tqdm(pool.imap_unordered(gen_transits, files), total = len(files)))
    for i in results:
        if 'Error' in i:
            print(i)