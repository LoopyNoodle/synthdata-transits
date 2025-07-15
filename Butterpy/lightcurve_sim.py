import butterpy as bp
import matplotlib.pyplot as plt
import numpy as np
import os

noise_dir = 'tess_noise'
out_dir = 'final_lcs'
os.makedirs(out_dir, exist_ok = True)

noise_files = sorted([f for f in os.listdir(noise_dir) if f.endswith('.csv')])
N = len(noise_files)

# os.makedirs('simulated_lcs', exist_ok = True)

# def gen_lc(seed= None):
#     if seed is not None:
#         np.random.seed(seed)
    
#     star = bp.Surface()
    
#     period = np.random.uniform(0.1, 180)  # days
#     activity_level = 10**np.random.uniform(-1, 1)  # log-uniform between 0.1 and 10
#     inclination = np.degrees(np.arcsin(np.sqrt(np.random.uniform(0, 1))))
#     tau_evol = np.random.uniform(5, 20)  # spot lifetime scale
#     shear = np.random.uniform(-1.0, 1.0)  # differential rotation

#     min_lat = np.random.uniform(10, 30)
#     max_lat = np.random.uniform(min_lat + 5, 50)

#     # emergence active regions -> butterfly like pattern
#     star.emerge_regions(butterfly = True,
#         activity_level = activity_level,
#         cycle_period = np.random.uniform(5, 30),
#         cycle_overlap = np.random.uniform(0.1, 5),
#         max_lat = max_lat,
#         min_lat = min_lat)

#     # time - close to 30 min cadence like TESS
#     t = np.arange(0, 365, 0.0208)

#     # simulating lightcurve
#     lc = star.evolve_spots(t,
#         alpha_med = 1e-4,
#         period = period,
#         inclination = inclination,
#         tau_evol = tau_evol,
#         shear = shear)

#     return lc

# # saving the data
# from astropy.table import Table

# def save(n):
#     for i in range(n):
#         lc = gen_lc(seed = i)
#         table = Table(data = [lc.time, lc.flux], names = ['time', 'flux'])
#         f_name = f'simulated_lcs/sim_lc_{i:03d}.csv'
#         table.write(f_name, format = 'csv', overwrite = True)
#         print(f'Saved {f_name}')

# save(120)

import pandas as pd

for i, f_name in enumerate(noise_files[:N]):
    path = os.path.join(noise_dir, f_name)
    out_path = os.path.join(out_dir, f'final_lc_{i:03d}.csv')

    '''
    Extracting flux values and timestamps from noise samples separately
    '''

    try:
        df = pd.read_csv(path)
        t = df['time'].values
        if 'nflux_dtr' not in df.columns:
            continue
        nflux = df['nflux_dtr'].values
    except:
        continue

    '''
    Generating lightcurves using Butterpy at the timestamps from the TESS noise samples generated using tessilator.py.
    Instead of using regression-corrected flux (reg_oflux), we're using detrended and normalised flux (nflux_dtr)
    '''

    np.random.seed(i)
    star = bp.Surface()
    star.emerge_regions(butterfly = True,
    activity_level = 10**np.random.uniform(-1, 1),
    cycle_period = np.random.uniform(5, 30),
    cycle_overlap = np.random.uniform(0.1, 5),
    max_lat = np.random.uniform(31, 50),
    min_lat = np.random.uniform(10, 30)
    )

    period = np.random.uniform(0.1, 180)
    inclination = np.degrees(np.arcsin(np.sqrt(np.random.uniform(0, 1))))
    tau_evol= np.random.uniform(5, 20)
    shear = np.random.uniform(-1, 1)

    lc = star.evolve_spots(t,
    alpha_med= 1e-4,
    period=period,
    inclination = inclination,
    tau_evol = tau_evol,
    shear = shear
    )

    '''
    Convolving the simulated lightcurves with the TESS noise samples to obtain final noisy lightcurves
    '''
    fin_flux = lc.flux * nflux

    fin_df = pd.DataFrame({'time': t,
    'nflux_dtr': nflux,
    'sim_flux': lc.flux,
    'final_flux': fin_flux,
    'period': period})

    fin_df.to_csv(out_path, index = False)
    print(f'\nSaved {out_path}')

sample = os.path.join(out_dir, 'final_lc_004.csv')

df = pd.read_csv(sample)
time = df['time']

fig, axs = plt.subplots(3, 1, figsize = (10, 6), sharex = True)
axs[0].scatter(time, df['final_flux'], s = 1, color = 'teal')
axs[0].set_ylabel('Flux')
axs[0].set_title('Final Lightcurve with TESS Noise')
axs[0].grid(True)

axs[1].scatter(time, df['sim_flux'], s = 1, color = 'darkviolet')
axs[1].set_ylabel('Flux')
axs[1].set_title('Butterpy Synthetic Lightcurve')
axs[1].grid(True)

axs[2].scatter(time, df['nflux_dtr'], s = 1, color = 'plum')
axs[2].set_xlabel('Time (days)')
axs[2].set_ylabel('Detrended Noise Flux')
axs[2].set_title('TESS Noise Sample')
axs[2].grid(True)
plt.show()