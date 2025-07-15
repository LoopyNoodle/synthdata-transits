import numpy as np
import os
import argparse
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
import sys
import datetime
import json

from urllib.parse import quote as urlencode
from urllib.parse import urlparse
import http.client as httplib
import scipy.optimize as opt
import base64

#################fixedconstants.py

'''

Alexander Binks & Moritz Guenther, 2024

Licence: MIT 2024

Module containing fixed constants for the tessilator

These are the pixel size, the typical full-width half maximum of the pixel
response function, the TESS zeropoints and the latest TESS sector available
for download.
'''

pixel_size=21.0
'''This is the pixel size for TESS (in arcseconds).

This is held constant. Do not change.
'''


exprf=0.65
'''This is the full-width half maximum of a TESS pixel.

This is held constant. Do not change.
'''


Zpt, eZpt = 20.44, 0.05
'''This is the zero-point TESS magnitude calculated in Vanderspek et al. 2018.

This is held constant. Do not change.
'''

sec_max = 89
'''The maximum sector number to be acquisitioned when looking for TESS data

This will change over time as more data is collected.
'''


########################acf_functions.py

# imports
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# UNIVERSAL VARIABLES

# # data points per day at even cadence (=24*30 for TESS, 24*2 for Kepler)
# cad_tess = 24.*30.
# cad_kepler = 24.*2.

cadence = 30*60    # default cadence is 30 minutes (in units of seconds)

# number of seconds in a day
day = 24*60*60


# GENERAL HELPER FUNCTIONS
def days_to_bins(days, bs):
    """
    Converts a time array given in units of days into units of bin number.

    Args:
        days (:obj:`array`): time array in units of days.
        bs (:obj:`float`): the cadence to bin to, in units of seconds.

    Returns:
        :obj:`array`: the time array in units of bin number.
    """
    return days*24*60*60//bs

def bins_to_days(bins, bs):
    """
    Converts a time array given in units of bin number into units of days.

    Args:
        bins (:obj:`array`): time array in units of bin number or (in the case of the acf) lag number.
        bs (:obj:`float`): the size of the bins, in units of seconds.

    Returns:
        :obj:`array`: the time array in units of days.
    """
    return bins*bs/(24*60*60)


# NEW PROCESS HELPER FUNCTIONS


def calc_fft_pgi(corr, bs=cadence):    
    """
    Uses a fast fourier transform to identify an initial guess (pgi) for the dominant periodicity in the autocorrelation function. This is the default pgi-finding function for SpinSpotter.

    Args:
        corr (:obj:`arr`): an autocorrelation function.
        bs (:obj:`float`): the size of the bins, in units of seconds.

    Returns:
    Two parameters.

        - pgi (:obj:`int`): the index in the acf of the initial period guess in units of lags.
        - results (:obj:`array`): dictionary of other relevant paramters from the pgi-finding code.
    """
    results = {}

    # Calculate the fft
    coeff = 10
    corr_fft = np.real(np.fft.rfft(corr,n=coeff*len(corr)))
    fft_period = np.divide(1., np.fft.rfftfreq(coeff*len(corr))[1:])

    # we're only interested in the positive half of the fft
    snip = np.where((fft_period >= 0) & (fft_period <= len(corr)))
    fft_period = fft_period[snip]
    corr_fft = corr_fft[snip]
    
    # add some results to the results dictionary
    results['fft'] = corr_fft
    results['fft_period'] = fft_period  # period array, in units of index of the acf
    results['fft_period_days'] = np.divide(fft_period, (day/bs))  # convert the periods to days
    # results['pgi'] = pgi

    # add the prominence of the fft peak
    pgi, fft_results = fft_find_peaks(results, plot=False)
    results.update(fft_results) 
    
    # if the pgi is more than half the period, automatic fail
    if pgi < 0 :
        results['fail'] = True
        return np.nan, results
    elif fft_period[pgi] > len(corr) / 2:
        results['fail'] = True
        return np.nan, results
    
    return pgi, results

def fft_find_peaks(result, plot=False):
    """
    Helper function used by `calc_fft_pgi` to calculate the prominence of the max 
    peak in the fft.Does this by finding the five tallest peaks, then calculates the standard 
    deviation of the tallest one from the other four. setting plot to True will plot the FFT and
    the five found peaks. Currently, returns the number of standard deviations from
    the mean that the highest peak is.

    Args:
        result (:obj:`dict`):the result dictionary as outputed by calc_fft_pgi(), which contains the fft.
        plot (:obj:`bool`): if True, will plot the FFT.

    Returns:
        results (:obj:`dict`): the updated result dictionary, which now contains info on the pgi and peaks in the fft.
    """
    output = result.copy()
    corr_fft = np.real(result['fft'])
    fft_period = result['fft_period']  # period array, in units of index of the acf

    # find the five tallest peaks, then see how much taller the tallest is
    fft_peaks, fft_properties = find_peaks(corr_fft)
    peak_heights = corr_fft[fft_peaks]

    #initialize the matrix to store the indices of the five peaks
    max_peaks = np.zeros(6,dtype=int)

    # if the max peak is too small, fail the test and exit
    # currently set to cut off when the max peak has index less than 5
    if fft_peaks[np.argmax(peak_heights)] < 5 :
        # print( "period is too short")
        return -1, {'pgi':-1, 'fft_pgi':-1}
    
    # retrieve the peaks
    for i,val in enumerate(max_peaks) :
        index = np.argmax(peak_heights)
        max_peaks[i] = int(fft_peaks[index])
        peak_heights[index] = 0
    
    # add the max_peaks and peak_heights to the results dictionary
    output['max_peaks'] = max_peaks
    output['peak_heights'] = peak_heights
    
    # the pgi is the index of the largest peak
    # pgi = int(np.round(fft_period[max_peaks[0]]))
    fft_pgi = max_peaks[0]  # index of the period guess in the fft
    acf_pgi = int(np.round(fft_period[max_peaks[0]]))  # index of the period guess in the acf
    
    # calculate the mean and std of all but the largest peak
    mean_mp = np.mean(corr_fft[max_peaks[1:]])
    std_mp = np.std(corr_fft[max_peaks[1:]])

    # calc how many stds the max peak is from the rest
    pgi_prom = (corr_fft[max_peaks[0]] - mean_mp) / std_mp

    # add the pgi and peak prominence to the result dict
    output['fft_pgi'] = fft_pgi
    output['pgi'] = acf_pgi
    output['pgi_prom'] = pgi_prom
    
    # if requested, then plot
    if plot :
        plt.figure(figsize=[12,5])
        plt.plot(fft_period,corr_fft)
        plt.plot(fft_period[max_peaks],corr_fft[max_peaks], marker='x',linestyle='')

    return acf_pgi, output



def gaussian(fwhm):
    """
    Creates a gaussian with a given FWHM as preparation for convolution with a light curve.

    Args:
        fwhm (:obj:`float`): the full width half max of the desired gaussian.
        bs (:obj:`float`): the size of the bins, in units of seconds.

    Returns:
        gaussian (:obj:`arr`): a gaussian.
    """
    sigma = fwhm / 2.355
    x = np.arange(-3*sigma, 3*sigma)
    # note that we divide by .997 to preserve the normalization and make the
    # area under the truncated gaussian equal to 1
    return 1./.997 * 1./(np.sqrt(2.*np.pi) * sigma) * np.exp(-(x/sigma)**2./2.)

def parabola(x,a,b,c):
    """
    Creates a parabola of the form y = ax^2 + bx + c. Intended to be used by curvefit in the process Test.

    Args:
        x (:obj:`arr`): the x range of the function.
        a, b, c (:obj:`float`): coefficients

    Returns:
        y (:obj:`arr`): a parabola.
    """
    return a*np.square(x) + b*x + c

def curvefit_peak(func,corr,pgi,peak_num,plot=False):
    """
    Bins a timeseries to the desired cadence. Works much faster than Lightkurve's built in binning function.

    Args:
        func (:obj:`func`): the function to be fitted to peaks in the ACF, intended to be the `parabola()` function.
        corr (:obj:`1D array`): the autocorrelation function.
        pgi (:obj:`int`): period_guess_index, i.e. the intial guess for the index location of the first peak in the `corr` array (aka the acf), usually identified from the highest peak in the FFT of the ACF.
        peak_num (:obj:`int`): which alias you are trying to fit (1 being the original peak, 2 being the first alias.)
        plot (:obj:`bool`, optional): if True, will plot the ACF and the fitted parabolas.

    Returns:
    Two parameters, or three if a flux_err is also provided.

        - fit_params (:obj:`array`): [a0,b0,c0], the coeffitients of the fitted parabola.
        - R_adj (:obj:`float`): the adjusted R^2 value of the parabola fit.
        - fitted_parabola (:obj:`array`): y-values of the fitted parabola.
        - hwhm (:obj:`float`): the half-width half-max of the fitted parabola. Used in error calculations.
    """
    # clip the acf to a length of a period guess, centered on the period guess
    k_snip = np.arange(pgi*peak_num - pgi/4, pgi*peak_num + pgi/4).astype(int)
    corr_snip = corr[k_snip]
    x_snip = np.arange(len(k_snip)).astype(int)  # synthetic x-axis
    
    # do a regression to fit the function
    # make an array for the weights
    # gauss = 1.0 - gaussian(len(x_snip)/2.4)
    # start = (len(gauss)-len(x_snip))/2
    # weights = gauss[start:(start+len(x_snip))]
    # params_opt stands for optimal parameters
    [a0,b0,c0], pcov = curve_fit(func, x_snip, corr_snip)
    # [a0,b0,c0], pcov = curve_fit(func, x_snip, corr_snip, sigma=weights, absolute_sigma=True)

    # calculate the expected curve from the fit parameters
    # note the the output is in lags, need to do conversions to get period in days again
    fitted_parabola = func(x_snip, *[a0,b0,c0])

    # calculate the other params you want
    # note correction to P_rot0 to make up for the shift in the window that was fit
    P_rot0 = -1.*b0 / (2*a0) + 3*pgi/4.   # note that P_rot0 is in lags, NOT time units
    A0 = c0 - np.square(b0) / (4*a0)
    
    # B0 = -1. * a0 * np.square(P_rot0)
    # new value of B0 is the width of the parabola at the zero crossing    
    # to avoid warning, check if there is an intercept    
    if (b0**2 - 4*a0*c0) < 0 :
        intercept1 = np.nan 
        intercept2 = np.nan
    else :
        intercept1 = (-b0 + np.sqrt(b0**2 - 4*a0*c0)) / (2 * a0)
        intercept2 = (-b0 - np.sqrt(b0**2 - 4*a0*c0)) / (2 * a0)
    B0 = np.abs((intercept1 - intercept2) / P_rot0)
    
    # gather things up
    fit_params = [a0,b0,c0]
    peak_params = [P_rot0, A0, B0]

    # check how good the fit is with the adjusted R^2 statistic
    R_adj = adjusted_R_sq(corr_snip, fitted_parabola)

    # calculate the half-width half-max of the peak
    if np.isnan(intercept1) or np.isnan(intercept2) :
        hwhm = np.nan
    else :
        x1 = (-b0 + np.sqrt(b0**2 - 4*a0*(c0-A0/2))) / (2 * a0)
        x2 = (-b0 - np.sqrt(b0**2 - 4*a0*(c0-A0/2))) / (2 * a0)
        hwhm = np.abs(x2-x1) / 2

    # plot if requested
    if plot:
        if peak_num == 1:
            plt.figure(figsize=[9,5])
            plt.plot(corr[:(len(corr))])
        plt.plot(k_snip,fitted_parabola)
        
    # # print the resulting params and R_adj if desired
    # if print_result:
    #     print( "Optimal params: " + str(params_opt))
    #     print( "Adjusted R^2 for curvefit: " + str(R_adj))

    return fit_params, peak_params, R_adj, fitted_parabola, hwhm

def adjusted_R_sq(obs,exp,num_param=3):
    """
    Calculates the adjusted R^2 statistic to estimate the success of a model. Formula implemented from equation 7.62 in Modern Statistical Methods for Astronomy by Feigelson and Babu.

    Args:
        obs (:obj:`float`): the observed value.
        exp (:obj:`float`): the expected value.
        num_param (:obj:`int`): the number of parameters used in the fit.

    Returns:
        R_adj (:obj:`float`): the adjusted R^2 statistic.
    """
    if len(obs) != len(exp):
        print( "The length of the observed and expected arrays should be equal.")
    n = len(obs)
    exp_mean = np.mean(obs) / float(n)

    # calculate the R^2 statistic first
    numerator = 0.
    denominator = 0.
    for i in range(n):
        numerator += np.square(obs[i] - exp[i])
        denominator += np.square(obs[i] - exp_mean)
    R_sq_inv = np.divide(numerator, denominator) # this is actually 1-R^2, not R^2

    # now calculate the adjusted R^2 to take into account the number of model parameters
    R_adj = 1. - (n - 1.)/(n - num_param) * R_sq_inv
    return R_adj


# # NEW PROCESS FUNCTIONS

def calc_acf(lc, bs=cadence, max_lag=None, smooth=None, sector_label=False):
    """
    Caculates the autocorrelation function (ACF) of a light curve..

    Args:
        lc (:obj:`LightCurve obj`): the cleaned lightcurve.
        bs (:obj:`float`): the size of the bins, in units of seconds.
        max_lag (:obj:`float`, optional): the maximum lag to which to calculate the ACF in units of days.
        smooth (:obj:`int`, optional): if supplied, will apply smoothing to the LC before fitting parabolas by convolving with a gaussian with a FWHM equal to this value.
        sector_label (:obj:`int` or :obj:`str`, optional): tallows you to set a custom sector label, must be castable to a string

    Returns:
        fits_result (:obj:`dict`): A dictionary containing information on the light curve and it's ACF.
    """
    fits_result = {'time_even':lc['time'], 'flux_even':lc['nflux_dtr'], 'flux_err_even':lc['nflux_err'],
       'acf_lags':np.array([]), 'acf':np.array([]), 'acf_smooth':np.array([])}

    # set the nlags
    if max_lag is None :
        nlags = days_to_bins(np.floor(lc['time'][-1] - lc['time'][0]), bs)
    else :
        nlags = days_to_bins(max_lag, bs)

    # calculate acf
    acf_corr = acf(lc['nflux_dtr'], missing='conservative',nlags=nlags,fft=True)
    lag_times = bins_to_days(np.arange(len(acf_corr)), bs)
    
    # try convolving with a gaussian
    if isinstance(smooth, int) :
        acf_smooth = np.convolve(acf_corr, gaussian(smooth), mode="same")
    else :
        acf_smooth = np.array([])

    # update fits_result
    fits_result['acf_lags'] = lag_times
    fits_result['acf'] = acf_corr
    fits_result['acf_smooth'] = acf_smooth

    return fits_result

def calc_parabolas(corr, TICID=None, bs=cadence, smooth=None, prot_prior_func=calc_fft_pgi, prot_prior_func_kwargs={}):
    """
    Calculates the best fit parabolas to peaks in an acf.

    Args:
        corr (:obj:`1D array`): the autocorrelation function.
        bs (:obj:`float`): the size of the bins, in units of seconds.
        TICID (:obj:`int` or :obj:`str`, optional): the ID for the object.
        smooth (:obj:`int`, optional): if supplied, will apply smoothing to the LC before fitting parabolas by convolving with a gaussian with a FWHM equal to this value.
        prot_prior_func (:obj:`func`, optional): the function to be used to idntify the period_guess_index (pgi) for the rotation period. Defaults to `calc_fft_pgi()`
        prot_prior_func_kwargs (:obj:`dict`, optional): keyword arguments for the `prot_prior_func` function.

    Returns:
        results (:obj:`dict`): a dictionary of info on the parabola fits.
    """
    # make a dictionary to store results in 
    # any key ending with _k is an array length 5, where the value at each index
    # is associated with a fit to a different peak in the acf. fitted_parabola_k is an
    # array of arrays, each of which is the parabola fit to one of the peaks.
    # 'fail', when set to True, indicates that the test could not be completed
    # due to finding a pgi greater than half the sample length.
    # 'half_period' describes whether there peaks in the ACF at half periods due to
    # having spots in opposite hemispheres. 'half_period_check' means that the peak
    # height difference is less than 5% and needs to be checked by hand.
    results = {'smooth':smooth, 'fft':None, 'fft_period':None, 'pgi':np.nan,
               'a_k':np.array([]), 'b_k':np.array([]), 'c_k':np.array([]), 
               'Rsq_k':np.array([]), 'hwhm_k':np.array([]), 'fitted_parabola_k':[[],[],[],[],[]],
               'P_k':np.array([]), 'A_k':np.array([]), 'B_k':np.array([]), 
               'P_avg':np.nan, 'A_avg':np.nan, 'B_avg':np.nan, 'R_avg':np.nan, 'fft_prom':np.nan,
               'P_err':np.nan,
               'half_period': False, 'half_period_check':False,
               'fail':False }
        
    # if smoothing of the acf is requested, apply it
    if smooth :
        corr_smooth = np.convolve(corr, gaussian(smooth), mode="same")
        corr = corr_smooth
    
    # if the pgi is given as a number, use that
    if (type(prot_prior_func)==float) or (type(prot_prior_func)==int):
        # the pgi will be the index of the acf_lags closes to the provided number in days
        prot_prior_lags = int(days_to_bins(prot_prior_func, bs))
        pgi = min(range(len(corr)), key=lambda i: abs(range(len(corr))[i]-prot_prior_lags))
        results['pgi'] = pgi
    else :
        # Calculate the pgi (initial guess for the period)
        # by default, this will use the funciton calc_fft_pgi, which selects the highest peak in the 
        # FFT of the ACF. You can also write a custom pgi-finding function and pass it in to process_test_raw
        pgi, pgi_results = prot_prior_func(corr, bs=bs, **prot_prior_func_kwargs)
        
        # update the results dictionary
        results.update(pgi_results)
        results['pgi'] = pgi
     
    # run curvefit_peaks for the first peak and up to four aliases
    for peak_num in range(1,6) :
        # check that the alias won't extend beyond end of the acf
        if peak_num*pgi + pgi/4 < len(corr) :
            # run curvefit
            fit_params, peak_params, R_adj, fitted_parabola, hwhm = curvefit_peak(parabola, corr, pgi, peak_num)
            
            # add results to the appropriate dictionary
            results['a_k'] = np.append(results['a_k'], fit_params[0])
            results['b_k'] = np.append(results['b_k'], fit_params[1])
            results['c_k'] = np.append(results['c_k'], fit_params[2])
            results['P_k'] = np.append(results['P_k'], peak_params[0])
            results['A_k'] = np.append(results['A_k'], peak_params[1])
            results['B_k'] = np.append(results['B_k'], peak_params[2])
            results['Rsq_k'] = np.append(results['Rsq_k'], R_adj)
            results['hwhm_k'] = np.append(results['hwhm_k'], hwhm)
            results['fitted_parabola_k'][peak_num-1] = fitted_parabola
        else :
            # trim fitted_parabola_k to the appropriate length
            results['fitted_parabola_k'] = results['fitted_parabola_k'][:peak_num]

    # add the averaged values to the results dictionary
    results['A_avg'] = np.nanmean(results['A_k'])
    results['B_avg'] = np.nanmean(results['B_k'])
    results['R_avg'] = np.nanmean(results['Rsq_k'])
    
    # calculate the error bar on P_avg
    if len(results['P_k']) >= 3 :
        results['P_err'] = np.std(results['P_k'])/np.sqrt(len(results['P_k']))
    else :
        results['P_err'] = np.nanmean(results['hwhm_k'])

    
    # Select the rotation period, keeping in mind that there may be spots in opposite hemispheres
    # Check for alternating peak heights in the ACF.
    if len(results['A_k']) > 2 :
        # check if the second peak is higher than the 1st or 3rd by more than 5%
        if results['A_k'][1]*.95 > results['A_k'][0] and results['A_k'][1]*.95 > results['A_k'][2] :
            # this is the unambiguous case, definitely a half period
            try: 
                results['P_avg'] = np.nanmean([results['P_k'][1], results['P_k'][3]]) * 2
            except:
                results['P_avg'] = results['P_k'][1]
            
            results['half_period'] = True
            
        elif results['A_k'][1] > results['A_k'][0] and results['A_k'][1] > results['A_k'][2]:
            # this is the ambiguous case, less than 5% difference in peak height
            # does NOT automatically update P_avg, this will have to be done by hand when checked
            results['P_avg'] = np.nanmean(results['P_k'])
            results['half_period_check'] = True
        else:
            results['P_avg'] = np.nanmean(results['P_k'])
              
    # now, everything in the results dictionary should be taken care of
    return results


def process_LightCurve(lc, bs=cadence, precleaned=False,
                        transit=None,
                        max_lag=None, smooth=None, sector_label=None,
                        prot_prior='fft', prot_prior_func=None, prot_prior_func_kwargs={}):
    """
    Bins a timeseries to the desired cadence. Works much faster than Lightkurve's built in binning function.

    Args:
        lc (:obj:`LightCurve obj`): the cleaned lightcurve.
        bs (:obj:`float`): the size of the bins, in units of seconds.
        precleaned (:obj:`bool`, optional): set to True if the provided `lc` argument has already been cleaned and normalized.
        transit (:obj:`array`, optional): array of transit parameters like [period, epoch, duration] in units of days each entry can be an array if there are multiple planets. Also used by `default_cleaning_func`.
        max_lag (:obj:`float`, optional): the maximum lag to which to calculate the ACF in units of days.
        smooth (:obj:`int`, optional): if supplißed, will apply smoothing to the LC before fitting parabolas by convolving with a gaussian with a FWHM equal to this value.
        sector_label (:obj:`int` or :obj:`str`, optional): tallows you to set a custom sector label, must be castable to a string.
        prot_prior_func (:obj:`func`, optional): the function to be used to identify the period_guess_index (pgi) for the rotation period. Defaults to `calc_fft_pgi()`
        prot_prior_func_kwargs (:obj:`dict`, optional): keyword arguments for the `prot_prior_func` function.

    Returns
    --------
    fits_result : dict
        dictionary containing information on the LC and ACF.
     process_result : dict
         dictionary containing information on the parabola fits.
    """
    if (prot_prior not in ["fft", "custom"]) and not isinstance(prot_prior, int):
        raise ValueError(
            r"Invalid argument passed to prot_prior. Please use 'fft', 'custom', or an int number."
        )

    lc_clean = lc
    
    # calculate the acf
    fits_result = calc_acf(lc_clean, bs=bs, max_lag=max_lag, smooth=smooth)

    # add the raw light curve to the fits_result for ease of inspection later on
    fits_result['time_raw'] = lc['time']
    fits_result['flux_raw'] = lc['nflux_dtr']
    fits_result['flux_err_raw'] = lc['nflux_err']

    # calculate parabola fits
    if prot_prior == 'fft':
        process_result = calc_parabolas(fits_result['acf'], bs=bs, prot_prior_func=calc_fft_pgi)
    elif prot_prior == 'custom':
        process_result = calc_parabolas(fits_result['acf'], bs=bs, prot_prior_func=prot_prior_func, 
                                prot_prior_func_kwargs=prot_prior_func_kwargs)
    else:  # it's an int. The check above only allows 'fft', 'custom', or an int
        process_result = calc_parabolas(
            fits_result["acf"], bs=bs, prot_prior_func=prot_prior
        )

    return fits_result, process_result


# # PLOTTING FUNCTIONS

def plot_fft(process_result, plot_peaks=True, **plt_kwargs):
    """
    Given the result dictionaries from process_LightCurve, plots the FFT of the ACF. Returns a figure object.

    Parameters
    ----------
    process_result : dict
        dictionary containing information on the parabola fits, as returned by `process_LightCurve()`.
    plot_peaks : bool
        if True, places a marker on the five tallest peaks in the FFT.
    plt_kwargs : dict
        keyword arguments for `matplotlib.plt.plot()`.
    """
    fft = process_result['fft']
    fft_period = process_result["fft_period_days"]
    fft_pgi = process_result['fft_pgi']
    
    # make le plot!
    fig, ax = plt.subplots(1, 1, figsize=[12,5], facecolor='white')
    ax.plot(fft_period,fft, color='black', **plt_kwargs)

    # also mark the highest peaks, with the highest one marked in red and the rest in green
    if plot_peaks :
        peaks_x = process_result['max_peaks']
        peaks_y = fft[peaks_x]
        ax.plot(fft_period[peaks_x], peaks_y, marker='o', color='green',linestyle='', markersize=8, fillstyle='none')
        ax.plot(fft_period[fft_pgi], fft[fft_pgi], marker='o', color='red', markersize=8, fillstyle='none')

    # labels
    ax.set_title("FFT of the ACF", fontsize=14)
    ax.set_xlabel("Period (days)", fontsize=14)
    ax.set_ylabel("FFT Power", fontsize=14)

    return fig, ax


# # PLOTTING AND PRINTING FUNCTIONS
def plot_acf(fits_result,process_result, plot_peaks=True, plot_line=None, cut=10):
    """
    Prints a summary of the descriptive parameters calculated by `process_LightCurve()`.

    Parameters
    ----------
    fits_result : dict
        dictionary containing information on the LC and ACF, as returned by `process_LightCurve()`.
    process_result : dict
        dictionary containing information on the parabola fits, as returned by `process_LightCurve()`.
    plot_peaks : bool
        if set to True, will overplot the parabola fits to the ACF peaks on the ACF plot
    plot_line : float or None
        if given a lag time in days, will plot a vertical line on the ACF at that x-value,
        intended to plot the found period for visual comparison.
    cut : int
        plots look better when you cut the first few points off the ACF, to avoid the high peak at (0,1).
        This keyword lets you adjust how many points get cut off the front.

    Returns
    -------
        fig :
            the figure object.
        ax :
            the axis object with the plotted function.
    """
    # make the base plot
    # fig_num=plt.figure().number + 1
    fig, ax = plt.subplots(1, 1, figsize=[10,5], facecolor='white')
    ax.plot(fits_result['acf_lags'][cut:],fits_result['acf'][cut:])
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("ACF")
    ax.set_title("ACF")

    pgi = process_result['pgi']
    # check if the test failed
    if pgi <= 0 or np.isnan(pgi) :
        print('Cannot plot peaks, no plausible rotation period detected.')
        return fig, ax
    if pgi > len(fits_result['acf_lags']//2) :
        print("Test failed due to pgi > 1/2*sample length") 
        return fig, ax

    # if desired, plot the fitted peaks to the acf
    if plot_peaks :
        # extract the needed info
        # plots look better when you cut off the first few points in the ACF
        lag_times = fits_result['acf_lags']
        pgi = process_result['pgi']
        fitted_parabolas = process_result['fitted_parabola_k']
        try : 
            acf_snip = pgi * 7
        except :
            acf_snip = pgi * len(process_result['a_k']) + pgi/2
        acf_snip = min(acf_snip, len(lag_times)-1)

        # Set a reasonable limit
        ax.set_xlim([0, lag_times[acf_snip]])
        
        # now plot each peak
        for i in range(len(process_result['a_k'])):
            curve = fitted_parabolas[i]
            window = len(curve)
            snip = np.arange(pgi*(i+1) - window/2, pgi*(i+1) + window/2).astype(int)
            lag_snip = lag_times[snip]
            ax.plot(lag_snip, curve,linewidth=4,color='r',linestyle='-')
            
        
        # now make it pretty
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("ACF")
        ax.set_title("ACF")

        # if provided, plot a line where requested
    if plot_line :
        line_y = np.arange(-1.5,1.5,.1)
        line_x = np.ones(len(line_y)) * (plot_line)# / (cad/bs)
        ax.plot(line_x, line_y, c='g', scaley=False)
    return fig, ax

###########################file.io.py

import os
import logging
import sys

def logger_tessilator(name_log, log_ext='logging'):
    '''A function to set up the logging files from each tessilator module
    
    parameters
    ----------
    ref_name : `str`
        The reference name for each subdirectory which will connect all output
        files.
    name_log : `str`
        The name of the python module.
    log_ext : `str`, optional, default='logging'
        The name of the directory to save the logging files to.

    returns
    -------
    logger : `logging.getLogger`
        The logging object created by the function.
    '''
    logger = logging.getLogger(name_log)   
    #print('ARGV:',sys.argv, len(sys.argv))
    if len(sys.argv)==6:
        log_dir = make_dir(log_ext, sys.argv[4])
    else:
        log_dir = make_dir(log_ext, 'target')
    

    f_handler = logging.FileHandler(f'{log_dir}/{name_log}.log')
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - '
                                 '%(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    return logger


def make_dir(extn, ref):
    '''Create a directory to store various tessilator results.
    
    parameters
    ----------
    extn : `str`
        The name of the parent directory to store the files.
    ref : `str`
        The name of the sub-directory to store the specific set of data files.
    
    returns
    -------
    dir_name : `str`
        The name of the full directory extension AND creates the directory
        (if needed).
    '''
    dir_name = f'./{extn}/{ref}'
    dir_path_exist = os.path.exists(dir_name)
    if not dir_path_exist:
        os.makedirs(dir_name)
    return dir_name

import os
import logging
import sys

def logger_tessilator(name_log, log_ext='logging'):
    '''A function to set up the logging files from each tessilator module
    
    parameters
    ----------
    ref_name : `str`
        The reference name for each subdirectory which will connect all output
        files.
    name_log : `str`
        The name of the python module.
    log_ext : `str`, optional, default='logging'
        The name of the directory to save the logging files to.

    returns
    -------
    logger : `logging.getLogger`
        The logging object created by the function.
    '''
    logger = logging.getLogger(name_log)   
    #print('ARGV:',sys.argv, len(sys.argv))
    if len(sys.argv)==6:
        log_dir = make_dir(log_ext, sys.argv[4])
    else:
        log_dir = make_dir(log_ext, 'target')
    

    f_handler = logging.FileHandler(f'{log_dir}/{name_log}.log')
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - '
                                 '%(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    return logger


def make_dir(extn, ref):
    '''Create a directory to store various tessilator results.
    
    parameters
    ----------
    extn : `str`
        The name of the parent directory to store the files.
    ref : `str`
        The name of the sub-directory to store the specific set of data files.
    
    returns
    -------
    dir_name : `str`
        The name of the full directory extension AND creates the directory
        (if needed).
    '''
    dir_name = f'./{extn}/{ref}'
    dir_path_exist = os.path.exists(dir_name)
    if not dir_path_exist:
        os.makedirs(dir_name)
    return dir_name

import inspect
import sys


# Third party
import numpy as np
import os
import json

from astropy.table import Table
from astropy.stats import akaike_info_criterion_lsq

from scipy.stats import median_abs_deviation as MAD
from scipy.optimize import curve_fit

import itertools as it
from operator import itemgetter



###############################################################################
###############################################################################
###############################################################################



# initialize the logger object
logger = logger_tessilator(__name__) 
    
    
    
    
def get_time_segments(t, t_fac=10.):
    '''Split the lightcurve into groups of contiguous data
    
    Group data into segments where the time difference between data points is
    less than some threshold factor, `t_fac`.
    
    parameters
    ----------
    t : `Iter`
        The list of time coordinates from the lightcurve
    t_fac : `float`, optional, default=10.
        A time factor, which is multiplied by the median cadence of the full
        lightcurve.
        
    returns
    -------
    ds : `list`
        The start indices of each time segment
    df : `list`
        The final indices of each time segment
    '''
    td     = np.zeros(len(t))
    td[1:] = np.diff(t)
    t_arr = (td <= t_fac*np.median(td)).astype(int)
    groups = (list(group) for key, group in it.groupby(enumerate(t_arr),
                                                       key=itemgetter(1))
                                                       if key)
    ss = [[group[0][0], group[-1][0]] for group in groups
                                      if group[-1][0] > group[0][0]]
    ss = np.array(ss).T
    ds, df = ss[0,:], ss[1,:]
    ds[1:] = [ds[i]-1 for i in range(1,len(ds))]
    df = np.array([(i+1) for i in df])
    return ds, df



    
def remove_sparse_data(x_first, x_last, min_crit_frac=.05, min_crit_num=50):
    '''Removes very sparse data groups, when there are 3 or more groups.

    Calculate the mean (mean_group) and standard deviation (std_group) for the
    number of data points in each group (n_points).
    If std_group > "std_crit", then only keep groups with n_points > std_crit.
    
    parameters
    ----------
    x_first : `Iterable`
        The index values of the first element in each group
    x_last : `Iterable`
        The index values of the last element each group
    min_crit_frac : `float`, optional, default=.05
        The minimum relative size of a flux component when correcting for
        sparse data in the cleaning functions.
    min_crit_num : `int`, optional, default=50
        The minimum number of data points required for a flux component in the
        sparse data cleaning functions.

    returns
    -------
    y_first : `np.array`
        The index values of the first element of the new arrays
    y_last : `np.array`
        The index values of the last element of the new arrays
    '''
    try:
        n_points = np.array([x_last[i] - x_first[i]
                            for i in range(len(x_first))])
        n_tot = np.sum(n_points)
        std_crit = max(min_crit_num, min_crit_frac*n_tot)
        y_first, y_last = np.array(x_first), np.array(x_last)
        if len(x_first) > 2:
            std_group = np.std(n_points)
            if std_group > std_crit:
                g = n_points > std_crit
                y_first, y_last = y_first[g], y_last[g]
                return y_first, y_last
        return y_first, y_last
    except:
        logger.warning("The sparse data removal algorithm failed. Retaining "
                       "the input indices.")
        return y_first, y_last


    
    
    
def aic_selector(x, y, poly_max=3, cov_min=1e-10):
    '''Select the detrending polynomial from the Aikaike Information Criterion
    
    This function uses the Aikaike Information Criterion (AIC) to find the most
    appropriate polynomial order to a set of X, Y data points.
    
    parameters
    ----------
    x : `Iterable`
        The independent variable
    y : `Iterable`
        The dependent variable
    poly_max : `int`, optional, default=3
        The maximum polynomial order to test
    cov_min : `float`
        A threshold value for the first element of the covariance matrix.
        Sometimes the AIC will automatically select a higher-order
        polynomial to a distribution that is clearly best fit by the
        preceeding lower-order fit. For example, a second-order fit provides a
        better fit for a perfect straight line. This is a bug in the numerical
        rounding. Therefore, if the value of the first element of the
        covariance matrix is less than cov_min for the lower order, then the
        lower order fit is selected.

    returns
    -------
    poly_ord : `int`
        The best polynomial order
    coeffs : `list`
        The polynomial coefficients.
    
    '''
    
    q = 0
    N = 1.*len(x)
    try:
        while q < poly_max:
            k1, k2 = q+1, q+2
            p1, r1, _,_,_ = np.polyfit(x, y, q, full=True)
            p2, r2, _,_,_ = np.polyfit(x, y, q+1, full=True)
            with np.errstate(invalid='ignore'):
                SSR1 = np.sum((np.polyval(p1, x) - y)**2)
                SSR2 = np.sum((np.polyval(p2, x) - y)**2)
            AIC1 = akaike_info_criterion_lsq(SSR1, k1, N)
            AIC2 = akaike_info_criterion_lsq(SSR2, k2, N)
            if (AIC1 < AIC2) | (r1 < cov_min):
                poly_ord, coeffs = q, p1
                return poly_ord, list(coeffs)
            else:
                q += 1
                if q >= poly_max:
                    poly_ord, coeffs = q, p2
                    return poly_ord, list(coeffs)
    except:
        return 0, [1.0]


def relative_root_mean_squared_error(true, pred):
    '''Return the relative root mean squared error (RRMSE)
    
    Given a list of predicted and true values, calculate the RRMSE
    
    parameters
    ----------
    true : `Iter`
        The set of true (measured) values
    pred : `Iter`
        The set of predicted (model) values
        
    returns
    -------
    rrmse : `float`
        The RRMSE value
    '''
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse = np.sqrt(squared_error)
    return rrmse


def smooth_test(time, flux, n_avg=10):
    '''Determine how to detrend a lightcurve, based on a smoothness algorithm
    
    The idea of this function is to catch lightcurves that appear to have
    periods longer than ~15 days, and are notably smooth. The function
    calculates a sine fit to the (linearly) detrended lightcurve, then
    smoothes it using a running mean. Finally, there are three criteria to
    decide how the lightcurve should be detrended. A boolean flag is returned,
    where False=individual groups and True=the whole lightcurve.
    
    parameters
    ----------
    time : `Iter`
        A set of time coordinates
    flux : `Iter`
        A set of flux coordinates
    n_avg : `float`, optional, default=10
        The number of datapoints to be used for the running mean calculation.
        
    returns
    -------
    smooth_flag : `bool`
        A boolean flag to (partially) determine how the lightcurve should be
        detrended (False=individual groups, True=the whole lightcurve)
    '''

    # 1) detrend the whole lightcurve by a linear fit, and calculate the MAD
    t_new = time-time[0]
    p1, r1, _,_,_ = np.polyfit(t_new, flux, 1, full=True)
    f_new = flux/np.polyval(p1, t_new)
    f_MAD = MAD(f_new, scale='normal')

    # 2) make a sine fit to the detrended lightcurve
    pops, popsc = curve_fit(sin_fit_per, t_new, f_new,
                            bounds=([0.5, 0.0, 0.0, 0.0],
                                    [1.5, 0.2, 100., 2.*np.pi]))
    yp = sin_fit_per(t_new, *pops)
    
    # 3) smooth the arrays using a running mean
    yp_sm = np.array(np.convolve(yp, np.ones(n_avg)/n_avg, mode='valid'))
    flux_sm = np.array(np.convolve(f_new, np.ones(n_avg)/n_avg, mode='valid'))
    t_sm = np.array(np.convolve(t_new, np.ones(n_avg)/n_avg, mode='valid'))
    
    # 4) subtract the smoothed "raw" flux by the smoothed sine fit
    diff_flux = flux_sm - yp_sm
    d_MAD = MAD(diff_flux, scale='normal')

    # 5) calculate the RRMSE between the detrended flux and the sine fit 
    rrmse = relative_root_mean_squared_error(f_new, yp)
    
    # 6) false if:
    #       a) the predicted period from the sine fit is > 15. days
    #       b) the RRMSE is < 0.01
    #       c) the ratio of the MAD between the differential and original flux
    #          is < 0.25   
    if (rrmse < 0.01) & \
       (pops[2] > 15.) & (pops[2] < 99.9) & \
       (d_MAD/f_MAD < 0.25):
        return True
    else:
        return False

       


def norm_choice(t_orig, f_orig, lc_part, MAD_fac=2., poly_max=4):
    '''Choose whether to detrend the lightcurve as one by individual groups.
    
    There are always at least two data components in a TESS sector because of
    the finite time needed for data retrieval. This can sometimes lead to
    discontinuities between components because of TESS systematics and the
    temperature gradients across the photometer. These discontinuities can
    cause the phase of the sinusoidal fit to change, leading to low power
    output in the periodograms. Alternatively, if the data are all detrended
    individually, but the data is relatively continuous, this can lead to
    shorter period measurements.
    
    The idea with this function is that a polynomial fit is made to each
    component (chosen using the Aikaike Information Criterion, AIC). The
    extrapolated flux value from component 1 is calculated at the point where
    component 2 starts. If the difference between this value and the actual
    normalised flux at this point is greater than a given threshold, the data
    should be detrended separately. Otherwise the full lightcurve can be
    detrended as a whole.
    
    In addition, if the "smooth" flag (calculated from the "smooth_test"
    function) is True, then the lightcurve is detrended as a whole, regardless
    of the outcome from this function.
    
    parameters
    ----------
    t_orig : `Iterable`
        The time coordinate
    f_orig : `Iterable`
        The original, normalised flux values
    lc_part : `Iterable`
        The running index for each contiguous data section in the lightcurve
    MAD_fac : `float`, optional, default=2.
        The factor which is to be multiplied by the median absolute deviation.
    poly_max : `float`, optional, default=4
        The maximum polynomial order to test for the AIC evaluation.        

    returns
    -------
    norm_flag : `bool`
        Determines whether the data should be detrended as one whole component
        (False) or by individual groups (True, providing the smooth flag is
        False).
    smooth_flag : `bool`
         Determines whether any detrending should be performed by testing how
         smooth the lightcurve is.
    f1_at_f2_0 : `list`
        The extrapolated fluxes from group 1, calculated at the start point of
        group 2.
    f2_at_f2_0 : `list`
        The first flux values from group 2
    f1_MAD : `list`
        The MAD fluxes from group 1
    f2_MAD : `list`
        The MAD fluxes from group 2
    '''
    norm_flag = False
    Ncomp = len(np.unique(lc_part))
    smooth_flag = smooth_test(t_orig, f_orig)

    f1_at_f2_0, f2_at_f2_0, f1_MAD, f2_MAD = [], [], [], []
    if Ncomp > 1:
        i = 1
        while i < Ncomp:
            g1 = np.array(lc_part == i)
            g2 = np.array(lc_part == i+1)
            try:
                s_fit1, coeff1 = aic_selector(t_orig[g1], f_orig[g1],
                                              poly_max=poly_max)
                s_fit2, coeff2 = aic_selector(t_orig[g2], f_orig[g2],
                                              poly_max=poly_max)
                f1_at_f2_0.append(np.polyval(coeff1, t_orig[g2][0]))
 # The line below IS supposed to be at index "g2"
                f2_at_f2_0.append(np.polyval(coeff2, t_orig[g2][0]))
                f1_n = f_orig[g1]/np.polyval(coeff1, t_orig[g1])
                f2_n = f_orig[g2]/np.polyval(coeff2, t_orig[g2])
                f1_MAD.append(MAD(f1_n, scale='normal'))
                f2_MAD.append(MAD(f2_n, scale='normal'))
                if 2.*abs(f1_at_f2_0[i-1] - f2_at_f2_0[i-i]) > \
                       MAD_fac*((f1_MAD[i-1]+f2_MAD[i-1])/2.):
                    norm_flag = True
                    break
                else:
                    i += 1
            except:
                logger.error('Could not run the AIC selector, '
                             'probably because of a zero-division.')
                f1_at_f2_0.append(np.polyval([1], t_orig[g2][0]))
                f2_at_f2_0.append(np.polyval([1], t_orig[g2][0]))
                f1_n = f_orig[g1]
                f2_n = f_orig[g2]
                f1_MAD.append(MAD(f1_n, scale='normal'))
                f2_MAD.append(MAD(f2_n, scale='normal'))
                norm_flag = False
                break
    if smooth_flag == True:
        norm_flag = False
    return norm_flag, smooth_flag, f1_at_f2_0, f2_at_f2_0, f1_MAD, f2_MAD


def detrend_lc(t,f,lc, MAD_fac=2., poly_max=3):
    '''Detrend and normalise the lightcurves.

    | This function runs 3 operations to detrend the lightcurve, as follows:
    | 1. Choose whether a zeroth- or first-order polynomial is the best fit to
         the full light-curve, using AIC, and detrend the full lightcurve.
    | 2. Decide whether to use the detrended lightcurve from part 1, or to
         detrend individual groups.
    | 3. Return the detrended flux.

    parameters
    ----------
    t : `Iterable`
        the time component of the lightcurve
    f : `Iterable`
        the flux component of the lightcurve.
    lc : `Iterable`
        The index representing the lightcurve component. Note this
        must be indexed starting from 1.
    MAD_fac : `float`, optional, default = 2.
        The factor to multiply the median absolute deviation by.
    poly_max : `int`, optional, default=8
        The maximum order of the polynomial fit.

    returns
    -------
    f_norm : `Iterable`
        The corrected lightcurve after the detrending procedures.
    detr_dict : `dict`
        A dictionary containing the parameters: norm_flag, smooth_flag,
        "f1_at_f2_0, f2_at_f2_0, f1_MAD, f2_MAD" (see norm_choice) and
        "s_fit, coeffs" (see aic_selector)
    '''

    # 1. Choose the best detrending polynomial using the Aikaike Information
    #    Criterion, and detrend the lightcurve as a whole.
    s_fit_0, coeffs_0 = aic_selector(t, f, poly_max=poly_max)
    f_norm = f/np.polyval(coeffs_0, t)

    # 2. Decide whether to use the detrended lightcurve from part 1, or to
    #    separate the lightcurve into individual components and detrend each
    #    one separately
    norm_flag, smooth_flag, f1_at_f2_0, f2_at_f2_0, f1_MAD, f2_MAD = \
                    norm_choice(t, f, lc, MAD_fac=MAD_fac, poly_max=poly_max)
    # 3. Detrend the lightcurve following steps 1 and 2.
    s_fit, coeffs = [], []
    if norm_flag:
        # normalise each component separately.
        f_detrend = np.array([])
        for l in np.unique(lc):
            g = np.array(lc == l)
            s_fit_n, coeffs_n = aic_selector(t[g], f[g], poly_max=poly_max)
            s_fit.append(s_fit_n)
            coeffs.append(coeffs_n)
            f_n = f[g]/np.polyval(coeffs_n, t[g])
            f_detrend = np.append(f_detrend, f_n)
        f_norm = f_detrend
    else:
        # normalise the entire lightcurve as a whole
        f_norm = f# f_norm
    detr_dict = {'norm_flag' : norm_flag,
                 'smooth_flag' : smooth_flag,
                 'f1_at_f2_0' : f1_at_f2_0,
                 'f2_at_f2_0' : f2_at_f2_0,
                 'f1_MAD' : f1_MAD,
                 'f2_MAD' : f2_MAD,
                 's_fit' : s_fit,
                 'coeffs' : coeffs}
    return f_norm, detr_dict




    
    
    
def clean_flux_algorithm(g):
    '''A basic algorithm that trims both sides of a contiguous data string
    if a condition is not satisfied, until the condition is met for the
    first time.
    
    parameters
    ----------
    g : `Iter`
        The outcome for each datapoint, qualified (=1) or not qualified (=0)
    
    returns
    -------
    first : `int`
        The trimmed first point of the array
    last : `int`
        The trimmed last point of the array
    '''
    i, j = 0, len(g)-1
    while i < j:
        if g[i] != 1:
            i+=1
        else:
            first=i
            break
    while j > 0:
        if g[j] != 1:
            j-=1
        else:
            last=j
            break
    if j <= i:
        first, last = 0, len(g)-1
        return first, last
    else:
        return first, last
    
    

    
def clean_edges_outlier(f, MAD_fac=2.):
    '''Remove spurious outliers at the start and end parts of groups.

    The start and end point of each group must have a flux value within a given
    number of MAD from the median flux in the group. This is done because
    during data downlinks, the temperature of the sensors can change notably.
    Therefore the outlying flux points at the group edges are probably from
    temperature instabilities. 

    parameters
    ----------
    f : `Iterable`
        The set of normalised flux coordinates
    MAD_fac : `float`, optional, default=2.
        The threshold number of MAD values to allow.

    returns
    -------
    first : `int`
       The start index for the data string.
    last : `int`
       The end index for the data string.
    '''
    f_med, f_MAD = np.median(f), MAD(f, scale='normal')
    try: 
        g = (np.abs(f-f_med) < MAD_fac*f_MAD).astype(int)
        first, last = clean_flux_algorithm(g)
        
    except:
        logger.error('Something went wrong with the arrays when doing the '
                     'lightcurve edge clipping')
        first, last = 0, len(g)-1
    return first, last


def clean_edges_scatter(f, MAD_fac=2., len_sub_raw=11, num_data_fac=0.1):
    '''Remove highly-scattered data at the edges of each group.

    Some groups have very scattered fluxes at the edges, presumably because the
    sensors are unstable before and after data downlinks. These can degrade the
    quality of the periodogram analysis, or even lead to an incorrect period.
    
    The idea is to group the first "n_sub" datapoints, and calculate the median
    absolute deviation (MAD). If this local MAD value is greater (less) than
    "MAD_fac" times the MAD of the full lightcurve, then the flag at this point
    is 0 (1). The first and last "(n_sub-1)/2" in the lightcurve are given a
    constant value. If the first/last MAD comparison yield a "1" value, then we
    include the full group, including the datapoints replaced with constant
    values -- i.e., no cleaning is necessary.
    
    The value for n_sub is chosen as the minimum value of "len_sub_raw", or
    num_data_fac*(the number of datapoints in the whole set).

    parameters
    ----------
    f : `Iterable`
        The set of normalised flux coordinates
    MAD_fac : `float`, optional, default=2.
        The threshold number of MAD values to allow.
    len_sub_raw : `int`, optional, default=11
        The number of data points to be used in the local MAD value.
    num_data_fac : `float`, optional, default=0.1
        The factor to multiply the number of data points by.

    returns
    -------
    first : `int`
       The start index for the data string.
    last : `int`
       The end index for the data string.
    '''
    n_sub =min(len_sub_raw, int(num_data_fac*len(f)))
    if n_sub // 2 == 0:
        n_sub += 1
    p_e = int((n_sub-1)/2)
    # get the median time and flux, the median absolute deviation in flux
    # and the time difference for each neighbouring point.
    f_med, f_MAD = np.median(f), MAD(f, scale='normal')
    f_diff = np.zeros(len(f))
    f_diff[1:] = np.diff(f)
    f_diff_med = np.median(np.absolute(f_diff))
    f_x = np.array([MAD(f[i:i+n_sub], scale='normal')
                    for i in range(len(f)-n_sub+1)])
    f_diff_run = np.pad(f_x, (p_e, p_e), 'constant',
                        constant_values=(MAD_fac*f_MAD, MAD_fac*f_MAD))

    try:
        g = (np.abs(f_diff_run) < MAD_fac*f_diff_med).astype(int)
        first, last = clean_flux_algorithm(g)
        if first <= p_e:
            first = 0
        elif first > p_e:
            first = np.where(g)[0][p_e]
        if last >= len(g)-1-(2*p_e+1):
            last = len(g)-1
        elif last < len(g)-1-(2*p_e+1):
            last = np.where(g)[0][-p_e]
    except:
        logger.error('Something went wrong with the arrays when doing the '
                     'lightcurve edge clipping')
        first, last = 0, len(g)-1
    return first, last


def run_make_lc_steps(f_lc, f_orig, min_crit_frac=0.1, min_crit_num=50,
                      outl_mad_fac=3.):
    '''Process the lightcurve: cleaning, normalisation and detrending functions
    
    | During each procedure, the function keeps a record of datapoints that are
    | kept or rejected, allowing users to assess the amount of data loss.
    
    | The function makes the following steps...
    | 1. normalise the original flux points
    | 2. split the lightcurve into 'time segments'
    | 3. remove very sparse elements from the lightcurve
    | 4. run the first detrending process to pass to the cleaning function.
    | 5. clean the lightcurve edges from outliers
    | 6. clean the lightcurve edges from scattered data
    | 7. finally cut out data that are extreme outliers.
    | 8. divide each lightcurve component by the median flux value
    | of qualifying data points.
    | 9. return the dictionary

    parameters
    ----------
    f_lc : `dict`
        The initial lightcurve with the minimum following keys required:
        (1) 'time' -> the time coordinate
        (2) 'eflux' -> the error in the flux
        (3) 'f_orig' -> see the f_orig parameter
    f_orig : `str`
        This string determines which of the original flux values to choose.
        It forms the final part of the f_lc keys.
        It could be either 'reg_oflux' (the regular, original flux) or
        'cbv_oflux' (the original flux corrected using co-trending basis
        vectors)
    min_crit_frac : `float`, optional, default=0.1
        The minimum relative size of a flux component when correcting for
        sparse data in the cleaning functions.
    min_crit_num : `int`, optional, default=50
        The minimum number of data points required for a flux component in the
        sparse data cleaning functions.
    outl_mad_fac : `float`, optional, default=3.
        The factor of MAD for the cleaned lightcurve flux values.
        
    returns
    -------
    f_lc : `dict`
        A dictionary storing the full set of results from the lightcurve
        analysis.
        As well as the keys from the inputs, the final keys returned are:
        1: "time" -> the time coordinate.
        2: "mag" -> the TESS magnitude.
        3: "(reg/cbv)_oflux" -> the flux calculated from aperture photometry.
        4: "eflux" -> the error bar on (reg/cbv)_oflux.
        5: "nflux_ori" -> the normalised fluxes from (3).
        6: "nflux_err" -> the error bars on (5).
        7: "nflux_dtr" -> the normalised fluxes after the detrending steps.
        8: "lc_part" -> an index referring to each group in the lightcurve.
        9: "pass_sparse" -> boolean from `remove_sparse_data`
        10: "pass_clean_outlier" -> boolean from clean_edges_outlier.
        11: "pass_clean_scatter" -> boolean from clean_edges_scatter.
        12: "pass_full_outlier" -> boolean from the final outlier rejection.
    detr_dict : `dict`
        The dictionary returned from `detrend_lc`
    '''


    # (1) normalise the original flux points
    f_lc['nflux_ori'] = f_lc[f'{f_orig}']/np.median(f_lc[f'{f_orig}'])
    f_lc['nflux_err'] = f_lc['eflux']/f_lc[f'{f_orig}']
    logger.info('part1: initial normalisation -> done!')

    # (2) split the lightcurve into 'time segments'
    ds1, df1 = get_time_segments(f_lc["time"])
    logger.info('part2: time segmentation -> done!')

    # (3) remove very sparse elements from the lightcurve
    ds2, df2 = remove_sparse_data(ds1, df1, min_crit_frac=min_crit_frac, 
                                  min_crit_num=min_crit_num)
    f_lc["pass_sparse"] = np.array(np.zeros(len(f_lc["time"])), dtype='bool')
    for s, f in zip(ds2, df2):
        f_lc["pass_sparse"][s:f] = True
    logger.info('part3: remove sparse data -> done!')

    # (4) run the first detrending process to pass to the cleaning function.
    f_lc["lc_part"] = np.zeros(len(f_lc["time"]), dtype=int)
    for i, (s, f) in enumerate(zip(ds2, df2)):
        f_lc["lc_part"][s:f] = int(i+1)
    g_cln = f_lc["pass_sparse"]
    f_lc["nflux_dtr"] = np.full(len(f_lc["time"]), -999.)
    f_lc["nflux_dtr"][g_cln], detr_dict = detrend_lc(f_lc["time"][g_cln],
                                                     f_lc["nflux_ori"][g_cln],
                                                     f_lc["lc_part"][g_cln],
                                                     poly_max=1)
    logger.info('part4: detrending -> done!')
    # (5) clean the lightcurve edges from outliers
    ds3, df3 = [], []
    for lc in np.unique(f_lc["lc_part"][g_cln]):
        g = np.where(f_lc["lc_part"] == lc)[0]
        s_o, f_o = clean_edges_outlier(f_lc["nflux_dtr"][g])
        ds3.append(g[s_o])
        df3.append(g[f_o])
    f_lc["pass_clean_outlier"] = np.array(np.zeros(len(f_lc["time"])),
                                          dtype='bool')
    for s, f in zip(ds3, df3):
        f_lc["pass_clean_outlier"][s:f] = True
    logger.info('part5: clean edges, outliers -> done!')

    # (6) clean the lightcurve edges from scattered data
    ds4, df4 = [], []
    for lc in np.unique(f_lc["lc_part"][g_cln]):
        g = np.where(f_lc["lc_part"] == lc)[0]
        s_s, f_s = clean_edges_scatter(f_lc["nflux_dtr"][g])
        ds4.append(g[s_s])
        df4.append(g[f_s])
    f_lc["pass_clean_scatter"] = np.array(np.zeros(len(f_lc["time"])),
                                          dtype='bool')
    for s, f in zip(ds4, df4):
        f_lc["pass_clean_scatter"][s:f] = True
    logger.info('part6: clean edges, scatter -> done!')

    # (7) finally cut out data that are extreme outliers.
    med_lc = np.median(f_lc["nflux_dtr"][g_cln])
    MAD_lc = MAD(f_lc["nflux_dtr"][g_cln], scale='normal')
    f_lc["pass_full_outlier"] = np.array(np.zeros(len(f_lc["time"])),
                                         dtype='bool')
    for f in range(len(f_lc["time"])):
        if abs(f_lc["nflux_dtr"][f] - med_lc) < outl_mad_fac*MAD_lc:
            f_lc["pass_full_outlier"][f] = True
    logger.info('part7: remove extreme points -> done!')

    # (8) divide each lightcurve component by the median flux value of
    #     qualifying data points.
    for lc in np.unique(f_lc["lc_part"][g_cln]):
        g = np.where(f_lc["lc_part"] == lc)[0]
        gx = np.logical_and.reduce([
                   f_lc["pass_sparse"][g], 
                   f_lc["pass_clean_scatter"][g],
                   f_lc["pass_clean_outlier"][g],
                   f_lc["pass_full_outlier"][g]
                   ])
        flux_vals = f_lc["nflux_dtr"][g[gx]]
        med_flux = np.median(flux_vals[flux_vals > 0.])

    f_lc["nflux_dtr"][f_lc["nflux_dtr"] < 0] = -999
    logger.info('part8: write the dictionary -> done!')

    # (9) return the dictionary
    logger.info('part9: FINISHED!')
    return f_lc, detr_dict




def sin_fit(x, y0, A, phi):
    '''
    Returns the best parameters (y_offset, amplitude, and phase) to a regular
    sinusoidal function.

    parameters
    ----------
    x : `Iterable`
        list of input values
    y0 : `float`
        The midpoint of the sine curve
    A : `float`
        The amplitude of the sine curve
    phi : `float`
        The phase angle of the sine curve

    returns
    -------
    sin_fit : `list`
        A list of sin curve values.
    '''
    sin_fit = y0 + A*np.sin(2.*np.pi*x + phi)
    return sin_fit

def sin_fit_per(t, y0, A, per, phi):
    '''
    Returns the best parameters (y_offset, amplitude, and phase) to a regular
    sinusoidal function.

    parameters
    ----------
    t : `Iterable`
        list of input values
    y0 : `float`
        The midpoint of the sine curve
    A : `float`
        The amplitude of the sine curve
    per : `float`
        The period of the sine curve
    phi : `float`
        The phase angle of the sine curve

    returns
    -------
    sin_fit_per : `list`
        A list of sin curve values.
    '''
    sin_fit_per = y0 + A*np.sin((2.*np.pi*t/per) + phi)
    return sin_fit_per






def cbv_fit_test(t, of, cf):
    '''Determine whether the cbv-corrected lightcurve should be considered.
    
    Whilst the cbv-corrected flux are designed to eliminate systematic
    artefacts by identifying features common to many stars (using principle 
    component analysis), the routine can overfit the data, and often the cbv
    corrections inject too much unwanted noise (particularly for targets with
    low signal to noise).
    
    Therefore the plan here is to assess lightcurves produced by the cbv
    corrections by comparing basic attributes with the regular (non-corrected)
    lightcurves. These scores come down to:
    
    1: the number of outliers
    2: the size of the median absolute deviation
    3: which lightcurve provides the lowest chi-squared value to a sine fit.
    
    If the "original lightcurve" scores higher, then the cbv-corrected
    lightcurve is not considered for further analysis.
    
    parameters
    ----------
    t : Iterable
        The time component of the lightcurve
    of : Iterable
        The original flux
    cf : Iterable
        The cbv-corrected flux
    
    returns
    -------
    use_cbv : bool
        True if cf score > of score, else False.
    '''
    
    of_score, cf_score = 0, 0

#1) number of outliers test
    of_nflux = np.array(of)/np.median(of)
    cf_nflux = np.array(cf)/np.median(cf)
    of_nMADf = MAD(of_nflux, scale='normal')
    cf_nMADf = MAD(cf_nflux, scale='normal')
    num_of = np.sum(abs(of_nflux-1.) > of_nMADf)
    num_cf = np.sum(abs(cf_nflux-1.) > cf_nMADf)
    if num_of > num_cf:
        cf_score += 1
    else:
        of_score += 1
#2) which has the largest MAD value
    if of_nMADf > cf_nMADf:
        cf_score += 1
    else:
        of_score += 1
#3) which makes the best sine fit?
    try:
        pops_of, popsc_of = curve_fit(sin_fit, t, of_nflux,
                                bounds=(0, [2., 2., 1000.]))
        pops_cf, popsc_cf = curve_fit(sin_fit, t, cf_nflux,
                                bounds=(0, [2., 2., 1000.]))
        yp_of = sin_fit(of_nflux, *pops_of)
        yp_cf = sin_fit(cf_nflux, *pops_cf)
        chi_of = np.sum((yp_of-of_nflux)**2)/(len(of_nflux)-len(pops_of)-1)
        chi_cf = np.sum((yp_cf-cf_nflux)**2)/(len(cf_nflux)-len(pops_cf)-1)
        if chi_of > chi_cf:
            cf_score += 1
        else:
            of_score += 1
    except:
        logger.error('Could not do the sine-fit comparison for ori vs cbv '
                     'lightcurves')
# get the final score - if cbv wins, then a True statement is returned.    
    if of_score >= cf_score:
        use_cbv = False
    else:
        use_cbv = True
    return use_cbv





def make_lc(phot_table, name_lc='target', store_lc=False, lc_dir='lc', cbv_flag=False):
    '''Construct the normalised, detrended, cleaned TESS lightcurve.
    
    This is essentially a parent function that performs all the steps in fixing
    the lightcurve.
    
    The returned product is an array containing the new tabulated lightcurve
    data for the original (unfiltered) aperture photometry, and (if necessary)
    another one for the CBV-corrected fluxes (see the 'cbv_fit_test' function
    for more information.)

    parameters
    ----------
    phot_table : `astropy.table.Table` or `dict`
        | The data table containing aperture photometry. Columns must include:
        | "time" -> The time coordinate for each image
        | "mag" -> The target magnitude
        | "(reg/cbv)_oflux" -> The total flux subtracted by the background flux
        | "flux_err" -> The error on flux_corr
    name_lc : `str`, optional, default='target'
        The name of the file which the lightcurve data will be saved to.
        The target name
    store_lc : `bool`, optional, default=False
        Choose to save the cleaned lightcurve to file
    lc_dir : `str`, optional, default='lc'
        The directory used to store the lightcurve files if lc_dir==True
    cbv_flag : `bool`, optional, default=False
        Choose whether to analyse the lightcurves for CBV-corrected data.

    returns
    -------
    final_tabs : `list`
        A list of tables containing the lightcurve data
        These are for the original lightcurve, and the cbv-corrected lightcurve
        if required and it satisfies the criteria from cbv_fit_test.
    norm_flags : `list`
        A list of norm_flag values from the detrending algorithm.
    smooth_flags : `list`
        A list of smooth_flag values from the detrending algorithm.
    '''
    logger.info(f'Running the lightcurve analysis for {name_lc}')
    f_labels = ['reg_oflux']
    cbv_ret = False
    
    if cbv_flag:
        if "cbv_oflux" in phot_table.colnames:
            f_labels.append('cbv_oflux')
            use_cbv = cbv_fit_test(phot_table["time"], phot_table["reg_oflux"],
                                   phot_table["cbv_oflux"])
            if use_cbv:
                cbv_ret = True
    final_tabs, norm_flags, smooth_flags = [], [], []
    for f_label in f_labels:
        logger.info(f'using the flux label: {f_label}')
        final_lc = {}
        final_lc["time"] = phot_table["time"].data
        final_lc["mag"] = phot_table["mag"].data
        final_lc[f'{f_label}'] = phot_table[f'{f_label}'].data
        final_lc["eflux"] = phot_table["flux_err"].data
        flux_dict, detr_dict = run_make_lc_steps(final_lc, f_label)
        norm_flag = detr_dict["norm_flag"]
        smooth_flag = detr_dict["smooth_flag"]

        keyorder = ['time','mag',f_label,'eflux','nflux_ori','nflux_err',
                    'nflux_dtr','lc_part','pass_sparse','pass_clean_outlier',
                    'pass_clean_scatter','pass_full_outlier']
        tab_format = ['.6f','.6f','.6f','.6f',
                      '.6f','.4e','.6f','%i',
                      '%s','%s','%s','%s']
        flux_dict = {k: flux_dict[k] for k in keyorder}
        if len(flux_dict["time"]) > 50:
            flux_tab = Table(flux_dict)
            for n, f in zip(keyorder, tab_format):
                flux_tab[n].info.format = f
            #            for k, f in zip(keyorder, tab_format):
            #                flux_tab[k].info.format =x f
            #            if f_label == "reg_oflux":
            final_tabs.append(flux_tab)
            norm_flags.append(norm_flag)
            smooth_flags.append(smooth_flag)
#            if (f_label == "cbv_oflux") and (cbv_ret):
#                final_tabs.append(flux_tab)
#                norm_flags.append(norm_flag)
#                smooth_flags.append(smooth_flag)
            if store_lc:
                path_exist = os.path.exists(f'./{lc_dir}')
                if not path_exist:
                    os.makedirs(f'./{lc_dir}')
                flux_tab.write(f'./{lc_dir}/{name_lc}_{f_label}.csv',
                               format='csv', overwrite=True)
                with open(f'./{lc_dir}/{name_lc}_{f_label}.json', 'w') \
                     as convert_file:
                    convert_file.write(json.dumps(detr_dict))
    return final_tabs, norm_flags, smooth_flags

###################aperture.py

import inspect
import sys

# Third party
import numpy as np


from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u

from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.aperture import aperture_photometry, ApertureStats


# initialize the logger object
logger = logger_tessilator(__name__)



def get_xy_pos(targets, head):
    '''Locate the X-Y position for targets in a given Sector/CCD/Camera mode
    
    The function reads in the RA and DEC positions of each target, and the
    metadata (header) of the input fits frame containing the WCS information.
    A WCS transformation is attempted first, which uses the
    `world_to_array_index` module to assign pixel values that match the
    indexing order of numpy arrays. If the WCS transformation fails, then the
    `Xpos` and `Ypos` columns from the input table are used. If `Xpos` and
    `Ypos` are not available, the function returns an error.

    parameters
    ----------
    targets : `astropy.table.Table`
        The table of input data with celestial coordinates.
    head : `astropy.io.fits`
        The fits header containing the WCS coordinate details.

    returns
    -------
    positions : `tuple`
        A tuple of X-Y pixel positions for each target.
    '''
    try:
        w = WCS(head)
        c = SkyCoord(targets['ra'], targets['dec'], unit=u.deg, frame='icrs')
        y_obj, x_obj = w.world_to_array_index(c)
        if len(y_obj) > 1:
            positions = tuple(zip(x_obj, y_obj))
        else:
            positions = (x_obj[0], y_obj[0])
        logger.info("The WCS coordinates were successfully applied.")
        return positions

    except:
        if ("Xpos" in targets.colnames) and ("Ypos" in targets.colnames):
            positions = tuple(zip(targets["Xpos"], targets["Ypos"]))
            logger.warning("x-y positions used directly - the aperture will be "
                           "offset by a few sub-pixels!")
        else:
            logger.error("Couldn't get the WCS coordinates to work...")
            return
    


def calc_rad(flux_vals, positions, f_lim=0.1, max_rad=4, default_rad=1, frame_num=0):
    '''Calculate the appropriate pixel radius for the aperture
    
    This function uses a basic algorithm to calculate the most appropriate
    radius size to use for the circular aperture photometry of the TESS image
    frames. If the ratio of the median value of neighbouring (8, in a square
    surrounding the central pixel) pixels compared to the central pixel is
    greater than 'f_lim', then expand the radius by one pixel, and test the
    next set of surrounding pixels. The pixel radius is linearly interpolated
    either side of the f_lim boundary. If, after 'n_pix' pixels the condition
    is still satisfied, set the pixel radius equal to 1. The latter constraint
    is intended to avoid contamination from neighbouring sources.
    
    parameters
    ----------
    flux_vals : `np.array`
        The raw flux values from each pixel in the image.
    positions : `tuple`
        The X,Y position of the central pixel.
    f_lim : `float`, optional, default=0.1
        The limiting threshold flux for the criterion.
    max_rad : `int`, optional, default=4
        The maximum number of pixels for the aperture radius.
    default_rad : `int`, optional, default=1
        The default aperture radius to be used in case of an error.
    frame_num : `int`
        The running number of the image frame of the input fits file.
        This is only used for logging purposes.

    returns
    -------
    aper_rad : `float`
        The pixel radius.
    '''
    try:
        x0, y0 = int(positions[0]), int(positions[1])
        f_max, f_old = float(flux_vals[x0,y0]), 1.
        mask_ori = np.zeros([flux_vals.shape[0], flux_vals.shape[1]])
        mask = mask_ori
        i = 1
        while i <= max_rad:
            mask[x0-i:x0+i+1,y0-i] = 1
            mask[x0-i:x0+i+1,y0+i] = 1
            mask[x0-i,y0-i:y0+i] = 1
            mask[x0+i,y0-i:y0+i] = 1
            f_sum = flux_vals[np.where(mask==1)]
            f_new = float(np.median(f_sum))/f_max
            if (f_new < f_lim) or (f_new > f_old):
                break
            else:
                mask = mask_ori
                f_old = f_new
                i += 1

        a_old, a_new = i-1, i
        if f_new < f_lim:
            z = (f_new-f_lim)/(f_lim-f_old)
            aper_rad = (a_new + z*a_old)/(1+z)
        else:
            aper_rad = a_old

        if (a_new > max_rad) or (a_new == 1):
            aper_rad = 0.5
    except:
        logger.warning(f"calc_rad ran into a problem for frame {frame_num}. "
                       f"Aperture radius set to {default_rad} pixel.")
        aper_rad = default_rad
    return aper_rad

def aper_run(file_in, targets, xy_pos=(10.,10.), ap_rad=1., sky_ann=(6.,8.),
             fix_rad=False):
    '''Perform aperture photometry for the image data.

    This function reads in each fits file, determines the pixel radius for an
    image aperture and performs aperture photometry. A table of aperture
    photometry results is returned, which forms the raw lightcurve to be
    processed in subsequent functions.

    parameters
    ----------
    file_in : `str`
        Name of the fits file containing image data.
    targets : `astropy.table.Table`
        The table of input data.
    xy_pos : `tuple`, optional, default=(10.,10.)
        The x-y centroid (in pixels) of the aperture.
    ap_rad : `float`, optional, default=1.
        The size of the aperture radius in pixels.
    sky_ann : `tuple`, optional, default=(6.,8.)
        A 2-element tuple defining the inner and outer annulus to calculate
        the background flux.
    fix_rad : `bool`, optional, default=False
        If True, then set the aperture radius equal to ap_rad, otherwise run the
        calc_rad algorithm.

    returns
    -------
    full_phot_table : `astropy.table.Table`
        The formatted table containing results from the aperture photometry.
    ap_rad : `float`
        The size of the aperture radius in pixels.
    '''
    if isinstance(file_in, np.ndarray):
        fits_files = file_in
    else:
        fits_files = [file_in]

    full_phot_table = Table(names=('run_no','gaia_dr3_id', 'aperture_rad', 'xcenter',
                                   'ycenter', 'flux', 'flux_err', 'bkg', 
                                   'total_bkg', 'reg_oflux', 'mag', 'mag_err',
                                   'time'),
                            dtype=(int, str, float, float, float, float, float,
                                   float, float, float, float, float, float))
    for f_num, f_file in enumerate(fits_files):
        logger.info(f'Running aperture photometry for {f_file}, #{f_num+1} of {len(fits_files)}')
        try:
            with fits.open(f_file) as hdul:
                data = hdul[1].data
                if data.ndim == 1:
                    if "FLUX_ERR" in data.names:
                        n_steps = data.shape[0]-1
                        flux_vals = data["FLUX"]
                        qual_val = data["QUALITY"]
                        time_val = data["TIME"]
                        erro_vals = data["FLUX_ERR"]
                    else:
                        n_steps = 1
                        flux_vals = data["FLUX"]
                        qual_val = [data["QUALITY"][0]]
                        time_val = [data["TIME"][0]]
                        erro_vals = 0.001*flux_vals
                    positions = xy_pos
                elif data.ndim == 2:
                    n_steps = 1
                    head_meta = hdul[0].header
                    head_data = hdul[1].header
                    qual_val = [head_data["DQUALITY"]]
                    time_val = [(head_meta['TSTART'] + head_meta['TSTOP'])/2.]
                    flux_vals = [data]
                    erro_vals = [hdul[2].data]
                    positions = get_xy_pos(targets, head_data)
                if not fix_rad:
                    rad_val = []
                    for n_step in range(n_steps):
                    #define a circular aperture around all objects
                        annulus_aperture = CircularAnnulus(positions,
                                                           sky_ann[0],
                                                           sky_ann[1])
                        aperstats = ApertureStats(flux_vals[n_step],
                                                  annulus_aperture)
                        bkg_rad = aperstats.median
#                        bkg_rad = aperstats.mode
                        flux_x = flux_vals[n_step]-bkg_rad
                        rad_x = calc_rad(flux_x, positions, frame_num=n_step)
                        rad_val.append(rad_x)
                    if len(rad_val) > 1:
#                        Rad = stats.mode(np.array(rad_val), keepdims=False)[0]
                        ap_rad = np.mean(rad_val)
                    else:
                        ap_rad = rad_val[0]
                else:
                    rad_val = np.repeat(ap_rad, n_steps)
                for n_step in range(n_steps):
                    if qual_val[n_step] == 0:
                        aperture = CircularAperture(positions, ap_rad)
                        #select a background annulus
                        annulus_aperture = CircularAnnulus(positions,
                                                           sky_ann[0],
                                                           sky_ann[1])
                        if flux_vals[:][:][n_step].ndim == 1:
                            flux_ap = flux_vals
                            erro_ap = erro_vals
                        else:
                            flux_ap = flux_vals[:][:][n_step]
                            erro_ap = erro_vals[:][:][n_step]
                        #get the image statistics for the background annulus
                        aperstats = ApertureStats(flux_ap, annulus_aperture)
                        #obtain the raw (source+background) flux
                        with np.errstate(invalid='ignore'):
                            t = aperture_photometry(flux_ap, aperture,
                                                    error=erro_ap)
                        #calculate the background contribution to the aperture
                        aperture_area = aperture.area_overlap(flux_ap)
                        #print out the data to "t"
                        t['run_no'] = n_step
                        t['aperture_rad'] = rad_val[n_step]
                        t['gaia_dr3_id'] = targets['source_id']
                        t['bkg'] = aperstats.median
                        t['tot_bkg'] = \
                            t['bkg'] * aperture_area
                        t['ap_sum_sub'] = \
                            t['aperture_sum'] - t['tot_bkg']
                        t['mag'] = -999.
                        t['mag_err'] = -999.
                        g = np.where(t['ap_sum_sub'] > 0.)[0]
                        t['mag'][g] = -2.5*np.log10(t['ap_sum_sub'][g].data)+\
                                      Zpt
                        t['mag_err'][g] = np.abs((-2.5/np.log(10))*\
                            t['aperture_sum_err'][g].data/\
                            t['aperture_sum'][g].data)
                        t['time'] = time_val[n_step]
                        fix_cols = ['run_no', 'gaia_dr3_id', 'aperture_rad', 'xcenter',
                                    'ycenter', 'aperture_sum',
                                    'aperture_sum_err', 'bkg', 'tot_bkg',
                                    'ap_sum_sub', 'mag', 'mag_err', 'time']
                        t = t[fix_cols]
                        for r in range(len(t)):
                            full_phot_table.add_row(t[r])
        except:
            print(f"There is a problem opening the file {f_file}")
            logger.error(f"There is a problem opening the file {f_file}")
            continue
    return full_phot_table, ap_rad

###############################contaminants.py

import inspect
import sys
import math

# Third party imports
import numpy as np
from astroquery.gaia import Gaia
from astropy.table import Table

# initialize the logger object
logger = logger_tessilator(__name__)





def run_sql_query_contaminants(t_target, cont_rad=10., mag_lim=3.,
                               tot_attempts=3):
    '''Perform an SQL query to identify neighbouring contaminants.

    This function generates the SQL query to identify targets within a
    specified pixel radius. The function returns a table of Gaia
    information on the neighbouring sources which is used to quantify the flux
    contamination within the target aperture.

    parameters
    ----------
    t_target : `astropy.table.Table`
        The input table
    cont_rad : `float`, optional, default=10.
        The maximum pixel radius to search for contaminants
    mag_lim : `float`, optional, default=3.
        The faint magnitude limit to search for contaminants, where this value
        is relative to the target. E.G., a value of 3. is 3. magnitudes
        fainter than the target.
    tot_attempts : `int`, optional, default=3
        The total number of SQL query attempts to be made, in case of http
        response issues.

    returns
    -------
    t_gaia : `astropy.table.Table`
        The Gaia results table from the SQL query.    
    '''
    # Generate an SQL query for each target.
    query = f"SELECT source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, \
              phot_rp_mean_mag, \
              DISTANCE(\
              POINT({t_target['ra']}, {t_target['dec']}),\
              POINT(ra, dec)) AS ang_sep\
              FROM gaiadr3.gaia_source\
              WHERE 1 = CONTAINS(\
              POINT({t_target['ra']}, {t_target['dec']}),\
              CIRCLE(ra, dec, {cont_rad*pixel_size/3600.})) \
              AND phot_g_mean_mag < {t_target['RPmag']+mag_lim} \
              ORDER BY phot_g_mean_mag ASC"

    # Attempt a synchronous SQL job, otherwise try the asyncronous method.

    num_attempts = 0
    while num_attempts < tot_attempts:
        print(f'attempting sql query for identifying contaminants: attempt '
              f'{num_attempts+1} of {tot_attempts}...')
        try:
            job = Gaia.launch_job(query)
            break
        except:
            logger.warning(f"Couldn't run the sync query for "
                           f"{t_target['source_id']}, attempt "
                           f"{num_attempts+1}")
            try:
                job = Gaia.launch_job_async(query)
                break
            except:
                logger.warning(f"Couldn't run the async query for "
                               f"{t_target['source_id']}, attempt "
                               f"{num_attempts+1}")
                num_attempts += 1
    if num_attempts < tot_attempts:
        t_gaia = job.get_results()
        return t_gaia
    else:
        print('Most likely there is a server problem. Try again later.')
        sys.exit()


def flux_fraction_contaminant(pix_sep, s, d_thr=5.e-6):
    r"""Quantify the flux contamination from a neighbouring source.

    Calculates the fraction of flux from a neighbouring contaminating source
    that gets scattered into the aperture. The analytic function uses equation
    3b-10 from `Biser & Millman (1965)
    <https://books.google.co.uk/books?id=5XBGAAAAYAAJ>`_, which is a double
    converging sum with infinite limits, given by

    .. math::

       f_{\rm bg} = e^{-t} \sum_{n=0}^{n\to{\infty}}
       {\Bigg\{\frac{t^{n}}{n!}\bigg[1-e^{-s}\sum_{k=0}^{n}
       {\frac{s^{k}}{k!}}} \bigg]\Bigg\}

    To solve the equation computationally, the summation terminates once the
    difference from the nth iteration is less than some given threshold value,
    `d_thr`.

    parameters
    ----------
    pix_sep : `float`
        The pixel distance between a contaminant and the aperture centre.
    s : `float`
        For a given aperture size, rad (in pixels)
        and an FWHM of the TESS PSF, exprf (set at 0.65 pixels),
        :math:`s = {\rm rad}^2/(2.0*{\rm exprf}^2)`
    d_thr : `float`, optional, default=5.e-6
        The threshold value to stop the summations. When the next component
        contributes a value which is less than d_thr, the summation ends.

    returns
    -------
    frac_flux_in_aperture : `float`
        Fraction of contaminant flux that gets scattered into the aperture.
    """
    n, n_z, n_0, n_sign = 0, 0, 0, 1.
    n_sign_lim = 0
    t = pix_sep**2/(2.0*exprf**(2))
    try:
        while True:
            n_old = n_0
            n_sign_old = n_sign
            sk = np.sum([(s**(k)/math.factorial(k)) for k in range(0,n+1)])
            sx = 1.0 - (np.exp(-s)*sk)
            n_0 = ((t**n)/math.factorial(n))*sx
            n_sign = np.sign(n_0-n_old)
            n_z += n_0

            if (n_sign_old-n_sign) != 0:
                n_sign_lim += 1
                if n_sign_lim > 1:
                    return 0.
            if np.abs(n_0) > d_thr:
                n += 1
            else:
                break
        frac_flux_in_aperture = n_z*np.exp(-t)
    except:
        logger.warning('contamination sum did not converge')
        frac_flux_in_aperture = 0.
    return frac_flux_in_aperture

def contamination(t_targets, ap_rad=1.0, n_cont=10, cont_rad=10., mag_lim=3.,
                  tot_attempts=3):
    '''Estimate flux from neighbouring contaminant sources.

    The purpose of this function is to estimate the amount of flux incident in
    the TESS aperture that originates from neighbouring, contaminating sources.
    Given that the passbands from TESS (T-band, 600-1000nm) are similar to Gaia
    RP magnitude, and that Gaia can observe targets down to G~21, the Gaia DR3
    catalogue is used to quantify contamination.
    
    For each target in the input file, the function
    "run_sql_query_contaminants" returns a catalogue of Gaia DR3 objects of all
    neighbouring sources that are within a chosen pixel radius and are brighter
    than $RP_{\\rm source}+$mag_lim.
    
    The Rayleigh formula is used to calculate the fraction of flux incident in
    the aperture from the target, and the function "flux_fraction_contaminant"
    uses an analytical formula `(Biser & Millman 1965, equation 3b-10)
    <https://books.google.co.uk/books?id=5XBGAAAAYAAJ>`_ to
    calculate the flux contribution from all neighbouring sources incident in
    the aperture.

    parameters
    ----------
    t_targets : `astropy.table.Table`
        The input table for all the targets.
    ap_rad : `float`, optional, default=1.0
        The size of the radius aperture (in pixels)
    n_cont : `int`, optional, default=10
        The maximum number of neighbouring contaminants to store to table.
    cont_rad : `float`, optional, default=10.
        The maximum pixel radius to search for contaminants
    mag_lim : `float`, optional, default=3.
        The faintest magnitude to search for contaminants.
    tot_attempts : `int`, optional, default=3
        The number of sql query attempts to be made to acquire Gaia DR3
        data for a contaminant before a time-out error occurs.

    returns
    -------
    t_targets : `astropy.table.Table`
        The input table for all the targets with 3 extra columns to quantify
        the flux contamination.
    t_cont : `astropy.table.Table`
        A table of Gaia DR3 data for the contaminants.
    '''
    con_tot, con_max, con_num = [], [], []
    # Create empty table to fill with results from the contamination analysis.
    t_cont = Table(names=('source_id_target', 'source_id', 'RA',
                          'DEC', 'Gmag', 'BPmag', 'RPmag', 'd_as',
                          'log_flux_frac'),\
                   dtype=(str, str, float, float, float, float, float, float,
                          float))

    for i, t_target in enumerate(t_targets):
        #print(t_target)
        
        r = run_sql_query_contaminants(t_target, cont_rad=cont_rad, mag_lim=mag_lim,
                                       tot_attempts=tot_attempts)
        #print(r)
        r["source_id"] = [f"Gaia DR3 {i}" for i in r["source_id"]]
        print(f"sql search for contaminants completed {t_target['source_id']},"
              f" target {i+1} of {len(t_targets)}.")
        # convert the angular separation from degrees to arcseconds
        r["pix_sep"] = r["ang_sep"]*3600./pixel_size
        if len(r) > 1:
            # make a table of all objects from the SQL except the target itself
            rx = Table(r[r["source_id"] != t_target["source_id"]])
            # calculate the fraction of flux from the source object that falls
            # into the aperture using the Rayleigh formula
            s = ap_rad**(2)/(2.0*exprf**(2)) # measured in pixels
            frp_star = (1.0-np.exp(-s))*10**(-0.4*t_target["RPmag"])
            frp_conts = []
            # calculate the fractional flux incident in the aperture from
            # each contaminant.
            for G_cont, RP_cont, pix_sep in zip(rx["phot_g_mean_mag"],
                                                rx["phot_rp_mean_mag"],
                                                rx["pix_sep"]):
                # if there is no RP magnitude, use the G magnitude, and add
                # 0.756 to it (equivalent to removing half the G-band flux, 
                # which could represent the red part of the G-magnitude
                # passband.)
                if type(RP_cont) == np.ma.core.MaskedConstant:
                    RP_cont = G_cont + 0.756
                    RP_cont = RP_cont.astype(np.float32)
                
                f_frac = flux_fraction_contaminant(pix_sep, s)
                frp_conts.append(f_frac*10**(-0.4*RP_cont))
            rx['log_flux_frac'] = 0.
            frp_tot, frp_max = 0., 0.
            for f, frp_cont in enumerate(frp_conts):
                if frp_cont > 0.:
                    rx['log_flux_frac'][f] = np.log10(frp_cont/frp_star)
                    frp_tot += frp_cont
                    if frp_cont > frp_max:
                        frp_max = frp_cont
                else:
                    rx['log_flux_frac'][f] = -99.

            rx['source_id_target'] = t_target["source_id"]
            new_order = ['source_id_target', 'source_id', 'ra', 'dec', 
                         'phot_g_mean_mag', 'phot_bp_mean_mag',
                         'phot_rp_mean_mag', 'pix_sep', 'log_flux_frac']
            rx.sort(['log_flux_frac'], reverse=True)
            rx = rx[new_order]
            rx['source_id_target'] = rx['source_id_target']
            rx['source_id'] = rx['source_id']

            # store the n_cont highest flux contributors to table
            for rx_row in rx[0:n_cont][:]:
                t_cont.add_row(rx_row)

            if frp_tot > 0.:
                con_tot.append(np.log10(frp_tot/frp_star))
            else:
                con_tot.append(-99.)
            if frp_max > 0.:
                con_max.append(np.log10(frp_max/frp_star))
            else:
                con_max.append(-99.)
            con_num.append(len(frp_conts))
        else:
            con_tot.append(-999)
            con_max.append(-999)
            con_num.append(0)

    t_targets["log_tot_bg"] = con_tot
    t_targets["log_max_bg"] = con_max
    t_targets["num_tot_bg"] = con_num

    return t_targets, t_cont


def is_period_cont(d_target, d_cont, t_cont, frac_amp_cont=0.5):
    '''Identify neighbouring contaminants that may cause the periodicity.

    If the user selects to measure periods for the neighbouring contaminants
    this function returns a flag to assess if a contaminant may actually be
    the source causing the observed periodicity. The function produces two
    flags: `false_flag` and `reliable_flag`, where the former assesses how
    likely a contaminant might be causing the periodicity, and the latter
    indicates how likely it is that the periodicity comes from the target.

    parameters
    ----------
    d_target : `dict`
        A dictionary containing periodogram data of the target star.
    d_cont : `dict`
        A dictionary containing periodogram data of the contaminant star.
    t_cont : `astropy.table.Table`
        A table containing Gaia data for the contaminant star
    frac_amp_cont : `float`, optional, default=0.5
        The threshold factor to account for the difference in amplitude
        of the two stars. If this is high, then the contaminants will be
        less likely to be flagged as the potential source
    
    returns
    -------
    false_flag : `int`
        Either 1 or 0, with 1 (0) indicating the contaminant is (un)likely to
        cause the periodicity.
    reliable_flag : `int`
        Either 1 or 0, with 1 (0) suggesting that the target provides an
        unreliable (reliable) rotation period.        
    '''
    pix_dist = t_cont["pix_sep"]

    per_targ = d_target["period_1"]
    err_targ = d_target["Gauss_fit_peak_parameters"][2]
    amp_targ = d_target["pops_vals"][1]
    RP_targ = d_target["RPmag"]

    per_cont = d_cont["period_best"]
    err_cont = d_cont["Gauss_fit_peak_parameters"][2]
    amp_cont = d_cont["pops_vals"][1]
    RP_cont = t_cont["RPmag"]
    
    
    false_flag, reliable_flag = 0, 0
    if abs(per_targ - per_cont) < (err_targ + err_cont):
        false_flag = 1

    if pix_dist < 1:
        if RP_targ > RP_cont:
            reliable_flag = 1
    else:
        if amp_targ < amp_cont:
            reliable_flag = 1    
    return false_flag, reliable_flag

#############periodogram.py

import warnings

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
import sys
import inspect

from astropy.table import Table
from astropy.timeseries import LombScargle

from scipy.stats import median_abs_deviation as MAD

from scipy.optimize import curve_fit
import itertools as it

from collections.abc import Iterable


# initialize the logger object
#logger = logger_tessilator(__name__) 




def check_for_jumps(time, flux, lc_part, n_avg=10, thresh_diff=10.):
    '''Identify if the lightcurve has jumps.
    
    A jumpy lightcurve is one that has small contiguous data points that change
    in flux significantly compared to the amplitude of the lightcurve. These
    could be due to some instrumental noise or response to a non-astrophysical
    effect. They may also be indicative of a stellar flare or active event.
    
    This function takes a running average of the differences in flux, and flags
    lightcurves if the absolute value exceeds a threshold. These will be
    flagged as "jumpy" lightcurves.

    parameters
    ----------
    time : `Iterable`
        The time coordinate
    flux : `Iterable`
        The original, normalised flux values
    lc_part : `Iterable`
        The running index for each contiguous data section in the lightcurve
    n_avg : `int`, optional, default=10
        The number of data points to calculate the running average
    thresh_diff : `float`, optional, default=10.
        The threshold value, which, if exceeded, will yield a "jumpy"
        lightcurve

    returns
    -------
    jump_flag : `bool`
        This will be True if a jumpy lightcurve is identified, otherwise False.
    '''
    
    jump_flag = False
    try:
        for lc in np.unique(lc_part):
            g = np.array(lc_part == lc)
        
            f_mean = np.convolve(flux[g], np.ones(n_avg), 'valid') / n_avg
            t_mean = np.convolve(flux[g], np.ones(n_avg), 'valid') / n_avg
        
            f_shifts = np.abs(np.diff(f_mean))

            median_f_shifts = np.median(f_shifts)
            max_f_shifts = np.max(f_shifts)
            if max_f_shifts/median_f_shifts > thresh_diff:
                jump_flag = True
                return jump_flag
    except:
        logger.error('Could not run the jump flag criteria for this target.')
        return jump_flag
    return jump_flag


def gauss_fit(x, a0, x_mean, sigma):
    '''Construct a simple Gaussian.

    Return Gaussian values from a given amplitude (a0), mean (x_mean) and
    uncertainty (sigma) for a distribution of values

    parameters
    ----------
    x : `Iterable`
        list of input values
    a0 : `float`
        Amplitude of a Gaussian
    x_mean : `float`
        The mean value of a Gaussian
    sigma : `float`
        The Gaussian uncertainty

    returns
    -------
    gaussian : `list`
        A list of Gaussian values.
    '''

    gaussian = a0*np.exp(-(x-x_mean)**2/(2*sigma**2))
    return gaussian


def gauss_fit_peak(period, power, max_power=1):
    '''Applies the Gaussian fit to the periodogram. If there are more than 3
    datapoints (i.e., more datapoints than fixed parameters), the "gauss_fit"
    module is used to return the fit parameters. If there are 3 or less points,
    the maximum peak is located and 9 datapoints are interpolated between the
    2 neighbouring data points of the maximum peak, and the "gauss_fit" module
    is applied.
    
    parameters
    ----------
    period : `Iterable`
        The period values around the peak.
    power : `Iterable`
        The power values around the peak.
    max_power : `float`, optional, default=1
        The maximum value for the power output.
        
    returns
    -------
    popt : `list`
        The best-fit Gaussian parameters: A, B and C where A is the amplitude,
        B is the mean and C is the uncertainty.
    ym : `list`
        The y values calculated from the Gaussian fit.
    '''
    if len(period) > 3:
        try:
            period_diff = period[-1]-period[0]
            popt, _ = curve_fit(gauss_fit, period, power,
                                bounds=([0, period[0], 0],
                                        [max_power, period[-1], period_diff]))
            ym = gauss_fit(period, *popt)
        except:
            logger.error(
                "Couldn't find the optimal parameters for the " "Gaussian fit!"
            )
            p_m = np.argmax(power)
            peak_vals = [p_m-1, p_m, p_m+1]
            x = period[peak_vals]
            y = power[peak_vals]
            xvals = np.linspace(x[0], x[-1], 9)
            yvals = np.interp(xvals, x, y)
            popt, _ = curve_fit(gauss_fit, xvals, yvals,
                                bounds=(0, [1., np.inf, np.inf]))
            ym = gauss_fit(xvals, *popt)     

    else:
        p_m = np.argmax(power)
        peak_vals = [p_m-1, p_m, p_m+1]
        x = period[peak_vals]
        y = power[peak_vals]
        xvals = np.linspace(x[0], x[-1], 9)
        yvals = np.interp(xvals, x, y)
        popt, _ = curve_fit(gauss_fit, xvals, yvals,
                            bounds=(0, [1., np.inf, np.inf]))
        ym = gauss_fit(xvals, *popt)        
    return popt, ym
    
    
def get_next_peak(power, frac_peak=0.85, option=None):
    '''An algorithm to identify the "next"-highest peak in the periodogram

    parameters
    ----------
    power : `Iterable`
        A set of power values calculated from the periodogram analysis.
    frac_peak : `float`, optional, default=0.85
        The relative height of the maximum peak, below which the data will be
        included.
    option : `float`, optional, default=None
        The value that the power must also be below. If kept as None, this is
        equal to frac_peak*the highest power in the array.

    returns
    -------
    a_o : `list`
        A list of indices corresponding to all other parts of the periodogram.
    '''
    # Get the left side of the peak
    a = np.arange(len(power))

    p_m = np.argmax(power)

    if not option:
        cond_2 = frac_peak*power[p_m]
    else:
        cond_2 = option

    x = p_m
    while (power[x-1] < power[x]) and (x > 0):
        x = x-1
    p_l = x
    p_lx = 0
    while (power[p_l] > cond_2) and (p_l > 1):
        p_lx = 1
        p_l = p_l - 1
    if p_lx == 1:
        while (power[p_l-1] < power[p_l]) and (p_l > 0):
            p_l = p_l - 1
    if p_l < 0:
        p_l = 0

    # Get the right side of the peak
    x = p_m
    if x < len(power)-1:
        while (power[x+1] < power[x]) and (x < len(power)-2):
            x = x+1
        p_r = x
        p_rx = 0
        while (power[p_r] > cond_2) and (p_r < len(power)-3):
            p_rx = 1
            p_r = p_r + 1
        if p_rx == 1:
           while (power[p_r+1] < power[p_r]) and (p_r < len(power)-2):
                p_r = p_r + 1
        if p_r > len(power)-1:
            p_r = len(power)-1
    elif x == len(power)-1:
        p_r = x

    #return the indices that do not constitute part of the specific periodogram
    #peak.
    a_g = a[p_l:p_r+2]
    a_o = a[np.setdiff1d(np.arange(a.shape[0]), a_g)] 
    return a_o

def mean_of_arrays(arr, num):
    '''Calculate the mean and standard deviation of an array which is split
    into N components.
    
    parameters
    ----------
    arr : `Iterable`
        The input array
    num : `int`
        The number of arrays in which to (equally) split the data

    returns
    -------
    mean_out : `float`
        The mean of the list of arrays.
    std_out : `float`
        The standard deviation of the list of arrays.
    '''
    x = np.array_split(arr, num)
    ar = np.array(list(it.zip_longest(*x, fillvalue=np.nan)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) 
        mean_out, std_out = np.nanmean(ar, axis=0), np.nanstd(ar, axis=0) 
    return mean_out, std_out
    

def get_Gauss_params_pg(period, power, indices=None, max_power=1,
                        gauss_min_frac=0.05, p_min_thresh=0.05,
                        p_max_thresh=100.):
    '''Calculate the best Gaussian-fit parameters to the periodogram output.
    
    parameters
    ----------
    period : `Iterable`
        The set of period outputs from the periodogram analysis.
    power : `Iterable`
        The set of power outputs from the periodogram analysis.
    indices : `Iterable`, optional, default=None
        The specific indices of the period/power array to be used.
    max_power : `float`, optional, default=1
        The maximum value for the power output.
    gauss_min_frac : `float`, optional, default=0.05
        The minimum ratio between the the Gaussian fit and max_power.
    p_min_thresh : `float`, optional, default=0.05
        The minimum period (in days) to be calculated.
    p_max_thresh : `float`, optional, default=100.
        The maximum period (in days) to be calculated.

    results
    -------
    returns
    -------
    popt : `list`
        The best-fit Gaussian parameters: A, B and C where A is the amplitude,
        B is the mean and C is the uncertainty.
    ym : `list`
        The y values calculated from the Gaussian fit.
    ''' 
# first -- check there are more than 3 values for the Gaussian fit.
    if isinstance(indices, Iterable):
        ind=indices
    else:
        ind=np.arange(len(period))

    if len(ind) > 3:
        pow_r = max(power[ind])-min(power[ind])
        ind_fit = ind[power[ind] >= min(power[ind]) + gauss_min_frac*pow_r]
        popt, ym = gauss_fit_peak(period[ind], power[ind], max_power=max_power)
    else:
        if np.isclose(period[ind], p_max_thresh, atol=0.001).any():
            popt = [1.0, p_max_thresh, 50.]
            ind_fit = np.arange(ind-10, ind)
            ym = power[ind]
        elif np.isclose(period[ind], p_min_thresh, atol=0.001).any():
            popt = [1.0, p_min_thresh, 50.]
            ind_fit = np.arange(ind, ind+10)
            ym = power[ind_fit]
        else:
            popt = [-999, -999, -999]
            n, i_n = 0, ind[-1]
            while i_n+1 < len(power):
                n+=1
                i_n+=1
                if n==3:
                    break
            m, i_m = 0, ind[0]
            while i_m-1 > 0:
                m-=1
                i_m-=1
                if m==-2:
                    break
            ind_fit = np.arange(ind[0]-m, ind[-1]+n)
            ym = power[ind_fit-1]
            
    return popt, ym


def initialise_LS_dict(lc_data, check_jump=False, p_min_thresh=0.05,
                       p_max_thresh=100., samples_per_peak=50):
    '''Run the periodogram analysis and store initial results to a dictionary.
    
    parameters
    ----------
    lc_data : `dict`
        A dictionary containing the lightcurve data. The keys must include
        | "time" -> The time coordinate relative to the first data point
        | "nflux" -> The detrended, cleaned, normalised flux values
        | "enflux" -> The uncertainty for each value of nflux
        | "lc_part" -> An running index describing the various contiguous
                       sections
    check_jump : `bool`, optional, default=False
        Choose to check the lightcurve for jumpy data, using the
        "check_for_jumps" function.
    p_min_thresh : `float`, optional, default=0.05
        The minimum period (in days) to be calculated.
    p_max_thresh : `float`, optional, default=100.
        The maximum period (in days) to be calculated.
    samples_per_peak : `int`, optional, default=10
        The number of samples to measure in each periodogram peak.

    results
    -------
    LS_dict : `dict`
        The dictionary of initial parameters from the periodogram analysis
    cln_lc : `dict`
        The lightcurve data, with any lines containing a False flag removed.
    ls : `astropy.timeseries.LombScargle`
        The Lomb-Scargle periodogram object for the given lightcurve.
    
    '''
    # a_g: array of datapoints that form the Gaussian around the highest power
    # a_o: the array for all other datapoints
    LS_dict = {}

    cln_cond = (np.logical_and.reduce([
                   lc_data["pass_clean_scatter"],
                   lc_data["pass_clean_outlier"],
                   lc_data["pass_full_outlier"]
                   ])) & (lc_data["nflux_dtr"] > -999.)
    cln_lc = lc_data[cln_cond]

    
   
    LS_dict["time"] = np.array(cln_lc["time"])
    LS_dict["nflux"] = np.array(cln_lc["nflux_dtr"])
    LS_dict["enflux"] = np.array(cln_lc["nflux_err"])
    LS_dict["lc_part"] = np.array(cln_lc["lc_part"])

    # calculate the median and MAD flux
    LS_dict['median_MAD_nLC'] = [np.median(LS_dict["nflux"]),
                                 MAD(LS_dict["nflux"], scale='normal')]

    # calculate the TESS magnitude (median and MAD value) 
    # note that this is from the "ORIGINAL LIGHTCURVE"
    try:
        mag = lc_data["mag"]
        LS_dict['Tmag_MED'] = np.median(mag[mag > -999])
        LS_dict['Tmag_MAD'] = MAD(mag[mag > -999])
    except:
        LS_dict['Tmag_MED'], LS_dict['Tmag_MAD'] = -999, -999    


    # assess the jump flag
    LS_dict['jump_flag'] = -999
    if check_jump:
        LS_dict['jump_flag'] = int(check_for_jumps(LS_dict["time"],
                                                   LS_dict["nflux"],
                                                   LS_dict["lc_part"]))


    # run the LS periodogram
    ls = LombScargle(LS_dict["time"], LS_dict["nflux"], dy=LS_dict["enflux"])
    frequency, power = ls.autopower(minimum_frequency=1./p_max_thresh,
                                    maximum_frequency=1./p_min_thresh,
                                    samples_per_peak=samples_per_peak)

    p_m = np.argmax(power)

    LS_dict['a_1'] = np.arange(len(power))
    LS_dict['period_a_1'] = 1./frequency[::-1]
    LS_dict['power_a_1'] = power[::-1]
    LS_dict['period_1'] = 1.0/frequency[p_m]
    LS_dict['power_1'] = power[p_m]



   # calculate the false alarm probability values, using the quick method
    try:
        FAP = ls.false_alarm_probability(power.max())
        probabilities = [0.1, 0.05, 0.01]
        LS_dict['FAPs'] = ls.false_alarm_level(probabilities)
    except:
        logger.error('Something went wrong with the FAP test, maybe division '
                     'by 0.')
        LS_dict['FAPs'] = np.array([0.3, 0.2, 0.1])
    LS_dict["FAP_001"] = LS_dict["FAPs"][2]




    LS_dict['shuffle_flag'] = 0
    LS_dict['period_shuffle'] = -999
    LS_dict['period_shuffle_err'] = -999

    return LS_dict, cln_lc, ls

def write_periodogram(LS_dict, name_pg='', pg_dir='', lc_type=''):
    '''Save the period and power output from the periodogram analysis to file.
    
    parameters
    ----------
    LS_dict : `dict`
        The dictionary of periodogram results produced in initialise_LS_dict
    name_pg : `str`, optional, default=''
        The name of the file to save the periodogram results
    pg_dir : `str`, optional, default=''
        The directory of the file to save the periodogram results
    lc_type : `str`, optional, default=''
        An additional string for reference in the file name.
    
    returns
    -------
    Nothing returned. The result is saved to file.
    '''
    if name_pg:
        res_table = Table(names=('period', 'power'), dtype=(float,float))
        for pe, po in zip(LS_dict['period_a_1'], LS_dict['power_a_1']):
            res_table.add_row([pe, po])
        res_table['period'].info.format = '.6f'
        res_table['power'].info.format = '.4e'
        res_table.write(f'{pg_dir}/{name_pg}_{lc_type}.csv', overwrite=True)


def get_periodogram_peaks(LS_dict, n_peaks=4):
    '''Calculate parameters for a given number of periodogram peaks
    
    This function locates the `n_peaks` highest peaks in the periodogram and
    calculates the associated period, power and Gaussian-fit parameters.
    
    parameters
    ----------
    LS_dict : `dict`
        The dictionary of periodogram results produced in initialise_LS_dict
    n_peaks : `int`, optional, default=4
        The number of periodogram peaks to analyse.
        
    results
    -------
    Nothing returned, the LS_dict dictionary is updated with new parameters.
    '''
    for i in 1+np.arange(n_peaks):
        try:
            # get the indices of all the peaks that were not part of the last
            # peak
            LS_dict[f'a_{i+1}'] = get_next_peak(LS_dict[f'power_a_{i}'])
            # all the indices that 'are' part of the peak
            LS_dict[f'a_g_{i}'] = np.delete(np.array(LS_dict[f'a_{i}']),
                                            np.array(LS_dict[f'a_{i+1}']))
            x1, x2 = get_Gauss_params_pg(LS_dict["period_a_1"],
                                         LS_dict["power_a_1"],
                                         indices=LS_dict[f'a_g_{i}'])
            LS_dict[f'Gauss_{i}'], LS_dict[f'Gauss_y_{i}'] = x1, x2 
            LS_dict[f'period_{i}_fit'] = LS_dict[f'Gauss_{i}'][1]
            LS_dict[f'period_{i}_err'] = LS_dict[f'Gauss_{i}'][2]


           # find all the new period values in the new array
            x3 = LS_dict[f'period_a_{i}'][LS_dict[f'a_{i+1}']]
            LS_dict[f'period_a_{i+1}'] = x3
            # find all the new power values in the new array
            x4 = LS_dict[f'power_a_{i}'][LS_dict[f'a_{i+1}']]
            LS_dict[f'power_a_{i+1}'] = x4
            # calculate the period of the maximum power peak
            x5 = LS_dict[f'period_a_{i+1}'][np.argmax(LS_dict[f'power_a_{i+1}'])]
            LS_dict[f'period_{i+1}'] = x5
            # return the maximum power peak value
            x6 = LS_dict[f'power_a_{i+1}'][np.argmax(LS_dict[f'power_a_{i+1}'])]
            LS_dict[f'power_{i+1}'] = x6
            
            
            
        except:
            logger.error('Something went wrong with the periods/powers of '
                         'subsequent peaks. Probably an empty array of '
                         'values.')
            LS_dict[f'Gauss_{i}'] = [-999, -999, -999]
            LS_dict[f'period_a_{i+1}'] = -999
            LS_dict[f'power_a_{i+1}'] = -999
            LS_dict[f'period_{i+1}'] = -999
            LS_dict[f'power_{i+1}'] = -999
            LS_dict[f'period_{i}_fit'] = -999
            LS_dict[f'period_{i}_err'] = -999

        LS_dict.pop(f'a_{n_peaks+1}', None)
        LS_dict.pop(f'period_a_{n_peaks+1}', None)
        LS_dict.pop(f'power_a_{n_peaks+1}', None)
        LS_dict.pop(f'period_{n_peaks+1}', None)
        LS_dict.pop(f'power_{n_peaks+1}', None)

    try:
        LS_dict['period_around_1'] = LS_dict["period_a_1"][LS_dict['a_g_1']]
        LS_dict['power_around_1'] = LS_dict["power_a_1"][LS_dict['a_g_1']]
    except:
        LS_dict['period_around_1'] = -999
        LS_dict['power_around_1'] = -999


def shuffle_periodogram(lc_data, n_shuf_runs=100, p_min=0.1, p_max=100.,
                        n_min=0.1, n_max=1.0):
    '''Generate period measurements by sampling subsets of the lightcurve.
    
    parameters
    ----------
    lc_data : `dict`
        A dictionary containing the lightcurve data. The keys must include
        | "time" -> The time coordinate relative to the first data point
        | "nflux" -> The detrended, cleaned, normalised flux values
        | "enflux" -> The uncertainty for each value of nflux
        | "lc_part" -> An running index describing the various contiguous
                       sections
    n_shuf_runs : `int`, optional, default=100
        The number of measurements to be made
    p_min : `float`, optional, default=0.1
        The minimum period for the shuffling method
    p_max : `float`, optional, default=100.
        The maximum period for the shuffling method
    n_min : `float`, optional, default=0.1
        The minimum fraction of a group to be used in the periodogram analysis
    n_max : `float`, optional, default=1.0
        The maximum fraction of a group to be used in the periodogram analysis
    
    results
    -------
    periods_out : `np.array`
        The array of calculated periods.
    '''
    period_arr = []
    sections = np.unique(lc_data['lc_part'])
    for n_run in range(n_shuf_runs):
        try:
            s = np.random.choice(sections)
            lc_s = lc_data[lc_data["lc_part"] == s]
            n = np.random.uniform(low=n_min, high=n_max)
            l_n = int(n*len(lc_s))
            n_start = int(np.random.uniform(low=0.0, high=len(lc_s)-l_n))
            n_fin = n_start + l_n 
            lc_use = lc_s[n_start:n_fin]
            p_max = 4.*(lc_use['time'][-1] - lc_use['time'][0])
            time = np.array(lc_use["time"])
            nflux = np.array(lc_use["nflux_dtr"])
            p1, r1, _,_,_ = np.polyfit(time, nflux, 1, full=True)
            nflux_n = nflux/np.polyval(p1, time)

            enflux = np.array(lc_use["nflux_err"])

            ls = LombScargle(time, nflux_n, dy=enflux)

            frequency, power = ls.autopower(minimum_frequency=1./p_max,
                                            maximum_frequency=1./p_min,
                                            samples_per_peak=50)
            p_m = np.argmax(power)
            period_max = 1.0/frequency[p_m]
            power_max = power[p_m]
            period_arr.append(period_max)
        except:
            continue
    periods_out = np.array(period_arr)
    return periods_out
    

def plotticks_shuffle(crit, xpos, ypos, ax):
    '''Simple function for plotting tick marks on the shuffle plots

    parameters
    ----------
    crit : `bool`
        True or False according to a given criteria
    xpos : `float`
        x-position for the text (in normalised coordinates)
    ypos : `float`
        y-position for the text (in normalised coordinates)
    ax : `matplotlib.pyplot.axes`
        the axes object to apply the tickmarks to.
        
    returns
    -------
    ax : `matplotlib.pyplot.axes`
        the axes object to apply the tickmarks to (after the tickmarks)  
    '''
    if crit:
        ax.text(xpos, ypos, "\u2714", fontsize=30, color='green',
                transform=ax.transAxes, horizontalalignment='right')
    else:
        ax.text(xpos, ypos, "\u2718", fontsize=30, color='red',
                transform=ax.transAxes, horizontalalignment='right')
    return ax


def shuffle_check(cln_lc, LS_dict, shuf_per=False, n_shuf_runs=5000,
                  p_min=0.05, p_max=100., n_min=.1, n_max=1., bin1=50,
                  bin2_fac=10, n_peaks=4, make_shuf_plot=False,
                  shuf_dir='plot_shuf',
                  name_shuf_plot='example_shuf_plot.png'):
    '''Choose the period from original periodogram, or from the shuffled method

    In the case of low signal to noise, the original periodogram analysis
    will often predict an incorrect period because the noise is too dominant.
    Therefore, an alternative period can be incorporated, which is capable of
    detecting periods in noisy data.
    
    The idea is that a large number of periods are calculated from smaller
    portions of the whole lightcurve, and then if the resulting distribution
    of periods is small enough, then the shuffled period measurement is used as
    the main period.
    
    The algorithm works as follows:
    | 1. run the shuffle_periodogram function.
    | 2. construct the period histogram from the results.
    | 3. find the highest population bin, and select all neighbouring (period)
         values either side until this value becomes less than the median.
    | 4. calculate the number of periods that lie within and outside the period
         range found in part (2)
    | 5. construct another histogram for all periods that are within the range
         calculated in part (2)
    | 6. normalise the histogram so the whole distribution integrates to 1.0.
    | 7. ensure that the number of bins in the new histogram must be greater
         than 3, the number of periods outside the range is less than 0.5, and
         that the histogram must not peak at the start or end points.
         The process only continues if these conditions pass. Otherwise the
         shuffled periodogram value is not returned.
    | 8. fit a Gaussian to the new histogram.
    | 9. calculate "rrmse", the relative root mean square error.
    | 10. if rrmse < 0.5, the FWHM to the Gaussian fit is < 0.05, and the final
          shuffled period (centroid of the Gaussian) differs from the original
          period measurement by more than 10%, then return the shuffled
          periodogram result as the determined period. We set the power
          output=1.0, and the uncertainty is given by the sigma-value
          calculated in the Gaussian fit.
    | 11. finally, replace each set of nth "period, power and error" with the
          (n+1)th set, so the output set from the shuffled periodogram values
          take the highest significance. 

    parameters
    ----------
    cln_lc : `dict`
        The lightcurve data, with any lines containing a False flag removed.
    LS_dict : `dict`
        The dictionary of periodogram results produced in initialise_LS_dict
        and modified by get_periodogram_peaks
    shuf_per : `bool`, optional, default=False
        Choose to run the shuffled period analysis (True=yes, False=no)
    n_shuf_runs : `int`, optional, default=5000
        The number of measurements to be made in shuffle_periodogram
    p_min : `float`, optional, default=0.05
        The minimum period for the shuffling method in shuffle_periodogram
    p_max : `float`, optional, default=100.
        The maximum period for the shuffling method in shuffle_periodogram
    n_min : `float`, optional, default=0.1
        The minimum fraction of a group to be used in the periodogram analysis
        in shuffle_periodogram
    n_max : `float`, optional, default=1.0
        The maximum fraction of a group to be used in the periodogram analysis
        in shuffle_periodogram
    bin1 : `int`, optional, default=50
        The number of histogram bins for the initial period distribution
    bin2_fac : `int`, optional, default=10
        The factor to use in calculating the number of histogram bins for the
        refined period distribution. The total number is the product of
        bin2_fac and the number of bins from the initial period distribution
        that are within the region surrounding the maximum bin occupancy.
    n_peaks : `int`, optional, default=4
        The number of peaks calculated in get_periodogram_peaks
    make_shuf_plot : `bool`, optional, default=False
        Choose to plot the outputs of the period distribution.
    shuf_dir : `str`, optional, default='plot_shuf'
        The name of the directory to save the plots of the shuffled period
        analysis. 
    name_plot_shuf : `str`, optional, default='example_plot_shuf.png'
        Choose the file name to save the period distribution plot.

    results
    -------
    Nothing returned, the LS_dict dictionary is updated with new parameters.
    '''
    if shuf_per:
        logger.info('running the shuffle periodogram')
        try:
#1) run the shuffle_periodogram function.
            period_arr = shuffle_periodogram(cln_lc, n_shuf_runs=n_shuf_runs,
                                             p_min=p_min, p_max=p_max,
                                             n_min=n_min, n_max=n_max)

#2) construct the period histogram from the results.
            num_log10_per1, log10_per1 = np.histogram(np.log10(period_arr),
                                                      bins=bin1)
            diff_log10_per1 = np.diff(log10_per1)[0]

#3) find the highest population bin, and select all neighbouring (period) 
#   values either side until this value becomes less than the median.
            ind_others = get_next_peak(num_log10_per1,
                                       option=5.*np.median(num_log10_per1))
            ind_log10_per = np.delete(np.arange(len(log10_per1)-1), ind_others)

#4) calculate the number of periods that lie within and outside the period
#   range found in part (2)
            n_others = np.sum(num_log10_per1[ind_others])
            n_log10 = np.sum(num_log10_per1[ind_log10_per])

            x_l = log10_per1[ind_log10_per[0]]
            x_u = log10_per1[ind_log10_per[-1]]
            n_bin = (log10_per1 > (x_l - diff_log10_per1)) & \
                    (log10_per1 < (x_u + diff_log10_per1))

#5) construct another histogram for all periods that are within the range
#   calculated in part (2)
            num_log10_per2, log10_per2 = np.histogram(np.log10(period_arr),
                                                      bins=bin2_fac*np.sum(n_bin),
                                                      range=(x_l, x_u))
            log10_per2 = np.array([(log10_per2[i]+log10_per2[i+1])/2.
                                  for i in range(len(log10_per2)-1)])

#6) normalise the histogram so the whole distribution integrates to 1.0.
            num_log10_per2 = num_log10_per2/np.sum(num_log10_per2)

#7) ensure that the number of bins in the new histogram must be greater than 3,
#   the number of periods outside the range is less than 0.5, and that the
#   histogram must not peak at the start or end of the distribution. The 
#   process only continues if these conditions pass. Otherwise the shuffled
#   periodogram value is not returned.
            crit_1, crit_2 = False, False
            crit_1a = (len(ind_log10_per) > 3)
            crit_1b = (n_log10/(n_log10+n_others) > 0.5)
            crit_1c = (np.argmax(num_log10_per2) != 0) & \
                      (np.argmax(num_log10_per2) != len(num_log10_per2)-1)
            crit_2a, crit_2b, crit_2c = False, False, False
            
            if crit_1a & crit_1b & crit_1c:
                crit_1 = True

#8) fit a Gaussian to the new histogram.
                Gauss_param_log10_per, Gauss_log10_per = get_Gauss_params_pg(log10_per2, num_log10_per2, max_power=1.2*max(num_log10_per2))

#9) calculate "rrmse", the relative root mean square error.
                rrmse = relative_root_mean_squared_error(num_log10_per2,
                                                         Gauss_log10_per)
                p_shuf = 10**(Gauss_param_log10_per[1])
                p_shufu = 10**(Gauss_param_log10_per[1] + Gauss_param_log10_per[2])
                p_shufl = 10**(Gauss_param_log10_per[1] - Gauss_param_log10_per[2])
                p_shuf_err = 0.5*(p_shufu-p_shufl)

#10) if rrmse < 0.5, the FWHM to the Gaussian fit is < 0.05, and the final shuffled period (centroid of the Gaussian) differs from the original period measurement by more than 10%, then return the shuffled periodogram result as the determined period. We set the power output=1.0, and the uncertainty is given by the sigma-value calculated in the Gaussian fit.
                crit_2a = (rrmse < 0.5)
                crit_2b = (Gauss_param_log10_per[2] < 0.05)
                crit_2c = (((p_shuf/LS_dict['period_1'])<0.9) | \
                          ((p_shuf/LS_dict['period_1'])>1.1))

                if crit_2a & crit_2b & crit_2c:
                    crit_2 = True
#11) finally, replace each set of nth "period, power and error" with the (n+1)th set, so the output set from the shuffled periodogram values take the highest significance.
                    LS_dict['shuffle_flag'] = 1
                    LS_dict['period_shuffle'] = p_shuf
                    LS_dict['period_shuffle_err'] = p_shuf_err
                    for i in range(2,n_peaks+1):
                        LS_dict[f'period_{i}'] = LS_dict[f'period_{i-1}']
                        LS_dict[f'Gauss_{i}'] = LS_dict[f'Gauss_{i-1}']
                    LS_dict['period_1'] = p_shuf
                    LS_dict['power_1'] = 1.0
                    LS_dict['Gauss_1'] = [1.0, p_shuf, p_shuf_err]
                else:
                    logger.warning(f'failed second set of criteria: 2a={crit_2a}, 2b={crit_2b}, 2c={crit_2c}')
            else:
                logger.warning(f'failed first set of criteria: 1a={crit_1a}, 1b={crit_1b}, 1c={crit_1c}')            

            if make_shuf_plot:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9,3))
                ax[0].set_xlabel(r"$\log_{10}$period [d]")
                ax[0].set_ylabel('number of trials')
                ax[0].hist(np.log10(period_arr), bins=bin1)
                ax[0].axhline(5.*np.median(num_log10_per1), linestyle='--',
                              color='darkorange', linewidth=0.5)
                ax[0].axvline(x_l, color='darkorange', linewidth=0.5)
                ax[0].axvline(x_u, color='darkorange', linewidth=0.5)
                ax[1].set_xlabel(r"$\log_{10}$period [d]")
                ax[1].set_ylabel('normalised PDF')
                ax[1].plot(log10_per2, num_log10_per2)
                
                plotticks_shuffle(crit_1a, 0.94, 0.60, ax[1])
                plotticks_shuffle(crit_1b, 0.94, 0.50, ax[1])
                plotticks_shuffle(crit_1c, 0.94, 0.40, ax[1])
                plotticks_shuffle(crit_2a, 0.99, 0.60, ax[1])
                plotticks_shuffle(crit_2b, 0.99, 0.50, ax[1])
                plotticks_shuffle(crit_2c, 0.99, 0.40, ax[1])
                if crit_2:
                    ax[1].plot(log10_per2, Gauss_log10_per, linestyle='--', color='darkorange')
                    nl = '\n'
                    ax[1].text(0.99, 0.85, f"$P_{{\\rm rot}}$ (shuf) [d]{nl}{p_shuf:.3f}+/-{p_shuf_err:.3f}", transform=ax[1].transAxes, horizontalalignment='right')
                plt.savefig(f'{shuf_dir}/{name_shuf_plot}', bbox_inches='tight')

        except:
            logger.error("An error occured with the the period shuffling method.")
            LS_dict['period_shuffle'] = -9
            LS_dict['period_shuffle_err'] = -9

def make_phase_curve(LS_dict, ls, n_sca=10):
    '''Generate the phase-folded lightcurve using the peak periodogram period.
    
    This function performs several steps:
    
    1) generate the phase-folded lightcurve
    2) use Aikake Information Criterion to determine whether a sine-fit or a
       straight line is the most appropriate.
    3) Calculate the reduced chi-squared value for the sine fit.
    4) Calculate the typical amplitude and scatter in the phase-folded
       lightcurve.
    
    parameters
    ----------
    LS_dict : `dict`
        The dictionary of periodogram results produced in initialise_LS_dict
        and modified by get_periodogram_peaks and shuffle_check
    ls : `astropy.timeseries.LombScargle`
        The Lomb-Scargle periodogram object for the given lightcurve.
    n_sca : `int`, optional, default=10
        The number of portions to split the phase-folded lightcurve.

    results
    -------
    Nothing returned, the LS_dict dictionary is updated with new parameters.
    '''
    time = LS_dict["time"]
    nflux = LS_dict["nflux"]
    enflux = LS_dict["enflux"]
    freq_best = 1./LS_dict['period_1']

    y_fit_sine = ls.model(time, freq_best)
    y_fit_sine_param = ls.model_parameters(freq_best)
    chisq_model_sine = np.sum((y_fit_sine-nflux)**2/enflux**2)/(len(nflux)-3-1)
    line_fit, _,_,_,_ = np.polyfit(time, nflux, 1, full=True)
    y_fit_line = np.polyval(line_fit, time)
    chisq_model_line = np.sum((y_fit_line-nflux)**2/enflux**2)/(len(nflux)-len(line_fit)-1)
    AIC_sine, AIC_line = 2.*3. + chisq_model_sine, 2.*2. + chisq_model_line

    tdiff = np.array(time-min(time))
    pha, cyc = np.modf(tdiff/LS_dict['period_1'])
    pha, cyc = np.array(pha), np.array(cyc)
    f = np.argsort(pha)
    p = np.argsort(tdiff/LS_dict['period_1'])

    pha_fit, nf_fit, ef_fit, cyc_fit = pha[f], nflux[f], enflux[f], cyc[f].astype(int)
    pha_plt, nf_plt, ef_plt, cyc_plt = pha[p], nflux[p], enflux[p], cyc[p].astype(int)
    try:
        pops, popsc = curve_fit(sin_fit, pha_fit, nf_fit,
                                bounds=(0, [2., 2., 1000.]))
    except Exception:
        logger.warning(Exception)
        pops, popsc = np.array([1., 0.001, 0.5]), 0
        pass

    # order the phase folded lightcurve by phase and split into N even parts.
    # find the standard deviation in the measurements for each bin and use
    # the median of the standard deviation values to represent the final scatter
    # in the phase curve.

    Ndata = len(nflux)
    yp = sin_fit(pha_fit, *pops)
    chi_sq = np.sum(((yp-pha_fit)/ef_fit)**2)/(len(pha_fit)-len(pops)-1)
    chi_sq = np.sum((yp-pha_fit)**2)/(len(pha_fit)-len(pops)-1)
    
    pha_sct = MAD(yp - nflux, scale='normal')
    fdev = 1.*np.sum(np.abs(nflux - yp) > 3.0*pha_sct)/Ndata
    sca_mean, sca_stdev = mean_of_arrays(nf_fit/yp, n_sca)
    sca_median = np.median(sca_stdev)

    LS_dict['y_fit_LS'] = y_fit_sine
    LS_dict['AIC_sine'] = AIC_sine
    LS_dict['AIC_line'] = AIC_line
    LS_dict['phase_fit_x'] = pha_fit
    LS_dict['phase_fit_y'] = yp
    LS_dict['phase_x'] = pha_plt
    LS_dict['phase_y'] = nf_plt
    LS_dict['chisq_phase'] = chi_sq
    LS_dict['phase_col'] = cyc_plt
    LS_dict['pops_vals'] = pops    
    LS_dict['amp'] = pops[1]
    LS_dict['pops_cov'] = popsc
    LS_dict['scatter'] = sca_median
    LS_dict['fdev'] = fdev
    LS_dict['Ndata'] = Ndata
    

def run_ls(lc_data, lc_type='reg', ref_name='targets', pg_dir='pg', name_pg='pg_target', n_sca=10, p_min_thresh=0.05, p_max_thresh=100., samples_per_peak=10, n_peaks=4, check_jump=False, shuf_per=False, n_shuf_runs=5000, make_shuf_plot=False, shuf_dir='shuf_plots', name_shuf_plot='example_shuf_plot.png'):
    '''Run Lomb-Scargle periodogram and return a dictionary of results.

    parameters
    ----------
    lc_data : `dict`
        A dictionary containing the lightcurve data. The keys must include
        | "time" -> The time coordinate relative to the first data point
        | "nflux" -> The detrended, cleaned, normalised flux values
        | "enflux" -> The uncertainty for each value of nflux
        | "lc_part" -> An running index describing the various contiguous sections
    lc_type : `str`, optional, default='reg'
        A label designating whether the lightcurve uses the original or CBV-corrected flux.
    ref_name : `str`, optional, default='targets'
        The reference name for each subdirectory which will connect all output
        files.
    pg_dir : `string`, optional, default='pg'
        The name of the directory to store the periodogram data
    name_pg : `string`, optional, default='pg_target'
        A file name which the periodogram output will be saved to.
    n_sca : `int`, optional, default=10
        The number of evenly-split lightcurve parts used to measure the flux scatter.
    p_min_thresh : `float`, optional, default=0.05
        The minimum period (in days) to be calculated.
    p_max_thresh : `float`, optional, default=100.
        The maximum period (in days) to be calculated.
    samples_per_peak : `int`, optional, default=10
        The number of samples to measure in each periodogram peak.
    n_peaks : `int`, optional, default=4
        The number of peaks calculated in get_periodogram_peaks
    check_jump : `bool`, optional, default=False
        Choose to check the lightcurve for jumpy data, using the "check_for_jumps"
        function.
    shuf_per : `bool`, optional, default=False
        Choose to run the shuffled period analysis (True=yes, False=no)
    n_shuf_runs : `int`, optional, default=5000
        The number of measurements to be made in shuffle_periodogram
    make_shuf_plot : `bool`, optional, default=False
        Choose to make a plot for the shuffled period analysis
    shuf_dir : `str`, optional, default='shuf_plots'
        The name of the directory to save the plots of the shuffled period analysis. 
    name_shuf_plot : `str`, optional, default='example_shuf_plot.png'
        A file name which the plots of the shuffled period analysis will be saved to.

    returns
    -------
    LS_dict : `dict`
        A dictionary of parameters calculated from the periodogram analysis. These are:
        | "median_MAD_nLC" : The median and median absolute deviation of the normalised lightcurve.
        | "jump_flag" : A flag determining if the lightcurve has sharp jumps in flux.
        | "period" : A list of period values from the periodogram analysis.
        | "power" :  A list of power values from the periodogram analysis.
        | "period_best" : The period corrseponding to the highest power output.
        | "power_best" : The highest power output.
        | "time" : The time coordinate corresponding to the normalised lightcurve.
        | "y_fit_LS" : The best fit sinusoidal function.
        | "AIC_sine" : The Aikaike Information Criterion value of the best-fit sinusoid
        | "AIC_line" : The Aikaike Information Criterion value of the best-fit linear function.
        | "FAPs" : The power output for the false alarm probability values of 0.1, 1 and 10%
        | "period_1" : The period corresponding to the highest peak
        | "power_1" : The power corresponding to the highest peak
        | "Gauss_fit_peak_parameters" : Parameters for the Gaussian fit to the highest power peak
        | "Gauss_fit_peak_y_values" : The corresponding y-values for the Gaussian fit
        | "period_around_1" : The period values covered by the Gaussian fit
        | "power_around_1" : The power values across the period range covered by the Gaussian fit
        | "period_not_1" : The period values not covered by the Gaussian fit
        | "power_not_1" : The power values across the period range not covered by the Gaussian fit
        | "period_2" : The period of the second highest peak.
        | "power_2" : The power of the second highest peak.
        | "period_3" : The period of the third highest peak.
        | "power_3" : The power of the third highest peak.
        | "period_4" : The period of the fourth highest peak.
        | "power_4" : The power of the fourth highest peak.
        | "phase_fit_x" : The time co-ordinates from the best-fit sinusoid to the phase-folded lightcurve.
        | "phase_fit_y" : The normalised flux co-ordinates from the best-fit sinusoid to the phase-folded lightcurve.
        | "phase_x" : The time co-ordinates from the phase-folded lightcurve.
        | "phase_y" : The normalised flux co-ordinates from the phase-folded lightcurve.
        | "phase_chisq" : The chi-square fit between the phase-folded lightcurve and the sinusoidal fit.
        | "phase_col" : The cycle number for each data point.
        | "pops_vals" : The best-fit parameters from the sinusoidal fit to the phase-folded lightcurve.
        | "pops_cov" : The corresponding co-variance matrix from the "pops_val" parameters.
        | "phase_scatter" : The typical scatter in flux around the best-fit.
        | "frac_phase_outliers" : The fraction of data points that are more than 3 median absolute deviation values from the best-fit.
        | "Ndata" : The number of data points used in the periodogram analysis.
    '''
    LS_dict, cln_lc, ls = initialise_LS_dict(lc_data, check_jump=check_jump)
    logger.info(f'LS dictionary successfully initialised: {name_pg}, {lc_type}.')

    write_periodogram(LS_dict, name_pg=name_pg, lc_type=lc_type, pg_dir=pg_dir)
    logger.info("Periodogram results written to file.")

    get_periodogram_peaks(LS_dict, n_peaks=n_peaks)
    logger.info(f'Top {n_peaks} peaks recorded to dictionary.')

    shuffle_check(cln_lc, LS_dict, shuf_per=shuf_per, n_shuf_runs=n_shuf_runs, p_min=p_min_thresh, p_max=p_max_thresh, n_min=1./10., n_max=1., bin1=50, bin2_fac=10, n_peaks=n_peaks, make_shuf_plot=make_shuf_plot, shuf_dir=shuf_dir, name_shuf_plot=name_shuf_plot)
    logger.info('Periodogram successfully shuffled.')

    make_phase_curve(LS_dict, ls, n_sca=n_sca)
    
    
    logger.info("Phase curve details stored to dictionary.")
    return LS_dict

##########################makeplots.py

import sys
import inspect

# Third party
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections.abc import Iterable


# initialize the logger object
#logger = logger_tessilator(__name__) 

def create_plot(im_plot, clean, LS, scc, t_table, name_target, plot_dir,
                xy_ctr=(10,10), xy_contam=None, p_min_thresh=0.1,
                p_max_thresh=50., ap_rad=1.0, sky_ann=(6.,8.), nc='nc'):
    '''Produce a plot of tessilator results.

    | This module produces a 4-panel plot displaying information from the
      tessilator analysis. These are:
    | 1) An TESS cut-out image of the target, with aperture and sky annulus.
    | 2) A power vs period plot from the Lomb-Scargle periodogram analysis.
    | 3) A lightcurve of the normalised flux.
    | 4) The phase-folded lightcurve.

    parameters
    ----------
    im_plot : `astropy.nddata.Cutout2D`
        The cut-out image of the target
    clean : `dict`
        The modified (cleaned) lightcurve after processing
    LS : `dict`
        The dictionary of parameters calculated by the Lomb-Scargle periodogram
    scc : `list`, size=3
        List containing the sector number, camera and CCD
    t_table : `astropy.table.Table`
        Table containing the input data for the target
    name_target : `str`
        The name of the target
    plot_dir : `str`
        The directory to save the plots.
    XY_ctr : `tuple`, optional, default=(10,10)
        The centroid (in pixels) of the target in the TESS image.
    XY_contam : `Iterable` or `None`, optional, default = `None`
        The pixel positions of the strongest contaminants
    p_min_thresh : `float`, optional, default=0.1
        The shortest period calculated in the Lomb-Scargle periodogram
    p_max_thresh : `float`, optional, default=50.
        The longest period calculated in the Lomb-Scargle periodogram
    ap_rad : `float`, optional, default=1.0
        The aperture radius from the aperture photometry
    sky_ann : `Iterable`, size=2, optional, default=[6.,8.]
        The inner and outer background annuli from aperture photometry  
    nc : `str`, optional, default='nc'
        Describes the type of noise correction applied to the lightcurve.

    returns
    -------
    Nothing returned. The resulting plot is saved to file.
    '''
    t_0 = clean["time"][0]
    c_1 = clean["pass_sparse"].data
    c_2 = clean["pass_clean_outlier"].data
    cln_cond = np.logical_and.reduce([
                   clean["pass_clean_scatter"],
                   clean["pass_clean_outlier"],
                   clean["pass_full_outlier"]
                   ])

    clean_orig_time = clean["time"]-t_0
    clean_orig_flux = clean["nflux_ori"]
    clean_orig_mag = clean["mag"]
    
    clean_norm_time = clean["time"][c_1]-t_0
    clean_norm_flux = clean["nflux_ori"][c_1]

    clean_detr_time = clean["time"][cln_cond]-t_0
    clean_detr_flux = clean["nflux_dtr"][cln_cond]
    mpl.rcParams.update({'font.size': 14})
    if LS["AIC_line"]+1. < LS["AIC_sine"]:
        best_fit_type = 'linear'
    else:
        best_fit_type = 'sine'
    fsize = 22.
    lsize = 0.9*fsize
    fig, axs = plt.subplots(2,2, figsize=(20,15))

    axs[0,0].set_position([0.05,0.55,0.40,0.40])
    axs[0,1].set_position([0.55,0.55,0.40,0.40])
    axs[1,0].set_position([0.05,0.3,0.90,0.2])
    axs[1,1].set_position([0.05,0.05,0.90,0.2])

    circ_aper = Circle(xy_ctr, ap_rad, linewidth=1.2, fill=False, color='r')
    circ_ann1 = Circle(xy_ctr, sky_ann[0], linewidth=1.2, fill=False, color='b')
    circ_ann2 = Circle(xy_ctr, sky_ann[1], linewidth=1.2, fill=False, color='b')

    #print(t_table)
    with np.errstate(all='ignore'):
        log_im_plot = np.log10(im_plot.data)
        image_plot = np.ma.array(log_im_plot, mask=np.isnan(log_im_plot))
    im_fig = axs[0,0].imshow(image_plot, cmap='binary')
    Gaia_name = f"{t_table['source_id'][0]}"
    targ_name = t_table['name'][0]
    fig.text(0.5,0.96,
             f"{targ_name}, Sector {scc[0]}, "
             f"Camera {scc[1]}, "
             f"CCD {scc[2]}",
             fontsize=lsize*2.0,
             horizontalalignment='center')
    axs[0,0].set_xlabel("x pixel", fontsize=fsize)
    axs[0,0].set_ylabel("y pixel", fontsize=fsize)
    axs[0,0].add_patch(circ_aper)
    axs[0,0].add_patch(circ_ann1)
    axs[0,0].add_patch(circ_ann2)
    axs[0,0].text(0.01,0.94, "$r_{\\rm ap}$ = "
                  f"{ap_rad:.2f}", color='red',
                  fontsize=lsize,horizontalalignment='left', 
                  transform=axs[0,0].transAxes)
    if isinstance(xy_contam, Iterable):
        axs[0,0].scatter(xy_contam[:, 0], xy_contam[:, 1], marker='X',
                         s=400, color='orange')
    divider = make_axes_locatable(axs[0,0])
    cax = divider.new_horizontal(size='5%', pad=0.4)
    fig.add_axes(cax)
    cbar = fig.colorbar(im_fig, cax=cax)
    cbar.set_label('log$_{10}$ counts (e$^-$/s)', rotation=270, labelpad=+15)

    axs[0,1].set_xlim([p_min_thresh, p_max_thresh])
    axs[0,1].grid(True)
    axs[0,1].set_xlabel("Period (days)", fontsize=fsize)
    axs[0,1].set_ylabel("Power", fontsize=fsize)
    axs[0,1].semilogx(LS['period_a_1'], LS['power_a_1'])
    [axs[0,1].axhline(y=i, linestyle='--', color='grey', alpha=0.8) \
     for i in LS['FAPs']]
    axs[0,1].text(0.01,0.94, f"Best fit: {best_fit_type}",
                  fontsize=lsize,horizontalalignment='left', 
                  transform=axs[0,1].transAxes)
    axs[0,1].text(0.99,0.94, "$P_{\\rm rot}^{\\rm (max)}$ = "
                  f"{LS['period_1']:.3f} days, "
                  f"power = {LS['power_1']:.3f}",
                  fontsize=lsize, horizontalalignment='right',
                  transform=axs[0,1].transAxes)
    axs[0,1].text(0.99,0.82, "$P_{\\rm rot}^{\\rm (2nd)}$ = "
                  f"{LS['period_2']:.3f}",
                  fontsize=lsize, horizontalalignment='right',
                  transform=axs[0,1].transAxes)
    axs[0,1].text(0.99,0.76, f"power ratio = "
                  f"{LS['power_1']/LS['power_2']:.3f}",
                  fontsize=lsize,horizontalalignment='right', 
                  transform=axs[0,1].transAxes)

    if (LS['Gauss_1'][1] != 15) & \
       (isinstance(LS['period_around_1'], Iterable)):
        axs[0,1].plot(LS['period_around_1'],
                      LS['Gauss_y_1'],
                      c='r', label='Best fit')
        axs[0,1].text(0.99,0.88, "$P_{\\rm rot}^{\\rm (Gauss)}$ = "
                      f"{LS['Gauss_1'][1]:.3f} $\\pm$"
                      f"{LS['Gauss_1'][2]:.3f}",
                      fontsize=lsize, horizontalalignment='right',
                      transform=axs[0,1].transAxes)    
    if LS['shuffle_flag'] > 0:
        axs[0,1].axvline(x=LS['period_1'], color='red', linewidth=3, alpha=0.3)
    axs[1,0].set_xlim([0, 30])
    axs[1,0].set_xlabel("Time (days)", fontsize=fsize)
    axs[1,0].set_ylim(
        [LS['median_MAD_nLC'][0]-(8.*LS['median_MAD_nLC'][1]),
        LS['median_MAD_nLC'][0]+(8.*LS['median_MAD_nLC'][1])])
    axs[1,0].set_ylabel("normalised flux", c='g', fontsize=fsize)
    axs[1,0].plot(LS["time"]-t_0, LS['y_fit_LS'], c='orange',
                  linewidth=1.5, label='LS best fit')
    axs[1,0].scatter(clean_orig_time, clean_orig_flux, s=1.0, c='pink',
                     alpha=0.5, label='raw, normalized')
    axs[1,0].scatter(clean_norm_time, clean_norm_flux, s=1.0, c='r', 
                     alpha=0.5, label='cleaned, normalized')
    axs[1,0].scatter(clean_detr_time, clean_detr_flux, s=1.2, c='g',
                     alpha=0.7, label='cleaned, normalized, detrended')
    if LS['jump_flag']:
        axs[1,0].text(0.01,0.90, 'Jumps detected', fontsize=lsize,
                      horizontalalignment='left',
                      transform=axs[1,0].transAxes)
    if LS["CBV_flag"] == 1:
        axs[1, 0].text(
            0.01,
            0.01,
            "best fit: original flux",
            fontsize=lsize,
            horizontalalignment="left",
            transform=axs[1, 0].transAxes,
        )
    if LS["CBV_flag"] == 2:
        axs[1, 0].text(
            0.01,
            0.01,
            "best fit: CBV corrected flux",
            fontsize=lsize,
            horizontalalignment="left",
            transform=axs[1, 0].transAxes,
        )
    axs[1,0].text(0.99,0.90, Gaia_name, fontsize=lsize,
                  horizontalalignment='right',
                  transform=axs[1,0].transAxes)
    axs[1,0].text(0.99,0.80, f"Gmag = {float(t_table['Gmag']):.3f}",
                  fontsize=lsize, horizontalalignment='right',
                  transform=axs[1,0].transAxes)
    axs[1,0].text(0.99,0.70, "$\log (f_{\\rm bg}/f_{*})$ = "
                  f"{float(t_table['log_tot_bg']):.3f}", fontsize=lsize,
                  horizontalalignment='right', transform=axs[1,0].transAxes)
    leg = axs[1,0].legend(loc='lower right')
    leg.legendHandles[1]._sizes = [30]
    leg.legendHandles[2]._sizes = [30]
    leg.legendHandles[3]._sizes = [30]
    ax2=axs[1,0].twinx()
    ax2.set_position([0.05,0.3,0.90,0.2])
    ax2.invert_yaxis()

    if not np.all(clean_orig_mag.data == -999.):
        ax2.scatter(clean_orig_time[clean_orig_mag>-999],
                    clean_orig_mag[clean_orig_mag>-999],
                    s=0.3, alpha=0.3, color="b", marker="x")
        ax2.set_ylabel("TESS magnitude", c="b",fontsize=fsize)

    axs[1,1].set_xlim([0,1])
    axs[1,1].set_xlabel("phase", fontsize=fsize)
    axs[1,1].set_ylabel("normalised flux", fontsize=fsize)
    axs[1,1].plot(LS['phase_fit_x'], LS['phase_fit_y'], c='b')
    LS["phase_col"] += 1
    N_cyc = int(max(LS["phase_col"]))
    cmap_use = plt.get_cmap('rainbow', N_cyc)
    s = axs[1,1].scatter(LS['phase_x'], LS["phase_y"],
                         c=LS['phase_col'], cmap=cmap_use, vmin=0.5,
                         vmax=N_cyc+0.5)
#    axs[1,1].text(0.01, 0.90, f"Amplitude = {LS['amp']:.3f}, "
#                  f"Scatter = {LS['scatter']:.3f}, "
#                  f"$\chi^{2}$ = {LS['chisq_phase']:.3f}, "
#                  "$f_{\\rm dev}$"+ f"= {LS['fdev']:.3f}",
#                  fontsize=lsize, horizontalalignment='left',
#                  transform=axs[1,1].transAxes)

#    cbaxes = inset_axes(axs[1,1], width="100%", height="100%",
#                        bbox_to_anchor=(0.79, 0.92, 0.20, 0.05),
#                        bbox_transform=axs[1,1].transAxes)
#    cbar = plt.colorbar(s, cax=cbaxes, orientation='horizontal',
#                        label='cycle number')
    plot_name = '_'.join([name_target, f"{scc[0]:04d}",
                          f"{scc[1]}", f"{scc[2]}", f"{nc}"])+'.png'
    plt.savefig(f'./{plot_dir}/{plot_name}', bbox_inches='tight')
    plt.close('all')

#######################maketable.py

import warnings
import sys
import inspect

#Third party
from astropy.table import Table
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord, ICRS
import astropy.units as u
import numpy as np

# initialize the logger object
#logger = logger_tessilator(__name__) 

Vizier.ROW_LIMIT = 1  # or -1 for unlimited
gaia_catalog = "I/355/gaiadr3"  # Gaia DR3 catalog in Vizier

Vizier.TIMEOUT = 120
Vizier.VIZIER_SERVER = "vizier.cds.unistra.fr"


def table_from_simbad(input_names):
    '''Generate the formatted astropy table from a list of target names.

    All characters can be parsed except commas, since the table is in comma
    separated variable (.csv) format.

    parameters
    ----------
    input_names : `astropy.table.Table`
        An input list of target names.

    returns
    -------
    gaia_table : `astropy.table.Table`
        The output table ready for further analysis.
    '''
    # Part 1: Use the SIMBAD database to retrieve the Gaia source identifier
    #         from the target names. 
    # set the column header = "ID"
    input_names.rename_column(input_names.colnames[0], 'ID')
    input_names["ID"] = input_names["ID"].astype(str)
    # create arrays to store naming variables
    name_arr, is_Gaia = [], [0 for i in input_names]
    for i, input_name in enumerate(input_names["ID"]):
    # if the target name is the numeric part of the Gaia DR3 source identifier
    # prefix the name with "Gaia DR3 "
        if input_name.isnumeric() and len(input_name) > 10:
            input_name = "Gaia DR3 " + input_name
            is_Gaia[i] = 1
        name_arr.append(input_name)

    # Get a list object identifiers from Simbad
    # suppress the Simbad.query_objectids warnings if there are no matches for
    # the input name
#    with warnings.catch_warnings():
#        warnings.simplefilter(action='ignore', category=UserWarning)
        try:
            result_table = [Simbad.query_objectids(name) for name in name_arr]
        except:
            result_table = [None for name in name_arr]
    NameList = []
    GaiaList = []
    for r, res in enumerate(result_table):
        input_name = input_names["ID"][r]
        if res is None: # no targets resolved by SIMBAD
            logger.warning(f"Simbad did not resolve {input_name} - checking "
                           f"Gaia")
            if is_Gaia[i] == 1:
                NameList.append("Gaia DR3 " + input_name)
                GaiaList.append(input_name)
            else:
                logger.error(f"Could not find any match for '{input_name}'")
        else: # Simbad returns at least one identifier
            r_list = [z for z in res["id"]]
            m = [s for s in r_list if "Gaia DR3" in s]
            if len(m) == 0: # if Gaia identifier is not in the Simbad list
                if is_Gaia[i] == 1:
                    logger.warning("Simbad didn't resolve Gaia DR3 "
                                   f"identifiers for {input_name}, "
                                   f"but we'll check anyway!")
                    NameList.append("Gaia DR3 " + input_name)
                    GaiaList.append(input_name)
                else:
                    logger.error(f"Could not find any match for "
                                 f"'{input_name}'")
            else:
                NameList.append(input_name)
                GaiaList.append(m[0].split(' ')[2])
    if len(NameList) == 0:
        logger.error(
            "No targets have been resolved, either by Simbad or "
            "Gaia DR3. Please check the target names are "
            "resolvable."
        )
        sys.exit()
    # Part 2: Query Gaia database using Gaia identifiers retrieved in part 1.
    ID_string = ""
    for g_i, gaia_name in enumerate(GaiaList):
        if g_i == len(GaiaList)-1:
            ID_string += gaia_name
        else:
            ID_string += gaia_name+','
    qry = "SELECT source_id,ra,dec,parallax,"\
          "phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag "\
          "FROM gaiadr3.gaia_source "\
          f"WHERE source_id in ({ID_string});"
    job = Gaia.launch_job_async( qry )
    gaia_table = job.get_results() # Astropy table
    logger.info('query completed!')
    # convert source_id column to str (astroquery returns type np.int64)
    gaia_table["source_id"] = gaia_table["source_id"].astype(str)
    list_ind = []
    # astroquery returns the table sorted numerically by the source identifier
    # the rows are rearranged to match with the input list.
    for row in GaiaList:
        list_ind.append(np.where(np.array(gaia_table["source_id"] == \
                        str(row)))[0][0])
    gaia_table = gaia_table[list_ind]
    gaia_table["source_id"] = [f"Gaia DR3 {i}" for i in gaia_table["source_id"]]
    gaia_table['name'] = NameList
    gaia_table.rename_column('phot_g_mean_mag', 'Gmag')
    gaia_table.rename_column('phot_bp_mean_mag', 'BPmag')
    gaia_table.rename_column('phot_rp_mean_mag', 'RPmag')
    new_order = ['name', 'source_id', 'ra', 'dec', 'parallax',
                 'Gmag', 'BPmag', 'RPmag']
    gaia_table = gaia_table[new_order]
    return gaia_table


def get_twomass_like_name(coords):
    '''If the Gaia DR3 system is not chosen, this function returns a string
    which has the same format as the 2MASS identifiers.
    
    parameters
    ----------
    coords : `astropy.coordinates.SkyCoord`
         The SkyCoord tuple of right ascencion and declination values.
         
    returns
    -------
    radec_fin : `list`
        A list of 2MASS-like identifiers.
    '''
    ra_hms = coords.ra.to_string(u.h, sep="", precision=2, alwayssign=False,
                                 pad=True)
    ra_hms_fin = [ra.replace(".","") for ra in ra_hms]
    dec_hms = coords.dec.to_string(sep="", precision=1, alwayssign=True,
                                   pad=True)
    dec_hms_fin = [dec.replace(".","") for dec in dec_hms]
    radec_fin = []
    for r,d in zip(ra_hms_fin, dec_hms_fin):
        radec_fin.append(f'{r}{d}')
    return radec_fin


def table_from_coords(coord_table, ang_max=10.0, type_coord='icrs',
                      gaia_sys=True):
    """Generate the formatted astropy table from a list of coordinates.

    Each entry needs to be in comma separated variable(.csv) format.

    parameters
    ----------
    coord_table : `astropy.table.Table`
        A table with two columns named 'col1' and 'col2'. If the coordinates
        are in the 'icrs' system, the columns should contain the right
        ascension and declination values in degrees. If the coordinates are in
        the 'galactic' or 'ecliptic' system, the columns contain the longitude and
        latitude in degrees.
    ang_max : `float`, optional, default=10.0
        the maximum angular distance in arcseconds from the input coordinates
        provided in the table.
    type_coord : `str`, optional, default='icrs'
        The coordinate system of the input positions. These can be 'icrs'
        (default), 'galactic' or 'ecliptic'.
    gaia_sys : `bool`, optional, default=True
        Choose to format the data based on Gaia DR3. Note that no contamination
        can be calculated if this is False.

    returns
    -------
    gaia_table : `astropy.table.Table`
        The output table ready for further analysis.
    """
    gaia_table = Table(names=('source_id', 'ra', 'dec', 'parallax',
                              'Gmag', 'BPmag', 'RPmag'), \
                       dtype=(int,float,float,float,float,float,float))
    if type_coord == 'galactic':
        gal = SkyCoord(l=coord_table['col1'],\
                       b=coord_table['col2'],\
                       unit=u.deg, frame='galactic')
        c = gal.transform_to(ICRS)
        coord_table['col1'], coord_table['col2'] = c.ra.deg, c.dec.deg
    elif type_coord == 'ecliptic':
        ecl = SkyCoord(lon=coord_table['col1'],\
                       lat=coord_table['col2'],\
                       unit=u.deg, frame='barycentricmeanecliptic')
        c = ecl.transform_to(ICRS)
    elif type_coord == 'icrs':
        c = SkyCoord(ra=coord_table['col1'], dec=coord_table['col2'],
                     unit=u.deg, frame='icrs')
        coord_table['col1'], coord_table['col2'] = c.ra.deg, c.dec.deg
    coord_table.rename_column(coord_table.colnames[0], 'ra')
    coord_table.rename_column(coord_table.colnames[1], 'dec')

    if gaia_sys:
        for i in range(len(coord_table)):
        # Generate an SQL query for each target, where the nearest source is
        # returned within a maximum radius set by ang_max.
            qry = f"SELECT source_id,ra,dec,parallax,phot_g_mean_mag,\
                    phot_bp_mean_mag,phot_rp_mean_mag, \
                    DISTANCE(\
                    POINT({coord_table['ra'][i]}, {coord_table['dec'][i]}),\
                    POINT(ra, dec)) AS ang_sep\
                    FROM gaiadr3.gaia_source \
                    WHERE 1 = CONTAINS(\
                    POINT({coord_table['ra'][i]}, {coord_table['dec'][i]}),\
                    CIRCLE(ra, dec, {ang_max}/3600.)) \
                    ORDER BY ang_sep ASC"
            job = Gaia.launch_job_async( qry )
            x = job.get_results() # Astropy table
            time.sleep(1.5)
            print(f"astroquery completed for target {i+1} of "
                  f"{len(coord_table)}")
            # Fill the empty table with results from astroquery
            if len(x) == 0:
                continue
            else:
                y = x[0]['source_id', 'ra', 'dec', 'parallax',
                         'phot_g_mean_mag', 'phot_bp_mean_mag',
                         'phot_rp_mean_mag']
                gaia_table.add_row((y))
        # For each source, query the identifiers resolved by SIMBAD and return
        # the target with the shortest number of characters (which is more
        # likely to be the most common reference name for the target).
        GDR3_Names = ["Gaia DR3 " + i for i in \
                      gaia_table['source_id'].astype(str)]
        try:
            result_table = [Simbad.query_objectids(i) for i in GDR3_Names]
        except:
            result_table = [None for i in GDR3_Names]
        NameList = []

        for i, r in enumerate(result_table):
            if r is None:
                NameList.append(gaia_table['source_id'][i].astype(str))
            else:
                NameList.append(sorted(r["id"], key=len)[0])
        gaia_table["name"] = NameList
        gaia_table["source_id"] = GDR3_Names
    else:
        twomass_name = get_twomass_like_name(c)
        for i in range(len(twomass_name)):
            source_id = f'{i+1:0{len(str(len(twomass_name)))}d}'
            row = [source_id,c[i].ra.deg,c[i].dec.deg,-999,-999,-999,-999]
            gaia_table.add_row(row)
        gaia_table['name'] = twomass_name

    new_order = ['name', 'source_id', 'ra', 'dec', 'parallax',
                 'Gmag', 'BPmag', 'RPmag']
    gaia_table = gaia_table[new_order]
    return gaia_table


def table_from_table(input_table, name_is_source_id=False, ang_max=10.):
    '''Generate the formatted astropy table from a pre-formatted astropy
    table.

    Each entry needs to be in comma separated variable(.csv) format. This
    is the quickest way to produce the table ready for analysis, but it is
    important the input data is properly formatted.
    
    parameters
    ----------
    input_table : `astropy.table.Table`
        The columns of table must be in the following order:
        
        * source_id (data type: `str`)

        * ra (data type: `float`)

        * dec (data type: `float`)

        * parallax (data type: `float`)

        * Gmag (data type: `float`)

        * BPmag (data type: `float`)

        * RPmag (data type: `float`)

        The column headers must not be included!
    name_is_source_id : `bool`, optional, default=False
        Choose if the name is to be the same as the Gaia DR3 source identifier.
    ang_max : `float`, optional, default=10.0
        the maximum angular distance in arcseconds from the input coordinates
        provided in the table.

    returns
    -------
    gaia_table : `astropy.table.Table`
        The output table ready for further analysis.
    '''

    gaia_table = Table(data=input_table, dtype=(str, float, float, float,
                                                float, float, float),
                       names=('source_id', 'ra', 'dec', 'parallax', 'Gmag',
                              'BPmag', 'RPmag'))
#    gaia_table['source_id'] = gaia_table['source_id']
    if name_is_source_id:
        gaia_table['name'] = gaia_table['source_id'].data
    else:
        GDR3_Names = [i for i in\
                      gaia_table['source_id']]
        source_list = []
        coords = SkyCoord(ra=gaia_table["ra"]*u.degree,
                         dec=gaia_table["dec"]*u.degree,
                         frame='icrs')
#        print(GDR3_Names)
        
        for c in coords:
            result = Vizier.query_region(c, radius=ang_max*u.arcsec, catalog=gaia_catalog)

            if result:
                source_list.append(f"Gaia DR3 {result[0][0]['Source']}")
            else:
                source_list.append(None)
        
#        try:
#            result_table =  [Simbad.query_objectids(i) for i in GDR3_Names]
#        except:
#            result_table = [None for i in GDR3_Names]
#        for i, r in enumerate(result_table):
#            if r is None:
#                NameList.append(str(gaia_table['source_id'][i]))
#            else:
#                NameList.append(sorted(r["id"], key=len)[0])
#        print(NameList)
        gaia_table["name"] = GDR3_Names
        gaia_table["source_id"] = source_list
    new_order = ['name', 'source_id', 'ra', 'dec', 'parallax', 'Gmag',
                 'BPmag', 'RPmag']
    gaia_table = gaia_table[new_order]
    return gaia_table


def get_gaia_data(gaia_table, name_is_source_id=False, type_coord='icrs',
                  gaia_sys=True):
    """Reads the input table and returns a table in the correct format for
    TESSilator.

    The table must be in comma-separated variable format, in either of these
    3 ways:

    1. A table with a single column containing the source identifier
       Note that this is the preferred method since the target identified
       in the Gaia query is unambiguously the same as the input value.
       Also, the name match runs faster than the coordinate match using
       astroquery.

    2. A table with sky-coordinates in either the 'icrs' (default),
       'galactic', or 'ecliptic' system.
       * note this is slower because of the time required to run the Vizier
       query.

    3. A table with all 7 columns already made.

    Parameters
    ----------
    gaia_table : `astropy.table.Table`
        The input table
    name_is_source_id : `bool`, optional, default=False
        If the input table has 7 columns, this provides the choice to set the
        name column equal to "source_id" (True), or to find a common target
        identifier (False)
    type_coord : `str`, optional, default='icrs'
        The coordinate system of the input data. Choose from 'icrs', 'galactic'
        or 'barycentricmeanecliptic', where the latter is the conventional
        coordinate system used by TESS.
    gaia_sys : `bool`, optional, default=True
        Choose to format the data based on Gaia DR3. Note that no contamination
        can be calculated if this is False.

    Returns
    -------
    tbl : `astropy.table.Table`
        The table ready for TESSilator analysis, with the columns:

        * name: the preferred choice of source identifier

        * source_id: the Gaia DR3 source identifier

        * ra: right ascension (icrs) or longditude (galactic,
          barycentricmeanecliptic)

        * dec: declination (icrs) or latitude (galactic,
          barycentricmeanecliptic)

        * parallax: parallax from Gaia DR3 (in mas)

        * Gmag: the apparent G-band magnitude from Gaia DR3

        * BPmag: the apparent BP-band magnitude from Gaia DR3

        * RPmag: the apparent RP-band magnitude from Gaia DR3

    """
    
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    
    if len(gaia_table.colnames) == 1:
        tbl = table_from_simbad(gaia_table)
    elif len(gaia_table.colnames) == 2:
        tbl = table_from_coords(gaia_table, type_coord=type_coord,
                                gaia_sys=gaia_sys)
    elif len(gaia_table.colnames) == 7:
        tbl = table_from_table(gaia_table, name_is_source_id=name_is_source_id)
    else:
        raise Exception(
            "Input table has invalid format. Please use one of "
            "the following formats: \n [1] source_id \n [2] ra "
            "and dec\n [3] source_id, ra, dec, parallax, Gmag, "
            "BPmag and RPmag"
        )
    return tbl

import warnings
import inspect
import sys


# Third party 
import numpy as np
import os
import subprocess

from astropy.io import fits

import pylab as pl


# Local application
import scipy.linalg
import scipy.special

# initialize the logger object
logger = logger_tessilator(__name__)

def logdet(a):
    '''
    Compute log of determinant of matrix a using Cholesky decomposition
    '''
    # First make sure that matrix is symmetric:
    if not np.allclose(a.T, a):
        print('MATRIX NOT SYMMETRIC')
    # Second make sure that matrix is positive definite:
    eigenvalues = scipy.linalg.eigvalsh(a)
    if min(eigenvalues) <=0:
        print('Matrix is NOT positive-definite')
        print('   min eigv = %.16f' % min(eigenvalues))
    step1 = scipy.linalg.cholesky(a)
    step2 = np.diag(step1.T)
    out = 2. * np.sum(np.log(step2), axis=0)
    return out


def bayes_linear_fit_ard(X, y):
    '''
    Fit linear basis model with design matrix X to data y.
    
    Calling sequence:
    w, V, invV, logdetV, an, bn, E_a, L = bayes_linear_fit_ard(X, y)
    
    Inputs:
    X: design matrix
    y: target data
    
    Outputs
    w: basis function weights
    ***need to document the others!***
    '''
    # uninformative priors
    a0 = 1e-2
    b0 = 1e-4
    c0 = 1e-2
    d0 = 1e-4
    # pre-process data
    [N, D] = X.shape
    X_corr = X.T * X
    Xy_corr = X.T * y    
    an = a0 + N / 2.    
    gammaln_an = scipy.special.gammaln(an)
    cn = c0 + 1 / 2.    
    D_gammaln_cn = D * scipy.special.gammaln(cn)
    # iterate to find hyperparameters
    L_last = -sys.float_info.max
    max_iter = 500
    E_a = np.matrix(np.ones(D) * c0 / d0).T
    #print('bayesloop begins')
    for _ in range(max_iter):
        #print(_)
        # covariance and weight of linear model
        #print(np.array(E_a)[:,0].shape,X_corr.shape)
        invV = np.matrix(np.diag(np.array(E_a)[:,0])) + X_corr   
        #print('a')
        V = np.matrix(scipy.linalg.inv(invV))
        #print('b')
        logdetV = -logdet(invV)    
        #print('c')
        w = np.dot(V, Xy_corr)[:,0]
        # parameters of noise model (an remains constant)
        #print('d')
        sse = np.sum(np.power(X*w-y, 2), axis=0)
        #print('e')
        if np.imag(sse)==0:
            sse = np.real(sse)[0]
            #print('f')
        else:
            print('Something went wrong')
        bn = b0 + 0.5 * (sse + np.sum((np.array(w)[:,0]**2) * np.array(E_a)[:,0], axis=0))
        #print('g')
        E_t = an / bn
        #print('h')
        # hyperparameters of covariance prior (cn remains constant)
        dn = d0 + 0.5 * (E_t * (np.array(w)[:,0]**2) + np.diag(V))
        #print('i')
        E_a = np.matrix(cn / dn).T
        # variational bound, ignoring constant terms for now
        
        # print('j')
        # print("E_t:", np.shape(E_t))
        # print("sse:", np.shape(sse))
        # print("X:", np.shape(X))
        # print("V:", np.shape(V))
        # print("X*V:", np.shape(X @ V))                                                                           
        # print("scipy.multiply(X, X*V):", np.shape(np.multiply(X, X @ V)))
        # print("np.sum(scipy.multiply(...)):", np.shape(np.sum(np.multiply(X, X @ V))))
        # print("logdetV:", np.shape(logdetV))
        # print("b0:", np.shape(b0))
        # print("gammaln_an:", np.shape(gammaln_an))
        # print("an:", np.shape(an))
        # print("bn:", np.shape(bn))
        # print("scipy.log(bn):", np.shape(np.log(bn)))
        # print("D_gammaln_cn:", np.shape(D_gammaln_cn))
        # print("cn:", np.shape(cn))
        # print("dn:", np.shape(dn))
        # print("scipy.log(dn):", np.shape(np.log(dn)))
        # print("np.sum(scipy.log(dn)):", np.shape(np.sum(np.log(dn))))

        L = -0.5 * (E_t*sse + np.sum(np.multiply(X,X@V))) + 0.5 * logdetV - b0 * E_t + gammaln_an - an * np.log(bn) + an + D_gammaln_cn - cn * np.sum(np.log(dn))
        # variational bound must grow!
        #print('k')
        if L_last > L:
            # if this happens, then something has gone wrong....
            #print('l')
            file = open('ERROR_LOG','w')
            file.write('Last bound %6.6f, current bound %6.6f' % (L, L_last))
            file.close()
            raise Exception('Variational bound should not reduce - see ERROR_LOG')
            return
        # stop if change in variation bound is < 0.001%
        if abs(L_last - L) < abs(0.00001 * L):        
            #print('m')
            break
        # print(L, L_last)
        L_last = L
        #print('n')
    if _ == max_iter:    
        warnings.warn('Bayes:maxIter ... Bayesian linear regression reached maximum number of iterations.') 
    # augment variational bound with constant terms

    
    # # Term 1: 0.5 * (N * np.log(2 * np.pi) - D)
    # term_1 = 0.5 * (N * np.log(2 * np.pi) - D)
    # print("Shape of term_1:", term_1.shape)
    
    # # Term 2: -scipy.special.gammaln(a0)
    # term_2 = -scipy.special.gammaln(a0)
    # print("Shape of term_2:", term_2.shape)
    
    # # Term 3: a0 * np.log(b0)
    # term_3 = a0 * np.log(b0)
    # print("Shape of term_3:", term_3.shape)
    
    # # Term 4: D * (-scipy.special.gammaln(c0) + c0 * np.log(d0))
    # term_4 = D * (-scipy.special.gammaln(c0) + c0 * np.log(d0))
    # print("Shape of term_4:", term_4.shape)
    
    # Final calculation of L
    # L = L - term_1 - term_2 + term_3 + term_4
    # print("Shape of L:", L.shape)
    L = L - 0.5 * (N * np.log(2 * np.pi) - D) - scipy.special.gammaln(a0) + a0 * np.log(b0) + D * (-scipy.special.gammaln(c0) + c0 * np.log(d0))
    # print('o')
    return w, V, invV, logdetV, an, bn, E_a, L


def savgol_filter(t, f, window = 2.0, order = 2):
    '''The savgol filter'''
    n = len(t)
    f_sm = np.zeros(n) + np.nan
    for i in np.arange(n):
        l = (abs(t-t[i]) <= window/2.)
        if l.sum() == 0:
            continue
        tt = t[l]
        ff = f[l]
        p = np.polyval(np.polyfit(tt, ff, order), t[l])
        j = np.where(tt == t[i])[0]
        f_sm[i] = p[j]
    return f_sm


def block_mean(t, f, block_size = 13, block_size_min = None):
    '''calculate the block mean'''
    block_size = int(block_size)
    if block_size_min is None:
        block_size_min = block_size / 2 + 1
    n = len(t)
    dt = np.median(t[1:]-t[:-1])
    t_blocks = []
    f_blocks = []
    i = 0
    while t[i] < t[-1]:
        j = np.copy(i)
        while (t[j] - t[i]) < (block_size * dt):
            j+=1
            if j >= n:
                break
        if j >= (i + block_size_min):
            t_blocks.append(t[i:j].mean())
            f_blocks.append(f[i:j].mean())
        i = np.copy(j)
        if i >= n:
            break
    t_blocks = np.array(t_blocks)
    f_blocks = np.array(f_blocks)
    return t_blocks, f_blocks


def cdpp(time, flux, dfilt = 2.0, bl_sz = 13, exclude=None, plot = False):
    '''cdpp'''
    m = np.isfinite(time) & np.isfinite(flux)
    if exclude is not None:
        assert exclude.size == m.size, 'Exclusion mask size != time and flux array size'
        m &= ~exclude
    t,f = time[m], flux[m] / flux[m].mean()

    # filter out long-term variations
    f_sm = savgol_filter(t, f, dfilt)
    f_res = f - f_sm

    # exclude obvious outliers
    m2, s2 = f_res.mean(), f_res.std()
    l2 = abs(f_res-m2) < (5 * s2)

    # compute bin-averaged fluxes
    t_b, f_b = block_mean(t[l2], f_res[l2], block_size = bl_sz)
    cdpp = f_b.std() * 1e6

    if plot:
        pl.clf()
        pl.subplot(211)
        pl.plot(time, flux/flux[m].mean(), '.', c = 'grey', mec = 'grey')
        pl.plot(t, f, 'k.')
        pl.plot(t, f_sm, 'r-')
        pl.xlim(t.min(), t.max())
        pl.subplot(212)
        pl.plot(t, f_res * 1e6, '.', c = 'grey', mec = 'grey')
        pl.plot(t[l2], f_res[l2] * 1e6, 'k.')
        pl.plot(t_b, f_b * 1e6, 'b.')
        pl.xlim(t.min(), t.max())

    return cdpp


def medransig(array):
    '''calculate the medransig'''
    l = np.isfinite(array)
    med = np.median(array[l])
    norm = array / med
    norm_s = np.sort(norm)
    ng = l.sum()
    ran = norm_s[int(ng*0.95)] - norm_s[int(ng*0.05)]
    diff = norm[1:] - norm[:-1]
    l = np.isfinite(diff)
    sig = 1.48 * np.median(np.abs(diff[l]))
    return med, ran * 1e6, sig * 1e6


def fit_basis(flux, basis, scl=None):
    '''Calculate the flux-correction weights for the co-trending basis vectors (CBVs).
    
    This routine is taken directly from `Aigrain et al. 2017 <https://github.com/saigrain/CBVshrink>`_

    parameters
    ----------
    flux : `iterable`
        The list of normalised flux values from the target lightcurve.
    basis : `iterable`
        The list of co-trending basis vectors.
    scl : `float`, optional, default='None'
        An additional scaling factor for the CBVs.

    returns
    -------
    weights : `iterable`
        The weights assigned for each lightcurve data point.
    '''
    # pre-process basis
    nb,nobs = basis.shape
    #print('nb&nobs',nb,nobs)
    B = np.matrix(basis.T)
    #print('Bshape',B.shape)
    if scl is None:
        scl = np.ones(nb)
    Bnorm = np.multiply(B, scl)
    #print('Bnorm.shape',Bnorm.shape)
    Bs = Bnorm.std()
    #print('Bs',Bs)
    Bnorm /= Bs
    #print('Bnorm.shape',Bnorm.shape)
    Bnorm = np.concatenate((Bnorm, np.ones((nobs,1))), axis=1)
    #print('Bnorm_concat.shape',Bnorm.shape)
    # array to store weights
    nobj = flux.shape[0]
    weights = np.zeros((nobj,nb))
    #print('loopbegins')
    for iobj in np.arange(nobj): 
        # pre-process flux
        F = np.matrix(flux[iobj,:]).T
        #print('Fshape',F.shape)
        l = np.isfinite(F)
        #print('l',l)
        Fm = F.mean()
        Fs = F.std()
        Fnorm = (F - Fm) / Fs
        #print('bayes begins')
        res = bayes_linear_fit_ard(Bnorm, Fnorm)
        #print(iobj)
        w, V, invV, logdetV, an, bn, E_a, L = res
        weights[iobj,:] = np.array(res[0][:-1]).flatten() * scl * Fs / Bs
    #print('loopends')
    return weights


def apply_basis(weights, basis):
    '''Calculate the dot product between the weights and the CBVs.
    
    parameters
    ----------
    weights : `Iterable`
        The weights for each lightcurve data point.
    basis : `Iterable`
        The CBVs for each data point.
    
    returns
    -------
        dot_prod_res : `Iterable`
            The dot product between the weights and the basis vectors.
        corr: (nobj x nobs) correction to apply to light curves
    '''

    dot_prod_res = np.dot(weights, basis)
    return dot_prod_res


def fixed_nb(flux, cbv, nB=4, use=None, doPlot=True):
    '''Correct light curve for systematics using first nB CBVs.

    parameters
    ----------
    flux : `Iter`
        The 1-D array of light curves 
    cbv : `Iter`
        The 2-D array of co-trending basis vectors trends
    nb : `int`, optional, default=4
        The number of CBVs to use (the first nB are used)
    use : `bool`, optional, default=True
        True for data points to use in evaluating correction, False for data points to ignore (NaNs are also ignored)
    doPlot : `bool`, optional, default=True
        Choose whether to produce a plot of the CBV corrections.

    returns
    -------
    corrected_flux : `Iter`
        The corrected light curves, with the same shape as flux
    weights : `Iter`
        An nB-sized array containing the basis vector coefficients
    '''
    nobs = len(flux)
    if cbv.shape[1] == nobs:
        cbv_ = cbv[:nB, :]
    else:
        cbv_ = cbv[:, :nB].T
    corrected_flux = np.copy(flux)
    l = np.isfinite(flux)
    if use is not None:
        l *= use
    weights = fit_basis(flux[l].reshape((1,l.sum())), cbv_[:,l])
    corr = apply_basis(weights, cbv_).reshape(flux.shape)
    corrected_flux = flux - corr
    if doPlot:
        pl.clf()
        x = np.arange(nobs)
        pl.plot(x, flux, '-', c = 'grey')
        pl.plot(x[l], flux[l], 'k-')
        pl.plot(x, corr, 'c-')
        pl.plot(x, corrected_flux, 'm-')
        pl.xlabel('Observation number')
        pl.xlabel('Flux')
    return corrected_flux, weights


def sel_nb(flux, cbv, nBmax=None, use=None):
    '''Correct light curve for systematics using upt to nB CBVs (automatically select best number).

    parameters
    ----------
    flux : `Iter`
        A 1-D array of light curves 
    cbv : `Iter`
        A 2-D array of co-trending basis vectors trends
    nBmax : `int`, optional, default=None
        The maximum number of CBVs to use (starting with the first)
    use : `bool`, optional, default=True
        True for data points to use in evaluating correction, False for data points to ignore (NaNs are also ignored)

    returns
    -------
    nBopt : `int`
        The automatically selected number of CBVs used (<= nBmax)
    corr_flux : `Iter`
        The corrected light curves (same shape as flux)
    weights : `Iter`
        The co-trending basis vector coefficients (same shape as nBopt).
    '''
    nobs = len(flux)
    if cbv.shape[1] == nobs:
        cbv_ = np.copy(cbv)
        #print('I')
    else:
        cbv_ = cbv.T
        #print('II')
    if nBmax is None:
        nBmax = cbv.shape[0]
        #print('III')
    else:
        cbv_ = cbv_[:nBmax, :]
        #print('IV')

    corr_flux = np.zeros(nobs)
    corr_flux_multi = np.zeros((nBmax,nobs))
    weights_multi = np.zeros((nBmax,nBmax))
    ran_multi = np.zeros(nBmax)
    sig_multi = np.zeros(nBmax)

    #print('V', nBmax)
    l = np.isfinite(flux)
    if use is not None:
        l *= use
        #print('VI')

    med_raw, ran_raw, sig_raw = medransig(flux[l])

    for i in range(nBmax):
        #print('VII')
        cbv_c = cbv_[:i+1,:]
        #print('VIIa')
        w_c = fit_basis(flux[l].reshape((1,l.sum())), cbv_c[:,l])
        #print('VIIb')
        w_ext = np.zeros(nBmax)
        #print('VIIc')
        w_ext[:i+1] = w_c
        #print('VIId')
        weights_multi[i,:] = w_ext
        #print('VIIe')
        corr = apply_basis(w_c, cbv_c).reshape(flux.shape)
        #print('VIIf')
        c = flux - corr
        #print('VIIg')
        med, ran, sig = medransig(c[l])
        #print('VIIh')
        corr_flux_multi[i,:] = c - med + med_raw
        #print('VIIi')
        ran_multi[i] = ran
        #print('VIIj')
        sig_multi[i] = sig
        #print('index',i)
    #print("loop ended")

    # Select the best number of basis functions       ###########DEBUG FROM HERE, MAY 1 2025- ADG
    # (smallest number that significantly reduces range)
    med_ran = np.median(ran_multi)
    sig_ran = 1.48 * np.median(abs(ran_multi - med_ran))
    jj = np.where(ran_multi < med_ran + 3 * sig_ran)[0][0]
    #print('VIII')
    # Does that introduce noise? If so try to reduce nB till it doesn't
    while (sig_multi[jj] > 1.1 * sig_raw) and (jj > 0): jj -= 1

    nb_opt = jj + 1
    flux_opt = corr_flux_multi[jj,:].flatten()
    weights_opt = weights_multi[jj,:][:jj+1].flatten()
    ran_opt = ran_multi[jj]
    sig_opt = sig_multi[jj]
    #print('IX')
    return (nb_opt, flux_opt, weights_opt), \
      (corr_flux_multi, weights_multi)


def interpolate_cbv(cbv_file, lc, type_cbv='Single'):
    '''Selects the type of CBV correction and applies them to the lightcurve
    
    parameters
    ----------
    cbv_file : `str`
        The name of the CBV-file
    lc : `astropy.table.Table`
        The lightcurve data in astropy-tabulated format
    type_cbv : `str`, optional, default='Single'
        The type of CBV-corrections to make. Choose from "Single",
        "Spike", "Multi1", "Multi2" or "Multi3"
        
   returns
   -------
   v_fin : `astropy.table.Table`
       The tabulated weights from the CBV fits to be applied to the lightcurve    
    '''

    with fits.open(cbv_file) as hdul:
        if type_cbv == 'Single':
            data = hdul[1].data
        if type_cbv == 'Spike':
            data = hdul[2].data
        if type_cbv == 'Multi1':
            data = hdul[3].data
        if type_cbv == 'Multi2':
            data = hdul[4].data
        if type_cbv == 'Multi3':
            data = hdul[5].data
 
        time = data["TIME"]
        vectors = [v for v in data.names if "VECTOR" in v]
        v_new = np.zeros((len(lc), len(vectors)))
        for i, v in enumerate(vectors):
            x = np.interp(lc["time"], time, data[f'{v}'])
            v_new[:,i] = np.interp(lc["time"], time, data[f'{v}'])
    v_fin = v_new.T
    return v_fin


def get_cbv_scc(scc, lc):
    '''Run the CBV fits for a given lightcurve
    
    parameters
    ----------
    scc : `Iter`
        A 3-element list containing the sector, camera and CCD of the TESS image
    lc : `astropy.table.Table`
        The tabulated lightcurve data.

    returns
    -------
    corrected_flux : `list`
        The flux values after ungoing CBV corrections
    weights : `list`
        The weights applied to each lightcurve data point.
    '''

    #print("GET CBVV")
    with open('./cbv/curl_cbv.scr', 'r') as curl_file:
        #print("GET CBV")
        lines = curl_file.readlines()
        cbv_comm = [l for l in lines if f'{scc[0]}-{scc[1]}-{scc[2]}' in l]
        #print('2',cbv_comm)
        #print('1')
        cbv_comm = cbv_comm[0]
        cbv_file = cbv_comm.split(' ')[6]
        #print('2',cbv_comm,cbv_file)
        if not os.path.exists(f'./cbv/{cbv_file}'):
            print(f'File does not exist: ./cbv/{cbv_file}')

        # if os.path.exists(f'./cbv/{cbv_file}') is False:
        #     subprocess.run(cbv_comm, shell=True)
        #     print('if')
        #     subprocess.run(f'mv {cbv_file} ./cbv/', shell=True)
    interpolated_cbv = interpolate_cbv(f'./cbv/{cbv_file}', lc)
    #print('interpolate')
    corrected_flux, weights = sel_nb(np.array(lc['flux'].data), interpolated_cbv)    
    print("Corrected flux length %d:" %len(corrected_flux))
    return corrected_flux, weights

########tessilator.py

from datetime import datetime
import sys
import os
import inspect
from glob import glob

# Third party imports
import numpy as np
import pyinputplus as pyip
from astropy.nddata.utils import Cutout2D
from astropy.table import Table, MaskedColumn
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
from astroquery.mast import Tesscut
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

# Local application imports
# from tessilator.aperture import aper_run
# from tessilator.lc_analysis import make_lc
# from tessilator.periodogram import run_ls
# from tessilator.detrend_cbv import get_cbv_scc
# from tessilator.contaminants import contamination, is_period_cont
# from tessilator.maketable import get_gaia_data
# from tessilator.makeplots import create_plot
# from tessilator.file_io import logger_tessilator, make_dir
# from tessilator.tess_stars2px import tess_stars2px_function_entry

# initialize the logger object
logger = logger_tessilator(__name__)


STARTUP_STRING = r"""
**********************************************************************
****|******_*********_*********_*********_*********_*********_********
****|*****/*\*******/*\*******/*\*******/*\*******/*\*******/*\*******
****|****/***\*****/***\*****/***\*****/***\*****/***\*****/***\******
****|***/*****\***/*****\***/*****\***/*****\***/*****\***/*****\*****
****|**/*******\_/*******\_/*******\_/*******\_/*******\_/*******\****
****|_____________________________________________________________****
**********************************************************************
**********************WELCOME TO THE TESSILATOR***********************
********The one-stop shop for measuring TESS rotation periods*********
**********************************************************************
**********************************************************************
If this package is useful for research leading to publication we
would appreciate the following acknowledgement:
'The data from the Transiting Exoplanet Survey Satellite (TESS) was
acquired using the tessilator software package (Binks et al. 2024).'
"""

print(STARTUP_STRING)


import requests
from bs4 import BeautifulSoup
import re

try:
    tess_web = requests.get('https://tess.mit.edu/observations/')
    soup = BeautifulSoup(tess_web.text, 'html.parser')
    tess_para = soup(text=re.compile("TESS is in Orbit"))
    sec_max = int(tess_para[0].split(',')[1].split(' ')[2].split('.')[0])
except:
    pass

_Template_table_format = [
    # column name, description, data type, format, fill_value
    ("original_id", r"Target identifier", str, None, "N/A"),
    ("source_id", r"Gaia DR3 source identifier", str, None, "N/A"),
    ("ra", r"Right ascension (epoch J2000)", float, ".12f", np.nan),
    ("dec", r"Declination (epoch J2000)", float, ".12f", np.nan),
    ("parallax", r"Gaia DR3 parallax", float, ".6f", np.nan),
    ("Gmag",r"Gaia DR3 $G$-band magnitude",float,".6f",np.nan,),
    ("BPmag", r"Gaia DR3 $G_{\rm BP}$-band magnitude", float, ".6f", np.nan),
    ("RPmag", r"Gaia DR3 $G_{\rm RP}$-band magnitude", float, ".6f", np.nan),
    ("Tmag_MED", r"Median TESS $T$-band magnitude", float, ".6f", 0),
    ("Tmag_MAD", r"MAD TESS $T$-band magnitude", float, ".6f", 0),
    ("Sector", r"TESS sector number", int, None, 0),
    ("Camera", r"TESS camera number", int, None, 0),
    ("CCD", r"TESS CCD number", int, None, 0),
    ("log_tot_bg", r"$\Sigma\eta$", float, ".6f", -999),
    ("log_max_bg", r"$\eta_{\rm max}$", float, ".6f", -999),
    ("num_tot_bg", r"Number of contaminating sources", int, None, 0),
    ("ap_rad", r"Aperture radius (pixels)", float, ".3f", -np.inf),
    ("false_flag", r"Test if a contaminant is the $P_{\rm rot}$ source", int, None, 4),
    ("reliable_flag", r"Test if the $P_{\rm rot}$ source is reliable", int, None, 4),
    ("CBV_flag", r"The CBV-correction category", int, None, 9),
    ("smooth_flag", r"Flag for detrending step 1", int, None, 9),
    ("norm_flag", r"Flag for detrending step 2$", int, None, 9),
    ("jump_flag", r"Test for jumps in the lightcurve", int, None, 9),
    ("AIC_line", r"AIC score: linear fit to the lightcurve", float, ".6f", np.nan),
    ("AIC_sine", r"AIC score: sine fit to the lightcurve", float, ".6f", np.nan),
    ("Ndata", r"Number of datapoints in the periodogram analysis", int, None, 0),
    ("FAP_001", r"1\% False Alarm Probability power", float, ".6f", np.nan),
    ("period_1", r"Primary $P_{\rm rot}$ (peak)", float, ".6f", np.nan),
    ("period_1_fit", r"Primary $P_{\rm rot}$ (Gaussian fit centroid)", float, ".6f",np.nan),
    ("period_1_err", r"Primary $P_{\rm rot}$ uncertainty", float, ".6f", np.nan),
    ("power_1", r"Power output of the primary $P_{\rm rot}$", float, ".6f", np.nan),
    ("period_2", r"Secondary $P_{\rm rot}$ (peak)", float, ".6f", np.nan),
    ("period_2_fit",r"Secondary $P_{\rm rot}$ (Gaussian fit centroid)",float,".6f",np.nan),
    ("period_2_err", r"Secondary $P_{\rm rot}$ uncertainty", float, ".6f", np.nan),
    ("power_2", r"Power output of the secondary $P_{\rm rot}$", float, ".6f", np.nan),
    ("period_3", r"Tertiary $P_{\rm rot}$ (peak)", float, ".6f", np.nan),
    ("period_3_fit",r"Tertiary $P_{\rm rot}$ (Gaussian fit centroid)",float,".6f",np.nan),
    ("period_3_err", r"Tertiary $P_{\rm rot}$ uncertainty", float, ".6f", np.nan),
    ("power_3", r"Power output of the tertiary $P_{\rm rot}$", float, ".6f", np.nan),
    ("period_4", r"Quarternary $P_{\rm rot}$ (peak)", float, ".6f", np.nan),
    ("period_4_fit", r"Quarternary $P_{\rm rot}$ (Gaussian fit centroid)", float, ".6f", np.nan),
    ("period_4_err", r"Quarternary $P_{\rm rot}$ uncertainty", float, ".6f", np.nan),
    ("power_4", r"Power output of the quarternary $P_{\rm rot}$", float, ".6f", np.nan),
    ("period_shuffle", r"$P_{\rm shuff}$", float, ".6f", np.nan),
    ("period_shuffle_err", r"Uncertainty in $P_{\rm shuff}$", float, ".6f", np.nan),
    ("shuffle_flag", r"Indicates if $P_{\rm shuff}$ was adopted", int, None, 9),
    ("amp", r"Amplitude of the PFL", float, ".6f", np.nan),
    ("scatter", r"Scatter of the PFL", float, ".6f", np.nan),
    ("chisq_phase",r"$\chi^{2}$ value of the sinusoidal fit to the PFL",float,".6f",np.nan),
    ("fdev", r"Number of extreme outliers in the PFL", float, ".6f", np.nan),
]

def create_table_template():
    '''Create a template astropy table to store tessilator results.

    returns
    -------
    res_table : `astropy.table.Table`
        A template table to store tessilator results.
    '''
    cols = []
    for name, description, dtype, format, fill_value in _Template_table_format:
        cols.append(
            MaskedColumn(
                name=name,
                description=description,
                dtype=dtype,
                format=format,
                fill_value=fill_value,
            )
        )
    return Table(cols, masked=True)


def setup_input_parameters():
    '''Retrieve the input parameters to run the tessilator program.

    The input parameters are:

    1) "file_ref" is a string expression used to reference the files produced.

    2) "t_filename" is the name of the input file (or target) required for
       analysis.

    If a program is called from the command line without all five input
    parameters, a set of prompts are initiated to receive input. If just one
    target is needed, then the user can simply supply either the target name,
    as long as it is preceeding by a hash (#) symbol. Otherwise, if the full
    set of command line parameters are supplied, the function will use these
    as the inputs, however, if they have the wrong format the program will
    return a warning message and exit.

    parameters
    ----------
    Either arguments supplied on the command line, or the function will prompt
    the user to provide input.
    
    returns
    -------
    flux_con : `bool`
        Run lightcurve analysis for contaminant sources.
    scc : `list`, size=3, only if sector data is used
        List containing the sector number, camera and CCD.
    lc_con : `bool`, only if cutout data is used
        Decides if a lightcurve analysis is to be performed for the 5 strongest
        contaminants. Here, the data required for further analysis are
        stored in a table.
    file_ref : `str`
        A common string to give all output files the same naming convention.
    t_filename : `str`
        The name of the input table containing the targets (or a single
        target).
     '''
    # first, set parameters in the case where inputs are not defined on the
    # command line
    if len(sys.argv) != 3:
        file_ref = pyip.inputStr("Enter the unique name for referencing the "
                   "output files : ")
        while True:
            t_filename = pyip.inputStr("Enter the file name of your input "
                         "table or object.\nIf this is a single target please "
                         "enter a hash (#) symbol before the identifier : ")
            if not os.path.exists(t_filename):
                logger.error(f'The file "{t_filename}" does not exist.')
            else:
                break
    # second, set parameters in the case where command line inputs are given
    else:
        file_ref = sys.argv[1]
        t_filename = sys.argv[2]

    if t_filename.startswith('#'):
        logger.info(t_filename)
        t_name = t_filename[1:]
        t_name_joined = t_name.replace(' ','_').replace(',', '_')+'.dat'
        if os.path.exists(t_name_joined):
            os.remove(t_name_joined)
        with open(t_name_joined, 'a') as single_target:
            if t_name.startswith("Gaia DR3 "):
                single_target.write(t_name[9:])
            else:
                single_target.write(t_name)
        t_filename = t_name_joined
    if not os.path.exists(t_filename):
        logger.critical(f'The file "{t_filename}" does not exist.')
        sys.exit()
    return file_ref, t_filename

def setup_filenames(file_ref, scc=None):
    '''Set up the file names to store data.

    parameters
    ----------
    file_ref : `str`
        A common string to give all output files the same naming convention.
    scc : `list` or `None`, size=3, optional, default = `None` 
        A list containing the Sector, Camera and CCD.

    returns
    -------
    period_file : `str`
        Name of file for recording parameters measured by the periodogram
        analysis.
    '''    
    if scc:
        name_str = 'sector'+f"{scc[0]:02d}"
    else:
        name_str = 'tesscut'
    period_file = '_'.join(['periods', file_ref, name_str])
    return period_file

def test_table_large_sectors(t_filename):
    '''Check if the input file needs modifying at all.

    If running the tessilator for a whole sector, read the input file and if
    the format is ready for analysis, make a couple of adjustments, then simply
    pass the file.
    
    For a straight pass, the columns must be ordered in two ways. Either:

    * exactly set out with the following columns:
    
      1) source_id: name of the Gaia DR3 source identifier
      
      2) ra: right ascension
      
      3) dec: declination
      
      4) parallax: parallax
      
      5) Gmag: Gaia DR3 apparent G-band magnitude
      
      6) BPmag: Gaia DR3 apparent BP-band magnitude
      
      7) RPmag: Gaia DR3 apparent RP-band magnitude
      
      8) Sector: The TESS sector
      
      9) Camera: The TESS camera
      
      10) CCD: The TESS CCD number
      
      11) Xpos: The TESS X-pixel
      
      12) Ypos: The TESS Y-pixel
        
    * The same, but with a preceding column entitled "name", which refers to a
      target identifier name. This can be any string.
    
    In any other case, None is returned and other functions are used to get the
    table into the correct format.

    parameters
    ----------
    t_filename : `str`
        The file name of the table which will be checked for formatting.

    returns
    -------
    t : `astropy.table.Table` or `None`
        either a ready-prepared table or nothing.
    '''

    t = Table.read(t_filename, format='csv')
    # The list of column names which must be spelled exactly.
    cnc = ['source_id', 'ra', 'dec', 'parallax', 'Gmag', 'BPmag', 'RPmag',
           'Sector', 'Camera', 'CCD', 'Xpos', 'Ypos']
    if cnc == t.colnames:
        t["name"] = t["source_id"]
        cnc.insert(0, 'name')
        t = t[cnc]
        # Ensure the dtype for "name" and "source_id" are strings.
        t["name"] = t["name"]
        t["source_id"] = t["source_id"]
    elif cnc == t.colnames[1:]:
        if t.colnames[0] == "name":
            # Ensure the dtype for "name" and "source_id" are strings.
            t["name"] = t["name"]
            t["source_id"] = t["source_id"]
    else:
        t = None
        # return nothing if neither of the above two conditions are met.
    return t

def read_data(t_filename, name_is_source_id=False, type_coord='icrs',
              gaia_sys=True):
    '''Read input data and convert to an astropy table ready for analysis.
    
    The input data must be in the form of a comma-separated variable and may
    take 3 forms:
    
    (a) a 1-column table of source identifiers.
    
    (b) a 2-column table of decimal sky coordinates (celestial or galactic).
    
    (c) a pre-prepared table of 7 columns consisting of source_id, ra, dec,
        parallax, Gmag, BPmag, RPmag (without column headers).
    
    Note that a single target can be quickly analysed directly from the
    command line by using option (a) with a #-sign preceding the target name,
    and then encompassed with double-quotation marks around the source
    identifier.
    
    E.G. >>> python run_tess_cutouts files "#AB Doradus"
    
    parameters
    ----------
    t_filename : `astropy.table.Table`
        Name of the file containing the input data.
    name_is_source_id : `bool`, optional, default=False
        When running option (c), the "name" column will automatically be set as
        the Gaia DR3 identifiers if this parameter is True. This avoids long
        sql queries for very large input tables.
    type_coord : `str`, optional, default='icrs'
        The coordinate system of the input data. Choose from 'icrs', 'galactic'
        or 'barycentricmeanecliptic', where the latter is the conventional
        coordinate system used by TESS.
    gaia_sys : `bool`, optional, default=True
        Choose to format the data based on Gaia DR3. Note that no contamination
        can be calculated if this is False.

    returns
    -------
    t_targets : `astropy.table.Table`
        a formatted astropy table ready for further analysis
    '''
    
    if isinstance(t_filename, str):
        t_input = ascii.read(t_filename, delimiter=',', format='no_header')
    elif isinstance(t_filename, Table):
        t_input = t_filename
    t_targets = get_gaia_data(t_input, name_is_source_id=name_is_source_id,
                              type_coord=type_coord, gaia_sys=gaia_sys)

    if len(t_targets) == 1:
        os.remove(t_filename)

    return t_targets


def collect_contamination_data(t_targets, ref_name, targ_name,
                               gaia_sys=True, ap_rad=1., n_cont=10, cont_rad=10.,
                               mag_lim=3., tot_attempts=3):
    '''Collect data on contamination sources around selected targets.

    This function takes a target table and, if requested, prints out details of
    the total flux contribution from neighbouring contaminating sources for
    each target.
    It also returns a table (in descending order) of neighbouring contaminants
    that contribute the most flux in the target aperture, if requested.
    
    Parameters
    ----------
    t_targets : `astropy.table.Table`
        A target table with columns:        

        * name: name of the target (`str`)

        * source_id: Gaia DR3 source identifier (`str`)

        * ra: right ascension

        * dec: declination

        * parallax: parallax

        * Gmag: Gaia DR3 apparent G-band magnitude

        * BPmag: Gaia DR3 apparent BP-band magnitude

        * RPmag: Gaia DR3 apparent RP-band magnitude

    ref_name : `str`
        The reference name for each subdirectory which will connect all output
        files.
    ap_rad : `float`, optional, default=1.
        The aperture radius from the aperture photometry (in pixels).
    gaia_sys : `bool`, optional, default=True
        Choose to format the data based on Gaia DR3. Note that no contamination
        can be calculated if this is False.

    Returns
    -------
    t_targets : `astropy.table.Table`
        Input target table with 3 columns added containing details of the
        contamination: "log_tot_bg", "log_max_bg", "num_tot_bg".
    t_cont : `astropy.table.Table`
        Table containing details of the contamination flux from nearby sources,
        in descending order of "log_tot_bg".
    cont_dir : `str`
        The directory that contains the contamination files.
    '''

    if gaia_sys:
        t_targets, t_cont = contamination(t_targets, ap_rad=ap_rad, n_cont=n_cont,
                                          cont_rad=cont_rad, mag_lim=mag_lim, tot_attempts=tot_attempts)
        t_contam = t_targets[['source_id', 'log_tot_bg', 'log_max_bg',\
                              'num_tot_bg']]
        cont_dir = make_dir("contaminants", ref_name)
        path_exist = os.path.exists(cont_dir)
        if not path_exist:
            os.mkdir(cont_dir)
        t_contam.write(f'{cont_dir}/{targ_name}.csv', overwrite=True)
        if t_cont is not None:
            t_cont.write(f'{cont_dir}/{targ_name}_individiual.csv',
                         overwrite=True)
    else:
        t_targets["log_tot_bg"] = -999.
        t_targets["log_max_bg"] = -999.
        t_targets["num_tot_bg"] = -999.
        if os.path.exists(f'{cont_dir}/{targ_name}.csv'):
            t_contam = Table.read(f'{cont_dir}/{targ_name}.csv')
            for i in range(len(t_contam)):
                g = (t_contam["source_id"] == \
                     t_targets["source_id"][i])
                if len(g) >= 1:
                    t_targets["log_tot_bg"][i] = t_contam["log_tot_bg"][g][0]
                    t_targets["log_max_bg"][i] = t_contam["log_max_bg"][g][0]
                    t_targets["num_tot_bg"][i] = t_contam["num_tot_bg"][g][0]
    return t_targets, t_cont, cont_dir

def make_target_row(t_targets, r, scc):
    """Construct table row with the input target data.

    If tessilator fails to run on a target, that's all you get.

    Parameters
    ----------
    t_targets : `astropy.table.Row`
        One row of input data for the tessilator, with the following columns:

        * name: name of the target (`str`)

        * source_id: Gaia DR3 source identifier (`str`)

        * ra: right ascension

        * dec: declination

        * parallax: parallax

        * Gmag: Gaia DR3 apparent G-band magnitude

        * BPmag: Gaia DR3 apparent BP-band magnitude

        * RPmag: Gaia DR3 apparent RP-band magnitude

        * log_tot_bg: log-10 value of the flux ratio between contaminants
          and target (optional)

        * log_max_bg: log-10 value of the flux ratio between the largest
          contaminant and target (optional)

        * num_tot_bg: number of contaminant sources (optional)

    r : `float`
        The pixel size of the aperture radius
    scc : tuple, size=3
        A list containing the Sector, Camera and CCD.

    returns
    -------
    dr : dict
        A dictionary for the target star containing input data
    """
    copycols = [
        "source_id",
        "ra",
        "dec",
        "parallax",
        "Gmag",
        "BPmag",
        "RPmag",
        "log_tot_bg",
        "log_max_bg",
        "num_tot_bg",
    ]
    dr = {col: t_targets[col] for col in copycols}
    dr["original_id"] = t_targets["name"]
    dr["Sector"] = scc[0]
    dr["Camera"] = scc[1]
    dr["CCD"] = scc[2]
    dr["ap_rad"] = r
    # All other lines we leave on default, so they will be maksed,
    # can can be filled if needed.
    return dr


def apply_noise_corr(targ_lc, sim_lc):
    '''Apply the correction to the lightcurve based on the noise simulation.
    
    parameters
    ----------
    targ_lc : `dict`
        A dictionary containing the lightcurve data for a target
        star (see tessilator.make_lc for the required inputs).
    sim_lc : `dict`
        A dictionary containing the simulated data.
        
    returns
    -------
    targ_lc : `dict`
        The same input lightcurve, corrected by the noise simulation.
    '''
    targ_lc['nflux_err'][np.where(targ_lc['nflux_err'] < 0)] = .01
    cln_cond = np.logical_and.reduce([
                   targ_lc["pass_clean_scatter"],
                   targ_lc["pass_clean_outlier"],
                   targ_lc["pass_full_outlier"]
                   ])
    targ_lc["nflux_noise_corr"] = 0
    targ_lc["sim_flux"] = 0

    nflux_noise_corr = np.array([])
    sim_flux = np.array([])
    for t, time in enumerate(targ_lc["time"]):
        sim_bit = np.interp(time, sim_lc["time"], sim_lc["nflux_dtr"])
        flux_bit = targ_lc["nflux_dtr"][t]/sim_bit
        sim_flux = np.append(sim_flux, sim_bit)
        nflux_noise_corr = np.append(nflux_noise_corr, flux_bit)
    targ_lc["nflux_noise_corr"][cln_cond] = nflux_noise_corr[cln_cond]
    targ_lc["sim_flux"][cln_cond] = sim_flux[cln_cond]
    return targ_lc


def fix_noise_lc_local(targ_lc, med_lc, scc, targ_name, ref_name,
                       make_plot=False):
    '''Apply a flux-corrected key to the target lightcurve dictionary, and make
    a plot of the corrections if required.
    
    This function uses the lightcurve data from neighbouring sources.
    
    parameters
    ----------
    targ_lc : `dict`
        a dictionary containing the lightcurve data for a target
        star (see tessilator.make_lc for the required inputs).
    med_lc : `dict`
        the "median" lightcurve data from the neigbourhing sources.
    scc : `list`, size=3
        List containing the sector number, camera and CCD.
    targ_name : `str`
        The name of the target
    ref_name : `str`
        The reference name for each subdirectory which will connect all output
        files.
    make_plot : `bool`, optional, default=False
        Choose to make plots of the noise corrections.

    returns
    -------
    targ_lc : `dict`
        The updated target lightcurve dictionary, with an extra key containing
        the noise-corrected flux.
    '''

    targ_lc = apply_noise_corr(targ_lc, med_lc)
    if make_plot:
        plot_name = f"{targ_name}_{scc[0]:04d}_{scc[1]}_"\
                    f"{scc[2]}_corr_local_lc.png"

        make_lc_corr_plot(plot_name, ref_name, targ_lc["time"],
                          targ_lc["nflux_dtr"], targ_lc["sim_flux"])
    return targ_lc

# NOTE - I'm cancelling this procedure until we get new lightcurves!
# 23.01.2024
def fix_noise_lc_sim(targ_lc, targ_name, t_targets, scc, ref_name,
                     mag_extr_lim=3., make_plot=False):
    '''Divide the target lightcurve by the noise lightcurve
    
    For a given sector, camera and CCD configuration, this function will search
    for the simulated noisy lightcurve and divide the target lightcurve by the
    noise lightcurve, which should mitigate some of the systematic features.
    
    The noisy lightcurves are described in the tessimulation.py module.
    
    parameters
    ----------
    targ_lc : `dict`
        A dictionary containing the lightcurve data for a target
        star (see tessilator.make_lc for the required inputs).
    targ_name : `str`
        The name of the target
    t_targets : `astropy.table.row.Row`
        Details of the target star.   
    scc : `list`, size=3
        A list containing the sector number, camera and CCD.    
    ref_name : `str`
        The reference name for each subdirectory which will connect all output
        files.
    mag_extr_lim : `float`, optional, default=3.
        The tolerated difference in magnitude between the target
        and the range of calculated simulated lightcurves if the
        magnitude is out of range.
    make_plot : `bool`, optional, default=False
        Choose to make plots of the noise corrections.

    returns
    -------
    targ_lc : `dict`
        The updated target lightcurve dictionary, with an extra key containing
        the noise-corrected flux.
    '''
    
    mag_files = sorted(glob(f"./tesssim/lc/{scc[0]:02d}_{scc[1]}_"\
                            f"{scc[2]}/mag*"))
    if not mag_files:
        logger.warning(f"No simulated lightcurve for {targ_name}, Sector "
                       f"{scc[0]:02d}, Camera {scc[1]}, CCD {scc[2]}")
        return targ_lc

    mag_target = t_targets['Gmag'][0]
    mag0 = np.array([float(mag_file.split("_")[-2]) for mag_file in mag_files])
    mag1 = np.array([float(mag_file.split("_")[-1]) for mag_file in mag_files])

    g = np.where((mag0 <= mag_target) & (mag1 > mag_target))[0]
    g = []
    if g.size > 0:
        sim_tab = f'{mag_files[g[0]]}/flux_fin.csv'
    elif (mag_target < mag0[0]) & \
         (np.abs(mag0[0] - mag_target) < mag_extr_lim):
        logger.info(f"Target {targ_name} is brighter than the magnitude range of "
                    f"simulated files, but within the extrapolation threshold")
        sim_tab = f'{mag_files[0]}/flux_fin.csv'
    elif (mag_target > mag1[-1]) & \
         (np.abs(mag1[-1] - mag_target) < mag_extr_lim):
        logger.info(f"Target {targ_name} is fainter than the magnitude range "
                    f"of simulated files, but within the extrapolation "
                    f"threshold")
        sim_tab = f'{mag_files[-1]}/flux_fin.csv'
    else:
        logger.warning(f"Target {targ_name} is out of the magnitude range of "
                       f"simulated files, and out of the extrapolation "
                       f"threshold")
        return targ_lc

    sim_lc = Table.read(sim_tab)
    targ_lc = apply_noise_corr(targ_lc, sim_lc)

    if make_plot:
        plot_name = f"{targ_name}_{scc[0]:04d}_{scc[1]}_{scc[2]}_"\
                    f"corr_sim_lc.png"
        make_lc_corr_plot(plot_name, ref_name, targ_lc["time"],
                          targ_lc["nflux_dtr"], targ_lc["sim_flux"])
    return targ_lc





def make_lc_corr_plot(plot_name, ref_name, targ_time, targ_flux, sim_flux,
                      im_dir='plots'):
    '''Make a plot of the noise corrections to the lightcurve.
    
    parameters
    ----------
    plot_name : `str`
        The file name which will be used to save the plot.
    ref_name : `str`
        The reference name for each subdirectory which will connect all output
        files.
    targ_time : `Iterable`
        The time coordinate of the data.
    targ_flux : `Iterable`
        The flux coordinate of the target data.
    sim_flux : `Iterable`
        The flux coordinate of the simulated noisy lightcurve.
    im_dir : `str`, optional, default='plots'
        The directory to save the plots.

    returns
    -------
    None. Plots are saved to file.
    '''    
    fig, ax = plt.subplots(figsize=(15,7))
    mpl.rcParams.update({'font.size': 20})
    ax.set_xlabel('time [days]')
    ax.set_ylabel('normalised flux')
    ax.plot(targ_time, targ_flux, '.', c='r', label='target flux')
    ax.plot(targ_time, sim_flux, '.', c='g', label='systematic flux')
    ax.legend()
    im_dir_tot = f'./{im_dir}/{ref_name}'
    path_exist = os.path.exists(im_dir_tot)
    if not path_exist:
        os.mkdir(im_dir_tot)
    plt.savefig(f'{im_dir_tot}/{plot_name}', bbox_inches='tight')
    plt.close('all')



def get_name_target(t_target):
    '''Quick function to capture a string containing the name of the target.
    
    parameters
    ----------
    t_target : `astropy.table.row.Row`
        The target data, which contains the target name.
        
    returns
    -------
    name_target : `str`
        The formatted target name to be used for reference.
    '''
    name_target = t_target.replace(" ", "_")
    name_spl = name_target.split("_")
    if name_spl[0] == 'Gaia':
        name_target = name_spl[-1]
    return name_target

def get_median_lc(tables, files_loc, scc, n_bin=10):
    '''A function that makes the final noisy lightcurve.
    
    For a given number of lightcurves, this function calculates the median flux
    at each time step if there are more than 2 measurements (or the mean if 2
    or less).
    
    parameters
    ----------
    tables : `list`
        A list of lightcurves in a given directory.
    files_loc : `str`
        The name of the directory containing the lightcurves.
    scc : `list`, size=3
        A list containing the sector number, camera and CCD.    
    n_bin : `int`, optional, default=10
        The maximum number of lightcurves to be used in the analysis.
        
    returns
    -------
    t_fin : `astropy.table.Table`
        A table containing the data for the final noisy lightcurve.
    '''
    f_name = files_loc.split("/")[3].split("_")[1]
    directory = ('/').join(files_loc.split("/")[:-1])+'/'
    num, time, flux, eflux = [], [], [], []
    if len(tables) > n_bin:
        chosen_indices = np.random.choice(len(tables), n_bin)
        tables_chosen = [tables[n] for n in chosen_indices]
    else:
        tables_chosen = tables
    for t, tab in enumerate(tables_chosen):
        tab['nflux_err'][np.where(tab['nflux_err'] < 0)] = .01
        cln_cond = np.logical_and.reduce([
                       tab["pass_clean_scatter"],
                       tab["pass_clean_outlier"],
                       tab["pass_full_outlier"]
                       ])

        tab = tab[cln_cond]
        for t_line in tab:
            num.append(t+1)
            time.append(t_line['time'])
            flux.append(t_line['nflux_dtr'])
            eflux.append(t_line['nflux_err'])
    t_uniq = np.unique(np.array(time))
    t_fin = Table(names=('time', 'nflux_dtr', 'nflux_err', 'n_lc'),
                  dtype=(float, float, float, int))
    for t in t_uniq:
        g = np.where(time == t)[0]
        flux_med = np.median(np.array(flux)[g])
        eflux_med = np.median(np.array(eflux)[g])
        flux_mean = np.mean(np.array(flux)[g])
        eflux_mean = np.mean(np.array(eflux)[g])
        num_lc = len(g)
        if num_lc > 2:
            t_fin.add_row([t, flux_med, eflux_med, num_lc])
        else:
            t_fin.add_row([t, flux_mean, eflux_mean, num_lc])
    t_fin.write(f'{directory}flux_med_{f_name}_{scc[0]:04d}_'
                f'{scc[1]}_{scc[2]}.csv', overwrite=True)
    return t_fin




def assess_lc(ls_results):
    '''Decide whether to use the periodogram results from the original, or from
    the CBV-corrected lightcurve.
    
    The function provides 5 different tests to find which analysis provides
    better results. For each test, the winning lightcurve scores a point. The
    one with the most points at the end of the tests is chosen as the
    periodogram results output.

    parameters
    ----------
    ls_results : `list`
        The list of dictionaries containing the periodogram scores for both
        lightcurves.
        
    returns
    -------
        lc_choice : `int`
            The chosen periodogram, where 0=original and 1=CBV-corrected.
    '''
    ori_sc, cbv_sc = 0, 0
    #print('I')
    ori_ls, cbv_ls = ls_results[0], ls_results[1]
    #print(list(ori_ls.keys()))

    #print(list(cbv_ls.keys()))

    #print('II')
    lc_choice = 0
    #print('III')

#1) Check the best fit sine vs best fit line scores...
    if (ori_ls["AIC_sine"]-ori_ls["AIC_line"]) < \
       (cbv_ls["AIC_sine"]-cbv_ls["AIC_line"]):
        ori_sc += 1
        #print('IVa')
    else:
        #print('IVb')
        cbv_sc += 1
#2) Check how jumpy the lightcurves are...
    if not ori_ls["jump_flag"]:
        #print('Va')
        ori_sc += 1
    if not cbv_ls["jump_flag"]:
        #print('Vb')
        cbv_sc += 1
#3) Check the max_power/FAP_001
    if (ori_ls["power_1"]/ori_ls["FAPs"][2]) > \
       (cbv_ls["power_1"]/cbv_ls["FAPs"][2]):
        ori_sc += 1
        #print('VIa')
    else:
        cbv_sc += 1
        #print('VIb')
#4) Check the height of the amplitude
    # print('ori_ls :', ori_ls["pops_vals"][1])
    # print('ori_ls_2 :', ori_ls["phase_scatter"])
    # print('cbv_ls :', cbv_ls["pops_vals"][1])
    # print('cbv_ls_2 :', cbv_ls["phase_scatter"])
    if ori_ls["pops_vals"][1]/ori_ls["scatter"] > \
       cbv_ls["pops_vals"][1]/cbv_ls["scatter"]:
        ori_sc += 1
        #print('VIIa')
    else:
        cbv_sc += 1
        #print('VIIb')
#5) Check number of datapoints
    if ori_ls["Ndata"] > cbv_ls["Ndata"]:
        ori_sc += 1
        #print('VIIIa')
    else:
        cbv_sc += 1
        #print('VIIIb')


#6) Check number of outliers in the phase-folded curve
#    if ori_ls["frac_phase_outliers"] < cbv_ls["frac_phase_outliers"]:
#        ori_sc += 1
#    else:
#        cbv_sc += 1
#    print('test fdev: ', ori_sc, cbv_sc)
#7) Check the max_power/2nd_max_power...
#    if (ori_ls["power_1"]/ori_ls["power_2"]) > \
#        (cbv_ls["power_1"]/cbv_ls["power_2"]):
#        ori_sc += 1
#    else:
#        cbv_sc += 1

        
    #test_fdev = cbv_ls["frac_phase_outliers"] < \
                #ori_ls["frac_phase_outliers"]
    #print('IX')
    if (ori_sc < cbv_sc):# & (test_fdev):
        #print('X')
        lc_choice = 1
    return lc_choice

def convert_ffi_date_to_tess(fd):
    tess_t0 = 2457000.
    fd = str(fd)
    y,d,h,m,s=fd[0:4],fd[4:7],fd[7:9],fd[9:11],fd[11:]
    time_str = f"{y}:{d}:{h}:{m}:{s}"
    time_now = Time(time_str, format='yday', scale='utc').jd
    return time_now-tess_t0
                
                
                
                
def run_clean_fail_modes(lc, scc):
    file_root = './fail_modes/'
    fail_fire_file = f'{file_root}tess_fireflies_and_fireworks.txt'
    fail_jitters_file = f'{file_root}tess_jitters.txt'

    time_cut = []
    lc_mask = np.ones(len(lc), dtype=bool)

    if os.path.exists(fail_fire_file):
        fail_fire = ascii.read(fail_fire_file)
        cut_lines = np.where((fail_fire["Sector"] == scc[0]) &
                            (fail_fire["Camera"] == scc[1]))[0]
        if len(cut_lines) >= 1:
            for c in cut_lines:
                fs = convert_ffi_date_to_tess(fail_fire["FFI_Start"][c])
                fe = convert_ffi_date_to_tess(fail_fire["FFI_End"][c])
                time_cut.append((fs,fe))
    if os.path.exists(fail_jitters_file):
        fail_jitters = ascii.read(fail_jitters_file)
        cut_lines = np.where(fail_jitters["Sector"] == scc[0])[0]
        for c in cut_lines:
            time_cut.append((fail_jitters["j_start"][c],
                             fail_jitters["j_fin"][c]))
    for tc in time_cut:
         lc_mask = (lc_mask) & ((lc["time"] <= tc[0]) | (lc["time"] >= tc[1]))

    return lc[lc_mask]

def full_run_lc(file_in, t_target, scc, res_table, gaia_sys=True,
                xy_pos=(10.,10.), ap_rad=1., sky_ann=(6.,8.), fix_rad=False,
                n_cont=10, cont_rad=10., mag_lim=3., tot_attempts=3,
                ref_name='targets', cutout_size=20, save_phot=False,
                cbv_flag=False, store_lc=False, lc_dir='lc', pg_dir='pg',
                plot_ext='plots', clean_fail_modes=False, keep_data=False,
                calc_cont=False, lc_cont=False, fix_noise=False,
                shuf_per=False, make_plot=False, make_shuf_plot=False,
                shuf_dir='plot_shuf'):
    '''Aperture photometry, lightcurve cleaning and periodogram analysis.

    This function calls a set of functions in the lc_analysis.py module to
    perform aperture photometry, clean the lightcurves from spurious data and
    runs the Lomb-Scargle periodogram to measure rotation periods.

    parameters
    ----------
    file_in : `str`
        Name of the input TESS fits file.
    t_target : `astropy.table.Table`
        Details of the target star.
    scc : `list`, size=3
        List containing the sector number, camera and CCD.
    res_table : `astropy.table.Table`
        The table to store the final tessilator results.
    gaia_sys : `bool`, optional, default=True
        Choose to format the data based on Gaia DR3. Note that no contamination
        can be calculated if this is False.
    xy_pos : `tuple`, size=2x2, optional, default=(10.,10.)
        The centroid of the target in pixels.
    ap_rad : `float`, optional, default=1.
        The size of the aperture radius in pixels.
    sky_ann : `tuple`, optional, default=(6.,8.)
        A 2-element tuple defining the inner and outer annulus to calculate
        the background flux.
    fix_rad : `bool`, optional, default=False
        If True, then set the aperture radius equal to ap_rad, otherwise run the
        calc_rad algorithm.
    n_cont : `int`, optional, default=10
        The maximum number of neighbouring contaminants to store to table.
    cont_rad : `float`, optional, default=10.
        The maximum pixel radius to search for contaminants
    mag_lim : `float`, optional, default=3.
        The faintest magnitude to search for contaminants.
    tot_attempts : `int`, optional, default=3
        The number of sql queries in case of request or server errors.
    ref_name : `str`, optional, default='targets'
        The reference name for each subdirectory which will connect all output
        files.
    cutout_size : `int`, optional, default=20
        The pixel length of the downloaded cutout.
    save_phot : `bool`, optional, default=False
        Decide whether to save the full results from the aperture photometry.
    cbv_flag : `bool`, optional, default=False
        Decide whether to run the lightcurve analysis with a CBV-correction
        applied.
    store_lc : `bool`, optional, default=False
        Choose to save the cleaned lightcurve to file.
    lc_dir : `str`, optional, default='lc'
        The directory used to store the lightcurve files if lc_dir==True.
    pg_dir : `str`, optional, default='pg'
        The directory used to store the periodogram data.
    plot_ext : `str`, optional, default='plots'
        The directory used to store the plots if make_plot==True.
    clean_fail_modes : `bool`, optional, default=False
        Choose to remove parts of the lightcurve that are reported as
        problematic in the TESS Data Release Notes
    keep_data : `bool`
        Choose to save the input data to file.
    flux_con : `bool`, optional, default=False
        Decides if the flux contribution from contaminants is to be calculated.
    lc_con : `bool`, optional, default=0
        Decides if a lightcurve analysis is to be performed for the n strongest
        contaminants.
    fix_noise : `bool`, optional, default=False
        Choose to apply corrections accounting for systematic noise.
    shuf_per : `bool`, optional, default=False
        Choose to run the shuffled period analysis (True=yes, False=no)
    make_shuf_plot : `bool`, optional, default=False
        Choose to make a plot for the shuffled period analysis
    shuf_dir : `str`, optional, default='plot_shuf'
        The name of the directory to save the plots of the shuffled period
        analysis. 

    returns
    -------
    * A data entry for the final period file.
    * A plot of the lightcurve (if requested).
    '''
    nc = 'nc'
    try:
        tpf, rad_calc = aper_run(file_in, t_target, xy_pos=xy_pos, ap_rad=ap_rad,
                                 sky_ann=sky_ann, fix_rad=fix_rad)
    except Exception as e:
        logger.error(f"aperture photometry: of {file_in} failed to run")
    if len(tpf) < 10:
        logger.error(f"aperture photometry: failed to produce enough data "
                     f"points for {t_target['source_id']}")
        for t in t_target:
            res_table.add_row(make_target_row(t, r=rad_calc, scc=scc))
        return None
    #print("CBV Flag  :",  cbv_flag)
    if cbv_flag:
        
        corrected_flux, weights = get_cbv_scc(scc, tpf)
        #print("YESSSSS")
        #print("tpf length %d" %len(tpf["reg_oflux"]))
        tpf["cbv_oflux"] = corrected_flux[1][:]
        
    else:
        tpf["cbv_oflux"] = tpf["reg_oflux"]
    keyorder = ['run_no','gaia_dr3_id','aperture_rad','time','xcenter','ycenter',
                'flux','flux_err','bkg','total_bkg','mag','mag_err',
                'reg_oflux','cbv_oflux']
    tab_format = [
        "%i",
        "%s",
        ".2f",
        ".6f",
        ".1f",
        ".1f",
        ".6f",
        ".6f",
        ".6f",
        ".6f",
        ".6f",
        ".4e",
        ".6f",
        r".6f",
    ]

    #print('A')
    
    tpf = tpf[keyorder]
    for n, f in zip(keyorder, tab_format):
        tpf[n].info.format = f

    phot_targets = tpf.group_by('gaia_dr3_id')
    #print("B")
    for key, group in zip(phot_targets.groups.keys, phot_targets.groups):
        g_c = group[group["flux"] > 0.0]
        #print('B1')
        if clean_fail_modes:
            #print('B2')
            g_c = run_clean_fail_modes(g_c, scc)
            #print('B3')
#        if scc[0] == 1:
#           sector01_rm = [1347,1349]
#           g_c = g_c[(g_c["time"] < sector01_rm[0]) |
#                     (g_c["time"] > sector01_rm[1])]
        if isinstance(t_target, Table):
            #print('B4')
            t_targets = t_target[t_target["source_id"] == key[0]]
            #print('B5')
        else:
            t_targets = Table(t_target)
            #print('B6')
        name_target = get_name_target(t_targets["name"][0])
        #print('B7')
        name_full = f'{name_target}_{scc[0]:04d}_{scc[1]}_{scc[2]}'
        #print('B8')
        if calc_cont:
            #print('B9')
            t_targets, t_cont, cont_dir = \
            collect_contamination_data(t_targets, ref_name, name_full,
                                       gaia_sys=gaia_sys, ap_rad=rad_calc, n_cont=n_cont, cont_rad=cont_rad, mag_lim=mag_lim, tot_attempts=tot_attempts
                                       )
            #print('B10')
        else:
            t_targets["log_tot_bg"] = -999.
            #print('B11')
            t_targets["log_max_bg"] = -999.
            #print('B12')
            t_targets["num_tot_bg"] = -999.
            #print('B13')
        if save_phot:
            #print('B14')
            tpf.write(f'{lc_dir}/ap_{name_full}.csv', overwrite=True)
        if len(g_c) >= 50:
            #print('B15')
            lcs, norm_flags, smooth_flags = make_lc(g_c,
                                                    name_lc='lc_'+name_full,
                                                    store_lc=store_lc,
                                                    lc_dir=lc_dir,
                                                    cbv_flag=cbv_flag)
        else:
            #print('B16')
            logger.error(f"No photometry was recorded for this group.")
            #print('B17')
            res_table.add_row(make_target_row(t_targets, r=rad_calc, scc=scc))
            #print('B18')
            continue
        if len(lcs) == 0:
            logger.error(f"no datapoints to make lightcurve analysis for "
                         f"{t_targets['source_id']}")
            res_table.add_row(make_target_row(t_targets, r=rad_calc, scc=scc))
            continue
        if fix_noise and not lc_con:
            logger.info('fixing the noise!')
            for l in range(len(lcs)):
                lcs[l] = fix_noise_lc_sim(lcs[l], name_target, t_targets, scc,
                                          ref_name, make_plot=make_plot)
            nc = 'corr_sim'
        ls_results = []
        #print('B19')
        for lc in lcs:
            #print('B20')
            lc_type = lc.colnames[2][:3]
            #print('B21')
            ls = run_ls(lc, lc_type=lc_type, ref_name=ref_name,
                        name_pg='pg_'+name_full,
                        check_jump=True, pg_dir=pg_dir, shuf_per=shuf_per,
                        n_shuf_runs=5000, make_shuf_plot=make_shuf_plot,
                        shuf_dir=shuf_dir,
                        name_shuf_plot=f'{name_full}_shuf.png')
            #print('B22')
            ls_results.append(ls)
            #print('B23')
        if len(lcs) == 1:
            #print('B24')
            choose_lc = 0
            best_lc = 0
        else:
            #print('B25')
            choose_lc = assess_lc(ls_results)
            #print('B25-a')
            best_lc = 1 + choose_lc
            #print('B25 b')
        #print('B26')
        d_target = ls_results[choose_lc]
        #print('B27')
        lc = lcs[choose_lc]
        #print('B28')
        d_target["CBV_flag"] = best_lc
        #print('B29')
        d_target["norm_flag"] = int(norm_flags[choose_lc])
        #print('B29')
        d_target["smooth_flag"] = int(smooth_flags[choose_lc])
        #print('B30')
        if d_target['period_1'] == -999:
            logger.error(f"the periodogram did not return any results for "
                         f"{t_targets['source_id']}")
            res_table.add_row(make_target_row(t_targets, r=rad_calc, scc=scc))
            continue

        false_flag, reliable_flag = 4, 4        
        #print('B31')
        if lc_cont:
            print('performing periodogram analysis of potential contaminants')
            lc_cont_files = []
            if not flux_cont:
                logger.warning("Contaminants not identified! "
                               "Please toggle lc_con=1")
                logger.warning("Continuing program using only the target.")
            else:
                logger.info('calculating contaminant lightcurves')
                if t_cont is not None:
                    t_cont = t_cont[t_cont["source_id_target"] ==
                                          t_targets["source_id"]]
                    xy_con = find_xy_cont(file_in, t_cont, cutout_size)
                    false_flag, reliable_flag = '', ''
                    for z in range(len(xy_con)):
                        name_lc, false_lab, reliable_lab, lc_contam, d_lc = \
                        run_test_for_contaminant(xy_con[z], file_in, t_cont[z],
                                                 d_target, scc,
                                                 lc_cont_dir=cont_dir)
                        false_flag += str(false_lab)
                        reliable_flag += str(reliable_lab)
                        if d_lc is not None:
                            if d_lc["AIC_sine"] > d_lc["AIC_line"]+1:
                                path_exist = os.path.exists(cont_dir)
                                if not path_exist:
                                    os.mkdir(cont_dir)
                                lc_contam.write(f'{cont_dir}/{name_lc}',
                                              overwrite=True)
                                lc_cont_files.append(lc_contam)
                        false_flag = int(false_flag)
                        reliable_flag = int(reliable_flag)
                else:
                    #print('B32')
                    false_flag, reliable_flag = 3, 3
                    xy_con = None
                if fix_noise:
                    print('fixing the noise')
                    logger.info('fixing the noise!')
                    if len(lc_cont_files) > 0:
                        nc = 'corr_local'
                        t_median_lc = \
                        get_median_lc(lc_cont_files, f'{cont_dir}/{name_lc}',
                                      scc)
                        clean_norm_lc = \
                        fix_noise_lc_local(lc, t_median_lc, scc, name_target,
                                           ref_name, make_plot=make_plot)
                        d_target = run_ls(clean_norm_lc, check_jump=True)
                        d_target["best_lc"] = best_lc
                    else:
                        nc = 'corr_sim'
                        lc = fix_noise_lc_sim(lc, name_target, t_targets, scc,
                                              ref_name, make_plot=make_plot)
                        d_target = run_ls(lc, check_jump=True)
                        d_target["best_lc"] = best_lc
                else:
                    #print('B33')
                    nc = 'nc'
        else:
            #print('B34')
            xy_con = None

        #print('C')
            
        d_target["false_flag"] = false_flag
        d_target["reliable_flag"] = reliable_flag


        #print('D')

        if make_plot:
            plot_dir = make_dir(plot_ext, ref_name)
            im_plot, xy_ctr = make_2d_cutout(file_in, group, 
                                             im_size=(cutout_size+1,
                                                      cutout_size+1))

            create_plot(im_plot, lc, d_target, scc, t_targets, name_target,
                      plot_dir, xy_contam=xy_con, p_min_thresh=0.1,
                      p_max_thresh=50., ap_rad=rad_calc, sky_ann=sky_ann, nc=nc)

        target_row = make_target_row(t_targets, r=rad_calc, scc=scc) | d_target
        #print('E')
        common_cols = {k: v for k, v in target_row.items() if k in res_table.colnames}
        res_table.add_row(common_cols)
        temp_dir = make_dir("temp_results", ref_name)
        res_table.write(f'{temp_dir}/{ref_name}_periods.csv', overwrite=True)
        #print('F')
        if not keep_data:
            if len(file_in) == 1:
                os.remove(file_in)
    print('completed!')

def print_time_taken(start, finish):
    '''Calculate the time taken for a process.

    This function takes a start and finish point a calculates the time taken in
    hours, minutes and seconds.

    parameters
    ----------
    start : `datetime.datetime`
        The start point of the process.
    
    finish : `datetime.datetime`
        The end point of the process.

    returns
    -------
    time_taken : `str`
        The time taken for the process to complete.
    '''
    time_in_secs = (finish - start).seconds
    mins, secs = divmod(time_in_secs, 60)
    hrs, mins = divmod(mins, 60)
    time_taken = f"{hrs} hours, {mins} minutes, {secs} seconds"
    return time_taken


def find_xy_cont(f_file, t_cont, cutout_size):
    '''Identify the pixel x-y positions for contaminant sources.

    If the user requests a periodogram analysis of neighbouring potential
    contaminants (lc_con=1), this function returns their x-y positions, which
    are used as the centroids for aperture photometry.

    parameters
    ----------
    f_file : `str`
        The name of the fits file.
    t_cont : `astropy.table.Table`
        The table containing Gaia data of the contaminants.
    cutout_size : `int`
        The length size of the TESS cutout image.

    returns
    -------
    cont_positions : `numpy.array`
        A 2-column array of the X-Y positions of the contaminants.
    '''
    xy_ctr = (cutout_size/2., cutout_size/2.)
    with fits.open(f_file) as hdul:
        head = hdul[0].header
        ra_ctr, dec_ctr = head["RA_OBJ"], head["DEC_OBJ"]
        ra_con, dec_con = t_cont["RA"], t_cont["DEC"]
        x_abs_con, y_abs_con = [], []
        _, _, _, _, _, _, col_ctr, row_ctr, _ = \
            tess_stars2px_function_entry(1, ra_ctr, dec_ctr,
                                         trySector=head["SECTOR"])
        for i in range(len(ra_con)):
            _, _, _, _, _, _, col_con, row_con, _ = \
                tess_stars2px_function_entry(1, ra_con[i], dec_con[i],
                                             trySector=head["SECTOR"])
            x_abs_con.append(col_con[0])
            y_abs_con.append(row_con[0])
        x_con = np.array(x_abs_con - col_ctr[0]).flatten() + xy_ctr[0]
        y_con = np.array(y_abs_con - row_ctr[0]).flatten() + xy_ctr[1]
        cont_positions = np.array([x_con, y_con]).T
        return cont_positions
        

def run_test_for_contaminant(
    xy_arr,
    file_in,
    t_cont,
    d_target,
    scc,
    aper_rad=1.0,
    sky_ann=(6.0, 8.0),
    store_lc=True,
    lc_cont_dir="lc_cont",
):
    """Run the periodogram analyses for neighbouring contaminants if required.

    parameters
    ----------
    xy_arr : `list`, size=2
        The X and Y positions of the contaminant (the output form the
        "find_XY_cont" function).
    file_in : `str`
        The name of the fits file containing the contaminant.
    t_cont : `astropy.table.Table`
        A single row from the contamination table which has
        details of the flux contribution.
    d_target : `dict`
        The dictionary returned from the periodogram analysis of
        the target star.
    scc : `list`, size=3
        List containing the sector number, camera and CCD.
    aper_rad : `float`, optional, default=1.
        The size of the aperture radius in pixels.
    sky_ann : `tuple`, optional, default=(6.,8.)
        A 2-element tuple defining the inner and outer annulus to calculate
        the background flux.
    store_lc : `bool`, optional, default=False
        Choose to save the cleaned lightcurve to file
    lc_cont_dir : `str`, optional, default='lc'
        The directory used to store the lightcurve files for contaminants if
        store_lc==True.

    returns
    -------
    name_lc : `str`
        The name of the file that the contaminant lightcurve will be saved to.
    labels_cont : `str` (a, b, c or d)
        A single character which assess if the calculated period for the target
        could actually come from the contaminant.

        a. At least 1 contaminant has a similar period to the target.

        b. No contaminants with similar periods

        c. The aperture photometry extraction failed for the contaminant

        d. Something went wrong with the routine.

    clean_norm_lc_cont : `astropy.table.Table`
    d_cont : `dict`
        The dictionary returned from the periodogram analysis of the
        contaminant star.
    """

    clean_norm_lc_cont, name_lc, d_cont = None, None, None
    try:
        xy_con = tuple((xy_arr[0], xy_arr[1]))
        phot_cont, _ = aper_run(file_in, t_cont, xy_pos=xy_con, aper_rad=aper_rad, sky_ann=sky_ann, fix_rad=False)
        if phot_cont is not None:
            name_lc = f'lc_{t_cont["source_id_target"]}_{t_cont["source_id"]}'\
                      f'_{scc[0]:04d}_{scc[1]}_{scc[2]}.csv'
            clean_norm_lc_cont, _, _ = make_lc(
                phot_cont, store_lc=store_lc, lc_dir=lc_cont_dir
            )[0]
            if len(clean_norm_lc_cont) != 0:
                d_cont = run_ls(clean_norm_lc_cont, name_pg=None)
                false_flag, reliable_flag = is_period_cont(d_target, d_cont, t_cont)
        else:
            false_flag, reliable_flag = 2, 2
    except:
        logger.error(f"something went wrong with measuring the period for"
                     f"{name_lc}")
        false_flag, reliable_flag = 2, 2
    logger.info(f"label for this contaminant: {name_lc}")
    return name_lc, false_flag, reliable_flag, clean_norm_lc_cont, d_cont


def get_tess_pixel_xy(t_targets):
    '''Get the pixel x-y positions for all targets in a Sector/Camera/CCD mode.

    For a given pair of celestial sky coordinates, this function returns table
    rows containing the sector, camera, CCD, and x/y position of the full-frame
    image fits file, so that all stars located in a given (large) fits file can
    be processed simultaneously. After the table is returned, the input table
    is joined to the input table on the source_id, to ensure this function only
    needs to be called once.

    This function is only used if the tessilator program runs over the
    calibrated full-frame images - i.e., when the "all_cc" function is called.
    
    parameters
    ----------
    t_targets : `astropy.table.Table`
        The input table created by the function get_gaia_data.py.

    returns
    -------
    xy_table : `astropy.table.Table`
        Output table containing the (*x*, *y*) pixel positions for each target.
    '''
    outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
           outCPix, outRPix, scinfo = tess_stars2px_function_entry(
           t_targets['source_id'], t_targets['ra'], t_targets['dec'])
    xy_table = Table([outID, outSec, outCam, outCcd, outCPix, outRPix],
            names=('source_id', 'Sector', 'Camera', 'CCD', 'Xpos', 'Ypos'))
    return xy_table



def make_2d_cutout(file_in, phot_table, im_size=(20,20)):
    '''Makes a 2D cutout object of a target using the median time-stacked
    image.

    parameters
    ----------
    file_in : `astropy.table.Table`
        The astropy table containing the output from the aperture photometry
        for a given target.
    phot_table : `list`
        The list of fits files used to make the aperture photometry.
    im_size : `tuple`, optional, default=(20,20)
        The required size of the 2D-cutout object.

    returns
    -------
    cutout : `astropy.nddata.Cutout2D`
        A 2D-cutout object.
    ctr_pt : `tuple`
        A tuple containing the X, Y position of the median time-stacked image.
    '''

    if isinstance(file_in, np.ndarray):
        image_index = math.floor((len(phot_table))/2)
        image_fits  = file_in[image_index]
        table_slice = phot_table[image_index]
        with fits.open(image_fits) as hdul:
            data = hdul[1].data
            head = hdul[1].header
            error = hdul[2].data
        ctr_pt = (table_slice["xcenter"], table_slice["ycenter"])
        cutout = Cutout2D(data, ctr_pt, im_size)
    elif isinstance(file_in, str):
        with fits.open(file_in) as hdul:
            data = hdul[1].data
        data_slice = data["FLUX"][:][:][int(data.shape[0]/2)]
        ctr_pt = ((im_size[0]-1)/2., (im_size[1]-1)/2.)
        cutout = Cutout2D(data_slice, ctr_pt, im_size)
    else:
        logger.error(f"Fits file {file_in} has the invalid type: "
                     f"{type(file_in)}")
        return None
    return cutout, ctr_pt

def get_cutouts(
    coord,
    cutout_size,
    name_target,
    choose_sec=None,
    tot_attempts=3,
    cap_sectors=13,
    fits_dir="fits",
):
    """Download TESS cutouts and store to a list for lightcurve analysis.

    The TESScut function will save fits files to the working directory.

    parameters
    ----------
    coord : `astropy.coordinates.SkyCoord`
        A set of coordinates in the SkyCoord format.
    cutout_size : `float`
        The pixel length of the downloaded cutout.
    name_target : `str`
        Name of the target.
    choose_sec : `None`, `int` or `Iterable`, optional, default=None
        The sector, or sectors required for download.

        * If `None`, TESScut will download all sectors available for the
          target.

        * If `int`, TESScut will attempt to download this sector number.

        * If `Iterable`, TESScut will attempt to download a list of sectors.

    tot_attempts : `int`, optional, default=3
        The number of sql queries in case of request or server errors.
    cap_sectors : `int`, optional, default=None
        The maximum number of sectors for each target.
    fits_dir : `str`, optional, default='fits'
        The name of the directory to store the fits files.

    Returns
    -------
    manifest : `list`
        A list of the fits files for lightcurve analysis.
    """
    if choose_sec is None:
        choose_sec = Tesscut.get_sectors(coordinates=coord)["sector"].data
        logger.info(f"There are {len(choose_sec)} in total: {choose_sec}")
        logger.info(f"There are {len(choose_sec)} in total: {choose_sec}")
        if len(choose_sec) == 0:
            logger.error(f"Sorry, no TESS data available for {name_target}")
            return []
        np.random.shuffle(choose_sec)
    if isinstance(choose_sec, int):
        choose_sec = [choose_sec]

    choose_sec = choose_sec[:cap_sectors]
    if not np.all(np.isin(choose_sec, np.arange(1, sec_max + 1))):
#        raise ValueError(
#            f"Invalid sector numbers: {~choose_sec[np.isin(choose_sec, , np.arange(1, sec_max + 1))]}, available sectors are 1-{sec_max}."
#        )
        logger.warning(f"Invalid sector numbers: {~choose_sec[np.isin(choose_sec, np.arange(1, sec_max + 1))]}, available sectors are 1-{sec_max}."
        )      
        choose_sec = choose_sec[np.isin(choose_sec, np.arange(1, sec_max + 1))]

    manifest = []
    for c in choose_sec:
        num_attempts = 0

        while num_attempts < tot_attempts:
            filename = glob(f"{fits_dir}/{name_target}_{c:04d}*.fits")
            if len(filename) == 1:
                manifest.append(filename[0])
                break
            else:
                try:
                    dl = Tesscut.download_cutouts(
                        coordinates=coord, size=cutout_size, sector=c, path=fits_dir
                    )
                    manifest.append(dl["Local Path"][0])
                    break
                except:
                    print(
                        f"Didn't get Sector {c} data for {name_target}, "
                        f"attempt {num_attempts+1} of {tot_attempts}"
                    )
                    logger.error(
                        f"Didn't get Sector {c} data for "
                        f"{name_target}, attempt {num_attempts+1} of "
                        f"{tot_attempts}"
                    )
            num_attempts += 1

            if num_attempts == tot_attempts:
                print(f"No data for {name_target} in Sector {c}")
                logger.error(f"No data for {name_target} in Sector {c}")
    return manifest


def one_source_cutout(
    target,
    res_table,
    ref_name,
    gaia_sys=True,
    xy_pos=(10.0, 10.0),
    ap_rad=1.0,
    sky_ann=(6.0, 8.0),
    fix_rad=False,
    n_cont=10,
    cont_rad=10.0,
    mag_lim=3.0,
    keep_data=False,
    save_phot=False,
    cbv_flag=False,
    choose_sec=None,
    store_lc=False,
    cutout_size=20,
    tot_attempts=3,
    cap_sectors=13,
    fits_dir="fits",
    lc_dir="lc",
    pg_dir="pg",
    plot_ext="plots",
    clean_fail_modes=False,
    fix_noise=False,
    shuf_per=False,
    make_plot=False,
    calc_cont=True,
    lc_cont=False,
    make_shuf_plot=False,
    shuf_dir="shuf_plots"
):
    """Download cutouts and run lightcurve/periodogram analysis for one target.

    Called by the function "all_sources_cutout".

    parameters
    ----------
    target : `astropy.table.row.Row`
        A row of data from the astropy table.
    lc_con : `bool`
        Decides if a lightcurve analysis is to be performed for the 5 strongest
        contaminants. Here, the data required for further analysis are
        stored in a table.
    flux_con : `bool`
        Decides if the flux contribution from contaminants is to be calculated.
    res_table : `astropy.table.Table`
        The table to store the final tessilator results.
    ref_name : `str`
        The reference name for each subdirectory which will connect all output
        files.
    gaia_sys : `bool`, optional, default=True
        Choose to format the data based on Gaia DR3. Note that no contamination
        can be calculated if this is False.
    xy_pos : `tuple`, size=2x2, optional, default=(10.,10.)
        The centroid of the target in pixels.
    ap_rad : `float`, optional, default=1.
        The size of the aperture radius in pixels.
    sky_ann : `tuple`, optional, default=(6.,8.)
        A 2-element tuple defining the inner and outer annulus to calculate
        the background flux.
    fix_rad : `bool`, optional, default=False
        If True, then set the aperture radius equal to ap_rad, otherwise run the
        calc_rad algorithm.
    n_cont : `int`, optional, default=10
        The maximum number of neighbouring contaminants to store to table.
    cont_rad : `float`, optional, default=10.
        The maximum pixel radius to search for contaminants
    mag_lim : `float`, optional, default=3.
        The faintest magnitude to search for contaminants.
    keep_data : `bool`
        Choose to save the input data to file.
    save_phot : `bool`, optional, default=False
        Decide whether to save the full results from the aperture photometry.
    cbv_flag : `bool`, optional, default=False
        Decide whether to run the lightcurve analysis with a CBV-correction
        applied.
    choose_sec : `None`, `int` or `Iterable`, optional, default=None
        The sector, or sectors required for download.

        * If `None`, TESScut will download all sectors available for the
          target.

        * If `int`, TESScut will attempt to download this sector number.

        * If `Iterable`, TESScut will attempt to download a list of sectors.
    store_lc : `bool`, optional, default=False
        Choose to save the cleaned lightcurve to file.
    cutout_size : `float`, optional, default=20.
        The pixel length of the downloaded cutout.
    tot_attempts : `int`, optional, default=3
        The number of sql queries in case of request or server errors.
    cap_sectors : `None`, `int`, optional, default=None
        The maximum number of sectors for each target.
    fits_dir : `str`, optional, default='fits'
        The name of the directory to store the fits files.
    lc_dir : `str`, optional, default='lc'
        The directory used to store the lightcurve files if store_lc==True.
    pg_dir : `str`, optional, default='pg'
        The directory used to store the periodogram data.
    clean_fail_modes : `bool`, optional, default=False
        Choose to remove parts of the lightcurve that are reported as
        problematic in the TESS Data Release Notes
    fix_noise : `bool`, optional, default=False
        Choose to apply corrections accounting for systematic noise.
    shuf_per : `bool`, optional, default=False
        Choose to run the shuffled period analysis (True=yes, False=no)
    make_shuf_plot : `bool`, optional, default=False
        Choose to make a plot for the shuffled period analysis
    shuf_dir : `str`, optional, default='plot_shuf'
        The name of the directory to save the plots of the shuffled period analysis.

    returns
    -------
    Nothing returned. Results are saved to table and plots are generated (if
    specified).
    """

    # Set the contaminant parameters to the default values in case
    # they have not been added
#    if 'log_tot_bg' not in target.colnames:
#        target.add_column(-999, name='log_tot_bg')
#        target.add_column(-999, name='log_max_bg')
#        target.add_column(0,    name='n_contaminants')
    name_target = target['name'].replace(" ", "_")
    name_spl = name_target.split("_")
    if name_spl[0] == 'Gaia':
        name_target = name_spl[-1]

    coo = SkyCoord(target["ra"], target["dec"], unit="deg")

    # use Tesscut to get the cutout fits files for the target star
    # there may be more than 1 fits file if the target lands in
    # multiple sectors!
    fits_files = get_cutouts(coo, cutout_size, name_target,
                             tot_attempts=tot_attempts, choose_sec=choose_sec,
                             cap_sectors=cap_sectors, fits_dir=fits_dir)
    #print(fits_files)
    if fits_files is None:
        logger.error(f"could not download any data for {target['name']}. "
                     f"Trying next target.")
    else:
        for m, file_in in enumerate(fits_files):
            fits_name = file_in.split('/')[-1]
            print(f'working on {fits_name}, #{m+1} of {len(fits_files)}')
            try:
    # rename the fits file to something more legible for users
                f_sp = file_in.split('/')[-1].split('-')
                #print(f_sp)
                if (len(f_sp) >=3) & (f_sp[0] == 'tess'):
                    f_new = f'{fits_dir}/'+'_'.join([name_target,
                                                     f_sp[1][1:],
                                                     f_sp[2],
                                                     f_sp[3][0]])+'.fits'
                    os.rename(f'./{file_in}', f_new)
                    logger.info(f"target: {target['source_id']}, "
                                f"sector: {f_sp[1][1:]}, "
                                f"{m+1}/{len(fits_files)}")
    # run the lightcurve analysis for the given target/fits file
                elif len(f_sp) == 1:
                    f_new = f'{fits_dir}/{f_sp[0]}'
                else:
                    f_new = file_in
                t_sp = f_new.split('_')
    # simply extract the sector, ccd and camera numbers from the fits file.
                scc = [int(t_sp[-3][1:]), int(t_sp[-2]), int(t_sp[-1][0])]
                #print("Hello!")
                
                full_run_lc(f_new, target, scc, res_table, gaia_sys=gaia_sys,
                            xy_pos=xy_pos, ap_rad=ap_rad, sky_ann=sky_ann,
                            fix_rad=fix_rad, n_cont=n_cont, cont_rad=cont_rad,
                            mag_lim=mag_lim, tot_attempts=tot_attempts,
                            ref_name=ref_name, cutout_size=cutout_size,
                            save_phot=save_phot, cbv_flag=cbv_flag,
                            store_lc=store_lc, calc_cont=calc_cont,
                            lc_cont=lc_cont, lc_dir=lc_dir, pg_dir=pg_dir,
                            plot_ext=plot_ext,
                            clean_fail_modes=clean_fail_modes,
                            fix_noise=fix_noise, keep_data = keep_data, shuf_per=shuf_per,
                            make_plot=make_plot, make_shuf_plot=make_shuf_plot,
                            shuf_dir=shuf_dir)
            except Exception as e:
                logger.error(f"Error occurred when processing {file_in}. "
                             f"Trying next target.")

def all_sources_cutout(t_targets, period_file, ref_name, gaia_sys=True,
                       xy_pos=(10.,10.), ap_rad=1., sky_ann=(6.,8.),
                       fix_rad=False, n_cont=10, cont_rad=10., mag_lim=3.,
                       choose_sec=None, save_phot=False, cbv_flag=False,
                       store_lc=False, tot_attempts=3, cap_sectors=13,
                       res_ext='results', lc_ext='lc', pg_ext='pg',
                       fits_ext='fits', clean_fail_modes=False,
                       keep_data=False, fix_noise=False, shuf_per=False,
                       shuf_ext='shuf_plots', calc_cont=True, lc_cont=False,
                       make_plot=False, make_shuf_plot=False):
    '''Run the tessilator for all targets.

    parameters
    ----------
    t_targets : `astropy.table.Table`
        Table of input data for the tessilator, with the following columns:

        * name: name of the target (`str`)

        * source_id: Gaia DR3 source identifier (`str`)

        * ra: right ascension

        * dec: declination

        * parallax: parallax

        * Gmag: Gaia DR3 apparent G-band magnitude

        * BPmag: Gaia DR3 apparent BP-band magnitude

        * RPmag: Gaia DR3 apparent RP-band magnitude

        * log_tot_bg: log-10 value of the flux ratio between contaminants
          and target (optional)

        * log_max_bg: log-10 value of the flux ratio between the largest
          contaminant and target (optional)

        * num_tot_bg: number of contaminant sources (optional)

    period_file : `str` 
        Name of the file to store periodogram results.
    lc_con : `bool`
        Decides if a lightcurve analysis is to be performed for the 5 strongest
        contaminants. Here, the data required for further analysis are
        stored in a table.
    flux_con : `bool`
        Decides if the flux contribution from contaminants is to be calculated.
    ref_name : `str`
        The reference name for each subdirectory which will connect all output
        files.
    gaia_sys : `bool`, optional, default=True
        Choose to format the data based on Gaia DR3. Note that no contamination
        can be calculated if this is False.
    xy_pos : `tuple`, size=2x2, optional, default=(10.,10.)
        The centroid of the target in pixels.
    ap_rad : `float`, optional, default=1.
        The size of the aperture radius in pixels.
    sky_ann : `tuple`, optional, default=(6.,8.)
        A 2-element tuple defining the inner and outer annulus to calculate
        the background flux.
    fix_rad : `bool`, optional, default=False
        If True, then set the aperture radius equal to ap_rad, otherwise run
        the calc_rad algorithm.
    n_cont : `int`, optional, default=10
        The maximum number of neighbouring contaminants to store to table.
    cont_rad : `float`, optional, default=10.
        The maximum pixel radius to search for contaminants
    mag_lim : `float`, optional, default=3.
        The faintest magnitude to search for contaminants.
    choose_sec : `None`, `int`, or `Iterable`, optional, default=None
        The sector, or sectors required for download.
        
        * If `None`, TESScut will download all sectors available for the
          target.

        * If `int`, TESScut will attempt to download this sector number.

        * If `Iterable`, TESScut will attempt to download a list of sectors.

    save_phot : `bool`, optional, default=False
        Decide whether to save the full results from the aperture photometry.
    cbv_flag : `bool`, optional, default=False
        Decide whether to run the lightcurve analysis with a CBV-correction
        applied.
    store_lc : `bool`, optional, default=False
        Choose to save the cleaned lightcurve to file.
    tot_attempts : `int`, optional, default=3
        The number of sql queries in case of request or server errors.
    cap_sectors : `int`, optional, default=None
        The maximum number of sectors for each target.
    res_ext : `str`, optional, default='results'
        The directory to store the final results file.
    lc_ext : `str`, optional, default='lc'
        The directory used to store the lightcurve files if lc_dir==True
    pg_dir : `str`, optional, default='pg'
        The directory used to store the periodogram data.
    clean_fail_modes : `bool`, optional, default=False
        Choose to remove parts of the lightcurve that are reported as
        problematic in the TESS Data Release Notes
    fits_ext : `str`, optional, default='fits'
        The name of the directory to store the fits files.
    keep_data : `bool`
        Choose to save the input data to file.
    fix_noise : `bool`, optional, default=False
        Choose to apply the noise correction to the cleaned lightcurve.
    shuf_per : `bool`, optional, default=False
        Choose to run the shuffled period analysis (True=yes, False=no)
    make_shuf_plot : `bool`, optional, default=False
        Choose to make a plot for the shuffled period analysis
    shuf_dir : `str`, optional, default='plot_shuf'
        The name of the directory to save the plots of the shuffled period
        analysis. 

    returns
    -------
    Nothing returned. The final table is saved to file and the program
    terminates.
    '''

    start = datetime.now()
    logger.info(f"Starting Time: {start}")
    print("Start time now: ", start.strftime("%d/%m/%Y %H:%M:%S"))
    #print("ALL", flush = True)
    fits_dir = make_dir(fits_ext, ref_name)
    lc_dir = make_dir(lc_ext, ref_name)
    pg_dir = make_dir(pg_ext, ref_name)
    shuf_dir = make_dir(shuf_ext, ref_name)
    res_dir = make_dir(res_ext, ref_name)

   

    res_table = create_table_template()
#    if 'log_tot_bg' not in t_targets.colnames:
#        t_targets.add_column(-999, name='log_tot_bg')
#        t_targets.add_column(-999, name='log_max_bg')
#        t_targets.add_column(0,    name='num_tot_bg')
    for i, target in enumerate(t_targets):
        logger.info(f"{target['name']} (Gaia DR3 {target['source_id']}), star #{i+1}"
                    f" of {len(t_targets)}")
        one_source_cutout(
            target,
            res_table,
            ref_name,
            gaia_sys=gaia_sys,
            xy_pos=xy_pos,
            ap_rad=ap_rad,
            sky_ann=sky_ann,
            fix_rad=fix_rad,
            keep_data=keep_data,
            n_cont=n_cont,
            cont_rad=cont_rad,
            mag_lim=mag_lim,
            save_phot=save_phot,
            cbv_flag=cbv_flag,
            store_lc=store_lc,
            choose_sec=choose_sec,
            tot_attempts=tot_attempts,
            cap_sectors=cap_sectors,
            fits_dir=fits_dir,
            lc_dir=lc_dir,
            pg_dir=pg_dir,
            clean_fail_modes=clean_fail_modes,
            fix_noise=fix_noise,
            shuf_per=shuf_per,
            shuf_dir=shuf_dir,
            calc_cont=calc_cont,
            lc_cont=lc_cont,
            make_plot=make_plot,
            make_shuf_plot=make_shuf_plot,
        )
    finish = datetime.now()
    dt_string = finish.strftime("%b-%d-%Y_%H:%M:%S")

    res_table.write(f'{res_dir}/{period_file}_{dt_string}.csv')

    

    hrs_mins_secs = print_time_taken(start, finish)
    print(f"Finished {len(t_targets)} targets in {hrs_mins_secs}")
    logger.info(f"Total time taken: {hrs_mins_secs}")


def get_fits(scc):
    '''Function which returns a list of fits files corresponding to a
    given Sector, Camera and CCD configuration.

    parameters
    ----------
    sector_num : `int`
        The required sector number.

    scc : `list`, size=3
        List containing the sector number, camera and CCD.
    
    file_dir : `str`
        The name of the base directory containin the fits files.

    returns
    -------
    fits_files : `list`
        A list of the fits files to be used for aperture photometry.
    '''
    
    list_fits = sorted(glob(f"../tess_fits_files/sector{scc[0]:02d}/*.fits"))
    l_cam = np.array([int(j.split('-')[2]) for j in list_fits])
    l_ccd = np.array([int(j.split('-')[3]) for j in list_fits])
    fits_indices = (l_cam == scc[1]) & (l_ccd == scc[2])
    fits_files = np.array(list_fits)[fits_indices]
    #print(fits_files)
    return fits_files




def one_cc(t_targets, scc, res_table, file_ref, ap_rad=1.0,
                      sky_ann=[6.,8.], keep_data=False, fix_noise=False):
    '''Run the tessilator for targets in a given Sector/Camera/CCD
    configuration.

    This routine finds the full-frame calibrated fits files and targets which
    land in a given Sector/Camera/CCD configuration (SCC). Aperture photometry
    is carried out simultaneously for all stars in a given SCC for each fits
    file in chronological order. This makes the method run much faster than
    doing it star-by-star (i.e. vectorisation). The output is a table for each
    SCC and plots for each target (if required).

    parameters
    ----------
    t_targets : `astropy.table.Table`
        Table containing the targets to be analysed.
    scc : `list`, size=3
        List containing the sector number, camera and CCD.
    res_table : `astropy.table.Table`
        The table to store tessilator results.
    file_ref : `str`
        A common string to give all output files the same naming convention.
    ap_rad : `float`, optional, default=1.0
        The pixel radius of the flux collecting area for aperture photometry.
    sky_ann : `Iterable`, size=2, optional, default=[6.,8.]
        The inner and outer background annuli used for aperture photometry.
    keep_data : `bool`
        Choose to save the input data to file.
    fix_noise : `bool`, optional, default=False
        Choose to apply the noise correction to the cleaned lightcurve.

    returns
    -------
    Nothing returned, but the final tessilator table is saved to file.
    '''
    fits_files = get_fits(scc)
    ind = (t_targets['Sector'] == scc[0]) & \
          (t_targets['Camera'] == scc[1]) & \
          (t_targets['CCD'] == scc[2])
    if ind.any() == False:
        return
    print(fits_files)
    suffix_file = f'scc_{scc[0]:02d}_{scc[1]}_{scc[2]}'
    lc_dir = make_dir(f'lc_{suffix_file}', file_ref)
    pg_dir = make_dir(f'pg_{suffix_file}', file_ref)
    full_run_lc(
        fits_files,
        t_targets[ind],
        scc,
        res_table,
        ap_rad=ap_rad,
        sky_ann=sky_ann,
        keep_data=keep_data,
        fix_noise=fix_noise,
        lc_dir=lc_dir,
        pg_dir=pg_dir,
    )


def all_sources_sector(t_targets, scc, period_file, file_ref,
                       keep_data=False, fix_noise=False):
    """Iterate over all cameras and CCDs for a given sector.

    This routine iterates over all cameras and CCDs for a given sector and
    performs the tessilator analysis for each camera/CCD configuration.

    Parameters
    ----------
    t_targets : `astropy.table.Table`
        Input data for the targets to be analysed.
    scc : tuple or int
        This can be a single integer or a tuple of size 3. If a single integer
        is given, then all cameras and CCDs for that sector are analysed. If a
        tuple is given, is has the format `(sector, camera, CCD)`.
    period_file : `str`
        Name of file for recording parameters measured by the periodogram
        analysis.
    file_ref : `str`
        A common string to give all output files the same naming convention.
    keep_data : `bool`
        Choose to save the input data to file.
    fix_noise : `bool`, optional, default=False
        Choose to apply the noise correction to the cleaned lightcurve.

    Returns
    -------
    Nothing returned. The Tessilator data for each camera/CCD configuration
    is saved to file.
    """
    if len(scc) == 3:
        sector = scc[0]
        cameras = [scc[1]]
        ccds = [scc[2]]
    elif isinstance(scc, int):
        sector = scc
        cameras, ccds = np.mgrid[1:5, 1:5]
        cameras = cameras.flatten()
        ccds = ccds.flatten()
    for cam, ccd in zip(cameras, ccds):
        start = datetime.now()
        logger.info(f"Starting Time: {start}")
        print("Start time: ", start.strftime("%d/%m/%Y %H:%M:%S"))

        res_table = create_table_template()
        one_cc(
            t_targets,
            [sector, cam, ccd],
            res_table,
            file_ref=file_ref,
            keep_data=keep_data,
            fix_noise=fix_noise,
        )
        res_table.write(f"{period_file}_{cam}_{ccd}.csv", overwrite=True)
        finish = datetime.now()
        hrs_mins_secs = print_time_taken(start, finish)
        print(
            f"Finished {len(res_table)} targets for Sector {scc[0]},"
            f" Camera {scc[1]}, CCD {scc[2]} in {hrs_mins_secs}"
        )

import os
import numpy as np
import logging
import ast
from multiprocessing import Pool

# Ensure the log directory exists
os.makedirs("outputs_test_ecliptic", exist_ok=True)

# Load tessilator_inputs once at the top
try:
    with open('tessilator_cutout_inputs.txt') as tci:
        data = tci.read()
        tessilator_inputs = ast.literal_eval(data)
except:
    tessilator_inputs = {
        'gaia_sys': True,
        'xy_pos': (10., 10.),
        'ap_rad': 1.,
        'sky_ann': (6., 8.),
        'fix_rad': False,
        'n_cont': 5,
        'cont_rad': 10.,
        'mag_lim': 3.,
        'choose_sec': None,
        'save_phot': True,
        'cbv_flag': False,
        'store_lc': True,
        'tot_attempts': 10,
        'cap_sectors': 13,
        'res_ext': 'results',
        'lc_ext': 'lc',
        'pg_ext': 'pg',
        'fits_ext': 'fits',
        'clean_fail_modes': True,
        'keep_data': False,
        'fix_noise': False,
        'shuf_per': True,
        'shuf_ext': 'shuf_plots',
        'make_plot': True,
        'calc_cont': True,
        'lc_cont': False,
        'make_shuf_plot': True
    }

# troubleshooting for connection timeouts
import time
import traceback

def safe_read_data(*args, retries=5, delay=5, **kwargs):
    for i in range(retries):
        try:
            return read_data(*args, **kwargs)
        except Exception as e:
            print(f"[{i+1}/{retries}] read_data failed: {type(e).__name__}: {e}")
            time.sleep(delay * (2 ** i))
    raise RuntimeError("All retries failed.")
    
import os
from glob import glob
# Wrapper function for parallel execution
def process_one(blah):
    fileRef = f'ecliptic_{blah}'
    tFile = r'/home/loopynoodle/exo_data/SCVZ_Samples/tic_dec66_64/batch2.1.csv'
    log_file = f'__main__.log'
    fits_path = r"/home/loopynoodle/exo_data/fits/{fileRef}"

    logging.basicConfig(filename=log_file, level=logging.INFO, force=True)

    print(f"Reading the table and formatting into astropy table structure for {tFile}")
    tTargets = safe_read_data(tFile, gaia_sys=True, type_coord='icrs', name_is_source_id=False)
    print(f"Done reading {tFile}... Now iterating over each source.")

    periodFile = setup_filenames(fileRef)

    all_sources_cutout(tTargets, periodFile, fileRef, **tessilator_inputs)

    # Delete all contents in fits_path
    for f in glob(os.path.join(fits_path, '*')):
        try:
            os.remove(f)
        except IsADirectoryError:
            print(f"Skipping subdirectory: {f}")


if __name__ == "__main__":
    N = 1
    num_cores = 1
    for i in range(1, N + 1):
        process_one(i)
    '''
    with Pool(processes=num_cores) as pool:
        pool.map(process_one, range(1, N + 1))
    '''
#############at the end do not forget to delete the downloaded fits files