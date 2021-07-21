import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
import gwsurrogate
import pycbc.noise
from pycbc.waveform import get_td_waveform
import pycbc.psd
import pylab
import pandas as pd
import random 

#For whitening the data
from pycbc.filter import highpass_fir, lowpass_fir
from pycbc.waveform import get_fd_waveform
from pycbc.psd import welch, interpolate

time_series_noise, t0 = None, None

def load(path, file_time):
    '''
    ONLY to be used if choice of noise is real LIGO noise
    format - ONLY hdf5 files allowed till now
    file_time :: starting GPS time of the file    
    '''

    global time_series_noise
    global t0
    time_series_noise = TimeSeries.read(path,format = 'hdf5.losc') 
    t0 = file_time

def real_noise(start_time, length): #t0 = 1239085056 - original .hdf5 file, t0 = 1184567296 for .gwf file
    """
    To generate LIGO noise
    
    time_series_noise :: entire noise file
    t0 :: starting time of LIGO hdf5/gwf file
    start_time :: time after t0 
    length :: length of noise array (default = 8)

    Returns real LIGO noise according to input parameters in a numpy array
    """   

    noise_ts = time_series_noise.crop(t0+start_time, t0+start_time+length)
    
    time_domain = np.array(noise_ts.times) - t0 - start_time
    noise_ts = np.array(noise_ts)
    
    return noise_ts 

def gauss_noise(seed, timeInterval, flow=6.0, delta_f=1.0/16):
    """
    Generating Gaussian Noise
    
    seed :: random variable for generating noise from psd 
    timeInterval ::  Total duration of noise signal to be returned in seconds 
    flow :: Lowest frequency bin with non-zero amplitude present in the noise [Default - 6]
    delta_f :: sampling rate of frequency   [Default - 1.0/16]

    Returns Gaussian noise according to input parameters in a numpy array
    """

    # The color of the noise matches a PSD which you provide
    #flow = 6.0 
    #delta_f = 1.0 / 16
    flen = int(2048 / delta_f) + 1
    #obtaining psd from pycbc's psd function
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
    timeInterval *= 4096
    # Generate 4.0 seconds of noise at 4096 Hz
    delta_t = 1.0 / 4096
    #generating noise from psd
    ts = pycbc.noise.noise_from_psd(timeInterval, delta_t, psd, seed = seed)
    return np.array(ts)


def add(strain,noise,insert_time):
    """
    Adding strain and noise by padding the strain.
    The strain usually has a smaller size than the noise file; 
    this function pads it with zeros
    
    Drawback: This function is NOT built for handling GWs bigger than
                noise file itself

    strain :: input GW strain 
    noise ::  Noise in which to embed strain
    insert_time :: time at which GW strain is to be inserted 
                   in the noise

    Returns the added noise + strain (with padding)
    """

    nullarray=np.zeros(len(noise)-len(strain))
    strain = np.insert(nullarray,insert_time,strain)
    event = noise + strain
    return event,strain #returning strain for denoising purposes

def truncate(time_domain, time_series, desiredLength, dt = 1/4096):
    """
    Truncation of waveform
    
    This function makes sure that the waveform 
    is within the limits of the noise array 

    time_domain :: time array
    time_series :: TS of values
    desiredLength :: total final signal duration
                     at the end of pipeline                     
    dt :: sampling interval


    Returns the truncated the time series and the time domain  
    """

    time_interval = dt * len(time_series)
    cropped_ts = time_series
    cropped_td = time_domain
    if time_interval > 0.75 * desiredLength: 
        crop_pos = len(time_series) - int((0.75/dt) * desiredLength) - 1 # change this 3 if necessary
        cropped_ts = time_series[crop_pos:len(time_series)]
        cropped_td = time_domain[crop_pos:len(time_domain)] # - time_domain[0]
    return cropped_td, cropped_ts


#Is there any modification we can make to the whitening process to make it even more efficient at lower SNR?
def whitening(l1,detector='L1'):
    '''Function to whiten data (with gaussian noise)
    takes numpy time series as input and converts it into a pyCBC time series for processing'''
    
    l1 = pycbc.types.timeseries.TimeSeries(initial_array = l1, delta_t = 1/4096)
    l1 = highpass_fir(l1, 15, 8)

        # Calculate the noise spectrum
    psd = interpolate(welch(l1), 1.0 / l1.duration)

        # whiten
    white_strainl1 = (l1.to_frequencyseries() / (psd ** 0.5)).to_timeseries()

        # remove some of the high and low
    smoothl1 = highpass_fir(white_strainl1, 35, 8)
    smoothl1 = lowpass_fir(white_strainl1, 300, 8)

        # time shift and flip L1
        
    if detector == 'L1':
        smoothl1 *= -1
        smoothl1.roll(int(.007 / smoothl1.delta_t))
        
    L1smoothdata=np.array(smoothl1)
    return L1smoothdata

def random_spin():
    '''
    Function to generate randomised spins in the domain [-0.8,+0.8]  
    
    '''
    chiA = list(np.array([0,0,random.randint(-8,8)])/10)
    chiB = list(np.array([0,0,random.randint(-8,8)])/10)
    
    return chiA, chiB


def generate_noise(noiseLen, rand, choice):
    '''
    Generates the noise depending on user choice
    Pass seed if you want to generate gauss noise, 
    start_time if you want to generate real noise

    Input:

    noiseLen - length of noise sample required
    rand - either start_time or seed depending on `choice`

    
    Returns the noise sample
    '''
    if choice == 'gauss':
        return gauss_noise(seed = rand, timeInterval = noiseLen)
    elif choice == 'real':
        return real_noise(start_time = rand, length = noiseLen)
    else:
        return None   
    
def wave_plot(time_series , continue_plot):
    '''
    For plotting waveforms
    
    Input: 

    time_series :: input waveform to be plotted    
    continue_plot :: whether any graphs will be plotted 
                    on the same figure or not
    
    '''
    lim = len(time_series)
    t = np.linspace(0,lim-1,lim)
    t = t/4096
    
    plt.plot(t,time_series)
    
    if not continue_plot: 
        plt.xlabel('Time')
        plt.ylabel('Strain')
        plt.show()
    
def combine(h, noise_ts, insert_pos, plot, whiten, crop):
    '''
    For combining waveform with noise, whitening it and finally cropping it
    
    Input:

    h - input time series
    noise_ts - additive noise time series
    insert_pos - position at which the time series is inserted in noise_ts
    plot - True/False whether to plot the combined waveform or not
    whiten - True/False whether to whiten the waveform or not
    crop - limits of cropping the final whitened strain    

    Output: 

    event - detector data 
    strain - actual GW strain
    '''
    event,strain = add(h, noise_ts,insert_pos)
    if whiten: event = whitening(event) #--------------undo this. This is just for denoising purposes
    event = event[(crop[0]*4096):(crop[1]*4096)] #---------------make this 2*4096:4*4096
    strain = strain[(crop[0]*4096):(crop[1]*4096)]

    if plot:
        wave_plot(event)
    return event,strain #returning strain for denoising purposes