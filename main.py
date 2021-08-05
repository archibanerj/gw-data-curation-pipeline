import sys
import numpy as np
import random 
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform

# Importing core script
import core

''' 
See this: https://pycbc.org/pycbc/latest/html/filter.html#matched-filter-snr
'''

'''
cmd-line arguments: 

fn - filename & path to original LIGO noise file
file_time - starting time for the file

n_sample - number of samples to be saved

m_lower - lower bound of masses
m_upper - upper bound of masses
m_samples - sampling rate in the mass space


dist_lower - lower bound of distances
dist_upper - upper bound of distances
dist_sample - sampling rate in the distance space

dir_wav - link to saving the ground waveforms
dir_detector_data - link to saving the detector data
dir_catalogue - path for saving the catalogue of parameters
dir_paramSpaceMass - path for saving the parameter space of masses
dir_paramSpaceSpin - path for saving the parameter space of spin

signalDuration - duration of signal to be outputted
type_of_noise - 'gauss' or 'noise'

noise_or_signal - 'noise', 'signal', 'all'
noise_path - path for saving the noise file
'''

flow = 25.0 # Lowest frequency in pycbc waveform

fn = sys.argv[1]
core.load(fn, file_time = int(sys.argv[2]), flow = flow)

n_sample = int(sys.argv[3]) 

m_lower = int(float(sys.argv[4]))
m_upper = int(float(sys.argv[5]))
m_sample = int(float(sys.argv[6]))

dist_lower = int(float(sys.argv[7]))
dist_upper = int(float(sys.argv[8]))
dist_sample = int(float(sys.argv[9]))

dir_wav = sys.argv[10]
dir_detector_data = sys.argv[11]
dir_catalogue = sys.argv[12]
dir_paramSpaceMass = sys.argv[13]
dir_paramSpaceSpin = sys.argv[14]

signalDuration = int(float(sys.argv[15]))
type_of_noise = sys.argv[16]

noise_or_signal = sys.argv[17]
noise_path = sys.argv[18]

'''
Issues:

Add header row to catalogue
Provide a way for generating Gaussian and real noise
Update README.md with full instructions

'''


def generate(mass1, mass2, spin1, spin2, dist,
            signalDuration, noiseType,
            dt = 1.0/4096):

    '''
    Input: 

    mass1 - lighter mass
    mass2 - heavier mass
    spin1 - spin along z direction of 1st BH
    spin2 - spin along z direction of 2nd BH
    dist - distance between BBH system and observer
    noiseType - either Gauss Noise or Real Noise
    signalDuration is in seconds

    Output:

    event - detector data
    strain - actual waveform
    insert_pos - index at which event can be inserted 
    ret - either start_time (time from which noise is cropped from the original noise file),
        or seed for Gaussian noise. Depends on noiseType

    '''
    global flow
    h, hc = get_td_waveform(approximant='IMRPhenomD_NRTidalv2',
                                 mass1 = mass1,
                                 mass2 = mass2,
                                 spin1z=spin1, spin2z = spin2,
                                 delta_t=1.0/4096,
                                 distance=dist, f_lower = flow)
    t = np.arange(len(h))
    h = np.array(h)

    # Total signal duration padded with 3s at start and 3s at end
    noiseDuration = 6 + signalDuration
    start_time = random.randint(0,4096-noiseDuration) # for Real Noise
    seed = np.random.randint(10000) # for Gaussian noise

    if len(t) * dt > 0.75 * signalDuration:
        t,h = core.truncate(t,h,signalDuration)

    # Generating the noise
    if noiseType == 'real' : 
        noise_ts, hp_noise_ts = core.generate_noise(noiseDuration, rand = start_time, choice = noiseType) 
    else :
        noise_ts = core.generate_noise(noiseDuration, rand = seed, choice = noiseType)


    # This limit is introduced so that the insert time + strain length 
    # does not exceed the length of the noise
    ''' Fatal error here. Check this out. '''
    insert_limit = ((3 + signalDuration)*4096) -len(t)
    insert_pos = random.randint(3*4096,insert_limit)

    event,strain,snr = core.combine(h, noise_ts, hp_noise_ts, flow, insert_pos, plot = False, whiten = True, crop = (3,3+signalDuration), snr_calc = True)

    event = event.reshape(1, signalDuration*4096)
    strain = strain.reshape(1, signalDuration*4096)

    # defining returned variable for noise generation
    # according to noise type
    if noiseType == 'real':
        ret = start_time
    else :
        ret = seed

    return event,strain,insert_pos,ret,snr


def generate_and_save_signal(num_sample, mass_lower, mass_upper, 
                            mass_sample, d_lower, d_upper, d_sample,
                            path_wav, path_detector_data, path_catalogue,
                            path_paramSpaceMass, path_paramSpaceSpin,
                            signalLen, noiseType):    
    '''
    This generates and saves the noisy whitened/unwhitened signals 
    along with the actual waveforms, and the catalogue and the parameter space plots.


    Input:

    num_sample - number of ground waveforms and detector data samples to be saved
    mass_lower - lower bound of masses
    mass_upper - upper bound of masses
    mass_samples - sampling rate in the mass space

    d_lower - lower bound of distances
    d_upper - upper bound of distances
    d_sample - sampling rate in the distance space

    path_wav - link to saving the ground waveforms
    path_detector_data - link to saving the detector data
    path_catalogue - path for saving the catalogue of parameters
    path_paramSpaceMass - path for saving the parameter space of masses
    path_paramSpaceSpin - path for saving the parameter space of spin

    signalLen - Length of signal to be saved
    noiseType - type of noise ('gauss'/'real')

    Output: 
    Saved waveforms and data from detector, and catalogue, parameter space of masses and spins.

    '''

    sl_no = 0
    catalogue = None
    
    f_low = 25.0
    dt = 1.0/4096

    # For Generating and saving the Mass Parameter Space
    M2,M1 = [],[]

    # m2 is heavier mass, m1 is lighter mass.
    for m2 in range(mass_lower, mass_upper, mass_sample):
        for m1 in range(mass_lower, m2+1, mass_sample):
            for d in range(d_lower, d_upper, d_sample):
                
                #Recording for getting the parameter space
                M1.append(m1)
                M2.append(m2)

                # obtaining the spins
                spin1,spin2 = core.random_spin()

                # obtaining the event in the detector and the trua strain
                event, strain, insert_pos, start_time, snr = generate(m1, m2, spin1[2], spin2[2], d, signalLen, noiseType)

                # updating catalogue
                temp = np.array([sl_no, start_time, insert_pos, m1, m2, d, spin1[2], spin2[2], snr]) #this is for real noises
                temp = np.reshape(temp,(1,len(temp)))                
                if catalogue is None: catalogue = temp 
                else: catalogue = np.r_[catalogue,temp]
                
                # saving detector data and true strain
                fn = path_detector_data +'detectorData_' + str(sl_no+1) + '.csv'       
                np.savetxt(fn, event, delimiter=',')
                fn = path_wav +'trueStrain_' + str(sl_no+1) + '.csv'       
                np.savetxt(fn, strain, delimiter=',')

                sl_no += 1

                if sl_no >= num_sample :
                    break
            if sl_no >= num_sample :
                break
        if sl_no >= num_sample :
            break

    # saving catalogue
    np.savetxt(path_catalogue + 'catalogue.csv', catalogue, fmt = '%.1f', delimiter = ',')
    
    # Save Parameter Space for Mass
    plt.scatter(M2,M1)
    plt.xlabel('Heavier Mass M2')
    plt.ylabel('Lighter Mass M1')
    plt.savefig(path_paramSpaceMass + 'ParamSpaceMass.jpg')
    plt.close()

    # Saving Parameter Space of Spin
    data = np.transpose(catalogue)
    chiAz = data[6]
    chiBz = data[7]
    plt.scatter(chiAz, chiBz)
    plt.xlabel('chi1z')
    plt.ylabel('chi2z')
    plt.savefig(path_paramSpaceSpin + 'ParamSpaceSpin.jpg')


def generate_and_save_noise(noiseType, noiseLen, num_noise, noisePath):
    '''
    This function generates and saves the noise according to the 
    passed arguments

    Input:
    
    noiseType - 'real', 'gauss'
    noiseLen - Length of the signal
    num_noise - number of noise samples
    noisePath - path for saving the file

    Output:

    Saved noise files in noisePath   
    
    '''
    global flow

    totalLen = 6 + noiseLen

    for i in range(num_noise):

        start_time = random.randint(0,4096-totalLen) # for Real Noise
        seed = np.random.randint(10000) # for Gaussian noise


        if noiseType == 'real': 
            noise_ts, hp_noise_ts = core.generate_noise(totalLen, rand = start_time, choice = noiseType) 
        else:
            noise_ts = core.generate_noise(totalLen, rand = seed, choice = noiseType)

        h = np.array([0])
        t = np.array([0])
        insert_limit = 6*4096 -len(t)
        insert_pos = random.randint(3*4096,insert_limit)
        pure_noise,strain,snr = core.combine(h, noise_ts, hp_noise_ts, flow, insert_pos, plot = False, whiten = True, crop = (3,3+noiseLen), snr_calc = False)

        # Saving the noise as a .csv file
        pure_noise = pure_noise.reshape(1,noiseLen*4096)
        np.savetxt(noisePath + 'noise-' + noiseType + '_' + str(i+1) + '.csv', pure_noise, delimiter = ',')


def main():
    '''
    This is the main function - which calls every other function

    The `noise_or_signal` variable is passed via the args.txt file
    this can be either 'noise' , 'signal' , 'all'

    '''

    if noise_or_signal == 'noise':

        generate_and_save_noise(type_of_noise, signalDuration, n_sample, noise_path)

    elif noise_or_signal == 'signal':

        generate_and_save_signal(n_sample, m_lower, m_upper, m_sample, dist_lower, dist_upper, 
                                dist_sample, dir_wav, dir_detector_data, dir_catalogue,
                                dir_paramSpaceMass, dir_paramSpaceSpin, signalDuration, type_of_noise)

    elif noise_or_signal == 'all':

        generate_and_save_noise(type_of_noise, signalDuration, n_sample, noise_path)

        generate_and_save_signal(n_sample, m_lower, m_upper, m_sample, dist_lower, dist_upper, 
                                dist_sample, dir_wav, dir_detector_data, dir_catalogue,
                                dir_paramSpaceMass, dir_paramSpaceSpin, signalDuration, type_of_noise)

    else: 

        print('Bad input - check the `noise_or_signal` variable in args file')


if __name__ == "__main__":
    main()