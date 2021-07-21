import sys
import numpy as np
import random 
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform

#Importing core script
import core

fn = sys.argv[1]
core.load(fn, file_time = int(sys.argv[2]))

num_sample = int(sys.argv[3])

mass_lower = int(float(sys.argv[4]))
mass_upper = int(float(sys.argv[5]))
mass_sample = int(float(sys.argv[6]))

d_lower = int(float(sys.argv[7]))
d_upper = int(float(sys.argv[8]))
d_sample = int(float(sys.argv[9]))

path_wav = sys.argv[10]
path_detector_data = sys.argv[11]
path_catalogue = sys.argv[12]
path_paramSpaceMass = sys.argv[13]
path_paramSpaceSpin = sys.argv[14]

signalLen = int(float(sys.argv[15]))
noiseType = sys.argv[16]


def generate(mass1, mass2, spin1, spin2, dist,
            signalDuration, noiseType,
            f_low = 25, dt = 1.0/4096):

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
    h, hc = get_td_waveform(approximant='IMRPhenomD_NRTidalv2',
                                 mass1 = mass1,
                                 mass2 = mass2,
                                 spin1z=spin1, spin2z = spin2,
                                 delta_t=1.0/4096,
                                 distance=dist,
                                 f_lower=f_low)
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
        noise_ts = core.generate_noise(noiseDuration, rand = start_time, choice = noiseType) 
    else :
        noise_ts = core.generate_noise(noiseDuration, rand = seed, choice = noiseType)


    # This limit is introduced so that the insert time + strain length 
    # does not exceed the length of the noise
    insert_limit = 6*4096 -len(t)
    insert_pos = random.randint(3*4096,insert_limit)

    event,strain = core.combine(h, noise_ts, insert_pos, plot = False, whiten = True, crop = (3,3+signalDuration))

    event = event.reshape(1, signalDuration*4096)
    strain = strain.reshape(1, signalDuration*4096)

    # defining returned variable for noise generation
    # according to noise type
    if noiseType == 'real':
        ret = start_time
    else :
        ret = seed

    return event,strain,insert_pos,ret 


# define main function
def main():    
    '''
    Main function which calls every other function

    Saves the files in the provided destination, and according to all the other specifications.
    Also saves the catalogue and the parameter space description for the masses and spins.
    
    '''

    sl_no = 1
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
                event, strain, insert_pos, start_time = generate(m1, m2, spin1[2], spin2[2], d, signalLen, noiseType)

                # updating catalogue
                temp = np.array([sl_no, start_time, insert_pos, m1, m2, d, spin1[2], spin2[2]]) #this is for real noises
                temp = np.reshape(temp,(1,len(temp)))                
                if catalogue is None: catalogue = temp 
                else: catalogue = np.r_[catalogue,temp]
                
                # saving detector data and true strain
                fn = path_detector_data +'detectorData_' + str(sl_no) + '.csv'       
                np.savetxt(fn, event, delimiter=',')
                fn = path_wav +'trueStrain_' + str(sl_no) + '.csv'       
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

if __name__ == "__main__":
    main()