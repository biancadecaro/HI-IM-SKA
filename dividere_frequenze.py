import numpy as np
import pickle
import healpy as hp
from needlets_analysis import analysis

path_data = 'sims_400freq_900_1280MHz_nside128'

with open(path_data+'.pkl', 'rb') as f:
        file = pickle.load(f)
print('file open')
nu_ch = np.array(file['frequencies'])
num_freq=len(nu_ch)
components = list(file.keys())
components.remove('frequencies')
components.remove('pol_leakage')

print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')

# divido frequenze
print(int(num_freq/2))

nu_ch_1=np.array([nu_ch[i] for i in range(int(num_freq/2))])
nu_ch_2=np.array([nu_ch[i] for i in range(int(num_freq/2), num_freq)])

import psutil
npix = np.shape(file['cosmological_signal'])[1]
nside = hp.get_nside(file['cosmological_signal'][0])
obs_maps = np.zeros((num_freq,npix))


for c in components:
    print(c)
    obs_maps += np.array(file[c])
    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    

## remove mean from maps
obs_maps_1_no_mean = np.array([obs_maps[i] -np.mean(obs_maps[i],axis=0) for i in range(len(nu_ch_1))])
obs_maps_2_no_mean = np.array([obs_maps[i] -np.mean(obs_maps[i],axis=0) for i in range(len(nu_ch_2), num_freq)])

HI_maps_1_no_mean = np.array([file['cosmological_signal'][i] -np.mean(file['cosmological_signal'][i],axis=0) for i in range(len(nu_ch_1))])
HI_maps_2_no_mean = np.array([file['cosmological_signal'][i] -np.mean(file['cosmological_signal'][i],axis=0) for i in range(len(nu_ch_2), num_freq)])

del file

file_sims_tot = {}
file_sims_tot['freq'] = nu_ch_2
file_sims_tot['maps_sims_tot'] = obs_maps_2_no_mean

file_sims_HI = {}
file_sims_HI['freq'] = nu_ch_2
file_sims_HI['maps_sims_HI'] = HI_maps_2_no_mean

filename_sims_tot = f'foregrounds+HI_sims_{len(nu_ch_2)}freq_{min(nu_ch_2)}_{max(nu_ch_2)}MHz_nside128'
with open(filename_sims_tot+'.pkl', 'wb') as f:
    pickle.dump(file_sims_tot, f)
    f.close()

filename_sims_HI = f'HI_sims_{len(nu_ch_2)}freq_{min(nu_ch_2)}_{max(nu_ch_2)}MHz_nside128'
with open(filename_sims_HI+'.pkl', 'wb') as ff:
    pickle.dump(file_sims_HI, ff)
    ff.close()