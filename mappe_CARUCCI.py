import h5py
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import seaborn as sns
from sys import getsizeof


def merging_maps(nu_ch_in,nu_ch_out,maps_in,deltanu_out):
	
	deltanu_in = abs(nu_ch_in[-1]-nu_ch_in[-2])
	maps_out  = [0] * len(nu_ch_out)  

	deltanu_out = abs(nu_ch_out[-1]-nu_ch_out[-2])
	N = int(deltanu_out/deltanu_in)
	if (deltanu_in*N)!=deltanu_out:
		print('just dnu multiples!')
		exit		

	for i in range(len(nu_ch_out)):
		maps_out[i] = sum(maps_in[N*i:N*i+N]) / N
		
	return maps_out

def nu_ch_f(nu_ch_in,dnu_out):
	du_in = abs(nu_ch_in[-1]-nu_ch_in[-2])
	a1 = nu_ch_in[0] - du_in/2; a2 = nu_ch_in[-1] + du_in/2
	M = int((a2-a1)/dnu_out)
	if (dnu_out*M)!=(a2-a1):
		print('just dnu multiples!')
		exit
	nu_ch_out = np.linspace(a1+dnu_out/2,a2-dnu_out/2,M)
	return nu_ch_out	


nside_out= 128

path_data = 'sim_PL05_from191030.hd5'
file = h5py.File(path_data,'r')

nu_ch = np.array(file['frequencies'])
nu0 =1420
print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')
print(f'corresponding to the redshift range z: [{min(nu0/nu_ch -1.0):.2f} - {max(nu0/nu_ch -1.0):.2f}] ')

npix = file['gal_synch'].shape[1]

ich =199
fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'Channel {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(231) 
hp.mollview(file['gal_synch'][ich], cmap='viridis', norm='log',title=f'Gal sync', hold=True)
fig.add_subplot(232) 
hp.mollview(file['gal_ff'][ich], cmap='viridis', norm='log',title=f'Gal ff',hold=True)
fig.add_subplot(233)
hp.mollview(file['point_sources'][ich], norm='log',title=f'Point sources',cmap='viridis', hold=True)
fig.add_subplot(234)
hp.mollview(file['pol_leakage'][ich], title=f'Polarization leakage',cmap='viridis', hold=True)
fig.add_subplot(235)
hp.mollview(file['cosmological_signal'][ich], norm='log', title=f'Cosmological signal',cmap='viridis', hold=True)
plt.savefig(f'maps_input_carucci_ch{nu_ch[ich]}_nside{hp.npix2nside(npix)}.png')
plt.show()

#print(getsizeof(file['cosmological_signal'][ich]), getsizeof(file['gal_synch'][ich]), getsizeof(file['gal_ff'][ich]), getsizeof(file['point_sources'][ich]))

print()

components = list(file.keys())
components.remove('frequencies')
components.remove('pol_leakage')

obs_maps = np.zeros((len(nu_ch),npix))#,dtype = np.float128)
fg_maps = np.zeros((len(nu_ch),npix))#,dtype = np.float128)
print(obs_maps.dtype)
for c in components:
    print(c)
    obs_maps += np.array(file[c])
	
for cc in components:
    if cc=='cosmological_signal':
      continue
    print(cc)
    fg_maps += np.array(file[cc])
diff =np.mean(obs_maps, axis=0)-np.mean(fg_maps, axis=0)-np.mean(file['cosmological_signal'], axis=0)
plt.hist(diff, bins=50, range= [-0.01e-10, -0.01e-10])
plt.show()	
hp.mollview(np.mean(obs_maps, axis=0)-np.mean(fg_maps, axis=0)-np.mean(file['cosmological_signal'], axis=0), cmap='viridis', hold=True)
plt.show()



