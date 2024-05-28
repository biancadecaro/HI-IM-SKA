""""
Questo script prende le mappe simulate di Isabella Carucci,
le degrada da Nside=256 a Nside=128, fas il merging tra le mappe per
ridurre il numero di canali (fa na semplice media tra le mappe in uno stesso canale)
e le stora in in tre file: 
uno con frequenze 900-1299 MHz con Delta_nu=2;
uno con frequenze 900-1099MHz con Delta-nu=2 MHz e l'altro con 
1100-1280 MHz con Delta-nu=2 MHz.
Le mappe finali saranno così composte:
'maps_sims_tot' = HI+gal_sync+gal_ff+ps
'maps_sims_HI' = HI
'maps_sims_fg' = gal_sync+gal_ff+ps
Tutte le mappe hanno media (su tutti i pixel) = 0
"""
import h5py
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


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

path_data = 'sim_PL05_from191030.hd5'
file = h5py.File(path_data,'r')
nu_ch = np.array(file['frequencies'])


file_new={}

delta_nu_out = 2
file_new['frequencies'] = nu_ch_f(nu_ch,delta_nu_out)#np.array([nu_ch[i*delta_nu] for i in range(0,int(len(nu_ch)/delta_nu))])

components = list(file.keys())
print(components)
components.remove('frequencies')
components.remove('pol_leakage')

#components.remove('gal_synch')
#components.remove('gal_ff')#
#components.remove('point_sources')


for c in components:
  print(c)
  file_new[c]=merging_maps(nu_ch,file_new['frequencies'],file[c], delta_nu_out )

print(len(file_new['frequencies']), hp.get_nside(file_new['cosmological_signal'][1]))
 
nside = hp.get_nside(file_new['cosmological_signal'][1])

del file

nu_ch_new = np.array(file_new['frequencies'])
num_freq_new=len(nu_ch_new)
npix = np.shape(file_new['cosmological_signal'])[1]

obs_maps = np.zeros((num_freq_new,npix))
fg_maps = np.zeros((num_freq_new,npix))

for c in components:
    print(c)
    obs_maps += np.array(file_new[c])

for cc in components:
    if cc=='cosmological_signal':
      continue
    print(cc)
    fg_maps += np.array(file_new[cc])
    
ich =100

obs_maps_no_mean = np.array([obs_maps[i] -np.mean(obs_maps[i],axis=0)  for i in range(num_freq_new)])
HI_maps_no_mean = np.array([file_new['cosmological_signal'][i] -np.mean(file_new['cosmological_signal'][i],axis=0) for i in range(num_freq_new)])
fg_maps_no_mean = np.array([fg_maps[i] -np.mean(fg_maps[i],axis=0) for i in range(num_freq_new)]) 
#

file_sims = {}
file_sims['freq'] = nu_ch_new
file_sims['maps_sims_tot'] =  obs_maps_no_mean
file_sims['maps_sims_fg'] = fg_maps_no_mean
file_sims['maps_sims_HI'] = HI_maps_no_mean

del obs_maps; del fg_maps; del file_new['cosmological_signal']

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch_new[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(file_sims['maps_sims_tot'][ich], cmap='viridis',title=f'Observations, freq={nu_ch_new[ich]}', hold=True)
fig.add_subplot(222) 
hp.mollview(file_sims['maps_sims_HI'][ich], cmap='viridis',title=f'HI signal, freq={nu_ch_new[ich]}',min=0, max=1,hold=True)
fig.add_subplot(223)
hp.mollview(file_sims['maps_sims_fg'][ich],title=f'Foregrounds, freq={nu_ch_new[ich]}',cmap='viridis',hold=True)
#plt.savefig('plots_PCA/maps_fg_HI_obs_input.png')
#plt.show()

import pickle
filename = f'Sims/no_mean_sims_synch_ff_ps_{len(file_sims['freq'])}freq_{min(file_sims['freq'])}_{max(file_sims['freq'])}MHz_nside{nside}'
with open(filename+'.pkl', 'wb') as f:
    pickle.dump(file_sims, f)
    f.close()
del file_sims
