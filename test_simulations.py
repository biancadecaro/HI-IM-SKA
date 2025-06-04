import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
from needlets_analysis import analysis, theory
import cython_mylibc as pippo
import os
import seaborn as sns
sns.set_theme(style = 'white')
from matplotlib import colors
import matplotlib as mpl
mpl.rc('xtick', direction='in', top=False, bottom = True)
mpl.rc('ytick', direction='in', right=False, left = True)

sns.palettes.color_palette()
import cython_mylibc as pippo

plt.rcParams['figure.figsize']=(11,7)
plt.rcParams['axes.titlesize']=20
plt.rcParams['lines.linewidth']  = 3.
plt.rcParams['lines.markersize']=6
plt.rcParams['axes.labelsize']  =20
plt.rcParams['legend.fontsize']=20
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.formatter.use_mathtext']=True
plt.rcParams['savefig.dpi']=300


from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
#############################
beam_s= 'SKA_AA4'
path_data_sims_tot = f'Sims/beam_{beam_s}_no_mean_sims_synch_ff_ps_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax767_nside256'
with open(path_data_sims_tot+'.pkl', 'rb') as f:
	file = pickle.load(f)
	f.close()
	
path_data_sims_pol_tot = f'Sims/beam_{beam_s}_no_mean_sims_synch_ff_ps_pol_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax767_nside256'
with open(path_data_sims_pol_tot+'.pkl', 'rb') as ff:
	file_pol = pickle.load(ff)
	ff.close()

nu_ch= file['freq']

num_freq = len(nu_ch)
nu0 =1420
print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')
print(f'corresponding to the redshift range z: [{min(nu0/nu_ch -1.0):.2f} - {max(nu0/nu_ch -1.0):.2f}] ')

HI_noise_maps_freq = file['maps_sims_HI'] + file['maps_sims_noise']
fg_maps_freq = file['maps_sims_fg']
full_maps_freq = file['maps_sims_tot'] + file['maps_sims_noise']
noise_maps_freq = file['maps_sims_noise']


npix = np.shape(HI_noise_maps_freq)[1]
nside = hp.get_nside(HI_noise_maps_freq[0])
lmax=3*nside-1#2*nside#


HI_noise_maps_freq_pol = file_pol['maps_sims_HI'] + file_pol['maps_sims_noise']
fg_maps_freq_pol = file_pol['maps_sims_fg']
full_maps_freq_pol = file_pol['maps_sims_tot'] + file_pol['maps_sims_noise']
noise_maps_freq_pol = file_pol['maps_sims_noise']

del file; del file_pol

ich=int(num_freq/2)

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(full_maps_freq[ich], cmap='viridis', title=f'Observation', min=-1e3, max=1e4, hold=True)
fig.add_subplot(222) 
hp.mollview(HI_noise_maps_freq[ich], cmap='viridis', title=f'HI signal + noise',min=0, max=1,hold=True)
fig.add_subplot(223)
hp.mollview(fg_maps_freq[ich], title=f'Fg signal',cmap='viridis', min=-1e3, max=1e4, hold=True)
fig.add_subplot(224)
hp.mollview(noise_maps_freq[ich], title=f'Noise',cmap='viridis', hold=True)



fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz, polarization leakage',fontsize=20)
fig.add_subplot(221) 
hp.mollview(full_maps_freq_pol[ich], cmap='viridis', title=f'Observation', min=-1e3, max=1e4, hold=True)
fig.add_subplot(222) 
hp.mollview(HI_noise_maps_freq_pol[ich], cmap='viridis', title=f'HI signal + noise',min=0, max=1,hold=True)
fig.add_subplot(223)
hp.mollview(fg_maps_freq_pol[ich], title=f'Fg signal',cmap='viridis', min=-1e3, max=1e4, hold=True)
fig.add_subplot(224)
hp.mollview(noise_maps_freq_pol[ich], title=f'Noise',cmap='viridis', hold=True)


fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz, polarization leakage',fontsize=20)

hp.mollview(full_maps_freq_pol[ich]-full_maps_freq[ich], cmap='viridis', title='obs with pol - obs w/o pol')
plt.show()