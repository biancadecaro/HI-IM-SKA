import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
from needlets_analysis import analysis
import cython_mylibc as pippo
import os

path_data_sims_tot = 'Sims/no_mean_sims_synch_ff_ps_200freq_901.0_1299.0MHz_nside128'
with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

nu_ch= file['freq']

num_freq = len(nu_ch)
nu0 =1420
print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')
print(f'corresponding to the redshift range z: [{min(nu0/nu_ch -1.0):.2f} - {max(nu0/nu_ch -1.0):.2f}] ')

HI_maps_freq = file['maps_sims_HI']
fg_maps_freq = file['maps_sims_fg']
full_maps_freq = file['maps_sims_tot']
print(HI_maps_freq.shape)

ich=100
fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(full_maps_freq[ich], cmap='viridis', title=f'Observation', min=-1e3, max=1e4, hold=True)
fig.add_subplot(222) 
hp.mollview(HI_maps_freq[ich], cmap='viridis', title=f'HI signal',min=0, max=1,hold=True)
fig.add_subplot(223)
hp.mollview(fg_maps_freq[ich], title=f'Fg signal',cmap='viridis', min=-1e3, max=1e4, hold=True)
fig.add_subplot(224)
hp.mollview(full_maps_freq[ich]-fg_maps_freq[ich], title=f'Observation - Fg',cmap='viridis',min=0, max=1, hold=True)
plt.show()

del file

npix = np.shape(HI_maps_freq)[1]
nside = hp.get_nside(HI_maps_freq[0])
lmax=256#2*nside#3*nside-1#
jmax=4
out_dir = './Maps_needlets/Maps_no_mean/'
if not os.path.exists(out_dir):
        os.makedirs(out_dir)

need_analysis = analysis.NeedAnalysis(jmax, lmax, out_dir, full_maps_freq)
need_analysis_HI = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_maps_freq)
need_analysis_fg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_maps_freq)

B=need_analysis.B

fname_obs_tot=f'bjk_maps_no_mean_obs_synch_ff_ps_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_jmax{jmax}_lmax{lmax}_B{B:0.2f}_nside{nside}'
fname_HI=f'bjk_maps_no_mean_HI_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_jmax{jmax}_lmax{lmax}_B{B:0.2f}_nside{nside}'
fname_fg=f'bjk_maps_no_mean_fg_synch_ff_ps_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_jmax{jmax}_lmax{lmax}_B{B:0.2f}_nside{nside}'


#try:
#        betajk_sims_tot = np.load(out_dir + fname_obs_tot+'.npy')
#        print("...simulations beta_jk's " + out_dir + fname_obs_tot + " found...")
#
#        betajk_sims_tot = np.load(out_dir + fname_obs_tot+'.npy')
#        print("...simulations beta_jk's " + out_dir + fname_obs_tot + " found...")
#
#        betajk_sims_tot = np.load(out_dir + fname_obs_tot+'.npy')
#        print("...simulations beta_jk's " + out_dir + fname_obs_tot + " found...")
#except:
#        print("...simulations beta_jk's " + out_dir + fname_obs_tot + " not found...")
#        print("...evaluating...")

map_need_output = np.zeros((num_freq, jmax+1, npix))
map_HI_need_output = np.zeros((num_freq, jmax+1, npix))
map_fg_need_output = np.zeros((num_freq, jmax+1, npix))
ich=100
j_test=4#7

for nu in range(num_freq):
        map_need_output[nu] = pippo.mylibpy_needlets_f2betajk_healpix_harmonic(full_maps_freq[nu], B, jmax,lmax )
np.save(out_dir+fname_obs_tot,map_need_output)

fig = plt.figure(figsize=(10, 7))
hp.mollview(map_need_output[ich,j_test], cmap='viridis', title=f'Observation, j={j_test}, freq={nu_ch[ich]}', hold=True)
del map_need_output; del full_maps_freq; del need_analysis

for nu in range(num_freq):        
        map_HI_need_output[nu] = pippo.mylibpy_needlets_f2betajk_healpix_harmonic(HI_maps_freq[nu], B, jmax,lmax )
np.save(out_dir+fname_HI,map_HI_need_output)
fig = plt.figure(figsize=(10, 7))
hp.mollview(map_HI_need_output[ich,j_test], cmap='viridis', title=f'HI, j={j_test}, freq={nu_ch[ich]}', hold=True)

del map_HI_need_output; del HI_maps_freq; del need_analysis_HI

for nu in range(num_freq):        
        map_fg_need_output[nu] = pippo.mylibpy_needlets_f2betajk_healpix_harmonic(fg_maps_freq[nu], B, jmax,lmax )
np.save(out_dir+fname_fg,map_fg_need_output)
fig = plt.figure(figsize=(10, 7))
hp.mollview(map_fg_need_output[ich,j_test], cmap='viridis', title=f'Fg, j={j_test}, freq={nu_ch[ich]}', hold=True)

del map_fg_need_output; del fg_maps_freq; del need_analysis_fg





plt.show()




