import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from needlets_analysis import analysis, theory
import os
import pickle

import seaborn as sns
sns.set_theme(style = 'white')
from matplotlib import colors
import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

#print(sns.color_palette("husl", 15).as_hex())
sns.palettes.color_palette()
import cython_mylibc as pippo

mpl.rcParams['font.size']=18
##########################################################################################

beam_s = 'SKA_AA4'
fg_comp = 'synch_ff_ps'
path_data_sims_tot = f'../Sims/beam_{beam_s}_no_mean_sims_{fg_comp}_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax767_nside256'
with open(path_data_sims_tot+'.pkl', 'rb') as f:
	file = pickle.load(f)
	f.close()
nu_ch= file['freq']
HI_noise_maps_freq = file['maps_sims_HI'] + file['maps_sims_noise']
del __file__
out_dir_maps_recon = './'
if not os.path.exists(out_dir_maps_recon):
		os.makedirs(out_dir_maps_recon)
		
num_ch=len(nu_ch)
min_ch = min(nu_ch)
max_ch = max(nu_ch)
nside=hp.get_nside(HI_noise_maps_freq[0])
npix= hp.nside2npix(nside)
jmax=4
lmax= 3*nside-1
if fg_comp=='synch_ff_ps':
	Nfg=3
if fg_comp=='synch_ff_ps_pol':
	Nfg=18
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)

##################################################


pix_mask = hp.query_strip(nside, theta1=np.pi*2/3, theta2=np.pi/3)
print(pix_mask)
mask_50 = np.zeros(npix)
mask_50[pix_mask] =1
fsky_50 = np.sum(mask_50)/hp.nside2npix(nside)

bad_v = np.where(mask_50==0)

for n in range(num_ch):
		HI_noise_maps_freq[n][bad_v] =  hp.UNSEEN
		HI_noise_maps_freq[n]=hp.remove_dipole(HI_noise_maps_freq[n])


j_test=2
ich=int(num_ch/2)
#fig = plt.figure(figsize=(10, 7))
#hp.mollview(HI_noise_maps_freq[ich], cmap='viridis', title=f'HI, freq={nu_ch[ich]}', hold=True)
#plt.show()

###################################################################

need_theory=theory.NeedletTheory(B,jmax, lmax)

b_values = pippo.mylibpy_needlets_std_init_b_values(B,jmax,lmax)

need_analysis_HI = analysis.NeedAnalysis(jmax, lmax, out_dir_maps_recon, HI_noise_maps_freq)

fname_HI=f'test_bjk_maps_HI_noise_{beam_s}_jmax{jmax}_lmax{lmax}_nside{nside}'
betajk_HI = np.zeros((num_ch, jmax+1, npix))
for nu in range(num_ch):        
	betajk_HI[nu] = pippo.mylibpy_needlets_f2betajk_healpix_harmonic(HI_noise_maps_freq[nu], B, jmax,lmax )
#np.save(out_dir_maps_recon+fname_HI,map_HI_need_output)
#betajk_HI[:,:,bad_v]=hp.UNSEEN


fig = plt.figure(figsize=(10, 7))
hp.mollview(betajk_HI[ich, j_test], cmap='viridis', title=f'HI, j={j_test}, freq={nu_ch[ich]}', hold=True)

###################################################################
map_recons_HI=np.zeros((len(nu_ch), npix))
for nu in range(len(nu_ch)):
	for j in range(betajk_HI.shape[1]):
		map_recons_HI[nu] += pippo.mylibpy_needlets_f2betajk_j_healpix_harmonic(betajk_HI[nu,j],b_values,j)
map_recons_HI[:,bad_v]=hp.UNSEEN		
#############################################################################

fig=plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, BEAM {beam_s}, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
fig.add_subplot(311)
hp.mollview(HI_noise_maps_freq[ich], min=0, max=1, title='Input HI+ noise', cmap='viridis', hold=True)
fig.add_subplot(312) 
hp.mollview(map_recons_HI[ich], min=0, max=1, title='Need recons HI + noise', cmap= 'viridis', hold=True)
fig.add_subplot(313) 
hp.mollview(100*(map_recons_HI[ich]/HI_noise_maps_freq[ich]-1), min=-0.2, max=0.2, title='% Need recons HI/HI -1', cmap= 'viridis', hold=True)

######################################################################################
lmax_cl= 2*nside

ell_cl = np.arange(lmax_cl+1)

cl_cosmo_HI_recons = np.zeros((len(nu_ch), lmax_cl+1))
cl_cosmo_HI = np.zeros((len(nu_ch), lmax_cl+1))

for n in range(len(nu_ch)):
	cl_cosmo_HI_recons[n] = hp.anafast(map_recons_HI[n], lmax=lmax_cl)
	cl_cosmo_HI[n]=hp.anafast(HI_noise_maps_freq[n], lmax=lmax_cl)
	
diff = cl_cosmo_HI_recons/cl_cosmo_HI-1

fig = plt.figure()
plt.plot(ell_cl[2:], diff.mean(axis=0)[2:]*100)
plt.show()