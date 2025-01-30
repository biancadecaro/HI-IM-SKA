import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import scipy.linalg as lng
sns.set_theme(style = 'white')
#sns.set_palette('husl',15)

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)
###########################################################################
beam_s = 'theta40arcmin'
dir_PCA_beam = f'maps_reconstructed/No_mean/Beam_{beam_s}_noise/'

dir_PCA_cl_beam = dir_PCA_beam+'cls_recons_need/'

fg_comp = 'synch_ff_ps'
path_sims = f'/home/bianca/Documents/HI IM SKA/Sims/beam_{beam_s}_no_mean_sims_{fg_comp}_noise_40freq_905.0_1295.0MHz_thick10MHz_lmax383_nside128'


with open(path_sims+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

nu_ch= file['freq']
del file
num_freq = len(nu_ch)

nu0 =1420
print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')
print(f'corresponding to the redshift range z: [{min(nu0/nu_ch -1.0):.2f} - {max(nu0/nu_ch -1.0):.2f}] ')
#########################################################################################################
nside =128
lmax=3*nside-1
lmax_cl=2*nside
Nfg=3
#####################################################################################################

cl_cosmo_HI_beam = np.loadtxt(f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise/power_spectra_cls_from_healpix_maps/cl_input_HI_noise_40_905.0_1295.0MHz_lmax256_nside128.dat')

cl_PCA_HI_beam_jmax4 = np.loadtxt(f'maps_reconstructed/No_mean/Beam_{beam_s}_noise/cls_recons_need/cl_PCA_HI_noise_{fg_comp}_{num_freq}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{Nfg}_jmax4_lmax{lmax_cl}_nside{nside}.dat')

cl_PCA_HI_beam_jmax12 = np.loadtxt(f'maps_reconstructed/No_mean/Beam_{beam_s}_noise/cls_recons_need/cl_PCA_HI_noise_{fg_comp}_{num_freq}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{Nfg}_jmax12_lmax{lmax_cl}_nside{nside}.dat')

cl_PCA_HI_beam_sph = np.loadtxt(f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise/power_spectra_cls_from_healpix_maps/cl_PCA_HI_noise_{fg_comp}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{Nfg}_lmax{lmax_cl}_nside{nside}.dat')


ich = int(num_freq/2)

#####################################################################################################################

ell = np.arange(0, lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)

diff_PCA_beam_jmax4 = cl_PCA_HI_beam_jmax4/cl_cosmo_HI_beam -1

diff_PCA_beam_jmax12 = cl_PCA_HI_beam_jmax12/cl_cosmo_HI_beam -1

diff_PCA_beam_sph = cl_PCA_HI_beam_sph/cl_cosmo_HI_beam -1

################## diff tra i due ################################

fig=plt.figure()
plt.suptitle('Rel diff , mean over channels')
plt.plot(ell[2:], 100*np.mean(diff_PCA_beam_jmax4, axis=0)[2:],'--.',mfc='none', label='jmax=4')
plt.plot(ell[2:], 100*np.mean(diff_PCA_beam_jmax12, axis=0)[2:],'--.',mfc='none', label='jmax=12')
plt.plot(ell[2:], 100*np.mean(diff_PCA_beam_sph, axis=0)[2:],'--.',c='k',mfc='none', label='=sph')

plt.xlim([0,200])
plt.ylim([-20,20])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%\langle C_{ell}^{4}/C_{ell}^{12} -1 \rangle$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.legend()
plt.tight_layout()


fig=plt.figure()
plt.suptitle(f'Rel diff, channel: {nu_ch[ich]} MHz')
plt.plot(ell[2:], 100*diff_PCA_beam_jmax4[ich][2:],'--.',mfc='none', label='jmax=4')
plt.plot(ell[2:], 100*diff_PCA_beam_jmax12[ich][2:],'--.',mfc='none', label='jmax=12')
plt.plot(ell[2:], 100*diff_PCA_beam_sph[ich][2:],'--.',c='k',mfc='none', label='=sph')

plt.xlim([0,200])
plt.ylim([-20,20])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%C_{ell}^{4}/C_{ell}^{12} -1 $')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.legend()
plt.tight_layout()


plt.show()

#del diff_PCA_beam; del cl_PCA_HI
