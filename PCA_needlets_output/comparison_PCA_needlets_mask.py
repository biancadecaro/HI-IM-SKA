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
dir_PCA_mask = 'maps_reconstructed/No_mean/Beam_40arcmin_mask/'
dir_PCA = 'maps_reconstructed/No_mean/Beam_40arcmin/'

dir_PCA_cl_mask = dir_PCA_mask+'cls_recons_need/'
dir_PCA_cl = dir_PCA+'cls_recons_need/'

fg_components='synch_ff_ps'
path_data_sims_tot = f'../Sims/beam_theta40arcmin_no_mean_sims_{fg_components}_40freq_905.0_1295.0MHz_lmax768_nside256'
with open(path_data_sims_tot+'.pkl', 'rb') as f:
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
nside =256
lmax=3*nside
lmax_cl=2*nside
Nfg=3
jmax=12
#####################################################################################################

cl_cosmo_HI_mask = np.loadtxt('../PCA_pixels_output/Maps_PCA/No_mean/Beam_40arcmin_mask/power_spectra_cls_from_healpix_maps/cl_input_HI_lmax512_nside256.dat')
cl_cosmo_HI = np.loadtxt('../PCA_pixels_output/Maps_PCA/No_mean/Beam_40arcmin/power_spectra_cls_from_healpix_maps/cl_input_HI_lmax512_nside256.dat')

cl_PCA_HI_mask = np.loadtxt(dir_PCA_cl_mask+f'cl_PCA_HI_Nfg{Nfg}_jmax{jmax}_lmax{lmax_cl}_nside{nside}.dat')
cl_PCA_HI = np.loadtxt(dir_PCA_cl+f'cl_PCA_HI_Nfg{Nfg}_jmax{jmax}_lmax{lmax_cl}_nside{nside}.dat')

ich = int(num_freq/2)

################################# PLOT ############################################################

cosmo_HI_mask = np.load(f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_40arcmin_mask/cosmo_HI_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_{lmax}.npy')
cosmo_HI = np.load(f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_40arcmin/cosmo_HI_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_{lmax}.npy')
res_PCA_mask = np.load(dir_PCA_mask +'maps_reconstructed_PCA_HI_40_jmax12_lmax768_Nfg3_nside256.npy')
res_PCA = np.load(dir_PCA +'maps_reconstructed_PCA_HI_40_jmax12_lmax768_Nfg3_nside256.npy')

delta_PCA_mask = cosmo_HI_mask[ich]-res_PCA_mask[ich]
delta_PCA = cosmo_HI[ich]-res_PCA[ich]
del cosmo_HI; del res_PCA; del res_PCA_mask

fig=plt.figure(figsize=(10, 7))
fig.suptitle(f'Need recons, channel: {nu_ch[ich]} MHz, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
fig.add_subplot(131) 
hp.gnomview(delta_PCA_mask, rot=[-50, 120],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-0.2, max=0.2, title=r'$\delta$HI PCA Mask', cmap= 'viridis', hold=True)
fig.add_subplot(132) 
hp.gnomview(delta_PCA, rot=[-50, 120],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-0.2, max=0.2, title=r'$\delta$HI PCA Full sky', cmap= 'viridis', hold=True)
fig.add_subplot(133) 
hp.gnomview(delta_PCA_mask-delta_PCA, rot=[-50, 120],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-2e-3, max=2e-3, title=r'$\delta$HI Mask - $\delta$HI Full sky', cmap= 'viridis', hold=True)
plt.tight_layout()
plt.show()

del delta_PCA_mask; del delta_PCA

#####################################################################################################################

ell = np.arange(0, lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)

diff_PCA_mask = cl_PCA_HI_mask/cl_cosmo_HI_mask -1
diff_PCA = cl_PCA_HI/cl_cosmo_HI-1


fig=plt.figure()
fig.suptitle(f'channel: {nu_ch[ich]} MHz, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.plot(ell[1:], 100*diff_PCA_mask[ich][1:],'--.',mfc='none', label = 'PCA Mask')
plt.plot(ell[1:], 100*diff_PCA[ich][1:],'--.',mfc='none', label = 'PCA No Mask')
plt.xlim([0,200])
plt.ylim([-10,10])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\% C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 $')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
print('% rel diff PCA mask:',min(100*diff_PCA_mask[ich]), max(100*diff_PCA_mask[ich]))
print('% rel diff PCA:',min(100*diff_PCA[ich]), max(100*diff_PCA[ich]))


fig=plt.figure()
fig.suptitle(f'mean over channels, lmax:{lmax}, Nfg:{Nfg}')
plt.plot(ell[1:], 100*np.mean(diff_PCA_mask, axis=0)[1:],'--.',mfc='none', label='PCA Mask')
plt.plot(ell[1:], 100*np.mean(diff_PCA, axis=0)[1:],'--.',mfc='none', label='PCA No Mask')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%\langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.xlim([0,200])
plt.ylim([-10,10])
plt.legend()
plt.tight_layout()
plt.show()
print('% rel diff mean PCA mask:',min(100*np.mean(diff_PCA_mask, axis=0)), max(100*np.mean(diff_PCA_mask, axis=0)))
print('% rel diff mean PCA:',min(100*np.mean(diff_PCA, axis=0)), max(100*np.mean(diff_PCA, axis=0)))

################## diff tra i due ################################

fig=plt.figure()
plt.plot(ell[1:], 100*np.mean(diff_PCA_mask/diff_PCA-1, axis=0)[1:],'--.',mfc='none')
plt.xlim([0,200])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%\langle diff \rangle$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)

plt.tight_layout()
plt.show()

#del diff_PCA_mask; del cl_PCA_HI


#######################################################################################################
############################## PROVIAMO A METTERE INSIEME I CL #######################################

dir_PCA_mask_sph = '../PCA_pixels_output/Maps_PCA/No_mean/Beam_40arcmin_mask/'
dir_PCA_sph= '../PCA_pixels_output/Maps_PCA/No_mean/Beam_40arcmin/'

dir_PCA_cl_mask_sph = dir_PCA_mask_sph+'power_spectra_cls_from_healpix_maps/'
dir_PCA_cl_sph = dir_PCA_sph+'power_spectra_cls_from_healpix_maps/'


cl_PCA_HI_mask_sph = np.loadtxt(dir_PCA_cl_mask_sph+f'cl_PCA_HI_{fg_components}_Nfg3_lmax{lmax_cl}_nside{nside}.dat')
cl_PCA_HI_sph = np.loadtxt(dir_PCA_cl_sph+f'cl_PCA_HI_{fg_components}_Nfg3_lmax{lmax_cl}_nside{nside}.dat')

diff_PCA_mask_sph = cl_PCA_HI_mask_sph/cl_cosmo_HI_mask-1
diff_PCA_sph = cl_PCA_HI_sph/cl_cosmo_HI-1


fig=plt.figure()
fig.suptitle(f'channel: {nu_ch[ich]} MHz, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.plot(ell[1:], 100*diff_PCA_mask[ich][1:],'--.',mfc='none', label = 'PCA Mask Need')
plt.plot(ell[1:], 100*diff_PCA[ich][1:],'--.',mfc='none', label = 'PCA No Mask Need')

plt.plot(ell[1:], 100*diff_PCA_mask_sph[ich][1:],'--.',mfc='none', label = 'PCA Mask Sph')
plt.plot(ell[1:], 100*diff_PCA_sph[ich][1:],'--.',mfc='none', label = 'PCA No Mask Sph')
plt.xlim([0,200])
plt.ylim([-10,10])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\% C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 $')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()



fig=plt.figure()
fig.suptitle(f'mean over channels, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.plot(ell[1:], 100*np.abs(np.mean(diff_PCA_mask, axis=0))[1:],'--',mfc='none', label='PCA Mask Need')
plt.plot(ell[1:], 100*np.abs(np.mean(diff_PCA, axis=0))[1:],'--',mfc='none', label='PCA No Mask Need')

plt.plot(ell[1:], 100*np.abs(np.mean(diff_PCA_mask_sph, axis=0))[1:],'--',mfc='none', label='PCA Mask Sph')
plt.plot(ell[1:], 100*np.abs(np.mean(diff_PCA_sph, axis=0))[1:],'--',mfc='none', label='PCA No Mask Sph')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%|\langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle|$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.xlim([0,256])
plt.ylim([-5,20])
plt.legend()
plt.tight_layout()
plt.show()