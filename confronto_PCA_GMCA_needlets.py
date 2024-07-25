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
dir_PCA = 'PCA_needlets_output/maps_reconstructed/No_mean/Beam/'
dir_GMCA = 'GMCA_needlets_output/maps_reconstructed/No_mean/Beam/'

dir_PCA_cl = dir_PCA+'cls_recons_need/'
dir_GMCA_cl = dir_GMCA+'cls_recons_need/'

fg_components='synch_ff_ps'
path_data_sims_tot = f'Sims/beam_no_mean_sims_{fg_components}_40freq_905.0_1295.0MHz_nside256'

with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

nu_ch= file['freq']

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
#####################################################################################################

cl_cosmo_HI = np.loadtxt('PCA_pixels_output/Maps_PCA/No_mean/Beam/power_spectra_cls_from_healpix_maps/cl_input_HI_lmax512_nside256.dat')

cl_PCA_HI = np.loadtxt(dir_PCA_cl+'cl_PCA_HI_Nfg3_lmax512_nside256.dat')
cl_GMCA_HI = np.loadtxt(dir_GMCA_cl+'cl_GMCA_HI_Nfg3_lmax512_nside256.dat')

ich = int(num_freq/2)

################################# PLOT ############################################################

cosmo_HI = np.load('PCA_pixels_output/Maps_PCA/No_mean/Beam/cosmo_HI_40_905.0_1295.0MHz.npy')
res_PCA = np.load(dir_PCA +'maps_reconstructed_PCA_HI_40_jmax12_lmax768_Nfg3_nside256.npy')
res_GMCA = np.load(dir_GMCA +'maps_reconstructed_GMCA_HI_40_jmax12_lmax768_Nfg3_nside256.npy')

delta_PCA = cosmo_HI[ich]-res_PCA[ich]
delta_GMCA = cosmo_HI[ich]-res_GMCA[ich]
del cosmo_HI; del res_PCA; del res_GMCA

fig=plt.figure(figsize=(10, 7))
fig.suptitle(f'Need recons, channel: {nu_ch[ich]} MHz, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
fig.add_subplot(131) 
hp.gnomview(delta_PCA, rot=[-50, 120],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-0.2, max=0.2, title=r'$\delta$HI PCA', cmap= 'viridis', hold=True)
fig.add_subplot(132) 
hp.gnomview(delta_GMCA, rot=[-50, 120],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-0.2, max=0.2, title=r'$\delta$HI GMCA', cmap= 'viridis', hold=True)
fig.add_subplot(133) 
hp.gnomview(delta_PCA-delta_GMCA, rot=[-50, 120],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-2e-3, max=2e-3, title=r'$\delta$HI GMCA - $\delta$HI PCA', cmap= 'viridis', hold=True)
plt.tight_layout()
plt.show()

del delta_GMCA; del delta_PCA

#####################################################################################################################

ell = np.arange(0, lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)

diff_PCA = cl_PCA_HI/cl_cosmo_HI-1
diff_GMCA = cl_GMCA_HI/cl_cosmo_HI-1


fig=plt.figure()
fig.suptitle(f'channel: {nu_ch[ich]} MHz, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.plot(ell[1:], 100*diff_PCA[ich][1:],'--.',mfc='none', label = 'PCA')
plt.plot(ell[1:], 100*diff_GMCA[ich][1:],'--.',mfc='none', label = 'GMCA')
plt.xlim([0,200])
plt.ylim([-10,10])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\% C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 $')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
print('% rel diff PCA:',min(100*diff_PCA[ich]), max(100*diff_PCA[ich]))
print('% rel diff GMCA:',min(100*diff_GMCA[ich]), max(100*diff_GMCA[ich]))


fig=plt.figure()
fig.suptitle(f'mean over channels, lmax:{lmax}, Nfg:{Nfg}')
plt.plot(ell[1:], 100*np.mean(diff_PCA, axis=0)[1:],'--.',mfc='none', label='PCA')
plt.plot(ell[1:], 100*np.mean(diff_GMCA, axis=0)[1:],'--.',mfc='none', label='GMCA')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%\langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.xlim([0,200])
plt.ylim([-10,10])
plt.legend()
plt.tight_layout()
plt.show()
print('% rel diff mean PCA:',min(100*np.mean(diff_PCA, axis=0)), max(100*np.mean(diff_PCA, axis=0)))
print('% rel diff mean GMCA:',min(100*np.mean(diff_GMCA, axis=0)), max(100*np.mean(diff_GMCA, axis=0)))

################## diff tra i due ################################

fig=plt.figure()
plt.plot(ell[1:], 100*np.mean(diff_PCA/diff_GMCA-1, axis=0)[1:],'--.',mfc='none')
plt.xlim([0,200])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%\langle diff \rangle$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)

plt.tight_layout()
plt.show()

del cl_GMCA_HI; del cl_PCA_HI


#######################################################################################################
############################## PROVIAMO A METTERE INSIEME I CL #######################################

dir_PCA_sph = 'PCA_pixels_output/Maps_PCA/No_mean/Beam/'
dir_GMCA_sph= 'GMCA_pixels_output/Maps_GMCA/No_mean/Beam/'

dir_PCA_cl_sph = dir_PCA_sph+'power_spectra_cls_from_healpix_maps/'
dir_GMCA_cl_sph = dir_GMCA_sph+'power_spectra_cls_from_healpix_maps/'


cl_PCA_HI_sph = np.loadtxt(dir_PCA_cl_sph+'cl_PCA_HI_Nfg3_lmax512_nside256.dat')
cl_GMCA_HI_sph = np.loadtxt(dir_GMCA_cl_sph+'cl_GMCA_HI_Nfg3_lmax512_nside256.dat')

diff_PCA_sph = cl_PCA_HI_sph/cl_cosmo_HI-1
diff_GMCA_sph = cl_GMCA_HI_sph/cl_cosmo_HI-1


fig=plt.figure()
fig.suptitle(f'channel: {nu_ch[ich]} MHz, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.plot(ell[1:], 100*diff_PCA[ich][1:],'--.',mfc='none', label = 'PCA Need')
plt.plot(ell[1:], 100*diff_GMCA[ich][1:],'--.',mfc='none', label = 'GMCA Need')

plt.plot(ell[1:], 100*diff_PCA_sph[ich][1:],'--.',mfc='none', label = 'PCA Sph')
plt.plot(ell[1:], 100*diff_GMCA_sph[ich][1:],'--.',mfc='none', label = 'GMCA Sph')
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
plt.plot(ell[1:], 100*np.abs(np.mean(diff_PCA, axis=0))[1:],'--',mfc='none', label='PCA Need')
plt.plot(ell[1:], 100*np.abs(np.mean(diff_GMCA, axis=0))[1:],'--',mfc='none', label='GMCA Need')

plt.plot(ell[1:], 100*np.abs(np.mean(diff_PCA_sph, axis=0))[1:],'--',mfc='none', label='PCA Sph')
plt.plot(ell[1:], 100*np.abs(np.mean(diff_GMCA_sph, axis=0))[1:],'--',mfc='none', label='GMCA Sph')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%|\langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle|$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.xlim([0,200])
plt.ylim([-10,10])
plt.legend()
plt.tight_layout()
plt.show()