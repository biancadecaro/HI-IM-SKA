import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as lng
sns.set_theme(style = 'white')
#sns.set_palette('husl',15)

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)
###########################################################################
dir_PCA = 'PCA_pixels_output/Maps_PCA/No_mean/Beam/'
dir_GMCA = 'GMCA_pixels_output/Maps_GMCA/No_mean/Beam/'

dir_PCA_cl = dir_PCA+'power_spectra_cls_from_healpix_maps/'
dir_GMCA_cl = dir_GMCA+'power_spectra_cls_from_healpix_maps/'

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
lmax_cl = 2*nside
Nfg=3
######################################################################################################

cl_cosmo_HI = np.loadtxt(dir_PCA_cl+f'cl_input_HI_lmax{lmax_cl}_nside256.dat')

cl_PCA_HI = np.loadtxt(dir_PCA_cl+f'cl_PCA_HI_Nfg3_lmax{lmax_cl}_nside256.dat')
cl_GMCA_HI = np.loadtxt(dir_GMCA_cl+f'cl_GMCA_HI_Nfg3_lmax{lmax_cl}_nside256.dat')

################################# PLOT ####################################################

ell = np.arange(0, lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)

ich = int(num_freq/2)


diff_PCA = cl_PCA_HI/cl_cosmo_HI-1
diff_GMCA = cl_GMCA_HI/cl_cosmo_HI-1


fig=plt.figure()
plt.plot(ell[1:], 100*diff_PCA[ich][1:],'--.',mfc='none', label = 'PCA')
plt.plot(ell[1:], 100*diff_GMCA[ich][1:],'--.',mfc='none', label = 'GMCA')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$\% C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 $')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.xlim([0,200])
plt.ylim([-10,10])
plt.legend()
plt.tight_layout()
plt.show()
print('% rel diff PCA:',min(100*diff_PCA[ich][0:lmax_cl]), max(100*diff_PCA[ich][0:lmax_cl]))
print('% rel diff GMCA:',min(100*diff_GMCA[ich]), max(100*diff_GMCA[ich][0:lmax_cl]))


fig=plt.figure()
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
print('% rel diff mean PCA:',min(100*np.mean(diff_PCA, axis=0)[0:lmax_cl]), max(100*np.mean(diff_PCA, axis=0)[0:lmax_cl]))
print('% rel diff mean GMCA:',min(100*np.mean(diff_GMCA, axis=0)[0:lmax_cl]), max(100*np.mean(diff_GMCA, axis=0)[0:lmax_cl]))

################## diff tra i due ################################

fig=plt.figure()
plt.plot(ell[1:], 100*np.mean(diff_PCA/diff_GMCA-1, axis=0)[1:],'--.',mfc='none')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%\langle diff \rangle$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.tight_layout()
plt.show()

del diff_PCA; del diff_GMCA; del cl_cosmo_HI; del cl_GMCA_HI; del cl_PCA_HI
#######################################################
cosmo_HI = np.load(dir_PCA+'cosmo_HI_40_905.0_1295.0MHz.npy')
res_PCA = np.load(dir_PCA +'res_PCA_HI_synch_ff_ps_40_905.0_1295.0MHz_Nfg3.npy')
res_GMCA = np.load(dir_GMCA +'res_GMCA_HI_synch_ff_ps_40_905.0_1295.0MHz_Nfg3.npy')

delta_PCA = cosmo_HI[ich]-res_PCA[ich]
delta_GMCA = cosmo_HI[ich]-res_GMCA[ich]

fig=plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
fig.add_subplot(131) 
hp.gnomview(delta_PCA, rot=[-50, 120],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-0.2, max=0.2, title=r'$\delta$HI PCA', cmap= 'viridis', hold=True)
fig.add_subplot(132) 
hp.gnomview(delta_GMCA, rot=[-50, 120],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-0.2, max=0.2, title=r'$\delta$HI GMCA', cmap= 'viridis', hold=True)
fig.add_subplot(133) 
hp.gnomview(delta_PCA-delta_GMCA, rot=[-50, 120],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-2e-3, max=2e-3, title=r'$\delta$HI GMCA - $\delta$HI PCA', cmap= 'viridis', hold=True)
plt.tight_layout()
plt.show()