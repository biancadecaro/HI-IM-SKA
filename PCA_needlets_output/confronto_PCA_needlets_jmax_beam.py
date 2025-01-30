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
dir_PCA_beam = 'maps_reconstructed/No_mean/Beam_40arcmin/'
dir_PCA = 'maps_reconstructed/No_mean/'

dir_PCA_cl_beam = dir_PCA_beam+'cls_recons_need/'
dir_PCA_cl = dir_PCA+'cls_recons_need/'

fg_components='synch_ff_ps'
path_data_sims_tot = f'../Sims/beam_no_mean_sims_{fg_components}_40freq_905.0_1295.0MHz_lmax768_nside256'

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
#####################################################################################################

cl_cosmo_HI_beam = np.loadtxt('../PCA_pixels_output/Maps_PCA/No_mean/Beam_40arcmin/power_spectra_cls_from_healpix_maps/cl_input_HI_lmax512_nside256.dat')
cl_cosmo_HI = np.loadtxt('../PCA_pixels_output/Maps_PCA/No_mean/power_spectra_cls_from_healpix_maps/cl_input_HI_lmax512_nside256.dat')

cl_PCA_HI_beam_jmax4 = np.loadtxt(dir_PCA_cl_beam+'cl_PCA_HI_Nfg3_jmax4_lmax512_nside256.dat')
cl_PCA_HI_jmax4 = np.loadtxt(dir_PCA_cl+'cl_PCA_HI_Nfg3_jmax4_lmax512_nside256.dat')

cl_PCA_HI_beam_jmax12 = np.loadtxt(dir_PCA_cl_beam+'cl_PCA_HI_Nfg3_jmax12_lmax512_nside256.dat')
cl_PCA_HI_jmax12 = np.loadtxt(dir_PCA_cl+'cl_PCA_HI_Nfg3_jmax12_lmax512_nside256.dat')


ich = int(num_freq/2)

################################# PLOT ############################################################

#cosmo_HI_beam = np.load('../PCA_pixels_output/Maps_PCA/No_mean/Beam/cosmo_HI_40_905.0_1295.0MHz.npy')
#cosmo_HI = np.load('../PCA_pixels_output/Maps_PCA/No_mean/cosmo_HI_40_905.0_1295.0MHz_768.npy')
#res_PCA_beam = np.load(dir_PCA_beam +'maps_reconstructed_PCA_HI_40_jmax12_lmax768_Nfg3_nside256.npy')
#res_PCA = np.load(dir_PCA +'maps_reconstructed_PCA_HI_40_jmax12_lmax768_Nfg3_nside256.npy')
#
#delta_PCA_beam = cosmo_HI_beam[ich]-res_PCA_beam[ich]
#delta_PCA = cosmo_HI[ich]-res_PCA[ich]
#del cosmo_HI; del res_PCA; del res_PCA_beam
#
#fig=plt.figure(figsize=(10, 7))
#fig.suptitle(f'Need recons, channel: {nu_ch[ich]} MHz, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
#fig.add_subplot(131) 
#hp.gnomview(delta_PCA_beam, rot=[-50, 120],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-0.2, max=0.2, title=r'$\delta$HI PCA Beam', cmap= 'viridis', hold=True)
#fig.add_subplot(132) 
#hp.gnomview(delta_PCA, rot=[-50, 120],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-0.2, max=0.2, title=r'$\delta$HI PCA No Beam', cmap= 'viridis', hold=True)
#fig.add_subplot(133) 
#hp.gnomview(delta_PCA_beam-delta_PCA, rot=[-50, 120],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-2e-3, max=2e-3, title=r'$\delta$HI Beam - $\delta$HI No Beam', cmap= 'viridis', hold=True)
#plt.tight_layout()
#plt.show()
#
#del delta_PCA_beam; del delta_PCA

#####################################################################################################################

ell = np.arange(0, lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)

diff_PCA_beam_jmax4 = cl_PCA_HI_beam_jmax4/cl_cosmo_HI_beam -1
diff_PCA_jmax4 = cl_PCA_HI_jmax4/cl_cosmo_HI-1

diff_PCA_beam_jmax12 = cl_PCA_HI_beam_jmax12/cl_cosmo_HI_beam -1
diff_PCA_jmax12 = cl_PCA_HI_jmax12/cl_cosmo_HI-1

fig=plt.figure()
fig.suptitle(f'mean over channels, jmax:4, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.plot(ell[2:], 100*np.mean(diff_PCA_beam_jmax4, axis=0)[2:],'--.',mfc='none', label = 'PCA Beam')
plt.plot(ell[2:], 100*np.mean(diff_PCA_jmax4, axis=0)[2:],'--.',mfc='none', label = 'PCA No Beam')
plt.xlim([0,200])
plt.ylim([-10,10])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\% \langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle $')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.legend()
plt.tight_layout()

print('% rel diff PCA beam:',min(100*diff_PCA_beam_jmax4[ich]), max(100*diff_PCA_beam_jmax4[ich]))
print('% rel diff PCA:',min(100*diff_PCA_jmax4[ich]), max(100*diff_PCA_jmax4[ich]))


fig=plt.figure()
fig.suptitle(f'mean over channels, jmax:12, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.plot(ell[2:], 100*np.mean(diff_PCA_beam_jmax12, axis=0)[2:],'--.',mfc='none', label='PCA Beam')
plt.plot(ell[2:], 100*np.mean(diff_PCA_jmax12, axis=0)[2:],'--.',mfc='none', label='PCA No Beam')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%\langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.xlim([0,200])
plt.ylim([-10,10])
plt.legend()
plt.tight_layout()
plt.show()
print('% rel diff mean PCA beam:',min(100*np.mean(diff_PCA_beam_jmax12, axis=0)), max(100*np.mean(diff_PCA_beam_jmax12, axis=0)))
print('% rel diff mean PCA:',min(100*np.mean(diff_PCA_jmax12, axis=0)), max(100*np.mean(diff_PCA_jmax12, axis=0)))

################## diff tra i due ################################

fig=plt.figure()
plt.suptitle('Diff beam no beam, mean over channels')
plt.plot(ell[2:], 100*np.mean(diff_PCA_beam_jmax4/diff_PCA_jmax4-1, axis=0)[2:],'--.',mfc='none', label='jmax=4')
plt.plot(ell[2:], 100*np.mean(diff_PCA_beam_jmax12/diff_PCA_jmax12-1, axis=0)[2:],'--.',mfc='none', label='jmax=12')
plt.xlim([0,200])
plt.ylim([-20,20])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%\langle diff \rangle$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

#del diff_PCA_beam; del cl_PCA_HI


#######################################################################################################
############################## PROVIAMO A METTERE INSIEME I CL #######################################

dir_PCA_beam_sph = '../PCA_pixels_output/Maps_PCA/No_mean/Beam_40arcmin/'
dir_PCA_sph= '../PCA_pixels_output/Maps_PCA/No_mean/'

dir_PCA_cl_beam_sph = dir_PCA_beam_sph+'power_spectra_cls_from_healpix_maps/'
dir_PCA_cl_sph = dir_PCA_sph+'power_spectra_cls_from_healpix_maps/'


cl_PCA_HI_beam_sph = np.loadtxt(dir_PCA_cl_beam_sph+'cl_PCA_HI_Nfg3_lmax512_nside256.dat')
cl_PCA_HI_sph = np.loadtxt(dir_PCA_cl_sph+'cl_PCA_HI_Nfg3_lmax512_nside256.dat')

diff_PCA_beam_sph = cl_PCA_HI_beam_sph/cl_cosmo_HI_beam-1
diff_PCA_sph = cl_PCA_HI_sph/cl_cosmo_HI-1


#fig=plt.figure()
#fig.suptitle(f'channel: {nu_ch[ich]} MHz, jmax:4, lmax:{lmax_cl}, Nfg:{Nfg}')
#plt.plot(ell[2:], 100*diff_PCA_beam_jmax4[ich][2:],'--.',mfc='none', label = 'PCA Beam Need')
#plt.plot(ell[2:], 100*diff_PCA_jmax4[ich][2:],'--.',mfc='none', label = 'PCA No Beam Need')
#
#plt.plot(ell[2:], 100*diff_PCA_beam_sph[ich][2:],'--.',mfc='none', label = 'PCA Beam Sph')
#plt.plot(ell[2:], 100*diff_PCA_sph[ich][2:],'--.',mfc='none', label = 'PCA No Beam Sph')
#plt.xlim([0,200])
#plt.ylim([-10,10])
#plt.xlabel(r'$\ell$')
#plt.ylabel(r'$\% C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 $')
#plt.axhline(y=0,c='k',ls='--',alpha=0.5)
#plt.legend()
#plt.tight_layout()
#plt.show()



fig=plt.figure()
fig.suptitle(f'mean over channels, jmax:4, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.plot(ell[2:], 100*(np.mean(diff_PCA_beam_jmax4, axis=0))[2:],'--',mfc='none', label='PCA Beam Need')
plt.plot(ell[2:], 100*(np.mean(diff_PCA_jmax4, axis=0))[2:],'--',mfc='none', label='PCA No Beam Need')

plt.plot(ell[2:], 100*(np.mean(diff_PCA_beam_sph, axis=0))[2:],'--',mfc='none', label='PCA Beam Sph')
plt.plot(ell[2:], 100*(np.mean(diff_PCA_sph, axis=0))[2:],'--',mfc='none', label='PCA No Beam Sph')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$\% \langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle $')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.xlim([0,256])
plt.ylim([-20,20])
plt.legend()
plt.tight_layout()

fig=plt.figure()
fig.suptitle(f'mean over channels, jmax:12, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.plot(ell[2:], 100*(np.mean(diff_PCA_beam_jmax12, axis=0))[2:],'--',mfc='none', label='PCA Beam Need')
plt.plot(ell[2:], 100*(np.mean(diff_PCA_jmax12, axis=0))[2:],'--',mfc='none', label='PCA No Beam Need')

plt.plot(ell[2:], 100*(np.mean(diff_PCA_beam_sph, axis=0))[2:],'--',mfc='none', label='PCA Beam Sph')
plt.plot(ell[2:], 100*(np.mean(diff_PCA_sph, axis=0))[2:],'--',mfc='none', label='PCA No Beam Sph')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%\langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.xlim([0,256])
plt.ylim([-20,20])
plt.legend()
plt.tight_layout()
plt.show()


fig=plt.figure()
fig.suptitle(f'mean over channels, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.plot(ell[2:], 100*(np.mean(diff_PCA_beam_jmax4, axis=0))[2:],'--',mfc='none', label='PCA Beam Need jmax=4')
plt.plot(ell[2:], 100*(np.mean(diff_PCA_jmax4, axis=0))[2:],'--',mfc='none', label='PCA No Beam Need jmax=4')

plt.plot(ell[2:], 100*(np.mean(diff_PCA_beam_jmax12, axis=0))[2:],'--',mfc='none', label='PCA Beam Need jmax=12')
plt.plot(ell[2:], 100*(np.mean(diff_PCA_jmax12, axis=0))[2:],'--',mfc='none', label='PCA No Beam Need jmax=12')


plt.plot(ell[2:], 100*(np.mean(diff_PCA_beam_sph, axis=0))[2:],'k--',mfc='none', label='PCA Beam Sph')
#plt.plot(ell[2:], 100*np.abs(np.mean(diff_PCA_sph, axis=0))[2:],'--',mfc='none', label='PCA No Beam Sph')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%\langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.xlim([0,256])
plt.ylim([-20,20])
plt.legend()
plt.tight_layout()
plt.savefig(f'beam_40arcmin_no_beam_diff_cls_need_sph_lmax{lmax}_nside{nside}.png')
plt.show()

fig=plt.figure()
fig.suptitle(f'mean over channels, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.plot(ell[2:], 100*(np.mean(diff_PCA_beam_jmax4, axis=0))[2:],'--',mfc='none', label='PCA Beam Need jmax = 4')
plt.plot(ell[2:], 100*(np.mean(diff_PCA_jmax4, axis=0))[2:],'--',mfc='none', label='PCA No Beam Need jmax = 4')

plt.plot(ell[2:], 100*(np.mean(diff_PCA_beam_jmax12, axis=0))[2:],'--',mfc='none', label='PCA Beam Need jmax = 12')
plt.plot(ell[2:], 100*(np.mean(diff_PCA_jmax12, axis=0))[2:],'--',mfc='none', label='PCA No Beam Need jmax = 12')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%\langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.xlim([0,256])
plt.ylim([-20,20])
plt.legend()
plt.tight_layout()
plt.show()


fig=plt.figure()
fig.suptitle(f'mean over channels, lmax:{lmax_cl}, Nfg:{Nfg}')
#plt.plot(ell[2:], 100*np.abs(np.mean(diff_PCA_beam_jmax4, axis=0))[2:],'--',mfc='none', label='PCA Beam 3 deg Need jmax=4')
#plt.plot(ell[2:], 100*np.abs(np.mean(diff_PCA_jmax4, axis=0))[2:],'--',mfc='none', label='PCA Beam 1.41 deg Need jmax=4')
#
#plt.plot(ell[2:], 100*np.abs(np.mean(diff_PCA_beam_jmax12, axis=0))[2:],'--',mfc='none', label='PCA Beam 3 deg Need jmax=12')
#plt.plot(ell[2:], 100*np.abs(np.mean(diff_PCA_jmax12, axis=0))[2:],'--',mfc='none', label='PCA Beam 1.41 deg Need jmax=12')

plt.plot(ell[2:], 100*(np.mean(diff_PCA_beam_sph, axis=0))[2:],'--',mfc='none', label='PCA Beam Sph')
plt.plot(ell[2:], 100*(np.mean(diff_PCA_sph, axis=0))[2:],'--',mfc='none', label='PCA No Beam Sph')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%\langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.xlim([0,256])
plt.ylim([-20,20])
plt.legend()
plt.tight_layout()
plt.savefig(f'beam_40arcmin_no_beam_diff_cls_sph_lmax{lmax}_nside{nside}.png')
plt.show()

fig=plt.figure()
fig.suptitle(f'mean over channels, lmax:{lmax_cl}, Nfg:{Nfg}')

plt.plot(ell[2:], factor[2:]*np.mean(cl_PCA_HI_beam_jmax4, axis=0)[2:],'--',mfc='none', label='PCA Beam Need jmax=4')
plt.plot(ell[2:], factor[2:]*np.mean(cl_PCA_HI_jmax4, axis=0)[2:],'--',mfc='none', label='PCA No Beam Need jmax=4')

plt.plot(ell[2:], factor[2:]*np.mean(cl_PCA_HI_beam_jmax12, axis=0)[2:],'--',mfc='none', label='PCA Beam Need jmax=12')
plt.plot(ell[2:], factor[2:]*np.mean(cl_PCA_HI_jmax12, axis=0)[2:],'--',mfc='none', label='PCA No Beam Need jmax=12')


plt.plot(ell[2:], factor[2:]*np.mean(cl_PCA_HI_beam_sph,axis=0)[2:],'--',mfc='none', label='PCA Beam Sph')
plt.plot(ell[2:], factor[2:]*np.mean(cl_PCA_HI_sph,axis=0)[2:],'--',mfc='none', label='PCA No Beam Sph')


plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)/(2\pi)\langle C_{\ell}^{\rm PCA} \rangle|$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.xlim([0,256])
#plt.ylim([-5,20])
plt.legend()
plt.tight_layout()
plt.savefig(f'beam_40arcmin_no_beam_cls_need_sph_lmax{lmax_cl}_nside{nside}.png')
plt.show()

fig=plt.figure()
fig.suptitle(f'BEAM theta 40 arcmin, mean over channels, lmax:{lmax_cl}, Nfg:{Nfg}')

plt.plot(ell[2:], factor[2:]*np.mean(cl_PCA_HI_beam_jmax4, axis=0)[2:],'--',mfc='none', label='PCA Beam Need jmax=4')
plt.plot(ell[2:], factor[2:]*np.mean(cl_PCA_HI_beam_jmax12, axis=0)[2:],'--',mfc='none', label='PCA Beam Need jmax=12')
plt.plot(ell[2:], factor[2:]*np.mean(cl_PCA_HI_beam_sph,axis=0)[2:],'--',mfc='none', label='PCA Beam Sph')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)/(2\pi)\langle C_{\ell}^{\rm PCA} \rangle|$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.xlim([0,256])
#plt.ylim([0,0.0030])
plt.legend()
plt.tight_layout()
plt.savefig(f'beam_40arcmin_cls_need_sph_lmax{lmax_cl}_nside{nside}.png')
plt.show()