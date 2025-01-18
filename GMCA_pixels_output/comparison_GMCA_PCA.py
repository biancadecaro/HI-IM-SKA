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

c_pal = sns.color_palette().as_hex()
###########################################################################
dir_GMCA = 'Maps_GMCA/No_mean/Beam_40arcmin/'
dir_PCA= '../PCA_pixels_output/Maps_PCA/No_mean/Beam_40arcmin/'

dir_GMCA_cl = dir_GMCA+'power_spectra_cls_from_healpix_maps/'
dir_PCA_cl = dir_PCA+'power_spectra_cls_from_healpix_maps/'

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

#####################################################################################################

cl_cosmo_HI = np.loadtxt('../PCA_pixels_output/Maps_PCA/No_mean/Beam_40arcmin/power_spectra_cls_from_healpix_maps/cl_input_HI_lmax512_nside256.dat')

cl_GMCA_HI = np.loadtxt(dir_GMCA_cl+f'cl_GMCA_HI_Nfg3_lmax512_nside256.dat')
cl_PCA_HI = np.loadtxt(dir_PCA_cl+f'cl_PCA_HI_Nfg3_lmax512_nside256.dat')

ich = int(num_freq/2)

################################# PLOT ############################################################

ell = np.arange(0, lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)

diff_PCA = cl_PCA_HI/cl_cosmo_HI -1
diff_GMCA = cl_GMCA_HI/cl_cosmo_HI-1


fig=plt.figure()
fig.suptitle(f'mean over channels, beam 40 arcmin, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.plot(ell[2:], 100*np.mean(diff_PCA, axis=0)[2:],'--',mfc='none', label = 'PCA')
plt.plot(ell[2:], 100*np.mean(diff_GMCA, axis=0)[2:],'--',mfc='none', label = 'GMCA')
plt.xlim([0,250])
plt.ylim([-20,20])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\% \langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle $')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(f'rel_diff_mean_cl_HI_PCA_GMCA_lmax{lmax_cl}_Nfg{Nfg}.png')

print('% rel diff PCA:',min(100*diff_PCA[ich]), max(100*diff_PCA[ich]))
print('% rel diff GMCA:',min(100*diff_GMCA[ich]), max(100*diff_GMCA[ich]))


##################### diff tra i due ###################################

fig=plt.figure()
fig.suptitle(f'mean over channels, beam 40 arcmin, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.scatter(ell[2:], 100*np.mean(diff_PCA/diff_GMCA-1, axis=0)[2:])
plt.xlim([0,250])
plt.ylim([-20,20])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\% \langle diff \rangle $')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.tight_layout()



plt.show()