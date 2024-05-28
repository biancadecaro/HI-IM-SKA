import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
from needlets_analysis import analysis
import os
import seaborn as sns
import scipy.linalg as lng
sns.set()
sns.set(style = 'white')
#sns.set_palette('husl',15)

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

fg_components='synch_ff_ps'
lmax=383

cl_PCA_HI_no_mean = np.loadtxt('Maps_PCA_no_mean/power_spectra_cls_from_healpix_maps/cl_PCA_HI_Nfg3_lmax383_nside128.dat')
cl_PCA_HI_mean = np.loadtxt('Maps_PCA/power_spectra_cls_from_healpix_maps/cl_PCA_HI_Nfg3_lmax383_nside128.dat')

cl_leak_fg_no_mean = np.loadtxt('Maps_PCA_no_mean/power_spectra_cls_from_healpix_maps/cl_leak_fg_Nfg3_lmax383_nside128.dat')
cl_leak_fg_mean = np.loadtxt('Maps_PCA/power_spectra_cls_from_healpix_maps/cl_leak_fg_Nfg3_lmax383_nside128.dat')

cl_leak_HI_no_mean = np.loadtxt('Maps_PCA_no_mean/power_spectra_cls_from_healpix_maps/cl_leak_HI_Nfg3_lmax383_nside128.dat')
cl_leak_HI_mean = np.loadtxt('Maps_PCA/power_spectra_cls_from_healpix_maps/cl_leak_HI_Nfg3_lmax383_nside128.dat')


cl_cosmo_HI_no_mean = np.loadtxt('Maps_PCA_no_mean/power_spectra_cls_from_healpix_maps/cl_input_HI_lmax383_nside128.dat')
cl_cosmo_HI_mean = np.loadtxt('Maps_PCA/power_spectra_cls_from_healpix_maps/cl_input_HI_lmax383_nside128.dat')

#caricare gli spettri invece delle mappe
ell = np.arange(0, lmax+1)
factor = ell*(ell+1)/(2*np.pi)

fig=plt.figure()
plt.semilogy(ell,factor*np.mean(cl_leak_fg_mean, axis=0),mfc='none', label='Fg leakage, mean')
plt.semilogy(ell,factor*np.mean(cl_leak_HI_mean, axis=0),mfc='none', label='HI leakage, mean')
plt.semilogy(ell,factor*np.mean(cl_leak_fg_no_mean, axis=0),mfc='none', label='Fg leakage, no mean')
plt.semilogy(ell,factor*np.mean(cl_leak_HI_no_mean, axis=0),mfc='none', label='HI leakage, no mean')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell*(\ell+1)}{2\pi} \langle C_{\ell} \rangle$')
#plt.xlim([0,200])
plt.legend()



fig=plt.figure()
plt.plot(ell[:], np.mean(cl_PCA_HI_no_mean/cl_cosmo_HI_no_mean-1, axis=0)[:],'--.',mfc='none',label=f'No mean')
plt.plot(ell[:], np.mean(cl_PCA_HI_mean/cl_cosmo_HI_mean-1, axis=0)[:],'--.',mfc='none',label=f'Mean')
#plt.plot(factor*np.mean(cl_fg_leak_Nfg, axis=0),'--',mfc='none', label='Foreground leakage')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle$')
#plt.xlim([0,200])
plt.ylim([-0.05,0.05])
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.legend()


fig=plt.figure()
plt.plot(ell[:], np.mean(cl_PCA_HI_no_mean/cl_PCA_HI_mean-1, axis=0)[:],mfc='none')
plt.xlabel(r'$\ell$')
plt.ylabel(r'% rel diff')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)


plt.show()