import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
sns.set(style = 'white')
#sns.set_palette('husl',15)

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

#print(sns.color_palette("husl").as_hex())
sns.palettes.color_palette()

from needlets_analysis import analysis
import cython_mylibc as pippo


nu_ch = np.linspace(901.0, 1299.0, 200)
num_freq = len(nu_ch)

ich=100
HI_cosmo = np.load('cosmo_HI_200_901.0_1299.0MHz.npy')

Nfg=3
nside=128
npix=hp.nside2npix(nside)
lmax=3*nside

#################################################################################
'''Devo estrarre i cl e creare una mappa con synfast con un seed'''

seed = 47234
cls_syn = hp.anafast(HI_cosmo[ich], lmax=lmax)#/hp.pixwin(nside)**2
#########################################################################
np.random.seed(seed)
map_syn = hp.synfast(cls_syn, nside=nside,lmax=lmax )

cls_syn_2 = hp.anafast(map_syn, lmax=lmax, use_pixel_weights=True, iter=3)
#cls_syn_2 = cls_syn_2/hp.pixwin(nside)**2

np.random.seed(seed)
map_syn_2 = hp.synfast(cls_syn_2,nside=nside, lmax=lmax )

fig = plt.figure()
plt.suptitle(f'Channel:{nu_ch[ich]}')
fig.add_subplot(311)
hp.mollview(map_syn, cmap= 'viridis',min=0, max=1, hold=True)
fig.add_subplot(312)
hp.mollview(map_syn_2, cmap= 'viridis', min=0, max=1,hold=True)
fig.add_subplot(313)
hp.mollview(map_syn-map_syn_2, cmap= 'viridis', min=0, max=0.1,hold=True)
plt.show()


cls_ana = hp.anafast(map_syn_2, lmax=lmax, use_pixel_weights=True, iter=3)#/hp.pixwin(nside)**2

ell = np.arange(lmax+1)
factor2 = ell*(ell+1)/(2*np.pi)

fig= plt.figure()
plt.scatter(ell[1:],factor2[1:]*cls_syn_2[1:], s=5,label = 'Input')
plt.scatter(ell[1:],factor2[1:]*cls_ana[1:],  s=3,label='Anafast')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}$')
plt.tight_layout()
plt.show()


fig= plt.figure()
plt.plot(ell[1:],(cls_ana[1:]/cls_syn[1:]-1)*100)
plt.xlabel(r'$\ell$')
plt.ylabel(f'% diff rel')
plt.axhline(y=0, ls='--', color='grey')
plt.xlabel(r'$\ell$')
plt.ylabel(f'% rel diff')
plt.tight_layout()
plt.show()
