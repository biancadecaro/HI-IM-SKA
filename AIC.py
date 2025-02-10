# Akaike information criterion

import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lng
from astropy.io import fits as pyfits
import seaborn as sns
from needlets_analysis import theory
import os
sns.set_theme(style = 'white')
sns.color_palette(n_colors=15)
#sns.set_palette('husl',15)

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)


#############################################################################
def alm2wavelets(alm, bands, nside, waveletsfile, nside_max_w):

    """This function computes wavelets maps for each wavelet band using the input spherical harmonics coefficients."""

    # Get the alm

    nbands = (bands[:,0]).size
    l_max = 3. * nside - 1

    # Write the bands in fits file

    pyfits.append(waveletsfile, bands)

    # Start the band loop 
    usenside_a = np.zeros(nbands)

    for i in range(0, nbands):

	# Now filter, restricting the alm and index to the needed ones
        uselmax = max(max(np.where(bands[i,:] != 0)))
         
        possiblenside = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
        
        # [+]
        possiblelmax = 3*possiblenside-1

        # [-] 
        # whereok = np.where(possiblenside >= uselmax)
        
        # [+]
        whereok = np.where(possiblelmax >= uselmax)

        usenside = min(possiblenside[whereok])

        if usenside > nside_max_w:
            usenside = nside_max_w


        nlm_tot = uselmax * (uselmax + 1.0) / 2.0 + uselmax + 1.0
        
        alm_bands = np.zeros(nlm_tot.astype(int), dtype=np.complex128)
        index_in = np.zeros(nlm_tot.astype(int))
        index_out = np.zeros(nlm_tot.astype(int))

        j = 0
        for l in range(0, uselmax + 1):
            for m in range(0, l + 1):
                index_in[j] = hp.Alm.getidx(l_max, l, m)
                index_out[j] = hp.Alm.getidx(uselmax, l, m)
                j = j + 1

        for k in range(0, nlm_tot.astype(int)):
            alm_bands[index_out[k].astype(int)] = alm[index_in[k].astype(int)]

        alm_write = hp.sphtfunc.almxfl(alm_bands[0:(nlm_tot).astype(int)], bands[0:(uselmax + 1).astype(int), i])
        
        # Transform back to Healpix format

        wavelets_write = hp.sphtfunc.alm2map(alm_write, usenside)
                
        # Write in wavelets fits file
        usenside_a[i] = int(usenside)
        pyfits.append(waveletsfile, wavelets_write)

    return usenside_a


###########################################################################3
dir_w = 'wavelets'
if not os.path.exists(dir_w):
        os.makedirs(dir_w)

fg_comp='synch_ff_ps'
path_data_sims_tot = f'Sims/beam_theta40arcmin_no_mean_sims_{fg_comp}_noise_40freq_905.0_1295.0MHz_thick10MHz_lmax383_nside128'
with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

nu_ch= file['freq']

HI_noise_maps_freq = file['maps_sims_HI'] + file['maps_sims_noise']
fg_maps_freq = file['maps_sims_fg']
full_maps_freq = file['maps_sims_tot'] + file['maps_sims_noise']
noise_maps_freq = file['maps_sims_noise']

num_freq = len(nu_ch)

nu_ch = np.linspace(905.0, 1295.0, num_freq)
min_ch = min(nu_ch)
max_ch = max(nu_ch)

jmax=4
npix = HI_noise_maps_freq.shape[1]
nside = hp.npix2nside(npix)
lmax=3*nside-1
B=pow(lmax,(1./jmax))

print(f'jmax:{jmax}, lmax:{lmax}, B:{B:1.2f}, num_freq:{num_freq}, min_ch:{min_ch}, max_ch:{max_ch}, nside:{nside}')

ich = int(num_freq/2)

need_theory=theory.NeedletTheory(B,jmax, lmax)
bvalues = need_theory.b_values

#fig, ax1  = plt.subplots(1,1,figsize=(7,5)) 
#plt.suptitle('bvalues')
#for j in range(bvalues.shape[0]):
#    ax1.plot(bvalues[j]*bvalues[j], label = 'j='+str(j) )
#ax1.set_xscale('log')
#ax1.set_xlabel(r'$\ell$')
#ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
#ax1.legend(loc='right')

nbands = bvalues.shape[0]  # number of bands
lmax_bands = np.zeros(nbands, dtype=np.int32)   # bands effective ell max
for j in range(0, nbands):
    lmax_bands[j] = max(np.where(bvalues[j,:] != 0.0)[0])

relevant_band_max = np.zeros(num_freq, dtype=np.int32)
#temp= np.zeros(bvalues.shape)
usenside = np.zeros((num_freq, nbands), dtype=np.int32)

for i in range(0, num_freq):
    alm_map = hp.sphtfunc.map2alm(full_maps_freq[i, :], lmax=lmax)

    # Wavelet transform channel maps: band-pass filtering in (l,m)-space and transform back to real (pixel) space

    # relevant bands for each channel map

    if lmax <= max(lmax_bands):
        relevant_band_max[i] = min(np.where(lmax_bands[:] >= lmax)[0])
    else:
        relevant_band_max[i] = nbands - 1
        
    if lmax_bands[relevant_band_max[i]] == max(lmax_bands):
        relevant_band_max[i] = nbands - 1

    #temp[:,:] =  bvalues[0:relevant_band_max[i] + 1,:]
    usenside[i] =alm2wavelets(alm_map, bvalues[0:relevant_band_max[i] + 1,:], nside, dir_w+'wavelet_' + str(i).strip() + '.fits', nside)


print(usenside)

tot_map_wav = np.zeros((num_freq, jmax+1))
for c in range(0, num_freq):
        for j in range(jmax+1):
            print(usenside[c,j])
            tot_map_wav[c,j] = np.zeros(usenside[c,j])
            tot_map_wav[c, j] = pyfits.getdata(dir_w+'wavelet_' + str(c).strip() + '.fits', j + 1) #* mask_wav


fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}',fontsize=20)
for j in range(jmax+1):
    fig.add_subplot(4, 4, j + 1)
    hp.mollview(tot_map_wav[ich][j], title=f'Res HI j={j}', cmap='viridis', hold=True)
plt.show()