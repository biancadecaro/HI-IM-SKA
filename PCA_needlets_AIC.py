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
        alm_write = hp.sphtfunc.almxfl(alm_bands[0:(nlm_tot).astype(int)], bands[ i, 0:(uselmax + 1).astype(int)])
        
        # Transform back to Healpix format

        wavelets_write = hp.sphtfunc.alm2map(alm_write, usenside)
                
        # Write in wavelets fits file
        #usenside_a[i] = int(usenside)
        pyfits.append(waveletsfile, wavelets_write)

    #return usenside_a

def wavelets2map(wavelets, nside):

    """This function combines wavelets maps into a single map (procedure reverse of map2wavelets)."""
    
    # Get synthesis windows

    windows = pyfits.getdata(wavelets, 0)
    nw = (windows[:,0]).size


    # Create output alm, alm index and ell values

    lmax = (windows[0,:]).size - 1
    nalm = hp.sphtfunc.Alm.getsize(lmax)

    alm_out= np.zeros(nalm, dtype=np.complex128)

    for i in range (1, nw + 1):
        map = pyfits.getdata(wavelets, i)
        wh = np.where(windows[i - 1, :] != 0)
        if max(max(wh)) < lmax:
            win_lmax = max(max(wh))
        else:
            win_lmax = lmax

        alm = hp.sphtfunc.map2alm(map, lmax=win_lmax)#, iter=1, use_weights=True)
        alm_win = hp.sphtfunc.almxfl(alm, windows[ i - 1,0:win_lmax + 1])

        for l in range(0, win_lmax + 1):
            for m in range(0, l + 1):
                ind_0 = hp.sphtfunc.Alm.getidx(win_lmax, l, m)
                ind_1 = hp.sphtfunc.Alm.getidx(lmax, l, m)
                alm_out[ind_1] = alm_out[ind_1] + alm_win[ind_0]

    # Make map

    map_o = hp.sphtfunc.alm2map(alm_out, nside)

    return map_o


###########################################################################3
dir_w = 'wavelets_AIC/'
if not os.path.exists(dir_w):
        os.makedirs(dir_w)

fg_comp='synch_ff_ps_pol'
path_data_sims_tot = f'Sims/beam_theta40arcmin_no_mean_sims_{fg_comp}_noise_40freq_905.0_1295.0MHz_thick10MHz_lmax383_nside128'
with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

nu_ch= file['freq']

HI_maps_freq = file['maps_sims_HI'] + file['maps_sims_noise']
fg_maps_freq = file['maps_sims_fg']
full_maps_freq = file['maps_sims_tot'] + file['maps_sims_noise']
noise_maps_freq = file['maps_sims_noise']

num_freq = len(nu_ch)

nu_ch = np.linspace(905.0, 1295.0, num_freq)
min_ch = min(nu_ch)
max_ch = max(nu_ch)

jmax=4
npix = HI_maps_freq.shape[1]
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
    
    alm2wavelets(alm_map, bvalues[0:relevant_band_max[i] + 1,:], nside, dir_w+'wavelet_' + str(i).strip() + '.fits', nside)


#fig, ax1  = plt.subplots(1,1,figsize=(7,5)) 
#ax1.plot( bvalues[relevant_band_max[i]-4]**2 )
#ax1.plot( bvalues[relevant_band_max[i]-3]**2 )
#ax1.plot( bvalues[relevant_band_max[i]-2]**2 )
#ax1.plot( bvalues[relevant_band_max[i]-1]**2 )
#ax1.plot( bvalues[relevant_band_max[i]]**2 )
#ax1.set_xscale('log')
#ax1.set_xlabel(r'$\ell$')
#ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
#plt.show()


cov = np.zeros((nbands,num_freq, num_freq))
corr = np.zeros((nbands,num_freq, num_freq))
#tot_map_wav = np.zeros((num_freq, jmax+1))
eigenval=np.zeros((nbands, num_freq))
eigenvec=np.zeros((nbands, num_freq,num_freq))
for j in range(nbands):
    for c in range(0, num_freq):
        tot_map_wav=pyfits.getdata(dir_w+'wavelet_' + str(c).strip() + '.fits', j + 1) 
        for cc in range(0,c+1):
            tot_map_wav2=pyfits.getdata(dir_w+'wavelet_' + str(cc).strip() + '.fits', j + 1) 
            npixx = hp.nside2npix(hp.get_nside(tot_map_wav))
            cov[j,c,cc]=np.dot(tot_map_wav,tot_map_wav2.T)
            cov[j, cc,c] = cov[j,c,cc]
            #corr[j,c,cc] = cov[j,c,cc]/np.sqrt(cov[j,c,c]*cov[j,cc,cc])
            #corr[j,c,cc]=corr[j,cc,c]
    tot_map_wav=0
    tot_map_wav2=0

#fig = plt.figure()
#plt.imshow(cov[-2], cmap='viridis')


for j in range(nbands):
    #eigenval[j], eigenvec[j] = np.linalg.s(cov[j])#np.linalg.eigh(Cov_channels[j])
    eigenvec[j], eigenval[j], Vr = np.linalg.svd(cov[j], full_matrices=True)

#fig = plt.figure(figsize=(10, 7))
#fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}',fontsize=20)
#for j in range(jmax+1):
#    fig.add_subplot(4, 4, j + 1)
#    hp.mollview(tot_map_wav[ich][j], title=f'Res HI j={j}', cmap='viridis', hold=True)
#plt.show()


fig = plt.figure(figsize=(8,4))
for j in range(eigenval.shape[0]):
    plt.semilogy(np.arange(1,num_freq+1),eigenval[j],'--o',mfc='none',markersize=5,label=f'j={j}')

plt.legend(fontsize=12, ncols=2)
x_ticks = np.arange(-10,num_freq+10, 10)
ax = plt.gca()
ax.set(xlim=[-10,num_freq+10],xticks=x_ticks,xlabel="eigenvalue number",ylabel="$\\lambda$",title='STANDARD NEED - Eigenvalues')
plt.show()

n_dim = np.zeros(nbands)
for j in range(nbands):
    AIC = np.zeros(num_freq)
    fun = eigenval[j] - np.log(eigenval[j]) - 1.0
    total = np.sum(fun)
    for r in range(1, num_freq + 1):
        if r < num_freq:
            total = total - fun[r - 1]
            AIC[r - 1] = 2 * r + total
        else:
            AIC[r - 1] = 2 * r
    n_dim[j] = max(np.where(AIC == np.ndarray.min(AIC))[0]) + 1

print(n_dim)
#n_dim = [3.,3.,3.,3.,3.]

for i in range(0, num_freq):
    pyfits.append(dir_w+'wavelet_pca_target_' + str(i).strip() + '.fits', bvalues[0:relevant_band_max[i] + 1,:])



for j in range(nbands):
    tot_map_wav =[]
    for c in range(num_freq):
        tot_map_wav.append(pyfits.getdata(dir_w+'wavelet_' + str(c).strip() + '.fits', j + 1) )
    #print(eigenvec_fg_Nfg[j].shape, eigenvec_fg_Nfg[j].T.shape, len(tot_map_wav))
    res_fg_maps = np.dot(eigenvec[j, :,:int(n_dim[j])],np.dot(eigenvec[j, :,:int(n_dim[j])].T,tot_map_wav))
    res_HI_maps =tot_map_wav-res_fg_maps
    #print(res_HI_maps.shape)
    #hp.mollview(tot_map_wav[0], cmap='viridis', title=f'Obs tot j:{j}, channel:{nu_ch[0]}')
    #hp.mollview(res_fg_maps[0], cmap='viridis', title=f'Res fg j:{j}, channel:{nu_ch[0]}')
    #hp.mollview(res_HI_maps[0], cmap='viridis', title=f'Obs-Res fg j:{j}, channel:{nu_ch[0]}')
    #plt.show()

    for cc in range(num_freq):
        #print(res_HI_maps[cc].shape)
        pyfits.append(dir_w+'wavelet_pca_target_' + str(cc).strip() + '.fits', res_HI_maps[cc])
    
recon_obs_maps = np.zeros((num_freq, hp.pixelfunc.nside2npix(nside)))
recon_map_HI = np.zeros((num_freq, hp.pixelfunc.nside2npix(nside)))

for i in range(0, num_freq):
   recon_map_HI[i,:] = wavelets2map(dir_w+'wavelet_pca_target_' + str(i).strip() + '.fits', nside)
   recon_obs_maps[i,:] = wavelets2map(dir_w+'wavelet_' + str(i).strip() + '.fits', nside)
   


fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}',fontsize=20)
fig.add_subplot(221)
hp.mollview(HI_maps_freq[ich], title=f'input HI ',min=0,max=1, cmap='viridis', hold=True)
fig.add_subplot(222)
hp.mollview(recon_map_HI[ich], title=f'Res HI', min=0,max=1, cmap='viridis', hold=True)
fig.add_subplot(223)
hp.mollview(full_maps_freq[ich], title=f'input obs ',min=-1e3,max=1e3, cmap='viridis', hold=True)
fig.add_subplot(224)
hp.mollview(recon_obs_maps[ich], title=f'recons obs', min=-1e3,max=1e3, cmap='viridis', hold=True)


del recon_obs_maps; del full_maps_freq
#############################################################################################
################################# CL #############################################
lmax_cl = 2*nside#512

cl_cosmo_HI = np.zeros((len(nu_ch), lmax_cl+1))
cl_PCA_HI_need2harm = np.zeros((len(nu_ch), lmax_cl+1))

for n in range(len(nu_ch)):
    cl_cosmo_HI[n]=hp.anafast(HI_maps_freq[n], lmax=lmax_cl)
    cl_PCA_HI_need2harm[n] = hp.anafast(recon_map_HI[n], lmax=lmax_cl)

del HI_maps_freq; del recon_map_HI

ell=np.arange(lmax_cl+1)
factor=ell*(ell+1)/(2*np.pi)


fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: channel:{nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{n_dim}')
plt.plot(ell[2:],factor[2:]*cl_cosmo_HI[ich][2:], label='Cosmo HI + noise')
plt.plot(ell[2:],factor[2:]*cl_PCA_HI_need2harm[ich][2:],'+', mfc='none', label='PCA HI + noise')
plt.ylabel(r'$\frac{\ell(\ell+1)}{2\pi}  C_{\ell} $')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} e C_{\ell} $')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 30))

diff_cl_need2sphe = cl_PCA_HI_need2harm/cl_cosmo_HI-1
frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe[ich][2:]*100, label='% PCA_HI/input_HI -1')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-10,10])
frame2.set_ylabel(r'%$  diff$')
frame2.set_xlabel(r'$\ell$')
frame2.set_xticks(np.arange(1,200+1, 30))
#plt.tight_layout()
plt.legend()



fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: mean over channels, jmax:{jmax}, lmax:{lmax}, Nfg:{n_dim}')
plt.plot(ell[2:], factor[2:]*cl_cosmo_HI.mean(axis=0)[2:], label = f'Cosmo HI + noise')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_need2harm.mean(axis=0)[2:],'+',mfc='none', label = f'PCA HI + noise')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 30))


frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe.mean(axis=0)[2:]*100, label='% PCA_HI/input_HI -1')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-10,10])
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
frame2.set_xticks(np.arange(1,200+1, 30))
#plt.tight_layout()
plt.legend()


#################################################################################################################
################################ confronto con la decomposizione normale ########################################
cl_PCA_HI_need2harm_vecchio = np.loadtxt(f'PCA_needlets_output/maps_reconstructed/No_mean/Beam_theta40arcmin_noise/cls_recons_need/cl_PCA_HI_noise_{fg_comp}_40_905_1295MHz_Nfg3_jmax4_lmax256_nside128.dat')

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: channel:{nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{n_dim}')
plt.plot(ell[2:],factor[2:]*cl_cosmo_HI[ich][2:], label='Cosmo HI + noise')
plt.plot(ell[2:],factor[2:]*cl_PCA_HI_need2harm[ich][2:],'+', mfc='none', label='PCA HI + noise')
plt.plot(ell[2:],factor[2:]*cl_PCA_HI_need2harm_vecchio[ich][2:],'+', mfc='none', label='vecchio PCA HI + noise')
plt.ylabel(r'$\frac{\ell(\ell+1)}{2\pi}  C_{\ell} $')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} e C_{\ell} $')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 30))

diff_cl_need2sphe = cl_PCA_HI_need2harm/cl_cosmo_HI-1
diff_cl_need2sphe_vecchio = cl_PCA_HI_need2harm_vecchio/cl_cosmo_HI-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe[ich][2:]*100, label='% PCA_HI/input_HI -1')
plt.plot(ell[2:], diff_cl_need2sphe_vecchio[ich][2:]*100, label='vecchio % PCA_HI/input_HI -1')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-10,10])
frame2.set_ylabel(r'%$  diff$')
frame2.set_xlabel(r'$\ell$')
frame2.set_xticks(np.arange(1,200+1, 30))
#plt.tight_layout()
plt.legend()


fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: mean over channels, jmax:{jmax}, lmax:{lmax}, Nfg:{n_dim}')
plt.plot(ell[2:], factor[2:]*cl_cosmo_HI.mean(axis=0)[2:], label = f'Cosmo HI + noise')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_need2harm.mean(axis=0)[2:],'+',mfc='none', label = f'PCA HI + noise')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_need2harm_vecchio.mean(axis=0)[2:],'+',mfc='none', label = f'vecchio PCA HI + noise')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 30))

del cl_PCA_HI_need2harm; del cl_cosmo_HI
frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe.mean(axis=0)[2:]*100, label='% PCA_HI/input_HI -1')
plt.plot(ell[2:], diff_cl_need2sphe_vecchio.mean(axis=0)[2:]*100, label='vecchio % PCA_HI/input_HI -1')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-10,10])
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
frame2.set_xticks(np.arange(1,200+1, 30))
#plt.tight_layout()
plt.legend()


plt.show()
##################################################################################################################
import subprocess
for i in range(0, num_freq):
    file =dir_w+ 'wavelet_' + str(i).strip() + '.fits'
    subprocess.call(["rm", file])
    file = dir_w+'wavelet_pca_target_' + str(i).strip() + '.fits' 
    subprocess.call(["rm", file])