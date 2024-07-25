import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

import seaborn as sns
sns.set()
sns.set(style = 'white')
#sns.set_palette('husl',15)

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

#print(sns.color_palette("husl", 15).as_hex())
sns.palettes.color_palette()

path_data_sims_tot = 'Sims/sims_synch_ff_ps_40freq_905.0_1295.0MHz_nside256'
with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

out_dir_output = 'PCA_needlets_output/'
out_dir_output_PCA = out_dir_output+'PCA_maps/Reshape_scales/'
out_dir_plot = out_dir_output+'Plots_PCA_needlets/Reshape_scales/'
if not os.path.exists(out_dir_output):
        os.makedirs(out_dir_output)
if not os.path.exists(out_dir_output_PCA):
        os.makedirs(out_dir_output_PCA)

nu_ch= file['freq']
HI_maps_freq = file['maps_sims_HI']
fg_maps_freq = file['maps_sims_fg']
full_maps_freq = file['maps_sims_tot']
del file



fg_comp = 'synch_ff_ps'

need_dir = 'Maps_needlets/'
need_tot_maps_filename = need_dir+f'bjk_maps_obs_{fg_comp}_40freq_905.0_1295.0MHz_jmax12_lmax512_B1.68_nside256.npy'
need_tot_maps = np.load(need_tot_maps_filename)

jmax=need_tot_maps.shape[1]-1

num_freq = need_tot_maps.shape[0]
nu_ch = np.linspace(905.0, 1295.0, num_freq)
min_ch = min(nu_ch)
max_ch = max(nu_ch)
npix = need_tot_maps.shape[2]
nside = hp.npix2nside(npix)
lmax=2*nside#3*nside-1#
B=pow(lmax,(1./jmax))



for nu in range(len(nu_ch)):
    alm_obs = hp.map2alm(full_maps_freq[nu], lmax=lmax)
    full_maps_freq[nu] = hp.alm2map(alm_obs, lmax=lmax, nside = nside)
    del alm_obs
    alm_HI = hp.map2alm(HI_maps_freq[nu], lmax=lmax)
    HI_maps_freq[nu] = hp.alm2map(alm_HI, lmax=lmax, nside = nside)
    del alm_HI
    alm_fg = hp.map2alm(fg_maps_freq[nu], lmax=lmax)
    fg_maps_freq[nu] = hp.alm2map(alm_fg, lmax=lmax, nside = nside)
    del alm_fg


print(f'jmax:{jmax}, lmax:{lmax}, B:{B:1.2f}, num_freq:{num_freq}, min_ch:{min_ch}, max_ch:{max_ch}, nside:{nside}')


#np.save(out_dir_output+f'need_HI_maps_{num_freq}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz.npy',need_HI_maps)
need_obs_reshape = np.zeros((num_freq,npix*jmax))
print(need_obs_reshape.shape, )
for nu in range(num_freq):
    need_obs_reshape[nu,:] = need_tot_maps[nu,0:jmax,:].reshape(1, -1)
print(need_obs_reshape.shape,npix*(jmax), npix*(jmax+1))

Cov_channels_need=np.cov(need_obs_reshape)

fig=plt.figure()
plt.imshow(Cov_channels_need, cmap='crest')
#plt.yticks(ticks=np.arange(nu_ch.shape[0]),labels=nu_ch)
#plt.xticks(ticks=np.arange(nu_ch.shape[0]),labels=nu_ch, rotation=45)
plt.xlabel('[MHz]')
plt.ylabel('[MHz]')
plt.colorbar()
plt.show()

eigenval_need, eigenvec_need= np.linalg.eig(Cov_channels_need)

Cov_channels=np.cov(full_maps_freq)

eigenval, eigenvec= np.linalg.eig(Cov_channels)
del Cov_channels

fig= plt.figure(figsize=(7,4))
plt.semilogy(np.arange(1,num_freq+1),eigenval_need,'--.',mfc='none',markersize=10, label='Need')
plt.semilogy(np.arange(1,num_freq+1),eigenval,'--.',mfc='none',markersize=10)

x_ticks = np.arange(-10, num_freq+1, 10 )
ax = plt.gca()
ax.set(xlim=[-10,num_freq+2],xticks=x_ticks,xlabel="eigenvalue number",ylabel="$\\lambda$",title='Eigenvalues')
plt.tight_layout()
plt.legend()
#plt.savefig(out_dir_plot+f'eigenvalues_Nfg_{fg_comp}.png')
plt.show()

num_sources=3

Nfg = num_freq - num_sources
eigenvec_fg_Nfg_need = eigenvec_need[:, 0:num_sources]#eigenvec_need[:num_freq, Nfg:num_freq]
eigenvec_fg_Nfg = eigenvec[:, 0:num_sources]#eigenvec_need[:num_freq, Nfg:num_freq]

fig=plt.figure()
plt.imshow(eigenvec_fg_Nfg_need, cmap='crest')
#plt.yticks(ticks=np.arange(nu_ch.shape[0]),labels=nu_ch)
#plt.xticks(ticks=np.arange(nu_ch.shape[0]),labels=nu_ch, rotation=45)
plt.xlabel('[MHz]')
plt.ylabel('[MHz]')
plt.colorbar()
plt.show()

del eigenvec_need, eigenvec

# gal freefree spectral index for reference
FF_col = np.array([nu_ch**(-2.13)]).T 

# gal synchrotron spectral index region for reference
sync_A = np.array([nu_ch**(-3.2)]).T; y1 = sync_A/np.linalg.norm(sync_A)
sync_B = np.array([nu_ch**(-2.6)]).T; y2 = sync_B/np.linalg.norm(sync_B)

### actual plotting
fig=plt.figure()
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams["axes.labelsize"] = 12

x = np.arange(0,len(nu_ch))

plt.fill_between(x,y1.T[0],y2.T[0],alpha=0.3,label='gal synch')
plt.plot(abs(eigenvec_fg_Nfg_need/np.linalg.norm(eigenvec_fg_Nfg_need,axis=0)),label='mix mat column')
plt.plot(FF_col/np.linalg.norm(FF_col),'m:',label='gal ff')

ax = plt.gca()
ax.set(ylim=[0.0,0.4],xlabel="frequency channel",title='mixing matrix columns')
plt.legend(fontsize=12)
plt.show()

#Foreground's maps from PCA

res_fg_maps_need=eigenvec_fg_Nfg_need@eigenvec_fg_Nfg_need.T@full_maps_freq
res_fg_maps=eigenvec_fg_Nfg@eigenvec_fg_Nfg.T@full_maps_freq


#The foreground residual that leaks into the recovered signal and noise
fg_leakage_need = fg_maps_freq - eigenvec_fg_Nfg_need@eigenvec_fg_Nfg_need.T@fg_maps_freq
HI_leakage_need = eigenvec_fg_Nfg_need@eigenvec_fg_Nfg_need.T@HI_maps_freq

fg_leakage = fg_maps_freq - eigenvec_fg_Nfg@eigenvec_fg_Nfg.T@fg_maps_freq
HI_leakage = eigenvec_fg_Nfg@eigenvec_fg_Nfg.T@HI_maps_freq

del eigenvec_fg_Nfg_need, eigenvec_fg_Nfg

## Res HI

res_HI_need = full_maps_freq - res_fg_maps_need
res_HI = full_maps_freq - res_fg_maps
### Plot

ich = int(num_freq/2)

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(np.abs(res_fg_maps_need[ich]/fg_maps_freq[ich]-1)*100,cmap='viridis', min=0, max=0.1, title=f'%(Res_fg/x_fg - 1), channel:{nu_ch[ich]}',unit='%' ,hold=True)
fig.add_subplot(222) 
hp.mollview(HI_maps_freq[ich], cmap='viridis', title=f'HI signal freq={nu_ch[ich]}',min=0, max =1,hold=True)
fig.add_subplot(223)
hp.mollview(res_HI_need[ich], title=f'PCA HI freq={nu_ch[ich]}',min=0, max =1,cmap='viridis', hold=True)
plt.show()

fig=plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
fig.add_subplot(131) 
hp.gnomview(HI_maps_freq[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='Input HI', cmap='viridis', hold=True)
fig.add_subplot(132) 
hp.gnomview(res_HI_need[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='PCA HI', cmap= 'viridis', hold=True)
fig.add_subplot(133) 
hp.gnomview(100*(res_HI_need[ich]/HI_maps_freq[ich]-1), rot=[-22,21],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-20, max=20, title='% PCA HI/HI -1', cmap= 'viridis', hold=True)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221)
hp.mollview(HI_maps_freq[ich]-res_HI_need[ich], title=f'Cosmo HI - PCA Res HI freq={nu_ch[ich]}', min=0, max=0.5,cmap='viridis', hold=True)
fig.add_subplot(222)
hp.mollview(fg_leakage_need[ich], title=f'Foreground leakage freq={nu_ch[ich]}', min=0, max=0.5,cmap='viridis', hold=True)
fig.add_subplot(223)
hp.mollview(HI_leakage_need[ich], title=f'HI leakage freq={nu_ch[ich]}', min=0, max=0.5,cmap='viridis', hold=True)
#plt.savefig(out_dir_plot+f'maps_ch{nu_ch[ich]}_cosmo_leak_fg_HI_{fg_components}_Nfg{num_sources}.png')
plt.show()

####################################################################################################

cl_Hi=np.zeros((num_freq, lmax+1))
cl_Hi_recons_Nfg_need=np.zeros((num_freq, lmax+1))
cl_Hi_recons_Nfg=np.zeros((num_freq, lmax+1))
cl_fg_leak_Nfg=np.zeros((num_freq, lmax+1))
cl_HI_leak_Nfg=np.zeros((num_freq, lmax+1))

for i in range(num_freq):
    cl_Hi[i] = hp.anafast(HI_maps_freq[i], lmax=lmax)
    cl_Hi_recons_Nfg_need[i] = hp.anafast(res_HI_need[i], lmax=lmax)
    cl_Hi_recons_Nfg[i] = hp.anafast(res_HI[i], lmax=lmax)
    cl_fg_leak_Nfg[i]=hp.anafast(fg_leakage_need[i], lmax=lmax)
    cl_HI_leak_Nfg[i]=hp.anafast(HI_leakage_need[i], lmax=lmax)

#del alm_map_res_ch; del alm_HI_ch
del res_HI_need; del HI_maps_freq; del fg_leakage_need; del HI_leakage_need;del fg_maps_freq

ell = np.arange(0, lmax+1)
factor = ell*(ell+1)/(2*np.pi)

fig=plt.figure()
plt.semilogy(ell,factor*np.mean(cl_fg_leak_Nfg, axis=0),mfc='none', label='Fg leakage')
plt.semilogy(ell,factor*np.mean(cl_HI_leak_Nfg, axis=0),mfc='none', label='HI leakage')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell*(\ell+1)}{2\pi} \langle C_{\ell} \rangle$')
#plt.xlim([0,200])
plt.legend()
#plt.savefig(out_dir_plot+f'cls_leakage_HI_fg_{fg_components}_Nfg{num_sources}.png')
plt.show()

fig=plt.figure()
plt.plot(ell, factor*np.mean(cl_Hi, axis=0),mfc='none', label='Cosmo HI')
plt.plot(ell, factor*np.mean(cl_Hi_recons_Nfg_need, axis=0),'+',mfc='none', label='Recovered HI')
#plt.plot(factor*np.mean(cl_fg_leak_Nfg, axis=0),mfc='none', label='Fg leakage')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell*(\ell+1)}{2\pi} \langle C_{\ell} \rangle$')
plt.xlim([15,200])
plt.legend()
#plt.savefig(out_dir_plot+f'cls_HI_cls_PCA_{fg_components}_Nfg{num_sources}.png')
plt.show()

fig=plt.figure()
plt.plot(ell[1:], np.mean(cl_Hi_recons_Nfg_need/cl_Hi-1, axis=0)[1:],'--.',mfc='none',label=f'Need')
plt.plot(ell[1:], np.mean(cl_Hi_recons_Nfg/cl_Hi-1, axis=0)[1:],'--.',mfc='none')
#plt.plot(factor*np.mean(cl_fg_leak_Nfg, axis=0),'--',mfc='none', label='Foreground leakage')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle$')
#plt.xlim([0,200])
plt.ylim([-0.1,0.1])
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.legend()
#plt.savefig(out_dir_plot+f'relative_diff_cls_PCA_cosmo_HI_{fg_components}_Nfg{num_sources}.png')
plt.show()

