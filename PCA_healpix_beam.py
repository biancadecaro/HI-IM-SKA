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

out_dir= 'PCA_pixels_output/Maps_PCA/No_mean/Beam_40arcmin/'
out_dir_plot = 'PCA_pixels_output/Plots_PCA_healpix/No_mean/Beam_40arcmin/'

if not os.path.exists(out_dir):
        os.makedirs(out_dir)
if not os.path.exists(out_dir_plot):
        os.makedirs(out_dir_plot)

###################################################################################

fg_components='synch_ff_ps_pol'
path_data_sims_tot = f'Sims/beam_theta40arcmin_no_mean_sims_{fg_components}_40freq_905.0_1295.0MHz_lmax383_nside128'

with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

nu_ch= file['freq']

num_freq = len(nu_ch)

nu0 =1420
print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')
print(f'corresponding to the redshift range z: [{min(nu0/nu_ch -1.0):.2f} - {max(nu0/nu_ch -1.0):.2f}] ')

HI_maps_freq = file['maps_sims_HI']
fg_maps_freq = file['maps_sims_fg']
full_maps_freq = file['maps_sims_tot']

######################################################################################################
ich = int(num_freq/2)

fig=plt.figure()
hp.mollview(fg_maps_freq[ich], cmap='viridis', title=f'Input foreground, channel:{nu_ch[ich]} MHz', hold=True)
plt.show()

del file
######################################################################################################


npix = np.shape(HI_maps_freq)[1]
nside = hp.get_nside(HI_maps_freq[0])
lmax=3*nside-1
num_sources = 18
print(f'nside:{nside}, lmax:{lmax}, num_ch:{num_freq}, min_ch:{min(nu_ch)}, max_ch:{max(nu_ch)}, Nfg:{num_sources}')


######################################################################################

Cov_channels=np.cov(full_maps_freq)

fig=plt.figure()
plt.imshow(Cov_channels, cmap='crest')
plt.xlabel('[MHz]')
plt.ylabel('[MHz]')
plt.colorbar()
plt.show()

eigenval, eigenvec= np.linalg.eig(Cov_channels)


fig= plt.figure(figsize=(7,4))
plt.semilogy(np.arange(1,num_freq+1),eigenval,'--.',mfc='none',markersize=10)
x_ticks = np.arange(-10, num_freq+1, 10 )
ax = plt.gca()
ax.set(xlim=[-10,num_freq+2],xticks=x_ticks,xlabel="eigenvalue number",ylabel="$\\lambda$",title='Eigenvalues')
plt.tight_layout()
#plt.savefig(out_dir_plot+f'eigenvalues_Nfg_{fg_components}.png')
plt.show()

#############################################################################
############################# PCA ##########################################

eigenvec_fg_Nfg = eigenvec[:, 0:num_sources]

fig=plt.figure()
plt.imshow(eigenvec_fg_Nfg, cmap='crest')
plt.xlabel('[MHz]')
plt.ylabel('[MHz]')
plt.colorbar()
plt.show()

del eigenvec

#for r in range(0,num_sources):
#    eigenvec_fg_Nfg[:,r] = eigenvec_fg_Nfg[:,r]/lng.norm(eigenvec_fg_Nfg[:,r])
######################################################################################

# gal freefree spectral index for reference
FF_col = np.array([nu_ch**(-2.13)]).T 

# gal synchrotron spectral index region for reference
sync_A = np.array([nu_ch**(-3.2)]).T 
sync_B = np.array([nu_ch**(-2.6)]).T 
y1 = sync_A/np.linalg.norm(sync_A)
y2 = sync_B/np.linalg.norm(sync_B)

### actual plotting
fig=plt.figure()
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams["axes.labelsize"] = 12

x = np.arange(0,len(nu_ch))

plt.fill_between(x,y1.T[0],y2.T[0],alpha=0.3,label='gal synch')
plt.plot(abs(eigenvec_fg_Nfg/np.linalg.norm(eigenvec_fg_Nfg,axis=0)),label='mix mat column')
plt.plot(FF_col/np.linalg.norm(FF_col),'m:',label='gal ff')

ax = plt.gca()
ax.set(ylim=[0.0,0.4],xlabel="frequency channel",ylabel="Spectral emission",title='PCA-mixing matrix columns')
plt.legend(fontsize=12)
plt.show()
################################################################################################

#Foreground's maps from PCA

res_fg_maps=eigenvec_fg_Nfg@eigenvec_fg_Nfg.T@full_maps_freq

#The foreground residual that leaks into the recovered signal and noise
fg_leakage = fg_maps_freq - eigenvec_fg_Nfg@eigenvec_fg_Nfg.T@fg_maps_freq
HI_leakage = eigenvec_fg_Nfg@eigenvec_fg_Nfg.T@HI_maps_freq

del eigenvec_fg_Nfg

res_HI=np.zeros((num_freq,npix))
res_HI = full_maps_freq - res_fg_maps

######################################################################################################

np.save(out_dir+f'cosmo_HI_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax}_nside{nside}.npy',HI_maps_freq)
np.save(out_dir+f'res_PCA_HI_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax}_nside{nside}.npy',res_HI)
np.save(out_dir+f'fg_leak_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax}_nside{nside}.npy',fg_leakage)
np.save(out_dir+f'HI_leak_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax}_nside{nside}.npy',HI_leakage)
np.save(out_dir+f'fg_input_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax}_nside{nside}.npy',fg_maps_freq)

##########################################################################################################

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(np.abs(res_fg_maps[ich]/fg_maps_freq[ich]-1)*100,cmap='viridis', min=0, max=0.2, title=f'%(Res_fg/x_fg - 1), channel:{nu_ch[ich]}',unit='%' ,hold=True)
fig.add_subplot(222) 
hp.mollview(HI_maps_freq[ich], cmap='viridis', title=f'HI signal freq={nu_ch[ich]}',min=0, max =1,hold=True)
fig.add_subplot(223)
hp.mollview(res_HI[ich], title=f'PCA HI freq={nu_ch[ich]}',min=0, max =1,cmap='viridis', hold=True)
plt.show()

fig=plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, lmax:{lmax}, Nfg:{num_sources}',fontsize=20)
fig.add_subplot(131) 
hp.gnomview(HI_maps_freq[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='Input HI', cmap='viridis', hold=True)
fig.add_subplot(132) 
hp.gnomview(res_HI[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='PCA HI', cmap= 'viridis', hold=True)
fig.add_subplot(133) 
hp.gnomview(HI_maps_freq[ich]-res_HI[ich], rot=[-22,21],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-0.2, max=0.2, title='HI -PCA', cmap= 'viridis', hold=True)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221)
hp.mollview(HI_maps_freq[ich]-res_HI[ich], title=f'Cosmo HI - PCA Res HI freq={nu_ch[ich]}', min=0, max=0.5,cmap='viridis', hold=True)
fig.add_subplot(222)
hp.mollview(fg_leakage[ich], title=f'Foreground leakage freq={nu_ch[ich]}', min=0, max=0.5,cmap='viridis', hold=True)
fig.add_subplot(223)
hp.mollview(HI_leakage[ich], title=f'HI leakage freq={nu_ch[ich]}', min=0, max=0.5,cmap='viridis', hold=True)
plt.show()

###############################################################################################################
out_dir_cl = out_dir+'power_spectra_cls_from_healpix_maps/'
if not os.path.exists(out_dir_cl):
        os.makedirs(out_dir_cl)

lmax_cl = 2*nside

cl_Hi=np.zeros((num_freq, lmax_cl+1))
cl_fg=np.zeros((num_freq, lmax_cl+1))
cl_Hi_recons_Nfg=np.zeros((num_freq, lmax_cl+1))
cl_fg_leak_Nfg=np.zeros((num_freq, lmax_cl+1))
cl_HI_leak_Nfg=np.zeros((num_freq, lmax_cl+1))
cl_diff_HI_cosmo_PCA_Nfg=np.zeros((num_freq, lmax_cl+1))
cl_diff_HI_fg_leak_Nfg=np.zeros((num_freq, lmax_cl+1))

for i in range(num_freq):
    cl_Hi[i] = hp.anafast(HI_maps_freq[i], lmax=lmax_cl)
    cl_fg[i] = hp.anafast(fg_maps_freq[i], lmax=lmax_cl)
    cl_Hi_recons_Nfg[i] = hp.anafast(res_HI[i], lmax=lmax_cl)
    cl_fg_leak_Nfg[i]=hp.anafast(fg_leakage[i], lmax=lmax_cl)
    cl_HI_leak_Nfg[i]=hp.anafast(HI_leakage[i], lmax=lmax_cl)
    cl_diff_HI_cosmo_PCA_Nfg[i] = hp.anafast(HI_maps_freq[i]-res_HI[i], lmax=lmax_cl)
    cl_diff_HI_fg_leak_Nfg[i] = hp.anafast(HI_leakage[i]-fg_leakage[i], lmax=lmax_cl)
#
np.savetxt(out_dir_cl+f'cl_input_HI_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax_cl}_nside{nside}.dat', cl_Hi)
np.savetxt(out_dir_cl+f'cl_input_fg_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax_cl}_nside{nside}.dat', cl_fg)
np.savetxt(out_dir_cl+f'cl_PCA_HI_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_Hi_recons_Nfg)
np.savetxt(out_dir_cl+f'cl_leak_HI_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_HI_leak_Nfg)
np.savetxt(out_dir_cl+f'cl_leak_fg_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_fg_leak_Nfg)

del res_HI; del HI_maps_freq; del fg_leakage; del HI_leakage;del fg_maps_freq

ell = np.arange(0, lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)

fig=plt.figure()
plt.suptitle('Mean over channels')
plt.semilogy(ell[1:],factor[1:]*np.mean(cl_fg_leak_Nfg, axis=0)[1:],mfc='none', label='Fg leakage')
plt.semilogy(ell[1:],factor[1:]*np.mean(cl_HI_leak_Nfg, axis=0)[1:],mfc='none', label='HI leakage')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell*(\ell+1)}{2\pi} \langle C_{\ell} \rangle$')
plt.xlim([0,200])
plt.legend()
plt.show()

fig=plt.figure()
plt.suptitle(f'Channel {nu_ch[ich]} MHz')
plt.plot(ell[1:], factor[1:]*cl_Hi[ich][1:],mfc='none', label='Cosmo HI')
plt.plot(ell[1:], factor[1:]*cl_Hi_recons_Nfg[ich][1:],'+',mfc='none', label='PCA HI')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell(\ell+1)}{2\pi} C_{\ell} $')
plt.xlim([15,200])
plt.legend()
plt.show()

fig=plt.figure()
plt.suptitle('Mean over channels')
plt.plot(ell[1:], factor[1:]*np.mean(cl_Hi, axis=0)[1:],mfc='none', label='Cosmo HI')
plt.plot(ell[1:], factor[1:]*np.mean(cl_Hi_recons_Nfg, axis=0)[1:],'+',mfc='none', label='PCA HI')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle $')
plt.xlim([0,200])
plt.legend()
plt.show()


#fig=plt.figure()
#plt.suptitle(f'channel {nu_ch[ich]} MHz')
#plt.plot(ell[1:], 100*((cl_Hi[ich]-cl_Hi_recons_Nfg[ich])/cl_Hi[ich])[1:],'--.',mfc='none')
#plt.xlabel(r'$\ell$')
#plt.ylabel(r'$\% C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 $')
#plt.axhline(y=0,c='k',ls='--',alpha=0.5)
#plt.xlim([0,200])
#plt.ylim([-30,30])
#plt.tight_layout()
#plt.show()
#print(min(100*((cl_Hi[ich]-cl_Hi_recons_Nfg[ich])/cl_Hi[ich])), max(100*((cl_Hi[ich]-cl_Hi_recons_Nfg[ich])/cl_Hi[ich])))


fig=plt.figure()
plt.suptitle('Mean over channels')
plt.plot(ell[1:], 100*np.mean((cl_Hi-cl_Hi_recons_Nfg)/cl_Hi, axis=0)[1:],'--',mfc='none')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%\langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.xlim([0,200])
plt.ylim([-50,50])
plt.tight_layout()
plt.show()
print(min(100*np.mean(cl_Hi_recons_Nfg/cl_Hi-1, axis=0)), max(100*np.mean(cl_Hi_recons_Nfg/cl_Hi-1, axis=0)))

#
#fig=plt.figure()
#plt.semilogy(ell, np.mean(cl_diff_HI_cosmo_PCA_Nfg, axis=0), label=r'$C_{\ell}$ diff Cosmo HI - Res PCA HI map')#f'Nfg={num_freq-Nfg}')
#plt.semilogy(ell, np.mean(cl_diff_HI_fg_leak_Nfg, axis=0),'--',mfc='none', label=r'$C_{\ell}$ HI-Fg leakage')#-cl_fg_leak_Nfg+cl_HI_leak_Nfg
#plt.xlabel(r'$\ell$')
#plt.ylabel(r'$\langle C_{\ell}\rangle $')
#plt.ylim(top=2e-1, bottom=1e-9)
#plt.axhline(y=0,c='k',ls='--',alpha=0.5)
#plt.legend()
#plt.show()
#
#
#fig=plt.figure()
#plt.plot(ell, (np.mean(cl_diff_HI_fg_leak_Nfg/cl_diff_HI_cosmo_PCA_Nfg-1, axis=0))*100,'--',mfc='none')#-cl_fg_leak_Nfg+cl_HI_leak_Nfg
#plt.xlabel(r'$\ell$')
#plt.ylabel(r'%$\langle C^{\rm HI-fg~leak}_{\ell}/ C^{\rm cosmo-PCA~HI}_{\ell}\rangle $')
#plt.ylim(top=1e-10, bottom=-1e-10)
#plt.axhline(y=0,c='k',ls='--',alpha=0.5)
#plt.show()
#
