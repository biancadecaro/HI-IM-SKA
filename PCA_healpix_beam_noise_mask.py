import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pymaster as nm
sns.set_theme(style = 'white')
#sns.set_palette('husl',15)
c_pal = sns.color_palette().as_hex()

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)
###########################################################################
beam_s = 'theta40arcmin'
out_dir= f'PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise_mask0.39/'
out_dir_plot = f'PCA_pixels_output/Plots_PCA/No_mean/Beam_{beam_s}_noise_mask0.39/'

if not os.path.exists(out_dir):
        os.makedirs(out_dir)
if not os.path.exists(out_dir_plot):
        os.makedirs(out_dir_plot)

###################################################################################

fg_components='synch_ff_ps'
path_data_sims_tot = f'Sims/beam_{beam_s}_no_mean_sims_{fg_components}_noise_40freq_905.0_1295.0MHz_thick10MHz_lmax383_nside128'

with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

nu_ch= file['freq']

num_freq = len(nu_ch)

nu0 =1420
print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')
print(f'corresponding to the redshift range z: [{min(nu0/nu_ch -1.0):.2f} - {max(nu0/nu_ch -1.0):.2f}] ')

HI_maps_freq = file['maps_sims_HI'] + file['maps_sims_noise']  #aggiungo il noise
fg_maps_freq = file['maps_sims_fg']
full_maps_freq = file['maps_sims_tot'] + file['maps_sims_noise']  #aggiungo il noise


npix = np.shape(HI_maps_freq)[1]
nside = hp.get_nside(HI_maps_freq[0])
lmax=3*nside-1
num_sources = 3
print(f'nside:{nside}, lmax:{lmax}, num_ch:{num_freq}, min_ch:{min(nu_ch)}, max_ch:{max(nu_ch)}, Nfg:{num_sources}')

#######################################

mask1_70 = hp.read_map('HFI_Mask_GalPlane_2048_R1.10.fits', field=3)#fsky 40 % coverage
mask_70t = hp.ud_grade(mask1_70, nside_out=256)
mask_70 = hp.ud_grade(mask_70t, nside_out=nside)
del mask1_70
#mask_70s = hp.sphtfunc.smoothing(mask_70, 3*np.pi/180,lmax=lmax) #apodization 3 deg come in Olivari
#del mask_70
fsky  = np.mean(mask_70) 

fig=plt.figure()
hp.mollview(mask_70, cmap='viridis', title=f'fsky={np.mean(mask_70):0.2f}', hold=True)
#plt.savefig(f'Plots_sims/mask_apo3deg_fsky{np.mean(mask_70s):0.2f}_nside{nside}.png')
plt.show()


#######################################################################################

HI_maps_freq_mask = np.zeros((num_freq, npix))
fg_maps_freq_mask = np.zeros((num_freq, npix))
full_maps_freq_mask = np.zeros((num_freq, npix))

print(HI_maps_freq[0].mean(),full_maps_freq[0].mean())
for n in range(num_freq):
        
        HI_maps_freq_mask[n] = HI_maps_freq[n]*mask_70 #- HI_maps_freq_mean

        fg_maps_freq_mask[n] = fg_maps_freq[n]*mask_70 #- fg_maps_freq_mean
        fg_maps_freq_mask[n] = hp.remove_dipole(fg_maps_freq_mask[n])

        full_maps_freq_mask[n] = full_maps_freq[n]*mask_70 #- full_maps_freq_mean
        full_maps_freq_mask[n] = hp.remove_dipole(full_maps_freq_mask[n])

print(HI_maps_freq[0].mean(),full_maps_freq[0].mean())
#########################################################################################


ich = 21#int(num_freq/2)

fig=plt.figure()
hp.mollview(fg_maps_freq[ich], cmap='viridis', title=f'Input foreground, channel:{nu_ch[ich]} MHz', hold=True)
plt.show()


######################################################################################################

print(full_maps_freq.shape)
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
#plt.tight_layout()
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

res_fg_maps=eigenvec_fg_Nfg@eigenvec_fg_Nfg.T@full_maps_freq_mask

#The foreground residual that leaks into the recovered signal and noise
fg_leakage = fg_maps_freq_mask - eigenvec_fg_Nfg@eigenvec_fg_Nfg.T@fg_maps_freq_mask
HI_leakage = eigenvec_fg_Nfg@eigenvec_fg_Nfg.T@HI_maps_freq_mask

del eigenvec_fg_Nfg

res_HI=np.zeros((num_freq,npix))
res_HI = full_maps_freq_mask - res_fg_maps

######################################################################################################

#np.save(out_dir+f'cosmo_HI_noise_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax}_nside{nside}.npy',HI_maps_freq_mask)
#np.save(out_dir+f'res_PCA_HI_noise{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax}_nside{nside}.npy',res_HI)
#np.save(out_dir+f'fg_leak_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax}_nside{nside}.npy',fg_leakage)
#np.save(out_dir+f'HI_leak_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax}_nside{nside}.npy',HI_leakage)
#np.save(out_dir+f'fg_input_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax}_nside{nside}.npy',fg_maps_freq_mask)

##########################################################################################################

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, lmax:{lmax}, Nfg:{num_sources}, fksy:{fsky:0.2f}',fontsize=20)
fig.add_subplot(331) 
hp.mollview(np.abs(res_fg_maps[ich]/fg_maps_freq_mask[ich]-1)*100,cmap='viridis', min=0, max=0.2, title=f'%(Res_fg/x_fg - 1)',unit='%' ,hold=True)
fig.add_subplot(332) 
hp.mollview(HI_maps_freq_mask[ich], cmap='viridis', title=f'HI signal + noise ',hold=True)#min=0, max =1,
fig.add_subplot(333)
hp.mollview(res_HI[ich], title=f'PCA HI + noise',min=0, max =1,cmap='viridis', hold=True)
fig.add_subplot(334)
hp.mollview(fg_leakage[ich], title=f'Foreground leakage', min=0, max=0.5,cmap='viridis', hold=True)
fig.add_subplot(335)
hp.mollview(HI_leakage[ich], title=f'HI leakage', min=0, max=0.5,cmap='viridis', hold=True)
plt.show()

del file

#fig=plt.figure(figsize=(10, 7))
#fig.suptitle(f'channel: {nu_ch[ich]} MHz, lmax:{lmax}, Nfg:{num_sources}, fksy:{fsky:0.2f}',fontsize=20)
#fig.add_subplot(131) 
#hp.gnomview(HI_maps_freq_mask[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='Input HI', cmap='viridis', hold=True)
#fig.add_subplot(132) 
#hp.gnomview(res_HI[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='PCA HI+noise', cmap= 'viridis', hold=True)
#fig.add_subplot(133) 
#hp.gnomview(HI_maps_freq_mask[ich]-res_HI[ich], rot=[-22,21],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-0.2, max=0.2, title='PCA residuals', cmap= 'viridis', hold=True)
##plt.tight_layout()
#plt.show()

#fig = plt.figure(figsize=(10, 7))
#fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
#fig.add_subplot(221)
#hp.mollview(HI_maps_freq_mask[ich]-res_HI[ich], title=f'PCA residuals', min=0, max=0.5,cmap='viridis', hold=True)
#fig.add_subplot(222)
#hp.mollview(fg_leakage[ich], title=f'Foreground leakage', min=0, max=0.5,cmap='viridis', hold=True)
#fig.add_subplot(223)
#hp.mollview(HI_leakage[ich], title=f'HI leakage', min=0, max=0.5,cmap='viridis', hold=True)
#plt.show()

###############################################################################################################
out_dir_cl = out_dir+'power_spectra_cls_from_healpix_maps/'
if not os.path.exists(out_dir_cl):
        os.makedirs(out_dir_cl)

lmax_cl = 2*nside

cl_HI_cosmo_full=np.zeros((num_freq, lmax_cl+1))
cl_HI_cosmo_mask=np.zeros((num_freq, lmax_cl+1))
#cl_fg_mask=np.zeros((num_freq, lmax_cl+1))
cl_Hi_recons_Nfg=np.zeros((num_freq, lmax_cl+1))
cl_fg_leak_Nfg=np.zeros((num_freq, lmax_cl+1))
cl_HI_leak_Nfg=np.zeros((num_freq, lmax_cl+1))

for i in range(num_freq):
    cl_HI_cosmo_full[i] = hp.anafast(HI_maps_freq[i], lmax=lmax_cl)    
    cl_HI_cosmo_mask[i] = hp.anafast(HI_maps_freq_mask[i], lmax=lmax_cl)
    #cl_fg_mask[i] = hp.anafast(fg_maps_freq_mask[i], lmax=lmax_cl)
    cl_Hi_recons_Nfg[i] = hp.anafast(res_HI[i], lmax=lmax_cl)
    cl_fg_leak_Nfg[i]=hp.anafast(fg_leakage[i], lmax=lmax_cl)
    cl_HI_leak_Nfg[i]=hp.anafast(HI_leakage[i], lmax=lmax_cl)
#
#np.savetxt(out_dir_cl+f'cl_input_HI_noise_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax_cl}_nside{nside}.dat', cl_Hi_mask)
#np.savetxt(out_dir_cl+f'cl_input_fg_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax_cl}_nside{nside}.dat', cl_fg)
#np.savetxt(out_dir_cl+f'cl_PCA_HI_noise_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_Hi_recons_Nfg)
#np.savetxt(out_dir_cl+f'cl_leak_HI_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_HI_leak_Nfg)
#np.savetxt(out_dir_cl+f'cl_leak_fg_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_fg_leak_Nfg)

del HI_maps_freq; del fg_leakage; del HI_leakage;del fg_maps_freq

ell = np.arange(0, lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)


fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Channel:{nu_ch[ich]} MHz, BEAM {beam_s}, lmax:{lmax}, Nfg:{num_sources}, fsky:{fsky:0.2f}')
plt.semilogy(ell[2:], factor[2:]*cl_HI_cosmo_full[ich][2:],'k--',mfc='none', label='Cosmo HI+noise full sky')
plt.semilogy(ell[2:], factor[2:]*cl_HI_cosmo_mask[ich][2:],mfc='none', label='Cosmo HI+noise mask / fsky')
plt.semilogy(ell[2:], factor[2:]*cl_Hi_recons_Nfg[ich][2:],'+',color=c_pal[1],mfc='none', label='PCA HI+noise fsky')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi}C_{\ell}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 10))


diff_cl_HI_PCA_cosmo_mask = cl_Hi_recons_Nfg/cl_HI_cosmo_mask -1 
#diff_cl_HI_cosmo = cl_Hi_recons_Nfg/cl_HI_cosmo_full -1 

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_HI_PCA_cosmo_mask[ich][2:]*100, color=c_pal[1],label='PCA HI mask/cosmo HI mask')

frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-50,50])
frame2.set_ylabel(r'%$ C_{\ell}^{\rm PCA} / C_{\ell}^{\rm cosmo} -1$')
frame2.set_xlabel(r'$\ell$')
frame1.set_xticks(np.arange(1,200+1, 10))
#plt.tight_layout()
plt.legend()

#####

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Mean over channel, BEAM {beam_s}, lmax:{lmax}, Nfg:{num_sources}, fsky:{fsky:0.2f}')
plt.semilogy(ell[2:], factor[2:]*np.mean(cl_HI_cosmo_full, axis=0)[2:],'k--',mfc='none', label='Cosmo HI+noise full sky')
plt.semilogy(ell[2:], factor[2:]*np.mean(cl_HI_cosmo_mask, axis=0)[2:],mfc='none', label='Cosmo HI+noise fsky')
plt.semilogy(ell[2:], factor[2:]*np.mean(cl_Hi_recons_Nfg, axis=0)[2:],'+',mfc='none', label='PCA HI+noise fsky')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\langle \frac{\ell(\ell+1)}{2\pi}C_{\ell} \rangle $')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 10))


frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], np.mean(diff_cl_HI_PCA_cosmo_mask, axis=0)[2:]*100,color=c_pal[1],label='PCA HI mask/cosmo HI mask')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-50,50])
frame2.set_ylabel(r'%$ \langle C_{\ell}^{\rm PCA} / C_{\ell}^{\rm cosmo} -1 \rangle $')
frame2.set_xlabel(r'$\ell$')
frame1.set_xticks(np.arange(1,200+1, 10))
#plt.tight_layout()



fig=plt.figure()
plt.suptitle('Mean over channels, leakage')
plt.semilogy(ell[2:],factor[2:]*np.mean(cl_fg_leak_Nfg, axis=0)[2:],mfc='none', label='Fg leakage')
plt.semilogy(ell[2:],factor[2:]*np.mean(cl_HI_leak_Nfg, axis=0)[2:],mfc='none', label='HI leakage')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell*(\ell+1)}{2\pi} \langle C_{\ell} \rangle$')
plt.xlim([0,200])
plt.legend()
plt.show()
#print(min(100*np.mean(cl_Hi_recons_Nfg/cl_Hi-1, axis=0)), max(100*np.mean(cl_Hi_recons_Nfg/cl_Hi-1, axis=0)))


####################################################################################
##### confronto maschera non maschera - mashchera deconvolta #######################

#### deconvoluzione
f_0_mask = nm.NmtField(mask_70,[res_HI[0]] )
b = nm.NmtBin.from_nside_linear(nside, 8)
ell_mask= b.get_effective_ells()

cl_PCA_HI_mask_deconv = np.zeros((num_freq, len(ell_mask)))
cl_PCA_HI_mask_deconv_interp = np.zeros((num_freq, lmax_cl+1))

for n in range(num_freq):
    f_0_mask = nm.NmtField(mask_70,[res_HI[n]] )
    cl_PCA_HI_mask_deconv[n] = nm.compute_full_master(f_0_mask, f_0_mask, b)[0]
    cl_PCA_HI_mask_deconv_interp[n] = np.interp(ell, ell_mask, cl_PCA_HI_mask_deconv[n])

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Channel:{nu_ch[ich]} MHz, BEAM {beam_s}, lmax:{lmax}, Nfg:{num_sources}, fsky:{fsky:0.2f}')
plt.plot(ell[2:], factor[2:]*cl_HI_cosmo_full[ich][2:],'k--',mfc='none', label='Cosmo HI+noise full sky')
plt.plot(ell[2:], factor[2:]*cl_HI_cosmo_mask[ich][2:]/fsky,mfc='none', label='Cosmo HI+noise mask / fsky')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_mask_deconv_interp[ich][2:],'+',color=c_pal[1],mfc='none', label='PCA HI+noise maskplot deconv')
plt.xlim([0,150])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi}C_{\ell}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 10))


diff_cl_HI_PCA_cosmo_deconv_full = cl_PCA_HI_mask_deconv_interp/cl_HI_cosmo_full -1 
#diff_cl_HI_cosmo = cl_Hi_recons_Nfg/cl_HI_cosmo_full -1 

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_HI_PCA_cosmo_deconv_full[ich][2:]*100, color=c_pal[1],label='PCA HI mask deconv/cosmo HI full')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,150])
frame2.set_ylim([-20,20])
frame2.set_ylabel(r'%$ C_{\ell}^{\rm PCA} / C_{\ell}^{\rm cosmo} -1$')
frame2.set_xlabel(r'$\ell$')
frame1.set_xticks(np.arange(1,200+1, 10))
#plt.tight_layout()
plt.legend()
plt.show()
#####

##############################################################################
##### confronto maschera non maschera #########################################

#cl_res_HI_no_mask = np.loadtxt(f'PCA_pixels_output/Maps_PCA/No_mean/Beam_theta40arcmin_noise/power_spectra_cls_from_healpix_maps/cl_PCA_HI_noise_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat')
#
#
#diff_cl_PCA_mask_full = cl_Hi_recons_Nfg/cl_res_HI_no_mask -1 
#
#fig = plt.figure(figsize=(10,7))
#frame1=fig.add_axes((.1,.3,.8,.6))
#plt.title(f'Channel:{nu_ch[ich]} MHz, BEAM {beam_s}, lmax:{lmax}, Nfg:{num_sources}')
#plt.semilogy(ell[2:], factor[2:]*cl_HI_cosmo_full[ich][2:],'k--', mfc='none', label='Cosmo HI full sky')
#plt.semilogy(ell[2:], factor[2:]*cl_res_HI_no_mask[ich][2:],mfc='none', label='PCA HI full sky')
#plt.xlim([0,200])
#plt.legend()
#frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi}C_{\ell}$')
#frame1.set_xlabel([])
#frame1.set_xticks(np.arange(1,200+1, 10))
#
#frame2=fig.add_axes((.1,.1,.8,.2))
#plt.plot(ell[2:], diff_cl_PCA_mask_full[ich][2:]*100)
#frame2.axhline(ls='--', c= 'k', alpha=0.3)
#frame2.set_xlim([0,200])
#frame2.set_ylim([-50,50])
#frame2.set_ylabel(r'%$ C_{\ell}^{\rm PCA,mask} / C_{\ell}^{\rm PCA,full} -1$')
#frame2.set_xlabel(r'$\ell$')
#frame1.set_xticks(np.arange(1,200+1, 10))
##plt.tight_layout()
#plt.legend()
#
#
#
#
#fig = plt.figure(figsize=(10,7))
#frame1=fig.add_axes((.1,.3,.8,.6))
#plt.title(f'Mean over channels, BEAM {beam_s}, lmax:{lmax}, Nfg:{num_sources}')
#plt.semilogy(ell[2:], factor[2:]*np.mean(cl_HI_cosmo_full, axis=0)[2:],'k--',mfc='none', label='Cosmo HI full sky')
#plt.semilogy(ell[2:], factor[2:]*np.mean(cl_res_HI_no_mask, axis=0)[2:],mfc='none', label='PCA HI full sky')
#plt.xlim([0,200])
#plt.legend()
#frame1.set_ylabel(r'$\langle \frac{\ell(\ell+1)}{2\pi}C_{\ell}\rangle$')
#frame1.set_xlabel([])
#frame1.set_xticks(np.arange(1,200+1, 10))
#
#frame2=fig.add_axes((.1,.1,.8,.2))
#plt.plot(ell[2:], np.mean(diff_cl_PCA_mask_full, axis=0)[2:]*100)
#frame2.axhline(ls='--', c= 'k', alpha=0.3)
#frame2.set_xlim([0,200])
#frame2.set_ylim([-50,50])
#frame2.set_ylabel(r'%$\langle C_{\ell}^{\rm PCA } / C_{\ell}^{\rm cosmo} -1 \rangle $')
#frame2.set_xlabel(r'$\ell$')
#frame1.set_xticks(np.arange(1,200+1, 10))
##plt.tight_layout()
#plt.legend()
#
#plt.show()