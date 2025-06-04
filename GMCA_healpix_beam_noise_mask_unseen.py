import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py
sys.path.insert(1, '/home/bianca/Documents/gmca4im-master/scripts/')
import gmca4im_lib2 as g4i
import seaborn as sns
import numpy.ma as ma
import copy
import pymaster as nm
sns.set_theme(style = 'white')
sns.color_palette(n_colors=15)
#sns.set_palette('husl',15)
sns.palettes.color_palette()
c_pal = sns.color_palette().as_hex()

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)
################################################################
beam_s = '1.3deg_SKA_AA4'
out_dir= f'GMCA_pixels_output/Maps_GMCA/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
out_dir_plot = f'GMCA_pixels_output/Plots_GMCA_healpix/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'

if not os.path.exists(out_dir):
        os.makedirs(out_dir)
if not os.path.exists(out_dir_plot):
        os.makedirs(out_dir_plot)

################################################################

fg_components='synch_ff_ps_pol'

path_data_sims_tot = f'Sims/beam_{beam_s}_no_mean_sims_{fg_components}_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax383_nside128'

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


######################################################################################################
npix = np.shape(HI_maps_freq)[1]
nside = hp.get_nside(HI_maps_freq[0])
lmax=3*nside-1
if fg_components=='synch_ff_ps':
    num_sources=3
if fg_components=='synch_ff_ps_pol':
    num_sources=3
print(num_sources)
print(f'nside:{nside}, lmax:{lmax}, num_ch:{num_freq}, min_ch:{min(nu_ch)}, max_ch:{max(nu_ch)}, Nfg:{num_sources}')

#######################################

pix_mask = hp.query_strip(nside, theta1=np.pi*2/3, theta2=np.pi/3)
print(pix_mask)
mask_50 = np.zeros(npix)
mask_50[pix_mask] =1
fsky_50 = np.sum(mask_50)/hp.nside2npix(nside)


#######################################################################################
bad_v = np.where(mask_50==0)

HI_maps_freq_mask = copy.deepcopy(HI_maps_freq)
fg_maps_freq_mask = copy.deepcopy(fg_maps_freq)
full_maps_freq_mask = copy.deepcopy(full_maps_freq)

print(full_maps_freq_mask.shape)

for n in range(num_freq):
		HI_maps_freq_mask[n][bad_v] =  hp.UNSEEN
		HI_maps_freq_mask[n]=hp.remove_dipole(HI_maps_freq_mask[n])

		fg_maps_freq_mask[n][bad_v] =  hp.UNSEEN
		fg_maps_freq_mask[n]=hp.remove_dipole(fg_maps_freq_mask[n])

		full_maps_freq_mask[n][bad_v] =  hp.UNSEEN
		full_maps_freq_mask[n]=hp.remove_dipole(full_maps_freq_mask[n])

#########################################################################################


ich = int(num_freq/2)
fig = plt.figure()
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(full_maps_freq_mask[ich], cmap='viridis', title=f'Observations', hold=True)
fig.add_subplot(222)
hp.mollview(fg_maps_freq_mask[ich], cmap='viridis', title=f'Input foreground', hold=True)
fig.add_subplot(223)
hp.mollview(HI_maps_freq_mask[ich], cmap='viridis', min=0, max=1,title=f'Input HI + noise', hold=True)
plt.show()


######################################################################################################

#mask_bool = mask_40
#mask_bool[bad_v] = True
#mask_bool[mask_bool==1] = False
##mask_bool = np.array(mask_40,dtype='bool')
#print(mask_bool)

full_maps_freq_masked=ma.zeros((num_freq,npix))
#fg_maps_freq_masked=ma.zeros((num_freq,npix))
#HI_maps_freq_masked=ma.zeros((num_freq,npix))
#full_maps_freq_masked= ma.asarray(full_maps_freq_masked)
maskt =np.zeros(mask_50.shape)
maskt[bad_v]=  1
mask = ma.make_mask(maskt, shrink=False)
#mask_b = np.array(maskt,dtype='int')
#hp.mollview(mask_b, title='mask b', cmap = 'viridis')



for n in range(num_freq):
        full_maps_freq_masked[n]  =ma.MaskedArray(full_maps_freq_mask[n], mask=mask)#np.isnan(full_maps_freq_mask[n])


#hp.mollview(full_maps_freq_masked[0], cmap='viridis', title='masked ma')
#plt.show()


######################################################################################################
Cov_channels=ma.cov(full_maps_freq_masked)

fig=plt.figure()
plt.imshow(Cov_channels, cmap='crest')
plt.xlabel('[MHz]')
plt.ylabel('[MHz]')
plt.colorbar()
#plt.show()

eigenval, eigenvec= np.linalg.eig(Cov_channels)
fig= plt.figure(figsize=(7,4))
plt.semilogy(np.arange(1,num_freq+1),eigenval,'--.',mfc='none',markersize=10)
x_ticks = np.arange(-10, num_freq+1, 10 )
ax = plt.gca()
ax.set(xlim=[-10,num_freq+2],xticks=x_ticks,xlabel="eigenvalue number",ylabel="$\\lambda$",title='Eigenvalues')
#plt.tight_layout()
#plt.savefig(out_dir_plot+f'eigenvalues_Nfg_{fg_components}.png')
#plt.show()

#############################################################################
############################# GMCA ##########################################

################   GMCA PARAMETERS   ##################
#num_sources   = 3   # number of sources to be estimated
mints = 0.1 # min threshold (what is sparse compared to noise?)
nmax  = 100 # number of iterations (usually 100 is safe)
L0    = 0   # switch between L0 norm (1) or L1 norm (0)
#######################################################

# initial guess for the mixing matrix?
# i.e. we could start from GMCA-determined mix matrix
AInit = None

# we can impose a column of the mixing matrix
ColFixed = None

# we can whiten the data
whitening = False; epsi = 1e-3

# estimated mixing matrix:
Ae = g4i.ma_run_GMCA(full_maps_freq_masked,AInit,num_sources,mints,nmax,L0,ColFixed,whitening,epsi)
#fig=plt.figure()
#plt.suptitle('mixing matrix')
#plt.imshow(Ae,cmap='crest')
#plt.colorbar()
#plt.show()
#np.save(out_dir+f'Ae_mixing_matrix_{num_freq}_Nfg{num_sources}_lmax{lmax}_nside{nside}', Ae)

#####################################################################################

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
plt.plot(abs(Ae/np.linalg.norm(Ae,axis=0)),label='mix mat column')
plt.plot(FF_col/np.linalg.norm(FF_col),'m:',label='gal ff')

ax = plt.gca()
ax.set(ylim=[0.0,0.4],xlabel="frequency channel",ylabel="Spectral emission",title='GMCA-mixing matrix columns')
plt.legend(fontsize=12)
plt.show()
#############################################################################################
#Foreground's maps from GMCA
#### reconstructed maps by GMCA

piA = np.ma.dot(np.linalg.inv(np.ma.dot(Ae.T,Ae)),Ae.T)
# reproject onto original maps 
Se_sph = np.ma.dot(piA,full_maps_freq_mask) # LS estimate of the sources in the pixel domain
res_fg_maps = np.ma.dot(Ae,Se_sph)
del Se_sph, piA

#The foreground residual that leaks into the recovered signal and noise
fg_leakage = fg_maps_freq_mask - np.ma.dot(np.ma.dot(Ae,np.linalg.inv(np.ma.dot(Ae.T,Ae))),np.ma.dot(Ae.T,fg_maps_freq_mask))
HI_leakage = np.ma.dot(np.ma.dot(Ae,np.linalg.inv(np.ma.dot(Ae.T,Ae))),np.ma.dot(Ae.T,HI_maps_freq_mask))
fg_leakage[:,bad_v]=hp.UNSEEN
HI_leakage[:,bad_v] = hp.UNSEEN

del Ae


res_HI=np.zeros((num_freq,npix))
res_HI = full_maps_freq_mask - res_fg_maps

res_HI[:,bad_v]=hp.UNSEEN
#########################################

HI_maps_freq_mask.dump(out_dir+f'cosmo_HI_noise_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax}_nside{nside}.npy')
res_HI.dump(out_dir+f'res_PCA_HI_noise{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax}_nside{nside}.npy')
#fg_leakage.dump(out_dir+f'fg_leak_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax}_nside{nside}.npy')
#HI_leakage.dump(out_dir+f'HI_leak_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax}_nside{nside}.npy')
fg_maps_freq_mask.dump(out_dir+f'fg_input_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax}_nside{nside}.npy')


#np.save(out_dir+f'cosmo_HI_noise_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax}_nside{nside}.npy',HI_maps_freq)
#np.save(out_dir+f'res_GMCA_HI_noise{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax}_nside{nside}.npy',res_HI)
#np.save(out_dir+f'fg_leak_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax}_nside{nside}.npy',fg_leakage)
#np.save(out_dir+f'HI_leak_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax}_nside{nside}.npy',HI_leakage)
#np.save(out_dir+f'fg_input_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax}_nside{nside}.npy',fg_maps_freq)

#####################################################################################################################

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(np.abs(res_fg_maps[ich]/fg_maps_freq_mask[ich]-1)*100,cmap='viridis', min=0, max=0.2, title=f'%(Res_fg/x_fg - 1), channel:{nu_ch[ich]}',unit='%' ,hold=True)
fig.add_subplot(222) 
hp.mollview(HI_maps_freq_mask[ich]-file['maps_sims_HI'][ich], cmap='viridis', title=f'HI signal + noise - HI freq={nu_ch[ich]}',hold=True)#min=0, max =1,
fig.add_subplot(223)
hp.mollview(res_HI[ich], title=f'GMCA HI + noise freq={nu_ch[ich]}',cmap='viridis', hold=True)
plt.show()

del file

fig=plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, lmax:{lmax}, Nfg:{num_sources}',fontsize=20)
fig.add_subplot(131) 
hp.gnomview(HI_maps_freq_mask[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='Input HI', cmap='viridis', hold=True)
fig.add_subplot(132) 
hp.gnomview(res_HI[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='GMCA HI+noise', cmap= 'viridis', hold=True)
fig.add_subplot(133) 
hp.gnomview(HI_maps_freq_mask[ich]-res_HI[ich], rot=[-22,21],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-0.2, max=0.2, title='GMCA residuals', cmap= 'viridis', hold=True)
#plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221)
hp.mollview(HI_maps_freq_mask[ich]-res_HI[ich], title=f'GMCA residuals freq={nu_ch[ich]}', min=0, max=0.5,cmap='viridis', hold=True)
fig.add_subplot(222)
hp.mollview(fg_leakage[ich], title=f'Foreground leakage freq={nu_ch[ich]}', min=0, max=0.5,cmap='viridis', hold=True)
fig.add_subplot(223)
hp.mollview(HI_leakage[ich], title=f'HI leakage freq={nu_ch[ich]}', min=0, max=0.5,cmap='viridis', hold=True)
plt.show()
###########################################################################################################
out_dir_cl = out_dir+'power_spectra_cls_from_healpix_maps/'
if not os.path.exists(out_dir_cl):
        os.makedirs(out_dir_cl)

lmax_cl = 2*nside

cl_HI_cosmo_full = np.zeros((num_freq, lmax_cl+1))
cl_Hi=np.zeros((num_freq, lmax_cl+1))
cl_Hi_recons_Nfg=np.zeros((num_freq, lmax_cl+1))
cl_fg_leak_Nfg=np.zeros((num_freq, lmax_cl+1))
cl_HI_leak_Nfg=np.zeros((num_freq, lmax_cl+1))

for i in range(num_freq):
    cl_Hi[i] = hp.anafast(HI_maps_freq_mask[i], lmax=lmax_cl)
    cl_HI_cosmo_full[i] = hp.anafast(HI_maps_freq[i], lmax=lmax_cl)
    cl_Hi_recons_Nfg[i] = hp.anafast(res_HI[i], lmax=lmax_cl)
    cl_fg_leak_Nfg[i]=hp.anafast(fg_leakage[i], lmax=lmax_cl)
    cl_HI_leak_Nfg[i]=hp.anafast(HI_leakage[i], lmax=lmax_cl)

#
#np.savetxt(out_dir_cl+f'cl_input_HI_noise_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax_cl}_nside{nside}.dat', cl_Hi)
#np.savetxt(out_dir_cl+f'cl_input_fg_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax_cl}_nside{nside}.dat', cl_fg)
#np.savetxt(out_dir_cl+f'cl_GMCA_HI_noise_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_Hi_recons_Nfg)
#np.savetxt(out_dir_cl+f'cl_leak_HI_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_HI_leak_Nfg)
#np.savetxt(out_dir_cl+f'cl_leak_fg_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_fg_leak_Nfg)

del fg_leakage; del HI_leakage

ell = np.arange(0, lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)


fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Channel:{nu_ch[ich]} MHz, BEAM {beam_s}, lmax:{lmax}, Nfg:{num_sources}, fsky_50:{fsky_50}')
plt.semilogy(ell[2:], factor[2:]*cl_HI_cosmo_full[ich][2:],'k--',mfc='none', label='Cosmo HI+noise full sky')
plt.semilogy(ell[2:], factor[2:]*cl_Hi[ich][2:],mfc='none', label='Cosmo HI+noise fsky_50')
plt.semilogy(ell[2:], factor[2:]*cl_Hi_recons_Nfg[ich][2:],'+',color=c_pal[1],mfc='none', label='GMCA HI+noise fsky_50 mask UNSEEN')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi}C_{\ell}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 10))


diff_cl_GMCA_cosmo = cl_Hi_recons_Nfg/cl_Hi -1 

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_GMCA_cosmo[ich][2:]*100, color=c_pal[1],label='mask UNSEEN')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-50,50])
frame2.set_ylabel(r'%$ C_{\ell}^{\rm GMCA} / C_{\ell}^{\rm cosmo} -1$')
frame2.set_xlabel(r'$\ell$')
frame1.set_xticks(np.arange(1,200+1, 10))
#plt.tight_layout()
plt.legend()

#####

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Mean over channel, BEAM {beam_s}, lmax:{lmax}, Nfg:{num_sources}, fsky_50:{fsky_50}')
plt.semilogy(ell[2:], factor[2:]*np.mean(cl_HI_cosmo_full, axis=0)[2:],'k--',mfc='none', label='Cosmo HI+noise full sky')
plt.semilogy(ell[2:], factor[2:]*np.mean(cl_Hi, axis=0)[2:],mfc='none', label='Cosmo HI+noise')
plt.semilogy(ell[2:], factor[2:]*np.mean(cl_Hi_recons_Nfg, axis=0)[2:],'+',mfc='none', label='GMCA HI+noise')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\langle \frac{\ell(\ell+1)}{2\pi}C_{\ell} \rangle $')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 10))


frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], np.mean(diff_cl_GMCA_cosmo, axis=0)[2:]*100)
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-50,50])
frame2.set_ylabel(r'%$ \langle C_{\ell}^{\rm GMCA} / C_{\ell}^{\rm cosmo} -1 \rangle $')
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



####################################################################################
##### confronto maschera non maschera - maschera deconvolta #######################

#### deconvoluzione
#f_0_mask = nm.NmtField(mask_50,[res_HI[0]], masked_on_input=True )
b = nm.NmtBin.from_nside_linear(nside, 8)
ell_mask= b.get_effective_ells()

cl_GMCA_HI_mask_deconv = np.zeros((num_freq, len(ell_mask)))
cl_GMCA_HI_mask_deconv_interp = np.zeros((num_freq, lmax_cl+1))

#cl_GMCA_HI_mask_0_deconv = np.zeros((num_freq, len(ell_mask)))
#cl_GMCA_HI_mask_0_deconv_interp = np.zeros((num_freq, lmax_cl+1))

for n in range(num_freq):
    f_0_mask = nm.NmtField(mask_50,[res_HI[n]] )#, masked_on_input=True)
    cl_GMCA_HI_mask_deconv[n] = nm.compute_full_master(f_0_mask, f_0_mask, b)[0]
    cl_GMCA_HI_mask_deconv_interp[n] = np.interp(ell, ell_mask, cl_GMCA_HI_mask_deconv[n])
    

np.savetxt(out_dir_cl+f'cl_deconv_GMCA_HI_noise_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_GMCA_HI_mask_deconv_interp)



fig = plt.figure(figsize=(10,7))
#frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Mean over channel, BEAM {beam_s}, lmax:{lmax}, Nfg:{num_sources}, fsky_50:{fsky_50}')
plt.semilogy(ell[2:], factor[2:]*np.mean(cl_HI_cosmo_full, axis=0)[2:],'k--',mfc='none', label='Cosmo HI+noise full sky')
plt.semilogy(ell[2:], factor[2:]*np.mean(cl_GMCA_HI_mask_deconv_interp, axis=0)[2:],'+',mfc='none', label='GMCA HI+noise')
plt.xlim([0,200])
plt.legend()
plt.ylabel(r'$\langle \frac{\ell(\ell+1)}{2\pi}C_{\ell} \rangle $')
plt.xlabel([])

plt.show()
###############

#cl_res_HI_no_mask = np.loadtxt(f'GMCA_pixels_output/Maps_GMCA/No_mean/Beam_theta40arcmin_noise/power_spectra_cls_from_healpix_maps/cl_GMCA_HI_noise_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat')
#
#diff_cl_GMCA_mask_full = cl_Hi_recons_Nfg/cl_res_HI_no_mask -1 
#diff_cl_GMCA_mask_full_deconv = cl_GMCA_HI_mask_deconv_interp/cl_res_HI_no_mask -1 
#diff_cl_GMCA_mask_cosmo_full_deconv = cl_GMCA_HI_mask_deconv_interp/cl_HI_cosmo_full -1 
#
#diff_cl_GMCA_mask_0_full_deconv = cl_GMCA_HI_mask_0_deconv_interp/cl_res_HI_no_mask -1 
#diff_cl_GMCA_mask_0_cosmo_full_deconv = cl_GMCA_HI_mask_0_deconv_interp/cl_HI_cosmo_full -1 
#
#fig = plt.figure(figsize=(10,7))
#frame1=fig.add_axes((.1,.3,.8,.6))
#plt.title(f'Channel:{nu_ch[ich]} MHz, BEAM {beam_s}, lmax:{lmax}, Nfg:{num_sources}')
#plt.semilogy(ell[2:], factor[2:]*cl_HI_cosmo_full[ich][2:],'k--', mfc='none', label='Cosmo HI full sky')
#plt.semilogy(ell[2:], factor[2:]*cl_res_HI_no_mask[ich][2:],mfc='none', label='GMCA HI full sky')
##plt.semilogy(ell[2:], factor[2:]*cl_Hi_recons_Nfg[ich][2:],'+',mfc='none', label='GMCA HI fsky_50')
#plt.semilogy(ell[2:], factor[2:]*cl_GMCA_HI_mask_deconv_interp[ich][2:],'+',mfc='none', label='GMCA HI fsky_50 deconvolution mask UNSEEN')
#plt.semilogy(ell[2:], factor[2:]*cl_GMCA_HI_mask_0_deconv_interp[ich][2:],'+',mfc='none', label='GMCA HI fsky_50 deconvolution mask 0')
#plt.xlim([0,200])
#plt.legend()
#frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi}C_{\ell}$')
#frame1.set_xlabel([])
#frame1.set_xticks(np.arange(1,200+1, 10))
#
#frame2=fig.add_axes((.1,.1,.8,.2))
##plt.plot(ell[2:], diff_cl_GMCA_mask_full_deconv[ich][2:]*100, label='% GMCA mask UNSEEN deconv/GMCA full sky -1 ')
#plt.plot(ell[2:], diff_cl_GMCA_mask_cosmo_full_deconv[ich][2:]*100,c_pal[1],  label='% GMCA mask UNSEEN deconv /cosmo full sky -1  ')
##plt.plot(ell[2:], diff_cl_GMCA_mask_0_full_deconv[ich][2:]*100, label='% GMCA mask 0 deconv/GMCA full sky -1 ')
#plt.plot(ell[2:], diff_cl_GMCA_mask_0_cosmo_full_deconv[ich][2:]*100, c_pal[2], label='% GMCA mask 0 deconv /cosmo full sky -1  ')
#frame2.axhline(ls='--', c= 'k', alpha=0.3)
#frame2.set_xlim([0,200])
#frame2.set_ylim([-50,50])
#frame2.set_ylabel(r'%$ C_{\ell}^{\rm GMCA,mask} / C_{\ell}^{\rm GMCA,full} -1$')
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
#plt.semilogy(ell[2:], factor[2:]*np.mean(cl_res_HI_no_mask, axis=0)[2:],mfc='none', label='GMCA HI full sky')
##plt.semilogy(ell[2:], factor[2:]*np.mean(cl_Hi_recons_Nfg, axis=0)[2:],'+',mfc='none', label='GMCA HI fsky_50')
#plt.semilogy(ell[2:], factor[2:]*np.mean(cl_GMCA_HI_mask_deconv_interp, axis=0)[2:],'+',mfc='none', label='GMCA HI fsky_50 deconvolution mask UNSEEN')
#plt.semilogy(ell[2:], factor[2:]*np.mean(cl_GMCA_HI_mask_0_deconv_interp, axis=0)[2:],'+',mfc='none', label='GMCA HI fsky_50 deconvolution mask 0')
#plt.xlim([0,200])
#plt.legend()
#frame1.set_ylabel(r'$\langle \frac{\ell(\ell+1)}{2\pi}C_{\ell}\rangle$')
#frame1.set_xlabel([])
#frame1.set_xticks(np.arange(1,200+1, 10))
#
#frame2=fig.add_axes((.1,.1,.8,.2))
##plt.plot(ell[2:], np.mean(diff_cl_GMCA_mask_full_deconv, axis=0)[2:]*100, label='% GMCA mask deconv/GMCA full sky -1 ')
#plt.plot(ell[2:], np.mean(diff_cl_GMCA_mask_cosmo_full_deconv, axis=0)[2:]*100,c_pal[1], label='% GMCA mask deconv /cosmo full sky -1  ')
##plt.plot(ell[2:], np.mean(diff_cl_GMCA_mask_0_full_deconv, axis=0)[2:]*100, label='% GMCA mask 0 deconv/GMCA full sky -1 ')
#plt.plot(ell[2:], np.mean(diff_cl_GMCA_mask_0_cosmo_full_deconv, axis=0)[2:]*100, c_pal[2],label='% GMCA mask 0 deconv /cosmo full sky -1  ')
#frame2.axhline(ls='--', c= 'k', alpha=0.3)
#frame2.set_xlim([0,200])
#frame2.set_ylim([-50,50])
#frame2.set_ylabel(r'%$\langle C_{\ell}^{\rm GMCA } / C_{\ell}^{\rm cosmo} -1 \rangle $')
#frame2.set_xlabel(r'$\ell$')
#frame1.set_xticks(np.arange(1,200+1, 10))
##plt.tight_layout()
#plt.legend()
#
#plt.show()