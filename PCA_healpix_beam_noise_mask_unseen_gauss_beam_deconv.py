import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import seaborn as sns
import numpy.ma as ma
import copy
import pymaster as nm
from convolution_func import create_bl_vec
sys.path.insert(1, '/home/bianca/Documents/gmca4im-master/scripts/')
from gmca4im_lib2 import convolve

sns.set_theme(style = 'white')

from matplotlib import colors
sns.palettes.color_palette()
c_pal = sns.color_palette().as_hex()

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)
###########################################################################
beam_s = 'SKA_AA4'
out_dir= f'PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_{beam_s}_noise_mask0.5_unseen_deconv/'
out_dir_plot = f'PCA_pixels_output/Plots_PCA/No_mean/Beam_{beam_s}_noise_mask0.5_unseen_deconv/'

if not os.path.exists(out_dir):
	os.makedirs(out_dir)
if not os.path.exists(out_dir_plot):
	os.makedirs(out_dir_plot)

###################################################################################

fg_components='synch_ff_ps'
path_data_sims_tot = f'Sims/nuovo_beam_{beam_s}_sims_{fg_components}_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax383_nside128'

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
noise = file['maps_sims_noise']


full_maps_freq = np.array([full_maps_freq[i] -np.mean(full_maps_freq[i],axis=0)  for i in range(num_freq)])
fg_maps_freq = np.array([fg_maps_freq[i] -np.mean(fg_maps_freq[i],axis=0)  for i in range(num_freq)])
HI_maps_freq = np.array([HI_maps_freq[i] -np.mean(HI_maps_freq[i],axis=0)  for i in range(num_freq)])


npix = np.shape(HI_maps_freq)[1]
nside = hp.get_nside(HI_maps_freq[0])
lmax=3*nside-1
if fg_components=='synch_ff_ps':
    num_sources=3
if fg_components=='synch_ff_ps_pol':
    num_sources=3#
print(f'nside:{nside}, lmax:{lmax}, num_ch:{num_freq}, min_ch:{min(nu_ch)}, max_ch:{max(nu_ch)}, Nfg:{num_sources}')


###########################################################################
######## Computing beam size using given survey specifics: ################
### initialise a dictionary with the instrument specifications
### for noise and beam calculation
c_light = 3.0*1e8  # m/s
dish_diam_MeerKat = 13.5 #m
dish_diam_SKA = 15 # m
Ndishes_MeerKAT = 64.
Ndishes_SKA = 133.
dish_diam = (Ndishes_MeerKAT*dish_diam_MeerKat+ Ndishes_SKA*dish_diam_SKA)/(Ndishes_MeerKAT+Ndishes_SKA) # m (effective)
Omega_sur     = 20000   # Survey area deg2
t_obs     = 10000. # hrs, observing time
Ndishes   = Ndishes_MeerKAT + Ndishes_SKA  # number of dishes
specs_dict = {'dish_diam': dish_diam,
			  'Omega_sur': Omega_sur, 't_obs': t_obs, 'Ndishes' : Ndishes}

##############################################################################


theta_FWMH = c_light*1e-6/nu_ch/float(dish_diam) #radians
theta_FWMH_max = c_light*1e-6/np.min(nu_ch)/float(dish_diam) #radians
beam_gauss =np.array( [hp.gauss_beam(theta_FWMH[i], lmax=lmax) for i in range(num_freq)])
beam_gauss_worst = hp.gauss_beam(theta_FWMH_max, lmax=lmax)

beam_to_worst=np.array([hp.gauss_beam(np.sqrt(theta_FWMH_max**2-theta_FWMH[i]**2),lmax) for i in range(num_freq)])

############################################################################
HI_maps_freq_beam_deconv=np.array([convolve(HI_maps_freq[i],beam_to_worst[i], lmax=lmax) for i in range(num_freq)])
fg_maps_freq_beam_deconv=np.array([convolve(fg_maps_freq[i],beam_to_worst[i], lmax=lmax) for i in range(num_freq)])
full_maps_freq_beam_deconv=np.array([convolve(full_maps_freq[i],beam_to_worst[i], lmax=lmax) for i in range(num_freq)])


#HI_maps_freq_beam_deconv=np.zeros((num_freq,npix))
#fg_maps_freq_beam_deconv=np.zeros((num_freq,npix))
#full_maps_freq_beam_deconv=np.zeros((num_freq,npix))
#
#length_of_alms=hp.Alm.getsize(lmax)
#for n in range(num_freq):
#	alm_HI_ms = hp.sphtfunc.map2alm(HI_maps_freq[n])
#	alm_HI_rs=np.zeros(length_of_alms,dtype=complex)
#
#	alm_fg_ms = hp.sphtfunc.map2alm(fg_maps_freq[n])
#	alm_fg_rs=np.zeros(length_of_alms,dtype=complex)
#
#	alm_tot_ms = hp.sphtfunc.map2alm(full_maps_freq[n])
#	alm_tot_rs=np.zeros(length_of_alms,dtype=complex)
#
#	counter=0
#	for m in range(lmax+1):
#			for l in range(m,lmax+1):
#					alm_HI_rs[counter]=alm_HI_ms[counter]*(beam_gauss_worst[l]/beam_gauss[n][l])
#					alm_fg_rs[counter]=alm_fg_ms[counter]*(beam_gauss_worst[l]/beam_gauss[n][l])
#					alm_tot_rs[counter]=alm_tot_ms[counter]*(beam_gauss_worst[l]/beam_gauss[n][l])
#					counter+=1
#
#	HI_maps_freq_beam_deconv[n] = hp.alm2map(alms=alm_HI_rs, nside=nside,inplace=False)
#	fg_maps_freq_beam_deconv[n] = hp.alm2map(alms=alm_fg_rs, nside=nside,inplace=False)
#	full_maps_freq_beam_deconv[n] = hp.alm2map(alms=alm_tot_rs, nside=nside,inplace=False)


del HI_maps_freq; del fg_maps_freq; del full_maps_freq#; del alm_fg_ms; del alm_HI_rs; del alm_fg_rs; del alm_tot_rs


hp.mollview(HI_maps_freq_beam_deconv[21], cmap='viridis', min=0, max=1)
#######################################################################################

pix_mask = hp.query_strip(nside, theta1=np.pi*2/3, theta2=np.pi/3)
print(pix_mask)
mask_50 = np.zeros(npix)
mask_50[pix_mask] =1
fsky_50 = np.sum(mask_50)/hp.nside2npix(nside)
######################################################################

bad_v = np.where(mask_50==0)

HI_maps_freq_mask = copy.deepcopy(HI_maps_freq_beam_deconv)
fg_maps_freq_mask = copy.deepcopy(fg_maps_freq_beam_deconv)
full_maps_freq_mask = copy.deepcopy(full_maps_freq_beam_deconv)


for n in range(num_freq):
		HI_maps_freq_mask[n][bad_v] =  hp.UNSEEN
		HI_maps_freq_mask[n]=hp.remove_dipole(HI_maps_freq_mask[n])

		fg_maps_freq_mask[n][bad_v] =  hp.UNSEEN
		fg_maps_freq_mask[n]=hp.remove_dipole(fg_maps_freq_mask[n])

		full_maps_freq_mask[n][bad_v] =  hp.UNSEEN
		full_maps_freq_mask[n]=hp.remove_dipole(full_maps_freq_mask[n])



hp.mollview(HI_maps_freq_mask[0], title='freq 0',cmap='viridis' )
hp.mollview(HI_maps_freq_mask[-1], title='freq -1',cmap='viridis' )
plt.show()

#########################################################################################


ich = 21

######################################################################################################


full_maps_freq_masked=ma.zeros((num_freq,npix))

maskt =np.zeros(mask_50.shape)
maskt[bad_v]=  1
mask = ma.make_mask(maskt, shrink=False)


for n in range(num_freq):
	full_maps_freq_masked[n]  =ma.MaskedArray(full_maps_freq_mask[n], mask=mask)#np.isnan(full_maps_freq_mask[n])

Cov_channels=ma.cov(full_maps_freq_masked)


eigenval, eigenvec= np.linalg.eig(Cov_channels)



fig= plt.figure(figsize=(7,4))
plt.semilogy(np.arange(1,num_freq+1),eigenval,'--.',mfc='none',markersize=10, label='mask unseen')

x_ticks = np.arange(-10, num_freq+1, 10 )
ax = plt.gca()
ax.set(xlim=[-10,num_freq+2],xticks=x_ticks,xlabel="eigenvalue number",ylabel="$\\lambda$",title='Eigenvalues')
plt.legend()

#############################################################################
############################# PCA ##########################################

eigenvec_fg_Nfg = eigenvec[:, 0:num_sources]

fig=plt.figure()
plt.suptitle(f'Mixing matrix, Nfg:{num_sources}, sources: {fg_components},\nbeam: {beam_s}, fsky:{fsky_50:0.2f}')
plt.imshow(eigenvec_fg_Nfg, cmap='crest')
plt.xlabel('[MHz]')
plt.ylabel('[MHz]')
plt.colorbar()
#plt.show()


del eigenvec

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
#plt.show()
################################################################################################

#Foreground's maps from PCA

print(eigenvec_fg_Nfg.shape,eigenvec_fg_Nfg.T.shape,  full_maps_freq_masked.shape)

res_fg_maps=ma.dot(eigenvec_fg_Nfg,ma.dot(eigenvec_fg_Nfg.T,full_maps_freq_mask))

#The foreground residual that leaks into the recovered signal and noise
fg_leakage = fg_maps_freq_mask - ma.dot(eigenvec_fg_Nfg,ma.dot(eigenvec_fg_Nfg.T,fg_maps_freq_mask))
fg_leakage_noise = (fg_maps_freq_mask + noise)- ma.dot(eigenvec_fg_Nfg,ma.dot(eigenvec_fg_Nfg.T,fg_maps_freq_mask+noise))

HI_leakage = ma.dot(eigenvec_fg_Nfg,ma.dot(eigenvec_fg_Nfg.T,HI_maps_freq_mask))
fg_leakage[:,bad_v]=hp.UNSEEN
HI_leakage[:,bad_v] = hp.UNSEEN

del eigenvec_fg_Nfg

res_HI=np.zeros((num_freq,npix))
res_HI = full_maps_freq_mask - res_fg_maps

res_HI[:,bad_v]=hp.UNSEEN


res_HI_mask_0 = copy.deepcopy(res_HI)

######################################################################################################

HI_maps_freq_mask.dump(out_dir+f'cosmo_HI_noise_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax}_nside{nside}.npy')
res_HI.dump(out_dir+f'res_PCA_HI_noise_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax}_nside{nside}.npy')
fg_maps_freq_mask.dump(out_dir+f'fg_input_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax}_nside{nside}.npy')

##########################################################################################################

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(np.abs(res_fg_maps[ich]/fg_maps_freq_mask[ich]-1)*100,cmap='viridis', min=0, max=0.2, title=f'%(Res_fg/x_fg - 1), channel:{nu_ch[ich]}',unit='%' ,hold=True)
fig.add_subplot(222) 
hp.mollview(HI_maps_freq_mask[ich]-file['maps_sims_HI'][ich], cmap='viridis', title=f'HI signal + noise - HI freq={nu_ch[ich]}',hold=True)#min=0, max =1,
fig.add_subplot(223)
hp.mollview(res_HI[ich], title=f'PCA HI + noise freq={nu_ch[ich]}',cmap='viridis', hold=True)
#plt.show()

del file

fig=plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, lmax:{lmax}, Nfg:{num_sources}',fontsize=20)
fig.add_subplot(131) 
hp.gnomview(HI_maps_freq_mask[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='Input HI', cmap='viridis', hold=True)
fig.add_subplot(132) 
hp.gnomview(res_HI[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='PCA HI+noise', cmap= 'viridis', hold=True)
fig.add_subplot(133) 
hp.gnomview(HI_maps_freq_mask[ich]-res_HI[ich], rot=[-22,21],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-0.2, max=0.2, title='PCA residuals', cmap= 'viridis', hold=True)
#plt.tight_layout()
#plt.show()

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221)
hp.mollview(HI_maps_freq_mask[ich]-res_HI[ich], title=f'PCA residuals freq={nu_ch[ich]}', min=0, max=0.5,cmap='viridis', hold=True)
fig.add_subplot(222)
hp.mollview(fg_leakage[ich], title=f'Foreground leakage freq={nu_ch[ich]}', min=0, max=0.5,cmap='viridis', hold=True)
fig.add_subplot(223)
hp.mollview(HI_leakage[ich], title=f'HI leakage freq={nu_ch[ich]}', min=0, max=0.5,cmap='viridis', hold=True)
#plt.show()

###############################################################################################################
out_dir_cl = out_dir+'power_spectra_cls_from_healpix_maps/'
if not os.path.exists(out_dir_cl):
	os.makedirs(out_dir_cl)

lmax_cl = 2*nside

cl_HI_cosmo_full = np.zeros((num_freq, lmax_cl+1))
cl_Hi=np.zeros((num_freq, lmax_cl+1))
#cl_Hi_mask_0=np.zeros((num_freq, lmax_cl+1))
cl_Hi_recons_Nfg=np.zeros((num_freq, lmax_cl+1))
#cl_Hi_recons_Nfg_mask_0=np.zeros((num_freq, lmax_cl+1))
cl_fg_leak_Nfg=np.zeros((num_freq, lmax_cl+1))
cl_HI_leak_Nfg=np.zeros((num_freq, lmax_cl+1))

for i in range(num_freq):
    cl_Hi[i] = hp.anafast(HI_maps_freq_mask[i], lmax=lmax_cl)
    cl_HI_cosmo_full[i] = hp.anafast(HI_maps_freq_beam_deconv[i], lmax=lmax_cl)
    cl_Hi_recons_Nfg[i] = hp.anafast(res_HI[i], lmax=lmax_cl)
    cl_fg_leak_Nfg[i]=hp.anafast(fg_leakage[i], lmax=lmax_cl)
    cl_HI_leak_Nfg[i]=hp.anafast(HI_leakage[i], lmax=lmax_cl)


ell = np.arange(0, lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)


fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Channel:{nu_ch[ich]} MHz, BEAM {beam_s}, lmax:{lmax}, Nfg:{num_sources}, fsky:{fsky_50}')
plt.semilogy(ell[2:], factor[2:]*cl_HI_cosmo_full[ich][2:],'k--',mfc='none', label='Cosmo HI+noise full sky')
plt.semilogy(ell[2:], factor[2:]*cl_Hi[ich][2:],mfc='none', label='Cosmo HI+noise fsky')
plt.semilogy(ell[2:], factor[2:]*cl_Hi_recons_Nfg[ich][2:],'+',color=c_pal[1],mfc='none', label='PCA HI+noise fsky mask UNSEEN')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi}C_{\ell}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 10))


diff_cl_pca_cosmo = cl_Hi_recons_Nfg/cl_Hi -1 

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_pca_cosmo[ich][2:]*100, color=c_pal[1],label='mask UNSEEN')
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
plt.title(f'Mean over channel, BEAM {beam_s}, lmax:{lmax}, Nfg:{num_sources}, fsky:{fsky_50}')
plt.semilogy(ell[2:], factor[2:]*np.mean(cl_HI_cosmo_full, axis=0)[2:],'k--',mfc='none', label='Cosmo HI+noise full sky')
plt.semilogy(ell[2:], factor[2:]*np.mean(cl_Hi, axis=0)[2:],mfc='none', label='Cosmo HI+noise')
plt.semilogy(ell[2:], factor[2:]*np.mean(cl_Hi_recons_Nfg, axis=0)[2:],'+',mfc='none', label='PCA HI+noise')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\langle \frac{\ell(\ell+1)}{2\pi}C_{\ell} \rangle $')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 10))


frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], np.mean(diff_cl_pca_cosmo, axis=0)[2:]*100)
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



####################################################################################
##### confronto maschera non maschera - maschera deconvolta #######################

#### deconvoluzione
#f_0_mask = nm.NmtField(mask_50,[res_HI[0]])#, masked_on_input=True )
b = nm.NmtBin.from_nside_linear(nside, 8)
ell_mask= b.get_effective_ells()


cl_PCA_HI_mask_deconv = np.zeros((num_freq, len(ell_mask)))
cl_PCA_HI_mask_deconv_interp = np.zeros((num_freq, lmax_cl+1))

cl_cosmo_HI_mask_deconv = np.zeros((num_freq, len(ell_mask)))
cl_cosmo_HI_mask_deconv_interp = np.zeros((num_freq, lmax_cl+1))


cl_leak_HI_mask_deconv = np.zeros((num_freq, len(ell_mask)))
cl_leak_HI_mask_deconv_interp = np.zeros((num_freq, lmax_cl+1))

cl_leak_fg_mask_deconv = np.zeros((num_freq, len(ell_mask)))
cl_leak_fg_mask_deconv_interp = np.zeros((num_freq, lmax_cl+1))

cl_leak_fg_noise_mask_deconv = np.zeros((num_freq, len(ell_mask)))
cl_leak_fg_noise_mask_deconv_interp = np.zeros((num_freq, lmax_cl+1))


#cl_PCA_HI_mask_0_deconv = np.zeros((num_freq, len(ell_mask)))
#cl_PCA_HI_mask_0_deconv_interp = np.zeros((num_freq, lmax_cl+1))

for n in range(num_freq):
    f_0_mask = nm.NmtField(mask_50,[res_HI[n]])#, masked_on_input=True )
    cl_PCA_HI_mask_deconv[n] = nm.compute_full_master(f_0_mask, f_0_mask, b)[0]
    cl_PCA_HI_mask_deconv_interp[n] = np.interp(ell, ell_mask, cl_PCA_HI_mask_deconv[n])
    
    f_0_cosmo_mask = nm.NmtField(mask_50,[HI_maps_freq_mask[n]] ) #qua
    cl_cosmo_HI_mask_deconv[n] = nm.compute_full_master(f_0_cosmo_mask, f_0_cosmo_mask, b)[0]
    cl_cosmo_HI_mask_deconv_interp[n] = np.interp(ell, ell_mask, cl_cosmo_HI_mask_deconv[n])

    f_0_leak_HI_mask = nm.NmtField(mask_50,[HI_leakage[n]] ) #qua
    cl_leak_HI_mask_deconv[n] = nm.compute_full_master(f_0_leak_HI_mask, f_0_leak_HI_mask, b)[0]
    cl_leak_HI_mask_deconv_interp[n] = np.interp(ell, ell_mask, cl_leak_HI_mask_deconv[n])

    f_0_leak_fg_mask = nm.NmtField(mask_50,[fg_leakage[n]] ) #qua
    cl_leak_fg_mask_deconv[n] = nm.compute_full_master(f_0_leak_fg_mask, f_0_leak_fg_mask, b)[0]
    cl_leak_fg_mask_deconv_interp[n] = np.interp(ell, ell_mask, cl_leak_fg_mask_deconv[n])

    f_0_leak_fg_noise_mask = nm.NmtField(mask_50,[fg_leakage_noise[n]] ) #qua
    cl_leak_fg_noise_mask_deconv[n] = nm.compute_full_master(f_0_leak_fg_noise_mask, f_0_leak_fg_noise_mask, b)[0]
    cl_leak_fg_noise_mask_deconv_interp[n] = np.interp(ell, ell_mask, cl_leak_fg_noise_mask_deconv[n])


del fg_leakage; del HI_leakage

np.savetxt(out_dir_cl+f'cl_PCA_HI_noise_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_PCA_HI_mask_deconv)

np.savetxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_PCA_HI_mask_deconv_interp)

np.savetxt(out_dir_cl+f'cl_deconv_cosmo_HI_noise_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_cosmo_HI_mask_deconv_interp)

np.savetxt(out_dir_cl+f'cl_deconv_leak_HI_noise_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_leak_HI_mask_deconv_interp)
np.savetxt(out_dir_cl+f'cl_deconv_leak_fg_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_leak_fg_mask_deconv_interp)
np.savetxt(out_dir_cl+f'cl_deconv_leak_fg_noise_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_leak_fg_noise_mask_deconv_interp)


ell=np.arange(lmax_cl+1)
factor=ell*(ell+1)/(2*np.pi)


fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'CLs: channel:{nu_ch[ich]} MHzlmax:{lmax}, Nfg:{num_sources}')
plt.plot(ell[2:],factor[2:]*cl_cosmo_HI_mask_deconv_interp[ich][2:], label='Cosmo HI + noise')
plt.plot(ell[2:],factor[2:]*cl_PCA_HI_mask_deconv_interp[ich][2:],'+', mfc='none', label='PCA HI + noise')
plt.ylabel(r'$\frac{\ell(\ell+1)}{2\pi}  C_{\ell} $')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} e C_{\ell} $')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 30))

diff_cl_need2sphe = cl_PCA_HI_mask_deconv_interp/cl_cosmo_HI_mask_deconv_interp-1
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
#plt.savefig(out_dir_plot+f'cls_need2pix_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.png')
#plt.show()


fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: mean over channels, lmax:{lmax}, Nfg:{num_sources}')
plt.plot(ell[2:], factor[2:]*cl_cosmo_HI_mask_deconv_interp.mean(axis=0)[2:], label = f'Cosmo HI + noise')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_mask_deconv_interp.mean(axis=0)[2:],'+',mfc='none', label = f'PCA HI + noise')
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
#plt.savefig(out_dir_plot+f'cls_need2pix_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.png')

#plt.show()

################

cl_PCA_HI_mask_deconv_interp_18=np.loadtxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg18_lmax{lmax_cl}_nside{nside}.dat')
fig = plt.figure(figsize=(10,7))
plt.title(f'NEEDLETS CLs: mean over channels, lmax:{lmax}, Nfg:{num_sources}')
plt.plot(ell[2:], factor[2:]*cl_cosmo_HI_mask_deconv_interp.mean(axis=0)[2:], label = f'Cosmo HI + noise')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_mask_deconv_interp.mean(axis=0)[2:],'+',mfc='none', label = f'PCA HI + noise Nfg=3')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_mask_deconv_interp_18.mean(axis=0)[2:],'+',mfc='none', label = f'PCA HI + noise Nfg=18')
plt.xlim([0,200])
plt.legend()

plt.show()