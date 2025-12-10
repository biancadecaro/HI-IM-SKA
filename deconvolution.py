import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy.ma as ma
import copy
import pymaster as nm
from convolution_func import create_bl_vec,createDtheta, single_convolution_bl, cos_beam

sns.set_theme(style = 'white')
sns.set_palette(palette='tab20',n_colors=15)
from matplotlib import colors
sns.palettes.color_palette()
c_pal = sns.color_palette().as_hex()

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)
###########################################################################
beam_s = 'cosine_Amp0.1_smooth_True_SKA_AA4'

################################################################

fg_components='synch_ff_ps_pol'

path_data_sims_tot = f'Sims/nuovo_beam_{beam_s}_sims_{fg_components}_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax383_nside128'

with open(path_data_sims_tot+'.pkl', 'rb') as f:
		file = pickle.load(f)
		f.close()

nu_ch= file['freq']

num_freq = len(nu_ch)
ich = int(num_freq/2)
nu0 =1420
print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')
print(f'corresponding to the redshift range z: [{min(nu0/nu_ch -1.0):.2f} - {max(nu0/nu_ch -1.0):.2f}] ')

HI_maps_freq = file['maps_sims_HI'] + file['maps_sims_noise']  #aggiungo il noise

HI_maps_freq = np.array([HI_maps_freq[i] -np.mean(HI_maps_freq[i],axis=0)  for i in range(num_freq)])

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


Amp=0.1
T_p = 20
smooth = True

##############################################################################

bl_beam_cos, delta_theta=create_bl_vec(beam='cosine', nside=nside,dish_diameter=dish_diam, T_p=T_p, Amp=Amp, smooth=smooth, ch_nu=nu_ch)
index, = np.where(delta_theta==delta_theta.max())
print(index, delta_theta[index])
bl_beam_cos_worst = bl_beam_cos[0]
delta_theta_worst = delta_theta[0]

theta_FWMH = c_light*1e-6/nu_ch/float(dish_diam) #radians
theta_FWMH_max = c_light*1e-6/np.min(nu_ch)/float(dish_diam) #radians
beam_gauss =np.array( [hp.gauss_beam(theta_FWMH[i], lmax=lmax) for i in range(num_freq)])
beam_gauss_worst = hp.gauss_beam(theta_FWMH_max, lmax=lmax)
############################################################################
HI_maps_freq_deconv_gauss=np.zeros((num_freq,npix))
HI_maps_freq_deconv=np.zeros((num_freq,npix))
length_of_alms=hp.Alm.getsize(lmax)
for n in range(num_freq):
	alm_ms = hp.sphtfunc.map2alm(HI_maps_freq[n])
	alm_ms_gauss = hp.sphtfunc.map2alm(HI_maps_freq[n])
	alm_rs=np.zeros(length_of_alms,dtype=complex)
	alm_rs_gauss=np.zeros(length_of_alms,dtype=complex)

	counter=0
	for m in range(lmax+1):
			for l in range(m,lmax+1):
					alm_rs[counter]=alm_ms[counter]*(bl_beam_cos_worst[l]/bl_beam_cos[n][l])
					alm_rs_gauss[counter]=alm_ms_gauss[counter]*(beam_gauss_worst[l]/beam_gauss[n][l])
					counter+=1

	HI_maps_freq_deconv[n] = hp.alm2map(alms=alm_rs, nside=nside,inplace=False)
	HI_maps_freq_deconv_gauss[n] = hp.alm2map(alms=alm_rs_gauss, nside=nside,inplace=False)

hp.mollview(HI_maps_freq_deconv[-1],min=0, max=1, cmap='viridis', title='cosine')
hp.mollview(HI_maps_freq_deconv_gauss[-1],min=0, max=1, cmap='viridis', title='gauss')
#plt.show()

############
lmax_cl = 2*nside
cl_HI_cos = np.zeros((num_freq, lmax_cl+1))
cl_HI_gauss = np.zeros((num_freq, lmax_cl+1))

for n in range(num_freq):
	cl_HI_cos[n] = hp.anafast(HI_maps_freq_deconv[n], lmax=lmax_cl)
	cl_HI_gauss[n] = hp.anafast(HI_maps_freq_deconv_gauss[n], lmax=lmax_cl)


ell = np.arange(lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)
fig = plt.figure()
plt.plot(ell,factor*np.mean(cl_HI_cos, axis=0), label='deconv w cosine')
plt.plot(ell,factor*np.mean(cl_HI_gauss, axis=0), label='deconv w gauss')
plt.xlabel('ell')
plt.ylabel('mean Cl')
plt.legend()
#plt.show()

#######################################################################



#####################################################################
################ effect of deconvolution ############################
cl_PCA_HI_Nfg3_deconv = np.loadtxt('PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_cosine_Amp0.1_smooth_True_SKA_AA4_noise_mask0.5_unseen_deconv/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')
cl_PCA_HI_Nfg4_deconv = np.loadtxt('PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_cosine_Amp0.1_smooth_True_SKA_AA4_noise_mask0.5_unseen_deconv/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg4_lmax256_nside128.dat')
cl_PCA_HI_Nfg5_deconv = np.loadtxt('PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_cosine_Amp0.1_smooth_True_SKA_AA4_noise_mask0.5_unseen_deconv/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg5_lmax256_nside128.dat')
cl_cosmo_deconv = np.loadtxt('PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_cosine_Amp0.1_smooth_True_SKA_AA4_noise_mask0.5_unseen_deconv/power_spectra_cls_from_healpix_maps/cl_deconv_cosmo_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')


cl_PCA_HI_Nfg3 = np.loadtxt('PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_cosine_Amp0.1_smooth_True_SKA_AA4_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')
cl_PCA_HI_Nfg4 = np.loadtxt('PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_cosine_Amp0.1_smooth_True_SKA_AA4_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg4_lmax256_nside128.dat')
cl_PCA_HI_Nfg5 = np.loadtxt('PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_cosine_Amp0.1_smooth_True_SKA_AA4_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg5_lmax256_nside128.dat')
cl_cosmo = np.loadtxt('PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_cosine_Amp0.1_smooth_True_SKA_AA4_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_cosmo_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')

diff_deconv_Nfg3 = cl_PCA_HI_Nfg3_deconv/cl_cosmo_deconv-1
diff_deconv_Nfg4 = cl_PCA_HI_Nfg4_deconv/cl_cosmo_deconv-1
diff_deconv_Nfg5 = cl_PCA_HI_Nfg5_deconv/cl_cosmo_deconv-1

diff_Nfg3 = cl_PCA_HI_Nfg3/cl_cosmo-1
diff_Nfg4 = cl_PCA_HI_Nfg4/cl_cosmo-1
diff_Nfg5 = cl_PCA_HI_Nfg5/cl_cosmo-1

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,4))

axs[0,0].plot(ell, np.mean(diff_deconv_Nfg3, axis=0), label= 'Cosine, exact deconv')
axs[0,0].plot(ell, np.mean(diff_Nfg3, axis=0), label= 'Cosine, no deconv')
axs[0,0].text( 0.5, 0.9, 'Nfg=3', transform=axs[0,0].transAxes)
axs[0,0].axhline(ls='--', c= 'k', alpha=0.3)
axs[0,0].set_xlim([15,200])
axs[0,0].set_ylim([-0.2,0.2])

axs[0,1].plot(ell, np.mean(diff_deconv_Nfg4, axis=0), label= 'Cosine, exact deconv')
axs[0,1].plot(ell, np.mean(diff_Nfg4, axis=0), label= 'Cosine, no deconv')
axs[0,1].text( 0.5, 0.9, 'Nfg=4', transform=axs[0,1].transAxes)
axs[0,1].axhline(ls='--', c= 'k', alpha=0.3)
axs[0,1].set_xlim([15,200])
axs[0,1].set_ylim([-0.2,0.2])

axs[1,0].plot(ell, np.mean(diff_deconv_Nfg5, axis=0), label= 'Cosine, exact deconv')
axs[1,0].plot(ell, np.mean(diff_Nfg5, axis=0), label= 'Cosine, no deconv')
axs[1,0].text( 0.5, 0.9, 'Nfg=5', transform=axs[1,0].transAxes)
axs[1,0].axhline(ls='--', c= 'k', alpha=0.3)
axs[1,0].set_xlim([15,200])
axs[1,0].set_ylim([-0.2,0.2])

axs[1,0].legend()
#plt.show()
plt.close('all')

###################################################
################# senza pol #######################

cl_cosmo = np.loadtxt('PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_cosine_Amp0.1_smooth_True_SKA_AA4_noise_mask0.5_unseen_deconv/power_spectra_cls_from_healpix_maps/cl_deconv_cosmo_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg4_lmax256_nside128.dat')
cl_PCA_no_pol_Nfg4 = np.loadtxt('PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_cosine_Amp0.1_smooth_True_SKA_AA4_noise_mask0.5_unseen_deconv/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg4_lmax256_nside128.dat')

diff_cl = cl_PCA_no_pol_Nfg4/cl_cosmo -1

lmin=3

fig = plt.figure()
plt.suptitle('Nfg=4, no pol, cosine beam with accurate deconvolution')
plt.plot(ell[lmin:], diff_cl.mean(axis=0)[lmin:], c=c_pal[0], label='PCA')
plt.axhline(ls='--', c= 'k', alpha=0.3)
plt.xlim([lmin,200])
plt.ylim([-0.2,0.2])
plt.ylabel(r'$ \langle \Delta\rangle_{\rm ch}$ [%]')
plt.xlabel(r'$\ell$')
plt.xticks(np.arange(10,200, 10))

plt.show()