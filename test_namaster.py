import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pymaster as nm
import copy
###############################

nside=256
npix = hp.nside2npix(nside)
lmax=3*nside-1

###################################

pix_mask = hp.query_strip(nside, theta2=np.pi*2/3, theta1=np.pi/3)

mask_50 = np.ones(npix)
mask_50[pix_mask] =0
fsky_50 = np.mean(mask_50)

#############################
beam_s = 'SKA_AA4'
fg_components='synch_ff_ps'
path_data_sims_tot = f'Sims/beam_{beam_s}_no_mean_sims_{fg_components}_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax767_nside256'

with open(path_data_sims_tot+'.pkl', 'rb') as f:
		file = pickle.load(f)
		f.close()

nu_ch= file['freq']

num_freq = len(nu_ch)

nu0 =1420
print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')
print(f'corresponding to the redshift range z: [{min(nu0/nu_ch -1.0):.2f} - {max(nu0/nu_ch -1.0):.2f}] ')

HI_maps_freq = file['maps_sims_HI'] + file['maps_sims_noise']

del file
###############################################################
HI_maps_freq_mask = copy.deepcopy(HI_maps_freq)
bad_v = np.where(mask_50==0)
for n in range(num_freq):
		HI_maps_freq_mask[n][bad_v] = hp.UNSEEN
		HI_maps_freq_mask[n]        = hp.remove_dipole(HI_maps_freq_mask[n])

#######################################################################
import pickle
with open(f'PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/res_PCA_HI_noise_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg3_lmax{lmax}_nside{nside}.npy', 'rb') as f:
	res_HI = pickle.load(f)
	f.close()


#cl_PCA_deconv = np.loadtxt(f'PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_PCA_HI_noise_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg3_lmax512_nside{nside}.dat')

lmax_cl = 2*nside
ell = np.arange(lmax_cl+1)
factor_ell = ell*(ell+1)/(2.*np.pi)

delta_ell = 2

b = nm.NmtBin.from_nside_linear(nside, delta_ell)
ell_mask= b.get_effective_ells()

b_8 = nm.NmtBin.from_nside_linear(nside, 8)
ell_mask_8= b_8.get_effective_ells()

factor = ell_mask*(ell_mask+1)/(2.*np.pi)
factor_8 = ell_mask_8*(ell_mask_8+1)/(2.*np.pi)

print(ell_mask, (lmax+1)/delta_ell)

cl_HI_mask_deconv = np.zeros((num_freq, len(ell_mask)))
cl_PCA_deconv = np.zeros((num_freq, len(ell_mask)))

cl_HI_mask_deconv_8 = np.zeros((num_freq, len(ell_mask_8)))
cl_PCA_deconv_8 = np.zeros((num_freq, len(ell_mask_8)))

cl_HI_mask_deconv_interp = np.zeros((num_freq, lmax_cl+1))
cl_PCA_HI_mask_deconv_interp = np.zeros((num_freq, lmax_cl+1))

cl_HI_mask_deconv_interp_8 = np.zeros((num_freq, lmax_cl+1))
cl_PCA_HI_mask_deconv_interp_8 = np.zeros((num_freq, lmax_cl+1))

for n in range(num_freq):
	f_0_mask = nm.NmtField(mask_50,[HI_maps_freq_mask[n]] )
	cl_HI_mask_deconv[n] = nm.compute_full_master(f_0_mask, f_0_mask, b)[0]
	cl_HI_mask_deconv_interp[n] = np.interp(ell, ell_mask, cl_HI_mask_deconv[n])
	

	cl_HI_mask_deconv_8[n] = nm.compute_full_master(f_0_mask, f_0_mask, b_8)[0]
	cl_HI_mask_deconv_interp_8[n] = np.interp(ell, ell_mask_8, cl_HI_mask_deconv_8[n])

	f_0_PCA_mask = nm.NmtField(mask_50,[res_HI[n]] )
	cl_PCA_deconv[n] = nm.compute_full_master(f_0_PCA_mask, f_0_PCA_mask, b)[0]
	cl_PCA_deconv_8[n] = nm.compute_full_master(f_0_PCA_mask, f_0_PCA_mask, b_8)[0]

	cl_PCA_HI_mask_deconv_interp[n] = np.interp(ell, ell_mask, cl_PCA_deconv[n])
	cl_PCA_HI_mask_deconv_interp_8[n] = np.interp(ell, ell_mask_8, cl_PCA_deconv_8[n])

plt.figure()
plt.plot(ell_mask, factor*cl_HI_mask_deconv.mean(axis=0), label='cl cosmo')
plt.plot(ell_mask, factor*cl_PCA_deconv.mean(axis=0), label='cl PCA')
plt.plot(ell_mask_8, factor_8*cl_HI_mask_deconv_8.mean(axis=0), label='cl cosmo 8')
plt.plot(ell_mask_8, factor_8*cl_PCA_deconv_8.mean(axis=0), label='cl PCA 8')
plt.xlim(left=0,right=200)
plt.legend()
plt.ylabel(r'$\langle \frac{\ell(\ell+1)}{2\pi}C_{\ell} \rangle $')
plt.xlabel(r'$\ell$')

plt.figure()
plt.plot(ell_mask, 100*(cl_PCA_deconv/cl_HI_mask_deconv-1).mean(axis=0), label= '2')
plt.plot(ell_mask_8, 100*(cl_PCA_deconv_8/cl_HI_mask_deconv_8-1).mean(axis=0), label= '8')
plt.axhline(ls='--', c='k', alpha=0.7)
plt.xlim(left=0,right=200)
plt.ylim(bottom=-20, top=20)
plt.legend()
plt.ylabel(r'% diff')
plt.xlabel(r'$\ell$')


plt.figure()
plt.suptitle('interpolation')
plt.plot(ell, factor_ell*cl_HI_mask_deconv_interp.mean(axis=0), label='cl cosmo')
plt.plot(ell, factor_ell*cl_PCA_HI_mask_deconv_interp.mean(axis=0), label='cl PCA')

plt.plot(ell, factor_ell*cl_HI_mask_deconv_interp_8.mean(axis=0), label='cl cosmo 8')
plt.plot(ell, factor_ell*cl_PCA_HI_mask_deconv_interp_8.mean(axis=0), label='cl PCA 8')
plt.xlim(left=0,right=200)
plt.legend()
plt.ylabel(r'$\langle \frac{\ell(\ell+1)}{2\pi}C_{\ell} \rangle $')
plt.xlabel(r'$\ell$')

plt.figure()
plt.suptitle('interpolation')
plt.plot(ell, 100*(cl_PCA_HI_mask_deconv_interp/cl_HI_mask_deconv_interp-1).mean(axis=0), label= '2')
plt.plot(ell, 100*(cl_PCA_HI_mask_deconv_interp_8/cl_HI_mask_deconv_interp_8-1).mean(axis=0), label= '8')
plt.axhline(ls='--', c='k', alpha=0.7)
plt.xlim(left=0,right=200)
plt.ylim(bottom=-20, top=20)
plt.legend()
plt.ylabel(r'% diff')
plt.xlabel(r'$\ell$')
plt.show()