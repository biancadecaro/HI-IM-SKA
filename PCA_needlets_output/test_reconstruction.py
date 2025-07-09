import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from needlets_analysis import analysis, theory
import os
import pickle
import copy
import pymaster as nm
import seaborn as sns
sns.set_theme(style = 'white')
from matplotlib import colors
import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

#print(sns.color_palette("husl", 15).as_hex())
sns.palettes.color_palette()
c_pal = sns.color_palette().as_hex()
import cython_mylibc as pippo

#mpl.rcParams['font.size']=18
##########################################################################################
plot_dir = 'test_reconstruction/'
if not os.path.exists(plot_dir):
	os.makedirs(plot_dir)

beam_s = 'SKA_AA4'
fg_comp = 'synch_ff_ps'
path_data_sims_tot = f'../Sims/nuovo_beam_{beam_s}_sims_{fg_comp}_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax383_nside128'
path_data_sims_tot_no_beam = f'../Sims/nuovo_sims_{fg_comp}_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax383_nside128'

with open(path_data_sims_tot+'.pkl', 'rb') as f:
	file = pickle.load(f)
	f.close()
nu_ch= file['freq']

with open(path_data_sims_tot_no_beam+'.pkl', 'rb') as ff:
	ffile = pickle.load(ff)
	ff.close()

HI_maps_freq_no_noise_no_beam = ffile['maps_sims_HI']

HI_maps_freq = file['maps_sims_HI'] + file['maps_sims_noise']

del file; del ffile

num_ch=len(nu_ch)
min_ch = min(nu_ch)
max_ch = max(nu_ch)
nside=hp.get_nside(HI_maps_freq[0])
npix= hp.nside2npix(nside)


HI_maps_freq_no_noise_no_beam = np.array([HI_maps_freq_no_noise_no_beam[i] -np.mean(HI_maps_freq_no_noise_no_beam[i],axis=0)  for i in range(num_ch)])
HI_maps_freq = np.array([HI_maps_freq[i] -np.mean(HI_maps_freq[i],axis=0)  for i in range(num_ch)])


out_dir_maps_recon = './'
if not os.path.exists(out_dir_maps_recon):
		os.makedirs(out_dir_maps_recon)
		

jmax=4
lmax= 3*nside-1
if fg_comp=='synch_ff_ps':
	Nfg=3
if fg_comp=='synch_ff_ps_pol':
	Nfg=3
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)

##################################################


pix_mask = hp.query_strip(nside, theta1=np.pi*2/3, theta2=np.pi/3)
print(pix_mask)
mask_50 = np.zeros(npix)
mask_50[pix_mask] =1
fsky_50 = np.sum(mask_50)/hp.nside2npix(nside)

bad_v = np.where(mask_50==0)

HI_maps_freq_mask = copy.deepcopy(HI_maps_freq)

for n in range(num_ch):
		HI_maps_freq_mask[n][bad_v] =  hp.UNSEEN
		HI_maps_freq_mask[n]=hp.remove_dipole(HI_maps_freq_mask[n])



j_test=2
ich=int(num_ch/2)

###################################################################

need_theory=theory.NeedletTheory(B,jmax, lmax)

b_values = pippo.mylibpy_needlets_std_init_b_values(B,jmax,lmax)

#need_analysis_HI = analysis.NeedAnalysis(jmax, lmax, out_dir_maps_recon, HI_maps_freq_no_noise_no_beam)
#
#fname_HI=f'test_bjk_maps_HI_noise_{beam_s}_jmax{jmax}_lmax{lmax}_nside{nside}'
#
#betajk_HI = np.zeros((num_ch, jmax+1, npix))
#
#for nu in range(num_ch):        
#	betajk_HI[nu] = pippo.mylibpy_needlets_f2betajk_healpix_harmonic(HI_maps_freq_no_noise_no_beam[nu], B, jmax,lmax )
#
#
#
#fig = plt.figure(figsize=(10, 7))
#hp.mollview(betajk_HI[ich, j_test], cmap='viridis', title=f'HI, j={j_test}, freq={nu_ch[ich]}', hold=True)
#
####################################################################
#map_recons_HI=np.zeros((len(nu_ch), npix))
#
#for nu in range(len(nu_ch)):
#	for j in range(betajk_HI.shape[1]):
#		map_recons_HI[nu] += pippo.mylibpy_needlets_f2betajk_j_healpix_harmonic(betajk_HI[nu,j],b_values,j)
#
#del betajk_HI
#
##############################################################################
#
#fig=plt.figure(figsize=(10, 7))
#fig.suptitle(f'channel: {nu_ch[ich]} MHz, BEAM {beam_s}, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
#fig.add_subplot(311)
#hp.mollview(HI_maps_freq_no_noise_no_beam[ich], min=0, max=1, title='Input HI+ noise', cmap='viridis', hold=True)
#fig.add_subplot(312) 
#hp.mollview(map_recons_HI[ich], min=0, max=1, title='Need recons HI + noise', cmap= 'viridis', hold=True)
#fig.add_subplot(313) 
#hp.mollview(100*(map_recons_HI[ich]/HI_maps_freq_no_noise_no_beam[ich]-1), min=-0.2, max=0.2, title='% Need recons HI/HI -1', cmap= 'viridis', hold=True)
#
########################################################################################
#
lmax_cl= 2*nside
#
ell_cl = np.arange(lmax_cl+1)
#
#cl_cosmo_HI_recons = np.zeros((len(nu_ch), lmax_cl+1))
#cl_cosmo_HI = np.zeros((len(nu_ch), lmax_cl+1))
#
#for n in range(len(nu_ch)):
#	cl_cosmo_HI_recons[n] = hp.anafast(map_recons_HI[n], lmax=lmax_cl)
#	cl_cosmo_HI[n]=hp.anafast(HI_maps_freq_no_noise_no_beam[n], lmax=lmax_cl)
#	
#diff = cl_cosmo_HI_recons/cl_cosmo_HI-1
#
#fig = plt.figure()
#plt.title('Cl jmax 4 lmax 383 nside 128, full sky')
#plt.plot(ell_cl[2:], diff.mean(axis=0)[2:]*100)
#plt.axhline(y=0, c='k', ls='--', alpha=0.3)
#plt.xlabel('ell')
#plt.ylabel('recons/cosmo -1 %')
#plt.savefig(plot_dir+'diff_cl_recons_HI_jmax4_lmax383_nside128.png')
#plt.show()
#
#del map_recons_HI; del HI_maps_freq; del cl_cosmo_HI_recons; del cl_cosmo_HI; del diff
del HI_maps_freq_no_noise_no_beam
###################################################
################### MASK ######################

need_analysis_HI_mask = analysis.NeedAnalysis(jmax, lmax, out_dir_maps_recon, HI_maps_freq_mask)

fname_HI=f'test_bjk_maps_HI_noise_{beam_s}_jmax{jmax}_lmax{lmax}_nside{nside}'

betajk_HI_mask = np.zeros((num_ch, jmax+1, npix))

for nu in range(num_ch):        
	betajk_HI_mask[nu] = pippo.mylibpy_needlets_f2betajk_healpix_harmonic(HI_maps_freq_mask[nu], B, jmax,lmax )

#betajk_HI_mask[:,:, bad_v] = 0#hp.UNSEEN


path_cosmo_HI_bjk = f'../Maps_needlets_nuovo_1/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/bjk_maps_HI_noise_{num_ch}freq_{min_ch:1.1f}_{max_ch:1.1f}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}'
cosmo_HI_bjk = np.load(path_cosmo_HI_bjk+'.npy')#[:,:jmax,:]
#

hp.mollview(betajk_HI_mask[ich, j_test] -cosmo_HI_bjk[ich, j_test] , cmap='viridis', title=f'betajk nuovo-vecchio, j={j_test}, freq={nu_ch[ich]}', hold=True)
plt.show()
del cosmo_HI_bjk
###################################################################
map_recons_HI_mask=np.zeros((len(nu_ch), npix))

for nu in range(len(nu_ch)):
	#for j in range(betajk_HI_mask.shape[1]):
	#	map_recons_HI_mask[nu] += pippo.mylibpy_needlets_f2betajk_j_healpix_harmonic(betajk_HI_mask[nu,j],b_values,j)
	map_recons_HI_mask[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(betajk_HI_mask[nu], B, lmax )

del betajk_HI_mask
map_recons_HI_mask[:,bad_v]=hp.UNSEEN		

b = nm.NmtBin.from_nside_linear(nside, 8)
ell_mask= b.get_effective_ells()

cl_cosmo_HI_mask_deconv = np.zeros((num_ch, len(ell_mask)))
cl_cosmo_HI_mask_deconv_interp = np.zeros((num_ch, lmax_cl+1))

cl_recons_HI_mask_deconv = np.zeros((num_ch, len(ell_mask)))
cl_recons_HI_mask_deconv_interp = np.zeros((num_ch, lmax_cl+1))

for n in range(num_ch):
	f_0_cosmo_mask = nm.NmtField(mask_50,[HI_maps_freq_mask[n]] )
	cl_cosmo_HI_mask_deconv[n] = nm.compute_full_master(f_0_cosmo_mask, f_0_cosmo_mask, b)[0]
	cl_cosmo_HI_mask_deconv_interp[n] = np.interp(ell_cl, ell_mask, cl_cosmo_HI_mask_deconv[n])

	f_0_mask = nm.NmtField(mask_50,[map_recons_HI_mask[n]] )
	cl_recons_HI_mask_deconv[n] = nm.compute_full_master(f_0_mask, f_0_mask, b)[0]
	cl_recons_HI_mask_deconv_interp[n] = np.interp(ell_cl, ell_mask, cl_recons_HI_mask_deconv[n])

hp.mollview(HI_maps_freq_mask[ich],title='cosmo', cmap='viridis')
hp.mollview(map_recons_HI_mask[ich],title='recons', cmap='viridis')

#out_dir_maps_recon = f'maps_reconstructed_nuovo/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
#out_dir_cl = out_dir_maps_recon+'cls_recons_need/'
#cl_recons_HI_mask_deconv_interp_vecchio=np.loadtxt(out_dir_cl+f'cl_deconv_cosmo_recon_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{lmax_cl}_nside{nside}.dat')
#cl_cosmo_HI_mask_deconv_interp_vecchio=np.loadtxt(out_dir_cl+f'cl_deconv_cosmo_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')

diff_deconv = cl_recons_HI_mask_deconv_interp/cl_cosmo_HI_mask_deconv_interp-1
#diff_deconv_vecchio = cl_recons_HI_mask_deconv_interp_vecchio/cl_cosmo_HI_mask_deconv_interp_vecchio-1

fig = plt.figure(figsize=(11,7))
plt.suptitle('Cl jmax 4, lmax 256, nside 128, mask 50%, noise, gaussian beam')
plt.plot(ell_cl[2:], diff_deconv.mean(axis=0)[2:]*100)
#plt.plot(ell_cl[2:], diff_deconv_vecchio.mean(axis=0)[2:]*100, label='vecchio')
plt.axhline(y=0, c='k', ls='--', alpha=0.3)
plt.xlabel('ell')
plt.ylabel('diff %')
#plt.legend()
plt.savefig(plot_dir+'diff_cl_recons_HI_mask_beam_noise_jmax4_lmax383_nside128.png')

plt.show()


del f_0_cosmo_mask;del f_0_mask
############################################
dir_PCA = f'PCA_maps_nuovo/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'# noise_mask0.39
out_dir_maps_recon = f'maps_reconstructed_nuovo_1/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
map_input_HI_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_cosmo_HI_noise_{num_ch}_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy', allow_pickle=True)
map_input_HI_need2pix[:, bad_v]=hp.UNSEEN


path_cosmo_HI = f'../PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/cosmo_HI_noise_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz_lmax{lmax}_nside{nside}'
cosmo_HI = np.load(path_cosmo_HI+'.npy', allow_pickle=True)

hp.mollview(map_input_HI_need2pix[ich]-map_recons_HI_mask[ich], title='diff recons vecchia nuova', cmap='viridis')

hp.mollview(map_input_HI_need2pix[ich], title= 'reconstructed vecchio', cmap='viridis')
hp.mollview(cosmo_HI[ich], title= 'cosmo vecchio', cmap='viridis')

cl_recons_HI_mask_deconv_1 = np.zeros((num_ch, len(ell_mask)))
cl_recons_HI_mask_deconv_1_interp = np.zeros((num_ch, lmax_cl+1)) 

cl_cosmo_HI_mask_deconv_1 = np.zeros((num_ch, len(ell_mask)))
cl_cosmo_HI_mask_deconv_1_interp = np.zeros((num_ch, lmax_cl+1)) 

for n in range(num_ch):
	f_0_mask = nm.NmtField(mask_50,[map_input_HI_need2pix[n]] )
	cl_recons_HI_mask_deconv_1[n] = nm.compute_full_master(f_0_mask, f_0_mask, b)[0]
	cl_recons_HI_mask_deconv_1_interp[n] = np.interp(ell_cl, ell_mask, cl_recons_HI_mask_deconv_1[n])

	f_0_cosmo_mask = nm.NmtField(mask_50,[HI_maps_freq_mask[n]] )
	cl_cosmo_HI_mask_deconv_1[n] = nm.compute_full_master(f_0_cosmo_mask, f_0_cosmo_mask, b)[0]
	cl_cosmo_HI_mask_deconv_1_interp[n] = np.interp(ell_cl, ell_mask, 	cl_cosmo_HI_mask_deconv_1[n])


fig = plt.figure()
plt.suptitle('Deconvolution')
plt.plot(ell_cl[2:], 100*((cl_recons_HI_mask_deconv_interp/cl_cosmo_HI_mask_deconv_interp)[ich][2:]-1),color=c_pal[0], label= 'nuovo' )
plt.plot(ell_cl[2:], 100*((cl_recons_HI_mask_deconv_1_interp/cl_cosmo_HI_mask_deconv_1_interp)[ich][2:]-1),color=c_pal[1], label= 'vecchio nuovo' )
#plt.plot(ell_cl[2:], (cl_recons_HI_mask_deconv_interp_vecchio/cl_cosmo_HI_mask_deconv_interp_vecchio)[ich][2:]-1,color=c_pal[2], label='vecchio')


plt.axhline(y=0, c='k', ls='--', alpha=0.3)
plt.xlabel('ell')
plt.ylabel('diff %')
plt.legend()
plt.show()


plt.show()