import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import pymaster as nm
import os

import seaborn as sns
sns.set_theme()
sns.set_theme(style = 'white')
#sns.set_palette('husl',15)
from matplotlib import colors
import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

#print(sns.color_palette("husl", 15).as_hex())
sns.palettes.color_palette()
import cython_mylibc as pippo
##########################################################################################

beam_s = 'theta40arcmin'

out_dir_plot = 'Plots_PCA_needlets/'
dir_PCA = f'PCA_maps/No_mean/Beam_{beam_s}_noise_mask0.39_unseen/'# noise_mask0.39
out_dir_maps_recon = f'maps_reconstructed/No_mean/Beam_{beam_s}_noise_mask0.39_unseen/'
if not os.path.exists(out_dir_maps_recon):
		os.makedirs(out_dir_maps_recon)


fg_comp = 'synch_ff_ps'
beam = 'theta 40 arcmin'


num_ch=40
min_ch = 905
max_ch = 1295
nside=128
npix= hp.nside2npix(nside)
jmax=4
lmax= 3*nside-1
Nfg=3
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)

path_PCA_HI=dir_PCA+f'res_PCA_HI_noise_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_nside{nside}'
path_PCA_fg=dir_PCA+f'res_PCA_fg_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_nside{nside}'
path_cosmo_HI = f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise_mask0.39_unseen/cosmo_HI_noise_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz_lmax{lmax}_nside{nside}'
path_cosmo_HI_fullsky = f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise/cosmo_HI_noise_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz_lmax{lmax}_nside{nside}'

path_fg = f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise_mask0.39_unseen/fg_input_{fg_comp}_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz_lmax{lmax}_nside{nside}'
path_leak_Fg = dir_PCA+f'leak_PCA_fg_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_nside{nside}'
path_leak_HI = dir_PCA+f'leak_PCA_HI_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_nside{nside}'
path_cosmo_HI_bjk = f'../Maps_needlets/No_mean/Beam_{beam_s}_noise_mask0.39_unseen/bjk_maps_HI_noise_{num_ch}freq_{min_ch:1.1f}_{max_ch:1.1f}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}'
path_input_fg_bjk = f'../Maps_needlets/No_mean/Beam_{beam_s}_noise_mask0.39_unseen/bjk_maps_fg_{fg_comp}_{num_ch}freq_{min_ch:1.1f}_{max_ch:1.1f}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}'


print(f'jmax:{jmax}, lmax:{lmax}, num_ch:{num_ch}, min_ch:{min_ch}, max_ch:{max_ch}, Nfg:{Nfg}')

nu_ch = np.linspace(min_ch, max_ch, num_ch)


#for nu in range(len(nu_ch)):
#        alm_HI = hp.map2alm(cosmo_HI[nu], lmax=lmax)
#        cosmo_HI[nu] = hp.alm2map(alm_HI, lmax=lmax, nside = nside)
#        del alm_HI
#        alm_fg = hp.map2alm(fg[nu], lmax=lmax)
#        fg[nu] = hp.alm2map(alm_fg, lmax=lmax, nside = nside)
#        del alm_fg

ich=int(num_ch/2)
###########################################################################################
mask1_40 = hp.read_map('../HFI_Mask_GalPlane_2048_R1.10.fits', field=1)#fsky 40 % coverage
mask_40t = hp.ud_grade(mask1_40, nside_out=256)
mask_40 = hp.ud_grade(mask_40t, nside_out=nside)
del mask1_40
mask_40s = hp.sphtfunc.smoothing(mask_40, 3*np.pi/180,lmax=lmax) #apodization 3 deg come in Olivari
#del mask_40
fsky  = np.mean(mask_40s) 

fig=plt.figure()
hp.mollview(mask_40, cmap='viridis', title=f'fsky={np.mean(mask_40s):0.2f}', hold=True)
#plt.savefig(f'Plots_sims/mask_apo3deg_fsky{np.mean(mask_40s):0.2f}_nside{nside}.png')
plt.show()

############################################################################################
####################### NEEDLETS2HARMONICS #################################################

b_values = pippo.mylibpy_needlets_std_init_b_values(B,jmax,lmax)
res_PCA_HI = np.load(path_PCA_HI+'.npy')
res_PCA_fg = np.load(path_PCA_fg+'.npy')

print(res_PCA_HI.shape)
map_PCA_HI_need2pix=np.zeros((len(nu_ch), npix))
map_PCA_fg_need2pix=np.zeros((len(nu_ch), npix))

for nu in range(len(nu_ch)):
    map_PCA_fg_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(res_PCA_fg[:,nu],B, lmax)
    map_PCA_HI_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(res_PCA_HI[:,nu],B, lmax)
np.save(out_dir_maps_recon+f'maps_reconstructed_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_PCA_HI_need2pix)
np.save(out_dir_maps_recon+f'maps_reconstructed_PCA_fg_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_PCA_fg_need2pix)

del res_PCA_HI; del res_PCA_fg

cosmo_HI_bjk = np.load(path_cosmo_HI_bjk+'.npy')#[:,:jmax,:]
fg_bjk = np.load(path_input_fg_bjk+'.npy')#[:,:jmax,:]
print(cosmo_HI_bjk.shape)

map_input_HI_need2pix=np.zeros((len(nu_ch), npix))
map_input_fg_need2pix=np.zeros((len(nu_ch), npix))
for nu in range(len(nu_ch)):
	for j in range(cosmo_HI_bjk.shape[1]):
		map_input_HI_need2pix[nu] += pippo.mylibpy_needlets_f2betajk_j_healpix_harmonic(cosmo_HI_bjk[nu,j],b_values,j)
		map_input_fg_need2pix[nu] += pippo.mylibpy_needlets_f2betajk_j_healpix_harmonic(fg_bjk[nu,j],b_values,j)
	#map_input_HI_need2pix[nu] = hp.remove_dipole(map_input_HI_need2pix[nu])
	#map_input_fg_need2pix[nu] = hp.remove_dipole(map_input_fg_need2pix[nu])
	#map_input_HI_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(cosmo_HI_bjk[nu,:],B, lmax)
	#map_input_fg_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(fg_bjk[nu,:],B, lmax)
np.save(out_dir_maps_recon+f'maps_reconstructed_cosmo_HI_noise_{num_ch}_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_input_HI_need2pix)
np.save(out_dir_maps_recon+f'maps_reconstructed_input_fg_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_input_fg_need2pix)
del cosmo_HI_bjk; del fg_bjk


#map_PCA_HI_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_PCA_HI_{fg_comp}_{num_ch}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
#map_PCA_fg_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_PCA_fg_{fg_comp}_{num_ch}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
#map_input_fg_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_input_fg_{fg_comp}_{num_ch}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
#map_input_HI_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_cosmo_HI_{num_ch}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')


fg = np.load(path_fg+'.npy', allow_pickle=True)
cosmo_HI = np.load(path_cosmo_HI+'.npy', allow_pickle=True)
cosmo_HI_fullsky = np.load(path_cosmo_HI_fullsky+'.npy')


fig=plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, BEAM {beam}, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
fig.add_subplot(311)
hp.mollview(cosmo_HI[ich], min=0, max=1, title='Input HI+ noise', cmap='viridis', hold=True)
fig.add_subplot(312) 
hp.mollview(map_input_HI_need2pix[ich], min=0, max=1, title='Need recons HI + noise', cmap= 'viridis', hold=True)
fig.add_subplot(313) 
hp.mollview(100*(map_input_HI_need2pix[ich]/cosmo_HI[ich]-1), min=-0.02, max=0.02, title='% Need recons HI/HI -1', cmap= 'viridis', hold=True)


fig=plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
fig.add_subplot(131) 
hp.gnomview(fg[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-1e3, max=1e4, title='Input fg', cmap='viridis', hold=True)
fig.add_subplot(132) 
hp.gnomview(map_input_fg_need2pix[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-1e3, max=1e4, title='Need recons fg', cmap= 'viridis', hold=True)
fig.add_subplot(133) 
hp.gnomview(100*(map_input_fg_need2pix[ich]/fg[ich]-1), rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-0.02, max=0.02, title='% Need recons fg/fg -1', cmap= 'viridis', hold=True)
#plt.tight_layout()
plt.show()
#del map_input_fg_need2pix

#fig=plt.figure(figsize=(10, 7))
#fig.suptitle(f'channel: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
#fig.add_subplot(131) 
#hp.gnomview(cosmo_HI[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='Input HI', cmap='viridis', hold=True)
#fig.add_subplot(132) 
#hp.gnomview(map_PCA_HI_need2pix[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='PCA HI', cmap= 'viridis', hold=True)
#fig.add_subplot(133) 
#hp.gnomview(cosmo_HI[ich]-map_PCA_HI_need2pix[ich], rot=[-22,21],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-0.2, max=0.2, title='% HI - PCA', cmap= 'viridis', hold=True)
##plt.tight_layout()
#plt.show()

rot = [-56,87]
xsize=150
reso = hp.nside2resol(nside, arcmin=True)
map0  = hp.gnomview(cosmo_HI[ich],rot=rot, coord='G', reso=reso,xsize=xsize, min=0, max=1,return_projected_map=True, no_plot=True)
map1  = hp.gnomview(cosmo_HI[ich]+fg[ich],rot=rot, coord='G', reso=reso,xsize=xsize, min=-1e3, max=1e3,return_projected_map=True, no_plot=True)
map2  = hp.gnomview(map_PCA_HI_need2pix[ich],rot=rot, coord='G', reso=reso,xsize=xsize, min=0, max=1,return_projected_map=True, no_plot=True)

fig, axs = plt.subplots(1,3, figsize=(12,6))
cmap= 'viridis'
fig.suptitle(f'STD NEED, BEAM {beam}, channel: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=19)
im0=axs[0].imshow(map0,cmap=cmap)
axs[0].set_title(f'Input HI + noise', fontsize=15)
axs[0].set_xlabel(r'$\theta$[deg]', fontsize=15)
axs[0].set_ylabel(r'$\theta$[deg]', fontsize=15)
im1=axs[1].imshow(map1,cmap=cmap)
axs[1].set_title(f'Input HI + noise + foregrounds', fontsize=15)
axs[1].set_xlabel(r'$\theta$[deg]', fontsize=15)
#axs[1].set_ylabel(r'$\theta$[deg]')
im2=axs[2].imshow(map2,cmap=cmap)
axs[2].set_title(f'Cleaned HI + noise', fontsize=15)
axs[2].set_xlabel(r'$\theta$[deg]', fontsize=15)
#axs[2].set_ylabel(r'$\theta$[deg]')
norm = colors.Normalize(vmin=0, vmax=1)
plt.subplots_adjust(wspace=0.2, hspace=0.4, bottom=0.3, left=0.05, top=0.85)
sub_ax = plt.axes([0.91, 0.367, 0.02, 0.46]) 
fig.colorbar(im2, ax=axs, cax=sub_ax,location='right',orientation='vertical',label='T [mK]')
#plt.savefig(f'Plots_PCA_needlets/gnomview_HI_HIfg_PCAHI_std_need_beam40arcmin_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.png')

print(f'mean % rel diff PCA fg/fg:{100*np.mean((np.abs(map_PCA_fg_need2pix[ich]/fg[ich]-1)))}')

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'STD NEED, BEAM {beam}, channel: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=19)
fig.add_subplot(221) 
hp.mollview(100*(np.abs(map_PCA_fg_need2pix[ich]/fg[ich]-1)), min=0, max=10,  title= '(Res fg/fg-1)%',cmap='viridis',unit='%', hold= True)
fig.add_subplot(222) 
hp.mollview(cosmo_HI[ich],min=0, max=1, title= 'Cosmo HI + noise',cmap='viridis', hold=True)
fig.add_subplot(223) 
hp.mollview(map_PCA_HI_need2pix[ich],min=0, max=1, title= 'Res PCA HI + noise Needlets 2 Pix',cmap='viridis', hold= True)
#plt.tight_layout()
plt.show()

del fg; del map_PCA_fg_need2pix;del map_input_fg_need2pix
################################################################################
############################# CL ###############################################
out_dir_cl = out_dir_maps_recon+'cls_recons_need/'
if not os.path.exists(out_dir_cl):
		os.makedirs(out_dir_cl)
lmax_cl = 2*nside
ell=np.arange(lmax_cl+1)
factor=ell*(ell+1)/(2*np.pi)


cl_cosmo_HI_recons = np.zeros((len(nu_ch), lmax_cl+1))
cl_cosmo_HI = np.zeros((len(nu_ch), lmax_cl+1))
cl_cosmo_HI_fullsky = np.zeros((len(nu_ch), lmax_cl+1))
cl_PCA_HI_need2harm = np.zeros((len(nu_ch), lmax_cl+1))
#cl_diff_cosmo_PCA_HI_need2harm = np.zeros((len(nu_ch), lmax_cl+1))

for n in range(len(nu_ch)):
	cl_cosmo_HI_recons[n] = hp.anafast(map_input_HI_need2pix[n], lmax=lmax_cl)
	cl_cosmo_HI[n]=hp.anafast(cosmo_HI[n], lmax=lmax_cl)
	cl_cosmo_HI_fullsky[n]=hp.anafast(cosmo_HI_fullsky[n], lmax=lmax_cl)
	cl_PCA_HI_need2harm[n] = hp.anafast(map_PCA_HI_need2pix[n], lmax=lmax_cl)
	#cl_diff_cosmo_PCA_HI_need2harm[n] = hp.anafast(cosmo_HI[n]-map_PCA_HI_need2pix[n], lmax=lmax_cl)

#### deconvoluzione
f_0_mask = nm.NmtField(mask_40,[map_PCA_HI_need2pix[0]] )
b = nm.NmtBin.from_nside_linear(nside, 8)
ell_mask= b.get_effective_ells()

cl_PCA_HI_mask_deconv = np.zeros((num_ch, len(ell_mask)))
cl_PCA_HI_mask_deconv_interp = np.zeros((num_ch, lmax_cl+1))

cl_PCA_HI_mask_0_deconv = np.zeros((num_ch, len(ell_mask)))
cl_PCA_HI_mask_0_deconv_interp = np.zeros((num_ch, lmax_cl+1))

for n in range(num_ch):
    f_0_mask = nm.NmtField(mask_40,[map_PCA_HI_need2pix[n]] )
    cl_PCA_HI_mask_deconv[n] = nm.compute_full_master(f_0_mask, f_0_mask, b)[0]
    cl_PCA_HI_mask_deconv_interp[n] = np.interp(ell, ell_mask, cl_PCA_HI_mask_deconv[n])
    

np.savetxt(out_dir_cl+f'cl_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{lmax_cl}_nside{nside}.dat', cl_PCA_HI_need2harm)
np.savetxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{lmax_cl}_nside{nside}.dat', cl_PCA_HI_mask_deconv_interp)


del map_PCA_HI_need2pix; del cosmo_HI; del map_input_HI_need2pix; 
##################################################################################################

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: channel:{nu_ch[ich]} MHz, BEAM {beam}, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}, fsky:0.39')
plt.semilogy(ell[2:],factor[2:]*cl_cosmo_HI[ich][2:],label='Cosmo + noise fsky')
plt.semilogy(ell[2:],factor[2:]*cl_PCA_HI_need2harm[ich][2:],'+', label='PCA HI + noise fsky')
plt.semilogy(ell[2:],factor[2:]*cl_PCA_HI_mask_deconv_interp[ich][2:],'+', label='PCA HI + noise fsky deconv')
plt.semilogy(ell[2:],factor[2:]*cl_cosmo_HI_fullsky[ich][2:],'k--' ,label='Cosmo + noise full sky')
#plt.semilogy(ell[2:],factor[2:]*cl_cosmo_HI_recons[ich][2:], label='Cosmo reconstructed')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 10))


diff_cl_need2sphe = cl_PCA_HI_need2harm/cl_cosmo_HI-1
diff_cl_need2sphe_full = cl_PCA_HI_mask_deconv_interp/cl_cosmo_HI_fullsky-1
#diff_cl_need2sphe_cosmo_recons = cl_cosmo_HI_recons/cl_cosmo_HI-1
frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe[ich][2:]*100, label='fsky')
plt.plot(ell[2:], diff_cl_need2sphe_full[ich][2:]*100, label='full sky')
#plt.plot(ell[2:], diff_cl_need2sphe_cosmo_recons[ich][2:]*100, label=f'% recons_HI/input_HI -1')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-10,10])
frame2.set_ylabel(r'%$ C_{\ell}^{\rm PCA}/C_{\ell}^{\rm cosmo} $-1')
frame2.set_xlabel(r'$\ell$')
frame1.set_xticks(np.arange(1,200+1, 10))
#plt.tight_layout()
plt.legend()
#plt.savefig(f'Plots_PCA_needlets/cl_std_need_ch{nu_ch[ich]}_{fg_comp}_noise_beam40arcmin_jmax{jmax}_lmax{lmax_cl}_Nfg{Nfg}_nside{nside}_mask0.39.png')

plt.show()

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: mean over channels, BEAM {beam}, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}')
plt.plot(ell[2:], factor[2:]*cl_cosmo_HI.mean(axis=0)[2:], label = f'Cosmo + noise')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_need2harm.mean(axis=0)[2:],'+',mfc='none', label = f'PCA HI + noise')
#plt.plot(ell[2:], factor[2:]*cl_cosmo_HI_recons.mean(axis=0)[2:], label = f'Cosmo reconstructed')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 10))

1
del cl_PCA_HI_need2harm
del cl_cosmo_HI_recons; del cl_cosmo_HI
frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe.mean(axis=0)[2:]*100, label='% PCA_HI/input_HI -1')
#plt.plot(ell[2:], diff_cl_need2sphe_cosmo_recons.mean(axis=0)[2:]*100, label=f'% recons_HI/input_HI -1')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-10,10])
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
frame1.set_xticks(np.arange(1,200+1, 10))
#plt.tight_layout()
plt.legend()


plt.show()

del diff_cl_need2sphe; #del diff_cl_need2sphe_cosmo_recons


#######################################################################
############################ LEAKAGE ##################################
print(' sto ricostruendo il leakage')

need_HI_leak=np.load(path_leak_HI+'.npy')
map_leak_HI_need2pix=np.zeros((len(nu_ch), npix))
for nu in range(len(nu_ch)):
    #map_leak_HI_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(need_HI_leak[:,nu],B, lmax)
    for j in range(need_HI_leak.shape[0]):
        map_leak_HI_need2pix[nu] += pippo.mylibpy_needlets_f2betajk_j_healpix_harmonic(need_HI_leak[j,nu],b_values,j)
    #map_leak_HI_need2pix[nu] = hp.remove_dipole(map_leak_HI_need2pix[nu])
np.save(out_dir_maps_recon+f'maps_reconstructed_leak_HI_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_leak_HI_need2pix)

del need_HI_leak

need_fg_leak=np.load(path_leak_Fg+'.npy')
map_leak_fg_need2pix=np.zeros((len(nu_ch), npix))
for nu in range(len(nu_ch)):
    for j in range(need_fg_leak.shape[0]):
        map_leak_fg_need2pix[nu] += pippo.mylibpy_needlets_f2betajk_j_healpix_harmonic(need_fg_leak[j,nu],b_values,j)
    #map_leak_fg_need2pix[nu] = hp.remove_dipole(map_leak_fg_need2pix[nu])
    #map_leak_fg_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(need_fg_leak[:,nu],B, lmax)
del need_fg_leak
np.save(out_dir_maps_recon+f'maps_reconstructed_leak_fg_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_leak_fg_need2pix)


#map_leak_HI_need2pix = np.load(out_dir_maps_recon+f'maps_reconstructed_leak_HI_{fg_comp}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
#map_leak_fg_need2pix = np.load(out_dir_maps_recon+f'maps_reconstructed_leak_fg_{fg_comp}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, BEAM {beam}, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
fig.add_subplot(211) 
hp.mollview(map_leak_HI_need2pix[ich],min=0, max=1, title= 'Leakage HI',cmap='viridis', hold=True)
fig.add_subplot(212) 
hp.mollview(map_leak_fg_need2pix[ich],min=0, max=1, title= 'Leakage Fg',cmap='viridis', hold= True)
#plt.tight_layout()
plt.show()

######################################################################

cl_leak_HI = np.zeros((len(nu_ch), lmax_cl+1))
cl_leak_fg = np.zeros((len(nu_ch), lmax_cl+1))
cl_diff_leak = np.zeros((len(nu_ch), lmax_cl+1))

for n in range(len(nu_ch)):
	cl_leak_HI[n] = hp.anafast(map_leak_HI_need2pix[n], lmax=lmax_cl)
	cl_leak_fg[n] = hp.anafast(map_leak_fg_need2pix[n], lmax=lmax_cl)
	cl_diff_leak[n] = hp.anafast(map_leak_HI_need2pix[n]-map_leak_fg_need2pix[n], lmax=lmax_cl)


np.savetxt(out_dir_cl+f'cl_leak_HI_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{lmax_cl}_nside{nside}.dat', cl_leak_HI)
np.savetxt(out_dir_cl+f'cl_leak_fg_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{lmax_cl}_nside{nside}.dat', cl_leak_fg)

del map_leak_HI_need2pix; del map_leak_fg_need2pix
#
fig=plt.figure()
fig.suptitle(f'channel: {nu_ch[ich]} MHz, BEAM {beam}, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}')
plt.semilogy(ell[2:], factor[2:]*np.mean(cl_leak_fg, axis=0)[2:],mfc='none', label='Fg leakage')
plt.semilogy(ell[2:], factor[2:]*np.mean(cl_leak_HI, axis=0)[2:],mfc='none', label='HI leakage')
plt.xlim([0,200])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell*(\ell+1)}{2\pi} \langle C_{\ell} \rangle$')
plt.legend()
#plt.tight_layout()
#plt.savefig(f'Plots_PCA_needlets/recons_factorxcl_beam40arcmin_leakage_jmax{jmax}_lmax{lmax_cl}.png')
plt.show()

#fig = plt.figure()
#plt.semilogy(ell[2:],np.mean(cl_diff_cosmo_PCA_HI_need2harm, axis=0)[2:],mfc='none', label='Cl diff Cosmo - PCA HI maps')
#plt.semilogy(ell[2:],np.mean(cl_diff_leak, axis=0)[2:],mfc='none', label='Cl HI - Fg leakage')
#plt.title(f'NEEDLETS CLs: mean over channels, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}')
#plt.xlabel(r'$\ell$')
#plt.ylabel(r'$ \frac{\ell*(\ell+1)}{2\pi} \langle C_{\ell} \rangle$')
#plt.legend()
#plt.show()

