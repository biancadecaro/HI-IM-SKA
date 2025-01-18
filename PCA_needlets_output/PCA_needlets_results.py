import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import re
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
import cython_mylibc as pippo

c_pal = sns.color_palette().as_hex()
##########################################################################################

out_dir_plot = 'Plots_PCA_needlets/'
dir_PCA = 'PCA_maps/No_mean/'
out_dir_maps_recon = 'maps_reconstructed/No_mean/'
if not os.path.exists(out_dir_maps_recon):
        os.makedirs(out_dir_maps_recon)


fg_comp = 'synch_ff_ps'


num_ch=40
min_ch = 905
max_ch = 1295
nside=256
npix= hp.nside2npix(nside)
jmax=12
lmax= 3*nside#2*nside#
Nfg=3
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)

path_PCA_HI=dir_PCA+f'res_PCA_HI_synch_ff_ps_jmax{jmax}_lmax{lmax}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_nside{nside}'
path_PCA_fg=dir_PCA+f'res_PCA_fg_synch_ff_ps_jmax{jmax}_lmax{lmax}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_nside{nside}'
path_cosmo_HI = f'../PCA_pixels_output/Maps_PCA/No_mean/cosmo_HI_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz_lmax{lmax}_nside{nside}'
path_fg = f'../PCA_pixels_output/Maps_PCA/No_mean/fg_input_synch_ff_ps_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz_lmax{lmax}_nside{nside}'
path_leak_Fg = dir_PCA+f'leak_PCA_synch_ff_ps_jmax{jmax}_lmax{lmax}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_nside{nside}'
path_leak_HI = dir_PCA+f'leak_PCA_HI_synch_ff_ps_jmax{jmax}_lmax{lmax}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_nside{nside}'
path_cosmo_HI_bjk = f'../Maps_needlets/No_mean/bjk_maps_HI_{num_ch}freq_{min_ch:1.1f}_{max_ch:1.1f}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}'
path_input_fg_bjk = f'../Maps_needlets/No_mean/bjk_maps_fg_synch_ff_ps_{num_ch}freq_{min_ch:1.1f}_{max_ch:1.1f}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}'


print(f'jmax:{jmax}, lmax:{lmax}, num_ch:{num_ch}, min_ch:{min_ch}, max_ch:{max_ch}, Nfg:{Nfg}')

nu_ch = np.linspace(min_ch, max_ch, num_ch)
del min_ch;del max_ch


#for nu in range(len(nu_ch)):
#        alm_HI = hp.map2alm(cosmo_HI[nu], lmax=lmax)
#        cosmo_HI[nu] = hp.alm2map(alm_HI, lmax=lmax, nside = nside)
#        del alm_HI
#        alm_fg = hp.map2alm(fg[nu], lmax=lmax)
#        fg[nu] = hp.alm2map(alm_fg, lmax=lmax, nside = nside)
#        del alm_fg

ich=int(num_ch/2)

############################################################################################
####################### NEEDLETS2HARMONICS #################################################

b_values = pippo.mylibpy_needlets_std_init_b_values(B,jmax,lmax)
#res_PCA_HI = np.load(path_PCA_HI+'.npy')
#res_PCA_fg = np.load(path_PCA_fg+'.npy')
#
#print(res_PCA_HI.shape)
#map_PCA_HI_need2pix=np.zeros((len(nu_ch), npix))
#map_PCA_fg_need2pix=np.zeros((len(nu_ch), npix))
#
#for nu in range(len(nu_ch)):
#	for j in range(res_PCA_HI.shape[0]):
#		map_PCA_fg_need2pix[nu] += pippo.mylibpy_needlets_f2betajk_j_healpix_harmonic(res_PCA_fg[j,nu],b_values,j)
#		map_PCA_HI_need2pix[nu] += pippo.mylibpy_needlets_f2betajk_j_healpix_harmonic(res_PCA_HI[j,nu],b_values,j)
#	#map_PCA_fg_need2pix[nu] = hp.remove_dipole(map_PCA_fg_need2pix[nu])
#	#map_PCA_HI_need2pix[nu] = hp.remove_dipole(map_PCA_HI_need2pix[nu])
##for nu in range(len(nu_ch)):
##    map_PCA_fg_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(res_PCA_fg[:,nu],B, lmax)
##    map_PCA_HI_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(res_PCA_HI[:,nu],B, lmax)
#np.save(out_dir_maps_recon+f'maps_reconstructed_PCA_HI_{num_ch}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_PCA_HI_need2pix)
#np.save(out_dir_maps_recon+f'maps_reconstructed_PCA_fg_{num_ch}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_PCA_fg_need2pix)
#del res_PCA_HI; del res_PCA_fg
#
#cosmo_HI_bjk = np.load(path_cosmo_HI_bjk+'.npy')#[:,:jmax,:]
#fg_bjk = np.load(path_input_fg_bjk+'.npy')#[:,:jmax,:]
#print(cosmo_HI_bjk.shape)
#
#map_input_HI_need2pix=np.zeros((len(nu_ch), npix))
#map_input_fg_need2pix=np.zeros((len(nu_ch), npix))
#for nu in range(len(nu_ch)):
#	for j in range(cosmo_HI_bjk.shape[1]):
#		map_input_HI_need2pix[nu] += pippo.mylibpy_needlets_f2betajk_j_healpix_harmonic(cosmo_HI_bjk[nu,j],b_values,j)
#		map_input_fg_need2pix[nu] += pippo.mylibpy_needlets_f2betajk_j_healpix_harmonic(fg_bjk[nu,j],b_values,j)
#	#map_input_HI_need2pix[nu] = hp.remove_dipole(map_input_HI_need2pix[nu])
#	#map_input_fg_need2pix[nu] = hp.remove_dipole(map_input_fg_need2pix[nu])
#	#map_input_HI_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(cosmo_HI_bjk[nu,:],B, lmax)
#	#map_input_fg_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(fg_bjk[nu,:],B, lmax)
#np.save(out_dir_maps_recon+f'maps_reconstructed_cosmo_HI_{num_ch}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_input_HI_need2pix)
#np.save(out_dir_maps_recon+f'maps_reconstructed_input_fg_{num_ch}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_input_fg_need2pix)
#del cosmo_HI_bjk; del fg_bjk


map_PCA_HI_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_PCA_HI_{num_ch}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
map_PCA_fg_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_PCA_fg_{num_ch}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
map_input_fg_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_input_fg_{num_ch}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
map_input_HI_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_cosmo_HI_{num_ch}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')

fg = np.load(path_fg+'.npy')
cosmo_HI = np.load(path_cosmo_HI+'.npy')

fig=plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
fig.add_subplot(311)
hp.mollview(cosmo_HI[ich], min=0, max=1, title='Input HI', cmap='viridis', hold=True)
fig.add_subplot(312) 
hp.mollview(map_input_HI_need2pix[ich], min=0, max=1, title='Need recons HI', cmap= 'viridis', hold=True)
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
plt.tight_layout()
plt.show()
#del map_input_fg_need2pix

fig=plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
fig.add_subplot(131) 
hp.gnomview(cosmo_HI[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='Input HI', cmap='viridis', hold=True)
fig.add_subplot(132) 
hp.gnomview(map_PCA_HI_need2pix[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='PCA HI', cmap= 'viridis', hold=True)
fig.add_subplot(133) 
hp.gnomview(cosmo_HI[ich]-map_PCA_HI_need2pix[ich], rot=[-22,21],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-0.2, max=0.2, title='% HI - PCA', cmap= 'viridis', hold=True)
plt.tight_layout()
plt.show()

print(f'mean % rel diff PCA fg/fg:{100*np.mean((np.abs(map_PCA_fg_need2pix[ich]/fg[ich]-1)))}')

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'STD NEED, channel: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=19)
fig.add_subplot(221) 
hp.mollview(100*(np.abs(map_PCA_fg_need2pix[ich]/fg[ich]-1)), min=0, max=10,  title= '(Res fg/fg-1)%',cmap='viridis',unit='%', hold= True)
fig.add_subplot(222) 
hp.mollview(cosmo_HI[ich],min=0, max=1, title= 'Cosmo HI',cmap='viridis', hold=True)
fig.add_subplot(223) 
hp.mollview(map_PCA_HI_need2pix[ich],min=0, max=1, title= 'Res PCA HI Needlets 2 Pix',cmap='viridis', hold= True)
plt.tight_layout()
plt.show()

del fg; del map_PCA_fg_need2pix;del map_input_fg_need2pix
################################################################################
############################# CL ###############################################
out_dir_cl = out_dir_maps_recon+'cls_recons_need/'
if not os.path.exists(out_dir_cl):
        os.makedirs(out_dir_cl)
lmax_cl = 2*nside

cl_cosmo_HI_recons = np.zeros((len(nu_ch), lmax_cl+1))
cl_cosmo_HI = np.zeros((len(nu_ch), lmax_cl+1))
cl_PCA_HI_need2harm = np.zeros((len(nu_ch), lmax_cl+1))
cl_diff_cosmo_PCA_HI_need2harm = np.zeros((len(nu_ch), lmax_cl+1))

for n in range(len(nu_ch)):
    cl_cosmo_HI_recons[n] = hp.anafast(map_input_HI_need2pix[n], lmax=lmax_cl)
    cl_cosmo_HI[n]=hp.anafast(cosmo_HI[n], lmax=lmax_cl)
    cl_PCA_HI_need2harm[n] = hp.anafast(map_PCA_HI_need2pix[n], lmax=lmax_cl)
    cl_diff_cosmo_PCA_HI_need2harm[n] = hp.anafast(cosmo_HI[n]-map_PCA_HI_need2pix[n], lmax=lmax_cl)

np.savetxt(out_dir_cl+f'cl_PCA_HI_Nfg{Nfg}_jmax{jmax}_lmax{lmax_cl}_nside{nside}.dat', cl_PCA_HI_need2harm)


del map_PCA_HI_need2pix; del cosmo_HI; del map_input_HI_need2pix; 
##################################################################################################
ell=np.arange(lmax_cl+1)
factor=ell*(ell+1)/(2*np.pi)

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: channel:{nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.semilogy(ell[1:],cl_PCA_HI_need2harm[ich][1:], label='PCA HI')
plt.semilogy(ell[1:],cl_cosmo_HI[ich][1:], label='Cosmo')
plt.semilogy(ell[1:],cl_cosmo_HI_recons[ich][1:], label='Cosmo reconstructed')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 10))


diff_cl_need2sphe = cl_PCA_HI_need2harm/cl_cosmo_HI-1
diff_cl_need2sphe_cosmo_recons = cl_cosmo_HI_recons/cl_cosmo_HI-1
frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[1:], diff_cl_need2sphe[ich][1:]*100, label='% PCA_HI/input_HI -1')
plt.plot(ell[1:], diff_cl_need2sphe_cosmo_recons[ich][1:]*100, label=f'% recons_HI/input_HI -1')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-10,10])
frame2.set_ylabel(r'%$ diff $')
frame2.set_xlabel(r'$\ell$')
frame1.set_xticks(np.arange(1,200+1, 10))
plt.tight_layout()
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'STANDARD NEED CLs: mean over channels, jmax:{jmax}, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.plot(ell[2:], factor[2:]*cl_cosmo_HI.mean(axis=0)[2:], label = f'Cosmo')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_need2harm.mean(axis=0)[2:],'+',mfc='none', label = f'PCA')
plt.plot(ell[2:], factor[2:]*cl_cosmo_HI_recons.mean(axis=0)[2:], label = f'Cosmo reconstructed')
plt.xlim([0,200])
plt.ylim(top=0.04)
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 10))

del cl_PCA_HI_need2harm
del cl_cosmo_HI_recons; del cl_cosmo_HI
frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe.mean(axis=0)[2:]*100, c=c_pal[1],label='% PCA_HI/input_HI -1')
plt.plot(ell[2:], diff_cl_need2sphe_cosmo_recons.mean(axis=0)[2:]*100, c=c_pal[2],label=f'% recons_HI/input_HI -1')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-10,10])
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
frame1.set_xticks(np.arange(1,200+1, 10))
plt.tight_layout()
plt.legend()
plt.savefig(f'recons_factorxcl_jmax{jmax}_lmax{lmax_cl}.png')

plt.show()

del diff_cl_need2sphe; del diff_cl_need2sphe_cosmo_recons


#######################################################################
############################ LEAKAGE ##################################
print(' sto ricostruendo il leakage')

need_HI_leak=np.load(path_leak_HI+'.npy')
map_leak_HI_need2pix=np.zeros((len(nu_ch), npix))
for nu in range(len(nu_ch)):
    #map_leak_HI_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(need_HI_leak[:,nu],B, lmax)
    for j in range(need_HI_leak.shape[0]):
        map_leak_HI_need2pix[nu] += pippo.mylibpy_needlets_f2betajk_j_healpix_harmonic(need_HI_leak[j,nu],b_values,j)
np.save(out_dir_maps_recon+f'maps_reconstructed_leak_HI_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_leak_HI_need2pix)

del need_HI_leak

need_fg_leak=np.load(path_leak_Fg+'.npy')
map_leak_fg_need2pix=np.zeros((len(nu_ch), npix))
for nu in range(len(nu_ch)):
    for j in range(need_fg_leak.shape[0]):
        map_leak_fg_need2pix[nu] += pippo.mylibpy_needlets_f2betajk_j_healpix_harmonic(need_fg_leak[j,nu],b_values,j)
    #map_leak_fg_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(need_fg_leak[:,nu],B, lmax)
del need_fg_leak
np.save(out_dir_maps_recon+f'maps_reconstructed_leak_fg_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_leak_fg_need2pix)


#map_leak_HI_need2pix = np.load(out_dir_maps_recon+f'maps_reconstructed_leak_HI_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
#map_leak_fg_need2pix = np.load(out_dir_maps_recon+f'maps_reconstructed_leak_fg_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
fig.add_subplot(211) 
hp.mollview(map_leak_HI_need2pix[ich],min=0, max=1, title= 'Leakage HI',cmap='viridis', hold=True)
fig.add_subplot(212) 
hp.mollview(map_leak_fg_need2pix[ich],min=0, max=1, title= 'Leakage Fg',cmap='viridis', hold= True)
plt.tight_layout()
plt.show()

######################################################################

cl_leak_HI = np.zeros((len(nu_ch), lmax_cl+1))
cl_leak_fg = np.zeros((len(nu_ch), lmax_cl+1))
cl_diff_leak = np.zeros((len(nu_ch), lmax_cl+1))

for n in range(len(nu_ch)):
    cl_leak_HI[n] = hp.anafast(map_leak_HI_need2pix[n], lmax=lmax_cl)
    cl_leak_fg[n] = hp.anafast(map_leak_fg_need2pix[n], lmax=lmax_cl)
    cl_diff_leak[n] = hp.anafast(map_leak_HI_need2pix[n]-map_leak_fg_need2pix[n], lmax=lmax_cl)


np.savetxt(out_dir_cl+f'cl_leak_HI_Nfg{Nfg}_{jmax}_lmax{lmax_cl}_nside{nside}.dat', cl_leak_HI)
np.savetxt(out_dir_cl+f'cl_leak_fg_Nfg{Nfg}_{jmax}_lmax{lmax_cl}_nside{nside}.dat', cl_leak_fg)

del map_leak_HI_need2pix; del map_leak_fg_need2pix
#
fig=plt.figure()
plt.semilogy(ell[2:], factor[2:]*np.mean(cl_leak_fg, axis=0)[2:],mfc='none', label='Fg leakage')
plt.semilogy(ell[2:], factor[2:]*np.mean(cl_leak_HI, axis=0)[2:],mfc='none', label='HI leakage')
plt.xlim([0,200])
plt.title(f'STANDARD NEED CLs: mean over channels, jmax:{jmax}, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell*(\ell+1)}{2\pi} \langle C_{\ell} \rangle$')
plt.legend()
plt.tight_layout()
plt.savefig(f'recons_factorxcl_leakage_jmax{jmax}_lmax{lmax_cl}.png')
plt.show()

#fig = plt.figure()
#plt.semilogy(ell[:],np.mean(cl_diff_cosmo_PCA_HI_need2harm, axis=0)[:],mfc='none', label='Cl diff Cosmo - PCA HI maps')
#plt.semilogy(ell[:],np.mean(cl_diff_leak, axis=0)[:],mfc='none', label='Cl HI - Fg leakage')
#plt.title(f'NEEDLETS CLs: mean over channels, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}')
#plt.xlabel(r'$\ell$')
#plt.ylabel(r'$ \frac{\ell*(\ell+1)}{2\pi} \langle C_{\ell} \rangle$')
#plt.legend()
#plt.show()

