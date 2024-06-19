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

nside=128
npix= hp.nside2npix(nside)

out_dir_plot = 'Plots_PCA_needlets/No_mean/'
dir_PCA = 'PCA_maps/No_mean/'
out_dir_maps_recon = 'maps_reconstructed/No_mean/'
out_dir_cls = 'cls_need2harm/No_mean/'
if not os.path.exists(out_dir_maps_recon):
        os.makedirs(out_dir_maps_recon)
if not os.path.exists(out_dir_cls):
        os.makedirs(out_dir_cls)


jmax=12
lmax=256
B=pippo.mylibpy_jmax_lmax2B(jmax, lmax)
Nfg=3

fg_comp = 'synch_ff_ps'


num_ch=200
min_ch = 901
max_ch = 1299
nside=128

path_PCA_HI=dir_PCA+f'res_PCA_HI_synch_ff_ps_jmax{jmax}_lmax{lmax}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_nside{nside}'
path_PCA_fg=dir_PCA+f'res_PCA_fg_synch_ff_ps_jmax{jmax}_lmax{lmax}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_nside{nside}'
path_cosmo_HI = f'../PCA_pixels_output/Maps_PCA_no_mean/cosmo_HI_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz'
path_fg = f'../PCA_pixels_output/Maps_PCA_no_mean/fg_input_synch_ff_ps_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz'
path_leak_Fg = dir_PCA+f'leak_PCA_fg_synch_ff_ps_jmax{jmax}_lmax{lmax}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_nside{nside}'
path_leak_HI = dir_PCA+f'leak_PCA_HI_synch_ff_ps_jmax{jmax}_lmax{lmax}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_nside{nside}'
path_cosmo_HI_bjk = f'../Maps_needlets/Maps_no_mean/bjk_maps_HI_{num_ch}freq_{min_ch:1.1f}_{max_ch:1.1f}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}'
path_input_fg_bjk = f'../Maps_needlets/Maps_no_mean/bjk_maps_fg_synch_ff_ps_{num_ch}freq_{min_ch:1.1f}_{max_ch:1.1f}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}'

#jmax=int(re.findall(r'\d+', path_PCA_HI)[0])
#lmax=int(re.findall(r'\d+', path_PCA_HI)[1])
#num_ch=int(re.findall(r'\d+', path_PCA_HI)[2])
#min_ch = float(re.findall(r'\d+', path_PCA_HI)[3])
#max_ch = float(re.findall(r'\d+', path_PCA_HI)[4])
#Nfg = int(re.findall(r'\d+', path_PCA_HI)[5])

print(f'jmax:{jmax}, lmax:{lmax}, num_ch:{num_ch}, min_ch:{min_ch}, max_ch:{max_ch}, Nfg:{Nfg}')

nu_ch = np.linspace(min_ch, max_ch, num_ch)
del min_ch;del max_ch

res_PCA_HI = np.load(path_PCA_HI+'.npy')
res_PCA_fg = np.load(path_PCA_fg+'.npy')

fg = np.load(path_fg+'.npy')
cosmo_HI = np.load(path_cosmo_HI+'.npy')


for nu in range(len(nu_ch)):
        alm_HI = hp.map2alm(cosmo_HI[nu], lmax=lmax)
        cosmo_HI[nu] = hp.alm2map(alm_HI, lmax=lmax, nside = nside)
        del alm_HI
        alm_fg = hp.map2alm(fg[nu], lmax=lmax)
        fg[nu] = hp.alm2map(alm_fg, lmax=lmax, nside = nside)
        del alm_fg

ich=100

#for j in range(jmax+1):
#    hp.mollview(res_PCA_HI[j, ich],cmap='viridis',min=0, max=1, title=f'Res PCA HI, j={j}, jmax={jmax}, Nfg:{Nfg}',hold=True,cbar=True)
#    plt.show()
#
#del res_PCA_HI; del res_PCA_fg
############################################################################################
####################### NEEDLETS2HARMONICS #################################################
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)

map_PCA_HI_need2pix=np.zeros((len(nu_ch), npix))
map_PCA_fg_need2pix=np.zeros((len(nu_ch), npix))
for nu in range(len(nu_ch)):
    map_PCA_HI_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(res_PCA_HI[:,nu],B, lmax)
    map_PCA_fg_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(res_PCA_fg[:,nu],B, lmax)
np.save(out_dir_maps_recon+f'maps_reconstructed_PCA_HI_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_PCA_HI_need2pix)
np.save(out_dir_maps_recon+f'maps_reconstructed_PCA_fg_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_PCA_fg_need2pix)
del res_PCA_HI; del res_PCA_fg

cosmo_HI_bjk = np.load(path_cosmo_HI_bjk+'.npy')
fg_bjk = np.load(path_input_fg_bjk+'.npy')

map_input_HI_need2pix=np.zeros((len(nu_ch), npix))
map_input_fg_need2pix=np.zeros((len(nu_ch), npix))
for nu in range(len(nu_ch)):
    map_input_HI_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(cosmo_HI_bjk[nu,:],B, lmax)
    map_input_fg_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(fg_bjk[nu,:],B, lmax)
np.save(out_dir_maps_recon+f'maps_reconstructed_cosmo_HI_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_input_HI_need2pix)
np.save(out_dir_maps_recon+f'maps_reconstructed_input_fg_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_input_fg_need2pix)
del cosmo_HI_bjk; del fg_bjk

#map_PCA_HI_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_PCA_HI_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
#map_PCA_fg_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_PCA_fg_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
#map_input_fg_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_input_fg_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
#map_input_HI_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_cosmo_HI_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')


fig=plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
fig.add_subplot(311)
hp.mollview(cosmo_HI[ich], min=0, max=1, title='Input HI', cmap='viridis', hold=True)
fig.add_subplot(312) 
hp.mollview(map_input_HI_need2pix[ich], min=0, max=1, title='Need recons HI', cmap= 'viridis', hold=True)
fig.add_subplot(313) 
hp.mollview(100*(np.abs(map_input_HI_need2pix[ich]/cosmo_HI[ich]-1)), min=0, max=50, title='% Need recons HI/HI -1', cmap= 'viridis', hold=True)


fig=plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
fig.add_subplot(131) 
hp.gnomview(fg[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-1e3, max=1e4, title='Input fg', cmap='viridis', hold=True)
fig.add_subplot(132) 
hp.gnomview(map_input_fg_need2pix[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-1e3, max=1e4, title='Need recons fg', cmap= 'viridis', hold=True)
fig.add_subplot(133) 
hp.gnomview(100*(np.abs(map_input_fg_need2pix[ich]/fg[ich]-1)), rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=20, title='% Need recons fg/fg -1', cmap= 'viridis', hold=True)
plt.tight_layout()
plt.show()

fig=plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
fig.add_subplot(131) 
hp.gnomview(cosmo_HI[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='Input HI', cmap='viridis', hold=True)
fig.add_subplot(132) 
hp.gnomview(map_PCA_HI_need2pix[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='PCA HI', cmap= 'viridis', hold=True)
fig.add_subplot(133) 
hp.gnomview(100*(np.abs(map_PCA_HI_need2pix[ich]/cosmo_HI[ich]-1)), rot=[-22,21],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=20, title='% PCA HI/HI -1', cmap= 'viridis', hold=True)
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
fig.add_subplot(221) 
hp.mollview(100*(np.abs(map_PCA_fg_need2pix[ich]/fg[ich]-1)), min=0, max=20,  title= '(Res fg/fg-1)%',cmap='viridis',unit='%', hold= True)
fig.add_subplot(222) 
hp.mollview(cosmo_HI[ich],min=0, max=1, title= 'Cosmo HI',cmap='viridis', hold=True)
fig.add_subplot(223) 
hp.mollview(map_PCA_HI_need2pix[ich],min=0, max=1, title= 'Res PCA HI Needlets 2 Pix',cmap='viridis', hold= True)
plt.tight_layout()
#plt.savefig(out_dir_plot+f'maps_recons_new_jmax{jmax}_lmax{lmax}_B{B:1.2f}_Nfg{Nfg}_nside{nside}.png')
plt.show()


cl_cosmo_HI_recons = np.zeros((len(nu_ch), lmax+1))
cl_cosmo_HI = np.zeros((len(nu_ch), lmax+1))
cl_PCA_HI_need2harm = np.zeros((len(nu_ch), lmax+1))
cl_diff_cosmo_PCA_HI_need2harm = np.zeros((len(nu_ch), lmax+1))

for n in range(len(nu_ch)):
    cl_cosmo_HI_recons[n] = hp.anafast(map_input_HI_need2pix[n], lmax=lmax)#, use_pixel_weights=True)
    cl_cosmo_HI[n]=hp.anafast(cosmo_HI[n], lmax=lmax)
    cl_PCA_HI_need2harm[n] = hp.anafast(map_PCA_HI_need2pix[n], lmax=lmax)#, use_pixel_weights=True)
    cl_diff_cosmo_PCA_HI_need2harm[n] = hp.anafast(cosmo_HI[n]-map_PCA_HI_need2pix[n], lmax=lmax)#, use_pixel_weights=True)


del map_PCA_HI_need2pix; del cosmo_HI#; del map_input_HI_need2pix; 

ell=np.arange(lmax+1)
factor=ell*(ell+1)/(2*np.pi)

fig = plt.figure()
plt.title(f'NEEDLETS CLs: channel:{nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}')
plt.semilogy(ell[1:],cl_PCA_HI_need2harm[ich][1:], label='PCA HI')
plt.semilogy(ell[1:],cl_cosmo_HI[ich][1:], label='Cosmo')
plt.semilogy(ell[1:],cl_cosmo_HI_recons[ich][1:], label='Cosmo reconstructed')
plt.legend()
plt.show()

#np.savetxt(out_dir_cls+f'cls_need2harm_PCA_HI_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.dat', cl_PCA_HI_need2harm)


fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: mean over channels, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}')
plt.plot(ell[1:], factor[1:]*cl_PCA_HI_need2harm.mean(axis=0)[1:],'+',mfc='none', label = f'PCA')
plt.plot(ell[1:], factor[1:]*cl_cosmo_HI.mean(axis=0)[1:], label = f'Cosmo')
plt.plot(ell[1:], factor[1:]*cl_cosmo_HI_recons.mean(axis=0)[1:], label = f'Cosmo reconstructed')
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,lmax+1, 10))


diff_cl_need2sphe = cl_PCA_HI_need2harm/cl_cosmo_HI-1
diff_cl_need2sphe_cosmo_recons = cl_cosmo_HI_recons/cl_cosmo_HI-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[1:], diff_cl_need2sphe.mean(axis=0)[1:]*100, label='% PCA_HI/input_HI -1')
plt.plot(ell[1:], diff_cl_need2sphe_cosmo_recons.mean(axis=0)[1:]*100, label=f'% recons_HI/input_HI -1')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
frame1.set_xticks(np.arange(1,lmax+1, 10))
plt.tight_layout()
plt.legend()
#plt.savefig(out_dir_plot+f'cls_need2pix_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.png')

plt.show()

del diff_cl_need2sphe; del cl_PCA_HI_need2harm; del diff_cl_need2sphe_cosmo_recons


#######################################################################
############################ LEAKAGE ##################################
print(' qua ci sono ')

need_fg_leak=np.load(path_leak_Fg+'.npy')
need_HI_leak=np.load(path_leak_HI+'.npy')


map_leak_HI_need2pix=np.zeros((len(nu_ch), npix))
map_leak_fg_need2pix=np.zeros((len(nu_ch), npix))
for nu in range(len(nu_ch)):
    map_leak_HI_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(need_HI_leak[:,nu],B, lmax)
    map_leak_fg_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(need_fg_leak[:,nu],B, lmax)
del need_HI_leak; del need_fg_leak
#np.save(out_dir_maps_recon+f'maps_reconstructed_leak_HI_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_leak_HI_need2pix)
#np.save(out_dir_maps_recon+f'maps_reconstructed_leak_fg_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_leak_fg_need2pix)


#map_leak_HI_need2pix = np.load(out_dir_maps_recon+f'maps_reconstructed_leak_HI_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
#map_leak_fg_need2pix = np.load(out_dir_maps_recon+f'maps_reconstructed_leak_fg_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=20)
fig.add_subplot(211) 
hp.mollview(map_leak_HI_need2pix[ich],min=0, max=1, title= 'Leakage HI',cmap='viridis', hold=True)
fig.add_subplot(212) 
hp.mollview(map_leak_fg_need2pix[ich],min=0, max=1, title= 'Leakage Fg',cmap='viridis', hold= True)
plt.tight_layout()
#plt.savefig(out_dir_plot+f'maps_recons_leak_jmax{jmax}_lmax{lmax}_B{B:1.2f}_Nfg{Nfg}_nside{nside}.png')
plt.show()


cl_leak_HI = np.zeros((len(nu_ch), lmax+1))
cl_leak_fg = np.zeros((len(nu_ch), lmax+1))
cl_diff_leak = np.zeros((len(nu_ch), lmax+1))

for n in range(len(nu_ch)):
    cl_leak_HI[n] = hp.anafast(map_leak_HI_need2pix[n], lmax=lmax)#, use_pixel_weights=True)
    cl_leak_fg[n] = hp.anafast(map_leak_fg_need2pix[n], lmax=lmax)#, use_pixel_weights=True)
    cl_diff_leak[n] = hp.anafast(map_leak_HI_need2pix[n]-map_leak_fg_need2pix[n], lmax=lmax)#, use_pixel_weights=True)

#np.savetxt(out_dir_cls+f'cls_need2harm_leak_HI_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.dat', cl_leak_HI)
#np.savetxt(out_dir_cls+f'cls_need2harm_leak_fg_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.dat', cl_leak_fg)
#
del map_leak_HI_need2pix; del map_leak_fg_need2pix
#
fig=plt.figure()
plt.semilogy(ell[1:], factor[1:]*np.mean(cl_leak_fg, axis=0)[1:],mfc='none', label='Fg leakage')
plt.semilogy(ell[1:], factor[1:]*np.mean(cl_leak_HI, axis=0)[1:],mfc='none', label='HI leakage')
plt.title(f'NEEDLETS CLs: mean over channels, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell*(\ell+1)}{2\pi} \langle C_{\ell} \rangle$')
plt.legend()
#plt.savefig(out_dir_plot+f'cls_need2pix_leak_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.png')
plt.show()

fig = plt.figure()
plt.semilogy(ell[1:],np.mean(cl_diff_cosmo_PCA_HI_need2harm, axis=0)[1:],mfc='none', label='Cl diff Cosmo - PCA HI maps')
plt.semilogy(ell[1:],np.mean(cl_diff_leak, axis=0)[1:],mfc='none', label='Cl HI - Fg leakage')
plt.title(f'NEEDLETS CLs: mean over channels, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell*(\ell+1)}{2\pi} \langle C_{\ell} \rangle$')
plt.legend()
#plt.savefig(out_dir_plot+f'cls_need2pix_leak_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.png')
plt.show()

