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

#jmax=8
npix= hp.nside2npix(nside)

out_dir_plot = 'Plots_PCA_needlets/'
dir_PCA = 'PCA_maps/'
out_dir_maps_recon = 'maps_reconstructed/'
out_dir_cls = 'cls_need2harm/'

path_PCA_HI=dir_PCA+'res_PCA_HI_jmax8_lmax256_200_901_1299MHz_Nfg3'
path_PCA_fg=dir_PCA+'res_PCA_fg_sync_ff_ps_jmax8_lmax256_901_1299MHz_Nfg3'
path_cosmo_HI = '../PCA_pixels_output/cosmo_HI_200_901.0_1299.0MHz'
path_fg = '../PCA_pixels_output/fg_input_200_901.0_1299.0MHz'
path_leak_Fg = dir_PCA+'leak_PCA_fg_sync_ff_ps_jmax8_lmax256_901_1299MHz_Nfg3'
path_leak_HI = dir_PCA+'leak_PCA_HI_jmax8_lmax256_901_1299MHz_Nfg3'

jmax=int(re.findall(r'\d+', path_PCA_HI)[0])
lmax=int(re.findall(r'\d+', path_PCA_HI)[1])
num_ch=int(re.findall(r'\d+', path_PCA_HI)[2])
min_ch = float(re.findall(r'\d+', path_PCA_HI)[3])
max_ch = float(re.findall(r'\d+', path_PCA_HI)[4])
Nfg = int(re.findall(r'\d+', path_PCA_HI)[5])

print(f'jmax:{jmax}, lmax:{lmax}, num_ch:{num_ch}, min_ch:{min_ch}, max_ch:{max_ch}, Nfg:{Nfg}')

nu_ch = np.linspace(min_ch, max_ch, num_ch)
del min_ch;del max_ch

res_PCA_HI = np.load(path_PCA_HI+'.npy')
res_PCA_fg = np.load(path_PCA_fg+'.npy')
cosmo_HI = np.load(path_cosmo_HI+'.npy')
fg = np.load(path_fg+'.npy')


ich=100

#for j in range(jmax+1):
#    hp.mollview(res_PCA_HI[j, ich],cmap='viridis',min=0, max=1, title=f'Res PCA HI, j={j}, jmax={jmax}, Nfg:{Nfg}',hold=True,cbar=True)
#    plt.show()
#
#del res_PCA_HI; del res_PCA_fg
############################################################################################
####################### NEEDLETS2HARMONICS #################################################
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)
print(B)

map_PCA_HI_need2pix=np.zeros((len(nu_ch), npix))
map_PCA_fg_need2pix=np.zeros((len(nu_ch), npix))
for nu in range(len(nu_ch)):
    map_PCA_HI_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(res_PCA_HI[:,nu],B, lmax)
    map_PCA_fg_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(res_PCA_fg[:,nu],B, lmax)
np.save(out_dir_maps_recon+f'maps_reconstructed_PCA_HI_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_PCA_HI_need2pix)
np.save(out_dir_maps_recon+f'maps_reconstructed_PCA_fg_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_PCA_fg_need2pix)
del res_PCA_HI; del res_PCA_fg

#map_PCA_HI_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_PCA_HI_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
#map_PCA_fg_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_PCA_fg_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')


fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(100*(map_PCA_fg_need2pix[ich]/fg[ich]-1), title= '(Res fg/fg-1)%',cmap='viridis',unit='%', hold= True)
fig.add_subplot(222) 
hp.mollview(cosmo_HI[ich],min=0, max=1, title= 'Cosmo HI',cmap='viridis', hold=True)
fig.add_subplot(223) 
hp.mollview(map_PCA_HI_need2pix[ich],min=0, max=1, title= 'Res PCA HI Needlets 2 Pix',cmap='viridis', hold= True)
plt.tight_layout()
plt.savefig(out_dir_plot+f'maps_recons_jmax{jmax}_lmax{lmax}_B{B:1.2f}_Nfg{Nfg}_nside{nside}.png')
plt.show()


cl_cosmo_HI = np.zeros((len(nu_ch), lmax+1))
cl_PCA_HI_need2harm = np.zeros((len(nu_ch), lmax+1))
cl_diff_cosmo_PCA_HI_need2harm = np.zeros((len(nu_ch), lmax+1))

for n in range(len(nu_ch)):
    cl_cosmo_HI[n] = hp.anafast(cosmo_HI[n], lmax=lmax, use_pixel_weights=True)
    cl_PCA_HI_need2harm[n] = hp.anafast(map_PCA_HI_need2pix[n], lmax=lmax, use_pixel_weights=True)
    cl_diff_cosmo_PCA_HI_need2harm[n] = hp.anafast(cosmo_HI[n]-map_PCA_HI_need2pix[n], lmax=lmax, use_pixel_weights=True)

ell=np.arange(lmax+1)
factor=ell*(ell+1)/(2*np.pi)

fig = plt.figure()
plt.semilogy(ell[2:],cl_PCA_HI_need2harm[ich][2:], label='PCA HI')
plt.semilogy(ell[2:],cl_cosmo_HI[ich][2:], label='Cosmo')
plt.legend()
plt.show()

np.savetxt(out_dir_cls+f'cls_need2harm_PCA_HI_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.dat', cl_PCA_HI_need2harm)

del map_PCA_HI_need2pix; del cosmo_HI

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: mean over channels, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_need2harm.mean(axis=0)[2:], label = f'PCA')
plt.plot(ell[2:], factor[2:]*cl_cosmo_HI.mean(axis=0)[2:], label = f'Cosmo')
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(2,lmax+1, 10))


diff_cl_need2sphe = cl_PCA_HI_need2harm/cl_cosmo_HI-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe.mean(axis=0)[2:]*100)
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
frame1.set_xticks(np.arange(2,lmax+1, 10))
plt.tight_layout()
plt.savefig(out_dir_plot+f'cls_need2pix_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.png')

plt.show()

del diff_cl_need2sphe; del cl_PCA_HI_need2harm

#############################################################################################
#cls_Nfg6_jmax12 = np.loadtxt('cls_need2harm_PCA_HI_jmax8_lmax383_nside128_Nfg6.dat')
#cls_Nfg6_jmax8 = np.loadtxt('cls_need2harm_PCA_HI_jmax12_lmax383_nside128_Nfg6.dat')
#
#diff_Nfg6_jmax12 = cls_Nfg6_jmax12/cl_cosmo_HI-1
#diff_Nfg6_jmax8 = cls_Nfg6_jmax8/cl_cosmo_HI-1
#
#
#fig, ax = plt.subplots(1,1)
#plt.title(f'NEEDLETS CLs: mean over channels, lmax:{lmax}, Nfg:6')
#ax.plot(ell[2:], diff_Nfg6_jmax12.mean(axis=0)[2:]*100, label = 'jmax=12')
#ax.plot(ell[2:], diff_Nfg6_jmax8.mean(axis=0)[2:]*100, label = 'jmax=8')
#ax.axhline(ls='--', c= 'k', alpha=0.3)
#ax.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
#ax.set_xlabel(r'$\ell$')
##ax.set_xticks(np.arange(2,lmax+1, 20))
#plt.legend()
#plt.tight_layout()
#plt.savefig(out_dir_plot+f'diff_cls_need2harm_jmax8_jmax12_Nfg6_lmax{lmax}_nside{nside}.png')
#plt.show()
#
#fig, ax = plt.subplots(1,1)
#plt.title(f'NEEDLETS CLs: mean over channels, lmax:{lmax}, Nfg:6')
#ax.plot(ell[2:], (diff_Nfg6_jmax12.mean(axis=0)[2:]/diff_Nfg6_jmax8.mean(axis=0)[2:]-1)*100)
#ax.axhline(ls='--', c= 'k', alpha=0.3)
#ax.set_ylabel(r'%$ diff_{jmax=12}/diff_{jmax=12}-1$')
#ax.set_xlabel(r'$\ell$')
##ax.set_xticks(np.arange(2,lmax+1, 20))
#plt.legend()
#plt.tight_layout()
#plt.savefig(out_dir_plot+f'diff_of_the_diff_cls_need2harm_jmax8_jmax12_Nfg6_lmax{lmax}_nside{nside}.png')
#plt.show()
#
#del diff_Nfg6_jmax8; del cls_Nfg6_jmax8; del cl_cosmo_HI; del diff_Nfg6_jmax12; del cls_Nfg6_jmax12


################################################################################
############################ LEAKAGE ###########################################
print('qua ci sono?')

need_fg_leak=np.load(path_leak_Fg+'.npy')
need_HI_leak=np.load(path_leak_HI+'.npy')


map_leak_HI_need2pix=np.zeros((len(nu_ch), npix))
map_leak_fg_need2pix=np.zeros((len(nu_ch), npix))
for nu in range(len(nu_ch)):
    map_leak_HI_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(need_HI_leak[:,nu],B, lmax)
    map_leak_fg_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(need_fg_leak[:,nu],B, lmax)
del need_HI_leak; del need_fg_leak
np.save(out_dir_maps_recon+f'maps_reconstructed_leak_HI_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_leak_HI_need2pix)
np.save(out_dir_maps_recon+f'maps_reconstructed_leak_fg_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_leak_fg_need2pix)

#map_leak_HI_need2pix = np.load(out_dir_maps_recon+f'maps_reconstructed_leak_HI_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
#map_leak_fg_need2pix = np.load(out_dir_maps_recon+f'maps_reconstructed_leak_fg_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(211) 
hp.mollview(map_leak_HI_need2pix[ich],min=0, max=1, title= 'Leakage HI',cmap='viridis', hold=True)
fig.add_subplot(212) 
hp.mollview(map_leak_fg_need2pix[ich],min=0, max=1, title= 'Leakage Fg',cmap='viridis', hold= True)
plt.tight_layout()
plt.savefig(out_dir_plot+f'maps_recons_leak_jmax{jmax}_lmax{lmax}_B{B:1.2f}_Nfg{Nfg}_nside{nside}.png')
plt.show()


cl_leak_HI = np.zeros((len(nu_ch), lmax+1))
cl_leak_fg = np.zeros((len(nu_ch), lmax+1))
cl_diff_leak = np.zeros((len(nu_ch), lmax+1))

for n in range(len(nu_ch)):
    cl_leak_HI[n] = hp.anafast(map_leak_HI_need2pix[n], lmax=lmax, use_pixel_weights=True)
    cl_leak_fg[n] = hp.anafast(map_leak_fg_need2pix[n], lmax=lmax, use_pixel_weights=True)
    cl_diff_leak[n] = hp.anafast(map_leak_HI_need2pix[n]-map_leak_fg_need2pix[n], lmax=lmax, use_pixel_weights=True)

np.savetxt(out_dir_cls+f'cls_need2harm_leak_HI_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.dat', cl_leak_HI)
np.savetxt(out_dir_cls+f'cls_need2harm_leak_fg_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.dat', cl_leak_fg)
#
del map_leak_HI_need2pix; del map_leak_fg_need2pix
#
fig=plt.figure()
plt.semilogy(factor*np.mean(cl_leak_fg, axis=0),mfc='none', label='Fg leakage')
plt.semilogy(factor*np.mean(cl_leak_HI, axis=0),mfc='none', label='HI leakage')
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


#cls_leak_fg_jmax8=np.loadtxt('cls_need2harm_leak_fg_jmax8_lmax383_nside128_Nfg6.dat')
#cls_leak_HI_jmax8=np.loadtxt('cls_need2harm_leak_HI_jmax8_lmax383_nside128_Nfg6.dat')
#
#plt.semilogy(factor*np.mean(cl_leak_fg, axis=0),mfc='none', label='Fg leakage, jmax=12')
#plt.semilogy(factor*np.mean(cl_leak_HI, axis=0),mfc='none', label='HI leakage, jmax=12')
#plt.semilogy(factor*np.mean(cls_leak_fg_jmax8, axis=0),mfc='none', label='Fg leakage, jmax=8')
#plt.semilogy(factor*np.mean(cls_leak_HI_jmax8, axis=0),mfc='none', label='HI leakage, jmax=8')
#plt.title(f'NEEDLETS CLs: mean over channels, lmax:{lmax}, Nfg:{Nfg}')
#plt.xlabel(r'$\ell$')
#plt.ylabel(r'$ \frac{\ell*(\ell+1)}{2\pi} \langle C_{\ell} \rangle$')
#plt.legend()
#plt.savefig(out_dir_plot+f'cls_need2pix_leak_jmax{jmax}_jmax8_lmax{lmax}_nside{nside}_Nfg{Nfg}.png')
#plt.show()

#############################################################################################
#cls_Nfg3_jmax8 = np.loadtxt('cls_need2harm_PCA_HI_jmax8_lmax383_nside128_Nfg3.dat')
#cls_Nfg6_jmax8 = np.loadtxt('cls_need2harm_PCA_HI_jmax8_lmax383_nside128_Nfg6.dat')
#
#diff_Nfg3_jmax8 = cls_Nfg3_jmax8/cl_cosmo_HI-1
#diff_Nfg6_jmax8 = cls_Nfg6_jmax8/cl_cosmo_HI-1
#
#
#fig, ax = plt.subplots(1,1)
#plt.title(f'NEEDLETS CLs: mean over channels, lmax:{lmax}, jmax:{jmax}')
#ax.plot(ell[2:], diff_Nfg3_jmax8.mean(axis=0)[2:]*100, label = 'Nfg=3')
#ax.plot(ell[2:], diff_Nfg6_jmax8.mean(axis=0)[2:]*100, label = 'Nfg=6')
#ax.axhline(ls='--', c= 'k', alpha=0.3)
#ax.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
#ax.set_xlabel(r'$\ell$')
##ax.set_xticks(np.arange(2,lmax+1, 20))
#plt.legend()
#plt.tight_layout()
#plt.savefig(out_dir_plot+f'diff_cls_need2harm_Nfg3_Nfg6_jmax{jmax}_lmax{lmax}_nside{nside}.png')
#plt.show()
#
#del diff_Nfg6_jmax8; del cls_Nfg6_jmax8; del cl_cosmo_HI; del diff_Nfg3_jmax8; del cls_Nfg3_jmax8



