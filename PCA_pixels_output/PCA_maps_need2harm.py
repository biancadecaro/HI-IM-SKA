import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
sns.set(style = 'white')
#sns.set_palette('husl',15)

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

#print(sns.color_palette("husl").as_hex())
sns.palettes.color_palette()

from needlets_analysis import analysis
import cython_mylibc as pippo


nu_ch = np.linspace(901.0, 1299.0, 200)
num_freq = len(nu_ch)

ich=100


#res_HI = np.load('res_PCA_HI_200_901.0_1299.0MHz_Nfg3.npy')
#cosmo_HI = np.load('cosmo_HI_200_901.0_1299.0MHz.npy')
#fg_input = np.load('fg_input_200_901.0_1299.0MHz.npy')
fg_lkg = np.load('fg_leak_200_901.0_1299.0MHz_Nfg3.npy')
HI_lkg = np.load('HI_leak_200_901.0_1299.0MHz_Nfg3.npy')

Nfg=3
nside=128
npix=hp.nside2npix(nside)
lmax=3*nside-1#256#
lmax_cl=256#3*nside-1#
betajk_dir = './Need_betajk_PCA/'
out_dir = './Maps_betajk2harm/'
######################################################################################################
################################ jmax=15  ############################################################

jmax=15

need_analysis_HI_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_lkg)
need_analysis_fg_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_lkg)

B = need_analysis_fg_lkg.B
print(B)

#fname_leak_HI=f'bjk_maps_leak_HI_jmax{jmax}_lmax{lmax}_B{B:0.2f}_Nfg{Nfg}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_nside{nside}'
#fname_leak_fg=f'bjk_maps_leak_fg_jmax{jmax}_lmax{lmax}_B{B:0.2f}_Nfg{Nfg}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_nside{nside}'
#
#map_leak_HI_need = np.load(betajk_dir+fname_leak_HI+'.npy')
#map_leak_fg_need = np.load(betajk_dir+fname_leak_fg+'.npy')
#
#map_leak_HI_need2harm=np.zeros((len(nu_ch), npix))
#map_leak_fg_need2harm=np.zeros((len(nu_ch), npix))
#for nu in range(len(nu_ch)):
#    map_leak_HI_need2harm[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(map_leak_HI_need[nu],B, lmax)
#    map_leak_fg_need2harm[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(map_leak_fg_need[nu],B, lmax)
#np.save(out_dir+f'maps_reconstructed_PCA_leak_HI_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}',map_leak_HI_need2harm)
#np.save(out_dir+f'maps_reconstructed_PCA_leak_fg_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}',map_leak_fg_need2harm)
#del map_leak_HI_need; del map_leak_fg_need; del need_analysis_HI_lkg; del need_analysis_fg_lkg
del need_analysis_HI_lkg; del need_analysis_fg_lkg

map_leak_HI_need2harm = np.load(out_dir+f'maps_reconstructed_PCA_leak_HI_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}.npy')
map_leak_fg_need2harm = np.load(out_dir+f'maps_reconstructed_PCA_leak_fg_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}.npy')



fig = plt.figure(figsize=(10, 7))
plt.suptitle(f'Need2harm, jmax:{jmax}, mean over channel',fontsize=20)
fig.add_subplot(221) 
hp.mollview(map_leak_HI_need2harm.mean(axis=0),min=0, max=0.1, title= 'Leakage HI',cmap='viridis', hold= True)
fig.add_subplot(222) 
hp.mollview(map_leak_fg_need2harm.mean(axis=0),min=0, max=0.1, title= 'Leakage fg',cmap='viridis', hold=True)
fig.add_subplot(223)
hp.mollview(HI_lkg.mean(axis=0)-map_leak_HI_need2harm.mean(axis=0),min=0, max=0.1, title= 'HI Healpix - HI recons',cmap='viridis', hold=True)
fig.add_subplot(224)
hp.mollview(fg_lkg.mean(axis=0)-map_leak_fg_need2harm.mean(axis=0), title= 'Fg Healpix - Fg recons',cmap='viridis', hold=True)
plt.savefig(f'Plots_need_est/maps_need2harm_leak_mean_ch_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.png')
plt.show()

cl_leak_HI_need2harm_jmax15 = np.zeros((len(nu_ch), lmax_cl+1))
cl_leak_fg_need2harm_jmax15 = np.zeros((len(nu_ch), lmax_cl+1))

for n in range(len(nu_ch)):
    cl_leak_HI_need2harm_jmax15[n] = hp.anafast(map_leak_HI_need2harm[n], lmax=lmax_cl)
    cl_leak_fg_need2harm_jmax15[n] = hp.anafast(map_leak_fg_need2harm[n], lmax=lmax_cl)

del map_leak_HI_need2harm; del map_leak_fg_need2harm
######################################################################################################
################################ jmax=12  ############################################################

jmax=12

need_analysis_HI_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_lkg)
need_analysis_fg_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_lkg)

B = need_analysis_fg_lkg.B
print(B)

#fname_leak_HI=f'bjk_maps_leak_HI_jmax{jmax}_lmax{lmax}_B{B:0.2f}_Nfg{Nfg}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_nside{nside}'
#fname_leak_fg=f'bjk_maps_leak_fg_jmax{jmax}_lmax{lmax}_B{B:0.2f}_Nfg{Nfg}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_nside{nside}'
#
#map_leak_HI_need = np.load(betajk_dir+fname_leak_HI+'.npy')
#map_leak_fg_need = np.load(betajk_dir+fname_leak_fg+'.npy')
#
#map_leak_HI_need2harm=np.zeros((len(nu_ch), npix))
#map_leak_fg_need2harm=np.zeros((len(nu_ch), npix))
#for nu in range(len(nu_ch)):
#    map_leak_HI_need2harm[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(map_leak_HI_need[nu],B, lmax)
#    map_leak_fg_need2harm[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(map_leak_fg_need[nu],B, lmax)
#np.save(out_dir+f'maps_reconstructed_PCA_leak_HI_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}',map_leak_HI_need2harm)
#np.save(out_dir+f'maps_reconstructed_PCA_leak_fg_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}',map_leak_fg_need2harm)
#del map_leak_HI_need; del map_leak_fg_need; del need_analysis_HI_lkg; del need_analysis_fg_lkg
del need_analysis_HI_lkg; del need_analysis_fg_lkg

map_leak_HI_need2harm = np.load(out_dir+f'maps_reconstructed_PCA_leak_HI_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}.npy')
map_leak_fg_need2harm = np.load(out_dir+f'maps_reconstructed_PCA_leak_fg_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}.npy')



fig = plt.figure(figsize=(10, 7))
plt.suptitle(f'Need2harm, jmax:{jmax}, mean over channel',fontsize=20)
fig.add_subplot(221) 
hp.mollview(map_leak_HI_need2harm.mean(axis=0),min=0, max=0.1, title= 'Leakage HI',cmap='viridis', hold= True)
fig.add_subplot(222) 
hp.mollview(map_leak_fg_need2harm.mean(axis=0),min=0, max=0.1, title= 'Leakage fg',cmap='viridis', hold=True)
fig.add_subplot(223)
hp.mollview(HI_lkg.mean(axis=0)-map_leak_HI_need2harm.mean(axis=0),min=0, max=0.1, title= 'HI Healpix - HI recons',cmap='viridis', hold=True)
fig.add_subplot(224)
hp.mollview(fg_lkg.mean(axis=0)-map_leak_fg_need2harm.mean(axis=0), title= 'Fg Healpix - Fg recons',cmap='viridis', hold=True)
plt.savefig(f'Plots_need_est/maps_need2harm_leak_mean_ch_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.png')
plt.show()

cl_leak_HI_need2harm_jmax12 = np.zeros((len(nu_ch), lmax_cl+1))
cl_leak_fg_need2harm_jmax12 = np.zeros((len(nu_ch), lmax_cl+1))

for n in range(len(nu_ch)):
    cl_leak_HI_need2harm_jmax12[n] = hp.anafast(map_leak_HI_need2harm[n], lmax=lmax_cl)
    cl_leak_fg_need2harm_jmax12[n] = hp.anafast(map_leak_fg_need2harm[n], lmax=lmax_cl)

del map_leak_HI_need2harm; del map_leak_fg_need2harm

#####################################################################################################
################################ jmax=8  ############################################################
jmax=8

need_analysis_HI_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_lkg)
need_analysis_fg_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_lkg)

B = need_analysis_fg_lkg.B
print(B)

#fname_leak_HI=f'bjk_maps_leak_HI_jmax{jmax}_lmax{lmax}_B{B:0.2f}_Nfg{Nfg}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_nside{nside}'
#fname_leak_fg=f'bjk_maps_leak_fg_jmax{jmax}_lmax{lmax}_B{B:0.2f}_Nfg{Nfg}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_nside{nside}'
#
#map_leak_HI_need = np.load(betajk_dir+fname_leak_HI+'.npy')
#map_leak_fg_need = np.load(betajk_dir+fname_leak_fg+'.npy')
#
#map_leak_HI_need2harm=np.zeros((len(nu_ch), npix))
#map_leak_fg_need2harm=np.zeros((len(nu_ch), npix))
#for nu in range(len(nu_ch)):
#    map_leak_HI_need2harm[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(map_leak_HI_need[nu],B, lmax)
#    map_leak_fg_need2harm[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(map_leak_fg_need[nu],B, lmax)
#np.save(out_dir+f'maps_reconstructed_PCA_leak_HI_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}',map_leak_HI_need2harm)
#np.save(out_dir+f'maps_reconstructed_PCA_leak_fg_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}',map_leak_fg_need2harm)
#del map_leak_HI_need; del map_leak_fg_need; del need_analysis_HI_lkg; del need_analysis_fg_lkg
del need_analysis_HI_lkg; del need_analysis_fg_lkg

map_leak_HI_need2harm = np.load(out_dir+f'maps_reconstructed_PCA_leak_HI_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}.npy')
map_leak_fg_need2harm = np.load(out_dir+f'maps_reconstructed_PCA_leak_fg_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}.npy')



fig = plt.figure(figsize=(10, 7))
plt.suptitle(f'Need2harm, jmax:{jmax}, mean over channel',fontsize=20)
fig.add_subplot(221) 
hp.mollview(map_leak_HI_need2harm.mean(axis=0),min=0, max=0.1, title= 'Leakage HI',cmap='viridis', hold= True)
fig.add_subplot(222) 
hp.mollview(map_leak_fg_need2harm.mean(axis=0),min=0, max=0.1, title= 'Leakage fg',cmap='viridis', hold=True)
fig.add_subplot(223)
hp.mollview(HI_lkg.mean(axis=0)-map_leak_HI_need2harm.mean(axis=0),min=0, max=0.1, title= 'HI Healpix - HI recons',cmap='viridis', hold=True)
fig.add_subplot(224)
hp.mollview(fg_lkg.mean(axis=0)-map_leak_fg_need2harm.mean(axis=0), title= 'Fg Healpix - Fg recons',cmap='viridis', hold=True)
plt.savefig(f'Plots_need_est/maps_need2harm_leak_mean_ch_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.png')
plt.show()

cl_leak_HI_need2harm_jmax8 = np.zeros((len(nu_ch), lmax_cl+1))
cl_leak_fg_need2harm_jmax8 = np.zeros((len(nu_ch), lmax_cl+1))

for n in range(len(nu_ch)):
    cl_leak_HI_need2harm_jmax8[n] = hp.anafast(map_leak_HI_need2harm[n], lmax=lmax_cl)
    cl_leak_fg_need2harm_jmax8[n] = hp.anafast(map_leak_fg_need2harm[n], lmax=lmax_cl)

del map_leak_HI_need2harm; del map_leak_fg_need2harm
#####################################################################################################
################################ jmax=4  ############################################################
jmax=4

need_analysis_HI_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_lkg)
need_analysis_fg_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_lkg)

B = need_analysis_fg_lkg.B
print(B)

#fname_leak_HI=f'bjk_maps_leak_HI_jmax{jmax}_lmax{lmax}_B{B:0.2f}_Nfg{Nfg}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_nside{nside}'
#fname_leak_fg=f'bjk_maps_leak_fg_jmax{jmax}_lmax{lmax}_B{B:0.2f}_Nfg{Nfg}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_nside{nside}'
#map_leak_HI_need = np.load(betajk_dir+fname_leak_HI+'.npy')
#map_leak_fg_need = np.load(betajk_dir+fname_leak_fg+'.npy')
#map_leak_HI_need2harm=np.zeros((len(nu_ch), npix))
#map_leak_fg_need2harm=np.zeros((len(nu_ch), npix))
#for nu in range(len(nu_ch)):
#    map_leak_HI_need2harm[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(map_leak_HI_need[nu],B, lmax)
#    map_leak_fg_need2harm[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(map_leak_fg_need[nu],B, lmax)
#np.save(out_dir+f'maps_reconstructed_PCA_leak_HI_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}',map_leak_HI_need2harm)
#np.save(out_dir+f'maps_reconstructed_PCA_leak_fg_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}',map_leak_fg_need2harm)
#del map_leak_HI_need; del map_leak_fg_need; del need_analysis_HI_lkg; del need_analysis_fg_lkg
del need_analysis_HI_lkg; del need_analysis_fg_lkg

map_leak_HI_need2harm = np.load(out_dir+f'maps_reconstructed_PCA_leak_HI_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}.npy')
map_leak_fg_need2harm = np.load(out_dir+f'maps_reconstructed_PCA_leak_fg_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}.npy')



fig = plt.figure(figsize=(10, 7))
plt.suptitle(f'Need2harm, jmax:{jmax}, mean over channel',fontsize=20)
fig.add_subplot(221) 
hp.mollview(map_leak_HI_need2harm.mean(axis=0),min=0, max=0.1, title= 'Leakage HI',cmap='viridis', hold= True)
fig.add_subplot(222) 
hp.mollview(map_leak_fg_need2harm.mean(axis=0),min=0, max=0.1, title= 'Leakage fg',cmap='viridis', hold=True)
fig.add_subplot(223)
hp.mollview(HI_lkg.mean(axis=0)-map_leak_HI_need2harm.mean(axis=0),min=0, max=0.1, title= 'HI Healpix - HI recons',cmap='viridis', hold=True)
fig.add_subplot(224)
hp.mollview(fg_lkg.mean(axis=0)-map_leak_fg_need2harm.mean(axis=0), title= 'Fg Healpix - Fg recons',cmap='viridis', hold=True)
plt.savefig(f'Plots_need_est/maps_need2harm_leak_mean_ch_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.png')
plt.show()

cl_leak_HI_need2harm_jmax4 = np.zeros((len(nu_ch), lmax_cl+1))
cl_leak_fg_need2harm_jmax4 = np.zeros((len(nu_ch), lmax_cl+1))

for n in range(len(nu_ch)):
    cl_leak_HI_need2harm_jmax4[n] = hp.anafast(map_leak_HI_need2harm[n], lmax=lmax_cl)
    cl_leak_fg_need2harm_jmax4[n] = hp.anafast(map_leak_fg_need2harm[n], lmax=lmax_cl)

del map_leak_HI_need2harm; del map_leak_fg_need2harm

################################################################################################################################
################################################## COMPARISON CLS ##############################################################
cl_fg_leak_Nfg=np.zeros((num_freq, lmax_cl+1))
cl_HI_leak_Nfg=np.zeros((num_freq, lmax_cl+1))

for i in range(num_freq):
    cl_fg_leak_Nfg[i]=hp.anafast(fg_lkg[i], lmax=lmax_cl)
    cl_HI_leak_Nfg[i]=hp.anafast(HI_lkg[i], lmax=lmax_cl)

ell = np.arange(0, lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)
fig = plt.figure()
plt.suptitle(r'$C_{\ell}$ Leak$_{HI}$, mean over frequency channels')
plt.semilogy(ell,factor*np.mean(cl_fg_leak_Nfg, axis=0),mfc='none', label='Fg Leak')
plt.semilogy(ell,factor*np.mean(cl_HI_leak_Nfg, axis=0),mfc='none', label='HI Leak')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell*(\ell+1)}{2\pi} \langle C_{\ell} \rangle$')
plt.legend()
plt.show()


fig = plt.figure()
plt.suptitle(r'$C_{\ell}$ Leak$_{HI}$, mean over frequency channels')
plt.semilogy(ell,factor*np.mean(cl_HI_leak_Nfg, axis=0),mfc='none', label='From Healpix map')
plt.semilogy(ell,factor*np.mean(cl_leak_HI_need2harm_jmax4, axis=0),mfc='none', label='From reconstructed need maps, jmax=4')
plt.semilogy(ell,factor*np.mean(cl_leak_HI_need2harm_jmax8, axis=0),mfc='none', label='From reconstructed need maps, jmax=8')
plt.semilogy(ell,factor*np.mean(cl_leak_HI_need2harm_jmax12, axis=0),mfc='none', label='From reconstructed need maps, jmax=12')
plt.semilogy(ell,factor*np.mean(cl_leak_HI_need2harm_jmax15, axis=0),mfc='none', label='From reconstructed need maps, jmax=15')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell*(\ell+1)}{2\pi} \langle C_{\ell} \rangle$')
plt.legend()
plt.show()

diff_leak_HI_jmax4 = cl_leak_HI_need2harm_jmax4/cl_HI_leak_Nfg-1#((hp.pixwin(nside)**2)*cl_HI_leak_Nfg)-1
diff_leak_HI_jmax8 = cl_leak_HI_need2harm_jmax8/cl_HI_leak_Nfg-1#((hp.pixwin(nside)**2)*cl_HI_leak_Nfg)-1
diff_leak_HI_jmax12 = cl_leak_HI_need2harm_jmax12/cl_HI_leak_Nfg-1#((hp.pixwin(nside)**2)*cl_HI_leak_Nfg)-1
diff_leak_HI_jmax15 = cl_leak_HI_need2harm_jmax15/cl_HI_leak_Nfg-1#((hp.pixwin(nside)**2)*cl_HI_leak_Nfg)-1

fig = plt.figure()
plt.suptitle(r'% Relative difference $C_{\ell}$ Leak$_{HI}$, mean over frequency channels')
plt.plot(ell[1:],100*np.mean(diff_leak_HI_jmax4, axis=0)[1:],mfc='none', label='jmax=4')
plt.plot(ell[1:],100*np.mean(diff_leak_HI_jmax8, axis=0)[1:],mfc='none', label='jmax=8')
plt.plot(ell[1:],100*np.mean(diff_leak_HI_jmax12, axis=0)[1:],mfc='none', label='jmax=12')
plt.plot(ell[1:],100*np.mean(diff_leak_HI_jmax15, axis=0)[1:],mfc='none', label='jmax=15')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel(r'$\ell$')
plt.ylabel(r'%$ \langle C_{\ell}^{\text{Recon need maps}} / C_{\ell}^{\text{Healpix maps}} -1\rangle_{ch}$')
plt.legend()
plt.savefig(f'Plots_need_est/diff_HI_leak_need_recons_Healpix_maps_jmax_lmax{lmax}_lmax_cl{lmax_cl}_Nfg{Nfg}_nside{nside}.png')
plt.show()


fig = plt.figure()
plt.suptitle(r'$C_{\ell}$ Leak$_{fg}$, mean over frequency channels')
plt.semilogy(ell, factor*np.mean(cl_fg_leak_Nfg, axis=0),mfc='none', label='From Healpix map')
plt.semilogy(ell, factor*np.mean(cl_leak_fg_need2harm_jmax4, axis=0),mfc='none', label='From reconstructed need maps, jmax=4')
plt.semilogy(ell, factor*np.mean(cl_leak_fg_need2harm_jmax8, axis=0),mfc='none', label='From reconstructed need maps, jmax=8')
plt.semilogy(ell, factor*np.mean(cl_leak_fg_need2harm_jmax12, axis=0),mfc='none', label='From reconstructed need maps, jmax=12')
plt.semilogy(ell, factor*np.mean(cl_leak_fg_need2harm_jmax15, axis=0),mfc='none', label='From reconstructed need maps, jmax=15')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell*(\ell+1)}{2\pi} \langle C_{\ell} \rangle$')
plt.legend()
plt.show()

diff_leak_fg_jmax4 = cl_leak_fg_need2harm_jmax4/cl_fg_leak_Nfg-1#((hp.pixwin(nside)**2)*cl_fg_leak_Nfg)-1
diff_leak_fg_jmax8 = cl_leak_fg_need2harm_jmax8/cl_fg_leak_Nfg-1#((hp.pixwin(nside)**2)*cl_fg_leak_Nfg)-1
diff_leak_fg_jmax12 = cl_leak_fg_need2harm_jmax12/cl_fg_leak_Nfg-1#((hp.pixwin(nside)**2)*cl_fg_leak_Nfg)-1
diff_leak_fg_jmax15 = cl_leak_fg_need2harm_jmax15/cl_fg_leak_Nfg-1#((hp.pixwin(nside)**2)*cl_fg_leak_Nfg)-1

fig = plt.figure()
plt.suptitle(r'% Relative difference $C_{\ell}$ Leak$_{fg}$, mean over frequency channels')
plt.plot(ell[1:],100*np.mean(diff_leak_fg_jmax4, axis=0)[1:],mfc='none', label='jmax=4')
plt.plot(ell[1:],100*np.mean(diff_leak_fg_jmax8, axis=0)[1:],mfc='none', label='jmax=8')
plt.plot(ell[1:],100*np.mean(diff_leak_fg_jmax12, axis=0)[1:],mfc='none', label='jmax=12')
plt.plot(ell[1:],100*np.mean(diff_leak_fg_jmax15, axis=0)[1:],mfc='none', label='jmax=15')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel(r'$\ell$')
plt.ylabel(r'%$ \langle C_{\ell}^{\text{Recon need maps}} / C_{\ell}^{\text{Healpix maps}} -1 \rangle_{ch}$')
plt.legend()
plt.savefig(f'Plots_need_est/diff_fg_leak_need_recons_Healpix_maps_jmax_lmax{lmax}_lmax_cl{lmax_cl}_Nfg{Nfg}_nside{nside}.png')
plt.show()


#fig = plt.figure()
#plt.suptitle('Pixel windows function, nside=128')
#plt.plot(ell, hp.pixwin(nside))
#plt.xlabel(r'$\ell$')
#plt.ylabel(r'$w_{\ell}$')
#plt.tight_layout()
#plt.show()


##############################################
''' Facciamo una cosa stupida 
prendiamo il cl e torniamo indietro alla mappa e vediamo se viene quella cosa strana
Non ha molto senso fare questa cosa perchè genera un solo alm dalla distribuzione gaussiana
con varianza cl e quindi è random'''''

#np.random.seed(1234)
#cls2map = hp.synfast(cls=cl_HI_leak_Nfg[ich], nside=nside)
#np.random.seed(4635345)
#cls2map_need2harm = hp.synfast(cls=cl_leak_HI_need2harm_jmax8[ich], nside=nside)
##fig = plt.figure()
##fig.add_subplot(211) 
##hp.mollview(cls2map, title=f'Cls2Map HI Leak, channel={nu_ch[ich]} MHz',  cmap='viridis', min=0, max=0.1, hold=True  )
##fig.add_subplot(212) 
##hp.mollview(HI_lkg[ich], title=f'HI Leak from PCA, channel={nu_ch[ich]} MHz',  cmap='viridis', min=0, max=0.1, hold=True  )
###plt.savefig(f'Plots_need_est/cls2map_HI_leak_Healpix_lmax{lmax}_Nfg{Nfg}_nside{nside}.png')
##plt.show()
#
#fig = plt.figure()
#fig.add_subplot(211) 
#hp.mollview(cls2map, title=f'Cls2Map HI Leak, channel={nu_ch[ich]} MHz',  cmap='viridis', min=0, max=0.1, hold=True  )
#fig.add_subplot(212) 
#hp.mollview(cls2map_need2harm, title=f'Cls2Map Need2Harm jmax=8 HI Leak channel={nu_ch[ich]} MHz',  cmap='viridis', min=0, max=0.1, hold=True  )
##plt.savefig(f'Plots_need_est/cls2map_HI_leak_Healpix_lmax{lmax}_Nfg{Nfg}_nside{nside}.png')
#plt.show()
rs = np.random.randint(543543, size=4)
print(rs)
cls2map_need2harm = np.zeros((len(rs), npix))
for n in range(len(rs)):
    np.random.seed(rs[n])
    cls2map_need2harm[n] = hp.synfast(cls=cl_leak_HI_need2harm_jmax8[ich], nside=nside, lmax=lmax)

fig = plt.figure()
fig.add_subplot(221) 
hp.mollview(cls2map_need2harm[0], title=f'seed={rs[0]}, channel={nu_ch[ich]} MHz',  cmap='viridis', min=0, max=0.1, hold=True  )
fig.add_subplot(222) 
hp.mollview(cls2map_need2harm[1], title=f'seed={rs[1]}, channel={nu_ch[ich]} MHz',  cmap='viridis', min=0, max=0.1, hold=True  )
fig.add_subplot(223) 
hp.mollview(cls2map_need2harm[2], title=f'seed={rs[2]}, channel={nu_ch[ich]} MHz',  cmap='viridis', min=0, max=0.1, hold=True  )
fig.add_subplot(224) 
hp.mollview(cls2map_need2harm[3], title=f'seed={rs[3]}, channel={nu_ch[ich]} MHz',  cmap='viridis', min=0, max=0.1, hold=True  )
plt.show()
