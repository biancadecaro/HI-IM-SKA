import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
sns.set(style = 'white')
sns.set_palette('husl',15)

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

print(sns.color_palette("husl").as_hex())
sns.palettes.color_palette()

from needlets_analysis import analysis
import cython_mylibc as pippo

#########################
####### jmax=8  ########

nside=128
lmax=3*nside-1
jmax=8
npix = hp.nside2npix(nside)

out_dir = './'

res_HI = np.load('res_PCA_HI_200_901.0_1299.0MHz_Nfg3.npy')
cosmo_HI = np.load('/home/bianca/Documents/HI IM SKA/PCA_pixels_output/cosmo_HI_200_901.0_1299.0MHz.npy')

need_analysis_res = analysis.NeedAnalysis(jmax, lmax, out_dir, res_HI)
need_analysis_HI = analysis.NeedAnalysis(jmax, lmax, out_dir, cosmo_HI)

B_res = need_analysis_res.B
B_HI = need_analysis_HI.B
print(B_res)

fname_res=f'bjk_maps_res_HI_200freq_901.0_1299.0MHz_Nfg3_jmax{jmax}_B{B_res:0.2f}_lmax{lmax}_nside{nside}'
fname_HI=f'bjk_maps_cosmo_HI_200freq_901.0_1299.0MHz_Nfg3_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}'

res_HI_need_output = need_analysis_res.GetBetajkSims(map_input=res_HI, nfreq=200, fname=fname_res)
cosmo_HI_need_output = need_analysis_HI.GetBetajkSims(map_input=cosmo_HI, nfreq=200, fname=fname_HI)
#cosmo_HI_need_output = np.load('/home/bianca/Documents/HI IM SKA/Maps_test_2/bjk_maps_HI_200freq_901.0_1299.0MHz_jmax12_B1.59_nside128.npy')

nu_ch = np.linspace(901.0, 1299.0, 200)

ich=100

#j_test=11
#for j_test in range(jmax+1):
#    fig = plt.figure(figsize=(10, 7))
#    fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
#    hp.mollview(cosmo_HI_need_output[ich,j_test], cmap='viridis', title=f'Res PCA HI, j={j_test}, freq={nu_ch[ich]}', min=0, max=1,hold=True)
#    plt.show()

#ottenere spettro betaj
HI_PCA_need_est = np.zeros((len(nu_ch), jmax+1))
HI_cosmo_need_est = np.zeros((len(nu_ch), jmax+1))
for ch in range(len(nu_ch)):
    HI_PCA_need_est[ch] =  need_analysis_res.Betajk2Betaj(betajk1=res_HI_need_output[ch])
    HI_cosmo_need_est[ch] =  need_analysis_HI.Betajk2Betaj(betajk1=cosmo_HI_need_output[ch])

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS ESTIMATORS mean over channels, B:{B_res:0.2f}, jmax:{jmax}, lmax:{lmax}')
plt.plot(np.arange(1,jmax+1), HI_PCA_need_est.mean(axis=0)[1:], label = f'PCA')
plt.plot(np.arange(1,jmax+1), HI_cosmo_need_est.mean(axis=0)[1:],c= '#3ba3ec', label = f'Cosmo')
plt.legend()
frame1.set_ylabel(r'$\langle \beta_j^{\rm HI} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,jmax+1))



diff_j8 = HI_PCA_need_est/HI_cosmo_need_est-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(np.arange(1,jmax+1), diff_j8.mean(axis=0)[1:]*100)
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel('j')
frame2.set_xticks(np.arange(1,jmax+1))
plt.show()

del HI_cosmo_need_est; del HI_PCA_need_est


###############################
##### SPHERICAL HARMONICS #####

ell = np.arange(lmax+1)
cl_cosmo_HI = np.zeros((len(nu_ch), lmax+1))
cl_res_HI = np.zeros((len(nu_ch), lmax+1))

for nu in range(len(nu_ch)):
    cl_cosmo_HI[nu] = hp.anafast(map1=cosmo_HI[nu],lmax=lmax)
    cl_res_HI[nu] = hp.anafast(map1=res_HI[nu],lmax=lmax)


factor=ell*(ell+1)/(2*np.pi)

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'SPHERICAL HARMONICS CLs mean over channels, lmax:{lmax}')
plt.plot(ell[2:], factor[2:]*cl_res_HI.mean(axis=0)[2:], label = f'PCA')
plt.plot(ell[2:], factor[2:]*cl_cosmo_HI.mean(axis=0)[2:],c= '#3ba3ec', label = f'Cosmo')
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(2,lmax+1, 10))


diff_cl = cl_res_HI/cl_cosmo_HI-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl.mean(axis=0)[2:]*100)
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel('j')
frame1.set_xticks(np.arange(2,lmax+1, 10))
plt.show()

del cl_res_HI; del cl_cosmo_HI

fig = plt.figure()
plt.plot(diff_cl.mean(axis= 0)[2:], label = 'Diff spherical harmonics')
plt.plot(diff_j8.mean(axis=0)[1:], label = 'Diff needlet estimator')
plt.ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
plt.xlabel('j')
plt.legend()
plt.show()

############################
#### NEEDLETS2HARMONICS ####

#cosmo_HI_need_jtot = cosmo_HI_need_output.sum(axis=1)
#res_HI_need_jtot = res_HI_need_output.sum(axis=1)
#
##map_cosmo_HI_need2pix=np.zeros((len(nu_ch), npix))
##map_PCA_HI_need2pix=np.zeros((len(nu_ch), npix))
##for nu in range(len(nu_ch)):
##    map_cosmo_HI_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(cosmo_HI_need_output[nu],B_HI, lmax)
##    map_PCA_HI_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(res_HI_need_output[nu],B_HI, lmax)
#
#map_cosmo_HI_need2pix=np.load(out_dir+f'maps_cosmo_HI_need2pix_jmax{jmax}_B{B_HI:1.2f}_lmax{lmax}_Nfg3.npy')
#map_PCA_HI_need2pix=np.load(out_dir+f'maps_PCA_HI_need2pix_jmax{jmax}_B{B_HI:1.2f}_lmax{lmax}_Nfg3.npy')
#
#fig = plt.figure(figsize=(10, 7))
#fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
#fig.add_subplot(221) 
#hp.mollview(map_cosmo_HI_need2pix[nu],min=0, max=1, title= 'Cosmo HI Needlets 2 Pix',cmap='viridis',cbar=False, hold= True)
#fig.add_subplot(222) 
#hp.mollview(cosmo_HI_need_jtot[ich],min=0, max=1, title= 'Cosmo HI Betajk',cmap='viridis',cbar=False, hold=True)
#
#fig.add_subplot(223) 
#hp.mollview(map_PCA_HI_need2pix[nu],min=0, max=1, title= 'Res PCA HI Needlets 2 Pix',cmap='viridis',cbar=False, hold= True)
#fig.add_subplot(224) 
#hp.mollview(res_HI_need_jtot[ich],min=0, max=1, title= 'Res PCA HI Betajk',cmap='viridis',cbar=False, hold=True)
#plt.show()
#
#del res_HI_need_jtot; del cosmo_HI_need_jtot
#
#cl_cosmo_HI_need2harm = np.zeros((len(nu_ch), lmax+1))
#cl_PCA_HI_need2harm = np.zeros((len(nu_ch), lmax+1))
#
#for n in range(len(nu_ch)):
#    cl_cosmo_HI_need2harm[n] = hp.anafast(map_cosmo_HI_need2pix[n], lmax=lmax)
#    cl_PCA_HI_need2harm[n] = hp.anafast(map_PCA_HI_need2pix[n], lmax=lmax)
#
#del map_PCA_HI_need2pix; del map_cosmo_HI_need2pix
#
#
#fig = plt.figure(figsize=(10,7))
#frame1=fig.add_axes((.1,.3,.8,.6))
#plt.title(f'NEEDLETS CLs: mean over channels, lmax:{lmax}')
#plt.plot(ell[2:], factor[2:]*cl_PCA_HI_need2harm.mean(axis=0)[2:], label = f'PCA')
#plt.plot(ell[2:], factor[2:]*cl_cosmo_HI_need2harm.mean(axis=0)[2:],c= '#3ba3ec', label = f'Cosmo')
#plt.legend()
#frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
#frame1.set_xlabel([])
#frame1.set_xticks(np.arange(2,lmax+1, 10))
#
#
#diff_cl_need2sphe = cl_PCA_HI_need2harm/cl_cosmo_HI_need2harm-1
#
#frame2=fig.add_axes((.1,.1,.8,.2))
#plt.plot(ell[2:], diff_cl_need2sphe.mean(axis=0)[2:]*100)
#frame2.axhline(ls='--', c= 'k', alpha=0.3)
#frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
#frame2.set_xlabel('j')
#frame1.set_xticks(np.arange(2,lmax+1, 10))
#plt.show()