import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py
sys.path.insert(1, '/home/bianca/Documents/gmca4im-master/scripts/')
import gmca4im_lib2 as g4i
import seaborn as sns
sns.set_theme(style = 'white')
sns.color_palette(n_colors=15)
#sns.set_palette('husl',15)

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)
################################################################
out_dir= 'GMCA_pixels_output/Maps_GMCA/No_mean/Beam/'
out_dir_plot = 'GMCA_pixels_output/Plots_GMCA/No_mean/Beam/'

if not os.path.exists(out_dir):
        os.makedirs(out_dir)
if not os.path.exists(out_dir_plot):
        os.makedirs(out_dir_plot)

################################################################

fg_components='synch_ff_ps'
path_data_sims_tot = f'Sims/beam_no_mean_sims_{fg_components}_40freq_905.0_1295.0MHz_nside256'

with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

nu_ch= file['freq']

num_freq = len(nu_ch)

nu0 =1420
print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')
print(f'corresponding to the redshift range z: [{min(nu0/nu_ch -1.0):.2f} - {max(nu0/nu_ch -1.0):.2f}] ')

HI_maps_freq = file['maps_sims_HI']
fg_maps_freq = file['maps_sims_fg']
full_maps_freq = file['maps_sims_tot']

########################################################################3

ich = int(num_freq/2)

fig=plt.figure()
hp.mollview(fg_maps_freq[ich], cmap='viridis', title=f'Input foreground, channel:{nu_ch[ich]} MHz', hold=True)
plt.show()

del file
############################################################################################

npix = np.shape(HI_maps_freq)[1]
nside = hp.get_nside(HI_maps_freq[0])
lmax=3*nside
num_sources = 3
print(f'nside:{nside}, lmax:{lmax}, num_ch:{num_freq}, min_ch:{min(nu_ch)}, max_ch:{max(nu_ch)}, Nfg:{num_sources}')

for nu in range(len(nu_ch)):
        alm_HI = hp.map2alm(HI_maps_freq[nu], lmax=lmax)
        HI_maps_freq[nu] = hp.alm2map(alm_HI, lmax=lmax, nside = nside)
        del alm_HI
        alm_fg = hp.map2alm(fg_maps_freq[nu], lmax=lmax)
        fg_maps_freq[nu] = hp.alm2map(alm_fg, lmax=lmax, nside = nside)
        del alm_fg
        alm_obs = hp.map2alm(full_maps_freq[nu], lmax=lmax)
        full_maps_freq[nu] = hp.alm2map(alm_obs, lmax=lmax, nside = nside)
        del alm_obs

Cov_channels=np.cov(full_maps_freq)

fig=plt.figure()
plt.imshow(Cov_channels, cmap='crest')
plt.xlabel('[MHz]')
plt.ylabel('[MHz]')
plt.colorbar()
plt.show()

eigenval, eigenvec= np.linalg.eig(Cov_channels)
fig= plt.figure(figsize=(7,4))
plt.semilogy(np.arange(1,num_freq+1),eigenval,'--.',mfc='none',markersize=10)
x_ticks = np.arange(-10, num_freq+1, 10 )
ax = plt.gca()
ax.set(xlim=[-10,num_freq+2],xticks=x_ticks,xlabel="eigenvalue number",ylabel="$\\lambda$",title='Eigenvalues')
plt.tight_layout()
plt.savefig(out_dir_plot+f'eigenvalues_Nfg_{fg_components}.png')
plt.show()

#############################################################################
############################# GMCA ##########################################

################   GMCA PARAMETERS   ##################
num_sources   = 3   # number of sources to be estimated
mints = 0.1 # min threshold (what is sparse compared to noise?)
nmax  = 100 # number of iterations (usually 100 is safe)
L0    = 0   # switch between L0 norm (1) or L1 norm (0)
#######################################################

# initial guess for the mixing matrix?
# i.e. we could start from GMCA-determined mix matrix
AInit = None

# we can impose a column of the mixing matrix
ColFixed = None

# we can whiten the data
whitening = False; epsi = 1e-3

# estimated mixing matrix:
Ae = g4i.run_GMCA(full_maps_freq,AInit,num_sources,mints,nmax,L0,ColFixed,whitening,epsi)
fig=plt.figure()
plt.suptitle('mixing matrix')
plt.imshow(Ae,cmap='crest')
plt.colorbar()
plt.show()

#####################################################################################

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
plt.plot(abs(Ae/np.linalg.norm(Ae,axis=0)),label='mix mat column')
plt.plot(FF_col/np.linalg.norm(FF_col),'m:',label='gal ff')

ax = plt.gca()
ax.set(ylim=[0.0,0.4],xlabel="frequency channel",ylabel="Spectral emission",title='GMCA-mixing matrix columns')
plt.legend(fontsize=12)
plt.show()
#############################################################################################
#Foreground's maps from GMCA
#### reconstructed maps by GMCA

piA = np.linalg.inv(Ae.T@Ae)@Ae.T
# reproject onto original maps 
Se_sph = piA@full_maps_freq # LS estimate of the sources in the pixel domain
res_fg_maps = Ae@Se_sph
del Se_sph, piA

#The foreground residual that leaks into the recovered signal and noise
fg_leakage = fg_maps_freq - Ae@np.linalg.inv(Ae.T@Ae)@Ae.T@fg_maps_freq
HI_leakage = Ae@np.linalg.inv(Ae.T@Ae)@Ae.T@HI_maps_freq

del Ae

res_HI = full_maps_freq - res_fg_maps

#########################################


np.save(out_dir+f'cosmo_HI_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz.npy',HI_maps_freq)
np.save(out_dir+f'res_GMCA_HI_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}.npy',res_HI)
np.save(out_dir+f'fg_leak_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}.npy',fg_leakage)
np.save(out_dir+f'HI_leak_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}.npy',HI_leakage)
np.save(out_dir+f'fg_input_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz.npy',fg_maps_freq)

#####################################################################################################################

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(np.abs(res_fg_maps[ich]/fg_maps_freq[ich]-1)*100,cmap='viridis', min=0, max=0.1, title=f'%(Res_fg/x_fg - 1), channel:{nu_ch[ich]}',unit='%' ,hold=True)
fig.add_subplot(222) 
hp.mollview(HI_maps_freq[ich], cmap='viridis', title=f'HI signal freq={nu_ch[ich]}',min=0, max =1,hold=True)
fig.add_subplot(223)
hp.mollview(res_HI[ich], title=f'GMCA HI freq={nu_ch[ich]}',min=0, max =1,cmap='viridis', hold=True)
plt.show()

fig=plt.figure(figsize=(10, 7))
fig.suptitle(f'channel: {nu_ch[ich]} MHz, lmax:{lmax}, Nfg:{num_sources}',fontsize=20)
fig.add_subplot(131) 
hp.gnomview(HI_maps_freq[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='Input HI', cmap='viridis', hold=True)
fig.add_subplot(132) 
hp.gnomview(res_HI[ich],rot=[-22,21], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='GMCA HI', cmap= 'viridis', hold=True)
fig.add_subplot(133) 
hp.gnomview(HI_maps_freq[ich]-res_HI[ich], rot=[-22,21],coord='G', reso=hp.nside2resol(nside, arcmin=True), min=-0.2, max=0.2, title='HI-GMCA', cmap= 'viridis', hold=True)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221)
hp.mollview(HI_maps_freq[ich]-res_HI[ich], title=f'Cosmo HI - GMCA Res HI freq={nu_ch[ich]}', min=0, max=0.5,cmap='viridis', hold=True)
fig.add_subplot(222)
hp.mollview(fg_leakage[ich], title=f'Foreground leakage freq={nu_ch[ich]}', min=0, max=0.5,cmap='viridis', hold=True)
fig.add_subplot(223)
hp.mollview(HI_leakage[ich], title=f'HI leakage freq={nu_ch[ich]}', min=0, max=0.5,cmap='viridis', hold=True)
plt.show()

###########################################################################################################
out_dir_cl = out_dir+'power_spectra_cls_from_healpix_maps/'
if not os.path.exists(out_dir_cl):
        os.makedirs(out_dir_cl)

lmax_cl = 2*nside

cl_Hi=np.zeros((num_freq, lmax_cl+1))
cl_fg=np.zeros((num_freq, lmax_cl+1))
cl_Hi_recons_Nfg=np.zeros((num_freq, lmax_cl+1))
cl_fg_leak_Nfg=np.zeros((num_freq, lmax_cl+1))
cl_HI_leak_Nfg=np.zeros((num_freq, lmax_cl+1))


for i in range(num_freq):
    cl_Hi[i] = hp.anafast(HI_maps_freq[i], lmax=lmax_cl)
    cl_fg[i] = hp.anafast(fg_maps_freq[i], lmax=lmax_cl)
    cl_Hi_recons_Nfg[i] = hp.anafast(res_HI[i], lmax=lmax_cl)
    cl_fg_leak_Nfg[i]=hp.anafast(fg_leakage[i], lmax=lmax_cl)
    cl_HI_leak_Nfg[i]=hp.anafast(HI_leakage[i], lmax=lmax_cl)

del res_HI; del HI_maps_freq; del fg_leakage; del HI_leakage;del fg_maps_freq

##################################################################################################
np.savetxt(out_dir_cl+f'cl_input_HI_lmax{lmax_cl}_nside{nside}.dat', cl_Hi)
np.savetxt(out_dir_cl+f'cl_input_fg_lmax{lmax_cl}_nside{nside}.dat', cl_fg)
np.savetxt(out_dir_cl+f'cl_GMCA_HI_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_Hi_recons_Nfg)
np.savetxt(out_dir_cl+f'cl_leak_HI_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_HI_leak_Nfg)
np.savetxt(out_dir_cl+f'cl_leak_fg_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_fg_leak_Nfg)
#################################################################################################

ell = np.arange(0, lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)

fig=plt.figure()
plt.semilogy(ell,factor*np.mean(cl_fg_leak_Nfg, axis=0),mfc='none', label='Fg leakage')
plt.semilogy(ell,factor*np.mean(cl_HI_leak_Nfg, axis=0),mfc='none', label='HI leakage')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell*(\ell+1)}{2\pi} \langle C_{\ell} \rangle$')
plt.xlim([0,200])
plt.legend()
plt.show()

fig=plt.figure()
plt.plot(ell[1:], factor[1:]*cl_Hi[ich][1:],mfc='none', label='Cosmo HI')
plt.plot(ell[1:], factor[1:]*cl_Hi_recons_Nfg[ich][1:],'+',mfc='none', label='GMCA HI')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell(\ell+1)}{2\pi} C_{\ell} $')
plt.xlim([15,200])
plt.legend()
plt.show()

fig=plt.figure()
plt.plot(ell[1:], factor[1:]*np.mean(cl_Hi, axis=0)[1:],mfc='none', label='Cosmo HI')
plt.plot(ell[1:], factor[1:]*np.mean(cl_Hi_recons_Nfg, axis=0)[1:],'+',mfc='none', label='GMCA HI')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle $')
plt.xlim([15,200])
plt.legend()
plt.show()


fig=plt.figure()
plt.plot(ell[1:], 100*(cl_Hi_recons_Nfg[ich]/cl_Hi[ich]-1)[1:],'--.',mfc='none')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\% C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 $')
plt.xlim([0,200])
plt.ylim([-10,10])
plt.tight_layout()
plt.show()
print(min(100*(cl_Hi_recons_Nfg[ich]/cl_Hi[ich]-1)), max(100*(cl_Hi_recons_Nfg[ich]/cl_Hi[ich]-1)))


fig=plt.figure()
plt.plot(ell[1:], 100*np.mean(cl_Hi_recons_Nfg/cl_Hi-1, axis=0)[1:],'--.',mfc='none')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\%\langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle$')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.xlim([0,200])
plt.ylim([-10,10])
plt.tight_layout()
plt.show()
print(min(100*np.mean(cl_Hi_recons_Nfg/cl_Hi-1, axis=0)), max(100*np.mean(cl_Hi_recons_Nfg/cl_Hi-1, axis=0)))