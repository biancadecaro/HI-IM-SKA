import healpy as hp
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import os
import pickle

import seaborn as sns
sns.set_theme(style = 'white')
sns.color_palette(n_colors=15)
#sns.set_palette('husl',15)

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

###########################################################################3
fg_comp = 'synch_ff_ps_pol'
beam_s = 'theta40arcmin'
path_data_sims_tot = f'Sims/beam_{beam_s}_no_mean_sims_{fg_comp}_noise_40freq_905.0_1295.0MHz_thick10MHz_lmax383_nside128'
with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()


out_dir_output = 'PCA_needlets_output/'
out_dir_output_PCA = out_dir_output+f'PCA_maps/No_mean/Beam_{beam_s}_noise_mask0.39_unseen/'
out_dir_plot = out_dir_output+f'Plots_PCA_needlets/No_mean/Beam_{beam_s}_noise_mask0.39_unseen/'#noise_mask_patch_stripe82_noise_mask0.39
if not os.path.exists(out_dir_output):
        os.makedirs(out_dir_output)
if not os.path.exists(out_dir_output_PCA):
        os.makedirs(out_dir_output_PCA)

nu_ch= file['freq']
del file



need_dir = f'Maps_needlets/No_mean/Beam_{beam_s}_noise_mask0.39_unseen/'
need_tot_maps_filename = need_dir+f'bjk_maps_obs_noise_{fg_comp}_40freq_905.0_1295.0MHz_jmax4_lmax383_B4.42_nside128.npy'
need_tot_maps = np.load(need_tot_maps_filename)

jmax=need_tot_maps.shape[1]-1

num_freq = need_tot_maps.shape[0]
nu_ch = np.linspace(905.0, 1295.0, num_freq)
min_ch = min(nu_ch)
max_ch = max(nu_ch)
npix = need_tot_maps.shape[2]
nside = hp.npix2nside(npix)
lmax=3*nside-1#2*nside#
B=pow(lmax,(1./jmax))

######################################################################################
mask1_40 = hp.read_map('HFI_Mask_GalPlane_2048_R1.10.fits', field=1)#fsky 20 %
mask_40t = hp.ud_grade(mask1_40, nside_out=256)
mask_40 = hp.ud_grade(mask_40t, nside_out=nside)
del mask1_40
#mask_40s = hp.sphtfunc.smoothing(mask_40, 3*np.pi/180,lmax=lmax)
fsky  = np.mean(mask_40) 
########################################################################
bad_v = np.where(mask_40==0)


maskt =np.zeros(mask_40.shape)

del mask_40
maskt[bad_v]=  1
mask = ma.make_mask(maskt, shrink=False)

need_tot_maps_masked=ma.zeros(need_tot_maps.shape)

for n in range(num_freq):
    for jj in range(jmax+1):
        need_tot_maps_masked[n,jj]  =ma.MaskedArray(need_tot_maps[n,jj], mask=mask)#np.isnan(full_maps_freq_mask[n])


Cov_channels = np.zeros((jmax+1,num_freq, num_freq))

for j in range(Cov_channels.shape[0]):
    #Cov_channels[j] = ma.cov(need_tot_maps_masked[:,j,:])
    for c in range(0, num_freq):
        for cc in range(0, num_freq):
            Cov_channels[j,c,cc]=ma.dot(need_tot_maps_masked[c,j,:],need_tot_maps_masked[cc,j,:].T)
            Cov_channels[j,cc,c] = Cov_channels[j,c,cc]
    #corr_coeff = ma.corrcoef(need_tot_maps_masked)


##########################################################################
print(f'jmax:{jmax}, lmax:{lmax}, B:{B:1.2f}, num_freq:{num_freq}, min_ch:{min_ch}, max_ch:{max_ch}, nside:{nside}')

eigenval=np.zeros((Cov_channels.shape[0], num_freq))
eigenvec=np.zeros((Cov_channels.shape[0], num_freq,num_freq))
eigenval1=np.zeros((Cov_channels.shape[0], num_freq))
eigenvec1=np.zeros((Cov_channels.shape[0], num_freq,num_freq))
for j in range(eigenval.shape[0]):
    eigenvec[j], eigenval[j], Vr = np.linalg.svd(Cov_channels[j], full_matrices=True)#np.linalg.eigh(Cov_channels[j])
    #eigenval1[j], eigenvec1[j] = np.linalg.eigh(Cov_channels[j])#np.linalg.eigh(Cov_channels[j])

del Cov_channels

############################################################################

fig = plt.figure(figsize=(8,4))
for j in range(eigenval.shape[0]):
    plt.semilogy(np.arange(1,num_freq+1),eigenval[j],'--o',mfc='none',markersize=5,label=f'j={j}')

plt.legend(fontsize=12, ncols=2)
x_ticks = np.arange(-10,num_freq+10, 10)
ax = plt.gca()
ax.set(xlim=[-10,num_freq+10],xticks=x_ticks,xlabel="eigenvalue number",ylabel="$\\lambda$",title='STANDARD NEED - Eigenvalues')

#fig = plt.figure(figsize=(8,4))
#for j in range(eigenval1.shape[0]):
#    plt.semilogy(np.arange(1,num_freq+1),eigenval1[j],'--o',mfc='none',markersize=5,label=f'j={j}')
#
#plt.legend(fontsize=12, ncols=2)
#x_ticks = np.arange(-10,num_freq+10, 10)
#ax = plt.gca()
#ax.set(xlim=[-10,num_freq+10],xticks=x_ticks,xlabel="eigenvalue number",ylabel="$\\lambda$",title='STANDARD NEED - Eigenvalues')


plt.show()
##############################################################################

n_dim = np.zeros(jmax+1)
for j in range(jmax+1):
    AIC = np.zeros(num_freq)
    fun = eigenval[j] - np.log(eigenval[j]) - 1.0
    total = np.sum(fun)
    for r in range(1, num_freq + 1):
        if r < num_freq:
            total = total - fun[r - 1]
            AIC[r - 1] = 2 * r + total
        else:
            AIC[r - 1] = 2 * r
    n_dim[j] = max(np.where(AIC == np.ndarray.min(AIC))[0]) + 1
    if np.sum(fun) < np.ndarray.min(AIC):
        n_dim[j] = 0   # allow zero dimension 
    print(n_dim[j])

