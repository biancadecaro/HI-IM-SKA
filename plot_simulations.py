import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import scipy.linalg as lng
sns.set_theme(style = 'white')
#sns.set_palette('husl',15)
from matplotlib import colors

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)
###########################################################################

beam = 'SKA_AA4'
fg_comp='synch_ff_ps'
path_data_sims_tot = f'Sims/beam_SKA_AA4_no_mean_synch_ff_ps_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax767_nside256'
with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

min_ch = 900.5
max_ch = 1004.5

num_ch =105

print(file.keys())

file['freq'] = np.linspace(min_ch, max_ch, num_ch)
print(file['freq'])

nu_ch= file['freq']

num_ch = len(nu_ch)

min_ch=min(nu_ch)
max_ch=max(nu_ch)

nu0 =1420
print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')
print(f'corresponding to the redshift range z: [{min(nu0/nu_ch -1.0):.2f} - {max(nu0/nu_ch -1.0):.2f}] ')

HI_maps_noise_freq = file['maps_sims_HI'] + file['maps_sims_noise']  #aggiungo il noise
fg_maps_freq = file['maps_sims_fg']
full_maps_freq = file['maps_sims_tot'] + file['maps_sims_noise']  #aggiungo il noise


nu_ch = np.linspace(min_ch, max_ch, num_ch)
del min_ch;del max_ch

nside = hp.get_nside(HI_maps_noise_freq[0])
lmax = 3*nside-1


ich=int(num_ch/2)
############################################################################

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'BEAM SKA AA4 channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(full_maps_freq[ich], cmap='viridis',title=f'Observations', hold=True)
fig.add_subplot(222) 
hp.mollview(HI_maps_noise_freq[ich], cmap='viridis',title=f'HI signal + noise',min=0, max=1,hold=True)
fig.add_subplot(223)
hp.mollview(fg_maps_freq[ich],title=f'Foregrounds',cmap='viridis',hold=True)
#fig.add_subplot(224)
#hp.mollview(file['maps_sims_noise'][ich],title=f'Noise, freq={nu_ch[ich]}',cmap='viridis',hold=True)
#plt.savefig(f'Plots_sims/sims_ch{nu_ch[ich]}_{fg_comp}_noise_beam_theta40arcmin_40freq_905.0_1295.0MHz_lmax{lmax}_nside{nside}.png')
########################################################################################################

z0= nu0/nu_ch[0] -1.0
z1= nu0/nu_ch[ich] -1.0
z2= nu0/nu_ch[-1] -1.0

rot = [-56,87]
xsize=150
reso = hp.nside2resol(nside, arcmin=True)
#fig=plt.figure(figsize=(10, 7))
#plt.suptitle('Brightness temperature')
#fig.add_subplot(311) 
#hp.gnomview(HI_maps_freq[0],rot=rot, coord='G', reso=reso,xsize=xsize, min=0, max=1, title=f'z={z0:0.2f}', unit='mK',cmap='viridis', cbar=False,notext=True,hold=True)
#fig.add_subplot(312) 
#hp.gnomview(HI_maps_freq[ich],rot=rot, coord='G', reso=reso,xsize=xsize,min=0, max=1, title=f'z={z1:0.2f}',unit='mK', cmap= 'viridis',cbar=False,notext=True, hold=True)
#fig.add_subplot(313) 
#hp.gnomview(HI_maps_freq[-1], rot=rot, coord='G', reso=reso,xsize=xsize,min=0, max=1, title=f'z={z2:0.2f}',unit='mK', cmap= 'viridis', cbar=False,notext=True,hold=True)
#plt.tight_layout()


map0  = hp.gnomview(HI_maps_noise_freq[0],rot=rot, coord='G', reso=reso,xsize=xsize, min=0, max=1,return_projected_map=True, no_plot=True)
map1  = hp.gnomview(HI_maps_noise_freq[ich],rot=rot, coord='G', reso=reso,xsize=xsize, min=0, max=1,return_projected_map=True, no_plot=True)
map2  = hp.gnomview(HI_maps_noise_freq[-1],rot=rot, coord='G', reso=reso,xsize=xsize, min=0, max=1,return_projected_map=True, no_plot=True)

fig, axs = plt.subplots(1,3, figsize=(12,5))
#fig.tight_layout()
cmap= 'viridis'
im0=axs[0].imshow(map0,cmap=cmap)
axs[0].set_title(f'z={z0:0.2f}')
axs[0].set_xlabel(r'$\theta$[deg]')
axs[0].set_ylabel(r'$\theta$[deg]')
im1=axs[1].imshow(map1,cmap=cmap)
axs[1].set_title(f'z={z1:0.2f}')
axs[1].set_xlabel(r'$\theta$[deg]')
#axs[1].set_ylabel(r'$\theta$[deg]')
im2=axs[2].imshow(map2,cmap=cmap)
axs[2].set_title(f'z={z2:0.2f}')
axs[2].set_xlabel(r'$\theta$[deg]')
#axs[2].set_ylabel(r'$\theta$[deg]')
norm = colors.Normalize(vmin=0, vmax=1)


plt.subplots_adjust(wspace=0.3, hspace=0.4, bottom=0.3, left=0.05)
sub_ax = plt.axes([0.93, 0.367, 0.02, 0.46]) 
fig.colorbar(im2, ax=axs, cax=sub_ax,location='right',orientation='vertical',label='T [mK]')
#plt.savefig('Plots_sims/gnomview_HI_simulations_z.png')
plt.show()