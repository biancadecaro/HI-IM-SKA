import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pymaster as nm
sns.set_theme(style = 'white')
#sns.set_palette('husl',15)
c_pal = sns.color_palette().as_hex()

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)
###########################################################################
beam_s = 'theta40arcmin'

fg_components='synch_ff_ps'
path_data_sims_tot = f'Sims/beam_{beam_s}_no_mean_sims_{fg_components}_noise_40freq_905.0_1295.0MHz_thick10MHz_lmax767_nside256'

with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

nu_ch= file['freq']

num_freq = len(nu_ch)

nu0 =1420
print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')
print(f'corresponding to the redshift range z: [{min(nu0/nu_ch -1.0):.2f} - {max(nu0/nu_ch -1.0):.2f}] ')

HI_maps_freq = file['maps_sims_HI']# + file['maps_sims_noise']  #aggiungo il noise

npix = np.shape(HI_maps_freq)[1]
nside = hp.get_nside(HI_maps_freq[0])
lmax=3*nside-1

del file
########################################################################################
#######################################

mask1_70 = hp.read_map('HFI_Mask_GalPlane_2048_R1.10.fits', field=3)
mask_70t = hp.ud_grade(mask1_70, nside_out=256)
mask_70 = hp.ud_grade(mask_70t, nside_out=nside)
del mask1_70
mask_70s = hp.sphtfunc.smoothing(mask_70, 3*np.pi/180,lmax=lmax) #apodization 3 deg come in Olivari
#del mask_70
fsky  = np.sum(mask_70)/hp.nside2npix(nside)
fsky_apo  = np.sum(mask_70s)/hp.nside2npix(nside)


fig=plt.figure()
hp.mollview(mask_70, cmap='viridis', title=f'fsky={fsky:0.2f}', hold=True)
########################################################################################


HI_maps_freq_mask = np.zeros((num_freq, npix))
HI_maps_freq_mask_apo = np.zeros((num_freq, npix))

for n in range(num_freq):
        
        HI_maps_freq_mask[n] = HI_maps_freq[n]*mask_70 #- HI_maps_freq_mean
        #HI_maps_freq_mask[n] = hp.remove_dipole(HI_maps_freq_mask[n])
        HI_maps_freq_mask_apo[n] = HI_maps_freq[n]*mask_70s #- HI_maps_freq_mean
        #HI_maps_freq_mask_apo[n] = hp.remove_dipole(HI_maps_freq_mask_apo[n])
ich = 21#int(num_freq/2)

fig=plt.figure()
hp.mollview(HI_maps_freq_mask[ich],min=0, max=1, cmap='viridis', title=f'Input HI, channel:{nu_ch[ich]} MHz', hold=True)
plt.show()
        
######################################################################################

lmax_cl = 2*nside
ell = np.arange(0, lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)

cl_HI_cosmo_full=hp.anafast(HI_maps_freq[ich], lmax=lmax_cl) #np.zeros((num_freq, lmax_cl+1))
cl_HI_cosmo_mask=hp.anafast(HI_maps_freq_mask[ich], lmax=lmax_cl)#np.zeros((num_freq, lmax_cl+1))
cl_HI_cosmo_mask_apo=hp.anafast(HI_maps_freq_mask_apo[ich], lmax=lmax_cl)#np.zeros((num_freq, lmax_cl+1))

#for i in range(num_freq):
#    cl_HI_cosmo_full[i] = hp.anafast(HI_maps_freq[i], lmax=lmax_cl)    
#    cl_HI_cosmo_mask[i] = hp.anafast(HI_maps_freq_mask[i], lmax=lmax_cl)


#### deconvoluzione
f_no_mask = nm.NmtField(np.ones(hp.nside2npix(nside)),[HI_maps_freq[ich]] )
b = nm.NmtBin.from_nside_linear(nside, 4)
ell_mask= b.get_effective_ells()
factor_mask = ell_mask*(ell_mask+1)/(2*np.pi)

f_mask_apo = nm.NmtField(mask_70s,[HI_maps_freq[ich]] )
f_mask = nm.NmtField(mask_70,[HI_maps_freq[ich]] )

cl_cosmo_HI_no_mask_deconv = nm.compute_full_master(f_no_mask, f_no_mask, b)[0]
#cl_cosmo_HI_no_mask_deconv_interp = np.interp(ell, ell_mask, cl_cosmo_HI_no_mask_deconv[ich])


cl_cosmo_HI_mask_deconv = nm.compute_full_master(f_mask, f_mask, b)[0]
#cl_cosmo_HI_mask_deconv_interp = np.interp(ell, ell_mask, cl_cosmo_HI_mask_deconv[ich])

cl_cosmo_HI_mask_apo_deconv = nm.compute_full_master(f_mask_apo, f_mask_apo, b)[0]
#cl_cosmo_HI_mask_apo_deconv_interp = np.interp(ell, ell_mask, cl_cosmo_HI_mask_apo_deconv[ich])

#for n in range(num_freq):
#    f_0_mask = nm.NmtField(mask_70,[HI_maps_freq_mask[n]] )
#    cl_cosmo_HI_mask_deconv[n] = nm.compute_full_master(f_0_mask, f_0_mask, b)[0]
#    cl_cosmo_HI_mask_deconv_interp[n] = np.interp(ell, ell_mask, cl_cosmo_HI_mask_deconv_interp[n])

fig=plt.figure(figsize=(12,9)) 
ax = fig.add_subplot(111)
plt.suptitle(f'Channel {nu_ch[ich]} MHz')
ax.plot(ell, factor*cl_HI_cosmo_full, label='ana, no mask')
ax.plot(ell, factor*cl_HI_cosmo_mask, label='ana, w/ binary mask',ls='--')
#ax.plot(ell, cl_HI_cosmo_mask_apo, label='ana, w/ apo-mask')
ax.plot(ell_mask, factor_mask*cl_cosmo_HI_mask_deconv,'-s',mfc='none',markersize=10,label='nam, w/ binary mask')
#ax.plot(ell_mask, cl_cosmo_HI_mask_apo_deconv,'-s',mfc='none',markersize=10,label='nam, w/ apo-mask')
ax.plot(ell_mask, factor_mask*cl_cosmo_HI_no_mask_deconv,'-o', label='nam, no mask')

ax.set(xlabel=r"$\ell$",ylabel=r"$\frac{\ell(\ell+1)}{2\pi} C_\ell$",xscale='linear',yscale='linear',xlim=[2,lmax_cl])#,ylim=[1e-7,1e-2])
ax.legend(fontsize=14)



fig=plt.figure(figsize=(12,9)) 
ax = fig.add_subplot(111)
plt.suptitle(f'Channel {nu_ch[ich]} MHz, nside={nside}, lmax cl:{lmax_cl}, beam theta 40 arcim')
ax.plot(ell, factor*cl_HI_cosmo_full, label='ana, no mask')
ax.plot(ell, factor*cl_HI_cosmo_mask/fsky, label='ana, w/ mask /fsky',ls='--')
#ax.plot(ell, cl_HI_cosmo_mask_apo/fsky_apo, label='ana, w/ apo-mask /fsky_apo')
ax.plot(ell_mask, factor_mask*cl_cosmo_HI_mask_deconv,'-s',mfc='none',markersize=10,label='nam, w/ binary mask')
#ax.plot(ell_mask, cl_cosmo_HI_mask_apo_deconv,'-s',mfc='none',markersize=10,label='nam, w/ apo-mask')
ax.plot(ell_mask, factor_mask*cl_cosmo_HI_no_mask_deconv,'-o', label='nam, no mask')

ax.set(xlabel=r"$\ell$",ylabel=r"$\frac{\ell(\ell+1)}{2\pi} C_\ell$",xscale='linear',yscale='linear',xlim=[2,lmax_cl])#,ylim=[1e-7,1e-2])
ax.legend(fontsize=14)


plt.show()

del cl_HI_cosmo_full; del cl_HI_cosmo_mask; del cl_cosmo_HI_mask_deconv; del cl_cosmo_HI_no_mask_deconv; del f_mask
############################################################################
cl_HI_cosmo_full = np.zeros((num_freq, lmax_cl+1))
cl_cosmo_HI_mask_deconv = np.zeros((num_freq, len(ell_mask)))
cl_cosmo_HI_mask_deconv_interp = np.zeros((num_freq, lmax_cl+1))

for n in range(num_freq):
    cl_HI_cosmo_full[n] = hp.anafast(HI_maps_freq[n], lmax=lmax_cl)

    f_mask = nm.NmtField(mask_70,[HI_maps_freq[n]] )
    cl_cosmo_HI_mask_deconv[n] = nm.compute_full_master(f_mask, f_mask, b)[0]
    cl_cosmo_HI_mask_deconv_interp[n] = np.interp(ell, ell_mask, cl_cosmo_HI_mask_deconv[n])

bias = cl_cosmo_HI_mask_deconv_interp[:,2:]/cl_HI_cosmo_full[:,2:]


ell_ticks = np.arange(2,lmax_cl+1,20)
print(ell_ticks)

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111)
plt.suptitle('Bias Cl')
im=ax.imshow(bias, cmap='crest', aspect='auto', vmin=0, vmax=6)
#ax.axhline(y=np.arange(num_freq)[ich], c='grey', ls='--')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel('channels [MHz]')
ax.set_yticks(np.arange(num_freq), labels=nu_ch)
ax.set_xticks(ell_ticks, labels=ell_ticks, rotation=45)
ax.get_yticklabels()[ich].set_color("red")
cb=fig.colorbar(im)
cb.set_label('cl mask deconv / cl full sky',size=18)
plt.show()

del cl_cosmo_HI_mask_deconv_interp; del cl_HI_cosmo_full; del cl_cosmo_HI_mask_deconv
#######################################################################################
##################### INVERSE MASK ###############################################
########################################################################################

bad_v = np.where(mask_70==0)
mask_70_inv =np.zeros(mask_70.shape)
mask_70_inv[bad_v]=  1
fsky_inv  = np.sum(mask_70_inv)/hp.nside2npix(nside)


fig=plt.figure()
hp.mollview(mask_70_inv, cmap='viridis', title=f'fsky={fsky_inv:0.2f}', hold=True)

HI_maps_freq_mask = np.zeros((num_freq, npix))

for n in range(num_freq):
        
        HI_maps_freq_mask[n] = HI_maps_freq[n]*mask_70_inv #- HI_maps_freq_mean
        

fig=plt.figure()
hp.mollview(HI_maps_freq_mask[ich],min=0, max=1, cmap='viridis', title=f'Input HI, channel:{nu_ch[ich]} MHz', hold=True)
plt.show()
        
######################################################################################

lmax_cl = 2*nside
ell = np.arange(0, lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)

cl_HI_cosmo_full=hp.anafast(HI_maps_freq[ich], lmax=lmax_cl) #np.zeros((num_freq, lmax_cl+1))
cl_HI_cosmo_mask=hp.anafast(HI_maps_freq_mask[ich], lmax=lmax_cl)#np.zeros((num_freq, lmax_cl+1))
cl_HI_cosmo_mask_apo=hp.anafast(HI_maps_freq_mask_apo[ich], lmax=lmax_cl)#np.zeros((num_freq, lmax_cl+1))

#for i in range(num_freq):
#    cl_HI_cosmo_full[i] = hp.anafast(HI_maps_freq[i], lmax=lmax_cl)    
#    cl_HI_cosmo_mask[i] = hp.anafast(HI_maps_freq_mask[i], lmax=lmax_cl)


################ deconvoluzione
f_no_mask = nm.NmtField(np.ones(hp.nside2npix(nside)),[HI_maps_freq[ich]] )
b = nm.NmtBin.from_nside_linear(nside, 4)
ell_mask= b.get_effective_ells()
factor_mask = ell_mask*(ell_mask+1)/(2*np.pi)

f_mask = nm.NmtField(mask_70_inv,[HI_maps_freq[ich]] )

cl_cosmo_HI_no_mask_deconv = nm.compute_full_master(f_no_mask, f_no_mask, b)[0]


cl_cosmo_HI_mask_deconv = nm.compute_full_master(f_mask, f_mask, b)[0]


##################################################################

fig=plt.figure(figsize=(12,9)) 
ax = fig.add_subplot(111)
plt.suptitle(f'Channel {nu_ch[ich]} MHz, INVERSE MASK, fsky:{fsky_inv:0.2f}')
ax.plot(ell, factor*cl_HI_cosmo_full, label='ana, no mask')
ax.plot(ell, factor*cl_HI_cosmo_mask, label='ana, w/ binary mask',ls='--')
#ax.plot(ell, cl_HI_cosmo_mask_apo, label='ana, w/ apo-mask')
ax.plot(ell_mask, factor_mask*cl_cosmo_HI_mask_deconv,'-s',mfc='none',markersize=10,label='nam, w/ binary mask')
#ax.plot(ell_mask, cl_cosmo_HI_mask_apo_deconv,'-s',mfc='none',markersize=10,label='nam, w/ apo-mask')
ax.plot(ell_mask, factor_mask*cl_cosmo_HI_no_mask_deconv,'-o', label='nam, no mask')

ax.set(xlabel=r"$\ell$",ylabel=r"$\frac{\ell(\ell+1)}{2\pi} C_\ell$",xscale='linear',yscale='linear',xlim=[2,lmax_cl])#,ylim=[1e-7,1e-2])
ax.legend(fontsize=14)



fig=plt.figure(figsize=(12,9)) 
ax = fig.add_subplot(111)
plt.suptitle(f'Channel {nu_ch[ich]} MHz, INVERSE MASK, fsky:{fsky_inv:0.2f}\nnside={nside}, lmax cl:{lmax_cl}, beam theta 40 arcim')
ax.plot(ell, factor*cl_HI_cosmo_full, label='ana, no mask')
ax.plot(ell, factor*cl_HI_cosmo_mask/fsky_inv, label='ana, w/ mask /fsky',ls='--')
#ax.plot(ell, cl_HI_cosmo_mask_apo/fsky_apo, label='ana, w/ apo-mask /fsky_apo')
ax.plot(ell_mask, factor_mask*cl_cosmo_HI_mask_deconv,'-s',mfc='none',markersize=10,label='nam, w/ binary mask')
#ax.plot(ell_mask, cl_cosmo_HI_mask_apo_deconv,'-s',mfc='none',markersize=10,label='nam, w/ apo-mask')
ax.plot(ell_mask, factor_mask*cl_cosmo_HI_no_mask_deconv,'-o', label='nam, no mask')

ax.set(xlabel=r"$\ell$",ylabel=r"$\frac{\ell(\ell+1)}{2\pi} C_\ell$",xscale='linear',yscale='linear',xlim=[2,lmax_cl])#,ylim=[1e-7,1e-2])
ax.legend(fontsize=14)


plt.show()

del cl_HI_cosmo_full; del cl_HI_cosmo_mask; del cl_cosmo_HI_mask_deconv; del cl_cosmo_HI_no_mask_deconv; del f_mask
############################################################################
cl_HI_cosmo_full = np.zeros((num_freq, lmax_cl+1))
cl_cosmo_HI_mask_deconv = np.zeros((num_freq, len(ell_mask)))
cl_cosmo_HI_mask_deconv_interp = np.zeros((num_freq, lmax_cl+1))

for n in range(num_freq):
    cl_HI_cosmo_full[n] = hp.anafast(HI_maps_freq[n], lmax=lmax_cl)

    f_mask = nm.NmtField(mask_70_inv,[HI_maps_freq[n]] )
    cl_cosmo_HI_mask_deconv[n] = nm.compute_full_master(f_mask, f_mask, b)[0]
    cl_cosmo_HI_mask_deconv_interp[n] = np.interp(ell, ell_mask, cl_cosmo_HI_mask_deconv[n])

bias = cl_cosmo_HI_mask_deconv_interp[:,2:]/cl_HI_cosmo_full[:,2:]


ell_ticks = np.arange(2,lmax_cl+1,20)
print(ell_ticks)

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111)
plt.suptitle(f'Bias Cl, INVERSE MASK, fsky:{fsky_inv:0.2f}')
im=ax.imshow(bias, cmap='crest', aspect='auto', vmin=0, vmax=6)
#ax.axhline(y=np.arange(num_freq)[ich], c='grey', ls='--')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel('channels [MHz]')
ax.set_yticks(np.arange(num_freq), labels=nu_ch)
ax.set_xticks(ell_ticks, labels=ell_ticks, rotation=45)
ax.get_yticklabels()[ich].set_color("red")
cb=fig.colorbar(im)
cb.set_label('cl mask deconv / cl full sky',size=18)
plt.show()


del bias; del cl_cosmo_HI_mask_deconv_interp; del cl_HI_cosmo_full;
####################################################################
pix_mask = hp.query_strip(nside, theta1=np.pi*2/3, theta2=np.pi/3)
print(pix_mask)
mask_50 = np.zeros(npix)
mask_50[pix_mask] =1
fsky_50 = np.sum(mask_50)/hp.nside2npix(nside)
hp.mollview(mask_50, cmap='viridis', title=f'fsky:{fsky_50:0.2f}')


HI_mask_50 = HI_maps_freq[ich]*mask_50
cl_HI_cosmo_50 = hp.anafast(HI_mask_50, lmax=lmax_cl)

##invertire

bad_v = np.where(mask_50==0)
mask_50_inv =np.zeros(mask_50.shape)
mask_50_inv[bad_v]=  1
fsky_50_inv  = np.sum(mask_50_inv)/hp.nside2npix(nside)
hp.mollview(mask_50_inv, cmap='viridis', title=f'INV fsky:{fsky_50_inv:0.2f}')
plt.show()

HI_mask_50_inv = HI_maps_freq[ich]*mask_50_inv
cl_HI_cosmo_50_inv = hp.anafast(HI_mask_50_inv, lmax=lmax_cl)

fig=plt.figure() 
ax = fig.add_subplot(111)
plt.suptitle(f'Channel {nu_ch[ich]} MHz, fsky:{fsky_50:0.2f},fsky inv :{fsky_50_inv:0.2f} \nnside={nside}, lmax cl:{lmax_cl}, beam theta 40 arcmin')
ax.plot(ell, factor*cl_HI_cosmo_50, label='ana, w/ mask',c=c_pal[0] )
ax.plot(ell, factor*cl_HI_cosmo_50/fsky, label='ana, w/ mask /fsky',c=c_pal[0],ls='--')
ax.plot(ell, factor*cl_HI_cosmo_50_inv, label='ana, w/ mask inv',c=c_pal[1] )
ax.plot(ell, factor*cl_HI_cosmo_50_inv/fsky_inv, label='ana, w/ mask inv /fsky',c=c_pal[1],ls='--')

#ax.plot(ell_mask, factor_mask*cl_cosmo_HI_mask_deconv,'-s',mfc='none',markersize=10,label='nam, w/ binary mask')
#
#ax.plot(ell_mask, factor_mask*cl_cosmo_HI_no_mask_deconv,'-o', label='nam, no mask')

ax.set(xlabel=r"$\ell$",ylabel=r"$\frac{\ell(\ell+1)}{2\pi} C_\ell$",xscale='linear',yscale='linear',xlim=[2,lmax_cl])#,ylim=[1e-7,1e-2])
ax.legend(fontsize=14)
plt.show()