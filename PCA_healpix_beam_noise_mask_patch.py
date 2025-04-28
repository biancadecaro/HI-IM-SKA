import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import math
from scipy import ndimage
from matplotlib import colors
import difflib
import pymaster as nm
sns.set_theme(style = 'white')
#sns.set_palette('husl',15)

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)
###########################################################################
beam_s = 'theta40arcmin'
out_dir= f'PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise_mask_patch_stripe82/'
out_dir_plot = f'PCA_pixels_output/Plots_PCA/No_mean/Beam_{beam_s}_noise_mask_patch_stripe82/'

if not os.path.exists(out_dir):
        os.makedirs(out_dir)
if not os.path.exists(out_dir_plot):
        os.makedirs(out_dir_plot)

###################################################################################

fg_components='synch_ff_ps'
path_data_sims_tot = f'Sims/beam_{beam_s}_no_mean_sims_{fg_components}_noise_40freq_905.0_1295.0MHz_thick10MHz_lmax383_nside128'

with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

nu_ch= file['freq']

num_freq = len(nu_ch)

nu0 =1420
print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')
print(f'corresponding to the redshift range z: [{min(nu0/nu_ch -1.0):.2f} - {max(nu0/nu_ch -1.0):.2f}] ')

HI_maps_freq = file['maps_sims_HI'] + file['maps_sims_noise']  #aggiungo il noise
fg_maps_freq = file['maps_sims_fg']
full_maps_freq = file['maps_sims_tot'] + file['maps_sims_noise']  #aggiungo il noise


npix = np.shape(HI_maps_freq)[1]
nside = hp.get_nside(HI_maps_freq[0])
lmax=3*nside-1
num_sources = 3
print(f'nside:{nside}, lmax:{lmax}, num_ch:{num_freq}, min_ch:{min(nu_ch)}, max_ch:{max(nu_ch)}, Nfg:{num_sources}')


######################################################################

'''stripe 82: as WiggleZ and other survey (although not a stripe at the moment)
all of them are 54.1 x 54.1 = 2927 deg^2 square patches (because of the size of the HI/galaxies sim)'''

#############################################################################
## functions defined in extractPiece.py

## size in degree
def get_region(size=54.1,name=None):

    ref_region = ["Barnard", "SCP", "GPlane", "Stripe82"]

    lc=np.array([208., 0.,0., 120.])
    bc=np.array([-18., -62., 0., -50.])
    dl = np.ones((4)) * size
    db = np.ones((4)) * size 

    lst_match=[i for i,k in enumerate(ref_region) if name in k]
    if len(lst_match) ==0:
        clm=difflib.get_close_matches(name,ref_region)
        lst_match=[i for i,k in enumerate(ref_region) if k in clm]

    return [{'Name':ref_region[k],'lc':lc[k],'bc':bc[k],'dl':dl[k],
                                                'db':db[k]} for k in lst_match]

def extract_region(hmap, region=None):

    dict_match = get_region(name=region)
    nside = hp.get_nside(hmap)
            
    rotation=(dict_match[0]["lc"], dict_match[0]["bc"], 0.)
    reso_arcmin=hp.nside2resol(nside, arcmin=True)
    nxpix = int(np.ceil(dict_match[0]["dl"]*60./reso_arcmin))
    nypix = int(np.ceil(dict_match[0]["db"]*60./reso_arcmin))
    print('working on ',region)
    print(f'{nxpix} x {nypix} pixels')
    print("STAT=", reso_arcmin, nxpix, nypix, rotation)
    patch = hp.visufunc.gnomview(map=hmap, coord='G', rot=rotation, xsize=nxpix, \
                ysize=nypix, reso=reso_arcmin, flip='astro', return_projected_map=True)
    plt.close()

    return patch

def reduce_pixels(hmap, region=None, final_pixel = [256,256]):
    
    patch = extract_region(hmap, region)
    
    zoom_param  = final_pixel/np.array(patch.shape)
    
    # spline of order 3 for interpolation:
    final_patch = ndimage.zoom(patch, zoom_param, order=3)
    
    print(f'downsized to {final_patch.shape[0]} x {final_patch.shape[1]} pixels\n')
    
    return final_patch

def get_vertices(region=None, size=54.1):

    delta = size/2
    dict_match = get_region(name=region)
    
    centre_b,centre_l= dict_match[0]["bc"],dict_match[0]["lc"]
    
    v1 = [centre_b-delta,centre_l+delta]
    v2 = [centre_b-delta,centre_l-delta]
    v3 = [centre_b+delta,centre_l-delta]
    v4 = [centre_b+delta,centre_l+delta]
    
    coords = []
    for v in [v1,v2,v3,v4]:
        coords.append(hp.pixelfunc.ang2vec(math.radians(90.-v[0]),math.radians(v[1])))
        
    coords = np.array(coords)
    
    return coords

#######################################################################################
HI_maps_freq_patch_sq=np.zeros((num_freq, 256,256))
fg_maps_freq_patch_sq=np.zeros((num_freq, 256,256))
full_maps_freq_patch_sq=np.zeros((num_freq, 256,256))

for cc in range(num_freq):
     print(HI_maps_freq[cc].shape, 128*128)
     HI_maps_freq_patch_sq[cc] = reduce_pixels(HI_maps_freq[cc], region='Stripe82')
     HI_maps_freq_patch_sq[cc] = HI_maps_freq_patch_sq[cc][::-1]
     fg_maps_freq_patch_sq[cc] = reduce_pixels(fg_maps_freq[cc], region='Stripe82')
     fg_maps_freq_patch_sq[cc] = fg_maps_freq_patch_sq[cc][::-1]
     full_maps_freq_patch_sq[cc] = reduce_pixels(full_maps_freq[cc], region='Stripe82')
     full_maps_freq_patch_sq[cc] = full_maps_freq_patch_sq[cc][::-1]
     
#########################################################################################


ich = int(num_freq/2)
#coord = get_vertices("Stripe82")
#fig=plt.figure()
#plt.suptitle(f'Input foreground, channel:{nu_ch[ich]} MHz, pos : [{coord}]')
##hp.mollview(fg_maps_freq[ich], cmap='viridis', title=f'Input foreground, channel:{nu_ch[ich]} MHz', hold=True)
#plt.imshow(full_maps_freq_patch_sq[ich], cmap = 'viridis')#, vmin=15, vmax=60)
#plt.show()


######################################################################################################

print(full_maps_freq.shape)
assert len(np.shape(full_maps_freq_patch_sq))==3
full_maps_freq_patch=full_maps_freq_patch_sq.reshape(np.shape(full_maps_freq_patch_sq)[0],-1)
fg_maps_freq_patch=fg_maps_freq_patch_sq.reshape(np.shape(fg_maps_freq_patch_sq)[0],-1)
HI_maps_freq_patch=HI_maps_freq_patch_sq.reshape(np.shape(HI_maps_freq_patch_sq)[0],-1)
Cov_channels=np.cov(full_maps_freq_patch)

#fig=plt.figure()
#plt.imshow(Cov_channels, cmap='crest')
#plt.xlabel('[MHz]')
#plt.ylabel('[MHz]')
#plt.colorbar()
#plt.show()

eigenval, eigenvec= np.linalg.eig(Cov_channels)


fig= plt.figure(figsize=(7,4))
plt.semilogy(np.arange(1,num_freq+1),eigenval,'--.',mfc='none',markersize=10)
x_ticks = np.arange(-10, num_freq+1, 10 )
ax = plt.gca()
ax.set(xlim=[-10,num_freq+2],xticks=x_ticks,xlabel="eigenvalue number",ylabel="$\\lambda$",title='Eigenvalues')
#plt.tight_layout()
#plt.savefig(out_dir_plot+f'eigenvalues_Nfg_{fg_components}.png')
#plt.show()

#############################################################################
############################# PCA ##########################################

eigenvec_fg_Nfg = eigenvec[:, 0:num_sources]

#fig=plt.figure()
#plt.imshow(eigenvec_fg_Nfg, cmap='crest')
#plt.xlabel('[MHz]')
#plt.ylabel('[MHz]')
#plt.colorbar()
#plt.show()

del eigenvec

#for r in range(0,num_sources):
#    eigenvec_fg_Nfg[:,r] = eigenvec_fg_Nfg[:,r]/lng.norm(eigenvec_fg_Nfg[:,r])
######################################################################################

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
plt.plot(abs(eigenvec_fg_Nfg/np.linalg.norm(eigenvec_fg_Nfg,axis=0)),label='mix mat column')
plt.plot(FF_col/np.linalg.norm(FF_col),'m:',label='gal ff')

ax = plt.gca()
ax.set(ylim=[0.0,0.4],xlabel="frequency channel",ylabel="Spectral emission",title='PCA-mixing matrix columns')
plt.legend(fontsize=12)
################################################################################################

#Foreground's maps from PCA

res_fg_maps_patch=eigenvec_fg_Nfg@eigenvec_fg_Nfg.T@full_maps_freq_patch

#The foreground residual that leaks into the recovered signal and noise
fg_leakage = fg_maps_freq_patch - eigenvec_fg_Nfg@eigenvec_fg_Nfg.T@fg_maps_freq_patch
HI_leakage = eigenvec_fg_Nfg@eigenvec_fg_Nfg.T@HI_maps_freq_patch

del eigenvec_fg_Nfg

#res_HI=np.zeros((num_freq,npix))
res_HI_patch = full_maps_freq_patch - res_fg_maps_patch
res_HI_patch_sq = res_HI_patch.reshape(np.shape(fg_maps_freq_patch_sq)[0],256,256)

######################################################################################################

np.save(out_dir+f'cosmo_HI_noise_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax}_nside{nside}.npy',HI_maps_freq_patch_sq)
np.save(out_dir+f'res_PCA_HI_noise{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax}_nside{nside}.npy',res_HI_patch_sq)
#np.save(out_dir+f'fg_leak_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax}_nside{nside}.npy',fg_leakage)
#np.save(out_dir+f'HI_leak_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax}_nside{nside}.npy',HI_leakage)
np.save(out_dir+f'fg_input_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax}_nside{nside}.npy',fg_maps_freq_patch_sq)

##########################################################################################################



del file
fig, axs = plt.subplots(1,3, figsize=(12,6))
cmap= 'viridis'
fig.suptitle(f'BEAM {beam_s}, channel: {nu_ch[ich]} MHz, lmax:{lmax}, Nfg:{num_sources}',fontsize=19)
im0=axs[0].imshow(HI_maps_freq_patch_sq[ich],cmap=cmap)
axs[0].set_title(f'Input HI + noise', fontsize=15)
#axs[0].set_xlabel(r'$\theta$[deg]', fontsize=15)
#axs[0].set_ylabel(r'$\theta$[deg]', fontsize=15)
im1=axs[1].imshow(full_maps_freq_patch_sq[ich],cmap=cmap)
axs[1].set_title(f'Input HI + noise + foregrounds', fontsize=15)
#axs[1].set_xlabel(r'$\theta$[deg]', fontsize=15)
#axs[1].set_ylabel(r'$\theta$[deg]')
im2=axs[2].imshow(res_HI_patch_sq[ich],cmap=cmap)
axs[2].set_title(f'Cleaned HI + noise', fontsize=15)
#axs[2].set_xlabel(r'$\theta$[deg]', fontsize=15)
#axs[2].set_ylabel(r'$\theta$[deg]')
norm = colors.Normalize(vmin=0, vmax=1)
plt.subplots_adjust(wspace=0.2, hspace=0.4, bottom=0.3, left=0.05, top=0.85)
sub_ax = plt.axes([0.91, 0.367, 0.02, 0.46]) 
fig.colorbar(im2, ax=axs, cax=sub_ax,location='right',orientation='vertical',label='T [mK]')

plt.show()

###############################################################################################################
# Import the NaMaster python wrapper
def compute_Cl(mask,patch,Nx=256,Ny=256,Lxdeg=54.1,Lydeg=54.1, nbin=8, demask=False):

    #both patch and mask (Nx,Ny) matrix
    #Define field
    #  - Nx and Ny: the number of pixels in the x and y dimensions
    #  - Lx and Ly: the size of the patch in the x and y dimensions (in radians)
    Lx =  Lxdeg* np.pi/180.
    Ly =  Lydeg* np.pi/180.
    #create Namaster Field
    f0 = nm.NmtFieldFlat(Lx, Ly, mask, [patch])
    #Define bandpowers
    l0_bins = np.arange(Nx/nbin) * nbin * np.pi/Lx
    lf_bins = (np.arange(Nx/nbin)+1) * nbin * np.pi/Lx
    #this one for Namaster
    b = nm.NmtBinFlat(l0_bins, lf_bins)
    ell=(lf_bins-l0_bins)/2.+l0_bins
    #compute coupling
    w00 = nm.NmtWorkspaceFlat()
    w00.compute_coupling_matrix(f0, f0, b)
    cl = nm.compute_coupled_cell_flat(f0, f0, b)
    if demask==True: cl = w00.decouple_cell(cl)
    return ell, cl[0,:]
############################################################################################################
out_dir_cl = out_dir+'power_spectra_cls_from_healpix_maps/'
if not os.path.exists(out_dir_cl):
        os.makedirs(out_dir_cl)
#mask_stripe = hp.read_map(f'mask_patch_stripe82_binary_lmax{lmax}_nside{nside}.fits')
#hp.mollview(mask_stripe, cmap='viridis')
#plt.show()

lmax_cl= 2*nside
ell_cl = np.arange(lmax_cl+1)
factor_cl = ell_cl*(ell_cl+1)/(2*np.pi)

# Masks:
# Let's now create a mask:
Nx, Ny = res_HI_patch_sq[0].shape
Lxdeg = 54.1
Lydeg = 54.1
Lx =  Lxdeg* np.pi/180.
Ly =  Lydeg* np.pi/180.
mask = np.ones_like(res_HI_patch_sq[0]).flatten()
xarr = np.ones(Ny)[:, None] * np.arange(Nx)[None, :] * Lx/Nx
yarr = np.ones(Nx)[None, :] * np.arange(Ny)[:, None] * Ly/Ny

# Let's also trim the edges
#mask[np.where(xarr.flatten() < Lx / 16.)] = 0
#mask[np.where(xarr.flatten() > 15 * Lx / 16.)] = 0
#mask[np.where(yarr.flatten() < Ly / 16.)] = 0
#mask[np.where(yarr.flatten() > 15 * Ly / 16.)] = 0
mask = mask.reshape([Ny, Nx])
# You can also apodize it in the same way you do for full-sky masks:
#mask = nm.mask_apodization_flat(mask, Lx, Ly, aposize=2., apotype="C1")
plt.figure()
plt.imshow(mask, interpolation='nearest', origin='lower', cmap='viridis')
plt.colorbar()


nbin=8
f0 = nm.NmtFieldFlat(Lx, Ly, mask, [res_HI_patch_sq[0]])
#Define bandpowers
l0_bins = np.arange(Nx/nbin) * nbin * np.pi/Lx
lf_bins = (np.arange(Nx/nbin)+1) * nbin * np.pi/Lx
#this one for Namaster
b = nm.NmtBinFlat(l0_bins, lf_bins)
num_ell = b.get_effective_ells().shape[0]
ell=(lf_bins-l0_bins)/2.+l0_bins


cl_Hi=np.zeros((num_freq, num_ell))
#cl_fg=np.zeros((num_freq, num_ell))
cl_Hi_recons_Nfg=np.zeros((num_freq, num_ell))
#cl_fg_leak_Nfg=np.zeros((num_freq, num_ell))
#cl_HI_leak_Nfg=np.zeros((num_freq, num_ell))

for i in range(num_freq):
    _,cl_Hi[i] = compute_Cl(mask,HI_maps_freq_patch_sq[i])#hp.anafast(HI_maps_freq[i], lmax=lmax_cl)
    #cl_fg[i] = compute_Cl(mask,fg_maps_freq[i])
    _,cl_Hi_recons_Nfg[i] = compute_Cl(mask,res_HI_patch_sq[i])
    #cl_fg_leak_Nfg[i]=compute_Cl(mask,fg_leakage[i])
    #cl_HI_leak_Nfg[i]=compute_Cl(mask,HI_leakage[i])

print(ell.shape, cl_Hi.shape)
np.savetxt(out_dir_cl+f'ell_cl_input_HI_noise_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax_cl}_nside{nside}.dat', ell)
np.savetxt(out_dir_cl+f'cl_input_HI_noise_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax_cl}_nside{nside}.dat', cl_Hi)
#np.savetxt(out_dir_cl+f'cl_input_fg_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_lmax{lmax_cl}_nside{nside}.dat', cl_fg)
np.savetxt(out_dir_cl+f'cl_PCA_HI_noise_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat',cl_Hi_recons_Nfg)
#np.savetxt(out_dir_cl+f'cl_leak_HI_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_HI_leak_Nfg)
#np.savetxt(out_dir_cl+f'cl_leak_fg_{fg_components}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_lmax{lmax_cl}_nside{nside}.dat', cl_fg_leak_Nfg)

del res_HI_patch; del HI_maps_freq; del fg_leakage; del HI_leakage;del fg_maps_freq

factor = ell*(ell+1)/(2*np.pi)

cl_Hi_recons_Nfg_interp = np.zeros((num_freq, lmax_cl+1))
cl_Hi_interp = np.zeros((num_freq, lmax_cl+1))
for c in range(num_freq):
    cl_Hi_recons_Nfg_interp[c] = np.interp(ell_cl, ell, cl_Hi_recons_Nfg[c])
    cl_Hi_interp[c] = np.interp(ell_cl, ell, cl_Hi[c])


dict_match = get_region(name="Stripe82")
    
theta,phi= dict_match[0]["bc"],dict_match[0]["lc"]


fig=plt.figure()
plt.suptitle(f'Channel {nu_ch[ich]} MHz, patch:{theta, phi} rad')
plt.plot(ell, factor*cl_Hi[ich],mfc='none', label='Cosmo HI+noise')
plt.plot(ell, factor*cl_Hi_recons_Nfg[ich],'+',mfc='none', label='PCA HI+noise')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell(\ell+1)}{2\pi} C_{\ell} $')
#plt.xlim([15,200])
plt.legend()

fig=plt.figure()
plt.suptitle(f'Mean over channels, patch:{theta, phi} rad')
plt.plot(ell, factor*np.mean(cl_Hi, axis=0),mfc='none', label='Cosmo HI+noise')
plt.plot(ell, factor*np.mean(cl_Hi_recons_Nfg, axis=0),'+',mfc='none', label='PCA HI+noise')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle $')
plt.xlim([0,200])
plt.legend()


fig=plt.figure()
plt.suptitle(f'Channel {nu_ch[ich]} MHz, patch:{theta, phi} rad, INTERPOLATION')
plt.plot(ell_cl[2:], factor_cl[2:]*cl_Hi_interp[ich][2:],mfc='none', label='Cosmo HI+noise')
plt.plot(ell_cl[2:], factor_cl[2:]*cl_Hi_recons_Nfg_interp[ich][2:],'+',mfc='none', label='PCA HI+noise')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell(\ell+1)}{2\pi} C_{\ell} $')
plt.xlim([15,200])
plt.legend()

fig=plt.figure()
plt.suptitle(f'Mean over channels, patch:{theta, phi} rad, INTERPOLATION')
plt.plot(ell_cl[2:], factor_cl[2:]*np.mean(cl_Hi_interp, axis=0)[2:],mfc='none', label='Cosmo HI+noise')
plt.plot(ell_cl[2:], factor_cl[2:]*np.mean(cl_Hi_recons_Nfg_interp, axis=0)[2:],'+',mfc='none', label='PCA HI+noise')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle $')
plt.xlim([0,200])
plt.legend()
plt.show()


