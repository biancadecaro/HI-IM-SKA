import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy import ndimage
import seaborn as sns
sns.set_theme()
sns.set_theme(style = 'white')
#sns.set_palette('husl',15)
from matplotlib import colors
import matplotlib as mpl
import difflib
import pymaster as nm
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

#print(sns.color_palette("husl", 15).as_hex())
sns.palettes.color_palette()
c_pal = sns.color_palette().as_hex()
import cython_mylibc as pippo
##########################################################################################
# functions defined in extractPiece.py

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
##########################################################################################

beam_s = 'theta40arcmin'

out_dir_plot = 'Plots_PCA_needlets/'
dir_PCA = f'PCA_maps/No_mean/Beam_{beam_s}_noise_mask_patch_stripe82/'
dir_PCA_full = f'PCA_maps/No_mean/Beam_{beam_s}_noise/'
out_dir_maps_recon = f'maps_reconstructed/No_mean/Beam_{beam_s}_noise_mask_patch_stripe82/'
if not os.path.exists(out_dir_maps_recon):
		os.makedirs(out_dir_maps_recon)


fg_comp = 'synch_ff_ps'
beam = 'theta 40 arcmin'


num_ch=40
min_ch = 905
max_ch = 1295
nside=128
npix= hp.nside2npix(nside)
jmax=4
lmax= 3*nside-1
Nfg=3
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)

path_PCA_HI=dir_PCA+f'res_PCA_HI_noise_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_nside{nside}'
path_PCA_fg=dir_PCA+f'res_PCA_fg_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_nside{nside}'
path_cosmo_HI = f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise_mask_stripe82/cosmo_HI_noise_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz_lmax{lmax}_nside{nside}'
path_cosmo_HI_full = f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise/cosmo_HI_noise_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz_lmax{lmax}_nside{nside}'

path_fg = f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise_mask_stripe82/fg_input_{fg_comp}_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz_lmax{lmax}_nside{nside}'
path_leak_Fg = dir_PCA+f'leak_PCA_fg_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_nside{nside}'
path_leak_HI = dir_PCA+f'leak_PCA_HI_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_nside{nside}'
path_cosmo_HI_bjk = f'../Maps_needlets/No_mean/Beam_{beam_s}_noise_mask_patch_stripe82/bjk_maps_HI_noise_{num_ch}freq_{min_ch:1.1f}_{max_ch:1.1f}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}'
path_input_fg_bjk = f'../Maps_needlets/No_mean/Beam_{beam_s}_noise_mask_patch_stripe82/bjk_maps_fg_{fg_comp}_{num_ch}freq_{min_ch:1.1f}_{max_ch:1.1f}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}'


mask_stripe = hp.read_map(f'../mask_patch_stripe82_binary_lmax{lmax}_nside{nside}.fits')


print(f'jmax:{jmax}, lmax:{lmax}, num_ch:{num_ch}, min_ch:{min_ch}, max_ch:{max_ch}, Nfg:{Nfg}')

nu_ch = np.linspace(min_ch, max_ch, num_ch)

ich=int(num_ch/2)

############################################################################################
####################### NEEDLETS2HARMONICS #################################################

b_values = pippo.mylibpy_needlets_std_init_b_values(B,jmax,lmax)
res_PCA_HI = np.load(path_PCA_HI+'.npy')
res_PCA_fg = np.load(path_PCA_fg+'.npy')

print(res_PCA_HI.shape)
map_PCA_HI_need2pix=np.zeros((len(nu_ch), npix))
map_PCA_fg_need2pix=np.zeros((len(nu_ch), npix))

for nu in range(len(nu_ch)):
    map_PCA_fg_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(res_PCA_fg[:,nu],B, lmax)
    map_PCA_HI_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(res_PCA_HI[:,nu],B, lmax)
np.save(out_dir_maps_recon+f'maps_reconstructed_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_PCA_HI_need2pix)
np.save(out_dir_maps_recon+f'maps_reconstructed_PCA_fg_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_PCA_fg_need2pix)

del res_PCA_HI; del res_PCA_fg

cosmo_HI_bjk = np.load(path_cosmo_HI_bjk+'.npy')#[:,:jmax,:]
fg_bjk = np.load(path_input_fg_bjk+'.npy')#[:,:jmax,:]

map_input_HI_need2pix=np.zeros((len(nu_ch), npix))
map_input_fg_need2pix=np.zeros((len(nu_ch), npix))
for nu in range(len(nu_ch)):
	for j in range(cosmo_HI_bjk.shape[1]):
		map_input_HI_need2pix[nu] += pippo.mylibpy_needlets_f2betajk_j_healpix_harmonic(cosmo_HI_bjk[nu,j],b_values,j)
		map_input_fg_need2pix[nu] += pippo.mylibpy_needlets_f2betajk_j_healpix_harmonic(fg_bjk[nu,j],b_values,j)
	#map_input_HI_need2pix[nu] = hp.remove_dipole(map_input_HI_need2pix[nu])
	#map_input_fg_need2pix[nu] = hp.remove_dipole(map_input_fg_need2pix[nu])
	#map_input_HI_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(cosmo_HI_bjk[nu,:],B, lmax)
	#map_input_fg_need2pix[nu] = pippo.mylibpy_needlets_betajk2f_healpix_harmonic(fg_bjk[nu,:],B, lmax)
np.save(out_dir_maps_recon+f'maps_reconstructed_cosmo_HI_noise_{num_ch}_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_input_HI_need2pix)
np.save(out_dir_maps_recon+f'maps_reconstructed_input_fg_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}',map_input_fg_need2pix)
del cosmo_HI_bjk; del fg_bjk

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'STD NEED, BEAM {beam}, channel: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=19)
fig.add_subplot(221) 
hp.mollview(100*(np.abs(map_PCA_fg_need2pix[ich]/map_input_fg_need2pix[ich]-1)), min=0, max=10,  title= '(Res fg/fg-1)%',cmap='viridis',unit='%', hold= True)
fig.add_subplot(222)
hp.mollview(map_PCA_HI_need2pix[ich],min=0, max=1, title= 'Res PCA HI + noise Needlets 2 Pix',cmap='viridis', hold= True)
#plt.tight_layout()
plt.show()

#map_PCA_HI_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_PCA_HI_{fg_comp}_{num_ch}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
#map_PCA_fg_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_PCA_fg_{fg_comp}_{num_ch}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
#map_input_fg_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_input_fg_{fg_comp}_{num_ch}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')
#map_input_HI_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_cosmo_HI_{num_ch}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy')


#prima di fare il confronto devo ridurre le zone nelle res ricostruite
fg = np.load(path_fg+'.npy')
cosmo_HI = np.load(path_cosmo_HI+'.npy')
print(cosmo_HI.shape)
#
dict_match = get_region(name='Stripe82')
size = 54.1
delta = size/2
rotation=(dict_match[0]["lc"], dict_match[0]["bc"], 0.) #deg
reso_arcmin=hp.nside2resol(nside, arcmin=True) #1/60 deg
nxpix = int(np.ceil(dict_match[0]["dl"]*60./reso_arcmin)) 
nypix = int(np.ceil(dict_match[0]["db"]*60./reso_arcmin)) 
#temp_PCA_HI_need2pix_sq = np.zeros((num_ch, nxpix,nypix))
#temp_PCA_fg_need2pix_sq = np.zeros((num_ch, nxpix,nypix))

#map_PCA_fg_need2pix_sq=np.zeros((num_ch, 256,256))
final_pixel = [256,256]
map_PCA_HI_need2pix_sq_cart=np.zeros((num_ch, 256,256))
map_PCA_HI_need2pix_sq_gnom=np.zeros((num_ch, 256,256))

print(nxpix, nypix)
low_lonra  = dict_match[0]["lc"] - size/2.   #deg
high_lonra = dict_match[0]["lc"] + size/2.   #deg
low_latra  = dict_match[0]["bc"] - size/2.   #deg
high_latra = dict_match[0]["bc"] + size/2.   #deg

for cc in range(num_ch):
     temp_cart = hp.visufunc.cartview(map=map_PCA_HI_need2pix[cc], coord='G', lonra=[low_lonra,high_lonra], latra=[low_latra,high_latra],  xsize=nxpix,\
                             ysize=nypix, return_projected_map=True, cmap='viridis', title='cartview')#reso=reso_arcmin,
     zoom_param = final_pixel/np.array(temp_cart.shape)
     map_PCA_HI_need2pix_sq_cart[cc] = ndimage.zoom(temp_cart, zoom_param, order=3)
     map_PCA_HI_need2pix_sq_cart[cc] = map_PCA_HI_need2pix_sq_cart[cc][::-1]

     temp_gnom = hp.visufunc.gnomview(map=map_PCA_HI_need2pix[cc], coord='G', reso=reso_arcmin,rot=rotation ,xsize=nxpix,ysize=nypix, \
                                return_projected_map=True,cmap='viridis',  title='Gnomview', no_plot=True)
     map_PCA_HI_need2pix_sq_gnom[cc] = ndimage.zoom(temp_gnom, zoom_param, order=3)
     map_PCA_HI_need2pix_sq_gnom[cc] = map_PCA_HI_need2pix_sq_gnom[cc][::-1]
     #temp_PCA_fg_need2pix_sq[cc] = hp.visufunc.cartview(map=map_PCA_fg_need2pix[cc], coord='G', lonra=[low_lonra,high_lonra], latra=[low_latra,high_latra],  xsize=nxpix,\
                             #ysize=nypix, return_projected_map=True, cmap='viridis', title='cartview')#reso=reso_arcmin,
     #map_PCA_fg_need2pix_sq[cc] = ndimage.zoom(temp_PCA_fg_need2pix_sq[cc], zoom_param, order=3)
     #map_PCA_fg_need2pix_sq[cc] = map_PCA_fg_need2pix_sq[cc][::-1]
     plt.close('all')

#del temp_PCA_HI_need2pix_sq; del temp_PCA_fg_need2pix_sq

#fig, axs = plt.subplots(1,4, figsize=(12,6))
#cmap= 'viridis'
#fig.suptitle(f'BEAM {beam_s}, channel: {nu_ch[ich]} MHz, lmax:{lmax}, Nfg:{Nfg}',fontsize=19)
#im0=axs[0].imshow(cosmo_HI[ich],cmap=cmap)
#axs[0].set_title(f'Input HI + noise', fontsize=15)
##axs[0].set_xlabel(r'$\theta$[deg]', fontsize=15)
##axs[0].set_ylabel(r'$\theta$[deg]', fontsize=15)
#im1=axs[1].imshow(cosmo_HI[ich]+fg[ich],cmap=cmap)
#axs[1].set_title(f'Input HI + noise + foregrounds', fontsize=15)
##axs[1].set_xlabel(r'$\theta$[deg]', fontsize=15)
##axs[1].set_ylabel(r'$\theta$[deg]')
#im2=axs[2].imshow(map_PCA_HI_need2pix_sq_cart[ich],cmap=cmap)
#axs[2].set_title(f'HI + noise cleaned Cartview', fontsize=15)
##axs[2].set_xlabel(r'$\theta$[deg]', fontsize=15)
##axs[2].set_ylabel(r'$\theta$[deg]')
#im3=axs[3].imshow(map_PCA_HI_need2pix_sq_gnom[ich],cmap=cmap)
#axs[3].set_title(f'HI + noise cleaned Gnomview', fontsize=15)
#norm = colors.Normalize(vmin=0, vmax=1)
#plt.subplots_adjust(wspace=0.2, hspace=0.4, bottom=0.3, left=0.05, top=0.85)
#sub_ax = plt.axes([0.91, 0.367, 0.02, 0.46]) 
#fig.colorbar(im2, ax=axs, cax=sub_ax,location='right',orientation='vertical',label='T [mK]')
#
#plt.show()
#plt.savefig(f'Plots_PCA_needlets/gnomview_HI_HIfg_PCAHI_std_need_beam40arcmin_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.png')


rot = [120,-50]
xsize=150
reso = hp.nside2resol(nside, arcmin=True)
map0  = hp.gnomview(cosmo_HI[ich],rot=rot, coord='G', reso=reso,xsize=xsize, min=0, max=1,return_projected_map=True, no_plot=True)
map1  = hp.gnomview(cosmo_HI[ich]+fg[ich],rot=rot, coord='G', reso=reso,xsize=xsize, min=-1e3, max=1e3,return_projected_map=True, no_plot=True)
map2  = hp.gnomview(map_PCA_HI_need2pix[ich],rot=rot, coord='G', reso=reso,xsize=xsize, min=0, max=1,return_projected_map=True, no_plot=True)

fig, axs = plt.subplots(1,3, figsize=(12,6))
cmap= 'viridis'
fig.suptitle(f'STD NEED, BEAM {beam}, channel: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}',fontsize=19)
im0=axs[0].imshow(map0,cmap=cmap)
axs[0].set_title(f'Input HI + noise', fontsize=15)
axs[0].set_xlabel(r'$\theta$[deg]', fontsize=15)
axs[0].set_ylabel(r'$\theta$[deg]', fontsize=15)
im1=axs[1].imshow(map1,cmap=cmap)
axs[1].set_title(f'Input HI + noise + foregrounds', fontsize=15)
axs[1].set_xlabel(r'$\theta$[deg]', fontsize=15)
#axs[1].set_ylabel(r'$\theta$[deg]')
im2=axs[2].imshow(map2,cmap=cmap)
axs[2].set_title(f'Cleaned HI + noise', fontsize=15)
axs[2].set_xlabel(r'$\theta$[deg]', fontsize=15)
#axs[2].set_ylabel(r'$\theta$[deg]')
norm = colors.Normalize(vmin=0, vmax=1)
plt.subplots_adjust(wspace=0.2, hspace=0.4, bottom=0.3, left=0.05, top=0.85)
sub_ax = plt.axes([0.91, 0.367, 0.02, 0.46]) 
fig.colorbar(im2, ax=axs, cax=sub_ax,location='right',orientation='vertical',label='T [mK]')

plt.show()

del fg; del map_PCA_fg_need2pix;del map_input_fg_need2pix
################################################################################
############################# CL ###############################################
out_dir_cl = out_dir_maps_recon+'cls_recons_need/'
if not os.path.exists(out_dir_cl):
		os.makedirs(out_dir_cl)
lmax_cl = 2*nside

ell_cl_cosmo_HI = np.loadtxt(f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise_mask_patch_stripe82/power_spectra_cls_from_healpix_maps/ell_cl_input_HI_noise_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz_lmax{lmax_cl}_nside{nside}.dat')

cl_cosmo_HI_patch = np.loadtxt(f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise_mask_patch_stripe82/power_spectra_cls_from_healpix_maps/cl_input_HI_noise_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz_lmax{lmax_cl}_nside{nside}.dat')

cl_cosmo_HI_patch_mask = np.loadtxt(f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise_mask_stripe82/power_spectra_cls_from_healpix_maps/cl_deconv_input_HI_noise_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz_lmax{lmax_cl}_nside{nside}.dat')
cl_cosmo_HI_full = np.loadtxt(f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise/power_spectra_cls_from_healpix_maps/cl_input_HI_noise_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz_lmax{lmax_cl}_nside{nside}.dat')


print(cl_cosmo_HI_patch_mask.shape)
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

lmax_cl= 2*nside
ell_cl = np.arange(lmax_cl+1)
factor_cl = ell_cl*(ell_cl+1)/(2*np.pi)

# Masks:
# Let's now create a mask:
Nx, Ny = map_PCA_HI_need2pix_sq_cart[0].shape
Lxdeg = size
Lydeg = size
Lx =  Lxdeg* np.pi/180.
Ly =  Lydeg* np.pi/180.
mask = np.ones_like(map_PCA_HI_need2pix_sq_cart[0]).flatten()
xarr = np.ones(Ny)[:, None] * np.arange(Nx)[None, :] * Lx/Nx
yarr = np.ones(Nx)[None, :] * np.arange(Ny)[:, None] * Ly/Ny

mask = mask.reshape([Ny, Nx])


ell_cart, _ = compute_Cl(mask,map_PCA_HI_need2pix_sq_cart[0])
ell_gnom, _ = compute_Cl(mask,map_PCA_HI_need2pix_sq_gnom[0])

cl_PCA_HI_need2harm_sq_cart=np.zeros((num_ch, len(ell_cart)))
cl_PCA_HI_need2harm_sq_gnom=np.zeros((num_ch, len(ell_gnom)))


for i in range(num_ch):
    _,cl_PCA_HI_need2harm_sq_cart[i] = compute_Cl(mask,map_PCA_HI_need2pix_sq_cart[i])
    _,cl_PCA_HI_need2harm_sq_gnom[i] = compute_Cl(mask,map_PCA_HI_need2pix_sq_gnom[i])




f_0_mask = nm.NmtField(mask_stripe,[map_PCA_HI_need2pix[0]] , masked_on_input=True)
b = nm.NmtBin.from_nside_linear(nside, 8)
ell_mask= b.get_effective_ells()

cl_PCA_HI_mask = np.zeros((num_ch, len(ell_mask)))


for n in range(num_ch):
    f_0_mask = nm.NmtField(mask_stripe,[map_PCA_HI_need2pix[n]], masked_on_input=True)
    cl_PCA_HI_mask[n] = nm.compute_full_master(f_0_mask, f_0_mask, b)[0]



cl_PCA_HI_need2harm_sq_cart_interp = np.zeros((num_ch, lmax_cl+1))
cl_PCA_HI_need2harm_sq_gnom_interp = np.zeros((num_ch, lmax_cl+1))
cl_cosmo_HI_patch_interp = np.zeros((num_ch, lmax_cl+1))
cl_PCA_HI_mask_interp = np.zeros((num_ch, lmax_cl+1))
cl_PCA_HI_mask_anafast = np.zeros((num_ch, lmax_cl+1))

for c in range(num_ch):
    cl_PCA_HI_need2harm_sq_cart_interp[c] = np.interp(ell_cl, ell_cart, cl_PCA_HI_need2harm_sq_cart[c])
    cl_PCA_HI_need2harm_sq_gnom_interp[c] = np.interp(ell_cl, ell_gnom, cl_PCA_HI_need2harm_sq_gnom[c])
    cl_cosmo_HI_patch_interp[c] = np.interp(ell_cl, ell_cl_cosmo_HI, cl_cosmo_HI_patch[c])
    cl_PCA_HI_mask_interp[c] = np.interp(ell_cl, ell_mask, cl_PCA_HI_mask[c])
    cl_PCA_HI_mask_anafast[c] = hp.anafast(map_PCA_HI_need2pix[c], lmax=lmax_cl)

cl_PCA_HI_full= np.loadtxt('/home/bianca/Documents/HI IM SKA/PCA_needlets_output/maps_reconstructed/No_mean/Beam_theta40arcmin_noise/cls_recons_need/cl_PCA_HI_noise_synch_ff_ps_40_905_1295MHz_Nfg3_jmax4_lmax256_nside128.dat')
cl_PCA_HI_patch_mask = np.loadtxt(f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise_mask_stripe82/power_spectra_cls_from_healpix_maps/cl_deconv_input_HI_noise_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz_lmax{lmax_cl}_nside{nside}.dat')




del map_PCA_HI_need2pix; del cosmo_HI; del map_input_HI_need2pix; 
##################################################################################################
factor_cart=ell_cart*(ell_cart+1)/(2*np.pi)
factor_gnom=ell_gnom*(ell_gnom+1)/(2*np.pi)
factor_mask=ell_mask*(ell_mask+1)/(2*np.pi)
factor_cosmo_HI=ell_cl_cosmo_HI*(ell_cl_cosmo_HI+1)/(2*np.pi)

ell=np.arange(lmax_cl+1)
factor=ell*(ell+1)/(2*np.pi)



fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: channel:{nu_ch[ich]} MHz, BEAM {beam}, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}')
plt.semilogy(ell_cart,factor_cart*cl_PCA_HI_need2harm_sq_cart[ich], label='PCA HI + noise Cartview')
plt.semilogy(ell_gnom,factor_gnom*cl_PCA_HI_need2harm_sq_gnom[ich], label='PCA HI + noise Gnomview')
plt.semilogy(ell_mask,factor_mask*cl_PCA_HI_mask[ich], label='PCA HI + noise Mask Decoupled')

#plt.semilogy(ell,factor_cl*cl_PCA_HI_full[ich], label='PCA HI + noise Full sky')
plt.semilogy(ell_cl_cosmo_HI,factor_cosmo_HI*cl_cosmo_HI_patch[ich], '--k',label='Cosmo + noise patch')

plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 10))


diff_cl_need2sphe_sq_cart = cl_PCA_HI_need2harm_sq_cart/cl_cosmo_HI_patch-1
diff_cl_need2sphe_sq_gnom = cl_PCA_HI_need2harm_sq_gnom/cl_cosmo_HI_patch-1
#diff_cl_need2sphe_sq_mask_dec = cl_PCA_HI_mask/cl_cosmo_HI_patch-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell_cart, diff_cl_need2sphe_sq_cart[ich]*100, label='Cartview')
plt.plot(ell_gnom, diff_cl_need2sphe_sq_gnom[ich]*100, label='Gnomview')
#plt.plot(ell_gnom, diff_cl_need2sphe_sq_mask_dec*100, label='Mask decoupled')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-50,50])
frame2.set_ylabel(r'%$ diff $')
frame2.set_xlabel(r'$\ell$')
frame1.set_xticks(np.arange(1,200+1, 10))
#plt.tight_layout()
plt.legend()

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: mean over channels, BEAM {beam}, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}')
plt.plot(ell_cart, factor_cart*cl_cosmo_HI_patch.mean(axis=0), 'k', label = f'Cosmo + noise Patch')
plt.plot(ell_gnom, factor_gnom*cl_PCA_HI_need2harm_sq_cart.mean(axis=0),mfc='none', label = f'PCA HI + noise Cartview')
plt.plot(ell_cl_cosmo_HI, factor_cosmo_HI*cl_PCA_HI_need2harm_sq_gnom.mean(axis=0),mfc='none', label = f'PCA HI + noise Gnomview')
plt.semilogy(ell_mask,factor_mask*cl_PCA_HI_mask.mean(axis=0), label='PCA HI + noise Mask Decoupled')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 10))

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell_cart, diff_cl_need2sphe_sq_cart.mean(axis=0)*100, label='Cartview')
plt.plot(ell_gnom, diff_cl_need2sphe_sq_gnom.mean(axis=0)*100, label='Gnomview')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-50,50])
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
frame1.set_xticks(np.arange(1,200+1, 10))
#plt.tight_layout()
plt.legend()

###################### interp ###########################################################

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: channel:{nu_ch[ich]} MHz, BEAM {beam}, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}, INTERP')
plt.semilogy(ell[2:],factor[2:]*cl_cosmo_HI_full[ich][2:], 'k', label='Cosmo + noise Full sky')
plt.semilogy(ell[2:],factor[2:]*cl_cosmo_HI_patch_mask[ich][2:], '--k', label='Cosmo + noise Patch deconv')
plt.semilogy(ell[2:],factor[2:]*cl_PCA_HI_full[ich][2:], c=c_pal[0], label='Need-PCA HI + noise Full sky')
plt.semilogy(ell[2:],factor[2:]*cl_PCA_HI_mask_anafast[ich][2:],  c=c_pal[1],label='Need-PCA HI + noise Mask Anafast')
#plt.semilogy(ell[2:],factor[2:]*cl_PCA_HI_need2harm_sq_cart_interp[ich][2:], label='PCA HI + noise Cartview')
#plt.semilogy(ell[2:],factor[2:]*cl_PCA_HI_need2harm_sq_gnom_interp[ich][2:], label='PCA HI + noise Gnomview')
plt.semilogy(ell[2:],factor[2:]*cl_PCA_HI_mask_interp[ich][2:], c=c_pal[0],ls='--',label='Need-PCA HI + noise Patch deconv')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi}C_{\ell}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 10))


diff_cl_need2sphe_sq_cart_interp = cl_PCA_HI_need2harm_sq_cart_interp/cl_PCA_HI_full-1
diff_cl_need2sphe_sq_gnom_interp = cl_PCA_HI_need2harm_sq_gnom_interp/cl_PCA_HI_full-1
diff_cl_need2sphe_mask_interp = cl_PCA_HI_mask_interp/cl_PCA_HI_full-1

frame2=fig.add_axes((.1,.1,.8,.2))
#plt.plot(ell[2:], diff_cl_need2sphe_sq_cart_interp[ich][2:]*100, label='Cartview')
#plt.plot(ell[2:], diff_cl_need2sphe_sq_gnom_interp[ich][2:]*100, label='Gnomview')
plt.plot(ell[2:], diff_cl_need2sphe_mask_interp[ich][2:]*100, label='Mask decoupled')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-50,50])
frame2.set_ylabel(r'%$ C_{\ell}^{\rm PCA,patch} / C_{\ell}^{\rm PCA,full} -1$')
frame2.set_xlabel(r'$\ell$')
frame1.set_xticks(np.arange(1,200+1, 10))
#plt.tight_layout()
plt.legend()
#plt.savefig(f'Plots_PCA_needlets/cls_need_ch{nu_ch[ich]}_{fg_comp}_noise_full_patch2d_beam40arcmin_jmax{jmax}_lmax{lmax_cl}_Nfg{Nfg}_nside{nside}.png')


fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: mean over channels, BEAM {beam}, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}, INTERP')
plt.plot(ell[2:], factor[2:]*cl_cosmo_HI_patch_mask.mean(axis=0)[2:], '--k',label = f'Cosmo + noise Patch deconv')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_full.mean(axis=0)[2:],'k', label = f'PCA HI + noise Full sky')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_mask_interp.mean(axis=0)[2:], label = f'PCA HI + noise mask deconv ')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_need2harm_sq_cart_interp.mean(axis=0)[2:],'+',mfc='none', label = f'PCA HI + noise Cartview')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_need2harm_sq_gnom_interp.mean(axis=0)[2:],'+',mfc='none', label = f'PCA HI + noise Gnomview')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 10))

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe_sq_cart_interp.mean(axis=0)[2:]*100, label='Cartview')
plt.plot(ell[2:], diff_cl_need2sphe_sq_gnom_interp.mean(axis=0)[2:]*100, label='Gnomview')
plt.plot(ell[2:], diff_cl_need2sphe_mask_interp.mean(axis=0)[2:]*100, label='Mask deconv')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-50,50])
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
frame1.set_xticks(np.arange(1,200+1, 10))
#plt.tight_layout()
plt.legend()


plt.show()
