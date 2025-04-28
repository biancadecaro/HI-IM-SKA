import numpy as np
import pymaster as nmt
import matplotlib.pyplot as plt
import camb
import healpy as hp
import difflib
import math 
from scipy import ndimage
from matplotlib import colors
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

def extract_region(hmap, region=None, size=54.1):

    dict_match = get_region(size=size,name=region)
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
def extract_region_cartview(hmap, region=None, size=54.1):

    dict_match = get_region(size=size,name=region)
    nside = hp.get_nside(hmap)
    rotation=(dict_match[0]["lc"], dict_match[0]["bc"], 0.)
    reso_arcmin=hp.nside2resol(nside, arcmin=True)
    nxpix = int(np.ceil(dict_match[0]["dl"]*60./reso_arcmin))
    nypix = int(np.ceil(dict_match[0]["db"]*60./reso_arcmin))


    low_lonra  = dict_match[0]["lc"] - size/2.   #deg
    high_lonra = dict_match[0]["lc"] + size/2.  #deg
    low_latra  = dict_match[0]["bc"] - size/2.  #deg
    high_latra = dict_match[0]["bc"] + size/2.   #deg

    print('working on ',region)
    print(f'{nxpix} x {nypix} pixels')
    print("STAT=", reso_arcmin, nxpix, nypix, rotation)
    patch = hp.visufunc.cartview(map=hmap, coord='G', lonra=[low_lonra,high_lonra], latra=[low_latra,high_latra],xsize=nxpix, \
                ysize=nypix,flip='astro', return_projected_map=True)
    plt.close()

    return patch

def reduce_pixels(hmap, region=None, size=54.1,final_pixel = [256,256]):
    
    patch = extract_region(hmap, region, size)
    
    zoom_param  = final_pixel/np.array(patch.shape)
    
    # spline of order 3 for interpolation:
    final_patch = ndimage.zoom(patch, zoom_param, order=3)
    
    print(f'downsized to {final_patch.shape[0]} x {final_patch.shape[1]} pixels\n')
    
    return final_patch

def reduce_pixels_cartview(hmap, region=None, size=54.1,final_pixel = [256,256]):
    
    patch = extract_region_cartview(hmap, region, size)
    
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


def compute_Cl(mask,patch,Nx,Ny,Lxdeg,Lydeg, nbin=8, demask=False):

    #both patch and mask (Nx,Ny) matrix
    #Define field
    #  - Nx and Ny: the number of pixels in the x and y dimensions
    #  - Lx and Ly: the size of the patch in the x and y dimensions (in radians)
    Lx =  Lxdeg* np.pi/180.
    Ly =  Lydeg* np.pi/180.
    #create Namaster Field
    f0 = nmt.NmtFieldFlat(Lx, Ly, mask, [patch])
    #Define bandpowers
    l0_bins = np.arange(Nx/nbin) * nbin * np.pi/Lx
    lf_bins = (np.arange(Nx/nbin)+1) * nbin * np.pi/Lx
    #this one for Namaster
    b = nmt.NmtBinFlat(l0_bins, lf_bins)
    ell=(lf_bins-l0_bins)/2.+l0_bins
    #compute coupling
    w00 = nmt.NmtWorkspaceFlat()
    w00.compute_coupling_matrix(f0, f0, b)
    cl = nmt.compute_coupled_cell_flat(f0, f0, b)
    if demask==True: cl = w00.decouple_cell(cl)
    return ell, cl[0,:]
####################################################################################
# This script describes the functionality of the flat-sky version of pymaster

# Dimensions:
# First, a flat-sky field is defined by four quantities:
#  - Lx and Ly: the size of the patch in the x and y dimensions (in radians)
nside = 128
lmax=3*nside-1
reso_arcmin=hp.nside2resol(nside, arcmin=True)
Lxdeg = 54.1#54.1
Lydeg = 54.1#54.1

Lxdeg_s = 10.41
Lydeg_s = 10.41

Lx =  Lxdeg* np.pi/180.
Ly =  Lydeg* np.pi/180.

Lx_s =  Lxdeg* np.pi/180.
Ly_s =  Lydeg* np.pi/180.
#  - Nx and Ny: the number of pixels in the x and y dimensions
Nx = 256
Ny = 256

# Gaussian simulations:
# pymaster allows you to generate random realizations of both spherical and
# flat fields given a power spectrum. These are returned as 2D arrays with
# shape (Ny,Nx)

pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,  
                       As=2e-9, ns=0.965, halofit_version='mead', lmax=2000)
results = camb.get_results(pars)
powers =results.get_cmb_unlensed_scalar_array_dict(pars, raw_cl=True)
print(powers.keys())
cl_tt=powers['TxT']
cl_ee = powers['ExE']
cl_te = powers['TxE']
l= np.arange(cl_tt.shape[0])
#l, cl_tt, cl_ee, cl_bb, cl_te = np.loadtxt('cls.txt', unpack=True)
theta_arcmin = 40 #arcmin
theta_FWMH = theta_arcmin*np.pi/(60*180)
beam = np.exp(-(theta_FWMH * l)**2)
#cl_tt *= beam
#cl_ee *= beam
#cl_bb *= beam
#cl_te *= beam
#mpt= nmt.synfast_flat(Nx, Ny, Lx, Ly,
#                                 np.array([cl_tt]),# cl_bb]),
#                                 [0])[0]
seed=7344234
np.random.seed(seed)
map_cltt = hp.synfast(cls=cl_tt, nside=nside, lmax=lmax)


dict_match = get_region(name="Stripe82", size=Lx)
lon, lat=dict_match[0]["lc"],dict_match[0]["bc"]

mpt = reduce_pixels(map_cltt, region='Stripe82', size=Lxdeg)
mpt_cart = reduce_pixels_cartview(map_cltt, region='Stripe82', size=Lxdeg)
mpt_s = reduce_pixels(map_cltt, region='Stripe82', size=Lxdeg_s)


# You can have a look at the maps using matplotlib's imshow:
fig, axs = plt.subplots(1,3, figsize=(12,6))
cmap= 'viridis'
im0=axs[0].imshow(mpt,cmap=cmap)
axs[0].set_title(f'Gnomview size : {Lxdeg}x{Lxdeg}', fontsize=15)

im1=axs[1].imshow(mpt,cmap=cmap)
axs[1].set_title(f'Cartview size : {Lxdeg}x{Lxdeg}', fontsize=15)

im2=axs[2].imshow(mpt_s,cmap=cmap)
axs[2].set_title(f'Gnomview size : {Lxdeg_s}x{Lxdeg_s}', fontsize=15)

norm = colors.Normalize(vmin=0, vmax=1)
plt.subplots_adjust(wspace=0.2, hspace=0.4, bottom=0.3, left=0.05, top=0.85)
sub_ax = plt.axes([0.91, 0.367, 0.02, 0.46]) 
fig.colorbar(im1, ax=axs, cax=sub_ax,location='right',orientation='vertical',label='T [mK]')

# Masks:
# Let's now create a mask:
mask = np.ones_like(mpt).flatten()
mask_s = np.ones_like(mpt_s).flatten()
mask = mask.reshape([Ny, Nx])
mask_s = mask.reshape([Ny, Nx])
# You can also apodize it in the same way you do for full-sky masks:
#mask = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=2., apotype="C1")
#fig, axs = plt.subplots(1,2, figsize=(12,6))
#cmap= 'viridis'
#im0=axs[0].imshow(mask,cmap=cmap)
#axs[0].set_title(f'size : {Lxdeg}x{Lxdeg}', fontsize=15)
##axs[0].set_xlabel(r'$\theta$[deg]', fontsize=15)
##axs[0].set_ylabel(r'$\theta$[deg]', fontsize=15)
#im1=axs[1].imshow(mask_s,cmap=cmap)
#axs[1].set_title(f'size : {Lxdeg_s}x{Lxdeg_s}', fontsize=15)
#norm = colors.Normalize(vmin=0, vmax=1)
#plt.subplots_adjust(wspace=0.2, hspace=0.4, bottom=0.3, left=0.05, top=0.85)
#sub_ax = plt.axes([0.91, 0.367, 0.02, 0.46]) 
#fig.colorbar(im1, ax=axs, cax=sub_ax,location='right',orientation='vertical',label='T [mK]')


# Fields:
# Once you have maps it's time to create pymaster fields.
# Note that, as in the full-sky case, you can also pass
# contaminant templates and flags for E and B purification
# (see the documentation for more details)
f0 = nmt.NmtFieldFlat(Lx, Ly, mask, [mpt])
f0_cart = nmt.NmtFieldFlat(Lx, Ly, mask, [mpt_cart])
f0_s = nmt.NmtFieldFlat(Lx_s, Ly_s, mask_s, [mpt_s])

# Bins:
# For flat-sky fields, bandpowers are simply defined as intervals in ell, and
# pymaster doesn't currently support any weighting scheme within each interval.
delta_ell=8
l0_bins = np.arange(Nx/delta_ell) * delta_ell * np.pi/Lx
lf_bins = (np.arange(Nx/delta_ell)+1) * delta_ell * np.pi/Lx
b = nmt.NmtBinFlat(l0_bins, lf_bins)

# The effective sampling rate for these bandpowers can be obtained calling:
ells_uncoupled = b.get_effective_ells()
print(b.get_n_bands())

l0_bins_s = np.arange(Nx/delta_ell) * delta_ell * np.pi/Lx_s
lf_bins_s = (np.arange(Nx/delta_ell)+1) * delta_ell * np.pi/Lx_s
b_s = nmt.NmtBinFlat(l0_bins_s, lf_bins_s)
# The effective sampling rate for these bandpowers can be obtained calling:
ells_uncoupled_s = b_s.get_effective_ells()

# Workspaces:
# As in the full-sky case, the computation of the coupling matrix and of
# the pseudo-CL estimator is mediated by a WorkspaceFlat case, initialized
# by calling its compute_coupling_matrix method:
w00 = nmt.NmtWorkspaceFlat()
w00.compute_coupling_matrix(f0, f0, b)

w00_cart = nmt.NmtWorkspaceFlat()
w00_cart.compute_coupling_matrix(f0_cart, f0_cart, b)

w00_s = nmt.NmtWorkspaceFlat()
w00_s.compute_coupling_matrix(f0_s, f0_s, b_s)


# Workspaces can be saved to and read from disk to avoid recomputing them:
#w00.write_to("w00_flat.fits")
#w00.read_from("w00_flat.fits")
#w02.write_to("w02_flat.fits")
#w02.read_from("w02_flat.fits")
#w22.write_to("w22_flat.fits")
#w22.read_from("w22_flat.fits")

# Computing power spectra:
# As in the full-sky case, you compute the pseudo-CL estimator by
# computing the coupled power spectra and then decoupling them by
# inverting the mode-coupling matrix. This is done in two steps below,
# but pymaster provides convenience routines to do this
# through a single function call
cl00_coupled=nmt.compute_full_master_flat(f0, f0, b)
cl00_coupled_cart=nmt.compute_full_master_flat(f0_cart, f0_cart, b)
cl00_coupled_s=nmt.compute_full_master_flat(f0_s, f0_s, b_s)
#cl00_coupled = nmt.compute_coupled_cell_flat(f0, f0, b)
#cl00_uncoupled = w00.decouple_cell(cl00_coupled)

factor = l*(l+1)/(2.*np.pi)
factor_uncoupled = ells_uncoupled*(ells_uncoupled+1)/(2.*np.pi)
factor_uncoupled_s = ells_uncoupled_s*(ells_uncoupled_s+1)/(2.*np.pi)


# Let's look at the results!
#plt.figure()
#plt.suptitle(f'Stripe82 lon:{lon}, lat:{lat}')
#plt.plot(l, factor*cl_tt, label='Input TT, full sky')
##plt.plot(l, cl_ee, 'g-', label='Input EE')
##plt.plot(l, cl_bb, 'b-', label='Input BB')
#plt.plot(ells_uncoupled, factor_uncoupled*cl00_coupled[0],  label=f'Gnomview size:{Lxdeg}x{Lxdeg}')
#plt.plot(ells_uncoupled, factor_uncoupled*cl00_coupled_cart[0],  label=f'Cartview size:{Lxdeg}x{Lxdeg}')
#plt.plot(ells_uncoupled_s, factor_uncoupled_s*cl00_coupled_s[0],  label=f'Gnomview size:{Lxdeg_s}x{Lxdeg_s}')
##plt.loglog()
#plt.xlim([0,1000])
#plt.xlabel(r'$\ell$')
#plt.ylabel(r'$D_{\ell}$')
#plt.legend()
#plt.show()


del mask_s; del mpt; del mpt_cart; del mpt_s; del f0; del f0_cart; del f0_s; del b; del b_s
##########################################################################################################
####################### rifacciamo indietro ############################################################
mask_stripe = hp.read_map(f'mask_patch_stripe82_binary_lmax{lmax}_nside{nside}.fits')

#

f_0_mask = nmt.NmtField(mask_stripe,[map_cltt] )

b = nmt.NmtBin.from_nside_linear(nside, delta_ell)
ells_mask= b.get_effective_ells()

cltt_mask = nmt.compute_full_master(f_0_mask, f_0_mask, b)[0]


##### applichiamo la maschera e poi togliamo il patch ####
map_cltt_mask = map_cltt*mask_stripe #- HI_maps_freq_mean
map_cltt_mask = hp.remove_dipole(map_cltt_mask)
hp.mollview(map_cltt_mask, cmap='viridis')

rotation=(dict_match[0]["lc"], dict_match[0]["bc"], 0.) #deg
low_lonra  = dict_match[0]["lc"] - Lxdeg/2.   #deg
high_lonra = dict_match[0]["lc"] + Lxdeg/2.  #deg
low_latra  = dict_match[0]["bc"] - Lxdeg/2.  #deg
high_latra = dict_match[0]["bc"] + Lxdeg/2.   #deg

grid_cart = hp.visufunc.cartview(map=map_cltt_mask, coord='G',rot=rotation ,  xsize=Nx,ysize=Ny, return_projected_map=True, cmap='viridis', title='Cartview')#lonra=[low_lonra,high_lonra], latra=[low_latra,high_latra]
grid_cart=grid_cart[::-1]
plt.close()
grid_cart_lonra_latra = hp.visufunc.cartview(map=map_cltt_mask, coord='G',lonra=[low_lonra,high_lonra], latra=[low_latra,high_latra],  xsize=Nx,ysize=Ny, return_projected_map=True, cmap='viridis', title='Cartview')
grid_cart_lonra_latra=grid_cart_lonra_latra[::-1]
plt.close()
grid_gnom = hp.visufunc.gnomview(map=map_cltt_mask, coord='G', rot=rotation ,reso=reso_arcmin, xsize=Nx,ysize=Ny, return_projected_map=True,cmap='viridis',  title='Gnomview', no_plot=True)
grid_gnom=grid_gnom[::-1]

mask_gnom = hp.visufunc.gnomview(map=mask_stripe, coord='G', rot=rotation ,reso=reso_arcmin, xsize=Nx,ysize=Ny, return_projected_map=True,cmap='viridis',  title='Gnomview', no_plot=True)
mask_gnom=mask_gnom[::-1]

fig = plt.figure()
plt.imshow(mask_gnom, cmap='viridis')




fig, axs = plt.subplots(1,3, figsize=(12,6))
cmap= 'viridis'
fig.suptitle(f'Patch projection',fontsize=19)
im0=axs[0].imshow(grid_cart,cmap=cmap)
axs[0].set_title(f'Cartview rot', fontsize=15)
im1=axs[1].imshow(grid_cart_lonra_latra,cmap=cmap)
axs[1].set_title(f'Cartview lon lat', fontsize=15)
im2=axs[2].imshow(grid_gnom,cmap=cmap)
axs[2].set_title(f'Gnomview', fontsize=15)
norm = colors.Normalize(vmin=0, vmax=1)
plt.subplots_adjust(wspace=0.2, hspace=0.4, bottom=0.3, left=0.05, top=0.85)
sub_ax = plt.axes([0.91, 0.367, 0.02, 0.46]) 
fig.colorbar(im2, ax=axs, cax=sub_ax,location='right',orientation='vertical',label='T [mK]')
plt.show()



ell_grid_cart,cl_cart = compute_Cl(mask=mask, patch=grid_cart, Nx=Nx, Ny=Ny, Lxdeg=Lxdeg, Lydeg=Lydeg, demask=True)
factor_grid_cart = ell_grid_cart*(ell_grid_cart+1)/(2.*np.pi)
ell_grid_cart_lonra_latra,cl_cart_lonra_latra = compute_Cl(mask=mask, patch=grid_cart_lonra_latra, Nx=Nx, Ny=Ny, Lxdeg=Lxdeg, Lydeg=Lydeg, demask=True)
factor_grid_cart1 = ell_grid_cart_lonra_latra*(ell_grid_cart_lonra_latra+1)/(2.*np.pi)
ell_grid_gnom,cl_gnom = compute_Cl(mask=mask, patch=grid_gnom, Nx=Nx, Ny=Ny, Lxdeg=Lxdeg, Lydeg=Lydeg, demask=True)
factor_grid_gnom = ell_grid_gnom*(ell_grid_gnom+1)/(2.*np.pi)
ell_grid_gnom_mask,cl_gnom_mask = compute_Cl(mask=mask_gnom, patch=grid_gnom, Nx=Nx, Ny=Ny, Lxdeg=Lxdeg, Lydeg=Lydeg, demask=True)
factor_grid_gnom_mask = ell_grid_gnom_mask*(ell_grid_gnom_mask+1)/(2.*np.pi)


#
factor_mask =ells_mask*(ells_mask+1)/(2.*np.pi)
#fig = plt.figure()
#plt.suptitle(f'Cl Stripe82 lon:{lon}, lat:{lat}')
#plt.plot(l, factor*cl_tt, 'k',label='Input TT, full sky')
#plt.plot(ells_mask,factor_mask*cltt_mask, label='Mask decoupled')
#plt.plot(ells_uncoupled, factor_uncoupled*cl00_coupled[0],  label=f'Gnomview size:{Lxdeg}x{Lxdeg}')
#plt.plot(ells_uncoupled, factor_uncoupled*cl00_coupled_cart[0],  label=f'Cartview size:{Lxdeg}x{Lxdeg}')
##plt.plot(ells_uncoupled_s, factor_uncoupled_s*cl00_coupled_s[0],  label=f'Gnomview size:{Lxdeg_s}x{Lxdeg_s}')
#plt.xlim([-10,500])
#plt.xlabel(r'$\ell$')
#plt.ylabel(r'$D_{\ell}$')
#plt.legend()

fig=plt.figure()
plt.suptitle(f'Map masked Stripe82 lon:{lon}, lat:{lat}')
plt.plot(l, factor*cl_tt, 'k',label='Input TT, full sky')
plt.plot(ells_mask,factor_mask*cltt_mask, label='Mask decoupled')
plt.plot(ell_grid_cart,factor_grid_cart*cl_cart, label = 'Patch cartview')
plt.plot(ell_grid_cart_lonra_latra,factor_grid_cart1*cl_cart_lonra_latra, label = 'Patch cartview lonra lontra')
plt.plot(ell_grid_gnom,factor_grid_gnom*cl_gnom, label = 'Patch gnomview')
plt.plot(ell_grid_gnom_mask,factor_grid_gnom_mask*cl_gnom_mask, label = 'Patch gnomview mask')
plt.xlim([-10,800])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell(\ell+1)}{2\pi} C_{\ell} $')
#plt.xlim(left=-0.01,right=20)
#plt.ylim(bottom=-0.001,top=0.1)
plt.legend()

plt.show()