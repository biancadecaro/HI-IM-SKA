import healpy as hp
from astropy.coordinates import spherical_to_cartesian
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import ndimage
import difflib
import pymaster as nmt
from matplotlib import colors
import pymaster as nm
sns.set_theme(style = 'white')
sns.color_palette(n_colors=15)
#sns.set_palette('husl',15)

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

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

def extract_region(hmap, region=None, size=54.1):

    dict_match = get_region(name=region, size=size)
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
###############################################################################################################
# Import the NaMaster python wrapper
def compute_Cl(mask,patch,Nx,Ny,Lxdeg,Lydeg, nbin=8, demask=False):

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
#########################################################################################################
#########################################################################################################

size = 54.1
nside = 128
lmax = 3*nside-1
npix = hp.nside2npix(nside)

map1 = np.ones(npix)

map1res82=  patch = extract_region(map1, region='Stripe82', size=size)
print(map1res82.shape)

map1res82 = map1res82[::-1]

#plt.imshow(map1res82, cmap = 'viridis', vmin=0, vmax=1)
#######################################################


delta = size/2
dict_match = get_region(name="Stripe82", size=size)

centre_b,centre_l= dict_match[0]["bc"],dict_match[0]["lc"]

v1 = [centre_b-delta, centre_l+delta]
v2 = [centre_b-delta, centre_l-delta]
v3 = [centre_b+delta, centre_l-delta]
v4 = [centre_b+delta, centre_l+delta]
#pix_v1 = hp.ang2pix(nside=nside, theta=math.radians(90.-v1[0]),phi=math.radians(v1[1]))
#pix_v2 = hp.ang2pix(nside=nside, theta=math.radians(90.-v2[0]),phi=math.radians(v2[1]))
#pix_v3 = hp.ang2pix(nside=nside, theta=math.radians(90.-v3[0]),phi=math.radians(v3[1]))
#pix_v4 = hp.ang2pix(nside=nside, theta=math.radians(90.-v4[0]),phi=math.radians(v4[1]))

hp.mollview(title="Stripe 82")
hp.graticule()

x= [math.radians(90.-v1[0]), math.radians(90.-v2[0]), math.radians(90.-v3[0]), math.radians(90.-v4[0])]
y= [math.radians(v1[1]), math.radians(v2[1]), math.radians(v3[1]), math.radians(v4[1])]

hp.projscatter(x, y, lonlat=False, coord='G')

hp.projtext(math.radians(90.-v1[0]),math.radians(v1[1]), 'v1', lonlat=False, coord='G')
hp.projtext(math.radians(90.-v2[0]),math.radians(v2[1]), 'v2', lonlat=False, coord='G')
hp.projtext(math.radians(90.-v3[0]),math.radians(v3[1]), 'v3', lonlat=False, coord='G')
hp.projtext(math.radians(90.-v4[0]),math.radians(v4[1]), 'v4', lonlat=False, coord='G')

plt.savefig(f'patch_Stripe82_poly_proj_lmax{lmax}_nside{nside}.png')
plt.show()
###########################################################
mask = np.zeros((npix))
st = hp.query_polygon(nside, get_vertices('Stripe82', size=size), inclusive=True, nest=False)
mask[st] += 1.0

#hp.write_map(f'mask_patch_stripe82_binary_lmax{lmax}_nside{nside}', mask, overwrite=True)

hp.mollview(mask, cmap = 'viridis', title='Stripe 82 mask', cbar=False)

plt.savefig(f'mask_patch_Stripe82_lmax{lmax}_nside{nside}.png')
plt.show()

##################################################################3#
################## torniamo indietro #################################

#patch_back = reduce_pixels(mask, region='Stripe82')
#patch_back = patch_back[::-1]
#plt.imshow(patch_back, cmap='viridis')

coord = get_vertices(region='Stripe82', size=size)
        
rotation=(dict_match[0]["lc"], dict_match[0]["bc"], 0.) #deg
print(dict_match[0]["lc"], dict_match[0]["bc"])

reso_arcmin=hp.nside2resol(nside, arcmin=True) #1/60 deg
nxpix = int(np.ceil((dict_match[0]["dl"])*60./reso_arcmin)) 
nypix = int(np.ceil((dict_match[0]["db"])*60./reso_arcmin)) 
print(nxpix, nypix)

low_lonra  = dict_match[0]["lc"] - size/2.   #deg
high_lonra = dict_match[0]["lc"] + size/2.  #deg
low_latra  = dict_match[0]["bc"] - size/2.  #deg
high_latra = dict_match[0]["bc"] + size/2.   #deg

grid_cart = hp.visufunc.cartview(map=mask, coord='G',rot=rotation ,  xsize=nxpix,ysize=nypix, return_projected_map=True, cmap='viridis', title='Cartview')#lonra=[low_lonra,high_lonra], latra=[low_latra,high_latra]
grid_cart=grid_cart[::-1]
plt.close()
grid_cart1 = hp.visufunc.cartview(map=mask, coord='G',lonra=[low_lonra,high_lonra], latra=[low_latra,high_latra],  xsize=nxpix,ysize=nypix, return_projected_map=True, cmap='viridis', title='Cartview')
grid_cart1=grid_cart1[::-1]
plt.close()
grid_gnom = hp.visufunc.gnomview(map=mask, coord='G', rot=rotation ,reso=reso_arcmin, xsize=nxpix,ysize=nypix, return_projected_map=True,cmap='viridis',  title='Gnomview', no_plot=True)
grid_gnom=grid_gnom[::-1]

fig, axs = plt.subplots(1,3, figsize=(12,6))
cmap= 'viridis'
fig.suptitle(f'Patch projection',fontsize=19)
im0=axs[0].imshow(grid_cart,vmin=0, vmax=1,cmap=cmap)
axs[0].set_title(f'Cartview rot', fontsize=15)
im1=axs[1].imshow(grid_cart1,vmin=0, vmax=1,cmap=cmap)
axs[1].set_title(f'Cartview lon lat', fontsize=15)
im2=axs[2].imshow(grid_gnom,vmin=0, vmax=1,cmap=cmap)
axs[2].set_title(f'Gnomview', fontsize=15)
norm = colors.Normalize(vmin=0, vmax=1)
plt.subplots_adjust(wspace=0.2, hspace=0.4, bottom=0.3, left=0.05, top=0.85)
sub_ax = plt.axes([0.91, 0.367, 0.02, 0.46]) 
fig.colorbar(im2, ax=axs, cax=sub_ax,location='right',orientation='vertical',label='T [mK]')


## spline of order 3 for interpolation:
#final_patch = ndimage.zoom(grid, zoom_param, order=3)
##final_patch = final_patch[::-1]
#fig=plt.figure()
#plt.suptitle('Final patch')
#plt.imshow(final_patch, cmap='viridis')
lmax_cl=2*nside

f_0_mask = nmt.NmtField(mask,[map1] )
b = nmt.NmtBin.from_nside_linear(nside, 8)
ells_full= b.get_effective_ells()
factor_full = ells_full*(ells_full+1)/(2.*np.pi)

cl_full = nmt.compute_full_master(f_0_mask, f_0_mask, b)[0]



#cl_full = hp.anafast(mask, lmax=lmax_cl)
#fig = plt.figure()
#plt.plot(factor*cl_full)
#

########################################################################################
Nx, Ny = grid_cart.shape
Lxdeg = size#54.1
Lydeg = size#54.1
Lx =  Lxdeg* np.pi/180.
Ly =  Lydeg* np.pi/180.
#mask = np.ones_like(grid_cart).flatten()
#mask = mask.reshape([Ny, Nx])

ell_grid_cart,cl_cart = compute_Cl(mask=grid_cart, patch=grid_cart, Nx=Nx, Ny=Ny, Lxdeg=Lxdeg, Lydeg=Lydeg)
factor_grid_cart = ell_grid_cart*(ell_grid_cart+1)/(2.*np.pi)
ell_grid_cart1,cl_cart1 = compute_Cl(mask=grid_cart1, patch=grid_cart1, Nx=Nx, Ny=Ny, Lxdeg=Lxdeg, Lydeg=Lydeg)
factor_grid_cart1 = ell_grid_cart1*(ell_grid_cart1+1)/(2.*np.pi)
ell_grid_gnom,cl_gnom = compute_Cl(mask=grid_gnom, patch=grid_gnom, Nx=Nx, Ny=Ny, Lxdeg=Lxdeg, Lydeg=Lydeg)
factor_grid_gnom = ell_grid_gnom*(ell_grid_gnom+1)/(2.*np.pi)

fig=plt.figure()
plt.plot(ells_full,factor_full*cl_full, label = 'Full sky')
plt.plot(ell_grid_cart,factor_grid_cart*cl_cart, label = 'Patch cartview')
plt.plot(ell_grid_cart1,factor_grid_cart1*cl_cart1, label = 'Patch cartview1')
plt.plot(ell_grid_gnom,factor_grid_gnom*cl_gnom, label = 'Patch gnomview')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$ \frac{\ell(\ell+1)}{2\pi} C_{\ell} $')
#plt.xlim(left=-0.01,right=20)
#plt.ylim(bottom=-0.001,top=0.1)
plt.legend()

#cl_cart_interp = np.interp(ell, ell_grid_cart,cl_cart)
#cl_cart1_interp = np.interp(ell, ell_grid_cart1,cl_cart1)
#cl_gnom_interp = np.interp(ell, ell_grid_gnom,cl_gnom)
#
#fig=plt.figure()
#plt.plot(ell,factor_full*cl_full, label = 'Full sky')
#plt.plot(ell,factor_full*cl_cart_interp, label = 'Patch cartview interp')
#plt.plot(ell,factor_full*cl_cart1_interp, label = 'Patch cartview1 interp')
#plt.plot(ell,factor_full*cl_gnom_interp, label = 'Patch gnomview interp')
#plt.xlabel(r'$\ell$')
#plt.ylabel(r'$ \frac{\ell(\ell+1)}{2\pi} C_{\ell} $')
##plt.xlim(left=-0.01,right=20)
##plt.ylim(bottom=-0.001,top=0.1)
#plt.legend()

plt.show()