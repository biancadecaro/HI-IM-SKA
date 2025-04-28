import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from copy import deepcopy
import pymaster as nmt
import healpy as hp
import difflib
import math 
from scipy import ndimage
from matplotlib import colors
import pickle
###################################################################################
" Curve starlets"

def tab2alm(tab):

    lmax = int(np.shape(tab)[0])-1
    taille = int(lmax*(lmax+3)/2)+1
    alm = np.zeros((taille,),dtype=complex)

    for r in range(0,taille):
        l,m = hp.sphtfunc.Alm.getlm(lmax,r)
        alm[r] = complex(tab[l,m,0],tab[l,m,1])

    return alm

def alm2tab(alm,lmax):

    taille = np.size(alm)
    tab = np.zeros((lmax+1,lmax+1,2))

    for r in range(0,taille):
        l,m = hp.sphtfunc.Alm.getlm(lmax,r)
        tab[l,m,0] = np.real(alm[r])
        tab[l,m,1] = np.imag(alm[r])

    return tab

def almtrans(map,lmax=None):

    # To be done

    if lmax==None:
        lmax = 3.*hp.get_nside(map)
        print("lmax = ",lmax)

    alm = hp.sphtfunc.map2alm(map,lmax=lmax)

    tab = alm2tab(alm,lmax)

    return tab

def almrec(tab,nside=512):

    alm = tab2alm(tab)

    map = hp.alm2map(alm,nside)

    return map

def alm_product(tab,filt):

    length=np.size(filt)
    lmax = np.shape(tab)[0]

    if lmax > length:
        print("Filter length is too small")

    for r in range(lmax):
        tab[r,:,:] = filt[r]*tab[r,:,:]

    return tab

def spline2(size,l,lc):

    res =np.linspace(0,size,size+1)
    
    res = 2.0 * l * res / (lc *size)
    #print(f'res:{res}')
    tab = (3.0/2.0)*1.0 /12.0 * (( abs(res-2))**3 - 4.0* (abs(res-1))**3 + 6 *(abs(res))**3 - 4.0 *( abs(res+1))**3+(abs(res+2))**3)
    #print(f'tab:{tab}')
    return tab

def compute_h(size,lc):

    tab1 = spline2(size,2.*lc,1)
    tab2 = spline2(size,lc,1)
    h = tab1/(tab2+0.000001)
    h[int(size/(2.*lc)):size]=0.

    return h

def compute_g(size,lc):

    tab1 = spline2(size,2.*lc,1)
    tab2 = spline2(size,lc,1)
    g = (tab2-tab1)/(tab2+0.000001)
    g[int(size/(2.*lc)):size]=1

    return g


def wttrans_getfilters(nscale=4,lmax=128):

    ech = 1

    filt = np.zeros((lmax+1,nscale))
    f = np.ones((lmax+1,))

    for j in range(nscale-1):

        h = compute_h(lmax,ech)

        filt[:,j] = f - h*f

        f = h*f

        ech = 2*ech

    filt[:,nscale-1] = f

    return filt

def wttrans(map,nscale=4,lmax=128):

    ech = 1

    taille = np.size(map)

    alm = almtrans(map,lmax=lmax)
    alm_temp = deepcopy(alm)

    LScale = deepcopy(map)
    nside = hp.get_nside(map)

    WT = np.zeros((taille,nscale))

    for j in range(nscale-1):

        h = compute_h(lmax,ech)
        # g = compute_g(lmax,ech) # Needed if the difference is computed in the spherical harmonics domain

        alm_temp = alm_product(alm,h)

        m = almrec(alm_temp,nside=nside)

        HScale = LScale - m
        LScale = m

        WT[:,j] = HScale

        ech = 2*ech

    WT[:,nscale-1] = LScale

    return WT


"Flat starlets"
############################################################

def length(x=0):

	l = np.max(np.shape(x))
	return l

################# 1D convolution

def filter_1d(xin=0,h=0,boption=3):

	import numpy as np
	import scipy.linalg as lng
	import copy as cp

	x = np.squeeze(cp.copy(xin));
	n = length(x);
	m = length(h);
	y = cp.copy(x);

	z = np.zeros(1,m);

	m2 = np.int(np.floor(m/2))

	for r in range(m2):

		if boption == 1: # --- zero padding

			z = np.concatenate([np.zeros(m-r-m2-1),x[0:r+m2+1]],axis=0)

		if boption == 2: # --- periodicity

			z = np.concatenate([x[n-(m-(r+m2))+1:n],x[0:r+m2+1]],axis=0)

		if boption == 3: # --- mirror

			u = x[0:m-(r+m2)-1];
			u = u[::-1]
			z = np.concatenate([u,x[0:r+m2+1]],axis=0)

		y[r] = np.sum(z*h)


	a = np.arange(np.int(m2),np.int(n-m+m2),1)

	for r in a:

		y[r] = np.sum(h*x[r-m2:m+r-m2])


	a = np.arange(np.int(n-m+m2+1)-1,n,1)

	for r in a:

		if boption == 1: # --- zero padding

			z = np.concatenate([x[r-m2:n],np.zeros(m - (n-r) - m2)],axis=0)

		if boption == 2: # --- periodicity

			z = np.concatenate([x[r-m2:n],x[0:m - (n-r) - m2]],axis=0)

		if boption == 3: # --- mirror

			u = x[n - (m - (n-r) - m2 -1)-1:n]
			u = u[::-1]
			z = np.concatenate([x[r-m2:n],u],axis=0)

		y[r] = np.sum(z*h)

	return y

################# 1D convolution with the "a trous" algorithm

def Apply_H1(x=0,h=0,scale=1,boption=3):

	m = length(h)

	if scale > 1:
		p = (m-1)*np.power(2,(scale-1)) + 1
		g = np.zeros( p)
		z = np.linspace(0,m-1,m)*np.power(2,(scale-1))
		g[z.astype(int)] = h

	else:
		g = h

	y = filter_1d(x,g,boption)

	return y

################# 2D "a trous" algorithm

def forward(x=0,h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3):

	nx = np.shape(x)
	m = np.zeros((nx[0],nx[1],nx[2],J+1))	
	for r in range(nx[0]):		
		c,w = Starlet_Forward2D(x=x[r,:,:],h=h,J=J,boption=boption)		
		m[r,:,:,0:J] = w		
		m[r,:,:,J] = c	
	return m

#def backward(c,w):
#
#  return c + np.sum(w,axis=3)


def Starlet_Forward2D(x=0,h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3):

	import copy as cp

	nx = np.shape(x)
	c = np.zeros((nx[0],nx[1]),dtype=complex)
	w = np.zeros((nx[0],nx[1],J))

	c = cp.copy(x)
	cnew = cp.copy(x)

	for scale in range(J):

		for r in range(nx[0]):

			cnew[r,:] = Apply_H1(c[r,:],h,scale,boption)

		for r in range(nx[1]):

			cnew[:,r] = Apply_H1(cnew[:,r],h,scale,boption)

		w[:,:,scale] = c - cnew

		c = cp.copy(cnew);

	return c,w


#def Starlet_Forward1D(x=0,h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3):
#
#	import copy as cp
#
#	c = np.zeros((len(x),),dtype=complex)
#	w = np.zeros((len(x),J))
#
#	c = cp.copy(x)
#	cnew = cp.copy(x)
#
#	for scale in range(J):
#
#		cnew = Apply_H1(c,h,scale,boption)
#		w[:,scale] = c - cnew
#
#		c = cp.copy(cnew);
#
#	return c,w
#
#def Starlet_Backward1D(c,w):
#
#	import numpy as np
#
#	return c + np.sum(w,axis=1)
#
#def Starlet_Backward2D(c,w):
#
#	import numpy as np
#
#	return c + np.sum(w,axis=2)


## Let's wavelet transform
def wt_transf_2d(Xin,Jin=3,hlist=[0.0625,0.25,0.375,0.25,0.0625]):
    """
    X has [dims,nx,ny] dimensions
    """
    assert len(np.shape(Xin))==3
    X_wt = forward(x=Xin,h=hlist,J=Jin)
    X_wt_c =X_wt[:,:,:,:].reshape(np.shape(Xin)[0],-1) 
    return X_wt_c

def X_nocoarse(X_wt_c,nchan,nx,ny,Jin=3):
    assert np.shape(X_wt_c)[0]==nchan
    assert np.shape(X_wt_c)[1]==(nx*ny*(Jin+1))
    X_wt = X_wt_c.reshape(nchan,nx,ny,-1) 
    X_wt_j =X_wt[:,:,:,0:Jin].reshape(nchan,-1) 
    return X_wt_j

## look at individual wavelet scale ##
def get_aWLscale(X_wt_c,j,nchan,nx,ny):
    assert np.shape(X_wt_c)[0]==nchan
    X_wt = X_wt_c.reshape(nchan,nx,ny,-1) 
    Xj   = X_wt[:,:,:,j].reshape(nchan,-1); del X_wt
    return Xj 


#################################################################################
"Proiezione patch"
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

def reduce_pixels(hmap, region=None, size=54.1,final_pixel = [256,256]):
    
    patch = extract_region(hmap, region, size)
    
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
##########################################################################################
"Prepariamo la sfera e il patch"

## map
beam_s = 'theta40arcmin'
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
#fg_maps_freq = file['maps_sims_fg']
#full_maps_freq = file['maps_sims_tot'] + file['maps_sims_noise']  #aggiungo il noise

npix = np.shape(HI_maps_freq)[1]
nside = hp.get_nside(HI_maps_freq[0])
lmax=3*nside-1
Jin = 1
print(f'nside:{nside}, lmax:{lmax}, num_ch:{num_freq}, min_ch:{min(nu_ch)}, max_ch:{max(nu_ch)}')

## mask
mask_stripe = hp.read_map(f'mask_patch_stripe82_binary_lmax{lmax}_nside{nside}.fits')

### selezioniano una mappa sola

ich = int(num_freq/2)

HI_maps_freq_ch = HI_maps_freq[ich]  #aggiungo il noise

del HI_maps_freq; del file

##############################################
# estraiamo il patch

HI_maps_freq_ch_sq = reduce_pixels(HI_maps_freq_ch, region='Stripe82')
HI_maps_freq_ch_sq = HI_maps_freq_ch_sq[::-1]

#########################
# ora estraiamo le starlets

### curve 

HI_maps_freq_ch_st = wttrans(map=HI_maps_freq_ch, nscale=Jin, lmax=lmax)

#### flat

HI_maps_freq_ch_sq_st=wt_transf_2d(HI_maps_freq_ch_sq, Jin=1)


