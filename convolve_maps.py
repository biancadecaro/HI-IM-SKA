import os
from scipy.special import j1
import numpy as np
import healpy as hp 
import math as mth
import time
import scipy
from scipy import interpolate
from scipy.special import j0
from scipy.integrate import quad
from scipy.integrate import trapezoid
from astropy.io import fits
from scipy.interpolate import CubicSpline
import sys
from convolution_func import*
import h5py



if __name__ == '__main__':
    from optparse import OptionParser
    o = OptionParser()
    o.set_usage('%prog [options]')
    o.set_description(__doc__)
    o.add_option('--nside',dest='nside',default=512,help='nside')
    o.add_option('--Amp',dest='Amp',default=0.0,help='ripple amplitude 0.0 or 0.1')
    o.add_option('--T_p',dest='T_p',default=20,help='Ripple oscillation in Mhz') 
    o.add_option('--smooth_flag',action='store_true',dest='smooth_flag',default=False,help='If True l/D is corrected with a 8 deg pol')
    o.add_option('--beam',dest='beam',default='cosine',help='beam type: cosine, gaussian, jinc')
    o.add_option('--maps_in',dest='maps_in',default='fake_hdf5',help='filename of full sky map to convolve')
    o.add_option('--outdir',dest='outdir',default='/home/spinelli/Documents/ISA/code/',help='name of full sky map to convolve')
    o.add_option('--file_out',dest='file_out',default='convolved_maps.hdf5',help='output file name')


    opts, args = o.parse_args(sys.argv[1:]) 

    print(opts, args) 
    nu0=1420 #ref 21cm
    Amp=float(opts.Amp)
    T_p=float(opts.T_p)
    nside=int(opts.nside)
    beam=str(opts.beam)
    outdir=str(opts.outdir)
    file_out=str(opts.file_out)
    maps_in=str(opts.maps_in)
    
    smooth=False
    if opts.smooth_flag: 
        smooth=True
        print('not lambda/D anymore.. unsing smooth model')
    
    #read in file
    filename = h5py.File(maps_in,'r')
    print(filename.keys())
    components = list(filename.keys()); components.remove('frequencies')
    nu_ch = np.array(filename['frequencies'])
    
    nside_in=hp.npix2nside(len(np.array(filename[components[0]][0,:])))
    print(nside_in)
    if nside!=nside_in: raise ValueError('check your nside')
    
    print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
    print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')
    print(f'corresponding to the redshift range z: [{min(nu0/nu_ch -1.0):.2f} - {max(nu0/nu_ch -1.0):.2f}] ')
    
    
    initial_freq=np.min(nu_ch) #start frequency channel 
    final_freq=np.max(nu_ch) #end frequency channel
    binwidth=nu_ch[1]-nu_ch[0]
    nfreqs=len(nu_ch)
    npix=hp.nside2npix(nside)
    
    
    #add all maps
    tot_maps=np.zeros((nfreqs,npix),dtype=float)
    for c in components:
        tmp=np.array(filename[c])
        tot_maps+=tmp
        
    filename.close()
    print('all components sum up!')

    print('Constructing beam model')
    
    bl_vec=create_bl_vec(beam,nside,T_p,Amp,smooth,nu_ch)

    convolved_maps=np.zeros((nfreqs,npix),dtype=float)
    convolved_maps=convolution_bl(tot_maps,bl_vec,nside)
    
    print('maps convolved with ', beam, ', now saving...')
    
    with h5py.File(outdir+file_out, "w") as f:
        dset = f.create_dataset("convolved_maps", data=convolved_maps)
        dset = f.create_dataset("frequencies", data=nu_ch)
        dset = f.create_dataset("components", data=np.array(components, dtype='S'))
        beam_info=[beam,str(Amp),str(T_p),str(smooth)]
        dset = f.create_dataset("beam_info", data=np.array(beam_info, dtype='S'))
        
        
        
     
    
    
    
    
    




    
