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



############################################generals
def decibel(x):
    return 10*np.log10(x)



def deg2rad(deg):
    return deg*(np.pi/180.)


def freq2lamda(frequency):
    c=299792458 
    return c/(frequency*1e6)

####################################################

##################################Beams

#---------------------------------DEFINE COS BEAM
def cos_beam(theta, FWHM):  
    
    x=1.189*theta/FWHM
 
    return (np.cos(x*np.pi)/(1-4*x**2))**2

#-----------------------------------------------



#-----------------------------DEFINE JINC FUNCTION

def jinc(x):    

    if x == 0.0:
        return 0.5
    return j1(x) / x

#-----------------------------------------------------------------


#---------------------------------DEFINE GAUSSIAN FUNCTION
def gaussian(theta, delta_theta):   # delta_theta is FWHM. 
    
    sigma=delta_theta/np.sqrt(8*np.log(2))
    coeff=1/sigma*(np.sqrt(2*np.pi))
  
    gauss=coeff*np.exp(-np.power((theta/sigma), 2.)/2.)

    return gauss

#-----------------------------------------------------------

#-----------------------------------------------Harper's (AIRY)

def E_r(rho,sigma):
    return np.exp(-0.5*(rho/sigma)**2) 




def generate_harper_beam(theta,freq):
    
    
    lamda=freq2lamda(freq)    # compute wavelength at a given frequency
    D=13.5                    # MeerKAT dish diameter
    delta_theta=lamda/D       # FWHM
    
    sigma=(1/(2*delta_theta)) # 0.5*
    
    
    rho=np.linspace(0,sigma,num=1e4)
    
    deno=E_r(rho,sigma)*rho         #  denominator function of equation 4  
    
    integral_deno=trapezoid(deno,rho)   #computing the integral of the denominator of equation 4 
    
    sine=np.sin(deg2rad(theta))
    
    num_integrand=E_r(rho,sigma)*rho*j0(2.0*np.pi*sine*rho)
    
    Beam_theta=(np.abs(((trapezoid(num_integrand,rho)/integral_deno)))**2)

    
    return Beam_theta



#-----------------------------------------------------------------------
def createDtheta(nu,dish_diameter,T_p,Amp,smooth):
    #compute FWHM for different models: use Amp=0 for no ripple, smooth=False &Amp=0 gives standard lambda/D
    speed_of_light=299792458   # speed of light in m/s
    first_zero_of_jinc_function=3.8317 #first zero of jinc function : position of the first null 
    #dish_diameter=13.5 #*2 # diameter of telescope dish in meters (m)
    frequency=nu*1e6   #frequency in hertz (Hz)
     
      #adding ripple
    T=2*np.pi/(float(T_p)*1e6) #angular frequency / cycle of the wave is 20 MHz

    Amp=Amp/60.  #arcmin to degrees   
    sine_wave=(Amp*np.sin(frequency*T))
 
    fwhm=np.degrees((speed_of_light/frequency)/dish_diameter) 
    if smooth==True: 
        
        smooth_par=[ 3.40234907e-21, -3.02516490e-17,  1.17019761e-13, -2.57168633e-10,3.51130410e-07, -3.04953935e-04,  1.64488782e-01, -5.03702034e+01,6.70428133e+03]
        p_smooth = np.poly1d(smooth_par)

        delta_theta=fwhm*(p_smooth(nu) + sine_wave)

    else: delta_theta=1.16*fwhm + sine_wave  
        
    return delta_theta
    
    
    
def beam2bl(beam,nside,nu,dish_diameter,T_p,Amp,smooth):
    
    delta_theta=createDtheta(nu,dish_diameter,T_p,Amp,smooth)*np.pi/180.

    npix=hp.nside2npix(nside)
    pixelized_beam=np.zeros(npix) # initials pixelized beam array to store beam function values for each frequency

    #---compute theta from the pixel position    
    theta_array=np.zeros(npix)
    pixel=np.arange(npix)
    theta=(hp.pix2ang(nside,pixel)[0])#np.degrees
    #print('Amp :',Amp,'Tp :',T_p,'beam: ',beam,'smooth :', smooth)
    
    
    for p in pixel:
        if beam=='gaussian': pixelized_beam[p]=gaussian(theta[p],delta_theta)
        if beam=='jinc': pixelized_beam[p]=gaussian(theta[p],delta_theta) #qua errore, dovrebbe essere jinc
        if beam=='cosine': pixelized_beam[p]=cos_beam(theta[p],delta_theta) #ma perche in deg se il coseno si prende i rad
    
    #np.savetxt('test_beam_cosine.dat', pixelized_beam)
          
    beam_bl=hp.sphtfunc.map2alm(pixelized_beam)
    lmax=3*nside 
    return np.real(beam_bl[:lmax])

def create_bl_vec(beam,nside,dish_diameter,T_p,Amp,smooth,ch_nu):
    
    lmax=3*nside 
    nfreqs=len(ch_nu)
    bl_vec=np.zeros((nfreqs,lmax),dtype=float) #beam will be real
    delta_theta=np.zeros(nfreqs,dtype=float)
    
    for ii, nu in enumerate(ch_nu):
        
        bl_vec[ii,:]=beam2bl(beam,nside,nu,dish_diameter,T_p,Amp,smooth)
        delta_theta[ii] = createDtheta(nu,dish_diameter,T_p,Amp,smooth)*np.pi/180.
     
    return bl_vec, delta_theta
        
        

def single_convolution_bl(sky_map,beam_bl,nside):
    
    number_of_pixels=hp.nside2npix(nside)
    lmax=3*nside-1
    length_of_alms=hp.Alm.getsize(lmax)
    convolved_alms=np.zeros(length_of_alms,dtype=complex)    #array to store computed shts
    #resultant_convolved_map=np.zeros((np.shape(sky_map)[0],number_of_pixels))
    resultant_convolved_map=np.zeros(number_of_pixels)#bianca
    #compute convolution the freq channel


        
    b_bl=beam_bl
    b_bl=b_bl/(np.sqrt(4*np.pi)*b_bl[0]) #normalising the beam
    skymap_alms=hp.sphtfunc.map2alm(sky_map)
        
    counter=0  
    # Since the beam only depends on theta [symmetric around phi], we consider 
    #only the bl0s instead of the blms.
    # This counter variable enables me to multiply the correct bl0s 
    #with the sky map alms coefficients 
       
    for m in range(lmax+1):
        for l in range(m,lmax+1):           
                convolved_alms[counter]=np.sqrt((4*np.pi)/(2*float(l)+1))*b_bl[l]*skymap_alms[counter]   
                counter+=1

        resultant_convolved_map=hp.alm2map(convolved_alms,nside,inplace=False)

    return resultant_convolved_map 

 #---------------------------------------------------------------------------------------

def convolution_bl(sky_maps,beam_bl,nside):
    
    npix=hp.nside2npix(nside)
    nfreqs=np.shape(sky_maps)[0] #[freqs, pix]
    lmax=3*nside-1
    length_of_alms=hp.Alm.getsize(lmax)
    convolved_alm_array=np.zeros((nfreqs,length_of_alms),dtype=complex)    #array to store computed shts
    resultant_convolved_maps=np.zeros((nfreqs,npix),dtype=float)

    #compute convolution for each freq channel

    for nmap in range(nfreqs):

        
        b_bl=beam_bl[nmap,:]
        b_bl=b_bl/(np.sqrt(4*np.pi)*b_bl[0]) #normalising the beam
        skymap_alms=hp.sphtfunc.map2alm(sky_maps[nmap,:])
        
        counter=0  
        # Since the beam only depends on theta [symmetric around phi], we consider 
        #only the bl0s instead of the blms.
       
        for m in range(lmax+1):
            for l in range(m,lmax+1):           
                convolved_alm_array[nmap,counter]=np.sqrt((4*np.pi)/(2*float(l)+1))*b_bl[l]*skymap_alms[counter]   
                counter+=1

        resultant_convolved_maps[nmap,:]=hp.alm2map(convolved_alm_array[nmap,:],nside,inplace=False)
        print('Number of maps convolved: ',nmap)

    print('Status Update : Convolution computation complete')
    return resultant_convolved_maps

 #---------------------------------------------------------------------------------------


