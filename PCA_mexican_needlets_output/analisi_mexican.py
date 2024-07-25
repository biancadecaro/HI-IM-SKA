import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cython_mylibc as pippo
import os

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 

######################################################################################
def mexicanneedlet(B,j,lmax,p=1,normalised=True):
    '''Return the needlet filter b(l) for a Mexican needlet with parameters ``B`` and ``j``.
    
    Parameters
    ----------
    B : float
        The parameter B of the needlet, should be larger that 1.
    j : int or np.ndarray
        The frequency j of the needlet. Can be an array with the values of ``j`` to be calculated.
    lmax : int
        The maximum value of the multipole l for which the filter will be calculated (included).
    p : int
        Order of the Mexican needlet.
    normalised : bool, optional
        If ``True``, the sum (in frequencies ``j``) of the squares of the filters will be 1 for all multipole ell.

    Returns
    -------
    np.ndarray
        A numpy array containing the values of a needlet filter b(l), starting with l=0. If ``j`` is
        an array, it returns an array containing the filters for each frequency.
    '''
    ls=np.arange(lmax+1)
    j=np.array(j,ndmin=1)
    needs=[]
    if normalised != True:
        for jj in j:
#            u=(ls/B**jj)
#            bl=u**(2.*p)*np.exp(-u**2.)
            u=((ls*(ls+1)/B**(2.*jj)))
            bl=(u**p)*np.exp(-u)
            needs.append(bl)
    else:
        K=np.zeros(lmax+1)
        #jmax=np.max((np.log(5.*lmax)/np.log(B),np.max(j)))
        jmax = np.max(j)+1
        for jj in np.arange(0,jmax+1):#np.arange(1,jmax+1)
#            u=(ls/B**jj)                   This is an almost identical definition
#            bl=u**2.*np.exp(-u**2.)
            u=((ls*(ls+1)/B**(2.*jj)))
            bl=(u**p)*np.exp(-u)
            K=K+bl**2.
            if np.isin(jj,j):
                needs.append(bl)
        needs=needs/np.sqrt(np.mean(K[int(lmax/3):int(2*lmax/3)]))
    return(np.squeeze(needs))
#########################################################################################

nside = 256
lmax=3*nside
jmax=12
p=1
B=pippo.mylibpy_jmax_lmax2B(jmax, lmax)
print(f'B={B:1.2f}')
jvec = np.arange(jmax+1)

b_values_p1 = mexicanneedlet(B,jvec,lmax,p=1,normalised=True)
b_values_p2 = mexicanneedlet(B,jvec,lmax,p=2,normalised=True)
b_values_p3 = mexicanneedlet(B,jvec,lmax,p=3,normalised=True)
b_values_p4 = mexicanneedlet(B,jvec,lmax,p=4,normalised=True)

npix=hp.nside2npix(nside)

m=np.zeros(npix)

pix = hp.ang2pix(nside=nside,theta=np.pi / 2, phi=0)

m[pix]=1

betajk_p1 = np.zeros((jmax+1, npix))
betajk_p2 = np.zeros((jmax+1, npix))
betajk_p3 = np.zeros((jmax+1, npix))
betajk_p4 = np.zeros((jmax+1, npix))

for j in jvec:
    betajk_p1[j] = hp.alm2map(hp.almxfl(hp.map2alm(m,lmax=lmax),b_values_p1[j,:]),lmax=lmax,nside=nside) 
    betajk_p2[j] = hp.alm2map(hp.almxfl(hp.map2alm(m,lmax=lmax),b_values_p3[j,:]),lmax=lmax,nside=nside) 
    betajk_p3[j] = hp.alm2map(hp.almxfl(hp.map2alm(m,lmax=lmax),b_values_p3[j,:]),lmax=lmax,nside=nside) 
    betajk_p4[j] = hp.alm2map(hp.almxfl(hp.map2alm(m,lmax=lmax),b_values_p4[j,:]),lmax=lmax,nside=nside) 


j_plot1=2
j_plot2=4
j_plot3=6
j_plot4=10
fig=plt.figure()
fig.suptitle(f'B={B:1.2f}')
fig.add_subplot(221)
hp.mollview(betajk_p1[j_plot3], title=f'j={jvec[j_plot3]},p=1',min=-1e-3,max=1e-3, cmap='viridis', hold=True)
fig.add_subplot(222)
hp.mollview(betajk_p2[j_plot3], title=f'j={jvec[j_plot3]},p=2',min=-1e-3,max=1e-3, cmap='viridis', hold=True)
fig.add_subplot(223)
hp.mollview(betajk_p3[j_plot3], title=f'j={jvec[j_plot3]},p=3',min=-1e-3,max=1e-3, cmap='viridis', hold=True)
fig.add_subplot(224)
hp.mollview(betajk_p4[j_plot3], title=f'j={jvec[j_plot3]},p=4',min=-1e-3,max=1e-3, cmap='viridis', hold=True)

plt.show()




theta_c=np.pi / 2
phi_c=0
prova= hp.query_strip(nside = nside, theta1=theta_c, theta2 = theta_c+100*np.sqrt(hp.nside2pixarea(nside))/2, inclusive=True)

ipix = np.arange(0,1000)

fig=plt.figure()
fig.subplots_adjust(wspace=0.4)
fig.subplots_adjust(hspace=0.6)
fig.suptitle(f'B={B:1.2f}')

ax1 = fig.add_subplot(221)
ax1.set_title(f'j={j_plot1}')
ax1.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_p1[j_plot1][prova][0:1000], label='p=1')
ax1.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_p2[j_plot1][prova][0:1000], label='p=2')
ax1.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_p3[j_plot1][prova][0:1000], label='p=3')
ax1.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_p4[j_plot1][prova][0:1000], label='p=4')
ax1.set_xlim([0., 100])
ax1.set_xlabel(r'$\theta$[deg]')
ax1.set_ylabel(r'$\beta_{jk}$')
ax1.yaxis.set_major_formatter(formatter) 
ax1.legend()

ax2 = fig.add_subplot(222)
ax2.set_title(f'j={j_plot3}')
ax2.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_p1[j_plot3][prova][0:1000], label='p=1')
ax2.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_p2[j_plot3][prova][0:1000], label='p=2')
ax2.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_p3[j_plot3][prova][0:1000], label='p=3')
ax2.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_p4[j_plot3][prova][0:1000], label='p=4')
ax2.set_xlim([0., 20])
ax2.set_xlabel(r'$\theta$[deg]')
ax2.set_ylabel(r'$\beta_{jk}$')
ax2.yaxis.set_major_formatter(formatter) 
ax2.legend()

ax3 = fig.add_subplot(223)
ax3.set_title(f'j={j_plot4}')
ax3.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_p1[j_plot4][prova][0:1000], label='p=1')
ax3.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_p2[j_plot4][prova][0:1000], label='p=2')
ax3.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_p3[j_plot4][prova][0:1000], label='p=3')
ax3.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_p4[j_plot4][prova][0:1000], label='p=4')
ax3.set_xlim([0., 2.])
ax3.set_xlabel(r'$\theta$[deg]')
ax3.set_ylabel(r'$\beta_{jk}$')
ax3.yaxis.set_major_formatter(formatter) 
ax3.legend()

#plt.savefig(f'mex_need_B{B:1.2f}_theta_loc_p1_p4.png')

fig=plt.figure()
fig.subplots_adjust(wspace=0.4)
fig.subplots_adjust(hspace=0.6)
fig.suptitle(f'B={B:1.2f}')

ax1 = fig.add_subplot(221)
ax1.set_title(f'j={j_plot1}')
ax1.plot(b_values_p1[j_plot1], label='p=1')
ax1.plot(b_values_p2[j_plot1], label='p=2')
ax1.plot(b_values_p3[j_plot1], label='p=3')
ax1.plot(b_values_p4[j_plot1], label='p=4')
ax1.set_xscale('log')
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$b_{\ell}$')
ax1.yaxis.set_major_formatter(formatter) 
ax1.legend()

ax2 = fig.add_subplot(222)
ax2.set_title(f'j={j_plot3}')
ax2.plot(b_values_p1[j_plot3], label='p=1')
ax2.plot(b_values_p2[j_plot3], label='p=2')
ax2.plot(b_values_p3[j_plot3], label='p=3')
ax2.plot(b_values_p4[j_plot3], label='p=4')
ax2.set_xscale('log')
ax2.set_xlabel(r'$\ell$')
ax2.set_ylabel(r'$b_{\ell}$')
ax2.yaxis.set_major_formatter(formatter) 
ax2.legend()

ax3 = fig.add_subplot(223)
ax3.set_title(f'j={j_plot4}')
ax3.plot(b_values_p1[j_plot4], label='p=1')
ax3.plot(b_values_p2[j_plot4], label='p=2')
ax3.plot(b_values_p3[j_plot4], label='p=3')
ax3.plot(b_values_p4[j_plot4], label='p=4')
ax3.set_xscale('log')
ax3.set_xlabel(r'$\ell$')
ax3.set_ylabel(r'$b_{\ell}$')
ax3.yaxis.set_major_formatter(formatter) 
ax3.legend()

plt.show()