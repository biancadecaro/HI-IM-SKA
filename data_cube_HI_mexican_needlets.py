import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
from needlets_analysis import analysis, theory
import cython_mylibc as pippo
import os

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
        print(jmax)
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

def ell_binning(b_values,lmax):#, ell):
        """
        Returns the binning scheme in  multipole space
        """
        bjl  = b_values**2
        ell  = np.arange(lmax+1)*np.ones((bjl.shape[0], lmax+1))
        ellj =np.zeros((bjl.shape[0], lmax+1))

        for j in range(bjl.shape[0]):
            bjl[j,:][bjl[j,:]!=0] = 1.
            ellj[j,:] = ell[j,:]*bjl[j,:]
        return ellj 

######################################################################################################

fg_components='synch_ff_ps'
path_data_sims_tot = f'Sims/beam_theta40arcmin_no_mean_sims_{fg_components}_40freq_905.0_1295.0MHz_lmax768_nside256'

with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

nu_ch= file['freq']

num_freq = len(nu_ch)
nu0 =1420
print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')
print(f'corresponding to the redshift range z: [{min(nu0/nu_ch -1.0):.2f} - {max(nu0/nu_ch -1.0):.2f}] ')

HI_maps_freq = file['maps_sims_HI']
fg_maps_freq = file['maps_sims_fg']
full_maps_freq = file['maps_sims_tot']
print(HI_maps_freq.shape)


ich=int(num_freq/2)
print(ich)
fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(full_maps_freq[ich], cmap='viridis', title=f'Observation', min=-1e3, max=1e4, hold=True)
fig.add_subplot(222) 
hp.mollview(HI_maps_freq[ich], cmap='viridis', title=f'HI signal',min=0, max=1,hold=True)
fig.add_subplot(223)
hp.mollview(fg_maps_freq[ich], title=f'Fg signal',cmap='viridis', min=-1e3, max=1e4, hold=True)
fig.add_subplot(224)
hp.mollview(full_maps_freq[ich]-fg_maps_freq[ich], title=f'Observation - Fg',cmap='viridis',min=0, max=1, hold=True)
plt.show()

del file

npix = np.shape(HI_maps_freq)[1]
nside = hp.get_nside(HI_maps_freq[0])
lmax=3*nside#2*nside#
jmax=12
out_dir = './Maps_needlets_mexican/No_mean/p1/Beam_40arcmin/'
if not os.path.exists(out_dir):
        os.makedirs(out_dir)

#for nu in range(num_freq):
#        alm_HI = hp.map2alm(HI_maps_freq[nu], lmax=lmax)
#        HI_maps_freq[nu] = hp.alm2map(alm_HI, lmax=lmax, nside = nside)
#        del alm_HI
#        alm_fg = hp.map2alm(fg_maps_freq[nu], lmax=lmax)
#        fg_maps_freq[nu] = hp.alm2map(alm_fg, lmax=lmax, nside = nside)
#        del alm_fg
#        alm_obs = hp.map2alm(full_maps_freq[nu], lmax=lmax)
#        full_maps_freq[nu] = hp.alm2map(alm_obs, lmax=lmax, nside = nside)
#        del alm_obs


need_analysis = analysis.NeedAnalysis(jmax, lmax, out_dir, full_maps_freq)
need_analysis_HI = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_maps_freq)
need_analysis_fg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_maps_freq)

B=need_analysis.B

fname_obs_tot=f'bjk_maps_obs_synch_ff_ps_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_jmax{jmax}_lmax{lmax}_B{B:0.2f}_nside{nside}'
fname_HI=f'bjk_maps_HI_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_jmax{jmax}_lmax{lmax}_B{B:0.2f}_nside{nside}'
fname_fg=f'bjk_maps_fg_synch_ff_ps_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_jmax{jmax}_lmax{lmax}_B{B:0.2f}_nside{nside}'


j_test=3#7

b_values = mexicanneedlet(B,np.arange(0,jmax+1),lmax, p=1)

fig, ax1  = plt.subplots(1,1,figsize=(7,5)) 
plt.suptitle(r'MEXICAN $D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))


for i in range(b_values.shape[0]):
    ax1.plot(b_values[i], label = 'j='+str(i) )
ax1.set_xscale('log')
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
ax1.legend(loc='right')
plt.tight_layout()
plt.savefig(f'PCA_mexican_needlets_output/windows_function_mex_p1_jmax{jmax}_lmax{lmax}')

ell_binning=ell_binning(b_values,lmax)
fig = plt.figure()
plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))
ax = fig.add_subplot(1, 1, 1)
for i in range(0,jmax+1):
    ell_range = ell_binning[i][ell_binning[i]!=0]
    plt.plot(ell_range, i*ell_range/ell_range, label= f'j={i}')
    plt.text(ell_range[0], i, r'$\ell_{min}=%d,\,\ell_{max}=%d$'%(ell_range[0],ell_range[-1]))

ax.set_xlabel(r'$\ell$')
ax.legend(loc='right', ncol=2)
plt.tight_layout()
plt.show()

map_need_output = np.zeros((num_freq, jmax+1, npix))
map_HI_need_output = np.zeros((num_freq, jmax+1, npix))
map_fg_need_output = np.zeros((num_freq, jmax+1, npix))

for nu in range(num_freq):
        for j in range(jmax+1):
                map_need_output[nu,j,:] = hp.alm2map(hp.almxfl(hp.map2alm(full_maps_freq[nu],lmax=lmax),b_values[j,:]),lmax=lmax,nside=nside) 
np.save(out_dir+fname_obs_tot,map_need_output)

fig = plt.figure(figsize=(10, 7))
hp.mollview(map_need_output[ich,j_test], cmap='viridis', title=f'Observation, j={j_test}, freq={nu_ch[ich]}', hold=True)
del map_need_output; del full_maps_freq; del need_analysis

for nu in range(num_freq):        
        for j in range(jmax+1):
                map_HI_need_output[nu,j,:] = hp.alm2map(hp.almxfl(hp.map2alm(HI_maps_freq[nu],lmax=lmax),b_values[j,:]),lmax=lmax,nside=nside) 
np.save(out_dir+fname_HI,map_HI_need_output)

fig = plt.figure(figsize=(10, 7))
hp.mollview(map_HI_need_output[ich,j_test], cmap='viridis', title=f'HI, j={j_test}, freq={nu_ch[ich]}', hold=True)

del map_HI_need_output; del HI_maps_freq; del need_analysis_HI

for nu in range(num_freq):        
        for j in range(jmax+1):
                map_fg_need_output[nu,j,:] = hp.alm2map(hp.almxfl(hp.map2alm(fg_maps_freq[nu],lmax=lmax),b_values[j,:]),lmax=lmax,nside=nside)
np.save(out_dir+fname_fg,map_fg_need_output)

fig = plt.figure(figsize=(10, 7))
hp.mollview(map_fg_need_output[ich,j_test], cmap='viridis', title=f'Fg, j={j_test}, freq={nu_ch[ich]}', hold=True)

del map_fg_need_output; del fg_maps_freq; del need_analysis_fg

plt.show()