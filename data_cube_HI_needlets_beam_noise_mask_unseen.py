import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
from needlets_analysis import analysis, theory
import cython_mylibc as pippo
import os


fg_comp = 'synch_ff_ps_pol'
beam_s= 'theta40arcmin'
path_data_sims_tot = f'Sims/beam_{beam_s}_no_mean_sims_{fg_comp}_noise_40freq_905.0_1295.0MHz_thick10MHz_lmax383_nside128'
with open(path_data_sims_tot+'.pkl', 'rb') as f:
	file = pickle.load(f)
	f.close()

nu_ch= file['freq']

num_freq = len(nu_ch)
nu0 =1420
print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')
print(f'corresponding to the redshift range z: [{min(nu0/nu_ch -1.0):.2f} - {max(nu0/nu_ch -1.0):.2f}] ')

HI_noise_maps_freq = file['maps_sims_HI'] + file['maps_sims_noise']
fg_maps_freq = file['maps_sims_fg']
full_maps_freq = file['maps_sims_tot'] + file['maps_sims_noise']
noise_maps_freq = file['maps_sims_noise']
print(HI_noise_maps_freq.shape)

ich=int(num_freq/2)

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(full_maps_freq[ich], cmap='viridis', title=f'Observation', min=-1e3, max=1e4, hold=True)
fig.add_subplot(222) 
hp.mollview(HI_noise_maps_freq[ich], cmap='viridis', title=f'HI signal + noise',min=0, max=1,hold=True)
fig.add_subplot(223)
hp.mollview(fg_maps_freq[ich], title=f'Fg signal',cmap='viridis', min=-1e3, max=1e4, hold=True)
fig.add_subplot(224)
hp.mollview(noise_maps_freq[ich], title=f'Noise',cmap='viridis', hold=True)
plt.show()

del file

npix = np.shape(HI_noise_maps_freq)[1]
nside = hp.get_nside(HI_noise_maps_freq[0])
lmax=3*nside-1#2*nside#
jmax=4

	
######################################################################################

mask1_40 = hp.read_map('HFI_Mask_GalPlane_2048_R1.10.fits', field=1)#fsky 40 % sky coverage
mask_40t = hp.ud_grade(mask1_40, nside_out=256)
mask_40 = hp.ud_grade(mask_40t, nside_out=nside)
#del mask1_40
mask_40s = hp.sphtfunc.smoothing(mask_40, 3*np.pi/180,lmax=lmax) #aposization 3 deg come in Olivari
#del mask_40
fsky  = np.mean(mask_40s) 

fig=plt.figure()
hp.mollview(mask_40s, cmap='viridis', title=f'Mask, fsky={np.mean(mask_40s):0.2f}', hold=True)
plt.show()

bad_v = np.where(mask_40==0)

for n in range(num_freq):
		HI_noise_maps_freq[n][bad_v] =  hp.UNSEEN
		HI_noise_maps_freq[n]=hp.remove_dipole(HI_noise_maps_freq[n])

		fg_maps_freq[n][bad_v] =  hp.UNSEEN
		fg_maps_freq[n]=hp.remove_dipole(fg_maps_freq[n])

		full_maps_freq[n][bad_v] =  hp.UNSEEN
		full_maps_freq[n]=hp.remove_dipole(full_maps_freq[n])
		
####################################################################################

ich=int(num_freq/2)
print(ich)
fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(full_maps_freq[ich], cmap='viridis', title=f'Observation', min=-1e3, max=1e3, hold=True)
fig.add_subplot(222) 
hp.mollview(HI_noise_maps_freq[ich], cmap='viridis', title=f'HI signal + noise',min=0, max=1,hold=True)
fig.add_subplot(223)
hp.mollview(fg_maps_freq[ich], title=f'Fg signal',cmap='viridis', min=-1e3, max=1e3, hold=True)
fig.add_subplot(224)
hp.mollview(full_maps_freq[ich]-fg_maps_freq[ich], title=f'Observation - Fg',cmap='viridis',min=0, max=1, hold=True)
plt.show()



jmax=4
out_dir = f'./Maps_needlets/No_mean/Beam_{beam_s}_noise_mask{fsky:0.2}_unseen/'
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

need_analysis = analysis.NeedAnalysis(jmax, lmax, out_dir, full_maps_freq)
need_analysis_HI = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_noise_maps_freq)
need_analysis_fg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_maps_freq)

B=need_analysis.B

fname_obs_tot=f'bjk_maps_obs_noise_{fg_comp}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_jmax{jmax}_lmax{lmax}_B{B:0.2f}_nside{nside}'
fname_HI=f'bjk_maps_HI_noise_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_jmax{jmax}_lmax{lmax}_B{B:0.2f}_nside{nside}'
fname_fg=f'bjk_maps_fg_{fg_comp}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_jmax{jmax}_lmax{lmax}_B{B:0.2f}_nside{nside}'


#try:
#        betajk_sims_tot = np.load(out_dir + fname_obs_tot+'.npy')
#        print("...simulations beta_jk's " + out_dir + fname_obs_tot + " found...")
#
#        betajk_sims_tot = np.load(out_dir + fname_obs_tot+'.npy')
#        print("...simulations beta_jk's " + out_dir + fname_obs_tot + " found...")
#
#        betajk_sims_tot = np.load(out_dir + fname_obs_tot+'.npy')
#        print("...simulations beta_jk's " + out_dir + fname_obs_tot + " found...")
#except:
#        print("...simulations beta_jk's " + out_dir + fname_obs_tot + " not found...")
#        print("...evaluating...")



j_test=2#7
need_theory=theory.NeedletTheory(B,jmax, lmax)
np.savetxt(out_dir+f'b_values_std_jmax{jmax}_lmax{lmax}_B{B:1.2f}.dat',need_theory.b_values)

fig, ax1  = plt.subplots(1,1,figsize=(7,5)) 
plt.suptitle(r'STANDARD $D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))

for i in range(need_theory.b_values.shape[0]):
    ax1.plot(need_theory.b_values[i]*need_theory.b_values[i], label = 'j='+str(i) )
ax1.set_xscale('log')
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
ax1.legend(loc='right')
#plt.tight_layout()
#plt.savefig(f'PCA_needlets_output/windows_function_jmax{jmax}_lmax{lmax}')


ell_binning=need_theory.ell_binning(lmax)
fig = plt.figure()
plt.suptitle(r'$D = %1.2f $' %B +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))
ax = fig.add_subplot(1, 1, 1)
for i in range(0,jmax+1):
    ell_range = ell_binning[i][ell_binning[i]!=0]
    plt.plot(ell_range, i*ell_range/ell_range, label= f'j={i}')
    plt.text(ell_range[0], i, r'$\ell_{min}=%d,\,\ell_{max}=%d$'%(ell_range[0],ell_range[-1]))

ax.set_xlabel(r'$\ell$')
ax.legend(loc='right', ncol=2)
#plt.tight_layout()
plt.show()

map_need_output = np.zeros((num_freq, jmax+1, npix))
map_HI_need_output = np.zeros((num_freq, jmax+1, npix))
map_fg_need_output = np.zeros((num_freq, jmax+1, npix))

for nu in range(num_freq):
	map_need_output[nu] = pippo.mylibpy_needlets_f2betajk_healpix_harmonic(full_maps_freq[nu], B, jmax,lmax )
np.save(out_dir+fname_obs_tot,map_need_output)

fig = plt.figure(figsize=(10, 7))
hp.mollview(map_need_output[ich,j_test], cmap='viridis', title=f'Observation, j={j_test}, freq={nu_ch[ich]}', hold=True)
del map_need_output; del full_maps_freq; del need_analysis

for nu in range(num_freq):        
	map_HI_need_output[nu] = pippo.mylibpy_needlets_f2betajk_healpix_harmonic(HI_noise_maps_freq[nu], B, jmax,lmax )
np.save(out_dir+fname_HI,map_HI_need_output)

fig = plt.figure(figsize=(10, 7))
hp.mollview(map_HI_need_output[ich,j_test], cmap='viridis', title=f'HI, j={j_test}, freq={nu_ch[ich]}', hold=True)

del map_HI_need_output; del HI_noise_maps_freq; del need_analysis_HI

for nu in range(num_freq):        
	map_fg_need_output[nu] = pippo.mylibpy_needlets_f2betajk_healpix_harmonic(fg_maps_freq[nu], B, jmax,lmax )
np.save(out_dir+fname_fg,map_fg_need_output)
fig = plt.figure(figsize=(10, 7))
hp.mollview(map_fg_need_output[ich,j_test], cmap='viridis', title=f'Fg, j={j_test}, freq={nu_ch[ich]}', hold=True)

del map_fg_need_output; del fg_maps_freq; del need_analysis_fg

plt.show()