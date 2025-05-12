import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/home/bianca/Documents/gmca4im-master/scripts/')
import gmca4im_lib2 as g4i
import scipy.linalg as lng
import os
import numpy.ma as ma

import seaborn as sns
sns.set_theme(style = 'white')
sns.color_palette(n_colors=15)
#sns.set_palette('husl',15)

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

#print(sns.color_palette("husl", 15).as_hex())
sns.palettes.color_palette()
###########################################################################3
fg_comp='synch_ff_ps_pol'
beam_s = '1.3deg_SKA_AA4'
path_data_sims_tot = f'Sims/beam_{beam_s}_no_mean_sims_{fg_comp}_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax767_nside256'
with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

out_dir_output = 'GMCA_needlets_output/'
out_dir_output_GMCA = out_dir_output+f'GMCA_maps/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
out_dir_plot = out_dir_output+f'Plots_GMCA_needlets/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
if not os.path.exists(out_dir_output):
        os.makedirs(out_dir_output)
if not os.path.exists(out_dir_output_GMCA):
        os.makedirs(out_dir_output_GMCA)

nu_ch= file['freq']
del file

need_dir = f'Maps_needlets/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
need_tot_maps_filename = need_dir+f'bjk_maps_obs_noise_{fg_comp}_105freq_900.5_1004.5MHz_jmax4_lmax767_B5.26_nside256.npy'
need_tot_maps = np.load(need_tot_maps_filename)

jmax=need_tot_maps.shape[1]-1

num_freq = need_tot_maps.shape[0]
nu_ch = np.linspace(900.5, 1004.5, num_freq)
min_ch = min(nu_ch)
max_ch = max(nu_ch)
npix = need_tot_maps.shape[2]
nside = hp.npix2nside(npix)
lmax=3*nside-1#2*nside#
B=pow(lmax,(1./jmax))

print(f'jmax:{jmax}, lmax:{lmax}, B:{B:1.2f}, num_freq:{num_freq}, min_ch:{min_ch}, max_ch:{max_ch}, nside:{nside}')

######################################################################################

pix_mask = hp.query_strip(nside, theta1=np.pi*2/3, theta2=np.pi/3)
print(pix_mask)
mask_50 = np.zeros(npix)
mask_50[pix_mask] =1
fsky_50 = np.sum(mask_50)/hp.nside2npix(nside)

fig=plt.figure()
hp.mollview(mask_50, cmap='viridis', title=f'fsky={np.mean(mask_50):0.2f}', hold=True)
#plt.savefig(f'Plots_sims/mask_apo3deg_fsky{np.mean(mask_40s):0.2f}_nside{nside}.png')
plt.show()

#############################################################################
############################# GMCA ##########################################

################   GMCA PARAMETERS   ##################
if fg_comp=='synch_ff_ps':
    num_sources=3
if fg_comp=='synch_ff_ps_pol':
    num_sources=18
print(f'Nfg={num_sources}')  # number of sources to be estimated
mints = 0.1 # min threshold (what is sparse compared to noise?)
nmax  = 100 # number of iterations (usually 100 is safe)
L0    = 0   # switch between L0 norm (1) or L1 norm (0)
#############################################################################
bad_v = np.where(mask_50==0)


maskt =np.zeros(mask_50.shape)
maskt[bad_v]=  1
mask = ma.make_mask(maskt, shrink=False)

need_tot_maps_masked=ma.zeros(need_tot_maps.shape)

for n in range(num_freq):
    for jj in range(jmax+1):
        need_tot_maps_masked[n,jj]  =ma.MaskedArray(need_tot_maps[n,jj], mask=mask)#np.isnan(full_maps_freq_mask[n])


hp.mollview(need_tot_maps_masked[0,3], cmap='viridis', title='masked ma')
plt.show()
##############################################################################

# initial guess for the mixing matrix?
# i.e. we could start from GMCA-determined mix matrix
AInit = None

# we can impose a column of the mixing matrix
ColFixed = None

# we can whiten the data
whitening = False; epsi = 1e-3

# estimated mixing matrix:
Ae = np.zeros((jmax+1, num_freq,num_sources))
for j in range(Ae.shape[0]): 
    Ae[j] = g4i.ma_run_GMCA(need_tot_maps_masked[:,j,:],AInit,num_sources,mints,nmax,L0,ColFixed,whitening,epsi)
    fig=plt.figure()
    plt.suptitle(f'j={j}')
    plt.imshow(Ae[j],cmap='crest')
    plt.colorbar()
plt.show()

np.save(out_dir_output_GMCA+f'Ae_mixing_matrix_{num_freq}_Nfg{num_sources}_jmax{jmax}_lmax{lmax}_nside{nside}', Ae)
#Ae = np.load(out_dir_output_GMCA+f'Ae_mixing_matrix_{num_freq}_Nfg{num_sources}_jmax{jmax}_lmax{lmax}_nside{nside}.npy')
#############################################################################
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
for j in [1,2,3]:
    plt.plot(abs(Ae[j]/np.linalg.norm(Ae[j],axis=0)),label=f'mix mat column,j={j}')
plt.plot(FF_col/np.linalg.norm(FF_col),'m:',label='gal ff')

ax = plt.gca()
ax.set(ylim=[0.0,0.4],xlabel="frequency channel",ylabel="Spectral emission",title='GMCA-mixing matrix columns')
plt.legend(ncols=2)
plt.show()

####################################################################################################
res_fg_maps = np.zeros((Ae.shape[0], num_freq, npix))
for j in range(Ae.shape[0]):
    piA = np.ma.dot(np.linalg.inv(np.ma.dot(Ae[j].T,Ae[j])),Ae[j].T)
    Se_sph = np.ma.dot(piA,need_tot_maps[:,j,:]) # LS estimate of the sources in the pixel domain
    res_fg_maps[j] = np.ma.dot(Ae[j],Se_sph)
    #Ae[j]@np.linalg.inv(Ae[j].T@Ae[j])@Ae[j].T@need_tot_maps[:,j,:]
print(res_fg_maps.shape)



#np.save(out_dir_output_GMCA+f'res_GMCA_fg_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{min_ch}_{max_ch}MHz_Nfg{num_sources}_nside{nside}.npy',res_fg_maps)

print('.. ho calcolato res fg .. ')

ich= int(num_freq/2)
j_test=3

res_HI_maps = np.zeros((Ae.shape[0], num_freq, npix))
for j in range(Ae.shape[0]):
    res_HI_maps[j,:,:] = need_tot_maps[:,j,:] - res_fg_maps[j,:,:]
    res_HI_maps[j,:,bad_v]=hp.UNSEEN
    hp.mollview(res_HI_maps[j][ich],min=0, max=0.2, cmap='viridis', title=f'j={j}')
plt.show()
del need_tot_maps


np.save(out_dir_output_GMCA+f'res_GMCA_HI_noise_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{min_ch}_{max_ch}MHz_Nfg{num_sources}_nside{nside}.npy',res_HI_maps)

print('.. ho calcolato res HI .. ')

res_fg_maps_totj=res_fg_maps.sum(axis=0)
res_HI_maps_totj = res_HI_maps.sum(axis=0)
res_fg_maps_totj[:, bad_v] = hp.UNSEEN
res_HI_maps_totj[:, bad_v] = hp.UNSEEN
del res_HI_maps; del res_fg_maps


print('fin qui ci sono')

############### leakage ###########################
#Foreground's maps

need_fg_maps_filename = need_dir+f'bjk_maps_fg_{fg_comp}_{num_freq}freq_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}.npy'
need_fg_maps = np.load(need_fg_maps_filename)#[:,:jmax,:]

leak_fg_maps = np.zeros((Ae.shape[0], num_freq, npix))
leak_HI_maps = np.zeros((Ae.shape[0], num_freq, npix))

for j in range(Ae.shape[0]):
    piA = np.ma.dot(np.linalg.inv(np.ma.dot(Ae[j].T,Ae[j])),Ae[j].T)
    Se_sph = np.ma.dot(piA,need_fg_maps[:,j,:]) # LS estimate of the sources in the pixel domain
    leak_fg_maps[j] = need_fg_maps[:,j,:]-np.ma.dot(Ae[j],Se_sph)
    leak_fg_maps[j,:,bad_v]=hp.UNSEEN


np.save(out_dir_output_GMCA+f'leak_GMCA_fg_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{min_ch}_{max_ch}MHz_Nfg{num_sources}_nside{nside}.npy',leak_fg_maps)

fig = plt.figure()
plt.suptitle(f'Frequency channel: {nu_ch[ich]} MHz, Nfg:{num_sources}, jmax:{jmax}, lmax:{lmax} ')
hp.mollview(leak_fg_maps.sum(axis=0)[ich], title='Foregroud leakage', cmap = 'viridis', min=0, max=0.5, hold=True)
#plt.savefig(out_dir_plot+f'betajk_leak_fg_jmax{jmax}_lmax{lmax}_nside_{nside}_Nfg{num_sources}.png')
plt.show()
del leak_fg_maps


need_HI_maps_filename = need_dir+f'bjk_maps_HI_noise_{num_freq}freq_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}.npy'
need_HI_maps = np.load(need_HI_maps_filename)#[:,:jmax,:]


for j in range(Ae.shape[0]):
    piA = np.ma.dot(np.linalg.inv(np.ma.dot(Ae[j].T,Ae[j])),Ae[j].T)
    Se_sph = np.ma.dot(piA,need_HI_maps[:,j,:]) # LS estimate of the sources in the pixel domain
    leak_HI_maps[j] = np.ma.dot(Ae[j],Se_sph)

del Ae; 
np.save(out_dir_output_GMCA+f'leak_GMCA_HI_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{min_ch}_{max_ch}MHz_Nfg{num_sources}_nside{nside}.npy',leak_HI_maps)

fig = plt.figure()
plt.suptitle(f'Frequency channel: {nu_ch[ich]} MHz, Nfg:{num_sources}, jmax:{jmax}, lmax:{lmax} ')
hp.mollview(leak_HI_maps.sum(axis=0)[ich], title='HI leakage', cmap = 'viridis', min=0, max=0.5, hold=True)
#plt.savefig(out_dir_plot+f'betajk_leak_HI_jmax{jmax}_{lmax}_nside_{nside}_Nfg{num_sources}.png')
plt.show()

del leak_HI_maps#; del leak_fg_maps


##############################################


need_HI_maps_totj = need_HI_maps.sum(axis=1)
need_fg_maps_totj = need_fg_maps.sum(axis=1)
need_HI_maps_totj[:, bad_v] = hp.UNSEEN
need_fg_maps_totj[:,bad_v] = hp.UNSEEN

del need_HI_maps;del need_fg_maps

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}',fontsize=20)
fig.add_subplot(222) 
hp.mollview(need_HI_maps_totj[ich], cmap='viridis', title=f'HI signal + noise',min=0, max =1,hold=True)
fig.add_subplot(223)
hp.mollview(res_HI_maps_totj[ich], title=f'GMCA HI + noise',min=0, max =1,cmap='viridis', hold=True)
fig.add_subplot(221)
hp.mollview(np.abs(res_fg_maps_totj[ich]/need_fg_maps_totj[ich]-1)*100, min=0, max=0.1,cmap='viridis', title=f'%(Res fg/Fg)-1',unit='%' ,hold=True)
#plt.savefig(out_dir_plot+f'betajk_res_need_GMCA_sumj_Nfg{num_sources}_jmax{jmax}_lmax{lmax}_nside{nside}.png')
plt.show()


###################### PROVIAMO CON I CL #########################

lmax_cl = 2*nside

cl_cosmo_HI = np.zeros((len(nu_ch), lmax_cl+1))
cl_GMCA_HI_need2harm = np.zeros((len(nu_ch), lmax_cl+1))

for n in range(len(nu_ch)):
    cl_cosmo_HI[n]=hp.anafast(need_HI_maps_totj[n], lmax=lmax_cl)
    cl_GMCA_HI_need2harm[n] = hp.anafast(res_HI_maps_totj[n], lmax=lmax_cl)

del need_HI_maps_totj; del res_HI_maps_totj

ell=np.arange(lmax_cl+1)
factor=ell*(ell+1)/(2*np.pi)

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: channel:{nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{num_sources}')
plt.plot(ell[2:],factor[2:]*cl_cosmo_HI[ich][2:], label='Cosmo HI + noise')
plt.plot(ell[2:],factor[2:]*cl_GMCA_HI_need2harm[ich][2:],'+', mfc='none', label='GMCA HI + noise')
plt.ylabel(r'$\frac{\ell(\ell+1)}{2\pi}  C_{\ell} $')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} e C_{\ell} $')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 30))

diff_cl_need2sphe = cl_GMCA_HI_need2harm/cl_cosmo_HI-1
frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe[ich][2:]*100, label='% GMCA_HI/input_HI -1')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-10,10])
frame2.set_ylabel(r'%$  diff$')
frame2.set_xlabel(r'$\ell$')
frame2.set_xticks(np.arange(1,200+1, 30))
#plt.tight_layout()
plt.legend()
#plt.savefig(out_dir_plot+f'cls_need2pix_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.png')
plt.show()

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: mean over channels, jmax:{jmax}, lmax:{lmax}, Nfg:{num_sources}')
plt.plot(ell[2:], factor[2:]*cl_cosmo_HI.mean(axis=0)[2:], label = f'Cosmo HI + noise')
plt.plot(ell[2:], factor[2:]*cl_GMCA_HI_need2harm.mean(axis=0)[2:],'+',mfc='none', label = f'GMCA HI + noise')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 30))

del cl_GMCA_HI_need2harm; del cl_cosmo_HI
frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe.mean(axis=0)[2:]*100, label='% GMCA_HI/input_HI -1')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-10,10])
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
frame2.set_xticks(np.arange(1,200+1, 30))
#plt.tight_layout()
plt.legend()
#plt.savefig(out_dir_plot+f'cls_need2pix_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.png')

plt.show()

del diff_cl_need2sphe
