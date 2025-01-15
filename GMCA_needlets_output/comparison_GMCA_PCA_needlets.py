import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import scipy.linalg as lng
sns.set_theme(style = 'white')
#sns.set_palette('husl',15)

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

c_pal = sns.color_palette().as_hex()
###########################################################################
dir_PCA= '../PCA_needlets_output/maps_reconstructed/No_mean/Beam_40arcmin/'
dir_GMCA = 'maps_reconstructed/No_mean/Beam_40arcmin/'

dir_PCA_cl = dir_PCA+'cls_recons_need/'
dir_GMCA_cl = dir_GMCA+'cls_recons_need/'

fg_components='synch_ff_ps'
path_data_sims_tot = f'../Sims/beam_theta40arcmin_no_mean_sims_{fg_components}_40freq_905.0_1295.0MHz_lmax768_nside256'

with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

nu_ch= file['freq']
del file
num_freq = len(nu_ch)
min_ch = min(nu_ch)
max_ch = max(nu_ch)

nu0 =1420
print(f'working with {len(nu_ch)} channels, from {min(nu_ch)} to {max(nu_ch)} MHz')
print(f'i.e. channels are {nu_ch[1]-nu_ch[0]} MHz thick')
print(f'corresponding to the redshift range z: [{min(nu0/nu_ch -1.0):.2f} - {max(nu0/nu_ch -1.0):.2f}] ')
#########################################################################################################
nside =256
lmax=3*nside
lmax_cl=2*nside
Nfg=3
jmax = 4
#####################################################################################################

cl_cosmo_HI = np.loadtxt('../PCA_pixels_output/Maps_PCA/No_mean/Beam_40arcmin/power_spectra_cls_from_healpix_maps/cl_input_HI_lmax512_nside256.dat')

cl_PCA_HI = np.loadtxt(dir_PCA_cl+f'cl_PCA_HI_Nfg3_jmax{jmax}_lmax512_nside256.dat')
cl_GMCA_HI = np.loadtxt(dir_GMCA_cl+f'cl_GMCA_HI_Nfg3_jmax{jmax}_lmax512_nside256.dat')

ich = int(num_freq/2)

################################# PLOT ############################################################

ell = np.arange(0, lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)

diff_PCA = cl_PCA_HI/cl_cosmo_HI -1
diff_GMCA = cl_GMCA_HI/cl_cosmo_HI-1


fig=plt.figure()
fig.suptitle(f'mean over channels, Std Need,\nbeam 40 arcmin, jmax:{jmax}, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.plot(ell[2:], 100*np.mean(diff_PCA, axis=0)[2:],'--',mfc='none', label = 'PCA')
plt.plot(ell[2:], 100*np.mean(diff_GMCA, axis=0)[2:],'--',mfc='none', label = 'GMCA')
plt.xlim([0,250])
plt.ylim([-20,20])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\% \langle C_{\ell}^{\rm rec}/C_{\ell}^{\rm cosmo}-1 \rangle $')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(f'rel_diff_mean_cl_HI_PCA_GMCA_std_need_jmax{jmax}_lmax{lmax_cl}_Nfg{Nfg}.png')
print('% rel diff PCA:',min(100*diff_PCA[ich]), max(100*diff_PCA[ich]))
print('% rel diff GMCA:',min(100*diff_GMCA[ich]), max(100*diff_GMCA[ich]))


##################### diff tra i due ###################################

fig=plt.figure()
fig.suptitle(f'mean over channels, Std Need,\nbeam 40 arcmin, jmax:{jmax}, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.scatter(ell[2:], 100*np.mean(diff_PCA/diff_GMCA-1, axis=0)[2:])
plt.xlim([0,250])
plt.ylim([-20,20])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\% \langle diff \rangle $')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.tight_layout()



plt.show()



#########################################################
#################    radial clustering   ################
#########################################################

## HOW to use these functions:
# # # find the lines of sight over which compute the radial P(k)
# # indexes_los = np.where(mask==1.0)[0]

# ## which lines of sight
# # indexes_los = np.arange(0,hp.nside2npix(NSIDE),10)
# indexes_los = np.arange(hp.nside2npix(NSIDE))

## field_array should be nu X pixels 
## equally spaced array
import scipy 

def clustering_nu(field_array,indexes_los,nu_ch,verbose=False):
	
	## sanity check
	if verbose:
		print('sanity check: ')
		print('  ',(len(field_array[:,0])==len(nu_ch)),' ',(len(field_array[0,:])>=len(indexes_los)))
		print('  ',(len(nu_ch) % 2) == 0)


	## cropping the array
	T_field = field_array[:,indexes_los]
	del field_array

	## how many LOS are we considering?
	nlos = len(indexes_los)
	if verbose: print(f'using {nlos} LoS')
	del indexes_los

	## defines cells 
	dims = len(nu_ch); dnu  = abs(nu_ch[-1]-nu_ch[-2])
	if verbose: print(f'each divided into {dims} cells of {dnu} MHz')

	## remove mean from maps
	if verbose: print('removing mean from maps . .')
	mean_T_mapwise = np.mean(T_field,axis=1)
	T_field_nm =  np.array([T_field[i,:] - mean_T_mapwise[i] for i in range(dims)])
	del T_field
	if verbose: print('defining DeltaT array . .')
	deltaT = np.array([T_field_nm[:,ipix]  for ipix in range(nlos)])
	# print('i.e. deltaT --> ',deltaT.shape)
	del T_field_nm

	if verbose: print('\nFFT the overdensity temperature field along LoS')
	delta_k = scipy.fftpack.fftn(deltaT,overwrite_x=True,axes=1)
	delta_k *= dnu;  del deltaT

	delta_k_auto  = np.absolute(delta_k)**2  

	if verbose: print('done!\n')
	return dims, dnu, delta_k_auto

def doing_Pk1D(dims,dnu,delta_k_auto):

    # compute the values of k of the modes for the 1D P(k)
    modes   = np.arange(dims,dtype=np.float64);  
    middle = int(dims/2)
    indexes = np.where(modes>middle)[0];
    modes[indexes] = modes[indexes]-dims
    k = modes*(2.0*np.pi/(dnu*dims)) # k in MHz-1
    k = np.absolute(k)               # just take the modulus
    del indexes, modes

    # define the k-bins
    k_bins = np.linspace(0,middle,middle+1)*(2.0*np.pi/(dnu*dims))

    # compute the number of modes and the average number-weighted value of k
    k_modes = np.histogram(k,bins=k_bins)[0]
    k_bin   = np.histogram(k,bins=k_bins,weights=k)[0]/k_modes

    # take all LoS and compute the average value for each mode
    delta_k2_stacked = np.mean(delta_k_auto,dtype=np.float64,axis=0)

    # compute the 1D P(k)
    Pk_mean = np.histogram(k,bins=k_bins,weights=delta_k2_stacked)[0]
    Pk_mean = Pk_mean/(dnu*dims*k_modes);  del delta_k2_stacked

    Pk_1D = np.transpose([k_bin[1:],Pk_mean[1:]])
    
    return Pk_1D


## to plot the frequency power spectrum
## returns knu and P for x and y axis
def plot_nuPk(fmap,indexes_los,nu_ch,verbose=False):

	Pk_1D = doing_Pk1D(*clustering_nu(fmap,indexes_los,nu_ch))
	if verbose:
		print("k_nu [MHz^-1] vs P [mK^2 MHz]")

	return Pk_1D[:,0],Pk_1D[:,1]

################################################################################################

indexes_los = np.arange(hp.nside2npix(nside))

path_cosmo_HI = f'../GMCA_pixels_output/Maps_GMCA/No_mean/Beam_40arcmin/cosmo_HI_{num_freq}_{min_ch:1.1f}_{max_ch:1.1f}MHz'
path_GMCA_HI=dir_GMCA+f'maps_reconstructed_GMCA_HI_{num_freq}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}'
path_PCA_HI=dir_PCA+f'maps_reconstructed_PCA_HI_{num_freq}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}'

map_cosmo_HI = np.load(path_cosmo_HI+'.npy')
map_HI_GMCA = np.load(path_GMCA_HI+'.npy')
map_HI_PCA = np.load(path_PCA_HI+'.npy')

k_cosmo, P_cosmo = plot_nuPk(map_cosmo_HI,indexes_los,nu_ch)
k_GMCA, P_GMCA = plot_nuPk(map_HI_GMCA,indexes_los,nu_ch)
k_PCA, P_PCA = plot_nuPk(map_HI_PCA,indexes_los,nu_ch)

fig=plt.figure()
plt.semilogx(k_cosmo, P_cosmo, label='Cosmo')
plt.semilogx(k_GMCA, P_GMCA, label='GMCA')
plt.semilogx(k_PCA, P_PCA, label='PCA')
plt.legend()
ax = plt.gca()
ax.set(xlabel="$k_{\\nu}$ [MHz$^{-1}$]",ylabel="$P$ [mK$^2$ MHz$^2$]")

diff_Pk_GMCA = P_GMCA/P_cosmo-1
diff_Pk_PCA = P_PCA/P_cosmo-1


ich = 10
fig=plt.figure()
fig.suptitle(f' Std Need,\nbeam 40 arcmin, jmax:{jmax}, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.semilogx(k_GMCA, diff_Pk_GMCA,'--',mfc='none', label='GMCA')
plt.semilogx(k_PCA, diff_Pk_PCA,'--',mfc='none', label='PCA')
print(k_PCA)
plt.ylim([-1,1])
plt.xlim([7e-3,4e-1])
plt.xlabel('$k_{\\nu}$ [MHz$^{-1}$]')
plt.ylabel(r'$P_{\rm res}/P_{\rm cosmo}$-1 ')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.legend()
plt.tight_layout()

print((diff_Pk_GMCA/diff_Pk_PCA-1)*100)


##################### diff tra i due ###################################

fig=plt.figure()
fig.suptitle(f'Std Need,\nbeam 40 arcmin, jmax:{jmax}, lmax:{lmax_cl}, Nfg:{Nfg}')
plt.semilogx(k_GMCA,(diff_Pk_GMCA/diff_Pk_PCA-1)*100)
#plt.xlim([0,250])
plt.ylim([-1,1])
plt.xlim([4e-3,4e1-1])
plt.xlabel('$k_{\\nu}$ [MHz$^{-1}$]')
plt.ylabel('% diff PCA-GMCA')
plt.axhline(y=0,c='k',ls='--',alpha=0.5)
plt.tight_layout()


plt.show()
