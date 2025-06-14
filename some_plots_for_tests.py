import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import string

import seaborn as sns
sns.set_theme(style = 'white')
from matplotlib import colors
import matplotlib as mpl
mpl.rc('xtick', direction='in', top=False, bottom = True)
mpl.rc('ytick', direction='in', right=False, left = True)

#print(sns.color_palette("husl", 15).as_hex())
sns.color_palette()
import cython_mylibc as pippo

plt.rcParams['figure.figsize']=(11,7)
plt.rcParams['axes.titlesize']=20
plt.rcParams['lines.linewidth']  = 3.
plt.rcParams['lines.markersize']=6
plt.rcParams['axes.labelsize']  =20
plt.rcParams['legend.fontsize']=20
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.formatter.use_mathtext']=True
plt.rcParams['savefig.dpi']=300
plt.rcParams['legend.columnspacing']=0.5
plt.rcParams['legend.title_fontsize'] =20


from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 

c_pal = sns.color_palette().as_hex()
#############################################################################

beam_s = 'SKA_AA4'
fg_comp = 'synch_ff_ps'

out_dir_maps_recon = f'PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise/'
out_dir_cl = out_dir_maps_recon+'power_spectra_cls_from_healpix_maps/'


num_ch=105
min_ch = 900.5
max_ch = 1004.5
nu_ch = np.linspace(min_ch, max_ch, num_ch)
ich=int(num_ch/2.)

##############################
c_light = 3.0*1e8
dish_diam_MeerKat = 13.5 #m
dish_diam_SKA = 15 # m
Ndishes_MeerKAT = 64.
Ndishes_SKA = 133.
dish_diam = (Ndishes_MeerKAT*dish_diam_MeerKat+ Ndishes_SKA*dish_diam_SKA)/(Ndishes_MeerKAT+Ndishes_SKA) 
theta_FWMH_max = c_light*1e-6/np.min(nu_ch)/float(dish_diam) #radians
lmax_theta_worst = int(np.pi/theta_FWMH_max)
###############################
###############################

nside=128
npix= hp.nside2npix(nside)
jmax=4
lmax= 3*nside-1

Nfg3,Nfg6,Nfg18=3,6,18


###############################################
########################## cl #################################
lmax_cl = 2*nside
cl_cosmo_HI=np.loadtxt(out_dir_cl+f'cl_input_HI_noise_105_900.5_1004.5MHz_lmax{lmax_cl}_nside{nside}.dat')

cl_PCA_HI = np.loadtxt(out_dir_cl+f'cl_PCA_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg{Nfg3}_lmax{lmax_cl}_nside{nside}.dat')

cl_PCA_HI_pol_Nfg3 = np.loadtxt(out_dir_cl+f'cl_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg{Nfg3}_lmax{lmax_cl}_nside{nside}.dat')
cl_PCA_HI_pol_Nfg6 = np.loadtxt(out_dir_cl+f'cl_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg{Nfg6}_lmax{lmax_cl}_nside{nside}.dat')
cl_PCA_HI_pol_Nfg18 = np.loadtxt(out_dir_cl+f'cl_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg{Nfg18}_lmax{lmax_cl}_nside{nside}.dat')


#######################################################################

ell=np.arange(lmax_cl+1)
factor=ell*(ell+1)/(2*np.pi)
lmin = 3

lmax_plot= lmax_theta_worst
print(lmax_plot)

diff_cl = cl_PCA_HI/cl_cosmo_HI-1
diff_cl_pol_Nfg3 = cl_PCA_HI_pol_Nfg3/cl_cosmo_HI-1
diff_cl_pol_Nfg6 = cl_PCA_HI_pol_Nfg6/cl_cosmo_HI-1
diff_cl_pol_Nfg18 = cl_PCA_HI_pol_Nfg18/cl_cosmo_HI-1
###################################################################

fig, ax = plt.subplots(1,1)
plt.title(f'Mean over frequency channels, full sky')

ax.plot(ell[lmin:], np.abs(diff_cl.mean(axis=0))[lmin:]*100, label='w/o pol leak')
ax.plot(ell[lmin:], np.abs(diff_cl_pol_Nfg3.mean(axis=0))[lmin:]*100, label='with pol leak, Nfg=3')
ax.plot(ell[lmin:], np.abs(diff_cl_pol_Nfg6.mean(axis=0))[lmin:]*100, label='with pol leak, Nfg=6')
ax.plot(ell[lmin:], np.abs(diff_cl_pol_Nfg18.mean(axis=0))[lmin:]*100, label='with pol leak, Nfg=18')
ax.axhline(ls='--', c= 'k', alpha=0.3)
ax.set_xlim([lmin, lmax_plot+1])
ax.set_ylim([-1, 90])
ax.set_ylabel(r'|$\langle \Delta \rangle $|[%]')
ax.set_xlabel(r'$\ell$')
ax.set_xticks(np.arange(lmin,lmax_plot+1, 10))
plt.legend()


######################################################################
###################3 leakage #################################3
cl_HI_leak = np.loadtxt(out_dir_cl+f'cl_leak_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg{Nfg3}_lmax{lmax_cl}_nside{nside}.dat')
cl_fg_leak = np.loadtxt(out_dir_cl+f'cl_leak_fg_synch_ff_ps_105_900.5_1004.5MHz_Nfg{Nfg3}_lmax{lmax_cl}_nside{nside}.dat')

cl_HI_leak_pol_Nfg3 = np.loadtxt(out_dir_cl+f'cl_leak_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg{Nfg3}_lmax{lmax_cl}_nside{nside}.dat')
cl_fg_leak_pol_Nfg3 = np.loadtxt(out_dir_cl+f'cl_leak_fg_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg{Nfg3}_lmax{lmax_cl}_nside{nside}.dat')

cl_HI_leak_pol_Nfg6 = np.loadtxt(out_dir_cl+f'cl_leak_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg{Nfg6}_lmax{lmax_cl}_nside{nside}.dat')
cl_fg_leak_pol_Nfg6 = np.loadtxt(out_dir_cl+f'cl_leak_fg_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg{Nfg6}_lmax{lmax_cl}_nside{nside}.dat')

cl_HI_leak_pol_Nfg18 = np.loadtxt(out_dir_cl+f'cl_leak_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg{Nfg18}_lmax{lmax_cl}_nside{nside}.dat')
cl_fg_leak_pol_Nfg18 = np.loadtxt(out_dir_cl+f'cl_leak_fg_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg{Nfg18}_lmax{lmax_cl}_nside{nside}.dat')

fig, ax= plt.subplots()
ax.set_title('Foreground and HI leakage, mean over channels, full sky')

ax.plot(ell[lmin:], factor[lmin:]*cl_HI_leak.mean(axis=0)[lmin:], c=c_pal[0],ls='-',label= 'HI leakage ')
ax.plot(ell[lmin:], factor[lmin:]*cl_HI_leak_pol_Nfg3.mean(axis=0)[lmin:], c=c_pal[0],ls='--', label= 'pol, Nfg=3 ')
ax.plot(ell[lmin:], factor[lmin:]*cl_HI_leak_pol_Nfg6.mean(axis=0)[lmin:], c=c_pal[0],ls=':', label= 'pol, Nfg=6 ')
ax.plot(ell[lmin:], factor[lmin:]*cl_HI_leak_pol_Nfg18.mean(axis=0)[lmin:], c=c_pal[0],ls='-.', label= 'pol, Nfg=18 ')

ax.plot(ell[lmin:], factor[lmin:]*cl_fg_leak.mean(axis=0)[lmin:], c=c_pal[1],ls='-',label= 'fg leakage ')
ax.plot(ell[lmin:], factor[lmin:]*cl_fg_leak_pol_Nfg3.mean(axis=0)[lmin:], c=c_pal[1],ls='--', label= 'pol, Nfg=3 ')
ax.plot(ell[lmin:], factor[lmin:]*cl_fg_leak_pol_Nfg6.mean(axis=0)[lmin:], c=c_pal[1],ls=':', label= 'pol, Nfg=6 ')
ax.plot(ell[lmin:], factor[lmin:]*cl_fg_leak_pol_Nfg18.mean(axis=0)[lmin:], c=c_pal[1],ls='-.', label= 'pol, Nfg=18 ')
ax.set_yscale('log')

ax.set_ylabel(r'$  \ell(\ell+1)/2\pi~ \langle C_{\ell} \rangle_{\rm ch}$ [mK$^{2}$]')
ax.set_xlabel(r'$\ell$')

fig.legend(ncols=2,loc='outside center right',bbox_to_anchor=(1, 0.80))

plt.show()