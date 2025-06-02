import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import pymaster as nm
import os
import pickle

import seaborn as sns
sns.set_theme(style = 'white')
from matplotlib import colors
import matplotlib as mpl
mpl.rc('xtick', direction='in', top=False, bottom = True)
mpl.rc('ytick', direction='in', right=False, left = True)

#print(sns.color_palette("husl", 15).as_hex())
sns.palettes.color_palette()
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



from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
#############################################################################

beam_s = 'SKA_AA4'
fg_comp = 'synch_ff_ps'

out_dir_maps_recon = f'maps_reconstructed/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
out_dir_cl = out_dir_maps_recon+'cls_recons_need/'


out_dir_maps_recon_std = f'../PCA_needlets_output/maps_reconstructed/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
out_dir_cl_std = out_dir_maps_recon_std+'cls_recons_need/'


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

nside=256
npix= hp.nside2npix(nside)
jmax_std=4
jmax_mex=12
lmax= 3*nside-1
if fg_comp=='synch_ff_ps':
	Nfg=3
if fg_comp=='synch_ff_ps_pol':
	Nfg=18


cl_cosmo_HI=np.loadtxt(out_dir_cl_std+f'cl_deconv_cosmo_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax_std}_lmax{2*nside}_nside{nside}.dat')

cl_PCA_HI_std=np.loadtxt(out_dir_cl_std+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax_std}_lmax{2*nside}_nside{nside}.dat')

cl_PCA_HI_mex=np.loadtxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax_mex}_lmax{2*nside}_nside{nside}.dat')

#####################################################################################################################

lmax_cl = 2*nside
ell=np.arange(lmax_cl+1)
factor=ell*(ell+1)/(2*np.pi)
lmin = 3

lmax_plot= lmax_theta_worst
print(lmax_plot)

fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(r'$\nu$='+f'{nu_ch[ich]} MHz')
plt.plot(ell[lmin:],factor[lmin:]*cl_cosmo_HI[ich][lmin:],'k--',label='Cosmo HI + noise')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_std[ich][lmin:],  label='Standard Need-PCA HI + noise')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_mex[ich][lmin:],  label='Mexican Need-PCA HI + noise')

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim([lmin,lmax_plot])
plt.legend()
frame1.set_ylabel(r'$  \ell(\ell+1)/2\pi~ C_{\ell} $ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])
#frame1.yaxis.set_major_formatter(formatter) 

diff_cl_std = cl_PCA_HI_std/cl_cosmo_HI-1
diff_cl_mex = cl_PCA_HI_mex/cl_cosmo_HI-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_std[ich][lmin:]*100,label='Standard Needlets')
plt.plot(ell[lmin:], diff_cl_mex[ich][lmin:]*100,label='Mexican Needlets')

frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,lmax_plot])
if fg_comp=='synch_ff_ps_pol':
	frame2.set_ylim([-50,50])
else:
	frame2.set_ylim([-50,50])
frame2.set_ylabel(r'%$ C_{\ell}^{\rm PCA}/C_{\ell}^{\rm cosmo} $-1')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))
plt.legend()
plt.savefig(f'../PCA_needlets_output/Plots_paper/comparison_cl_std_mex_need_ch{nu_ch[ich]}_{fg_comp}_noise_beam_{beam_s}_jmax_std{jmax_std}_jmax_mex{jmax_mex}_Nfg{Nfg}_nside{nside}_mask0.5.png')


#plt.show()
####################


fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Mean over frequency channels')
plt.plot(ell[lmin:],factor[lmin:]*cl_cosmo_HI.mean(axis=0)[lmin:],'k--',label='Cosmo HI + noise')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_std.mean(axis=0)[lmin:],  label='Standard Need-PCA HI + noise')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_mex.mean(axis=0)[lmin:],  label='Mexican Need-PCA HI + noise')

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim([lmin,lmax_plot])
plt.legend()
frame1.set_ylabel(r'$  \ell(\ell+1)/2\pi~ \langle C_{\ell} \rangle_{\rm ch}$ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])
#frame1.yaxis.set_major_formatter(formatter) 

diff_cl_std = cl_PCA_HI_std/cl_cosmo_HI-1
diff_cl_mex = cl_PCA_HI_mex/cl_cosmo_HI-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_std.mean(axis=0)[lmin:]*100,label='Standard')
plt.plot(ell[lmin:], diff_cl_mex.mean(axis=0)[lmin:]*100,label='Mexican')

frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,lmax_plot])
if fg_comp=='synch_ff_ps_pol':
	frame2.set_ylim([-50,50])
else:
	frame2.set_ylim([-50,50])
frame2.set_ylabel(r'%$ \langle C_{\ell}^{\rm PCA}/C_{\ell}^{\rm cosmo} -1\rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))
plt.legend()
plt.savefig(f'../PCA_needlets_output/Plots_paper/comparison_cl_std_mex_need_mean_ch_{fg_comp}_noise_beam_{beam_s}_jmax_std{jmax_std}_jmax_mex{jmax_mex}_Nfg{Nfg}_nside{nside}_mask0.5.png')

fig = plt.figure()
plt.plot(ell[lmin:], (diff_cl_mex/diff_cl_std-1).mean(axis=0)[lmin:])
plt.axhline(ls='--', c= 'k', alpha=0.3)
plt.xlim([lmin,lmax_plot])
plt.ylim([-0.5,0.5])
plt.ylabel('% diff of the diff')
plt.xlabel(r'$\ell$')
plt.show()
