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

num_ch=105
min_ch = 900.5
max_ch = 1004.5
nu_ch = np.linspace(min_ch, max_ch, num_ch)
ich=int(num_ch/2.)

nside=256
npix= hp.nside2npix(nside)
jmax=4
lmax= 3*nside-1
if fg_comp=='synch_ff_ps':
	Nfg=3
if fg_comp=='synch_ff_ps_pol':
	Nfg=18
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)

cl_cosmo_HI=np.loadtxt(out_dir_cl+f'cl_deconv_cosmo_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')
cl_PCA_HI=np.loadtxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')

#####################################################################

lmax_cl = 2*nside
ell=np.arange(lmax_cl+1)
factor=ell*(ell+1)/(2*np.pi)
lmin = 3


fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(r'$\nu$='+f'{nu_ch[ich]} MHz')
plt.plot(ell[lmin:],factor[lmin:]*cl_cosmo_HI[ich][lmin:],'k--',label='Cosmo HI + noise')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI[ich][lmin:],'+', label='PCA HI + noise')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim([lmin,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} C_{\ell} $')
frame1.set_xlabel([])
frame1.set_xticks([])
#frame1.yaxis.set_major_formatter(formatter) 

diff_cl_need2sphe = cl_PCA_HI/cl_cosmo_HI-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_need2sphe[ich][lmin:]*100,)

frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,200])
if fg_comp=='synch_ff_ps_pol':
	frame2.set_ylim([-50,50])
else:
	frame2.set_ylim([-20,20])
frame2.set_ylabel(r'%$ C_{\ell}^{\rm PCA}/C_{\ell}^{\rm cosmo} $-1')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame2.set_xticks(np.arange(lmin,lmax_cl+1, 20))

plt.savefig(f'Plots_paper/cl_std_need_ch{nu_ch[ich]}_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg}_nside{nside}_mask0.5.png')

plt.show()

fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Mean over frequency channels')
plt.plot(ell[lmin:], factor[lmin:]*cl_cosmo_HI.mean(axis=0)[lmin:],'k--',label = f'Cosmo HI + noise')
plt.plot(ell[lmin:], factor[lmin:]*cl_PCA_HI.mean(axis=0)[lmin:],'+',mfc='none', label = f'PCA HI + noise')
plt.xlim([lmin,200])
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,200+1, 10))
#frame1.yaxis.set_major_formatter(formatter) 

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_need2sphe.mean(axis=0)[lmin:]*100)
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,200])
if fg_comp=='synch_ff_ps_pol':
	frame2.set_ylim([-50,50])
else:
	frame2.set_ylim([-20,20])
frame2.set_ylabel(r'%$ \langle C_{\ell}^{\rm PCA}/C_{\ell}^{\rm cosmo} -1\rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame1.set_xticks(np.arange(lmin,200+1, 10))
#plt.tight_layout()
plt.savefig(f'Plots_paper/cl_std_need_mean_ch_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg}_nside{nside}_mask0.5.png')


plt.show()
###################################################################################################################################
###################################################################################################################################
beam_s = 'SKA_AA4'
fg_comp = 'synch_ff_ps_pol'

out_dir_maps_recon = f'maps_reconstructed/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
out_dir_cl = out_dir_maps_recon+'cls_recons_need/'

if fg_comp=='synch_ff_ps':
	Nfg=3
if fg_comp=='synch_ff_ps_pol':
	Nfg=18

cl_PCA_HI_pol=np.loadtxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')

#####################################################################
fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(r'$\nu$='+f'{nu_ch[ich]} MHz, with pol leakage')
plt.plot(ell[lmin:],factor[lmin:]*cl_cosmo_HI[ich][lmin:],'k--',label='Cosmo HI + noise')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_pol[ich][lmin:],'+', label='PCA HI + noise')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim([lmin,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} C_{\ell} $')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,200+1, 10))
#frame1.yaxis.set_major_formatter(formatter) 


diff_cl_need2sphe_pol = cl_PCA_HI_pol/cl_cosmo_HI-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_need2sphe[ich][lmin:]*100,)

frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,200])
if fg_comp=='synch_ff_ps_pol':
	frame2.set_ylim([-50,50])
else:
	frame2.set_ylim([-20,20])
frame2.set_ylabel(r'%$ C_{\ell}^{\rm PCA}/C_{\ell}^{\rm cosmo} $-1')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame1.set_xticks(np.arange(lmin,200+1, 10))
plt.savefig(f'Plots_paper/cl_std_need_ch{nu_ch[ich]}_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg}_nside{nside}_mask0.5.png')

plt.show()

fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Mean over frequency channels, with pol leakage')
plt.plot(ell[lmin:], factor[lmin:]*cl_cosmo_HI.mean(axis=0)[lmin:],'k--',label = f'Cosmo HI + noise')
plt.plot(ell[lmin:], factor[lmin:]*cl_PCA_HI_pol.mean(axis=0)[lmin:],'+',mfc='none', label = f'PCA HI + noise')
plt.xlim([lmin,200])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,200+1, 10))
#frame1.yaxis.set_major_formatter(formatter) 

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_need2sphe_pol.mean(axis=0)[lmin:]*100)
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,200])
if fg_comp=='synch_ff_ps_pol':
	frame2.set_ylim([-50,50])
else:
	frame2.set_ylim([-20,20])
frame2.set_ylabel(r'%$ \langle C_{\ell}^{\rm PCA}/C_{\ell}^{\rm cosmo} -1\rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame1.set_xticks(np.arange(lmin,200+1, 10))
plt.savefig(f'Plots_paper/cl_std_need_mean_ch_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg}_nside{nside}_mask0.5.png')

plt.show()


#################################################################################################################
###################################### costant beam #############################################################
beam_s = '1.3deg_SKA_AA4'
fg_comp = 'synch_ff_ps'

out_dir_maps_recon = f'maps_reconstructed/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
out_dir_cl = out_dir_maps_recon+'cls_recons_need/'

num_ch=105
min_ch = 900.5
max_ch = 1004.5
nu_ch = np.linspace(min_ch, max_ch, num_ch)
ich=int(num_ch/2.)

nside=256
npix= hp.nside2npix(nside)
jmax=4
lmax= 3*nside-1
if fg_comp=='synch_ff_ps':
	Nfg=3
if fg_comp=='synch_ff_ps_pol':
	Nfg=18
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)

cl_cosmo_HI_1p3deg=np.loadtxt(out_dir_cl+f'cl_deconv_cosmo_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')
cl_PCA_HI_1p3deg=np.loadtxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')

#####################################################################

lmax_cl = 2*nside
ell=np.arange(lmax_cl+1)
factor=ell*(ell+1)/(2*np.pi)
lmin = 3


fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(r'$\theta_{\rm FMWH}=1.3$ deg, $\nu$='+f'{nu_ch[ich]} MHz')
plt.plot(ell[lmin:],factor[lmin:]*cl_cosmo_HI_1p3deg[ich][lmin:],'k--',label='Cosmo HI + noise')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_1p3deg[ich][lmin:],'+', label='PCA HI + noise')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim([lmin,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} C_{\ell} $')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,200+1, 10))
#frame1.yaxis.set_major_formatter(formatter) 


diff_cl_need2sphe_1p3deg = cl_PCA_HI_1p3deg/cl_cosmo_HI_1p3deg-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_need2sphe_1p3deg[ich][lmin:]*100,)

frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,200])
if fg_comp=='synch_ff_ps_pol':
	frame2.set_ylim([-50,50])
else:
	frame2.set_ylim([-20,20])
frame2.set_ylabel(r'%$ C_{\ell}^{\rm PCA}/C_{\ell}^{\rm cosmo} $-1')
frame2.set_xlabel(r'$\ell$')
frame1.set_xticks(np.arange(lmin,200+1, 10))
#frame2.yaxis.set_major_formatter(formatter) 
plt.savefig(f'Plots_paper/cl_std_need_ch{nu_ch[ich]}_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg}_nside{nside}_mask0.5.png')

plt.show()

fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(r'$\theta_{\rm FMWH}=1.3$ deg, '+f'Mean over frequency channels')
plt.plot(ell[lmin:], factor[lmin:]*cl_cosmo_HI_1p3deg.mean(axis=0)[lmin:],'k--',label = f'Cosmo HI + noise')
plt.plot(ell[lmin:], factor[lmin:]*cl_PCA_HI_1p3deg.mean(axis=0)[lmin:],'+',mfc='none', label = f'PCA HI + noise')
plt.xlim([lmin,200])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,200+1, 10))
#frame1.yaxis.set_major_formatter(formatter) 

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_need2sphe_1p3deg.mean(axis=0)[lmin:]*100)
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,200])
if fg_comp=='synch_ff_ps_pol':
	frame2.set_ylim([-50,50])
else:
	frame2.set_ylim([-20,20])
frame2.set_ylabel(r'%$ \langle C_{\ell}^{\rm PCA}/C_{\ell}^{\rm cosmo} -1\rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame1.set_xticks(np.arange(lmin,200+1, 10))
#plt.tight_layout()
plt.savefig(f'Plots_paper/cl_std_need_mean_ch_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg}_nside{nside}_mask0.5.png')


plt.show()
###################################################################################################################################
###################################################################################################################################
beam_s = '1.3deg_SKA_AA4'
fg_comp = 'synch_ff_ps_pol'

out_dir_maps_recon = f'maps_reconstructed/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
out_dir_cl = out_dir_maps_recon+'cls_recons_need/'

if fg_comp=='synch_ff_ps':
	Nfg=3
if fg_comp=='synch_ff_ps_pol':
	Nfg=18

cl_PCA_HI_1p3deg_pol=np.loadtxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')

#####################################################################
fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(r'$\theta_{\rm FMWH}=1.3$ deg, $\nu$='+f'{nu_ch[ich]} MHz, with pol leakage')
plt.plot(ell[lmin:],factor[lmin:]*cl_cosmo_HI_1p3deg[ich][lmin:],'k--',label='Cosmo HI + noise')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_1p3deg_pol[ich][lmin:],'+', label='PCA HI + noise')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim([lmin,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} C_{\ell} $')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,200+1, 10))
#frame1.yaxis.set_major_formatter(formatter) 

diff_cl_need2sphe_1p3deg_pol = cl_PCA_HI_1p3deg_pol/cl_cosmo_HI_1p3deg-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_need2sphe_1p3deg_pol[ich][lmin:]*100,)

frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,200])
if fg_comp=='synch_ff_ps_pol':
	frame2.set_ylim([-50,50])
else:
	frame2.set_ylim([-20,20])
frame2.set_ylabel(r'%$ C_{\ell}^{\rm PCA}/C_{\ell}^{\rm cosmo} $-1')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame1.set_xticks(np.arange(lmin,200+1, 10))
plt.savefig(f'Plots_paper/cl_std_need_ch{nu_ch[ich]}_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg}_nside{nside}_mask0.5.png')

plt.show()

fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(r'$\theta_{\rm FMWH}=1.3$ deg, '+f'Mean over frequency channels, with pol leakage')
plt.plot(ell[lmin:], factor[lmin:]*cl_cosmo_HI_1p3deg.mean(axis=0)[lmin:],'k--',label = f'Cosmo HI + noise')
plt.plot(ell[lmin:], factor[lmin:]*cl_PCA_HI_1p3deg_pol.mean(axis=0)[lmin:],'+',mfc='none', label = f'PCA HI + noise')
plt.xlim([lmin,200])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,200+1, 10))
#frame1.yaxis.set_major_formatter(formatter) 

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_need2sphe_1p3deg_pol.mean(axis=0)[lmin:]*100)
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,200])
if fg_comp=='synch_ff_ps_pol':
	frame2.set_ylim([-50,50])
else:
	frame2.set_ylim([-20,20])
frame2.set_ylabel(r'%$ \langle C_{\ell}^{\rm PCA}/C_{\ell}^{\rm cosmo} -1\rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame1.set_xticks(np.arange(lmin,200+1, 10))
plt.savefig(f'Plots_paper/cl_std_need_mean_ch_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg}_nside{nside}_mask0.5.png')

plt.show()
