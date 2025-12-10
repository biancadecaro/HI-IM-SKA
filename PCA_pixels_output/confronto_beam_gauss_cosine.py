import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy.ma as ma
import copy
import pymaster as nm
sns.set_theme(style = 'white')
c_pal = sns.color_palette().as_hex()
linestyle = ['solid','dotted', 'dashed','dashdot']  


import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)


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
plt.rcParams["figure.titlesize"] = 18


from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 

###########################################################################
beam_gauss = 'SKA_AA4'
beam_gauss_const = '1.3deg_SKA_AA4'

beam_cosine = 'cosine_Amp0.1_smooth_True_SKA_AA4'
beam_cosine_const = '1.55deg_cosine_Amp0.1_smooth_True_SKA_AA4'

min_ch = 900.5
max_ch = 1004.5
nu_ch = np.arange(min_ch, max_ch)
num_ch = len(nu_ch)

ich = int(num_ch/2)


nside = 128


###########################################################################

map_beam_gauss = np.load(f'Maps_PCA_nuovo/No_mean/Beam_{beam_gauss}_noise_mask0.5_unseen/res_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg3_lmax383_nside128.npy', allow_pickle=True)
map_beam_cosine = np.load(f'Maps_PCA_nuovo/No_mean/Beam_{beam_cosine}_noise_mask0.5_unseen/res_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg3_lmax383_nside128.npy', allow_pickle=True)


fig=plt.figure(figsize=(10, 7))
fig.add_subplot(121) 
hp.gnomview(map_beam_gauss[ich],rot=[-25,61], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='Gaussian beam', cmap='viridis', hold=True)
fig.add_subplot(122) 
hp.gnomview(map_beam_cosine[ich],rot=[-25,61], coord='G', reso=hp.nside2resol(nside, arcmin=True), min=0, max=1, title='Cosine beam', cmap= 'viridis', hold=True)

del map_beam_cosine; del map_beam_gauss
######################################################################

cl_beam_gauss = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_gauss}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')
cl_cosmo_beam_gauss = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_gauss}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_cosmo_HI_noise_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')

cl_beam_cosine_3 = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_cosine}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')
cl_beam_cosine_6 = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_cosine}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg6_lmax256_nside128.dat')
cl_beam_cosine_18 = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_cosine}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg18_lmax256_nside128.dat')
cl_cosmo_beam_cosine = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_cosine}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_cosmo_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')


lmax_cl = 2*nside
ell = np.arange(0, lmax_cl+1)
factor = ell*(ell+1)/(2*np.pi)
lmax_plot = 136
lmin=3


fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Channel:{nu_ch[ich]} MHz, lmax:{3*nside-1}, fsky:0.5, with pol leakage')
plt.semilogy(ell[2:], factor[2:]*cl_beam_gauss[ich][2:],color=c_pal[0],ls=linestyle[0],mfc='none', label='Gaussian beam')
plt.semilogy(ell[2:], factor[2:]*cl_beam_cosine_3[ich][2:],color=c_pal[1],ls=linestyle[1],mfc='none', label='Cosine beam Nfg 3')
plt.semilogy(ell[2:], factor[2:]*cl_beam_cosine_6[ich][2:],color=c_pal[2],ls=linestyle[2],mfc='none', label='Cosine beam Nfg 6')
plt.semilogy(ell[2:], factor[2:]*cl_beam_cosine_18[ich][2:],color=c_pal[3],ls=linestyle[3],mfc='none', label='Cosine beam Nfg 18')
plt.xlim([lmin,lmax_plot])
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(ncols=2, columnspacing=0.5)
frame1.set_ylabel(r'$\ell(\ell+1)/2\pi~ C_{\ell} $ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])


diff_cl_beam_gass = cl_beam_gauss/cl_cosmo_beam_gauss -1 
diff_cl_beam_cosine_3 = cl_beam_cosine_3/cl_cosmo_beam_cosine -1 
diff_cl_beam_cosine_6 = cl_beam_cosine_6/cl_cosmo_beam_cosine -1 
diff_cl_beam_cosine_18 = cl_beam_cosine_18/cl_cosmo_beam_cosine -1 

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_beam_gass[ich][2:]*100, color=c_pal[0], ls=linestyle[0],label='Gaussian beam')
plt.plot(ell[2:], diff_cl_beam_cosine_3[ich][2:]*100, color=c_pal[1], ls=linestyle[1],label='Cosine beam Nfg 3')
plt.plot(ell[2:], diff_cl_beam_cosine_6[ich][2:]*100, color=c_pal[2], ls=linestyle[2],label='Cosine beam Nfg 6')
plt.plot(ell[2:], diff_cl_beam_cosine_18[ich][2:]*100, color=c_pal[3], ls=linestyle[3],label='Cosine beam Nfg 18')

frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,lmax_plot])
frame2.set_ylim([-50,50])
frame2.set_ylabel(r'$\Delta$ [%]')
frame2.set_xlabel(r'$\ell$')
frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))

#########################

fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Mean over channels, lmax:{3*nside-1}, fsky:0.5, with pol leakage')
plt.semilogy(ell[2:], factor[2:]*cl_beam_gauss.mean(axis=0)[2:],color=c_pal[0],ls=linestyle[0],mfc='none', label='Gaussian beam')
plt.semilogy(ell[2:], factor[2:]*cl_beam_cosine_3.mean(axis=0)[2:],color=c_pal[1],ls=linestyle[1],mfc='none', label='Cosine beam Nfg 3')
plt.semilogy(ell[2:], factor[2:]*cl_beam_cosine_6.mean(axis=0)[2:],color=c_pal[2],ls=linestyle[2],mfc='none', label='Cosine beam Nfg 6')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylim([-5e-4,6e-3])
plt.xlim([lmin,lmax_plot])
plt.legend(ncols=2, columnspacing=0.5)
frame1.set_ylabel(r'$\ell(\ell+1)/2\pi~ C_{\ell} $ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])


diff_cl_beam_gass = cl_beam_gauss/cl_cosmo_beam_gauss -1 
diff_cl_beam_cosine_3 = cl_beam_cosine_3/cl_cosmo_beam_cosine -1 
diff_cl_beam_cosine_6 = cl_beam_cosine_6/cl_cosmo_beam_cosine -1 

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_beam_gass.mean(axis=0)[2:]*100, color=c_pal[0], ls=linestyle[0],label='Gaussian beam')
plt.plot(ell[2:], diff_cl_beam_cosine_3.mean(axis=0)[2:]*100, color=c_pal[1], ls=linestyle[1],label='Cosine beam Nfg 3')
plt.plot(ell[2:], diff_cl_beam_cosine_6.mean(axis=0)[2:]*100, color=c_pal[2], ls=linestyle[2],label='Cosine beam Nfg 6')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,lmax_plot])
frame2.set_ylim([-30,2])
frame2.set_ylabel(r'$\Delta$ [%]')
frame2.set_xlabel(r'$\ell$')

frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))

plt.close('all')
############################################################################################

cl_beam_gauss = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_gauss}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')
cl_cosmo_beam_gauss = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_gauss}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_cosmo_HI_noise_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')
diff_cl_beam_guass=cl_beam_gauss/cl_cosmo_beam_gauss -1 

cl_beam_const_gauss = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_gauss_const}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')
cl_cosmo_beam_const_gauss = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_gauss_const}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_cosmo_HI_noise_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')
diff_cl_beam_const_gass = cl_beam_const_gauss/cl_cosmo_beam_const_gauss -1 

cl_beam_deconv_gauss = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_gauss}_noise_mask0.5_unseen_deconv/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')
cl_cosmo_beam_deconv_gauss = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_gauss}_noise_mask0.5_unseen_deconv/power_spectra_cls_from_healpix_maps/cl_deconv_cosmo_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')
diff_cl_beam_deconv_gass = cl_beam_deconv_gauss/cl_cosmo_beam_deconv_gauss -1 



fig=plt.figure()
plt.suptitle('PCA, mean over channels, no pol leak, Nfg=3')
plt.plot(ell[2:], diff_cl_beam_guass.mean(axis=0)[2:]*100, color=c_pal[0], ls=linestyle[0],label='Gaussian beam')
plt.plot(ell[2:], diff_cl_beam_const_gass.mean(axis=0)[2:]*100, color=c_pal[1], ls=linestyle[0],label='Costant Gaussian beam')
plt.plot(ell[2:], diff_cl_beam_deconv_gass.mean(axis=0)[2:]*100, color=c_pal[2], ls=linestyle[0],label='Deconv Gaussian beam')
plt.axhline(ls='--', c= 'k', alpha=0.3)
plt.xlim([lmin,lmax_plot])
plt.ylim([-30,2])
plt.ylabel(r'$ \langle \Delta\rangle_{\rm ch}$ [%]')
plt.xlabel(r'$\ell$')
plt.legend()

############################################################################################

cl_beam_cos = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_cosine}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')
cl_beam_cos_nfg4 = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_cosine_Amp0.1_smooth_True_SKA_AA4_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg4_lmax256_nside128.dat')
cl_cosmo_beam_cos = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_cosine_Amp0.1_smooth_True_SKA_AA4_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_cosmo_HI_noise_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')

diff_cl_beam_cos=cl_beam_cos/cl_cosmo_beam_cos -1 
diff_cl_beam_cos_Nfg4=cl_beam_cos_nfg4/cl_cosmo_beam_cos -1 

cl_beam_const_cos = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_cosine_const}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')
cl_beam_const_cos_nfg4 = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_cosine_const}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg4_lmax256_nside128.dat')
cl_cosmo_beam_const_cos = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_cosine_const}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_cosmo_HI_noise_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')

diff_cl_beam_const_cos = cl_beam_const_cos/cl_cosmo_beam_const_cos -1 
diff_cl_beam_const_cos_nfg4 = cl_beam_const_cos_nfg4/cl_cosmo_beam_const_cos -1 

cl_beam_deconv_cos = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_cosine}_noise_mask0.5_unseen_deconv/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')
cl_beam_deconv_cos_nfg4 = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_cosine}_noise_mask0.5_unseen_deconv/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg4_lmax256_nside128.dat')
cl_cosmo_beam_deconv_cos = np.loadtxt(f'Maps_PCA_nuovo/No_mean/Beam_{beam_cosine}_noise_mask0.5_unseen_deconv/power_spectra_cls_from_healpix_maps/cl_deconv_cosmo_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')

diff_cl_beam_deconv_cos = cl_beam_deconv_cos/cl_cosmo_beam_deconv_cos -1 
diff_cl_beam_deconv_cos_Nfg4 = cl_beam_deconv_cos_nfg4/cl_cosmo_beam_deconv_cos -1 



fig = plt.figure()
plt.suptitle('Input HI')
plt.plot(ell, factor*np.mean(cl_cosmo_beam_cos, axis=0), ls=linestyle[0], c=c_pal[0],label='Cosine')
plt.plot(ell, factor*np.mean(cl_cosmo_beam_const_cos, axis=0), ls=linestyle[1], c=c_pal[0],label='Cosine const')
plt.plot(ell, factor*np.mean(cl_cosmo_beam_deconv_cos, axis=0), ls=linestyle[2], c=c_pal[0],label='Cosine deconv')
plt.plot(ell, factor*np.mean(cl_cosmo_beam_gauss, axis=0), ls=linestyle[0], c=c_pal[1],label='Gauss')
plt.plot(ell, factor*np.mean(cl_cosmo_beam_const_gauss, axis=0), ls=linestyle[1], c=c_pal[1],label='Gauss const')
plt.plot(ell, factor*np.mean(cl_cosmo_beam_deconv_gauss, axis=0), ls=linestyle[2], c=c_pal[1],label='Gauss deconv')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylim([-0.001,0.008])
plt.xlim([0,200 +1 ])
plt.ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle $')
plt.xlabel(r'$\ell$')
plt.legend()



fig=plt.figure()
plt.suptitle('PCA, mean over channels, no pol leak, Nfg=4', fontsize=18)
plt.plot(ell[10:], diff_cl_beam_cos_Nfg4.mean(axis=0)[10:], color=c_pal[0], ls=linestyle[0],label='Cosine beam no deconv')
plt.plot(ell[10:], diff_cl_beam_deconv_cos_Nfg4.mean(axis=0)[10:], color=c_pal[1], ls=linestyle[0],label='Cosine beam deconv')
plt.xlim([15,200])
plt.ylim([-0.2,0.2])
plt.axhline(ls='--', color='grey')
plt.ylabel(r'$ \langle \Delta\rangle_{\rm ch}$ [%]')
plt.xlabel(r'$\ell$')
plt.legend()



fig=plt.figure()
plt.suptitle('PCA, mean over channels, no pol leak, Nfg=3', fontsize=18)
plt.plot(ell[2:], diff_cl_beam_cos.mean(axis=0)[2:]*100, color=c_pal[0], ls=linestyle[0],label='Cosine beam')
plt.plot(ell[2:], diff_cl_beam_const_cos.mean(axis=0)[2:]*100, color=c_pal[1], ls=linestyle[0],label='Costant Cosine beam')
plt.plot(ell[2:], diff_cl_beam_deconv_cos.mean(axis=0)[2:]*100, color=c_pal[2], ls=linestyle[0],label='Deconv Cosine beam')
plt.axhline(ls='--', c= 'k', alpha=0.3)
plt.xlim([lmin,200])
plt.ylim([-30,2])
plt.ylabel(r'$ \langle \Delta\rangle_{\rm ch}$ [%]')
plt.xlabel(r'$\ell$')
plt.legend()




fig, (ax1,ax2,ax3)=plt.subplots(1,3, sharey=True) 
plt.suptitle('PCA, mean over channels, no pol leak,')

ax1.set_title('Gaussian, Nfg=3')
ax1.plot(ell[2:], diff_cl_beam_guass.mean(axis=0)[2:]*100, color=c_pal[0], ls=linestyle[0],label='Gaussian beam')
ax1.plot(ell[2:], diff_cl_beam_const_gass.mean(axis=0)[2:]*100, color=c_pal[1], ls=linestyle[0],label='Costant Gaussian beam')
ax1.plot(ell[2:], diff_cl_beam_deconv_gass.mean(axis=0)[2:]*100, color=c_pal[2], ls=linestyle[0],label='Deconv Gaussian beam')
ax1.axhline(ls='--', c= 'k', alpha=0.3)
ax1.set_xlim([lmin,lmax_plot])
ax1.set_ylim([-30,2])
ax1.set_ylabel(r'$ \langle \Delta\rangle_{\rm ch}$ [%]')
ax1.set_xlabel(r'$\ell$')
#ax1.legend()

ax2.set_title('Cosine, Nfg=3')
ax2.plot(ell[2:], diff_cl_beam_cos.mean(axis=0)[2:]*100, color=c_pal[0], ls=linestyle[0],label='Freq-depend,\nno deconv')
ax2.plot(ell[2:], diff_cl_beam_const_cos.mean(axis=0)[2:]*100, color=c_pal[1], ls=linestyle[0],label='Costant')
ax2.plot(ell[2:], diff_cl_beam_deconv_cos.mean(axis=0)[2:]*100, color=c_pal[2], ls=linestyle[0],label='Deconv')
ax2.axhline(ls='--', c= 'k', alpha=0.3)
ax2.set_xlim([lmin,lmax_plot])
ax2.set_ylim([-30,2])
ax2.set_xlabel(r'$\ell$')
#ax2.legend()


ax3.set_title('Cosine, Nfg=4')
ax3.plot(ell[2:], diff_cl_beam_cos_Nfg4.mean(axis=0)[2:]*100, color=c_pal[0], ls=linestyle[0],label='Freq-depend,\nno deconv')
ax3.plot(ell[2:], diff_cl_beam_const_cos_nfg4.mean(axis=0)[2:]*100, color=c_pal[1], ls=linestyle[0],label='Costant')
ax3.plot(ell[2:], diff_cl_beam_deconv_cos_Nfg4.mean(axis=0)[2:]*100, color=c_pal[2], ls=linestyle[0],label='Deconv')
ax3.axhline(ls='--', c= 'k', alpha=0.3)
ax3.set_xlim([lmin,lmax_plot])
ax3.set_ylim([-30,2])
ax3.set_xlabel(r'$\ell$')
ax3.legend()


plt.show()