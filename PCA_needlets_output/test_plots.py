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

out_dir_maps_recon = f'maps_reconstructed_nuovo_1/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
out_dir_cl = out_dir_maps_recon+'cls_recons_need/'

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

nside=128
npix= hp.nside2npix(nside)
jmax=4
lmax= 3*nside-1
if fg_comp=='synch_ff_ps':
	Nfg=3
if fg_comp=='synch_ff_ps_pol':
	Nfg3,Nfg18=3,18
B = pippo.mylibpy_jmax_lmax2B(jmax, lmax)


###############################################
########################## cl #################################
cl_cosmo_HI=np.loadtxt(out_dir_cl+f'cl_deconv_cosmo_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')
cl_cosmo_recon_HI=np.loadtxt(out_dir_cl+f'cl_deconv_cosmo_recon_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')


cl_PCA_HI=np.loadtxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')
cl_PCA_std = np.loadtxt('/home/bianca/Documents/HI IM SKA/PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_SKA_AA4_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')
#####################################################################

lmax_cl = 2*nside
ell=np.arange(lmax_cl+1)
factor=ell*(ell+1)/(2*np.pi)
lmin = 3

lmax_plot= lmax_theta_worst
print(lmax_plot)

diff_cl = cl_PCA_HI/cl_cosmo_HI-1

####################################################################################
############################### test reconstruction ##################################

diff_cl_need2sphe_recon = cl_cosmo_recon_HI/cl_cosmo_HI-1
diff_cl_need_std = cl_PCA_HI/cl_PCA_std - 1
diff_cl_PCA_cosmo_recons = cl_PCA_HI/cl_cosmo_recon_HI-1

chh = [0,ich,num_ch-1]

fig=plt.figure()
plt.title(r'Needlet reconstruction of $C_{\ell}^{\rm cosmo}$')
for i, ch in enumerate(chh):
	plt.plot(ell[lmin:],diff_cl_need2sphe_recon[i][lmin:]*100, label=f'{nu_ch[ch]} MHz')
plt.plot(ell[lmin:],diff_cl_need2sphe_recon.mean(axis=0)[lmin:]*100, 'k--',label='mean')
plt.axhline(ls='--',c='k', alpha=0.3)
plt.ylim([-22,2])
plt.xlim([lmin,lmax_plot])
plt.xticks(np.arange(lmin,lmax_plot+1, 10))
plt.legend(ncols=2)
plt.ylabel(r'$C_{\ell}^{\rm cosmo}/C_{\ell}^{\rm cosmo~recons}$ -1 [%]')
plt.xlabel(r'$\ell$')
plt.savefig(f'Plots_paper/cosmo_recons_std_need_{num_ch}_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_nside{nside}.png')

fig=plt.figure()
plt.title('PCA need / cosmo recons')
for ch in range(0,num_ch, 10):
	plt.plot(ell[lmin:],diff_cl_PCA_cosmo_recons[ch][lmin:], label=f'ch:{nu_ch[ch]}')
plt.plot(ell[lmin:],diff_cl_PCA_cosmo_recons.mean(axis=0)[lmin:], 'k--',label='mean')
plt.ylim([-0.5,0.5])
plt.legend(ncols=4, loc='upper left', frameon=False,columnspacing=1.)
plt.ylabel(r'$C_{\ell}^{\rm PCA~need}/C_{\ell}^{\rm cosmo~recons}$ -1 [%]')
plt.xlabel(r'$\ell$')

###################################################################################################################################
###################################################################################################################################
beam_s = 'SKA_AA4'
fg_comp = 'synch_ff_ps_pol'

out_dir_maps_recon = f'maps_reconstructed_nuovo_1/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
out_dir_cl = out_dir_maps_recon+'cls_recons_need/'

if fg_comp=='synch_ff_ps':
	Nfg=3
if fg_comp=='synch_ff_ps_pol':
	Nfg18, Nfg3, Nfg6, Nfg4 =18, 3, 6, 4

cl_PCA_HI_pol_18=np.loadtxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg18}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')
cl_PCA_HI_pol_3=np.loadtxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg3}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')
#cl_PCA_HI_pol_4=np.loadtxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg4}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')
cl_PCA_HI_pol_6=np.loadtxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg6}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')

###################################################################
####  comparison with standard PCA ##############################
out_dir_cl_std = f'../PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/'
cl_standard_PCA_HI=np.loadtxt(out_dir_cl_std+f'cl_deconv_PCA_HI_noise_synch_ff_ps_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg3}_lmax{2*nside}_nside{nside}.dat')

cl_standard_PCA_HI_pol_18=np.loadtxt(out_dir_cl_std+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg18}_lmax{2*nside}_nside{nside}.dat')
cl_standard_PCA_HI_pol_3=np.loadtxt(out_dir_cl_std+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg3}_lmax{2*nside}_nside{nside}.dat')
#cl_standard_PCA_HI_pol_4=np.loadtxt(out_dir_cl_std+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg4}_lmax{2*nside}_nside{nside}.dat')
cl_standard_PCA_HI_pol_6=np.loadtxt(out_dir_cl_std+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg6}_lmax{2*nside}_nside{nside}.dat')
#####################################################################
fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Mean over frequency channels, with pol leakage, '+r'N$_{\rm fg}$='+f'{Nfg}')
plt.plot(ell[lmin:], factor[lmin:]*cl_cosmo_HI.mean(axis=0)[lmin:],'k--',label = f'Cosmo HI + noise')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_pol_3.mean(axis=0)[lmin:], c=c_pal[0], label=r'Need-PCA, $N_{\rm fg}$=3')
plt.plot(ell[lmin:],factor[lmin:]*cl_standard_PCA_HI_pol_3.mean(axis=0)[lmin:], ls='--',c=c_pal[0], label=r'PCA HI, $N_{\rm fg}$=3')

#plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_pol_4.mean(axis=0)[lmin:], c=c_pal[1], label=r'Need-PCA, $N_{\rm fg}$=4')
#plt.plot(ell[lmin:],factor[lmin:]*cl_standard_PCA_HI_pol_4.mean(axis=0)[lmin:], ls='--',c=c_pal[1], label=r'PCA HI, $N_{\rm fg}$=4')

plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_pol_6.mean(axis=0)[lmin:], c=c_pal[1], label=r'Need-PCA, $N_{\rm fg}$=6')
plt.plot(ell[lmin:],factor[lmin:]*cl_standard_PCA_HI_pol_6.mean(axis=0)[lmin:], ls='--',c=c_pal[2], label=r'PCA, $N_{\rm fg}$=6')

plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_pol_18.mean(axis=0)[lmin:], c=c_pal[2],  label=r'Need-PCA, $N_{\rm fg}$=18')
plt.plot(ell[lmin:],factor[lmin:]*cl_standard_PCA_HI_pol_18.mean(axis=0)[lmin:], ls='--',c=c_pal[3],  label=r'PCA, $N_{\rm fg}$=18')
plt.xlim([lmin,lmax_plot])
plt.ylim([-5e-4,6e-3])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(ncols=2)
frame1.set_ylabel(r'$  \ell(\ell+1)/2\pi~ \langle C_{\ell} \rangle_{\rm ch}$ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])
#frame1.yaxis.set_major_formatter(formatter) 

diff_cl_pol_3 = cl_PCA_HI_pol_3/cl_cosmo_HI-1
#diff_cl_pol_4 = cl_PCA_HI_pol_4/cl_cosmo_HI-1
diff_cl_pol_6 = cl_PCA_HI_pol_6/cl_cosmo_HI-1
diff_cl_pol_18 = cl_PCA_HI_pol_18/cl_cosmo_HI-1


diff_std_PCA_cl_pol_3 = cl_standard_PCA_HI_pol_3/cl_cosmo_HI-1
diff_std_PCA_cl_pol_6 = cl_standard_PCA_HI_pol_6/cl_cosmo_HI-1
#diff_std_PCA_cl_pol_4 = cl_standard_PCA_HI_pol_4/cl_cosmo_HI-1
diff_std_PCA_cl_pol_18 = cl_standard_PCA_HI_pol_18/cl_cosmo_HI-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_pol_3.mean(axis=0)[lmin:]*100, c=c_pal[0], label=r'Need-PCA, $N_{\rm fg}$=3')
plt.plot(ell[lmin:], diff_std_PCA_cl_pol_3.mean(axis=0)[lmin:]*100,ls='--', c=c_pal[0], label=r'PCA, $N_{\rm fg}$=3')

#plt.plot(ell[lmin:], diff_cl_pol_4.mean(axis=0)[lmin:]*100, c=c_pal[1], label=r'Need-PCA, $N_{\rm fg}$=4')
#plt.plot(ell[lmin:], diff_std_PCA_cl_pol_4.mean(axis=0)[lmin:]*100,ls='--', c=c_pal[1], label=r'PCA, $N_{\rm fg}$=4')

plt.plot(ell[lmin:], diff_cl_pol_6.mean(axis=0)[lmin:]*100, c=c_pal[1], label=r'Need-PCA, $N_{\rm fg}$=6')
plt.plot(ell[lmin:], diff_std_PCA_cl_pol_6.mean(axis=0)[lmin:]*100,ls='--',c=c_pal[2], label=r'PCA HI, $N_{\rm fg}$=6')

plt.plot(ell[lmin:], diff_cl_pol_18.mean(axis=0)[lmin:]*100, c=c_pal[2], label=r'Need-PCA, $N_{\rm fg}$=18')
plt.plot(ell[lmin:], diff_std_PCA_cl_pol_18.mean(axis=0)[lmin:]*100,ls='--', c=c_pal[3], label=r'PCA HI, $N_{\rm fg}$=18')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_ylim([-60,2])
plt.xlim([lmin,lmax_plot])
frame2.set_ylabel(r'$ \langle \Delta\rangle_{\rm ch}$ [%]')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))

plt.close('all')

########################################################################
############################### test pol ##################################

chh = [0,ich,num_ch-1]

fig, ax = plt.subplots(1,1)
plt.title(f'Mean over frequency channels')
for i, cc in enumerate(chh):
	ax.plot(ell[lmin:], np.abs(diff_cl[cc])[lmin:]*100, color=c_pal[i], label = f'nu={nu_ch[cc]}')
	#ax.plot(ell[lmin:], np.abs(diff_cl_pol_4[cc])[lmin:]*100, ls='--', color=c_pal[i])
ax.axhline(ls='--', c= 'k', alpha=0.3)
ax.set_xlim([lmin, lmax_plot+1])
ax.set_ylim([-2,40])
ax.set_ylabel(r'|$\Delta$| [%]')
ax.set_xlabel(r'$\ell$')
ax.set_xticks(np.arange(lmin,lmax_plot+1, 10))
plt.legend(title='- without pol leak, Nfg=3, -- with pol leakage, Nfg=4')



#plt.show()

fig, ax = plt.subplots(1,1)
plt.title(f'Mean over frequency channels, Nfg=3')
#for i, cc in enumerate(chh):
#	ax.plot(ell[lmin:], np.abs(diff_cl[cc])[lmin:]*100, color=c_pal[i], label = f'nu={nu_ch[cc]}')
#	ax.plot(ell[lmin:], np.abs(diff_cl_pol_4[cc])[lmin:]*100, ls='--', color=c_pal[i])

ax.plot(ell[lmin:], diff_cl.mean(axis=0)[lmin:]*100, label='w/o pol leak')
ax.plot(ell[lmin:], diff_cl_pol_3.mean(axis=0)[lmin:]*100, label='with pol leak')
ax.axhline(ls='--', c= 'k', alpha=0.3)
ax.set_xlim([lmin, lmax_plot+1])
ax.set_ylim([-30,2])
ax.set_ylabel(r'$\langle \Delta \rangle $[%]')
ax.set_xlabel(r'$\ell$')
ax.set_xticks(np.arange(lmin,lmax_plot+1, 10))
plt.legend()

#######################################################################################
############################### CROSS CORR MASK #########################################

map_cosmo=np.load('../PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_SKA_AA4_noise_mask0.5_unseen/cosmo_HI_noise_105_900.5_1004.5MHz_lmax383_nside128.npy',allow_pickle=True)[ich]
map_PCA=np.load('../PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_SKA_AA4_noise_mask0.5_unseen/res_PCA_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg3_lmax383_nside128.npy',allow_pickle=True)[ich]
map_PCA_pol3=np.load('../PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_SKA_AA4_noise_mask0.5_unseen/res_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg3_lmax383_nside128.npy',allow_pickle=True)[ich]
map_PCA_pol6=np.load('../PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_SKA_AA4_noise_mask0.5_unseen/res_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg6_lmax383_nside128.npy',allow_pickle=True)[ich]
map_PCA_pol18=np.load('../PCA_pixels_output/Maps_PCA_nuovo/No_mean/Beam_SKA_AA4_noise_mask0.5_unseen/res_PCA_HI_noise_synch_ff_ps_pol_105_900.5_1004.5MHz_Nfg18_lmax383_nside128.npy',allow_pickle=True)[ich]


cl_cross_cosmo = hp.anafast(map1=map_cosmo, map2=map_cosmo, lmax=lmax_cl)
cl_cross_cosmo_PCA = hp.anafast(map1=map_cosmo, map2=map_PCA, lmax=lmax_cl)
cl_cross_cosmo_PCA_pol3 = hp.anafast(map1=map_cosmo, map2=map_PCA_pol3, lmax=lmax_cl)
cl_cross_cosmo_PCA_pol6 = hp.anafast(map1=map_cosmo, map2=map_PCA_pol6, lmax=lmax_cl)
cl_cross_cosmo_PCA_pol18 = hp.anafast(map1=map_cosmo, map2=map_PCA_pol18, lmax=lmax_cl)


fig, ax = plt.subplots(1,1)
ax.set_title('PCA standard, fsky=50%')
ax.plot(ell[lmin:], factor[lmin:]*cl_cross_cosmo[lmin:], 'k' ,label='Input cosmo')
ax.plot(ell[lmin:], factor[lmin:]*cl_cross_cosmo_PCA[lmin:],ls='--',c=c_pal[0], mfc='none', label='PCA w/o pol leak Nfg=3')
ax.plot(ell[lmin:], factor[lmin:]*cl_cross_cosmo_PCA_pol3[lmin:],ls=':',c=c_pal[1], mfc='none', label='PCA with pol leak Nfg=3')
ax.plot(ell[lmin:], factor[lmin:]*cl_cross_cosmo_PCA_pol6[lmin:],ls='-.', c=c_pal[2],mfc='none', label='PCA with pol leak Nfg=6')
ax.plot(ell[lmin:], factor[lmin:]*cl_cross_cosmo_PCA_pol18[lmin:],ls=(0, (5, 1)),c=c_pal[3], mfc='none', label='PCA with pol leak Nfg=18')

ax.set_xlim([lmin, lmax_plot+1])
ax.set_ylabel(r'$  \ell(\ell+1)/2\pi~ C_{\ell}^{\rm cosmo \times }  $ [mK$^{2}$]')
ax.set_xlabel(r'$\ell$')
ax.set_xticks(np.arange(lmin,lmax_plot+1, 10))
plt.legend()





################################################################
##################### leakage #########################


cl_leak_HI_mask_deconv_interp=np.loadtxt(out_dir_cl+f'cl_deconv_leak_HI_synch_ff_ps_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{lmax_cl}_nside{nside}.dat')
cl_leak_fg_mask_deconv_interp=np.loadtxt(out_dir_cl+f'cl_deconv_leak_fg_synch_ff_ps_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{lmax_cl}_nside{nside}.dat')

cl_leak_HI_pol_mask_deconv_interp=np.loadtxt(out_dir_cl+f'cl_deconv_leak_HI_synch_ff_ps_pol_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg3}_jmax{jmax}_lmax{lmax_cl}_nside{nside}.dat')
cl_leak_fg_pol_mask_deconv_interp=np.loadtxt(out_dir_cl+f'cl_deconv_leak_fg_synch_ff_ps_pol_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg3}_jmax{jmax}_lmax{lmax_cl}_nside{nside}.dat')

fig, ax= plt.subplots()
ax.set_title('Foreground and HI leakage, mean over channels, Nfg=3')

ax.plot(ell[lmin:], factor[lmin:]*cl_leak_HI_mask_deconv_interp.mean(axis=0)[lmin:], c=c_pal[0],ls='-',label= 'HI leakage')
ax.plot(ell[lmin:], factor[lmin:]*cl_leak_HI_pol_mask_deconv_interp.mean(axis=0)[lmin:], c=c_pal[0],ls='--')

ax.plot(ell[lmin:], factor[lmin:]*cl_leak_fg_mask_deconv_interp.mean(axis=0)[lmin:], c=c_pal[1],ls='-',label= 'Fg leakage')
ax.plot(ell[lmin:], factor[lmin:]*cl_leak_fg_pol_mask_deconv_interp.mean(axis=0)[lmin:], c=c_pal[1],ls='--')
ax.set_yscale('log')

ax.set_ylabel(r'$  \ell(\ell+1)/2\pi~ \langle C_{\ell} \rangle_{\rm ch}$ [mK$^{2}$]')
ax.set_xlabel(r'$\ell$')

fig.legend(title='- w/o pol leak,\n-- with pol leakage', loc='outside center right',bbox_to_anchor=(1, 0.72))


##############################################################################
cl_standard_leak_HI_mask_deconv_interp=np.loadtxt(out_dir_cl_std+f'cl_deconv_leak_HI_noise_synch_ff_ps_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_lmax{lmax_cl}_nside{nside}.dat')
cl_standard_leak_fg_mask_deconv_interp=np.loadtxt(out_dir_cl_std+f'cl_deconv_leak_fg_synch_ff_ps_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_lmax{lmax_cl}_nside{nside}.dat')

#cl_standard_leak_HI_pol_mask_deconv_interp=np.loadtxt(out_dir_cl+f'cl_deconv_leak_HI_synch_ff_ps_pol_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg3}_jmax{jmax}_lmax{lmax_cl}_nside{nside}.dat')
#cl_standard_leak_fg_pol_mask_deconv_interp=np.loadtxt(out_dir_cl+f'cl_deconv_leak_fg_synch_ff_ps_pol_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg3}_jmax{jmax}_lmax{lmax_cl}_nside{nside}.dat')


fig, ax= plt.subplots()
ax.set_title('Foreground and HI leakage, mean over channels, Nfg=3')

ax.plot(ell[lmin:], factor[lmin:]*cl_leak_HI_mask_deconv_interp.mean(axis=0)[lmin:], c=c_pal[0],ls='-',label= 'HI leakage')
ax.plot(ell[lmin:], factor[lmin:]*cl_standard_leak_HI_mask_deconv_interp.mean(axis=0)[lmin:], c=c_pal[0],ls='--')

ax.plot(ell[lmin:], factor[lmin:]*cl_leak_fg_mask_deconv_interp.mean(axis=0)[lmin:], c=c_pal[1],ls='-',label= 'Fg leakage')
ax.plot(ell[lmin:], factor[lmin:]*cl_standard_leak_fg_mask_deconv_interp.mean(axis=0)[lmin:], c=c_pal[1],ls='--')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel(r'$  \ell(\ell+1)/2\pi~ \langle C_{\ell} \rangle_{\rm ch}$ [mK$^{2}$]')
ax.set_xlabel(r'$\ell$')

fig.legend(title='- Need-PCA,\n-- Standard PCA', loc='outside center right',bbox_to_anchor=(1, 0.72))


fig, ax= plt.subplots()
ax.set_title('Foreground and HI leakage, mean over channels, Nfg=3')

ax.plot(ell[lmin:], 100*((cl_standard_PCA_HI/cl_cosmo_recon_HI).mean(axis=0)[lmin:]-1), c=c_pal[0],label= 'Cl PCA HI')
ax.plot(ell[lmin:], 100*((cl_standard_leak_HI_mask_deconv_interp/cl_leak_HI_mask_deconv_interp).mean(axis=0)[lmin:]-1),ls='--', c=c_pal[0],label= 'HI leakage')
#ax.plot(ell[lmin:], 100*((cl_leak_fg_mask_deconv_interp/cl_standard_leak_fg_mask_deconv_interp).mean(axis=0)[lmin:]-1), ls='-.', c=c_pal[0],label= 'Fg leakage')
#ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_xlim([lmin, 10])
ax.set_ylim([-20, 2])
ax.set_ylabel(r'$ \%  \langle C_{\ell}^{\rm Std-PCA}/C_{\ell}^{\rm Need-PCA} \rangle_{\rm ch}$ -1 ')
ax.set_xlabel(r'$\ell$')
fig.legend( loc='outside center right',bbox_to_anchor=(1, 0.72))

print((cl_standard_PCA_HI/cl_cosmo_recon_HI).mean(axis=0)[lmin:30]-1)

plt.show()

#########################################################################