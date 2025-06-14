import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

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



from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 

c_pal = sns.color_palette().as_hex()
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



###########################################################################################
pix_mask = hp.query_strip(nside, theta1=np.pi*2/3, theta2=np.pi/3)

mask_50 = np.zeros(npix)
mask_50[pix_mask] =1
fsky_50 = np.sum(mask_50)/hp.nside2npix(nside)


bad_v = np.where(mask_50==0)
del mask_50

#########################################################################
####################### mappe #############################################

map_PCA_HI_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy', allow_pickle=True)
map_PCA_fg_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_PCA_fg_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy', allow_pickle=True)
map_input_HI_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_cosmo_HI_noise_{num_ch}_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy', allow_pickle=True)
map_input_fg_need2pix=np.load(out_dir_maps_recon+f'maps_reconstructed_input_fg_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.npy', allow_pickle=True)

#map_PCA_HI_need2pix[:, bad_v]=hp.UNSEEN
#map_PCA_fg_need2pix[:, bad_v]=hp.UNSEEN 
#map_input_HI_need2pix[:, bad_v]=hp.UNSEEN
#map_input_fg_need2pix[:, bad_v]=hp.UNSEEN


path_cosmo_HI = f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/cosmo_HI_noise_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz_lmax{lmax}_nside{nside}'
path_fg = f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/fg_input_{fg_comp}_{num_ch}_{min_ch:1.1f}_{max_ch:1.1f}MHz_lmax{lmax}_nside{nside}'
fg = np.load(path_fg+'.npy', allow_pickle=True)
cosmo_HI = np.load(path_cosmo_HI+'.npy', allow_pickle=True)

##################
map_res = cosmo_HI-map_PCA_HI_need2pix
lon=0 #deg
lat=67 #deg
rot = [lon, lat, 0.]
xsize=100
ysize=xsize

reso = hp.nside2resol(nside, arcmin=True)
map0  = hp.gnomview(cosmo_HI[ich],rot=rot,reso=reso,xsize=xsize,ysize=ysize, min=0, max=1,return_projected_map=True, no_plot=True)
#map1  = hp.gnomview(cosmo_HI[ich]+fg[ich],rot=rot, coord='G', reso=reso,xsize=xsize,ysize=ysize, min=-1e3, max=1e3,return_projected_map=True, no_plot=True)
map2  = hp.gnomview(map_PCA_HI_need2pix[ich],rot=rot, reso=reso,xsize=xsize,ysize=ysize, min=0, max=1,return_projected_map=True, no_plot=True)
map3  = hp.gnomview(map_res[ich],rot=rot,  reso=reso,xsize=xsize,ysize=ysize, min=-0.5, max=0.5,return_projected_map=True, no_plot=True)


final_pixel = [256,256]
zoom_param  = final_pixel/np.array(map3.shape)


final_map0 = ndimage.zoom(map0, zoom_param, order=3)
final_map2 = ndimage.zoom(map2, zoom_param, order=3)
final_map3 = ndimage.zoom(map3, zoom_param, order=3)

std_patch = np.std(map3)
print(f'std_patch={std_patch:.3f}, dimension:{xsize*reso/60.:0.2f}X{ysize*reso/60.:0.2f} deg ')

maps = [map0,map2,map3]
#del map0; del map2; del map3

final_maps = [final_map0,final_map2,final_map3]

cmap= 'viridis'
titles = [f'Input HI + noise',f'Cleaned HI + noise', f'Residuals']
for mp,title in zip(final_maps, titles):
	fig, ax = plt.subplots(1,1)
	image=ax.imshow(mp,cmap = cmap)
	ax.set_title(title)
	ax.set_xlabel(r'$\theta$[deg]')
	ax.set_ylabel(r'$\theta$[deg]')
	ax.margins(x=0)
	norm = colors.Normalize(vmin=0, vmax=1)
	plt.subplots_adjust(hspace=0.4, bottom=0.26, left=0.1, top=0.95)
	sub_ax = plt.axes([0.38, 0.12, 0.24, 0.02])
	fig.colorbar(image,cax=sub_ax,orientation='horizontal',label='T [mK]')
	title=title.replace(' ', '_')
	print(title)
	plt.savefig(f'Plots_paper/gnomview_vertical_{title}_{fg_comp}_PCAHI_std_need_beam{beam_s}_jmax{jmax}_lmax{lmax}_Nfg{Nfg}_nside{nside}.png',bbox_inches='tight')

del map_input_fg_need2pix; del map_input_HI_need2pix; del map_PCA_fg_need2pix; del map_PCA_HI_need2pix; del map_res; del maps 
############################################################
########################## cl #################################
cl_cosmo_HI=np.loadtxt(out_dir_cl+f'cl_deconv_cosmo_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')
cl_cosmo_recon_HI=np.loadtxt(out_dir_cl+f'cl_deconv_cosmo_recon_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')

cl_PCA_HI=np.loadtxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')
cl_PCA_std = np.loadtxt('/home/bianca/Documents/HI IM SKA/PCA_pixels_output/Maps_PCA/No_mean/Beam_SKA_AA4_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/cl_deconv_PCA_HI_noise_synch_ff_ps_105_900.5_1004.5MHz_Nfg3_lmax256_nside128.dat')
#####################################################################

lmax_cl = 2*nside
ell=np.arange(lmax_cl+1)
factor=ell*(ell+1)/(2*np.pi)
lmin = 3

lmax_plot= lmax_theta_worst
print(lmax_plot)

fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(r'$\nu$='+f'{nu_ch[ich]} MHz, '+r'N$_{\rm fg}$='+f'{Nfg}')
plt.plot(ell[lmin:],factor[lmin:]*cl_cosmo_HI[ich][lmin:],'k--',label='Cosmo HI + noise')
#plt.plot(ell[lmin:],factor[lmin:]*cl_cosmo_recon_HI[ich][lmin:],'k--',label='Cosmo HI + noise recons')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI[ich][lmin:],  label='Need-PCA HI + noise')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim([lmin,lmax_plot])
plt.ylim([-5e-4,6e-3])
plt.legend()
frame1.set_ylabel(r'$  \ell(\ell+1)/2\pi~ C_{\ell} $ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])
#frame1.yaxis.set_major_formatter(formatter) 

diff_cl = cl_PCA_HI/cl_cosmo_HI-1


frame2=fig.add_axes((.1,.1,.8,.2))
#plt.plot(ell[lmin:], diff_cl_need2sphe_recon[ich][lmin:]*100,label='recons')
plt.plot(ell[lmin:], diff_cl[ich][lmin:]*100,)

frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,lmax_plot])
frame2.set_ylim([-30,2])
frame2.set_ylabel(r'$\Delta$ [%]')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))
#plt.legend()
plt.savefig(f'Plots_paper/cl_std_need_ch{nu_ch[ich]}_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg}_nside{nside}_mask0.5.png', bbox_inches='tight')

#plt.show()

fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Mean over frequency channels, '+r'N$_{\rm fg}$='+f'{Nfg}')
plt.plot(ell[lmin:], factor[lmin:]*cl_cosmo_HI.mean(axis=0)[lmin:],'k--',label = f'Cosmo HI + noise')
#plt.plot(ell[lmin:],factor[lmin:]*cl_cosmo_recon_HI.mean(axis=0)[lmin:],'k--',label='Cosmo HI + noise recons')
plt.plot(ell[lmin:], factor[lmin:]*cl_PCA_HI.mean(axis=0)[lmin:], label = f'Need-PCA HI + noise')
plt.xlim([lmin,lmax_plot])
plt.ylim([-5e-4,6e-3])
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
frame1.set_ylabel(r'$  \ell(\ell+1)/2\pi~ \langle C_{\ell} \rangle_{\rm ch}$ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])
#frame1.yaxis.set_major_formatter(formatter) 

frame2=fig.add_axes((.1,.1,.8,.2))
#plt.plot(ell[lmin:], diff_cl_need2sphe_recon.mean(axis=0)[lmin:]*100,label='recons')
plt.plot(ell[lmin:], diff_cl.mean(axis=0)[lmin:]*100)
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,lmax_plot])
frame2.set_ylim([-30,2])
frame2.set_ylabel(r'$ \langle \Delta\rangle_{\rm ch}$ [%]')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))
#plt.tight_layout()
plt.savefig(f'Plots_paper/cl_std_need_mean_ch_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg}_nside{nside}_mask0.5.png', bbox_inches='tight')


###################################################################################################################################
###################################################################################################################################
beam_s = 'SKA_AA4'
fg_comp = 'synch_ff_ps_pol'

out_dir_maps_recon = f'maps_reconstructed/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
out_dir_cl = out_dir_maps_recon+'cls_recons_need/'

if fg_comp=='synch_ff_ps':
	Nfg=3
if fg_comp=='synch_ff_ps_pol':
	Nfg18, Nfg3, Nfg6 =18, 3, 6

cl_PCA_HI_pol_18=np.loadtxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg18}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')
cl_PCA_HI_pol_3=np.loadtxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg3}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')
cl_PCA_HI_pol_6=np.loadtxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg6}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')

###################################################################
####  comparison with standard PCA ##############################
out_dir_cl_std = f'../PCA_pixels_output/Maps_PCA/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/power_spectra_cls_from_healpix_maps/'
cl_standard_PCA_HI_pol_18=np.loadtxt(out_dir_cl_std+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg18}_lmax{2*nside}_nside{nside}.dat')
cl_standard_PCA_HI_pol_3=np.loadtxt(out_dir_cl_std+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg3}_lmax{2*nside}_nside{nside}.dat')
cl_standard_PCA_HI_pol_6=np.loadtxt(out_dir_cl_std+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg6}_lmax{2*nside}_nside{nside}.dat')
#####################################################################
fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Mean over frequency channels, with pol leakage, '+r'N$_{\rm fg}$='+f'{Nfg}')
plt.plot(ell[lmin:], factor[lmin:]*cl_cosmo_HI.mean(axis=0)[lmin:],'k--',label = f'Cosmo HI + noise')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_pol_3.mean(axis=0)[lmin:], c=c_pal[0], label=r'Need-PCA, $N_{\rm fg}$=3')
plt.plot(ell[lmin:],factor[lmin:]*cl_standard_PCA_HI_pol_3.mean(axis=0)[lmin:], ls='--',c=c_pal[0], label=r'PCA HI, $N_{\rm fg}$=3')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_pol_6.mean(axis=0)[lmin:], c=c_pal[1], label=r'Need-PCA, $N_{\rm fg}$=6')
plt.plot(ell[lmin:],factor[lmin:]*cl_standard_PCA_HI_pol_6.mean(axis=0)[lmin:], ls='--',c=c_pal[1], label=r'PCA, $N_{\rm fg}$=6')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_pol_18.mean(axis=0)[lmin:], c=c_pal[2],  label=r'Need-PCA, $N_{\rm fg}$=18')
plt.plot(ell[lmin:],factor[lmin:]*cl_standard_PCA_HI_pol_18.mean(axis=0)[lmin:], ls='--',c=c_pal[2],  label=r'PCA, $N_{\rm fg}$=18')
plt.xlim([lmin,lmax_plot])
plt.ylim([-5e-4,6e-3])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(ncols=2)
frame1.set_ylabel(r'$  \ell(\ell+1)/2\pi~ \langle C_{\ell} \rangle_{\rm ch}$ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])
#frame1.yaxis.set_major_formatter(formatter) 

diff_cl_pol_3 = cl_PCA_HI_pol_3/cl_cosmo_HI-1
diff_cl_pol_6 = cl_PCA_HI_pol_6/cl_cosmo_HI-1
diff_cl_pol_18 = cl_PCA_HI_pol_18/cl_cosmo_HI-1


diff_std_PCA_cl_pol_3 = cl_standard_PCA_HI_pol_3/cl_cosmo_HI-1
diff_std_PCA_cl_pol_6 = cl_standard_PCA_HI_pol_6/cl_cosmo_HI-1
diff_std_PCA_cl_pol_18 = cl_standard_PCA_HI_pol_18/cl_cosmo_HI-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_pol_3.mean(axis=0)[lmin:]*100, c=c_pal[0], label=r'Need-PCA, $N_{\rm fg}$=3')
plt.plot(ell[lmin:], diff_std_PCA_cl_pol_3.mean(axis=0)[lmin:]*100,ls='--', c=c_pal[0], label=r'PCA, $N_{\rm fg}$=3')
plt.plot(ell[lmin:], diff_cl_pol_6.mean(axis=0)[lmin:]*100, c=c_pal[1], label=r'Need-PCA, $N_{\rm fg}$=6')
plt.plot(ell[lmin:], diff_std_PCA_cl_pol_6.mean(axis=0)[lmin:]*100,ls='--',c=c_pal[1], label=r'PCA HI, $N_{\rm fg}$=6')
plt.plot(ell[lmin:], diff_cl_pol_18.mean(axis=0)[lmin:]*100, c=c_pal[2], label=r'Need-PCA, $N_{\rm fg}$=18')
plt.plot(ell[lmin:], diff_std_PCA_cl_pol_18.mean(axis=0)[lmin:]*100,ls='--', c=c_pal[2], label=r'PCA HI, $N_{\rm fg}$=18')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,lmax_plot])
frame2.set_ylim([-60,2])
frame2.set_ylabel(r'$ \langle \Delta\rangle_{\rm ch}$ [%]')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))

plt.savefig(f'Plots_paper/comparison_cl_PCA_standard_need_mean_ch_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg3_Nfg6_Nfg18_nside{nside}_mask0.5.png', bbox_inches='tight')

#####################################################################
fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(r'$\nu$='+f'{nu_ch[ich]} MHz, with pol leakage')
plt.plot(ell[lmin:],factor[lmin:]*cl_cosmo_HI[ich][lmin:],'k--',label='Cosmo HI + noise')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_pol_3[ich][lmin:], c=c_pal[0], label=r'Need-PCA HI + noise, $N_{\rm fg}$=3')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_pol_6[ich][lmin:], c=c_pal[1], label=r'Need-PCA HI + noise, $N_{\rm fg}$=6')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_pol_18[ich][lmin:], c=c_pal[2],  label=r'Need-PCA HI + noise, $N_{\rm fg}$=18')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim([lmin,lmax_plot])
plt.ylim([-5e-4,6e-3])
plt.legend()
frame1.set_ylabel(r'$  \ell(\ell+1)/2\pi~ C_{\ell} $ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])
#frame1.yaxis.set_major_formatter(formatter) 


diff_cl_pol_3 = cl_PCA_HI_pol_3/cl_cosmo_HI-1
diff_cl_pol_6 = cl_PCA_HI_pol_6/cl_cosmo_HI-1
diff_cl_pol_18 = cl_PCA_HI_pol_18/cl_cosmo_HI-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_pol_3[ich][lmin:]*100, c=c_pal[0], label=r'Need-PCA HI + noise, $N_{\rm fg}$=3')
plt.plot(ell[lmin:], diff_cl_pol_6[ich][lmin:]*100, c=c_pal[1], label=r'Need-PCA HI + noise, $N_{\rm fg}$=6')
plt.plot(ell[lmin:], diff_cl_pol_18[ich][lmin:]*100, c=c_pal[2], label=r'Need-PCA HI + noise, $N_{\rm fg}$=18')


frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,lmax_plot])
frame2.set_ylim([-30,2])
frame2.set_ylabel(r'$\Delta$ [%]')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))
plt.savefig(f'Plots_paper/cl_std_need_ch{nu_ch[ich]}_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg3}_{Nfg6}_{Nfg18}_nside{nside}_mask0.5.png', bbox_inches='tight')

#plt.show()

fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Mean over frequency channels, with pol leakage')
plt.plot(ell[lmin:], factor[lmin:]*cl_cosmo_HI.mean(axis=0)[lmin:],'k--',label = f'Cosmo HI + noise')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_pol_3.mean(axis=0)[lmin:], c=c_pal[0], label=r'Need-PCA HI + noise, $N_{\rm fg}$=3')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_pol_6.mean(axis=0)[lmin:], c=c_pal[1], label=r'Need-PCA HI + noise, $N_{\rm fg}$=6')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_pol_18.mean(axis=0)[lmin:], c=c_pal[2],  label=r'Need-PCA HI + noise, $N_{\rm fg}$=18')
plt.xlim([lmin,lmax_plot])
plt.ylim([-5e-4,6e-3])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
frame1.set_ylabel(r'$  \ell(\ell+1)/2\pi~ \langle C_{\ell} \rangle_{\rm ch}$ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])
#frame1.yaxis.set_major_formatter(formatter) 

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_pol_3.mean(axis=0)[lmin:]*100, c=c_pal[0], label=r'Need-PCA HI + noise, $N_{\rm fg}$=3')
plt.plot(ell[lmin:], diff_cl_pol_6.mean(axis=0)[lmin:]*100, c=c_pal[1], label=r'Need-PCA HI + noise, $N_{\rm fg}$=6')
plt.plot(ell[lmin:], diff_cl_pol_18.mean(axis=0)[lmin:]*100, c=c_pal[2], label=r'Need-PCA HI + noise, $N_{\rm fg}$=18')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,lmax_plot])
frame2.set_ylim([-30,2])
frame2.set_ylabel(r'$ \langle \Delta\rangle_{\rm ch}$ [%]')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))
plt.savefig(f'Plots_paper/cl_std_need_mean_ch_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg3}_{Nfg6}_{Nfg18}_nside{nside}_mask0.5.png', bbox_inches='tight')

#plt.show()
###########################################################################################
fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(r'$\nu$='+f'{nu_ch[ich]} MHz, with pol leakage, '+r'N$_{\rm fg}$='+f'{Nfg3}')
plt.plot(ell[lmin:],factor[lmin:]*cl_cosmo_HI[ich][lmin:],'k--',label='Cosmo HI + noise')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_pol_3[ich][lmin:], c=c_pal[0], label=r'Need-PCA HI + noise')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim([lmin,lmax_plot])
plt.ylim([-5e-4,6e-3])
plt.legend()
frame1.set_ylabel(r'$  \ell(\ell+1)/2\pi~ C_{\ell} $ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])
#frame1.yaxis.set_major_formatter(formatter) 

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_pol_3[ich][lmin:]*100, c=c_pal[0], label=r'Need-PCA HI + noise')


frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,lmax_plot])
frame2.set_ylim([-30,2])
frame2.set_ylabel(r'$\Delta$ [%]')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))
plt.savefig(f'Plots_paper/cl_std_need_ch{nu_ch[ich]}_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg3}_nside{nside}_mask0.5.png', bbox_inches='tight')

#plt.show()

fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'Mean over frequency channels, with pol leakage, '+r'N$_{\rm fg}$='+f'{Nfg3}')
plt.plot(ell[lmin:], factor[lmin:]*cl_cosmo_HI.mean(axis=0)[lmin:],'k--',label = f'Cosmo HI + noise')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_pol_3.mean(axis=0)[lmin:], c=c_pal[0], label=r'Need-PCA HI + noise')
plt.xlim([lmin,lmax_plot])
plt.ylim([-5e-4,6e-3])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
frame1.set_ylabel(r'$  \ell(\ell+1)/2\pi~ \langle C_{\ell} \rangle_{\rm ch}$ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])
#frame1.yaxis.set_major_formatter(formatter) 

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_pol_3.mean(axis=0)[lmin:]*100, c=c_pal[0], label=r'Need-PCA HI + noise')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,lmax_plot])
frame2.set_ylim([-30,2])
frame2.set_ylabel(r'$ \langle \Delta\rangle_{\rm ch}$ [%]')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))
plt.savefig(f'Plots_paper/cl_std_need_mean_ch_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg3}_nside{nside}_mask0.5.png', bbox_inches='tight')


##################################################################################################################
#################################################################################################################
###################################### costant beam #############################################################
beam_s = '1.3deg_SKA_AA4'
fg_comp = 'synch_ff_ps'

out_dir_maps_recon = f'maps_reconstructed/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
out_dir_cl = out_dir_maps_recon+'cls_recons_need/'

if fg_comp=='synch_ff_ps':
	Nfg=3
if fg_comp=='synch_ff_ps_pol':
	Nfg3,Nfg18=3,18
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
plt.title(r'$\theta_{\rm FMWH}=1.3$ deg, $\nu$='+f'{nu_ch[ich]} MHz,'+r'N$_{\rm fg}$='+f'{Nfg}')
plt.plot(ell[lmin:],factor[lmin:]*cl_cosmo_HI_1p3deg[ich][lmin:],'k--',label='Cosmo HI + noise')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_1p3deg[ich][lmin:],  label='Need-PCA HI + noise')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim([lmin,lmax_plot])
plt.ylim([-5e-4,6e-3])
plt.legend()
frame1.set_ylabel(r'$  \ell(\ell+1)/2\pi~ C_{\ell} $ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])
#frame1.yaxis.set_major_formatter(formatter) 


diff_cl_1p3deg = cl_PCA_HI_1p3deg/cl_cosmo_HI_1p3deg-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_1p3deg[ich][lmin:]*100,)

frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,lmax_plot])
frame2.set_ylim([-30,2])
frame2.set_ylabel(r'$\Delta$ [%]')
frame2.set_xlabel(r'$\ell$')
frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))
#frame2.yaxis.set_major_formatter(formatter) 
plt.savefig(f'Plots_paper/cl_std_need_ch{nu_ch[ich]}_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg}_nside{nside}_mask0.5.png', bbox_inches='tight')


fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(r'$\theta_{\rm FMWH}=1.3$ deg, '+f'Mean over frequency channels, '+r'N$_{\rm fg}$='+f'{Nfg}')
plt.plot(ell[lmin:], factor[lmin:]*cl_cosmo_HI_1p3deg.mean(axis=0)[lmin:],'k--',label = f'Cosmo HI + noise')
plt.plot(ell[lmin:], factor[lmin:]*cl_PCA_HI_1p3deg.mean(axis=0)[lmin:], mfc='none', label = f'Need-PCA HI + noise')
plt.xlim([lmin,lmax_plot])
plt.ylim([-5e-4,6e-3])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
frame1.set_ylabel(r'$  \ell(\ell+1)/2\pi~ \langle C_{\ell} \rangle_{\rm ch}$ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])
#frame1.yaxis.set_major_formatter(formatter) 

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_1p3deg.mean(axis=0)[lmin:]*100)
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,lmax_plot])
frame2.set_ylim([-30,2])
frame2.set_ylabel(r'$ \langle \Delta\rangle_{\rm ch}$ [%]')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))
#plt.tight_layout()
plt.savefig(f'Plots_paper/cl_std_need_mean_ch_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg}_nside{nside}_mask0.5.png', bbox_inches='tight')


###################################################################################################################################
###################################################################################################################################
beam_s = '1.3deg_SKA_AA4'
fg_comp = 'synch_ff_ps_pol'

if fg_comp=='synch_ff_ps':
	Nfg=3
if fg_comp=='synch_ff_ps_pol':
	Nfg3,Nfg18=3,18

out_dir_maps_recon = f'maps_reconstructed/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
out_dir_cl = out_dir_maps_recon+'cls_recons_need/'

cl_PCA_HI_1p3deg_pol=np.loadtxt(out_dir_cl+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg3}_jmax{jmax}_lmax{2*nside}_nside{nside}.dat')
cl_standard_PCA_HI_1p3deg_pol_3=np.loadtxt(out_dir_cl_std+f'cl_deconv_PCA_HI_noise_{fg_comp}_{num_ch}_{min_ch}_{max_ch}MHz_Nfg{Nfg3}_lmax{2*nside}_nside{nside}.dat')

#####################################################################
fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(r'$\theta_{\rm FMWH}=1.3$ deg, $\nu$='+f'{nu_ch[ich]} MHz, with pol leakage, '+r'N$_{\rm fg}$='+f'{Nfg3}')
plt.plot(ell[lmin:],factor[lmin:]*cl_cosmo_HI_1p3deg[ich][lmin:],'k--',label='Cosmo HI + noise')
plt.plot(ell[lmin:],factor[lmin:]*cl_PCA_HI_1p3deg_pol[ich][lmin:],  label='Need-PCA HI + noise')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim([lmin,lmax_plot])
plt.ylim([-5e-4,6e-3])
plt.legend()
frame1.set_ylabel(r'$  \ell(\ell+1)/2\pi~ C_{\ell} $ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])
#frame1.yaxis.set_major_formatter(formatter) 

diff_cl_1p3deg_pol = cl_PCA_HI_1p3deg_pol/cl_cosmo_HI_1p3deg-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_1p3deg_pol[ich][lmin:]*100,)

frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,lmax_plot])
frame2.set_ylim([-30,2])
frame2.set_ylabel(r'$\Delta$ [%]')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))
plt.savefig(f'Plots_paper/cl_std_need_ch{nu_ch[ich]}_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg3}_nside{nside}_mask0.5.png', bbox_inches='tight')


fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(r'$\theta_{\rm FMWH}=1.3$ deg, '+f'Mean over frequency channels, with pol leakage, '+r'N$_{\rm fg}$='+f'{Nfg}')
plt.plot(ell[lmin:], factor[lmin:]*cl_cosmo_HI_1p3deg.mean(axis=0)[lmin:],'k--',label = f'Cosmo HI + noise')
plt.plot(ell[lmin:], factor[lmin:]*cl_PCA_HI_1p3deg_pol.mean(axis=0)[lmin:], mfc='none', label = f'Need-PCA HI + noise')
plt.xlim([lmin,lmax_plot])
plt.ylim([-5e-4,6e-3])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
frame1.set_ylabel(r'$  \ell(\ell+1)/2\pi~ \langle C_{\ell} \rangle_{\rm ch}$ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])
#frame1.yaxis.set_major_formatter(formatter) 

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_1p3deg_pol.mean(axis=0)[lmin:]*100)
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,lmax_plot])
frame2.set_ylim([-30,2])
frame2.set_ylabel(r'$ \langle \Delta\rangle_{\rm ch}$ [%]')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))
plt.savefig(f'Plots_paper/cl_std_need_mean_ch_{fg_comp}_noise_beam_{beam_s}_jmax{jmax}_Nfg{Nfg3}_nside{nside}_mask0.5.png', bbox_inches='tight')
##############################################################################
fig = plt.figure()
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(r'$\theta_{\rm FMWH}=1.3$ deg, '+f'Mean over frequency channels, with pol leakage, '+r'N$_{\rm fg}$='+f'{Nfg}')
plt.plot(ell[lmin:], factor[lmin:]*cl_cosmo_HI_1p3deg.mean(axis=0)[lmin:],'k--',label = f'Cosmo HI + noise')
plt.plot(ell[lmin:], factor[lmin:]*cl_PCA_HI_1p3deg_pol.mean(axis=0)[lmin:], mfc='none', c=c_pal[0],label = f'Need-PCA HI + noise')
plt.plot(ell[lmin:], factor[lmin:]*cl_standard_PCA_HI_1p3deg_pol_3.mean(axis=0)[lmin:], mfc='none', c=c_pal[1],label = f'PCA HI + noise')
plt.xlim([lmin,lmax_plot])
plt.ylim([-5e-4,6e-3])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
frame1.set_ylabel(r'$  \ell(\ell+1)/2\pi~ \langle C_{\ell} \rangle_{\rm ch}$ [mK$^{2}$]')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(lmin,lmax_plot+1, 10), labels=[])
#frame1.yaxis.set_major_formatter(formatter) 

diff_cl_std_1p3deg_pol = cl_standard_PCA_HI_1p3deg_pol_3/cl_cosmo_HI_1p3deg-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[lmin:], diff_cl_1p3deg_pol.mean(axis=0)[lmin:]*100,c=c_pal[0],label = f'Need-PCA HI + noise')
plt.plot(ell[lmin:], diff_cl_std_1p3deg_pol.mean(axis=0)[lmin:]*100,c=c_pal[1],label = f'Need-PCA HI + noise')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([lmin,lmax_plot])
frame2.set_ylim([-30,2])
frame2.set_ylabel(r'$ \langle \Delta\rangle_{\rm ch}$ [%]')
frame2.set_xlabel(r'$\ell$')
#frame2.yaxis.set_major_formatter(formatter) 
frame2.set_xticks(np.arange(lmin,lmax_plot+1, 10))

#######################################################################################################################
################################## confronto beam #############################################
#diff_cl_need2sphe_1p3deg
#diff_cl_need2sphe 

fig, ax = plt.subplots(1,1)
plt.title(r'$\nu$='+f'{nu_ch[0]} MHz')
ax.plot(ell[lmin:], diff_cl[0][lmin:]*100, label = 'Frequency-dependent beam')
ax.plot(ell[lmin:], diff_cl_1p3deg[0][lmin:]*100, label = r'$\theta_{\rm FMWH}=$ 1.3 deg')
ax.axhline(ls='--', c= 'k', alpha=0.3)
ax.set_xlim([lmin, lmax_plot+1])
ax.set_ylim([-40,2])
ax.set_ylabel(r'$\Delta$ [%]')
ax.set_xlabel(r'$\ell$')
ax.set_xticks(np.arange(lmin,lmax_plot+1, 10))
plt.legend()



fig, ax = plt.subplots(1,1)
plt.title(f'Mean over frequency channels')
ax.plot(ell[lmin:], diff_cl.mean(axis=0)[lmin:]*100, label = 'Frequency-dependent beam')
ax.plot(ell[lmin:], diff_cl_1p3deg.mean(axis=0)[lmin:]*100, label = r'$\theta_{\rm FMWH}=$ 1.3 deg')
ax.axhline(ls='--', c= 'k', alpha=0.3)
ax.set_xlim([lmin, lmax_plot+1])
ax.set_ylim([-40,2])
ax.set_ylabel(r'$\Delta$ [%]')
ax.set_xlabel(r'$\ell$')
ax.set_xticks(np.arange(lmin,lmax_plot+1, 10))
plt.legend()

plt.savefig(f'Plots_paper/diff_beam_need_mean_ch_synch_ff_ps_noise_jmax{jmax}_Nfg3_nside{nside}_mask0.5.png', bbox_inches='tight')

fig, ax = plt.subplots(1,1)
plt.title(f'Mean over frequency channels, with pol leakage')
ax.plot(ell[lmin:], diff_cl_pol_3.mean(axis=0)[lmin:]*100, label = 'Frequency-dependent beam')
ax.plot(ell[lmin:], diff_cl_1p3deg_pol.mean(axis=0)[lmin:]*100, label = r'$\theta_{\rm FMWH}=$ 1.3 deg')
ax.axhline(ls='--', c= 'k', alpha=0.3)
ax.set_xlim([lmin, lmax_plot+1])
ax.set_ylim([-40,2])
ax.set_ylabel(r'$\Delta$ [%]')
ax.set_xlabel(r'$\ell$')
ax.set_xticks(np.arange(lmin,lmax_plot+1, 10))
plt.legend()

plt.savefig(f'Plots_paper/diff_beam_need_mean_ch_synch_ff_ps_pol_noise_jmax{jmax}_Nfg{Nfg}_nside{nside}_mask0.5.png', bbox_inches='tight')


plt.show()