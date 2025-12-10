import numpy as np
import healpy as hp 
from astropy.io import fits
import matplotlib.pyplot as plt
from convolution_func import*
import pickle
import seaborn as sns
from Beams import AiryBeam


import matplotlib as mpl
mpl.rc('xtick', direction='in', top=False, bottom = True)
mpl.rc('ytick', direction='in', right=False, left = True)

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

c_pal = sns.color_palette().as_hex()

#######################################################################
###########################################################################
######## Computing beam size using given survey specifics: ################
### initialise a dictionary with the instrument specifications
### for noise and beam calculation
c_light=3.0*1e8 #m/s
dish_diam_MeerKat = 13.5 #m
dish_diam_SKA = 15 # m
Ndishes_MeerKAT = 64.
Ndishes_SKA = 133.
dish_diam = (Ndishes_MeerKAT*dish_diam_MeerKat+ Ndishes_SKA*dish_diam_SKA)/(Ndishes_MeerKAT+Ndishes_SKA) # m (effective)
#######################################################################
nside=128
lmax=3*nside-1
npix = hp.nside2npix(nside)
pixel=np.arange(npix)
theta_rad = hp.pix2ang(nside,pixel)[0]
phi_rad = hp.pix2ang(nside,pixel)[1]
theta=np.degrees(theta_rad)
phi=np.degrees(phi_rad)

pix_c=hp.ang2pix(nside, theta=np.pi/2, phi=0)

path_data = 'Sims/nuovo_sims_synch_ff_ps_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax383_nside128'
with open(path_data+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

nu_ch_array = np.array(file['freq'])
idx_nu_max, = np.where(nu_ch_array==950.5)[0]
nu_ch = np.array([file['freq'][idx_nu_max]])
num_ch=len(nu_ch_array)
nu_ch_max = nu_ch_array[-1]
#####################
##### Airy beam #####

#thetas = np.linspace(0,theta.max(),len(theta))*np.pi/180.
#print('thetas max',thetas.max())

airy =AiryBeam(nside=nside,dish_diameter=dish_diam,thetamax=theta.max(), nsamples=len(theta))
airy(nu_ch)


#beam_airy= airy.data['model'][0]

theta_airy = np.deg2rad(airy.data['theta'])


print(theta[0], theta_airy[0])

#x = np.cos(theta_rad)
#y = np.sin(theta_rad)
#
#x_airy = np.cos(theta_airy)
#y_airy = np.sin(theta_airy)
#
#
#X,Y = np.meshgrid(x,y)
#r = X**2 + Y**2
#X_airy,Y_airy = np.meshgrid(x_airy,y_airy)
#r_airy = X_airy**2 + Y_airy**2
#
#fig= plt.figure()
#ax=fig.subplots()
#plt.contour(X,Y,r, [1])
#plt.contour(X_airy,Y_airy,r_airy, [1], linestyles='dotted')
#
#ticks =  np.linspace(x.min(), x.max(), 5)
#yticks =  np.linspace(y.min(), y.max(), 5)
#labels =  np.linspace(np.arccos(x).min(), np.arccos(x).max(), 5)/np.pi
#major_labels = [r'%0.1f $\pi$'% i for i in labels]
#print(labels.shape)
#ax.set_xticks(ticks,labels=major_labels )
#ax.set_yticks(yticks,labels=major_labels)


#ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_two_digits))
#ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_two_digits))

#plt.plot(r_airy, 'o')

phi_plot = np.ones(theta.shape)*0

fig = plt.figure()
#ax.axis('off')
hp.mollview(title="Galactic map of Test Fields", fig=fig)
hp.graticule()
hp.projscatter(theta_rad, phi_rad, s=1, color='k', label='healpy')
hp.projscatter(theta_rad,phi_plot, s=2,label='cosine')
#hp.projscatter(theta_airy,phi_plot, label='airy')#np.pi/2+np.rad2deg(theta_airy)
fig.legend()
plt.close('all')

#import sys
#sys.exit()
############################
Amp=0.0
T_p = 20
smooth = False

delta_theta_array = np.array([createDtheta(nu_ch_array[i],dish_diam,T_p,Amp,smooth) for i in range(num_ch)])
delta_theta_array =  np.deg2rad(delta_theta_array)

delta_theta=createDtheta(nu_ch,dish_diam,T_p,Amp,smooth)#*np.pi/180.
delta_theta = np.deg2rad(delta_theta)

delta_theta_min=createDtheta(nu_ch_max,dish_diam,T_p,Amp,smooth)#*np.pi/180.
delta_theta_min = np.deg2rad(delta_theta_min)
print(delta_theta_min*180./np.pi, delta_theta*180./np.pi)


Amp_rip=0.1
T_p_rip = 20
smooth_rip = True

delta_theta_rip=createDtheta(nu_ch,dish_diam,T_p_rip,Amp_rip,smooth_rip)#*np.pi/180.
delta_theta_rip = np.deg2rad(delta_theta_rip)


beam_cos=np.zeros(len(theta))
beam_gauss=np.zeros(len(theta))
beam_gauss_min=np.zeros(len(theta))
#beam_jinc=np.zeros(npix)


beam_cos_rip=np.zeros(len(theta))
beam_gauss_rip=np.zeros(len(theta))
beam_gauss_test=np.zeros(len(theta))


for p in pixel:
    beam_cos[p]=cos_beam(theta_airy[p], delta_theta)
    beam_gauss[p]=gaussian(theta_airy[p], delta_theta)
    beam_gauss_min[p]=gaussian(theta_airy[p], delta_theta_min)
    beam_gauss_test[p]=gaussian(theta_airy[p], 3*delta_theta)
    #beam_jinc[p]=jinc(theta[p])

    beam_cos_rip[p]=cos_beam(theta_airy[p], delta_theta_rip)
    beam_gauss_rip[p]=gaussian(theta_airy[p], delta_theta_rip)
    ##beam_jinc_rip[p]=jinc(theta[p])



fig, ax = plt.subplots(1,1)
plt.suptitle(f'{nu_ch} MHz, FWMH = {np.rad2deg(delta_theta[0]):0.2f} deg', fontsize=20)
ax.plot(airy.data['theta'],10*np.log10(beam_cos_rip/beam_cos_rip[0]),color=c_pal[0], label='Cosine beam, ripple corrections')
#ax.plot(theta,10*np.log10(beam_cos_rip/beam_cos_rip[0]),'--', color=c_pal[0], label='cosine w ripples')
ax.plot(airy.data['theta'],10*np.log10(beam_gauss/beam_gauss[0]), color=c_pal[1],label='Gaussian beam')
ax.plot(airy.data['theta'],10*np.log10(beam_gauss_min/beam_gauss_min[0]), color='k', ls='--',label='Gaussian beam min')
ax.plot(airy.data['theta'],10*np.log10(beam_gauss_test/beam_gauss_test[0]), color='k', ls='-.',label='Gaussian beam test')
#ax.plot(theta,10*np.log10(beam_gauss_rip/beam_gauss_rip[0]), '--', color=c_pal[1],label='gauss w ripples')
#ax.plot(theta_airy,10*np.log10(beam_airy/beam_airy[0]), color=c_pal[2],label='airy')

xticks  = np.linspace(0.0,10, 5)
xlabels = np.linspace(0.1,10,5 )
ax.set_xticks(ticks=xticks)

ax.set_xlim([0, 10])
ax.set_ylim([-50,0])

ax.set_xlabel(r'$\theta$ ')
ax.set_ylabel('Normalized Beam [dB]')

plt.legend()
plt.savefig('plots_different_beams.png')


#### proviamo in altro modo
print(10*np.log10(beam_gauss/beam_gauss[0]))

#hp.cartview(10*np.log10(beam_gauss/beam_gauss[0]),cmap='viridis')
#plt.show()

#bl2beam_gauss = hp.anafast(beam_gauss)
#bl2beam_gauss_min = hp.anafast(beam_gauss_min)
#bl2beam_gauss_test = hp.anafast(beam_gauss_test)
#bl2beam_cosine_rip=hp.anafast(beam_cos_rip)
#
#fig,ax1 = plt.subplots(1,1)
#ax1.set_title('beam gauss')
#ax1.plot(bl2beam_gauss, label=f'gauss,theta={delta_theta[0]*180/np.pi:1.2f} deg', c=c_pal[0])
#ax1.plot(bl2beam_cosine_rip, label=f'cosine, theta={delta_theta[0]*180/np.pi:1.2f} deg', c=c_pal[1])
#ax1.plot(bl2beam_gauss_min, label=f'gauss, theta={delta_theta_min*180/np.pi:1.2f} deg', c='k', ls='--')
#ax1.plot(bl2beam_gauss_test, label=f'gauss, theta={3*delta_theta[0]*180/np.pi:1.2f} deg', c='k', ls='-.')
#ax1.yaxis.set_major_formatter(formatter) 
#ax1.set_ylim([0,1.1])
#ax1.set_xlabel('ell')
#ax1.set_ylabel('b_ell')
#ax1.legend(fontsize=15)

#strip= hp.query_strip(nside = nside, theta1=np.pi/2, theta2 = np.pi/2+np.sqrt(hp.nside2pixarea(nside))/2, inclusive=True)
##strip = hp.query_strip(nside, theta1= np.pi/2-np.pi/50, theta2=np.pi/2+np.pi/50)
#test_map = np.zeros(hp.nside2npix(nside))
#test_map[strip]=1
#test_map_s = hp.smoothing(test_map, fwhm=delta_theta)
#hp.mollview(test_map_s, min=-1e-7,max=0.05,cmap='gray')

#bl_beam_cos, delta_theta=create_bl_vec(beam='cosine', nside=nside,dish_diameter=dish_diam, T_p=T_p, Amp=Amp, smooth=smooth, ch_nu=nu_ch)

#saved_beam_cos = np.loadtxt('test_beam_cosine.dat')
#
#fig, ax = plt.subplots(1,1)
#plt.suptitle(f'{nu_ch} MHz, FWMH = {delta_theta[0]:0.2f} deg')
#ax.plot(theta_rad,beam_cos/saved_beam_cos-1,color=c_pal[0], label='cosine')
##ax.plot(theta_rad,10*np.log10(saved_beam_cos/saved_beam_cos[0]),color=c_pal[1], label='saved cosine')
#
#xticks  = np.arange(0,np.deg2rad(10), 0.05)
#xlabels = np.linspace(0,10, )
#ax.set_xticks(ticks=xticks)
#
#ax.set_xlim([0, np.deg2rad(10)])
#ax.set_ylim([-10,10])
#
#ax.set_xlabel(r'$\theta$ ')
#ax.set_ylabel('Normalized Beam [dB]')


#map_convolved = single_convolution_bl(test_map,bl_beam_cos[0], nside )
#
#hp.mollview(map_convolved, cmap='gray', title=f'map convolved, {nu_ch} MHz')


############################################################
nu_ch = file['freq']
num_ch=len(nu_ch)
print(num_ch)
theta_FWMH_max = c_light*1e-6/np.min(nu_ch)/float(dish_diam) #radians
theta_FWMH = c_light*1e-6/nu_ch/float(dish_diam) #radians

Amp=0.1
T_p = 20
smooth = True

print()
#####################################################################################
def beam_gauss_func(theta, lmax):
       ell = np.arange(lmax+1)
       sigma =theta/np.sqrt(8*np.log(2))
       gauss = np.exp(-0.5*ell*(ell+1)*np.power(sigma,2))#(ell*(ell+1))
       return gauss
gauss_func_test= beam_gauss_func(theta=3*max(delta_theta_array), lmax=lmax)

bl_beam_cos, delta_theta=create_bl_vec(beam='cosine', nside=nside,dish_diameter=dish_diam, T_p=T_p, Amp=Amp, smooth=smooth, ch_nu=nu_ch)
bl_gauss =np.array( [hp.gauss_beam(delta_theta_array[i], lmax=lmax) for i in range(num_ch)])
gauss_func_min = beam_gauss_func(theta=delta_theta_min, lmax=lmax)
gauss_func_max = beam_gauss_func(theta=max(delta_theta_array), lmax=lmax)
#gauss_func_test= beam_gauss_func(theta=3*max(theta_FWMH), lmax=lmax)

bl_beam_cos=bl_beam_cos/np.max(bl_beam_cos)

fig,(ax1,ax2)= plt.subplots(1,2, figsize=(13,7), sharey=True)

ax1.set_title('beam gauss')
for n in range(0, num_ch, 20):
	ax1.plot(bl_gauss[n], label=f'theta={theta_FWMH[n]*180/np.pi:1.2f} deg')
ax1.yaxis.set_major_formatter(formatter) 
ax1.set_ylim([0,1.1])
ax1.set_xlabel('ell')
ax1.set_ylabel('b_ell')
ax1.legend(fontsize=15)

ax2.set_title('beam cosine w ripples')
for n in range(0, num_ch, 20):
	ax2.plot(bl_beam_cos[n]/max(bl_beam_cos[n]), label=f'theta={delta_theta[n]*180/np.pi:1.2f} deg')
ax2.yaxis.set_major_formatter(formatter) 
ax2.set_ylim([0,1.1])
ax2.set_xlabel('ell')
#ax2.set_ylabel('b_ell')
ax2.legend(fontsize=15)

##########################################################

fig,ax1 = plt.subplots(1,1)
ax1.set_title('beam gauss')
ax1.plot(bl_gauss[0], label=f'theta={delta_theta_array[0]*180/np.pi:1.2f} deg', c=c_pal[0])
ax1.plot(gauss_func_max, label=f'mio, theta={max(delta_theta_array)*180/np.pi:1.2f} deg', c='grey', ls='--')
ax1.plot(bl_gauss[-1], label=f'theta={delta_theta_array[-1]*180/np.pi:1.2f} deg', c=c_pal[1])
ax1.plot(gauss_func_min, label=f'mio, theta={delta_theta_min*180/np.pi:1.2f} deg', c='k', ls='--')

ax1.plot(gauss_func_test, label=f'test, theta={3*max(delta_theta_array)*180/np.pi:1.2f} deg', c='k', ls='-.')
ax1.yaxis.set_major_formatter(formatter) 
ax1.set_ylim([0,1.1])
ax1.set_xlabel('ell')
ax1.set_ylabel('b_ell')
ax1.legend(fontsize=15)

##########################################################
plt.show()