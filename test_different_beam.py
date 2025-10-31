import numpy as np
import healpy as hp 
from astropy.io import fits
import matplotlib.pyplot as plt
from convolution_func import*
import pickle
import seaborn as sns
from Beams import AiryBeam


from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=False, bottom = True)
mpl.rc('ytick', direction='in', right=False, left = True)


#sns.set_theme(style = 'white')
#sns.set_palette('husl',15)
c_pal = sns.color_palette().as_hex()
#######################################################################
###########################################################################
######## Computing beam size using given survey specifics: ################
### initialise a dictionary with the instrument specifications
### for noise and beam calculation
dish_diam_MeerKat = 13.5 #m
dish_diam_SKA = 15 # m
Ndishes_MeerKAT = 64.
Ndishes_SKA = 133.
dish_diam = (Ndishes_MeerKAT*dish_diam_MeerKat+ Ndishes_SKA*dish_diam_SKA)/(Ndishes_MeerKAT+Ndishes_SKA) # m (effective)
#######################################################################
nside=128
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

nu_ch = np.array(file['freq'])
idx_nu_max, = np.where(nu_ch==950.5)[0]
nu_ch = np.array([file['freq'][idx_nu_max]])

del file
#####################
##### Airy beam #####

#thetas = np.linspace(0,theta.max(),len(theta))*np.pi/180.
#print('thetas max',thetas.max())

airy =AiryBeam(nside=nside,dish_diameter=dish_diam,thetamax=theta.max(), nsamples=len(theta))
airy(nu_ch)


beam_airy= airy.data['model'][0]

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


delta_theta=createDtheta(nu_ch,dish_diam,T_p,Amp,smooth)#*np.pi/180.
delta_theta = np.deg2rad(delta_theta)


Amp_rip=0.1
T_p_rip = 20
smooth_rip = True

delta_theta_rip=createDtheta(nu_ch,dish_diam,T_p_rip,Amp_rip,smooth_rip)#*np.pi/180.



beam_cos=np.zeros(len(theta))
beam_gauss=np.zeros(len(theta))
#beam_jinc=np.zeros(npix)


beam_cos_rip=np.zeros(len(theta))
beam_gauss_rip=np.zeros(len(theta))
#beam_jinc_rip=np.zeros(npix)


for p in pixel:
    beam_cos[p]=cos_beam(theta_rad[p], delta_theta)
    beam_gauss[p]=gaussian(theta_rad[p], delta_theta)
    #beam_jinc[p]=jinc(theta[p])

    #beam_cos_rip[p]=cos_beam(thetas[p], delta_theta_rip)
    #beam_gauss_rip[p]=gaussian(thetas[p], delta_theta_rip)
    ##beam_jinc_rip[p]=jinc(theta[p])



fig, ax = plt.subplots(1,1)
plt.suptitle(f'{nu_ch[0]} MHz, FWMH = {delta_theta[0]:0.2f} deg')
ax.plot(theta_rad,10*np.log10(beam_cos/beam_cos[0]),color=c_pal[0], label='cosine')
#ax.plot(theta,10*np.log10(beam_cos_rip/beam_cos_rip[0]),'--', color=c_pal[0], label='cosine w ripples')
ax.plot(theta_rad,10*np.log10(beam_gauss/beam_gauss[0]), color=c_pal[1],label='gauss')
#ax.plot(theta,10*np.log10(beam_gauss_rip/beam_gauss_rip[0]), '--', color=c_pal[1],label='gauss w ripples')
ax.plot(theta_airy,10*np.log10(beam_airy/beam_airy[0]), color=c_pal[2],label='airy')

xticks  = np.arange(0,np.deg2rad(10), 0.05)
xlabels = np.linspace(0,10, )
ax.set_xticks(ticks=xticks)

ax.set_xlim([0, np.deg2rad(10)])
ax.set_ylim([-50,0])

ax.set_xlabel(r'$\theta$ ')
ax.set_ylabel('Normalized Beam [dB]')

plt.legend()
plt.savefig('plots_different_beams.png')


#### proviamo in altro modo

strip= hp.query_strip(nside = nside, theta1=np.pi/2, theta2 = np.pi/2+np.sqrt(hp.nside2pixarea(nside))/2, inclusive=True)
#strip = hp.query_strip(nside, theta1= np.pi/2-np.pi/50, theta2=np.pi/2+np.pi/50)
test_map = np.zeros(hp.nside2npix(nside))
test_map[strip]=1
#test_map_s = hp.smoothing(test_map, fwhm=delta_theta)
#hp.mollview(test_map_s, min=-1e-7,max=0.05,cmap='gray')

bl_beam_cos, delta_theta=create_bl_vec(beam='cosine', nside=nside,dish_diameter=dish_diam, T_p=T_p, Amp=Amp, smooth=smooth, ch_nu=nu_ch)

saved_beam_cos = np.loadtxt('test_beam_cosine.dat')

fig, ax = plt.subplots(1,1)
plt.suptitle(f'{nu_ch[0]} MHz, FWMH = {delta_theta[0]:0.2f} deg')
ax.plot(theta_rad,beam_cos/saved_beam_cos-1,color=c_pal[0], label='cosine')
#ax.plot(theta_rad,10*np.log10(saved_beam_cos/saved_beam_cos[0]),color=c_pal[1], label='saved cosine')

xticks  = np.arange(0,np.deg2rad(10), 0.05)
xlabels = np.linspace(0,10, )
ax.set_xticks(ticks=xticks)

ax.set_xlim([0, np.deg2rad(10)])
ax.set_ylim([-10,10])

ax.set_xlabel(r'$\theta$ ')
ax.set_ylabel('Normalized Beam [dB]')


map_convolved = single_convolution_bl(test_map,bl_beam_cos[0], nside )

hp.mollview(map_convolved, cmap='gray', title=f'map convolved, {nu_ch} MHz')


############################################################


#ipix = np.arange(0,1000)
#
#fig = plt.figure()
#plt.plot(ipix*np.sqrt(hp.nside2pixarea(nside)),map_convolved[strip][0:1000])
#plt.xlabel(r'$\theta$')

plt.show()