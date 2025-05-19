import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

####################################
nside=256
npix = hp.nside2npix(nside)


pix_mask = hp.query_strip(nside, theta1=np.pi*2/3, theta2=np.pi/3)

mask_50 = np.zeros(npix)
mask_50[pix_mask] =1
fsky_50 = np.sum(mask_50)/hp.nside2npix(nside)

fig=plt.figure()
hp.mollview(mask_50, cmap='viridis', title=f'fsky={fsky_50:0.2f}', hold=True)


###############################################################
pix_mask_1 = hp.query_strip(nside, theta2=np.pi*2/3, theta1=np.pi/3)

mask_50_1 = np.ones(npix)
mask_50_1[pix_mask_1] =0
fsky_50_1 = np.mean(mask_50_1)#np.sum(mask_50_1)/hp.nside2npix(nside)

fig=plt.figure()
hp.mollview(mask_50_1, cmap='viridis', title=f'fsky1={fsky_50_1:0.2f}', hold=True)
plt.show()

theta,phi = hp.pix2ang(nside,pix_mask_1, lonlat=True)

print(theta, phi)

hp.mollview(title='')
hp.graticule()

hp.projscatter(theta, phi, lonlat=True, coord='G')
hp.projtext(0,phi[np.where(phi==max(phi))][0], f'phi_1={phi[np.where(phi==max(phi))][0]:0.2f}', lonlat=True, coord='G')
hp.projtext(0,phi[np.where(phi==min(phi))][0], f'phi_2={phi[np.where(phi==min(phi))][0]:0.2f}', lonlat=True, coord='G')
plt.show()
