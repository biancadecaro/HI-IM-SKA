""""
Questo script prende le mappe simulate di Isabella Carucci a nside=256, fa il merging tra le mappe per
ridurre il numero di canali (fa na semplice media tra le mappe in uno stesso canale) con Dnu =10
Le mappe finali saranno così composte:
'maps_sims_tot' = HI+gal_sync+gal_ff+ps
'maps_sims_HI' = HI
'maps_sims_fg' = gal_sync+gal_ff+ps
Tutte le mappe hanno media (su tutti i pixel) = 0
"""
import h5py
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

c_light = 3.0*1e8  # m/s

def merging_maps(nu_ch_in,nu_ch_out,maps_in,deltanu_out):
	
	deltanu_in = abs(nu_ch_in[-1]-nu_ch_in[-2])
	maps_out  = [0] * len(nu_ch_out)  

	deltanu_out = abs(nu_ch_out[-1]-nu_ch_out[-2])
	N = int(deltanu_out/deltanu_in)
	if (deltanu_in*N)!=deltanu_out:
		print('just dnu multiples!')
		exit		

	for i in range(len(nu_ch_out)):
		maps_out[i] = sum(maps_in[N*i:N*i+N]) / N
		
	return maps_out

def nu_ch_f(nu_ch_in,dnu_out):
	du_in = abs(nu_ch_in[-1]-nu_ch_in[-2])
	a1 = nu_ch_in[0] - du_in/2; a2 = nu_ch_in[-1] + du_in/2
	M = int((a2-a1)/dnu_out)
	if (dnu_out*M)!=(a2-a1):
		print('just dnu multiples!')
		exit
	nu_ch_out = np.linspace(a1+dnu_out/2,a2-dnu_out/2,M)	

	return nu_ch_out

## from vector to matrix and viceversa
def alm2tab(alm,lmax):

	size = np.size(alm)
	tab  = np.zeros((lmax+1,lmax+1,2))

	for r in range(0,size):
		l,m = hp.sphtfunc.Alm.getlm(lmax,r)
		tab[l,m,0] = np.real(alm[r])
		tab[l,m,1] = np.imag(alm[r])

	return tab

def tab2alm(tab):

	lmax = int(np.shape(tab)[0])-1
	taille = int(lmax*(lmax+3)/2)+1
	alm = np.zeros((taille,),dtype=complex)

	for r in range(0,taille):
		l,m = hp.sphtfunc.Alm.getlm(lmax,r)
		alm[r] = complex(tab[l,m,0],tab[l,m,1])

	return alm


## getting the spherical harmonic coefficients
## from a map
def almtrans(map_in,lmax=None):

	if lmax==None:
		lmax = 3.*hp.get_nside(map_in)
		print("lmax = ",lmax)

	alm = hp.sphtfunc.map2alm(map_in,lmax=lmax)

	tab = alm2tab(alm,lmax)

	return tab


## convolution:
## multiplying the spherical harmonic coefficients
def alm_product(tab,beam_l):
	length=np.size(beam_l)
	lmax = np.shape(tab)[0]

	if lmax > length:
		print("Filter length is too small")

	for r in range(lmax):
		tab[r,:,:] = beam_l[r]*tab[r,:,:]

	return tab


## from a_lm back to map
def almrec(tab,nside):

	alm = tab2alm(tab)
	map_out = hp.alm2map(alm,nside,verbose=False)

	return map_out

def convolve(map_in,beam_l,lmax):
	alm = almtrans(map_in,lmax=lmax)
	tab = alm_product(alm,beam_l)
	m = almrec(tab,nside=hp.get_nside(map_in))
	return m


## angle in radians of the FWHM
def theta_FWHM(nu,dish_diam): # nu in MHz, dish_diam in m
	return c_light*1e-6/nu/float(dish_diam) # rad

## solid angle of beam in steradian 
def Omega_beam(nu,dish_diam): # nu in MHz, dish_diam in m 
	return np.pi/(4.*np.log(2))*theta_FWHM(nu,dish_diam)**2

## how many beams to cover my survey area (fraction of sky)
def N_beams(f_sky,nu,dish_diam): # nu in MHz, dish_diam in m 
	return 4*np.pi*f_sky/Omega_beam(nu,dish_diam)


####THERMAL NOISE#####

def T_sky(nu): # K
	return 60.*(300./nu)**2.55  # K

def T_rcvr(nu,T_inst): # K
	temp_sky = T_sky(nu)
	return 0.1* temp_sky + T_inst

def T_sys(nu,T_inst): # K
	return T_rcvr(nu,T_inst) + T_sky(nu)

## final sigma in mK 
def sigma_N(nu,dnu,T_inst,f_sky,t_obs,Ndishes,dish_diam):
	t_obs = t_obs * 3600 # hrs to s
	dnu = dnu * 1.e6 # MHz to Hz

	temp_sys = T_sys(nu,T_inst)  # in K
	A = np.sqrt(N_beams(f_sky,nu,dish_diam)/dnu/t_obs/Ndishes)
	
	return temp_sys * A *1e3  # mK

def noise_map(sigma,nside=512):
	npixels = hp.nside2npix(nside)
	m = np.random.normal(0.0, sigma, npixels)
	return m
######################################################


nside_out= 128

path_data = 'sim_PL05_from191030.hd5'
file = h5py.File(path_data,'r')
nu_ch = np.array(file['frequencies'])


file_new={}
file_ud={}

delta_nu_out = 10
file_new['frequencies'] = nu_ch_f(nu_ch,delta_nu_out)#np.array([nu_ch[i*delta_nu] for i in range(0,int(len(nu_ch)/delta_nu))])
file_ud['frequencies'] = file_new['frequencies']

components = list(file.keys())
print(components)
components.remove('frequencies')
components.remove('pol_leakage')

#components.remove('gal_synch')
#components.remove('gal_ff')#
#components.remove('point_sources')

fg_comp = 'synch_ff_ps'


for c in components:
  print(c)
  file_new[c]=merging_maps(nu_ch,file_new['frequencies'],file[c], delta_nu_out )
  file_ud[c] = hp.pixelfunc.ud_grade(map_in=file_new[c], nside_out=128)

print(len(file_ud['frequencies']), hp.get_nside(file_ud['cosmological_signal'][1]))

del file; del file_new

nu_ch_new = np.array(file_ud['frequencies'])
num_freq_new=len(nu_ch_new)
npix = np.shape(file_ud['cosmological_signal'])[1]
lmax = 3*nside_out-1

obs_maps = np.zeros((num_freq_new,npix))
fg_maps = np.zeros((num_freq_new,npix))

for c in components:
    print(c)
    obs_maps += np.array(file_ud[c])

for cc in components:
    if cc=='cosmological_signal':
      continue
    print(cc)
    fg_maps += np.array(file_ud[cc])

obs_maps_no_mean = np.array([obs_maps[i] -np.mean(obs_maps[i],axis=0)  for i in range(num_freq_new)])
HI_maps_no_mean = np.array([file_ud['cosmological_signal'][i] -np.mean(file_ud['cosmological_signal'][i],axis=0) for i in range(num_freq_new)])
fg_maps_no_mean = np.array([fg_maps[i] -np.mean(fg_maps[i],axis=0) for i in range(num_freq_new)]) 
#


file_sims_no_mean = {}
file_sims_no_mean['freq'] = nu_ch_new
file_sims_no_mean['maps_sims_tot'] = obs_maps_no_mean
file_sims_no_mean['maps_sims_fg'] = fg_maps_no_mean
file_sims_no_mean['maps_sims_HI'] = HI_maps_no_mean

del obs_maps_no_mean; del fg_maps_no_mean; del HI_maps_no_mean


for nu in range(num_freq_new):
		alm_HI = hp.map2alm(file_sims_no_mean['maps_sims_HI'][nu], lmax=lmax)
		file_sims_no_mean['maps_sims_HI'][nu] = hp.alm2map(alm_HI, lmax=lmax, nside = nside_out)
		file_sims_no_mean['maps_sims_HI'][nu] = hp.remove_dipole(file_sims_no_mean['maps_sims_HI'][nu])
		del alm_HI
		alm_fg = hp.map2alm(file_sims_no_mean['maps_sims_fg'][nu], lmax=lmax)
		file_sims_no_mean['maps_sims_fg'][nu] = hp.alm2map(alm_fg, lmax=lmax, nside = nside_out)
		file_sims_no_mean['maps_sims_fg'][nu] = hp.remove_dipole(file_sims_no_mean['maps_sims_fg'][nu])
		del alm_fg
		alm_obs = hp.map2alm(file_sims_no_mean['maps_sims_tot'][nu], lmax=lmax)
		file_sims_no_mean['maps_sims_tot'][nu] = hp.alm2map(alm_obs, lmax=lmax, nside = nside_out)
		file_sims_no_mean['maps_sims_tot'][nu] = hp.remove_dipole(file_sims_no_mean['maps_sims_tot'][nu])
		del alm_obs

ich =int(num_freq_new/2)

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'No mean, channel {ich}: {nu_ch_new[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(file_sims_no_mean['maps_sims_tot'][ich], cmap='viridis',title=f'Observations, freq={nu_ch_new[ich]}',hold=True)
fig.add_subplot(222) 
hp.mollview(file_sims_no_mean['maps_sims_HI'][ich], cmap='viridis',title=f'HI signal, freq={nu_ch_new[ich]}',min=0, max=1,hold=True)
fig.add_subplot(223)
hp.mollview(file_sims_no_mean['maps_sims_fg'][ich],title=f'Foregrounds, freq={nu_ch_new[ich]}',cmap='viridis', hold=True)
#plt.savefig('plots_PCA/maps_no_mean_fg_HI_obs_input.png')
plt.show()

#filename = f'Sims/no_mean_sims_{fg_comp}_{len(nu_ch_new)}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside_out}'
#with open(filename+'.pkl', 'wb') as f:
#    pickle.dump(file_sims_no_mean, f)
#    f.close()


###########################################################################
######## Computing beam size using given survey specifics: ################
### initialise a dictionary with the instrument specifications
### for noise and beam calculation
dish_diam = 13.5  # m
T_inst    = 20.0  # K
f_sky     = 0.1   # Survey area (sky fraction)
t_obs     = 4000. # hrs, observing time
Ndishes   = 64.   # number of dishes
specs_dict = {'dish_diam': dish_diam, 'T_inst': T_inst,
			  'f_sky': f_sky, 't_obs': t_obs, 'Ndishes' : Ndishes}

theta_FWMH_max = c_light*1e-6/np.min(nu_ch_new)/float(dish_diam) #radians
theta_FWMH = c_light*1e-6/nu_ch_new/float(dish_diam) #radians

print()

#beam_worst = hp.gauss_beam(theta_FWMH_max, lmax=3*nside)


################################## NOISE ################################################
dnu = nu_ch_new[1]-nu_ch_new[0]

sigma_noise = sigma_N(nu_ch_new,dnu,**specs_dict)

noise = [noise_map(sigma,nside=nside_out) for sigma in sigma_noise]
del sigma_noise

#########################################################################################

theta_arcmin = 40 #arcmin
theta_FWMH = theta_arcmin*np.pi/(60*180)
beam = hp.gauss_beam(theta_FWMH, lmax=3*nside_out-1)
lmax_fwmh = int(np.pi/theta_FWMH) 
print(f'theta_FWMH : {theta_FWMH} rad,  {theta_FWMH*180./np.pi} deg')
print(f'ell_beam = {lmax_fwmh}')



file_sims_beam = {}
file_sims_beam['freq'] = nu_ch_new
file_sims_beam['maps_sims_tot'] =  np.array([convolve(file_sims_no_mean['maps_sims_tot'][i],beam, lmax=lmax) for i in range(num_freq_new)])

file_sims_beam['maps_sims_fg'] = np.array([convolve(file_sims_no_mean['maps_sims_fg'][i],beam, lmax=lmax) for i in range(num_freq_new)])

file_sims_beam['maps_sims_HI'] = np.array([convolve(file_sims_no_mean['maps_sims_HI'][i],beam, lmax=lmax) for i in range(num_freq_new)])

file_sims_beam['maps_sims_noise'] = np.array([convolve(noise[i],beam, lmax=lmax) for i in range(num_freq_new)])
#del temp_noise



for nu in range(num_freq_new):
		alm_HI = hp.map2alm(file_sims_beam['maps_sims_HI'][nu], lmax=lmax)
		file_sims_beam['maps_sims_HI'][nu] = hp.alm2map(alm_HI, lmax=lmax, nside = nside_out)
		file_sims_beam['maps_sims_HI'][nu] = hp.remove_dipole(file_sims_beam['maps_sims_HI'][nu])
		del alm_HI
		alm_fg = hp.map2alm(file_sims_beam['maps_sims_fg'][nu], lmax=lmax)
		file_sims_beam['maps_sims_fg'][nu] = hp.alm2map(alm_fg, lmax=lmax, nside = nside_out)
		file_sims_beam['maps_sims_fg'][nu] = hp.remove_dipole(file_sims_beam['maps_sims_fg'][nu])
		del alm_fg
		alm_obs = hp.map2alm(file_sims_beam['maps_sims_tot'][nu], lmax=lmax)
		file_sims_beam['maps_sims_tot'][nu] = hp.alm2map(alm_obs, lmax=lmax, nside = nside_out)
		file_sims_beam['maps_sims_tot'][nu] = hp.remove_dipole(file_sims_beam['maps_sims_tot'][nu])
		del alm_obs
		alm_noise = hp.map2alm(file_sims_beam['maps_sims_noise'][nu], lmax=lmax)
		file_sims_beam['maps_sims_noise'][nu] = hp.alm2map(alm_noise, lmax=lmax, nside = nside_out)
		file_sims_beam['maps_sims_noise'][nu] = hp.remove_dipole(file_sims_beam['maps_sims_noise'][nu])
		del alm_noise


import pickle
filename = f'Sims/beam_theta40arcmin_no_mean_sims_{fg_comp}_noise_{len(nu_ch_new)}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_thick{delta_nu_out}MHz_lmax{lmax}_nside{nside_out}'
with open(filename+'.pkl', 'wb') as ff:
	pickle.dump(file_sims_beam, ff)
	ff.close()





fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'BEAM, channel {ich}: {nu_ch_new[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(file_sims_beam['maps_sims_tot'][ich], cmap='viridis',title=f'Observations, freq={nu_ch_new[ich]}', hold=True)
fig.add_subplot(222) 
hp.mollview(file_sims_beam['maps_sims_HI'][ich], cmap='viridis',title=f'HI signal, freq={nu_ch_new[ich]}',min=0, max=1,hold=True)
fig.add_subplot(223)
hp.mollview(file_sims_beam['maps_sims_fg'][ich],title=f'Foregrounds, freq={nu_ch_new[ich]}',cmap='viridis',hold=True)
fig.add_subplot(224)
hp.mollview(file_sims_beam['maps_sims_noise'][ich],title=f'Noise, freq={nu_ch_new[ich]}',cmap='viridis',hold=True)

del file_sims_beam

plt.show()