import h5py
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 


mpl.rcParams['font.size']=18
##########################################################
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
	map_out = hp.alm2map(alm,nside)

	return map_out

def convolve(map_in,beam_l,lmax):
	alm = almtrans(map_in,lmax=lmax)
	tab = alm_product(alm,beam_l)
	m = almrec(tab,nside=hp.get_nside(map_in))
	return m


## angle in radians of the FWHM
def theta_FWHM(nu,dish_diam): # nu in MHz, dish_diam in m
	return c_light*1e-6/nu/float(dish_diam) # rad  #questa ok 

## solid angle of beam in steradian 
def Omega_beam(nu,dish_diam): # nu in MHz, dish_diam in m 
	return np.pi/(4.*np.log(2))*theta_FWHM(nu,dish_diam)**2 #questa ok, theta in rad

## how many beams to cover my survey area (fraction of sky)
def N_beams(Omega_sur,nu,dish_diam): # nu in MHz, dish_diam in m , Omega_sur in deg2
	return Omega_sur*(np.pi/180.)**2/Omega_beam(nu,dish_diam)


####THERMAL NOISE#####
# SKA Cosmology Redbook pag 3 eq 1, 2
def T_sky(nu): # K
	return 25*(408./nu)**2.75  # K, nu [MHz]

def T_rcvr(nu):#,T_inst): # K
	return 15.+30.*(nu/1e3-0.75)**2 # K, nu [MHz]

def T_sys(nu): # K
	T_cmb = 2.73 # K 
	T_spill = 3 # K
	return T_rcvr(nu) + T_sky(nu) + T_cmb + T_spill

## final sigma in mK 
def sigma_N(nu,dnu,Omega_sur,t_obs,Ndishes,dish_diam):
	t_obs = t_obs * 3600 # hrs to s
	dnu = dnu * 1.e6 # MHz to Hz

	temp_sys = T_sys(nu)  # in K
	A = np.sqrt(N_beams(Omega_sur,nu,dish_diam)/dnu/t_obs/Ndishes)
	
	return temp_sys * A *1e3  # mK

def noise_map(sigma,nside=512):
	npixels = hp.nside2npix(nside)
	seed = 3423232
	np.random.seed(seed)
	m = np.random.normal(0.0, sigma, npixels)
	return m

############################################################################################
path_data = 'sim_PL05_from191030.hd5'
file = h5py.File(path_data,'r')
nu_ch = np.array(file['frequencies'])
idx_nu_max, = np.where(nu_ch==1005.5)[0]
print(idx_nu_max)

file_new={}


file_new['frequencies'] = nu_ch[:idx_nu_max]

print(file_new['frequencies'], len(file_new['frequencies']))

components = list(file.keys())
print(components)
components.remove('frequencies')
#components.remove('pol_leakage')


fg_comp = 'synch_ff_ps'

if 'pol_leakage' in components:
	fg_comp = 'synch_ff_ps_pol'


print(fg_comp)
for c in components:
  print(c)
  file_new[c]=file[c][:idx_nu_max]

del file

print(len(file_new['frequencies']), hp.get_nside(file_new['cosmological_signal'][1]))
 
nside = hp.get_nside(file_new['cosmological_signal'][1])

nu_ch_new = np.array(file_new['frequencies'])
num_freq_new=len(nu_ch_new)
npix = np.shape(file_new['cosmological_signal'])[1]

ich = int(num_freq_new/2)
print(ich, nu_ch[ich])

lmax=3*nside-1


#synch_maps_no_mean = np.array([file_new['gal_synch'][i] -np.mean(file_new['gal_synch'][i],axis=0)  for i in range(num_freq_new)])
#ff_maps_no_mean = np.array([file_new['gal_ff'][i] -np.mean(file_new['gal_ff'][i],axis=0) for i in range(num_freq_new)])
#ps_maps_no_mean = np.array([file_new['point_sources'][i] -np.mean(file_new['point_sources'][i],axis=0) for i in range(num_freq_new)]) 
#HI_maps_no_mean = np.array([file_new['cosmological_signal'][i] -np.mean(file_new['cosmological_signal'][i],axis=0) for i in range(num_freq_new)]) 
#pl_maps_no_mean = np.array([file_new['pol_leakage'][i] -np.mean(file_new['pol_leakage'][i],axis=0) for i in range(num_freq_new)]) 
#
#
#
#for nu in range(num_freq_new):
#		alm_synch = hp.map2alm(synch_maps_no_mean[nu], lmax=lmax)
#		synch_maps_no_mean[nu] = hp.alm2map(alm_synch, lmax=lmax, nside = nside)
#		synch_maps_no_mean[nu] = hp.remove_dipole(synch_maps_no_mean[nu])
#		del alm_synch
#		
#		alm_ff = hp.map2alm(ff_maps_no_mean[nu], lmax=lmax)
#		ff_maps_no_mean[nu] = hp.alm2map(alm_ff, lmax=lmax, nside = nside)
#		ff_maps_no_mean[nu] = hp.remove_dipole(ff_maps_no_mean[nu])
#		del alm_ff
#		
#		alm_ps = hp.map2alm(ps_maps_no_mean[nu], lmax=lmax)
#		ps_maps_no_mean[nu] = hp.alm2map(alm_ps, lmax=lmax, nside = nside)
#		ps_maps_no_mean[nu] = hp.remove_dipole(ps_maps_no_mean[nu])
#		del alm_ps
#
#		alm_HI = hp.map2alm(HI_maps_no_mean[nu], lmax=lmax)
#		HI_maps_no_mean[nu] = hp.alm2map(alm_HI, lmax=lmax, nside = nside)
#		HI_maps_no_mean[nu] = hp.remove_dipole(HI_maps_no_mean[nu])
#		del alm_HI
#
#		alm_pl = hp.map2alm(pl_maps_no_mean[nu], lmax=lmax)
#		pl_maps_no_mean[nu] = hp.alm2map(alm_pl, lmax=lmax, nside = nside)
#		pl_maps_no_mean[nu] = hp.remove_dipole(pl_maps_no_mean[nu])
#		del alm_pl
#
#fig = plt.figure(figsize=(15, 7))
##fig.suptitle(f'channel: {nu_ch_new[ich]} MHz, Nside {nside}',fontsize=20)
#fig.add_subplot(221) 
#hp.mollview(synch_maps_no_mean[ich],  min=-1e3, max=1e3,unit='T[mK]',cmap='viridis',title=f'Gal synch', hold=True)
#fig.add_subplot(222) 
#hp.mollview(ff_maps_no_mean[ich], min=-1e3, max=1e3, unit='T[mK]',cmap='viridis',title=f'Gal ff',hold=True)
#fig.add_subplot(223)
#hp.mollview(ps_maps_no_mean[ich],  min=-1e2, max=1e2,unit='T[mK]',title=f'Point sources',cmap='viridis', hold=True)
#fig.add_subplot(224)
#hp.mollview(pl_maps_no_mean[ich],  unit= 'T[mK]', min=-1e2, max=1e2, title=f'Polarization leakage',cmap='viridis', hold=True)
#plt.savefig(f'comp_HI_fg_input_no_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')
#plt.show()
#del synch_maps_no_mean; del ff_maps_no_mean; del ps_maps_no_mean; del HI_maps_no_mean; del pl_maps_no_mean
#
#
#for nu in range(num_freq_new):
#		alm_synch = hp.map2alm(file_new['gal_synch'][nu], lmax=lmax)
#		file_new['gal_synch'][nu] = hp.alm2map(alm_synch, lmax=lmax, nside = nside)
#		#file_new['gal_synch'][nu] = hp.remove_dipole(file_new['gal_synch'][nu])
#		del alm_synch
#		
#		alm_ff = hp.map2alm(file_new['gal_ff'][nu], lmax=lmax)
#		file_new['gal_ff'][nu] = hp.alm2map(alm_ff, lmax=lmax, nside = nside)
#		#file_new['gal_ff'][nu] = hp.remove_dipole(file_new['gal_ff'][nu])
#		del alm_ff
#		
#		alm_ps = hp.map2alm(file_new['point_sources'][nu], lmax=lmax)
#		file_new['point_sources'][nu] = hp.alm2map(alm_ps, lmax=lmax, nside = nside)
#		#file_new['point_sources'][nu] = hp.remove_dipole(file_new['point_sources'][nu])
#		del alm_ps
#
#		alm_HI = hp.map2alm(file_new['cosmological_signal'][nu], lmax=lmax)
#		file_new['cosmological_signal'][nu] = hp.alm2map(alm_HI, lmax=lmax, nside = nside)
#		#file_new['cosmological_signal'][nu] = hp.remove_dipole(file_new['cosmological_signal'][nu])
#		del alm_HI
#
#		alm_pl = hp.map2alm(file_new['pol_leakage'][nu], lmax=lmax)
#		file_new['pol_leakage'][nu] = hp.alm2map(alm_pl, lmax=lmax, nside = nside)
#		#file_new['pol_leakage'][nu] = hp.remove_dipole(file_new['pol_leakage'][nu])
#		del alm_pl
#
#
#fig = plt.figure(figsize=(15, 7))
##fig.suptitle(f'channel: {nu_ch_new[ich]} MHz, Nside {nside}',fontsize=20)
#fig.add_subplot(221) 
#hp.mollview(file_new['gal_synch'][ich],  min=100, max=70000, norm='log',unit='T[mK]',cmap='viridis',title=f'Gal synchrotron', hold=True)
#fig.add_subplot(222) 
#hp.mollview(file_new['gal_ff'][ich], min=1, max=10000, unit='T[mK]',norm='log',cmap='viridis',title=f'Gal free-free',hold=True)
#fig.add_subplot(223)
#hp.mollview(file_new['point_sources'][ich],  min=200, max=500,unit='T[mK]',title=f'Point sources',cmap='viridis', hold=True)
#fig.add_subplot(224)
#hp.mollview(file_new['pol_leakage'][ich],  unit= 'T[mK]', min=-1, max=4, title=f'Polarization leakage',cmap='viridis', hold=True)
#plt.savefig(f'comp_HI_fg_input_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')
#plt.show()

###########################################################################
######## Computing beam size using given survey specifics: ################
### initialise a dictionary with the instrument specifications
### for noise and beam calculation
dish_diam_MeerKat = 13.5 #m
dish_diam_SKA = 15 # m
Ndishes_MeerKAT = 64.
Ndishes_SKA = 133.
dish_diam = (Ndishes_MeerKAT*dish_diam_MeerKat+ Ndishes_SKA*dish_diam_SKA)/(Ndishes_MeerKAT+Ndishes_SKA) # m (effective)
Omega_sur     = 20000   # Survey area deg2
t_obs     = 10000. # hrs, observing time
Ndishes   = Ndishes_MeerKAT + Ndishes_SKA  # number of dishes
specs_dict = {'dish_diam': dish_diam,
			  'Omega_sur': Omega_sur, 't_obs': t_obs, 'Ndishes' : Ndishes}

theta_FWMH_max = c_light*1e-6/np.min(nu_ch_new)/float(dish_diam) #radians
theta_FWMH = c_light*1e-6/nu_ch_new/float(dish_diam) #radians

print()

#beam_worst = hp.gauss_beam(theta_FWMH_max, lmax=3*nside)
##############################################################################

beam =np.array( [hp.gauss_beam(theta_FWMH[i], lmax=lmax) for i in range(num_freq_new)])
lmax_fwmh = np.array([int(np.pi/theta_FWMH[i]) for i in range(num_freq_new)])
print(f'theta_max : {theta_FWMH_max*180./np.pi} deg')
for cc in range(num_freq_new):
	print(f'theta_F at {nu_ch_new[cc]} MHz:{theta_FWMH[cc]*180./np.pi} deg\n')

#synch_maps_beam =  np.array([convolve(file_new['gal_synch'][i],beam[i], lmax=lmax) for i in range(num_freq_new)])
#ff_maps_beam =  np.array([convolve(file_new['gal_ff'][i],beam[i], lmax=lmax) for i in range(num_freq_new)])
#ps_maps_beam =  np.array([convolve(file_new['point_sources'][i],beam[i], lmax=lmax) for i in range(num_freq_new)])
#pl_maps_beam =  np.array([convolve(file_new['pol_leakage'][i],beam[i], lmax=lmax) for i in range(num_freq_new)])
#HI_maps_beam =  np.array([convolve(file_new['cosmological_signal'][i],beam[i], lmax=lmax) for i in range(num_freq_new)])


synch_maps_beam=np.load(f'Sims/synch_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy')
ff_maps_beam=np.load(f'Sims/ff_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy')
ps_maps_beam=np.load(f'Sims/ps_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy')
pl_maps_beam=np.load(f'Sims/pol_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy')
HI_maps_beam=np.load(f'Sims/HI_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy')



del file_new

fig = plt.figure(figsize=(15, 7))
#fig.suptitle(f'channel: {nu_ch_new[ich]} MHz, Nside {nside}',fontsize=20)
fig.add_subplot(221) 
hp.mollview(synch_maps_beam[ich],  min=100, max=70000, norm='log',unit='T[mK]',cmap='viridis',title=f'Gal synchrotron', hold=True)
fig.add_subplot(222) 
hp.mollview(ff_maps_beam[ich], min=1, max=10000, unit='T[mK]',norm='log',cmap='viridis',title=f'Gal free-free',hold=True)
fig.add_subplot(223)
hp.mollview(ps_maps_beam[ich],  min=200, max=500,unit='T[mK]',title=f'Point sources',cmap='viridis', hold=True)
fig.add_subplot(224)
hp.mollview(pl_maps_beam[ich],  unit= 'T[mK]', min=-1, max=4, title=f'Polarization leakage',cmap='viridis', hold=True)
plt.savefig(f'comp_HI_fg_input_beam_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')
plt.show()


fig = plt.figure(figsize=(15, 7))
#fig.suptitle(f'channel: {nu_ch_new[ich]} MHz, Nside {nside}',fontsize=20)
fig.add_subplot(141) 
hp.mollview(synch_maps_beam[ich],  min=100, max=70000, norm='log',unit='T[mK]',cmap='viridis',title=f'Gal synchrotron', hold=True)
fig.add_subplot(142) 
hp.mollview(ff_maps_beam[ich], min=1, max=10000, unit='T[mK]',norm='log',cmap='viridis',title=f'Gal free-free',hold=True)
fig.add_subplot(143)
hp.mollview(ps_maps_beam[ich],  min=200, max=500,unit='T[mK]',title=f'Point sources',cmap='viridis', hold=True)
fig.add_subplot(144)
hp.mollview(pl_maps_beam[ich],  unit= 'T[mK]', min=-1, max=4, title=f'Polarization leakage',cmap='viridis', hold=True)
plt.savefig(f'comp_HI_fg_input_beam_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}_1.png')



###########################################
########## singole figure ###################

hp.mollview(synch_maps_beam[ich],  min=100, max=70000, norm='log',unit='T[mK]',cmap='viridis',title=f'Gal synchrotron')
plt.savefig(f'comp_synch_beam_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')

hp.mollview(ff_maps_beam[ich], min=1, max=10000, unit='T[mK]',norm='log',cmap='viridis',title=f'Gal free-free')
plt.savefig(f'comp_ff_beam_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')

hp.mollview(ps_maps_beam[ich],  min=200, max=500,unit='T[mK]',title=f'Point sources',cmap='viridis')
plt.savefig(f'comp_ps_beam_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')

hp.mollview(pl_maps_beam[ich],  unit= 'T[mK]', min=-1, max=4, title=f'Polarization leakage',cmap='viridis')
plt.savefig(f'comp_pol_beam_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')

plt.show()


np.save(f'Sims/synch_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy',synch_maps_beam)
np.save(f'Sims/ff_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy',ff_maps_beam)
np.save(f'Sims/ps_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy',ps_maps_beam)
np.save(f'Sims/pol_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy',pl_maps_beam)
np.save(f'Sims/HI_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy',HI_maps_beam)
