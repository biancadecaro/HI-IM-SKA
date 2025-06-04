import h5py
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set_theme(style = 'white')
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
plt.rcParams['legend.columnspacing'] = 0.5
plt.rcParams['savefig.dpi']=300


from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 


import matplotlib as mpl
mpl.rc('xtick', direction='in', top=False, bottom = True)
mpl.rc('ytick', direction='in', right=False, left = True)

c_pal = sns.color_palette().as_hex()
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
#print(idx_nu_max)

file_new={}


file_new['frequencies'] = nu_ch[:idx_nu_max]

#print(file_new['frequencies'], len(file_new['frequencies']))

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
#print(ich, nu_ch[ich])

lmax=3*nside-1




#hp.mollview(file_new['gal_synch'][ich],  min=100, max=70000, norm='log',unit='T[mK]',cmap='viridis',title=f'Gal synchrotron')
#plt.savefig(f'Plots_paper/comp_synch_beam_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')
#
#hp.mollview(file_new['gal_ff'][ich], min=1, max=10000, unit='T[mK]',norm='log',cmap='viridis',title=f'Gal free-free')
#plt.savefig(f'Plots_paper/comp_ff_beam_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')
#
#hp.mollview(file_new['point_sources'][ich],  min=200, max=500,unit='T[mK]',title=f'Point sources',cmap='viridis')
#plt.savefig(f'Plots_paper/comp_ps_beam_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')
#
#hp.mollview(file_new['pol_leakage'][ich],  unit= 'T[mK]', min=-1, max=4, title=f'Polarization leakage',cmap='viridis')
#plt.savefig(f'Plots_paper/comp_pol_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')
#
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

dnu = nu_ch_new[1]-nu_ch_new[0]

print(f'dnu={dnu} MHz')

sigma_noise = sigma_N(nu_ch_new,dnu,**specs_dict)

noise = np.array([noise_map(sigma,nside=nside) for sigma in sigma_noise])
del sigma_noise

print(noise.shape)

components.append('noise')


#beam_worst = hp.gauss_beam(theta_FWMH_max, lmax=3*nside)
##############################################################################

beam =np.array( [hp.gauss_beam(theta_FWMH[i], lmax=lmax) for i in range(num_freq_new)])
lmax_fwmh = np.array([int(np.pi/theta_FWMH[i]) for i in range(num_freq_new)])
lmax_fwmh_max = int(np.pi/theta_FWMH_max)
#print(f'theta_max : {theta_FWMH_max*180./np.pi} deg')
#for cc in range(num_freq_new):
#	print(f'theta_F at {nu_ch_new[cc]} MHz:{theta_FWMH[cc]*180./np.pi} deg\n')


#noise_maps_beam = np.array([convolve(noise[i],beam[i], lmax=lmax) for i in range(num_freq_new)])
#np.save(f'Sims/noise_sims_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy',noise_maps_beam)


synch_maps_beam=np.load(f'Sims/synch_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy')
ff_maps_beam=np.load(f'Sims/ff_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy')
ps_maps_beam=np.load(f'Sims/ps_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy')
pl_maps_beam=np.load(f'Sims/pol_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy')
HI_maps_beam=np.load(f'Sims/HI_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy')
noise_maps_beam = np.load(f'Sims/noise_sims_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy')

#fig = plt.figure(figsize=(15, 7))
##fig.suptitle(f'channel: {nu_ch_new[ich]} MHz, Nside {nside}',fontsize=20)
#fig.add_subplot(221) 
#hp.mollview(synch_maps_beam[ich],  min=100, max=70000, norm='log',unit='T[mK]',cmap='viridis',title=f'Gal synchrotron', hold=True)
#fig.add_subplot(222) 
#hp.mollview(ff_maps_beam[ich], min=1, max=10000, unit='T[mK]',norm='log',cmap='viridis',title=f'Gal free-free',hold=True)
#fig.add_subplot(223)
#hp.mollview(ps_maps_beam[ich],  min=200, max=500,unit='T[mK]',title=f'Point sources',cmap='viridis', hold=True)
#fig.add_subplot(224)
#hp.mollview(pl_maps_beam[ich],  unit= 'T[mK]', min=-1, max=4, title=f'Polarization leakage',cmap='viridis', hold=True)
##plt.savefig(f'Plots_paper/comp_HI_fg_input_beam_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')
#plt.show()

###########################################
########## singole figure ###################

hp.mollview(synch_maps_beam[ich],  min=100, max=70000, norm='log',unit='T[mK]',cmap='viridis',title=f'Gal synchrotron')
plt.savefig(f'Plots_paper/comp_synch_beam_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')

hp.mollview(ff_maps_beam[ich], min=1, max=10000, unit='T[mK]',norm='log',cmap='viridis',title=f'Gal free-free')
plt.savefig(f'Plots_paper/comp_ff_beam_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')

hp.mollview(ps_maps_beam[ich],  min=200, max=500,unit='T[mK]',title=f'Point sources',cmap='viridis')
plt.savefig(f'Plots_paper/comp_ps_beam_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')

hp.mollview(pl_maps_beam[ich],  unit= 'T[mK]', min=-1, max=4, title=f'Polarization leakage',cmap='viridis')
plt.savefig(f'Plots_paper/comp_pol_beam_mean_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')

#plt.show()

###############################################################################################
################################### cl ########################################################


#file_mean = {}
#for c in components:
#	if c == 'noise':
#		print(c, 'sono dentro all if')
#		file_mean[c]= np.array([noise[i]  for i in range(num_freq_new)])
#	else:
#		file_mean[c] = np.array([file_new[c][i] for i in range(num_freq_new)])

file_beam_mean = {'cosmological_signal':HI_maps_beam,'gal_ff':ff_maps_beam,'gal_synch':synch_maps_beam,'point_sources':ps_maps_beam, 'pol_leakage':pl_maps_beam, 'noise':noise_maps_beam}


file_no_mean = {}
for c in components:
	if c == 'noise':
		print(c, 'sono dentro all if')
		file_no_mean[c]= np.array([noise[i]  for i in range(num_freq_new)])
	else:
		file_no_mean[c] = np.array([file_new[c][i] -np.mean(file_new[c][i],axis=0)  for i in range(num_freq_new)])

#del file_new


synch_maps_beam_no_mean = np.array([synch_maps_beam[i] -np.mean(synch_maps_beam[i],axis=0)  for i in range(num_freq_new)])
ff_maps_beam_no_mean = np.array([ff_maps_beam[i] -np.mean(ff_maps_beam[i],axis=0) for i in range(num_freq_new)])
ps_maps_beam_no_mean = np.array([ps_maps_beam[i] -np.mean(ps_maps_beam[i],axis=0) for i in range(num_freq_new)]) 
HI_maps_beam_no_mean = np.array([HI_maps_beam[i] -np.mean(HI_maps_beam[i],axis=0) for i in range(num_freq_new)]) 
pl_maps_beam_no_mean = np.array([pl_maps_beam[i] -np.mean(pl_maps_beam[i],axis=0) for i in range(num_freq_new)]) 

file_beam_no_mean = {'cosmological_signal':HI_maps_beam_no_mean,'gal_ff':ff_maps_beam_no_mean,'gal_synch':synch_maps_beam_no_mean,'point_sources':ps_maps_beam_no_mean, 'pol_leakage':pl_maps_beam_no_mean, 'noise':noise_maps_beam}


ls_dic = {'cosmological_signal':"-",'gal_ff':"--",'gal_synch':"-.",'point_sources':':', 'pol_leakage':(0, (3, 1, 1, 1)), 'noise':(0, (3, 10, 1, 10))}
lab_dic = {'cosmological_signal':"21-cm signal",'gal_ff':"Gal free-free",'gal_synch':"Gal synchrotron",'point_sources':"Point sources", 'pol_leakage':"Pol leakage", 'noise':'Noise'}
col_dic = {'cosmological_signal':c_pal[0],'gal_ff':c_pal[1],'gal_synch':c_pal[2],'point_sources':c_pal[3], 'pol_leakage': c_pal[4], 'noise':c_pal[5]}



#for c in components:
#	for nu in range(num_freq_new):
#		alm=hp.map2alm(file_no_mean[c][nu], lmax=lmax)
#		file_no_mean[c][nu] = hp.alm2map(alm, lmax=lmax, nside = nside)
#		file_no_mean[c][nu] = hp.remove_dipole(file_no_mean[c][nu])
#
#		alm_beam = hp.map2alm(file_beam_no_mean[c][nu], lmax=lmax)
#		file_beam_no_mean[c][nu] = hp.alm2map(alm_beam, lmax=lmax, nside = nside)
#		file_beam_no_mean[c][nu] = hp.remove_dipole(file_beam_no_mean[c][nu])
#		#alm=0
#		#alm_beam=0
#	print(f'fatto {c}')

##############################################################################
################# proviamo a fare il plot per frequenza#######################
lat = -5#deg
long = 0

pix_dir = hp.ang2pix(nside=nside, theta=long, phi=lat, lonlat=True)

print(f'pix_dir={pix_dir}')#, np.abs(HI_maps_beam_no_mean[0,pix_dir]), nu_ch_new.shape)

fig, ax = plt.subplots(1,1)
for c in components:
	if c =='noise':
		ax.plot(nu_ch_new,np.abs(file_beam_no_mean[c][:,pix_dir]), color=col_dic[c], ls=ls_dic[c], label=lab_dic[c])
		print(noise_maps_beam[:,pix_dir])
	else:
		ax.plot(nu_ch_new,np.abs(file_beam_no_mean[c][:,pix_dir]), color=col_dic[c], ls=ls_dic[c], label=lab_dic[c])
ax.set_xticks(np.arange(min(nu_ch_new), max(nu_ch_new), 20))
#ax.set_xlabel(np.arange(min(nu_ch_new), max(nu_ch_new), 20))
ax.set_yscale('log')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_ylabel('|T| [mK]')
ax.set_xlabel(r'$\nu$')
plt.legend(ncols=1, loc='upper right')
plt.savefig(f'Plots_paper/brightness_temperature_beam_no_mean_lat{lat}_lon{long}_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_nside{nside}.png')
plt.show()
###########################################################
##################### cl ##################################
lmax_cl = 2*nside
ell = np.arange(lmax_cl+1)
factor = ell*(ell+1)/(2.*np.pi)

lmax_cl_plot = 400

cl_comp = {}

for c in components:
	cl_comp[c]= np.zeros((num_freq_new, lmax_cl+1))
	for nu in range(num_freq_new):
		cl_comp[c][nu] = hp.anafast(file_no_mean[c][nu], lmax=lmax_cl)

plt.figure()
plt.title(r'$\nu$='+f'{nu_ch[ich]} MHz')
for c in components:
	plt.plot(ell[2:], factor[2:]*cl_comp[c][ich][2:], ls=ls_dic[c], color=col_dic[c], label=lab_dic[c])
plt.yscale('log')
plt.ylim([1e-7, 1e7])
plt.xlim([0,lmax_cl_plot])
plt.ylabel(r'$\frac{\ell(\ell+1)}{2\pi} C_{\ell}$ [mK$^2$]')
plt.xlabel(r'$\ell$')
plt.legend(ncols=1 )
plt.savefig(f'Plots_paper/cl_components_HI_sync_ff_ps_pol_ch{nu_ch[ich]}MHz_lmax{lmax_cl}_nside{nside}.png')
#plt.show()


#synch_maps_beam =  np.array([convolve(file_new['gal_synch'][i],beam[i], lmax=lmax) for i in range(num_freq_new)])
#ff_maps_beam =  np.array([convolve(file_new['gal_ff'][i],beam[i], lmax=lmax) for i in range(num_freq_new)])
#ps_maps_beam =  np.array([convolve(file_new['point_sources'][i],beam[i], lmax=lmax) for i in range(num_freq_new)])
#pl_maps_beam =  np.array([convolve(file_new['pol_leakage'][i],beam[i], lmax=lmax) for i in range(num_freq_new)])
#HI_maps_beam =  np.array([convolve(file_new['cosmological_signal'][i],beam[i], lmax=lmax) for i in range(num_freq_new)])


#np.save(f'Sims/synch_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy',synch_maps_beam)
#np.save(f'Sims/ff_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy',ff_maps_beam)
#np.save(f'Sims/ps_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy',ps_maps_beam)
#np.save(f'Sims/pol_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy',pl_maps_beam)
#np.save(f'Sims/HI_sims_mean_beam_SKA_AA4_noise_{num_freq_new}freq_{min(nu_ch_new)}_{max(nu_ch_new)}MHz_lmax{lmax}_nside{nside}.npy',HI_maps_beam)
############################################################################################################


cl_comp_beam = {}

for c in components:
	cl_comp_beam[c]= np.zeros((num_freq_new, lmax_cl+1))
	for nu in range(num_freq_new):
		cl_comp_beam[c][nu] = hp.anafast(file_beam_no_mean[c][nu], lmax=lmax_cl)

plt.figure()
plt.title(r'$\nu$='+f'{nu_ch[ich]} MHz')
for c in components:
	plt.plot(ell[2:], factor[2:]*cl_comp_beam[c][ich][2:], ls=ls_dic[c], color=col_dic[c],label=lab_dic[c])
plt.yscale('log')
plt.ylim([1e-7, 1e7])
plt.xlim([0,lmax_fwmh_max])
plt.ylabel(r'$\frac{\ell(\ell+1)}{2\pi} C_{\ell}$ [mK$^2$]')
plt.xlabel(r'$\ell$')
plt.legend(ncols=1, loc='upper right')
plt.savefig(f'Plots_paper/cl_components_beam_SKA_AA4_HI_sync_ff_ps_pol_ch{nu_ch[ich]}MHz_lmax{lmax_cl}_nside{nside}.png')


#cl_HI_min = cl_comp_beam['cosmological_signal'][0]
#cl_HI_max = cl_comp_beam['cosmological_signal'][-1]

#plt.figure()
#plt.title(r'$\nu$='+f'{nu_ch[ich]} MHz')
##plt.fill_between(ell[2:],cl_HI_max[2:],cl_HI_min[2:], alpha=0.3,label='21cm signal' )
#for c in components:
#	if c=='cosmological_signal':
#		plt.fill_between(ell[2:],factor[2:]*cl_comp_beam[c][0][2:],factor[2:]*cl_comp_beam[c][-1][2:], alpha=0.5,label='21cm signal' )
#	else:
#		plt.plot(ell[2:], factor[2:]*cl_comp_beam[c][ich][2:], ls=ls_dic[c],color=col_dic[c], label=lab_dic[c])
#plt.axvline(x=lmax_fwmh_max, color='k', ls='--', alpha=0.5, label = r'$\ell_{\rm beam}$=%d'%lmax_fwmh_max)
#plt.yscale('log')
#plt.ylim([1e-7, 1e7])
#plt.xlim([0,lmax_cl_plot])
#plt.ylabel(r'$\frac{\ell(\ell+1)}{2\pi} C_{\ell}$ [mK$^2$]')
#plt.xlabel(r'$\ell$')
#plt.legend(ncols=1 )
#plt.savefig(f'Plots_paper/cl_components_beam_SKA_AA4_fill_between_HI_sync_ff_ps_pol_ch{nu_ch[ich]}MHz_lmax{lmax_cl}_nside{nside}.png')


plt.figure()
#plt.title(r'$\nu_{\rm min}$='+f'{nu_ch[0]} MHz, '+ r'$\nu_{\rm max}$='+f'{nu_ch[-1]} MHz')
plt.title(r'$\nu \in $'+f'[{nu_ch[0]}, {nu_ch[-1]}] MHz')
#plt.fill_between(ell[2:],cl_HI_max[2:],cl_HI_min[2:], alpha=0.3,label='21cm signal' )
for c in components:

	plt.fill_between(ell[2:],factor[2:]*cl_comp_beam[c][0][2:],factor[2:]*cl_comp_beam[c][-1][2:], alpha=0.7,label=lab_dic[c] )
#plt.axvline(x=lmax_fwmh_max, color='k', ls='--', alpha=0.5, label = r'$\ell_{\rm beam}$=%d'%lmax_fwmh_max)
plt.yscale('log')
plt.ylim([1e-7, 1e7])
plt.xlim([0,lmax_fwmh_max])
plt.ylabel(r'$\frac{\ell(\ell+1)}{2\pi} C_{\ell}$ [mK$^2$]')
plt.xlabel(r'$\ell$')
plt.legend(ncols=1, loc='upper right')
plt.savefig(f'Plots_paper/cl_components_beam_SKA_AA4_fill_between_HI_sync_ff_ps_pol_ch_min_max{lmax_cl}_nside{nside}.png')

plt.show()
