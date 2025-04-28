import h5py
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import cython_mylibc as pippo

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
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

############################################################################################
path_data = 'sim_PL05_from191030.hd5'
file = h5py.File(path_data,'r')
nu_ch = np.array(file['frequencies'])


file_new={}

delta_nu_out = 10
file_new['frequencies'] = nu_ch_f(nu_ch,delta_nu_out)#np.array([nu_ch[i*delta_nu] for i in range(0,int(len(nu_ch)/delta_nu))])

components = list(file.keys())
print(components)
components.remove('frequencies')
#components.remove('pol_leakage')

for c in components:
  print(c)
  file_new[c]=merging_maps(nu_ch,file_new['frequencies'],file[c], delta_nu_out )

print(len(file_new['frequencies']), hp.get_nside(file_new['cosmological_signal'][1]))
 
nside = hp.get_nside(file_new['cosmological_signal'][1])

del file

nu_ch_new = np.array(file_new['frequencies'])
num_freq_new=len(nu_ch_new)
npix = np.shape(file_new['cosmological_signal'])[1]

ich = int(num_freq_new/2)

lmax=3*nside


synch_maps_no_mean = np.array([file_new['gal_synch'][i] -np.mean(file_new['gal_synch'][i],axis=0)  for i in range(num_freq_new)])
ff_maps_no_mean = np.array([file_new['gal_ff'][i] -np.mean(file_new['gal_ff'][i],axis=0) for i in range(num_freq_new)])
ps_maps_no_mean = np.array([file_new['point_sources'][i] -np.mean(file_new['point_sources'][i],axis=0) for i in range(num_freq_new)]) 
HI_maps_no_mean = np.array([file_new['cosmological_signal'][i] -np.mean(file_new['cosmological_signal'][i],axis=0) for i in range(num_freq_new)]) 
pl_maps_no_mean = np.array([file_new['pol_leakage'][i] -np.mean(file_new['pol_leakage'][i],axis=0) for i in range(num_freq_new)]) 


del file_new

for nu in range(num_freq_new):
		alm_synch = hp.map2alm(synch_maps_no_mean[nu], lmax=lmax)
		synch_maps_no_mean[nu] = hp.alm2map(alm_synch, lmax=lmax, nside = nside)
		synch_maps_no_mean[nu] = hp.remove_dipole(synch_maps_no_mean[nu])
		del alm_synch
		
		alm_ff = hp.map2alm(ff_maps_no_mean[nu], lmax=lmax)
		ff_maps_no_mean[nu] = hp.alm2map(alm_ff, lmax=lmax, nside = nside)
		ff_maps_no_mean[nu] = hp.remove_dipole(ff_maps_no_mean[nu])
		del alm_ff
		
		alm_ps = hp.map2alm(ps_maps_no_mean[nu], lmax=lmax)
		ps_maps_no_mean[nu] = hp.alm2map(alm_ps, lmax=lmax, nside = nside)
		ps_maps_no_mean[nu] = hp.remove_dipole(ps_maps_no_mean[nu])
		del alm_ps

		alm_HI = hp.map2alm(HI_maps_no_mean[nu], lmax=lmax)
		HI_maps_no_mean[nu] = hp.alm2map(alm_HI, lmax=lmax, nside = nside)
		HI_maps_no_mean[nu] = hp.remove_dipole(HI_maps_no_mean[nu])
		del alm_HI

		alm_pl = hp.map2alm(pl_maps_no_mean[nu], lmax=lmax)
		pl_maps_no_mean[nu] = hp.alm2map(alm_pl, lmax=lmax, nside = nside)
		pl_maps_no_mean[nu] = hp.remove_dipole(pl_maps_no_mean[nu])
		del alm_pl


#fig = plt.figure(figsize=(10, 7))
#fig.suptitle(f'channel: {nu_ch_new[ich]} MHz, Nside {nside}',fontsize=20)
#fig.add_subplot(221) 
#hp.mollview(np.abs(synch_maps_no_mean[ich]), min=0.0001,norm='log', cmap='viridis',title=f'Gal synch', hold=True)
#fig.add_subplot(222) 
#hp.mollview(np.abs(ff_maps_no_mean[ich]), min=0.0001,norm='log', cmap='viridis',title=f'Gal ff',hold=True)
#fig.add_subplot(223)
#hp.mollview(np.abs(ps_maps_no_mean[ich]),min=0.0001, norm='log', title=f'Point sources',cmap='viridis', hold=True)
##fig.add_subplot(224)
##hp.mollview(file_new['cosmological_signal'][ich], norm='log', title=f'Cosmological signal, freq={nu_ch[ich]}',cmap='viridis', hold=True)
#plt.savefig(f'fg_input_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')
#plt.show()

fig = plt.figure(figsize=(15, 7))
fig.suptitle(f'channel: {nu_ch_new[ich]} MHz, Nside {nside}',fontsize=20)
fig.add_subplot(141) 
hp.mollview(synch_maps_no_mean[ich],  min=-1e3, max=1e3,unit='mK',cmap='viridis',title=f'Gal synch', hold=True)
fig.add_subplot(142) 
hp.mollview(ff_maps_no_mean[ich], min=-1e3, max=1e3, unit='mK',cmap='viridis',title=f'Gal ff',hold=True)
fig.add_subplot(143)
hp.mollview(ps_maps_no_mean[ich],  min=-1e2, max=1e2,unit='mK',title=f'Point sources',cmap='viridis', hold=True)
fig.add_subplot(144)
hp.mollview(HI_maps_no_mean[ich],  unit= 'mK', min=0, max=1, title=f'Cosmological signal',cmap='viridis', hold=True)
plt.savefig(f'comp_HI_fg_input_lmax{lmax}_nside{nside}_ch{nu_ch_new[ich]}.png')
plt.show()



#########################################################################
####################### MEXICAN NEEDLETS #################################
def mexicanneedlet(B,j,lmax,p=1,normalised=True):
	'''Return the needlet filter b(l) for a Mexican needlet with parameters ``B`` and ``j``.
	
	Parameters
	----------
	B : float
		The parameter B of the needlet, should be larger that 1.
	j : int or np.ndarray
		The frequency j of the needlet. Can be an array with the values of ``j`` to be calculated.
	lmax : int
		The maximum value of the multipole l for which the filter will be calculated (included).
	p : int
		Order of the Mexican needlet.
	normalised : bool, optional
		If ``True``, the sum (in frequencies ``j``) of the squares of the filters will be 1 for all multipole ell.

	Returns
	-------
	np.ndarray
		A numpy array containing the values of a needlet filter b(l), starting with l=0. If ``j`` is
		an array, it returns an array containing the filters for each frequency.
	'''
	ls=np.arange(lmax+1)
	j=np.array(j,ndmin=1)
	needs=[]
	if normalised != True:
		for jj in j:
#            u=(ls/B**jj)
#            bl=u**(2.*p)*np.exp(-u**2.)
			u=((ls*(ls+1)/B**(2.*jj)))
			bl=(u**p)*np.exp(-u)
			needs.append(bl)
	else:
		K=np.zeros(lmax+1)
		#jmax=np.max((np.log(5.*lmax)/np.log(B),np.max(j)))
		jmax = np.max(j)+1
		for jj in np.arange(0,jmax+1):#np.arange(1,jmax+1)
#            u=(ls/B**jj)                   This is an almost identical definition
#            bl=u**2.*np.exp(-u**2.)
			u=((ls*(ls+1)/B**(2.*jj)))
			bl=(u**p)*np.exp(-u)
			K=K+bl**2.
			if np.isin(jj,j):
				needs.append(bl)
		needs=needs/np.sqrt(np.mean(K[int(lmax/3):int(2*lmax/3)]))
	return(np.squeeze(needs))
#########################################################################################

nside = 256
lmax=3*nside
jmax=12
B=pippo.mylibpy_jmax_lmax2B(jmax, lmax)
jvec = np.arange(jmax+1)

b_values_p1 = mexicanneedlet(B,jvec,lmax,p=1,normalised=True)
b_values_p4 = mexicanneedlet(B,jvec,lmax,p=4,normalised=True)

betajk_synch_p1 = np.zeros((jmax+1, npix))
betajk_synch_p4 = np.zeros((jmax+1, npix))

betajk_ff_p1 = np.zeros((jmax+1, npix))
betajk_ff_p4 = np.zeros((jmax+1, npix))

betajk_ps_p1 = np.zeros((jmax+1, npix))
betajk_ps_p4 = np.zeros((jmax+1, npix))

for j in jvec:
	betajk_synch_p1[j] = hp.alm2map(hp.almxfl(hp.map2alm(synch_maps_no_mean[ich],lmax=lmax),b_values_p1[j,:]),lmax=lmax,nside=nside) 
	betajk_synch_p4[j] = hp.alm2map(hp.almxfl(hp.map2alm(synch_maps_no_mean[ich],lmax=lmax),b_values_p4[j,:]),lmax=lmax,nside=nside)
	
	betajk_ff_p1[j] = hp.alm2map(hp.almxfl(hp.map2alm(ff_maps_no_mean[ich],lmax=lmax),b_values_p1[j,:]),lmax=lmax,nside=nside) 
	betajk_ff_p4[j] = hp.alm2map(hp.almxfl(hp.map2alm(ff_maps_no_mean[ich],lmax=lmax),b_values_p4[j,:]),lmax=lmax,nside=nside)

	betajk_ps_p1[j] = hp.alm2map(hp.almxfl(hp.map2alm(ps_maps_no_mean[ich],lmax=lmax),b_values_p1[j,:]),lmax=lmax,nside=nside) 
	betajk_ps_p4[j] = hp.alm2map(hp.almxfl(hp.map2alm(ps_maps_no_mean[ich],lmax=lmax),b_values_p4[j,:]),lmax=lmax,nside=nside)


j_plot=2

fig=plt.figure()
fig.suptitle(f'B={B:1.2f}, p=1, j={j_plot}, channel: {nu_ch[ich]} MHz')
fig.add_subplot(221)
hp.mollview(betajk_synch_p1[j_plot], title=f'Gal synch',min=betajk_synch_p1[j_plot].min(),max=betajk_synch_p1[j_plot].max(), cmap='viridis', hold=True)
fig.add_subplot(222)
hp.mollview(betajk_ff_p1[j_plot], title=f'Gal ff',min=betajk_ff_p1[j_plot].min(),max=betajk_ff_p1[j_plot].max(), cmap='viridis', hold=True)
fig.add_subplot(223)
hp.mollview(betajk_ps_p1[j_plot], title=f'Point sources',min=betajk_ps_p1[j_plot].min(),max=betajk_ps_p1[j_plot].max(), cmap='viridis', hold=True)
plt.savefig(f'fg_comp_synch_ff_ps_mex_need_B{B:1.2f}_j{j_plot}_p1.png')


fig=plt.figure()
fig.suptitle(f'B={B:1.2f}, p=4, j={j_plot}, channel: {nu_ch[ich]} MHz')
fig.add_subplot(221)
hp.mollview(betajk_synch_p4[j_plot], title=f'Gal synch',min=betajk_synch_p1[j_plot].min(),max=betajk_synch_p1[j_plot].max(), cmap='viridis', hold=True)
fig.add_subplot(222)
hp.mollview(betajk_ff_p4[j_plot], title=f'Gal ff',min=betajk_ff_p1[j_plot].min(),max=betajk_ff_p1[j_plot].max(), cmap='viridis', hold=True)
fig.add_subplot(223)
hp.mollview(betajk_ps_p4[j_plot], title=f'Point sources',min=betajk_ps_p1[j_plot].min(),max=betajk_ps_p1[j_plot].max(), cmap='viridis', hold=True)
plt.savefig(f'fg_comp_synch_ff_ps_mex_need_B{B:1.2f}_j{j_plot}_p4.png')

j_plot1=6
fig=plt.figure()
fig.suptitle(f'B={B:1.2f}, p=1, j={j_plot1}, channel: {nu_ch[ich]} MHz')
fig.add_subplot(221)
hp.mollview(betajk_synch_p1[j_plot1], title=f'Gal synch',min=betajk_synch_p1[j_plot1].min(),max=betajk_synch_p1[j_plot1].max(), cmap='viridis', hold=True)
fig.add_subplot(222)
hp.mollview(betajk_ff_p1[j_plot1], title=f'Gal ff',min=betajk_ff_p1[j_plot1].min(),max=betajk_ff_p1[j_plot1].max(), cmap='viridis', hold=True)
fig.add_subplot(223)
hp.mollview(betajk_ps_p1[j_plot1], title=f'Point sources',min=betajk_ps_p1[j_plot1].min(),max=betajk_ps_p1[j_plot1].max(), cmap='viridis', hold=True)
plt.savefig(f'fg_comp_synch_ff_ps_mex_need_B{B:1.2f}_j{j_plot1}_p1.png')


fig=plt.figure()
fig.suptitle(f'B={B:1.2f}, p=4, j={j_plot1}, channel: {nu_ch[ich]} MHz')
fig.add_subplot(221)
hp.mollview(betajk_synch_p4[j_plot1], title=f'Gal synch',min=betajk_synch_p1[j_plot1].min(),max=betajk_synch_p1[j_plot1].max(), cmap='viridis', hold=True)
fig.add_subplot(222)
hp.mollview(betajk_ff_p4[j_plot1], title=f'Gal ff',min=betajk_ff_p1[j_plot1].min(),max=betajk_ff_p1[j_plot1].max(), cmap='viridis', hold=True)
fig.add_subplot(223)
hp.mollview(betajk_ps_p4[j_plot1], title=f'Point sources',min=betajk_ps_p1[j_plot1].min(),max=betajk_ps_p1[j_plot1].max(), cmap='viridis', hold=True)
plt.savefig(f'fg_comp_synch_ff_ps_mex_need_B{B:1.2f}_j{j_plot1}_p4.png')


theta_c=np.pi / 2
phi_c=0
prova= hp.query_strip(nside = nside, theta1=theta_c, theta2 = theta_c+100*np.sqrt(hp.nside2pixarea(nside))/2, inclusive=True)

ipix = np.arange(0,1000)

fig=plt.figure()
fig.subplots_adjust(wspace=0.4)
fig.subplots_adjust(hspace=0.6)

fig.suptitle(f'channel: {nu_ch[ich]}, B={B:1.2f}, j={j_plot}')
ax1 = fig.add_subplot(221)
ax1.set_title('Gal synch')
ax1.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_synch_p1[j_plot][prova][0:1000], label='p=1')
ax1.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_synch_p4[j_plot][prova][0:1000], label='p=4')
ax1.set_xlim([0.,100.])
ax1.yaxis.set_major_formatter(formatter) 
ax1.set_xlabel(r'$\theta$[deg]')
ax1.set_ylabel(r'$\beta_{jk}$')
ax1.legend()

ax2 = fig.add_subplot(222)
ax2.set_title('Gal ff')
ax2.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_ff_p1[j_plot][prova][0:1000], label='p=1')
ax2.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_ff_p4[j_plot][prova][0:1000], label='p=4')
ax2.set_xlim([0.,100.])
ax2.yaxis.set_major_formatter(formatter) 
ax2.set_xlabel(r'$\theta$[deg]')
ax2.set_ylabel(r'$\beta_{jk}$')
ax2.legend()

ax3 = fig.add_subplot(223)
ax3.set_title('Gal ps')
ax3.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees= True)),betajk_ps_p1[j_plot][prova][0:1000], label='p=1')
ax3.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees= True)),betajk_ps_p4[j_plot][prova][0:1000], label='p=4')
ax3.set_xlim([0.,100.])
ax3.yaxis.set_major_formatter(formatter) 
ax3.set_xlabel(r'$\theta$[deg]')
ax3.set_ylabel(r'$\beta_{jk}$')
ax3.legend()

plt.savefig(f'loc_angle_fg_comp_synch_ff_ps_mex_need_B{B:1.2f}_j{j_plot}_p1_p4.png')
##########

fig=plt.figure()
fig.subplots_adjust(wspace=0.4)
fig.subplots_adjust(hspace=0.6)

fig.suptitle(f'channel: {nu_ch[ich]}, B={B:1.2f}, j={j_plot1}')
ax1 = fig.add_subplot(221)
ax1.set_title('Gal synch')
ax1.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_synch_p1[j_plot1][prova][0:1000], label='p=1')
ax1.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_synch_p4[j_plot1][prova][0:1000], label='p=4')
ax1.set_xlim([0.,100.])
ax1.yaxis.set_major_formatter(formatter) 
ax1.set_xlabel(r'$\theta$[deg]')
ax1.set_ylabel(r'$\beta_{jk}$')
ax1.legend()

ax2 = fig.add_subplot(222)
ax2.set_title('Gal ff')
ax2.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_ff_p1[j_plot1][prova][0:1000], label='p=1')
ax2.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees=True)),betajk_ff_p4[j_plot1][prova][0:1000], label='p=4')
ax2.set_xlim([0.,100.])
ax2.yaxis.set_major_formatter(formatter) 
ax2.set_xlabel(r'$\theta$[deg]')
ax2.set_ylabel(r'$\beta_{jk}$')
ax2.legend()

ax3 = fig.add_subplot(223)
ax3.set_title('Gal ps')
ax3.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees= True)),betajk_ps_p1[j_plot1][prova][0:1000], label='p=1')
ax3.plot(ipix*np.sqrt(hp.nside2pixarea(nside, degrees= True)),betajk_ps_p4[j_plot1][prova][0:1000], label='p=4')
ax3.set_xlim([0.,100.])
ax3.yaxis.set_major_formatter(formatter) 
ax3.set_xlabel(r'$\theta$[deg]')
ax3.set_ylabel(r'$\beta_{jk}$')
ax3.legend()
plt.savefig(f'loc_angle_fg_comp_synch_ff_ps_mex_need_B{B:1.2f}_j{j_plot1}_p1_p4.png')


##########################################
# proviamo a fare il plot nello spazio armonico 

bl_synch_p1 = np.zeros((jmax+1, 296065))
bl_synch_p4 = np.zeros((jmax+1, 296065))

bl_ff_p1 = np.zeros((jmax+1, 296065))
bl_ff_p4 = np.zeros((jmax+1, 296065))

bl_ps_p1 = np.zeros((jmax+1, 296065))
bl_ps_p4 = np.zeros((jmax+1, 296065))

for j in jvec:
	bl_synch_p1[j] = hp.almxfl(hp.map2alm(synch_maps_no_mean[ich],lmax=lmax),b_values_p1[j,:]) 
	bl_synch_p4[j] = hp.almxfl(hp.map2alm(synch_maps_no_mean[ich],lmax=lmax),b_values_p4[j,:])
	
	bl_ff_p1[j] = hp.almxfl(hp.map2alm(ff_maps_no_mean[ich],lmax=lmax),b_values_p1[j,:]) 
	bl_ff_p4[j] = hp.almxfl(hp.map2alm(ff_maps_no_mean[ich],lmax=lmax),b_values_p4[j,:])

	bl_ps_p1[j] = hp.almxfl(hp.map2alm(ps_maps_no_mean[ich],lmax=lmax),b_values_p1[j,:]) 
	bl_ps_p4[j] = hp.almxfl(hp.map2alm(ps_maps_no_mean[ich],lmax=lmax),b_values_p4[j,:])


fig=plt.figure()
fig.subplots_adjust(wspace=0.4)
fig.subplots_adjust(hspace=0.6)

fig.suptitle(f'channel: {nu_ch[ich]}, B={B:1.2f}, j={j_plot1}')
ax1 = fig.add_subplot(221)
ax1.set_title('Gal synch')
ax1.plot(bl_synch_p1[j_plot1], label='p=1')
ax1.plot(bl_synch_p4[j_plot1], label='p=4')
ax1.set_xlim([0.,100.])
ax1.yaxis.set_major_formatter(formatter) 
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$b_{\ell}$')
ax1.legend()

ax2 = fig.add_subplot(222)
ax2.set_title('Gal ff')
ax2.plot(bl_ff_p1[j_plot1], label='p=1')
ax2.plot(bl_ff_p4[j_plot1], label='p=4')
ax2.set_xlim([0.,100.])
ax2.yaxis.set_major_formatter(formatter) 
ax2.set_xlabel(r'$\ell$')
ax2.set_ylabel(r'$b_{\ell}$')
ax2.legend()

ax3 = fig.add_subplot(223)
ax3.set_title('Gal ps')
ax3.plot(bl_ps_p1[j_plot1], label='p=1')
ax3.plot(bl_ps_p4[j_plot1], label='p=4')
ax3.set_xlim([0.,100.])
ax3.yaxis.set_major_formatter(formatter) 
ax3.set_xlabel(r'$\ell$')
ax3.set_ylabel(r'$b_{\ell}$')
ax3.legend()



plt.show()