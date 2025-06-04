import healpy as hp
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import os
import pickle
import seaborn as sns
sns.set_theme(style = 'white')
from matplotlib import colors
import matplotlib as mpl
mpl.rc('xtick', direction='in', top=False, bottom = True)
mpl.rc('ytick', direction='in', right=False, left = True)

#print(sns.color_palette("husl", 15).as_hex())
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
plt.rcParams['savefig.dpi']=300



from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 

###########################################################################3
fg_comp = 'synch_ff_ps_pol'
beam_s = 'SKA_AA4'
path_data_sims_tot = f'Sims/beam_{beam_s}_no_mean_sims_{fg_comp}_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax383_nside128'
with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()


out_dir_output = 'PCA_mexican_needlets_output/'
out_dir_output_PCA = out_dir_output+f'PCA_maps/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
out_dir_plot = out_dir_output+f'Plots_PCA_needlets/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'#noise_mask_patch_stripe82_noise_mask0.39
if not os.path.exists(out_dir_output):
        os.makedirs(out_dir_output)
if not os.path.exists(out_dir_output_PCA):
        os.makedirs(out_dir_output_PCA)

nu_ch= file['freq']
del file



need_dir = f'Maps_mexican_needlets/No_mean/Beam_{beam_s}_noise_mask0.5_unseen/'
need_tot_maps_filename = need_dir+f'bjk_maps_obs_noise_{fg_comp}_105freq_900.5_1004.5MHz_jmax12_lmax383_B1.64_nside128.npy'
need_tot_maps = np.load(need_tot_maps_filename)

jmax=need_tot_maps.shape[1]-1

num_freq = need_tot_maps.shape[0]
nu_ch = np.linspace(900.5, 1004.5, num_freq)
min_ch = min(nu_ch)
max_ch = max(nu_ch)
npix = need_tot_maps.shape[2]
nside = hp.npix2nside(npix)
lmax=3*nside-1#2*nside#
B=pow(lmax,(1./jmax))


hp.mollview(need_tot_maps[4][4], cmap='viridis')
plt.show()
######################################################################################
pix_mask = hp.query_strip(nside, theta1=np.pi*2/3, theta2=np.pi/3)
print(pix_mask)
mask_50 = np.zeros(npix)
mask_50[pix_mask] =1
fsky_50 = np.sum(mask_50)/hp.nside2npix(nside)

fig=plt.figure()
hp.mollview(mask_50, cmap='viridis', title=f'fsky={np.mean(mask_50):0.2f}', hold=True)
#plt.savefig(f'Plots_sims/mask_apo3deg_fsky{np.mean(mask_40s):0.2f}_nside{nside}.png')
plt.show()
########################################################################
bad_v = np.where(mask_50==0)


maskt =np.zeros(mask_50.shape)
maskt[bad_v]=  1
mask = ma.make_mask(maskt, shrink=False)

need_tot_maps_masked=ma.zeros(need_tot_maps.shape)

for n in range(num_freq):
    for jj in range(jmax+1):
        need_tot_maps_masked[n,jj]  =ma.MaskedArray(need_tot_maps[n,jj], mask=mask)#np.isnan(full_maps_freq_mask[n])


hp.mollview(need_tot_maps_masked[0,3], cmap='viridis', title='masked ma')
plt.show()


Cov_channels = np.zeros((jmax+1,num_freq, num_freq))

for j in range(Cov_channels.shape[0]):
    Cov_channels[j]=ma.cov(need_tot_maps_masked[:,j,:])
    #corr_coeff = ma.corrcoef(need_tot_maps_masked)


##########################################################################
print(f'jmax:{jmax}, lmax:{lmax}, B:{B:1.2f}, num_freq:{num_freq}, min_ch:{min_ch}, max_ch:{max_ch}, nside:{nside}')


eigenval=np.zeros((Cov_channels.shape[0], num_freq))
eigenvec=np.zeros((Cov_channels.shape[0], num_freq,num_freq))
for j in range(eigenval.shape[0]):
    eigenval[j], eigenvec[j] = np.linalg.eigh(Cov_channels[j])#np.linalg.eigh(Cov_channels[j])
del Cov_channels


fig = plt.figure()#(figsize=(8,4))
for j in range(eigenval.shape[0]):
    plt.semilogy(np.arange(1,num_freq+1),eigenval[j],'--o',mfc='none',label=f'j={j}')#markersize=5,

plt.legend( ncols=2)
x_ticks = np.arange(-10,num_freq+10, 10)
ax = plt.gca()
ax.set(xlim=[-10,num_freq+10],xticks=x_ticks,xlabel="eigenvalue number",ylabel="$\\lambda$",title='Eigenvalues')
plt.savefig(f'Plots_paper/eigenvalue_cov_mask_unseen_mexican_need_no_mean_{fg_comp}_beam_{beam_s}_jmax{jmax}_lmax{lmax}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}_nside{nside}.png')
plt.show()

if fg_comp=='synch_ff_ps':
    num_sources=3
if fg_comp=='synch_ff_ps_pol':
    num_sources=3

Nfg = num_freq - num_sources
print(f'Nfg:{num_sources}')
print(eigenvec.shape)
eigenvec_fg_Nfg = eigenvec[:, :,Nfg:num_freq]#eigenvec[:, :,:num_sources]#

print(eigenvec_fg_Nfg.shape)

del eigenvec, eigenval
#############################################################################
# gal freefree spectral index for reference
FF_col = np.array([nu_ch**(-2.13)]).T 

# gal synchrotron spectral index region for reference
sync_A = np.array([nu_ch**(-3.2)]).T 
sync_B = np.array([nu_ch**(-2.6)]).T 
y1 = sync_A/np.linalg.norm(sync_A)
y2 = sync_B/np.linalg.norm(sync_B)

### actual plotting
fig=plt.figure()
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams["axes.labelsize"] = 12

x = np.arange(0,len(nu_ch))

plt.fill_between(x,y1.T[0],y2.T[0],alpha=0.3,label='gal synch')
for j in [1,2,3]:
    plt.plot(abs(eigenvec_fg_Nfg[j]/np.linalg.norm(eigenvec_fg_Nfg[j],axis=0)),label=f'mix mat column,j={j}')
plt.plot(FF_col/np.linalg.norm(FF_col),'m:',label='gal ff')

ax = plt.gca()
ax.set(ylim=[0.0,0.4],xlabel="frequency channel",ylabel="Spectral emission",title='PCA-mixing matrix columns')
plt.legend(ncols=2)
plt.show()
####################################################################################################
res_fg_maps = np.zeros((eigenvec_fg_Nfg.shape[0], num_freq, npix))

for j in range(eigenvec_fg_Nfg.shape[0]):
    res_fg_maps[j] = ma.dot(eigenvec_fg_Nfg[j],ma.dot(eigenvec_fg_Nfg[j].T,need_tot_maps[:,j,:]))

print(res_fg_maps.shape)

filename = out_dir_output_PCA+f'res_PCA_fg_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_nside{nside}'
with open(filename+'.pkl', 'wb') as f:
    pickle.dump(res_fg_maps, f)
    f.close()
del f
#np.save(out_dir_output_PCA+f'res_PCA_fg_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{num_sources}_nside{nside}.npy',res_fg_maps)

print('.. ho calcolato res fg .. ')

ich= int(num_freq/2)
j_test=7

res_HI_maps = np.zeros((eigenvec_fg_Nfg.shape[0], num_freq, npix))
for j in range(eigenvec_fg_Nfg.shape[0]):
    res_HI_maps[j,:,:] = need_tot_maps[:,j,:] - res_fg_maps[j,:,:]
    res_HI_maps[j,:,bad_v]=hp.UNSEEN

    hp.mollview(res_HI_maps[j][ich],min=0, max=0.39, cmap='viridis', title=f'j={j}')
plt.show()
del need_tot_maps


filename = out_dir_output_PCA+f'res_PCA_HI_noise_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_nside{nside}'
with open(filename+'.pkl', 'wb') as f:
    pickle.dump(res_HI_maps, f)
    f.close()
#res_HI_maps.dump(out_dir_output_PCA+f'res_PCA_HI_noise_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{num_sources}_nside{nside}.npy')
#np.save(out_dir_output_PCA+f'res_PCA_HI_noise_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{num_sources}_nside{nside}.npy',res_HI_maps)

print('.. ho calcolato res HI .. ')

res_fg_maps_totj=res_fg_maps.sum(axis=0)
res_HI_maps_totj = res_HI_maps.sum(axis=0)
res_fg_maps_totj[:, bad_v] = hp.UNSEEN
res_HI_maps_totj[:, bad_v] = hp.UNSEEN
del res_HI_maps; del res_fg_maps


print('fin qui ci sono')

############### leakage ###########################
#Foreground's maps

need_fg_maps_filename = need_dir+f'bjk_maps_fg_{fg_comp}_{num_freq}freq_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}.npy'
need_fg_maps = np.load(need_fg_maps_filename)#[:,:jmax,:]

#need_fg_maps_masked=ma.zeros(need_fg_maps.shape)
#for n in range(num_freq):
#    for jj in range(jmax+1):
#        need_fg_maps_masked[n,jj]  =ma.MaskedArray(need_fg_maps[n,jj], mask=mask)#np.isnan(full_maps_freq_mask[n])


leak_fg_maps = np.zeros((eigenvec_fg_Nfg.shape[0], num_freq, npix))
leak_HI_maps = np.zeros((eigenvec_fg_Nfg.shape[0], num_freq, npix))

for j in range(eigenvec_fg_Nfg.shape[0]):
    leak_fg_maps[j] = need_fg_maps[:,j,:] - ma.dot(eigenvec_fg_Nfg[j],ma.dot(eigenvec_fg_Nfg[j].T,need_fg_maps[:,j,:]))
    leak_fg_maps[j,:,bad_v]=hp.UNSEEN


filename = out_dir_output_PCA+f'leak_PCA_fg_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_nside{nside}'
with open(filename+'.pkl', 'wb') as ff:
    pickle.dump(leak_fg_maps, ff)
    ff.close()

#leak_fg_maps.dump(out_dir_output_PCA+f'leak_PCA_fg_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{num_sources}_nside{nside}.npy')
#np.save(out_dir_output_PCA+f'leak_PCA_fg_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{num_sources}_nside{nside}.npy',leak_fg_maps)

fig = plt.figure()
plt.suptitle(f'Frequency channel: {nu_ch[ich]} MHz, Nfg:{num_sources}, jmax:{jmax}, lmax:{lmax} ')
hp.mollview(leak_fg_maps.sum(axis=0)[ich], title='Foregroud leakage', cmap = 'viridis', min=0, max=0.5, hold=True)
#plt.savefig(out_dir_plot+f'betajk_leak_fg_jmax{jmax}_lmax{lmax}_nside_{nside}_Nfg{num_sources}.png')
plt.show()
del leak_fg_maps


need_HI_maps_filename = need_dir+f'bjk_maps_HI_noise_{num_freq}freq_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}.npy'
need_HI_maps = np.load(need_HI_maps_filename)#[:,:jmax,:]



#need_HI_maps_masked=ma.zeros(need_HI_maps.shape)
#for n in range(num_freq):
#    for jj in range(jmax+1):
#        need_HI_maps_masked[n,jj]  =ma.MaskedArray(need_HI_maps[n,jj], mask=mask)#np.isnan(full_maps_freq_mask[n])



for j in range(eigenvec_fg_Nfg.shape[0]):
    leak_HI_maps[j] = ma.dot(eigenvec_fg_Nfg[j],ma.dot(eigenvec_fg_Nfg[j].T,need_HI_maps[:,j,:]))
    leak_HI_maps[j,:,bad_v]=hp.UNSEEN   

del eigenvec_fg_Nfg; 


filename = out_dir_output_PCA+f'leak_PCA_HI_noise_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_nside{nside}'
with open(filename+'.pkl', 'wb') as fff:
    pickle.dump(leak_HI_maps, fff)
    fff.close()

#leak_HI_maps.dump(out_dir_output_PCA+f'leak_PCA_HI_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{num_sources}_nside{nside}.npy')
#np.save(out_dir_output_PCA+f'leak_PCA_HI_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{num_sources}_nside{nside}.npy',leak_HI_maps)

fig = plt.figure()
plt.suptitle(f'Frequency channel: {nu_ch[ich]} MHz, Nfg:{num_sources}, jmax:{jmax}, lmax:{lmax} ')
hp.mollview(leak_HI_maps.sum(axis=0)[ich], title='HI leakage', cmap = 'viridis', min=0, max=0.5, hold=True)
#plt.savefig(out_dir_plot+f'betajk_leak_HI_jmax{jmax}_{lmax}_nside_{nside}_Nfg{num_sources}.png')
plt.show()

del leak_HI_maps#; del leak_fg_maps


##############################################


need_HI_maps_totj = ma.sum(need_HI_maps, axis=1)#need_HI_maps.sum(axis=1)np.nansum(need_HI_maps, axis=1)#
need_fg_maps_totj = ma.sum(need_fg_maps, axis=1)#need_fg_maps.sum(axis=1)np.nansum(need_fg_maps, axis=1)#
need_HI_maps_totj[:, bad_v] = hp.UNSEEN
need_fg_maps_totj[:,bad_v] = hp.UNSEEN

del need_HI_maps;del need_fg_maps

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}',fontsize=20)
fig.add_subplot(221)
hp.mollview(np.abs(res_fg_maps_totj[ich]/need_fg_maps_totj[ich]-1)*100, min=0, max=0.1,cmap='viridis', title=f'%(Res fg/Fg)-1',unit='%' ,hold=True)
fig.add_subplot(222) 
hp.mollview(need_HI_maps_totj[ich], cmap='viridis', title=f'HI signal + noise',min=0, max =1,hold=True)
fig.add_subplot(223)
hp.mollview(res_HI_maps_totj[ich], title=f'PCA HI + noise',min=0, max =1,cmap='viridis', hold=True)
#plt.savefig(out_dir_plot+f'betajk_res_need_PCA_sumj_Nfg{num_sources}_jmax{jmax}_lmax{lmax}_nside{nside}.png')
plt.show()


###################### PROVIAMO CON I CL #########################

lmax_cl = 2*nside#512

cl_cosmo_HI = np.zeros((len(nu_ch), lmax_cl+1))
cl_PCA_HI_need2harm = np.zeros((len(nu_ch), lmax_cl+1))

for n in range(len(nu_ch)):
    cl_cosmo_HI[n]=hp.anafast(need_HI_maps_totj[n], lmax=lmax_cl)
    cl_PCA_HI_need2harm[n] = hp.anafast(res_HI_maps_totj[n], lmax=lmax_cl)

del need_HI_maps_totj; del res_HI_maps_totj

ell=np.arange(lmax_cl+1)
factor=ell*(ell+1)/(2*np.pi)


fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: channel:{nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{num_sources}')
plt.plot(ell[2:],factor[2:]*cl_cosmo_HI[ich][2:], label='Cosmo HI + noise')
plt.plot(ell[2:],factor[2:]*cl_PCA_HI_need2harm[ich][2:],'+', mfc='none', label='PCA HI + noise')
plt.ylabel(r'$\frac{\ell(\ell+1)}{2\pi}  C_{\ell} $')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} e C_{\ell} $')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 30))

diff_cl_need2sphe = cl_PCA_HI_need2harm/cl_cosmo_HI-1
frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe[ich][2:]*100, label='% PCA_HI/input_HI -1')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-10,10])
frame2.set_ylabel(r'%$  diff$')
frame2.set_xlabel(r'$\ell$')
frame2.set_xticks(np.arange(1,200+1, 30))
#plt.tight_layout()
plt.legend()
#plt.savefig(out_dir_plot+f'cls_need2pix_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.png')
plt.show()


fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: mean over channels, jmax:{jmax}, lmax:{lmax}, Nfg:{num_sources}')
plt.plot(ell[2:], factor[2:]*cl_cosmo_HI.mean(axis=0)[2:], label = f'Cosmo HI + noise')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_need2harm.mean(axis=0)[2:],'+',mfc='none', label = f'PCA HI + noise')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 30))

del cl_PCA_HI_need2harm; del cl_cosmo_HI
frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe.mean(axis=0)[2:]*100, label='% PCA_HI/input_HI -1')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-10,10])
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
frame2.set_xticks(np.arange(1,200+1, 30))
#plt.tight_layout()
plt.legend()
#plt.savefig(out_dir_plot+f'cls_need2pix_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.png')

plt.show()

del diff_cl_need2sphe


