import healpy as hp
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import os
import pickle

import seaborn as sns
sns.set_theme(style = 'white')
sns.color_palette(n_colors=15)
#sns.set_palette('husl',15)
c_pal = sns.color_palette().as_hex()
import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

###########################################################################3
fg_comp = 'synch_ff_ps_pol'
beam_s = 'theta40arcmin'
path_data_sims_tot = f'Sims/beam_{beam_s}_no_mean_sims_{fg_comp}_noise_40freq_905.0_1295.0MHz_thick10MHz_lmax383_nside128'
with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()


out_dir_output = 'PCA_needlets_output/'
out_dir_output_PCA = out_dir_output+f'PCA_maps/No_mean/Beam_{beam_s}_noise_mask0.39_unseen_vary_Nfg/'
out_dir_plot = out_dir_output+f'Plots_PCA_needlets/No_mean/Beam_{beam_s}_noise_mask0.39_unseen_vary_Nfg/'#noise_mask_patch_stripe82_noise_mask0.39
if not os.path.exists(out_dir_output):
        os.makedirs(out_dir_output)
if not os.path.exists(out_dir_output_PCA):
        os.makedirs(out_dir_output_PCA)

nu_ch= file['freq']
del file



need_dir = f'Maps_needlets/No_mean/Beam_{beam_s}_noise_mask0.39_unseen/'
need_tot_maps_filename = need_dir+f'bjk_maps_obs_noise_{fg_comp}_40freq_905.0_1295.0MHz_jmax4_lmax383_B4.42_nside128.npy'
need_tot_maps = np.load(need_tot_maps_filename)

jmax=need_tot_maps.shape[1]-1

num_freq = need_tot_maps.shape[0]
nu_ch = np.linspace(905.0, 1295.0, num_freq)
min_ch = min(nu_ch)
max_ch = max(nu_ch)
npix = need_tot_maps.shape[2]
nside = hp.npix2nside(npix)
lmax=3*nside-1#2*nside#
B=pow(lmax,(1./jmax))

######################################################################################
mask1_40 = hp.read_map('HFI_Mask_GalPlane_2048_R1.10.fits', field=1)#fsky 40 %
mask_40t = hp.ud_grade(mask1_40, nside_out=256)
mask_40 = hp.ud_grade(mask_40t, nside_out=nside)
del mask1_40
#mask_40s = hp.sphtfunc.smoothing(mask_40, 3*np.pi/180,lmax=lmax)
fsky  = np.mean(mask_40) 
########################################################################
bad_v = np.where(mask_40==0)


maskt =np.zeros(mask_40.shape)

del mask_40
maskt[bad_v]=  1
mask = ma.make_mask(maskt, shrink=False)

need_tot_maps_masked=ma.zeros(need_tot_maps.shape)

for n in range(num_freq):
    for jj in range(jmax+1):
        need_tot_maps_masked[n,jj]  =ma.MaskedArray(need_tot_maps[n,jj], mask=mask)#np.isnan(full_maps_freq_mask[n])


Cov_channels = np.zeros((jmax+1,num_freq, num_freq))

for j in range(Cov_channels.shape[0]):
    Cov_channels[j]=ma.cov(need_tot_maps_masked[:,j,:])
    #corr_coeff = ma.corrcoef(need_tot_maps_masked)
#for j in range(Cov_channels.shape[0]):
#    for c in range(0, num_freq):
#        for cc in range(0, num_freq):
#            Cov_channels[j,c,cc]=ma.dot(need_tot_maps_masked[c,j,:],need_tot_maps_masked[cc,j,:].T)
#            Cov_channels[j,cc,c] = Cov_channels[j,c,cc]

##########################################################################
print(f'jmax:{jmax}, lmax:{lmax}, B:{B:1.2f}, num_freq:{num_freq}, min_ch:{min_ch}, max_ch:{max_ch}, nside:{nside}')


eigenval=np.zeros((Cov_channels.shape[0], num_freq))
eigenvec=np.zeros((Cov_channels.shape[0], num_freq,num_freq))
for j in range(eigenval.shape[0]):
    #eigenval[j], eigenvec[j] = np.linalg.eigh(Cov_channels[j])#np.linalg.eigh(Cov_channels[j])
    eigenvec[j], eigenval[j], Vr = np.linalg.svd(Cov_channels[j], full_matrices=True)#np.linalg.eigh(Cov_channels[j])

del Cov_channels


fig = plt.figure(figsize=(8,4))
for j in range(eigenval.shape[0]):
    plt.semilogy(np.arange(1,num_freq+1),eigenval[j],'--o',mfc='none',markersize=5,label=f'j={j}')

plt.legend(fontsize=12, ncols=2)
x_ticks = np.arange(-10,num_freq+10, 10)
ax = plt.gca()
ax.set(xlim=[-10,num_freq+10],xticks=x_ticks,xlabel="eigenvalue number",ylabel="$\\lambda$",title='STANDARD NEED - Eigenvalues')
#plt.savefig(out_dir_output+f'eigenvalue_cov_need_no_mean_jmax{jmax}_lmax{lmax}_nside{nside}.png')
plt.show()

num_sources = np.array([40,20,18,3,3])#np.array([3,3,3,3,3,3,3,3,3,3,3,40,40])
num_sources_3 = 18

Nfg = np.array([ num_sources[i] for i in range(len(num_sources)) ])
Nfg_3 = num_sources_3

eigenvec_fg_Nfg3 = eigenvec[:, :,:Nfg_3]


eigenvec_fg_Nfg_0 = eigenvec[0, :, :Nfg[0]]
eigenvec_fg_Nfg_1 = eigenvec[1, :, :Nfg[1]]
eigenvec_fg_Nfg_2 = eigenvec[2, :, :Nfg[2]]
eigenvec_fg_Nfg_3 = eigenvec[3, :, :Nfg[3]]
eigenvec_fg_Nfg_4 = eigenvec[4, :, :Nfg[4]]

del eigenvec, eigenval

####################################################################################################
res_fg_maps = np.zeros((jmax+1, num_freq, npix))
res_fg_maps_3 = np.zeros((jmax+1, num_freq, npix))

print(eigenvec_fg_Nfg3.shape)

for j in range(eigenvec_fg_Nfg3.shape[0]):
    res_fg_maps_3[j] = ma.dot(eigenvec_fg_Nfg3[j],ma.dot(eigenvec_fg_Nfg3[j].T,need_tot_maps[:,j,:]))

res_fg_maps[0] = ma.dot(eigenvec_fg_Nfg_0,ma.dot(eigenvec_fg_Nfg_0.T,need_tot_maps[:,0,:]))
res_fg_maps[1] = ma.dot(eigenvec_fg_Nfg_1,ma.dot(eigenvec_fg_Nfg_1.T,need_tot_maps[:,1,:]))
res_fg_maps[2] = ma.dot(eigenvec_fg_Nfg_2,ma.dot(eigenvec_fg_Nfg_2.T,need_tot_maps[:,2,:]))
res_fg_maps[3] = ma.dot(eigenvec_fg_Nfg_3,ma.dot(eigenvec_fg_Nfg_3.T,need_tot_maps[:,3,:]))
res_fg_maps[4] = ma.dot(eigenvec_fg_Nfg_4,ma.dot(eigenvec_fg_Nfg_4.T,need_tot_maps[:,4,:]))


filename = out_dir_output_PCA+f'res_PCA_fg_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{min(nu_ch)}_{max(nu_ch)}MHz_Nfg{num_sources}_nside{nside}'
with open(filename+'.pkl', 'wb') as f:
    pickle.dump(res_fg_maps, f)
    f.close()
del f
#np.save(out_dir_output_PCA+f'res_PCA_fg_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{num_sources}_nside{nside}.npy',res_fg_maps)

print('.. ho calcolato res fg .. ')

ich= int(num_freq/2)
j_test=7

res_HI_maps = np.zeros((jmax+1, num_freq, npix))
res_HI_maps_3 = np.zeros((jmax+1, num_freq, npix))
for j in range(jmax+1):
    res_HI_maps_3[j,:,:] = need_tot_maps[:,j,:] - res_fg_maps_3[j,:,:]
    res_HI_maps[j] = need_tot_maps[:,j,:] - res_fg_maps[j]

    res_HI_maps[j,:,bad_v]=hp.UNSEEN
    res_HI_maps_3[j,:,bad_v]=hp.UNSEEN

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'Res HI channel {ich}: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, num_sources:{num_sources}',fontsize=20)
for j in range(res_HI_maps.shape[0]):
    fig.add_subplot(4, 4, j + 1)
    hp.mollview(res_HI_maps[j][ich], title=f'j={j}', cmap='viridis', hold=True)

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'Res HI channel {ich}: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax} num_sources:{num_sources_3}',fontsize=20)
for j in range(res_HI_maps.shape[0]):
    fig.add_subplot(4, 4, j + 1)
    hp.mollview(res_HI_maps_3[j][ich], title=f'j={j}', cmap='viridis', hold=True)
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
res_fg_maps_3_totj=res_fg_maps_3.sum(axis=0)
res_HI_maps_totj = res_HI_maps.sum(axis=0)
res_HI_maps_3_totj = res_HI_maps_3.sum(axis=0)

res_fg_maps_3_totj[:, bad_v] = hp.UNSEEN
res_HI_maps_3_totj[:, bad_v] = hp.UNSEEN
res_fg_maps_totj[:, bad_v] = hp.UNSEEN
res_HI_maps_totj[:, bad_v] = hp.UNSEEN
del res_HI_maps; del res_fg_maps; del res_fg_maps_3


print('fin qui ci sono')

############### leakage ###########################
#Foreground's maps

need_fg_maps_filename = need_dir+f'bjk_maps_fg_{fg_comp}_{num_freq}freq_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}.npy'
need_fg_maps = np.load(need_fg_maps_filename)#[:,:jmax,:]

#need_fg_maps_masked=ma.zeros(need_fg_maps.shape)
#for n in range(num_freq):
#    for jj in range(jmax+1):
#        need_fg_maps_masked[n,jj]  =ma.MaskedArray(need_fg_maps[n,jj], mask=mask)#np.isnan(full_maps_freq_mask[n])


leak_fg_maps = np.zeros((jmax+1, num_freq, npix))
leak_HI_maps = np.zeros((jmax+1, num_freq, npix))

for j in range(jmax+1):
    leak_fg_maps[j] = need_fg_maps[:,j,:] - ma.dot(eigenvec_fg_Nfg3[j],ma.dot(eigenvec_fg_Nfg3[j].T,need_fg_maps[:,j,:]))
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



for j in range(jmax+1):
    leak_HI_maps[j] = ma.dot(eigenvec_fg_Nfg3[j],ma.dot(eigenvec_fg_Nfg3[j].T,need_HI_maps[:,j,:]))
    leak_HI_maps[j,:,bad_v]=hp.UNSEEN   

del eigenvec_fg_Nfg3; 


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
hp.mollview(np.abs(res_fg_maps_3_totj[ich]/need_fg_maps_totj[ich]-1)*100, min=0, max=0.1,cmap='viridis', title=f'%(Res fg/Fg)-1 Nfg 3',unit='%' ,hold=True)
fig.add_subplot(223) 
hp.mollview(need_HI_maps_totj[ich], cmap='viridis', title=f'HI signal',min=0, max =1,hold=True)
fig.add_subplot(224)
hp.mollview(res_HI_maps_totj[ich], title=f'PCA HI',min=0, max =1,cmap='viridis', hold=True)
#plt.savefig(out_dir_plot+f'betajk_res_need_PCA_sumj_Nfg{num_sources}_jmax{jmax}_lmax{lmax}_nside{nside}.png')
plt.show()


###################### PROVIAMO CON I CL #########################

lmax_cl = 2*nside#512

cl_cosmo_HI = np.zeros((len(nu_ch), lmax_cl+1))
cl_PCA_HI_need2harm = np.zeros((len(nu_ch), lmax_cl+1))
cl_PCA_HI_3_need2harm = np.zeros((len(nu_ch), lmax_cl+1))
#cl_fg = np.zeros((len(nu_ch), lmax_cl+1))

for n in range(len(nu_ch)):
    cl_cosmo_HI[n]=hp.anafast(need_HI_maps_totj[n], lmax=lmax_cl)
    cl_PCA_HI_need2harm[n] = hp.anafast(res_HI_maps_totj[n], lmax=lmax_cl)
    cl_PCA_HI_3_need2harm[n] = hp.anafast(res_HI_maps_3_totj[n], lmax=lmax_cl)
    #cl_fg[n] = hp.anafast(need_fg_maps_jtot[n], lmax=lmax_cl)


del need_HI_maps_totj; del res_HI_maps_totj

ell=np.arange(lmax_cl+1)
factor=ell*(ell+1)/(2*np.pi)


fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: channel:{nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{num_sources}')
plt.plot(ell[2:],factor[2:]*cl_cosmo_HI[ich][2:], 'k--',label='Cosmo HI + noise')
plt.plot(ell[2:],factor[2:]*cl_PCA_HI_need2harm[ich][2:],'+', mfc='none', label='PCA HI + noise')
plt.plot(ell[2:],factor[2:]*cl_PCA_HI_3_need2harm[ich][2:], '+',mfc='none',label='PCA HI Nfg 3')

plt.ylabel(r'$\frac{\ell(\ell+1)}{2\pi}  C_{\ell} $')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} e C_{\ell} $')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 30))

diff_cl_need2sphe = cl_PCA_HI_need2harm/cl_cosmo_HI-1
diff_cl_3_need2sphe = cl_PCA_HI_3_need2harm/cl_cosmo_HI-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe[ich][2:]*100, label='% PCA_HI/input_HI -1')
plt.plot(ell[2:], diff_cl_3_need2sphe[ich][2:]*100, label='% PCA_HI/input_HI -1 Nfg 3')

frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-50,50])
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
plt.plot(ell[2:], factor[2:]*cl_cosmo_HI.mean(axis=0)[2:], 'k--',label = f'Cosmo HI + noise')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_need2harm.mean(axis=0)[2:],'+',mfc='none', label = f'PCA HI + noise')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_3_need2harm.mean(axis=0)[2:],'+',mfc='none', label = f'PCA Nfg 3')

plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 30))

del cl_PCA_HI_need2harm; del cl_cosmo_HI
frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe.mean(axis=0)[2:]*100, label='% PCA_HI/input_HI -1')
plt.plot(ell[2:], diff_cl_3_need2sphe.mean(axis=0)[2:]*100, label='% PCA_HI/input_HI -1 Nfg 3')

frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_xlim([0,200])
frame2.set_ylim([-50,50])
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
frame2.set_xticks(np.arange(1,200+1, 30))
#plt.tight_layout()
plt.legend()
#plt.savefig(out_dir_plot+f'cls_need2pix_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.png')


fig = plt.figure(figsize=(10,7))
plt.plot(ell[2:],100*((diff_cl_need2sphe/diff_cl_3_need2sphe).mean(axis=0)[2:]-1))

plt.show()

del diff_cl_need2sphe