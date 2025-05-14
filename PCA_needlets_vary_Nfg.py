import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
from needlets_analysis import analysis
import os

import seaborn as sns
sns.set_theme(style = 'white')
custom_palette=sns.color_palette(n_colors=15)
#sns.palettes.color_palette()

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

############################################################

fg_comp='synch_ff_ps'
path_data_sims_tot = f'Sims/beam_theta40arcmin_no_mean_sims_{fg_comp}_noise_40freq_905.0_1295.0MHz_thick10MHz_lmax383_nside128'
with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

out_dir_output = 'PCA_needlets_output/'
out_dir_output_PCA = out_dir_output+'PCA_maps/No_mean/Beam_theta40arcmin_noise/Diff_Nfg/'
out_dir_plot = out_dir_output+'Plots_PCA_needlets/No_mean/Beam_theta40arcmin_noise/Diff_Nfg/'
if not os.path.exists(out_dir_output):
        os.makedirs(out_dir_output)
if not os.path.exists(out_dir_output_PCA):
        os.makedirs(out_dir_output_PCA)

nu_ch= file['freq']
del file

need_dir = 'Maps_needlets/No_mean/Beam_theta40arcmin_noise/'
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

print(f'jmax:{jmax}, lmax:{lmax}, B:{B:1.2f}, num_freq:{num_freq}, min_ch:{min_ch}, max_ch:{max_ch}, nside:{nside}')

####################################################################################################

Cov_channels = np.zeros((jmax+1,num_freq, num_freq))

for j in range(Cov_channels.shape[0]):
    Cov_channels[j]=np.cov(need_tot_maps[:,j,:])

eigenval=np.zeros((Cov_channels.shape[0], num_freq))
eigenvec=np.zeros((Cov_channels.shape[0], num_freq,num_freq))
for j in range(eigenval.shape[0]):
    eigenval[j], eigenvec[j] = np.linalg.eigh(Cov_channels[j])#np.linalg.eigh(Cov_channels[j])
del Cov_channels

##########################################################################################################

fig = plt.figure(figsize=(8,4))
for j in range(eigenval.shape[0]):
    plt.semilogy(np.arange(1,num_freq+1),eigenval[j],'--o',mfc='none',markersize=5,label=f'j={j}')

plt.legend(fontsize=10, ncols=4)
x_ticks = np.arange(-10,num_freq+10, 10)
ax = plt.gca()
ax.set(xlim=[-10,num_freq+10],xticks=x_ticks,xlabel="eigenvalue number",ylabel="$\\lambda$",title='STANDARD NEED - Eigenvalues')
#plt.savefig(out_dir_output+f'eigenvalue_cov_need_no_mean_jmax{jmax}_lmax{lmax}_nside{nside}.png')
plt.show()
##############################################################################################
num_sources = np.array([num_freq,num_freq, 3, 3, 3, num_freq])#np.array([3,3,3,3,3,3,3,3,3,3,3,40,40])
num_sources_3 = np.array([3,3, 3, 3, 3, 3])#np.array([3,3,3,3,3,3,3,3,3,3,3,40,40])

Nfg = np.array([num_freq - num_sources[i] for i in range(len(num_sources)) ])
Nfg_3 = np.array([num_freq - num_sources_3[i] for i in range(len(num_sources)) ])
print(f'Nfg:{Nfg}')


eigenvec_fg_Nfg_0_Nfg3 = eigenvec[0, :, Nfg_3[0]:num_freq]
eigenvec_fg_Nfg_1_Nfg3 = eigenvec[1, :, Nfg_3[1]:num_freq]
eigenvec_fg_Nfg_2_Nfg3 = eigenvec[2, :, Nfg_3[2]:num_freq]
eigenvec_fg_Nfg_3_Nfg3 = eigenvec[3, :, Nfg_3[3]:num_freq]
eigenvec_fg_Nfg_4_Nfg3 = eigenvec[4, :, Nfg_3[4]:num_freq]


eigenvec_fg_Nfg_0 = eigenvec[0, :, Nfg[0]:num_freq]
eigenvec_fg_Nfg_1 = eigenvec[1, :, Nfg[1]:num_freq]
eigenvec_fg_Nfg_2 = eigenvec[2, :, Nfg[2]:num_freq]
eigenvec_fg_Nfg_3 = eigenvec[3, :, Nfg[3]:num_freq]
eigenvec_fg_Nfg_4 = eigenvec[4, :, Nfg[4]:num_freq]
#eigenvec_fg_Nfg_5 = eigenvec[5, :num_freq, Nfg[5]:num_freq]
#eigenvec_fg_Nfg_6 = eigenvec[6, :num_freq, Nfg[6]:num_freq]
#eigenvec_fg_Nfg_7 = eigenvec[7, :num_freq, Nfg[7]:num_freq]
#eigenvec_fg_Nfg_8 = eigenvec[8, :num_freq, Nfg[8]:num_freq]
#eigenvec_fg_Nfg_9 = eigenvec[9, :num_freq, Nfg[9]:num_freq]
#eigenvec_fg_Nfg_10 = eigenvec[10, :num_freq, Nfg[10]:num_freq]
#eigenvec_fg_Nfg_11 = eigenvec[11, :num_freq, Nfg[11]:num_freq]
#eigenvec_fg_Nfg_12 = eigenvec[12, :num_freq, Nfg[12]:num_freq]

print(eigenvec_fg_Nfg_0.shape, eigenvec_fg_Nfg_1.shape, eigenvec_fg_Nfg_2.shape, eigenvec_fg_Nfg_3.shape)

del eigenvec, eigenval


res_fg_maps = np.zeros((jmax+1, num_freq, npix))
res_fg_maps_3 = np.zeros((jmax+1, num_freq, npix))


res_fg_maps_3[0] = eigenvec_fg_Nfg_0_Nfg3@eigenvec_fg_Nfg_0_Nfg3.T@need_tot_maps[:,0,:]
res_fg_maps_3[2] = eigenvec_fg_Nfg_0_Nfg3@eigenvec_fg_Nfg_0_Nfg3.T@need_tot_maps[:,2,:]
res_fg_maps_3[1] = eigenvec_fg_Nfg_0_Nfg3@eigenvec_fg_Nfg_0_Nfg3.T@need_tot_maps[:,1,:]
res_fg_maps_3[4] = eigenvec_fg_Nfg_0_Nfg3@eigenvec_fg_Nfg_0_Nfg3.T@need_tot_maps[:,4,:]
res_fg_maps_3[3] = eigenvec_fg_Nfg_0_Nfg3@eigenvec_fg_Nfg_0_Nfg3.T@need_tot_maps[:,3,:]

res_fg_maps[0] = eigenvec_fg_Nfg_0@eigenvec_fg_Nfg_0.T@need_tot_maps[:,0,:]
res_fg_maps[1] = eigenvec_fg_Nfg_1@eigenvec_fg_Nfg_1.T@need_tot_maps[:,1,:]
res_fg_maps[2] = eigenvec_fg_Nfg_2@eigenvec_fg_Nfg_2.T@need_tot_maps[:,2,:]
res_fg_maps[3] = eigenvec_fg_Nfg_3@eigenvec_fg_Nfg_3.T@need_tot_maps[:,3,:]
res_fg_maps[4] = eigenvec_fg_Nfg_4@eigenvec_fg_Nfg_4.T@need_tot_maps[:,4,:]
#res_fg_maps[5] = eigenvec_fg_Nfg_5@eigenvec_fg_Nfg_5.T@need_tot_maps[:,5,:]
#res_fg_maps[6] = eigenvec_fg_Nfg_6@eigenvec_fg_Nfg_6.T@need_tot_maps[:,6,:]
#res_fg_maps[7] = eigenvec_fg_Nfg_7@eigenvec_fg_Nfg_7.T@need_tot_maps[:,7,:]
#res_fg_maps[8] = eigenvec_fg_Nfg_8@eigenvec_fg_Nfg_8.T@need_tot_maps[:,8,:]
#res_fg_maps[9] = eigenvec_fg_Nfg_9@eigenvec_fg_Nfg_9.T@need_tot_maps[:,9,:]
#res_fg_maps[10] = eigenvec_fg_Nfg_10@eigenvec_fg_Nfg_10.T@need_tot_maps[:,10,:]
#res_fg_maps[11] = eigenvec_fg_Nfg_11@eigenvec_fg_Nfg_11.T@need_tot_maps[:,11,:]
#res_fg_maps[12] = eigenvec_fg_Nfg_12@eigenvec_fg_Nfg_12.T@need_tot_maps[:,12,:]
#
#np.save(out_dir_output_PCA+f'res_PCA_fg_sync_ff_ps_jmax{jmax}_lmax{lmax}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{num_sources}.npy',res_fg_maps)

ich=20
j_test=7


res_HI_maps = np.zeros((jmax+1, num_freq, npix))
res_HI_maps_3 = np.zeros((jmax+1, num_freq, npix))
for j in range(jmax+1):
    res_HI_maps[j] = need_tot_maps[:,j,:] - res_fg_maps[j]
    res_HI_maps_3[j] = need_tot_maps[:,j,:] - res_fg_maps_3[j]
del need_tot_maps

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


#np.save(out_dir_output_PCA+f'res_PCA_HI_jmax{jmax}_lmax{lmax}_{num_freq}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{num_sources}.npy',res_HI_maps)

res_fg_maps_totj=res_fg_maps.sum(axis=0)
res_fg_maps_3_totj=res_fg_maps_3.sum(axis=0)
res_HI_maps_totj = res_HI_maps.sum(axis=0)
res_HI_maps_3_totj = res_HI_maps_3.sum(axis=0)
del res_HI_maps; del res_fg_maps; del res_fg_maps_3

print('fin qui ci sono')

############### le-akage ###########################
#Foreground's maps
need_fg_maps_filename = need_dir+f'bjk_maps_fg_{fg_comp}_{num_freq}freq_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}.npy'
need_fg_maps = np.load(need_fg_maps_filename)

leak_fg_maps = np.zeros((jmax+1, num_freq, npix))
leak_HI_maps = np.zeros((jmax+1, num_freq, npix))

leak_fg_maps[0] = need_fg_maps[:,0,:]-eigenvec_fg_Nfg_0@eigenvec_fg_Nfg_0.T@need_fg_maps[:,0,:]
leak_fg_maps[1] = need_fg_maps[:,1,:]-eigenvec_fg_Nfg_1@eigenvec_fg_Nfg_1.T@need_fg_maps[:,1,:]
leak_fg_maps[2] = need_fg_maps[:,2,:]-eigenvec_fg_Nfg_2@eigenvec_fg_Nfg_2.T@need_fg_maps[:,2,:]
leak_fg_maps[3] = need_fg_maps[:,3,:]-eigenvec_fg_Nfg_3@eigenvec_fg_Nfg_3.T@need_fg_maps[:,3,:]
leak_fg_maps[4] = need_fg_maps[:,4,:]-eigenvec_fg_Nfg_4@eigenvec_fg_Nfg_4.T@need_fg_maps[:,4,:]
#leak_fg_maps[5] = need_fg_maps[:,5,:]-eigenvec_fg_Nfg_5@eigenvec_fg_Nfg_5.T@need_fg_maps[:,5,:]
#leak_fg_maps[6] = need_fg_maps[:,6,:]-eigenvec_fg_Nfg_6@eigenvec_fg_Nfg_6.T@need_fg_maps[:,6,:]
#leak_fg_maps[7] = need_fg_maps[:,7,:]-eigenvec_fg_Nfg_7@eigenvec_fg_Nfg_7.T@need_fg_maps[:,7,:]
#leak_fg_maps[8] = need_fg_maps[:,8,:]-eigenvec_fg_Nfg_8@eigenvec_fg_Nfg_8.T@need_fg_maps[:,8,:]
#leak_fg_maps[9] = need_fg_maps[:,9,:]-eigenvec_fg_Nfg_9@eigenvec_fg_Nfg_9.T@need_fg_maps[:,9,:]
#leak_fg_maps[10] = need_fg_maps[:,10,:]-eigenvec_fg_Nfg_10@eigenvec_fg_Nfg_10.T@need_fg_maps[:,10,:]
#leak_fg_maps[11] = need_fg_maps[:,11,:]-eigenvec_fg_Nfg_11@eigenvec_fg_Nfg_11.T@need_fg_maps[:,11,:]
#leak_fg_maps[12] = need_fg_maps[:,12,:]-eigenvec_fg_Nfg_12@eigenvec_fg_Nfg_12.T@need_fg_maps[:,12,:]

#np.save(out_dir_output_PCA+f'leak_PCA_fg_sync_ff_ps_jmax{jmax}_lmax{lmax}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{num_sources}.npy',leak_fg_maps)

fig = plt.figure()
plt.suptitle(f'Frequency channel: {nu_ch[ich]} MHz, Nfg:{num_sources}, jmax:{jmax}, lmax:{lmax} ')
hp.mollview(leak_fg_maps.sum(axis=0)[ich], title='Foregroud leakage', cmap = 'viridis', min=0, max=0.5, hold=True)
#plt.savefig(out_dir_plot+f'betajk_leak_fg_jmax{jmax}_lmax{lmax}_nside_{nside}_Nfg{num_sources}.png')
plt.show()
del leak_fg_maps


need_HI_maps_filename = need_dir+f'bjk_maps_HI_noise_{num_freq}freq_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}.npy'
need_HI_maps = np.load(need_HI_maps_filename)


leak_HI_maps[0] = eigenvec_fg_Nfg_0@eigenvec_fg_Nfg_0.T@need_HI_maps[:,0,:]
leak_HI_maps[1] = eigenvec_fg_Nfg_1@eigenvec_fg_Nfg_1.T@need_HI_maps[:,1,:]
leak_HI_maps[2] = eigenvec_fg_Nfg_2@eigenvec_fg_Nfg_2.T@need_HI_maps[:,2,:]
leak_HI_maps[3] = eigenvec_fg_Nfg_3@eigenvec_fg_Nfg_3.T@need_HI_maps[:,3,:]
leak_HI_maps[4] = eigenvec_fg_Nfg_4@eigenvec_fg_Nfg_4.T@need_HI_maps[:,4,:]
#leak_HI_maps[5] = eigenvec_fg_Nfg_5@eigenvec_fg_Nfg_5.T@need_HI_maps[:,5,:]
#leak_HI_maps[6] = eigenvec_fg_Nfg_6@eigenvec_fg_Nfg_6.T@need_HI_maps[:,6,:]
#leak_HI_maps[7] = eigenvec_fg_Nfg_7@eigenvec_fg_Nfg_7.T@need_HI_maps[:,7,:]
#leak_HI_maps[8] = eigenvec_fg_Nfg_8@eigenvec_fg_Nfg_8.T@need_HI_maps[:,8,:]
#leak_HI_maps[9] = eigenvec_fg_Nfg_9@eigenvec_fg_Nfg_9.T@need_HI_maps[:,9,:]
#leak_HI_maps[10] = eigenvec_fg_Nfg_10@eigenvec_fg_Nfg_10.T@need_HI_maps[:,10,:]
#leak_HI_maps[11] = eigenvec_fg_Nfg_11@eigenvec_fg_Nfg_11.T@need_HI_maps[:,11,:]
#leak_HI_maps[12] = eigenvec_fg_Nfg_12@eigenvec_fg_Nfg_12.T@need_HI_maps[:,12,:]

del eigenvec_fg_Nfg_0; del eigenvec_fg_Nfg_1; del eigenvec_fg_Nfg_2; del eigenvec_fg_Nfg_3#; del eigenvec_fg_Nfg_4
 
#np.save(out_dir_output_PCA+f'leak_PCA_HI_jmax{jmax}_lmax{lmax}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{num_sources}.npy',leak_HI_maps)

fig = plt.figure()
plt.suptitle(f'Frequency channel: {nu_ch[ich]} MHz, Nfg:{num_sources}, jmax:{jmax}, lmax:{lmax} ')
hp.mollview(leak_HI_maps.sum(axis=0)[ich], title='HI leakage', cmap = 'viridis', min=0, max=0.5, hold=True)
#plt.savefig(out_dir_plot+f'betajk_leak_HI_jmax{jmax}_{lmax}_nside_{nside}_Nfg{num_sources}.png')
plt.show()
#
#fig = plt.figure()
#plt.suptitle(f'Frequency channel: {nu_ch[ich]} MHz, Nfg:{num_sources}, jmax:{jmax}, lmax:{lmax} ')
#fig.add_subplot(211)
#hp.mollview(leak_fg_maps.sum(axis=0)[ich], title='Foregroud leakage', cmap = 'viridis', min=0, max=0.5, hold=True)
#fig.add_subplot(212)
#hp.mollview(leak_HI_maps.sum(axis=0)[ich], title='HI leakage', cmap = 'viridis', min=0, max=0.5, hold=True)
#plt.savefig(out_dir_plot+f'betajk_leak_fg_HI_jmax{jmax}_lmax{lmax}_nside_{nside}_Nfg{num_sources}.png')
#plt.show()

del leak_HI_maps; #del leak_fg_maps


##############################################


need_HI_maps_totj = need_HI_maps.sum(axis=1)
need_fg_maps_jtot = need_fg_maps.sum(axis=1)

del need_HI_maps;del need_fg_maps

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}',fontsize=20)
fig.add_subplot(221)
hp.mollview(np.abs(res_fg_maps_totj[ich]/need_fg_maps_jtot[ich]-1)*100, min=0, max=0.1,cmap='viridis', title=f'%(Res fg/Fg)-1',unit='%' ,hold=True)
fig.add_subplot(222)
hp.mollview(np.abs(res_fg_maps_3_totj[ich]/need_fg_maps_jtot[ich]-1)*100, min=0, max=0.1,cmap='viridis', title=f'%(Res fg/Fg)-1 Nfg 3',unit='%' ,hold=True)
fig.add_subplot(223) 
hp.mollview(need_HI_maps_totj[ich], cmap='viridis', title=f'HI signal',min=0, max =1,hold=True)
fig.add_subplot(224)
hp.mollview(res_HI_maps_totj[ich], title=f'PCA HI',min=0, max =1,cmap='viridis', hold=True)
#plt.savefig(out_dir_plot+f'betajk_res_need_PCA_sumj_Nfg{num_sources}_jmax{jmax}_lmax{lmax}_nside{nside}.png')
plt.show()

##############################################################
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

fig = plt.figure()
plt.title(f'NEEDLETS CLs: channel:{nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}')
plt.plot(ell[2:],factor[2:]*cl_cosmo_HI[ich][2:], 'k--',label='Cosmo')
plt.plot(ell[2:],factor[2:]*cl_PCA_HI_need2harm[ich][2:],'+', mfc='none',label='PCA HI')
plt.plot(ell[2:],factor[2:]*cl_PCA_HI_3_need2harm[ich][2:], '+',mfc='none',label='PCA HI Nfg 3')
#plt.plot(ell[2:],factor[2:]*cl_fg[ich][2:], label='Fg')
plt.xlim([0,200])
#plt.ylim([-10,10])
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: mean over channels, jmax:{jmax}, lmax:{lmax}, Nfg:{num_sources}')
plt.plot(ell[2:], factor[2:]*cl_cosmo_HI.mean(axis=0)[2:], 'k--',label = f'Cosmo')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_need2harm.mean(axis=0)[2:],'+',mfc='none', label = f'PCA')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_3_need2harm.mean(axis=0)[2:],'+',mfc='none', label = f'PCA Nfg 3')
#plt.plot(ell[2:], factor[2:]*cl_fg.mean(axis=0)[2:],'+',mfc='none', label = f'Fg')
plt.xlim([0,200])
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,200+1, 30))


diff_cl_need2sphe = cl_PCA_HI_need2harm/cl_cosmo_HI-1
diff_cl_3_need2sphe = cl_PCA_HI_3_need2harm/cl_cosmo_HI-1
#diff_cl_fg = cl_PCA_HI_need2harm/cl_fg-1
del cl_PCA_HI_need2harm; del cl_cosmo_HI
frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe.mean(axis=0)[2:]*100, label='% PCA_HI/input_HI -1')
plt.plot(ell[2:], diff_cl_3_need2sphe.mean(axis=0)[2:]*100, label='% PCA_HI/input_HI -1 Nfg 3')
#plt.plot(ell[2:], diff_cl_fg.mean(axis=0)[2:]*100, label='% PCA_HI/Fg -1')
frame2.set_xlim([0,200])
frame2.set_ylim([-10,10])
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
frame2.set_xticks(np.arange(1,200+1, 30))
#plt.tight_layout()
plt.legend()
#plt.savefig(out_dir_plot+f'cls_need2pix_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.png')

plt.show()

del diff_cl_need2sphe