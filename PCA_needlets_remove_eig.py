import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

import seaborn as sns
sns.set()
sns.set(style = 'white')
#sns.set_palette('husl',15)

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

#print(sns.color_palette("husl", 15).as_hex())
sns.palettes.color_palette()

path_data_sims_tot = 'Sims/sims_synch_ff_ps_40freq_905.0_1295.0MHz_nside256'
with open(path_data_sims_tot+'.pkl', 'rb') as f:
        file = pickle.load(f)
        f.close()

out_dir_output = 'PCA_needlets_output/'
out_dir_output_PCA = out_dir_output+'PCA_maps/Rm_eig/'
out_dir_plot = out_dir_output+'Plots_PCA_needlets/Rm_eig/'
if not os.path.exists(out_dir_output):
        os.makedirs(out_dir_output)
if not os.path.exists(out_dir_output_PCA):
        os.makedirs(out_dir_output_PCA)

nu_ch= file['freq']
del file

fg_comp = 'synch_ff_ps'

need_dir = 'Maps_needlets/'
need_tot_maps_filename = need_dir+f'bjk_maps_obs_{fg_comp}_40freq_905.0_1295.0MHz_jmax12_lmax512_B1.68_nside256.npy'
need_tot_maps_o = np.load(need_tot_maps_filename)

jmax=need_tot_maps_o.shape[1]-1

need_tot_maps = need_tot_maps_o[:,:jmax-1,:]
del need_tot_maps_o

num_freq = need_tot_maps.shape[0]
nu_ch = np.linspace(905.0, 1295.0, num_freq)
min_ch = min(nu_ch)
max_ch = max(nu_ch)
npix = need_tot_maps.shape[2]
nside = hp.npix2nside(npix)
lmax=2*nside#3*nside-1#
B=pow(lmax,(1./jmax))


print(f'jmax:{jmax}, lmax:{lmax}, B:{B:1.2f}, num_freq:{num_freq}, min_ch:{min_ch}, max_ch:{max_ch}, nside:{nside}')


#np.save(out_dir_output+f'need_HI_maps_{num_freq}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz.npy',need_HI_maps)

Cov_channels = np.zeros((need_tot_maps.shape[1],num_freq, num_freq))
#Corr_channels=np.zeros((jmax+1,num_freq, num_freq))

for j in range(need_tot_maps.shape[1]):
    Cov_channels[j]=np.cov(need_tot_maps[:,j,:])
    #Corr_channels[j]=np.corrcoef(need_tot_maps[:,j,:])

eigenval=np.zeros((need_tot_maps.shape[1], num_freq))
eigenvec=np.zeros((need_tot_maps.shape[1], num_freq,num_freq))
for j in range(need_tot_maps.shape[1]):
    eigenval[j], eigenvec[j] = np.linalg.eigh(Cov_channels[j])#np.linalg.eigh(Cov_channels[j])
del Cov_channels

fig = plt.figure(figsize=(8,4))
for j in range(need_tot_maps.shape[1]):
    plt.semilogy(np.arange(1,num_freq+1),eigenval[j],'--o',mfc='none',markersize=5,label=f'j={j}')

plt.legend(fontsize=12, ncols=2)
x_ticks = np.arange(-10,num_freq+10, 10)
ax = plt.gca()
ax.set(xlim=[-10,num_freq+10],xticks=x_ticks,xlabel="eigenvalue number",ylabel="$\\lambda$",title='Eigenvalues')
#plt.savefig(out_dir_plot+f'eigenvalue_cov_need_jmax{jmax}_lmax{lmax}_nside{nside}.png')
plt.show()

num_sources=3

Nfg = num_freq - num_sources
print(f'Nfg:{num_sources}')
print(eigenvec.shape)
eigenvec_fg_Nfg = eigenvec[:, :,Nfg:num_freq]

del eigenvec, eigenval

rm_eig = 1 #ultimo

res_fg_maps= np.zeros(( need_tot_maps.shape[1], num_freq, npix))#tolgo anche il primo

for j in range(need_tot_maps.shape[1]):
    res_fg_maps[j] = eigenvec_fg_Nfg[j]@eigenvec_fg_Nfg[j].T@need_tot_maps[:,j,:]

np.save(out_dir_output_PCA+f'res_PCA_fg_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{num_sources}_nside{nside}.npy',res_fg_maps)

print('.. ho calcolato res fg .. ')

ich= int(num_freq/2)
j_test=7

res_HI_maps = np.zeros_like(res_fg_maps)
for j in range(need_tot_maps.shape[1]):
    res_HI_maps[j] = need_tot_maps[:,j,:] - res_fg_maps[j]
    #hp.mollview(res_HI_maps[j][ich],min=0, max=0.2, cmap='viridis', title=f'j={j}')
#plt.show()
del need_tot_maps


np.save(out_dir_output_PCA+f'res_PCA_HI_{fg_comp}_jmax{jmax}_lmax{lmax}_{num_freq}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{num_sources}_nside{nside}.npy',res_HI_maps)

print('.. ho calcolato res HI .. ')

res_fg_maps_totj=res_fg_maps.sum(axis=0)
res_HI_maps_totj = res_HI_maps.sum(axis=0)
del res_HI_maps; del res_fg_maps


print('fin qui ci sono')

############### leakage ###########################
#Foreground's maps

need_fg_maps_filename = need_dir+f'bjk_maps_fg_{fg_comp}_{num_freq}freq_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}.npy'
need_fg_maps = np.load(need_fg_maps_filename)[:,:jmax-1,:]

leak_fg_maps = np.zeros((need_fg_maps.shape[1], num_freq, npix))

for j in range(need_fg_maps.shape[1]):
    leak_fg_maps[j] = need_fg_maps[:,j,:]-eigenvec_fg_Nfg[j]@eigenvec_fg_Nfg[j].T@need_fg_maps[:,j,:]
np.save(out_dir_output_PCA+f'leak_PCA_{fg_comp}_jmax{jmax}_lmax{lmax}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{num_sources}_nside{nside}.npy',leak_fg_maps)

fig = plt.figure()
plt.suptitle(f'Frequency channel: {nu_ch[ich]} MHz, Nfg:{num_sources}, jmax:{jmax}, lmax:{lmax} ')
hp.mollview(leak_fg_maps.sum(axis=0)[ich], title='Foregroud leakage', cmap = 'viridis', min=0, max=0.5, hold=True)
#plt.savefig(out_dir_plot+f'betajk_leak_fg_jmax{jmax}_lmax{lmax}_nside_{nside}_Nfg{num_sources}.png')
plt.show()
del leak_fg_maps


need_HI_maps_filename = need_dir+f'bjk_maps_HI_{num_freq}freq_{min_ch}_{max_ch}MHz_jmax{jmax}_lmax{lmax}_B{B:1.2f}_nside{nside}.npy'
need_HI_maps = np.load(need_HI_maps_filename)[:,:jmax-1,:]

leak_HI_maps = np.zeros((need_HI_maps.shape[1], num_freq, npix))

for j in range(need_HI_maps.shape[1]):
    leak_HI_maps[j] = eigenvec_fg_Nfg[j]@eigenvec_fg_Nfg[j].T@need_HI_maps[:,j,:]

#del eigenvec_fg_Nfg; 
np.save(out_dir_output_PCA+f'leak_PCA_HI_{fg_comp}_jmax{jmax}_lmax{lmax}_{int(min(nu_ch))}_{int(max(nu_ch))}MHz_Nfg{num_sources}_nside{nside}.npy',leak_HI_maps)

fig = plt.figure()
plt.suptitle(f'Frequency channel: {nu_ch[ich]} MHz, Nfg:{num_sources}, jmax:{jmax}, lmax:{lmax} ')
hp.mollview(leak_HI_maps.sum(axis=0)[ich], title='HI leakage', cmap = 'viridis', min=0, max=0.5, hold=True)
#plt.savefig(out_dir_plot+f'betajk_leak_HI_jmax{jmax}_{lmax}_nside_{nside}_Nfg{num_sources}.png')
plt.show()


del leak_HI_maps#; del leak_fg_maps


##############################################
need_HI_maps_new = np.zeros((need_HI_maps.shape[1], num_freq, npix) )
need_fg_maps_new = np.zeros((need_HI_maps.shape[1], num_freq, npix) )


for j in range(need_HI_maps.shape[1]): 
    need_HI_maps_new[j] = need_HI_maps[:,j,:]#need_HI_maps.sum(axis=1)
    need_fg_maps_new[j] = need_fg_maps[:,j,:]#need_fg_maps.sum(axis=1)
need_HI_maps_totj=need_HI_maps_new.sum(axis=0)
need_fg_maps_totj=need_fg_maps_new.sum(axis=0)
del need_HI_maps;del need_fg_maps

fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}',fontsize=20)
fig.add_subplot(222) 
hp.mollview(need_HI_maps_totj[ich], cmap='viridis', title=f'HI signal',min=0, max =1,hold=True)
fig.add_subplot(223)
hp.mollview(res_HI_maps_totj[ich], title=f'PCA HI',min=0, max =1,cmap='viridis', hold=True)
fig.add_subplot(221)
hp.mollview(np.abs(res_fg_maps_totj[ich]/need_fg_maps_totj[ich]-1)*100, min=0, max=0.1,cmap='viridis', title=f'%(Res fg/Fg)-1',unit='%' ,hold=True)
#plt.savefig(out_dir_plot+f'betajk_res_need_PCA_sumj_Nfg{num_sources}_jmax{jmax}_lmax{lmax}_nside{nside}.png')
plt.show()

del res_fg_maps_totj; del need_fg_maps_totj

###################### PROVIAMO CON I CL #########################

lmax_cl = 300

cl_cosmo_HI = np.zeros((len(nu_ch), lmax_cl+1))
cl_PCA_HI_need2harm = np.zeros((len(nu_ch), lmax_cl+1))

for n in range(len(nu_ch)):
    cl_cosmo_HI[n]=hp.anafast(need_HI_maps_totj[n], lmax=lmax_cl)
    cl_PCA_HI_need2harm[n] = hp.anafast(res_HI_maps_totj[n], lmax=lmax_cl)

del need_HI_maps_totj; del res_HI_maps_totj

ell=np.arange(lmax_cl+1)
factor=ell*(ell+1)/(2*np.pi)

fig = plt.figure()
plt.title(f'NEEDLETS CLs: channel:{nu_ch[ich]} MHz, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}')
plt.semilogy(ell[2:],factor[2:]*cl_PCA_HI_need2harm[ich][2:], label='PCA HI')
plt.semilogy(ell[2:],factor[2:]*cl_cosmo_HI[ich][2:], label='Cosmo')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'NEEDLETS CLs: mean over channels, jmax:{jmax}, lmax:{lmax}, Nfg:{Nfg}')
plt.plot(ell[2:], factor[2:]*cl_cosmo_HI.mean(axis=0)[2:], label = f'Cosmo')
plt.plot(ell[2:], factor[2:]*cl_PCA_HI_need2harm.mean(axis=0)[2:],'+',mfc='none', label = f'PCA')
plt.legend()
frame1.set_ylabel(r'$\frac{\ell(\ell+1)}{2\pi} \langle C_{\ell} \rangle_{\rm ch}$')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,lmax_cl+1, 30))


diff_cl_need2sphe = cl_PCA_HI_need2harm/cl_cosmo_HI-1
del cl_PCA_HI_need2harm; del cl_cosmo_HI
frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(ell[2:], diff_cl_need2sphe.mean(axis=0)[2:]*100, label='% PCA_HI/input_HI -1')
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel(r'$\ell$')
frame2.set_xticks(np.arange(1,lmax_cl+1, 30))
plt.tight_layout()
plt.legend()
#plt.savefig(out_dir_plot+f'cls_need2pix_jmax{jmax}_lmax{lmax}_nside{nside}_Nfg{Nfg}.png')

plt.show()

del diff_cl_need2sphe

