import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
sns.set(style = 'white')
#sns.set_palette('husl',15)

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True, bottom = True)
mpl.rc('ytick', direction='in', right=True, left = True)

#print(sns.color_palette("husl").as_hex())
sns.palettes.color_palette()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

nside=128
lmax=256#3*nside-1

dir_est = './Need_est_ps/'
out_dir_plot = './Plots_need_est/'

nu_ch = np.linspace(901.0, 1299.0, 200)

ich=100
#################################################################################################
################################### jmax=15 #####################################################

jmax=15
B_15 = 1.45#1.49
HI_PCA_need_est=np.loadtxt(dir_est+f'need_est_ps_HI_PCA_Nfg3_jmax{jmax}_B{B_15:0.2f}_lmax{lmax}_nside{nside}.dat' )
HI_cosmo_need_est=np.loadtxt(dir_est+f'need_est_ps_cosmo_HI_jmax{jmax}_B{B_15:0.2f}_lmax{lmax}_nside{nside}.dat' )
fg_input_need_est=np.loadtxt(dir_est+f'need_est_ps_fg_input_jmax{jmax}_B{B_15:0.2f}_lmax{lmax}_nside{nside}.dat' )
fg_lkg_need_est=np.loadtxt(dir_est+f'need_est_ps_fg_leak_PCA_Nfg3_jmax{jmax}_B{B_15:0.2f}_lmax{lmax}_nside{nside}.dat' )
HI_lkg_need_est=np.loadtxt(dir_est+f'need_est_ps_HI_leak_PCA_Nfg3_jmax{jmax}_B{B_15:0.2f}_lmax{lmax}_nside{nside}.dat' )


fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'mean over channels, B:{B_15:0.2f}, jmax:{jmax}, lmax:{lmax}')
plt.plot(np.arange(1,jmax+1), HI_PCA_need_est.mean(axis=0)[1:], label = f'PCA')
plt.plot(np.arange(1,jmax+1), HI_cosmo_need_est.mean(axis=0)[1:],label = f'Cosmo')#c= '#3ba3ec', label = f'Cosmo')
plt.legend()
frame1.set_ylabel(r'$\langle \beta_j^{\rm HI} \rangle_{\rm ch} $')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,jmax+1))


diff_j15 = HI_PCA_need_est/HI_cosmo_need_est-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(np.arange(1,jmax+1), diff_j15.mean(axis=0)[1:]*100)
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel('j')
frame2.set_xticks(np.arange(1,jmax+1))
plt.savefig(out_dir_plot+f'need_est_PCA_cosmo_HI_jmax{jmax}_B{B_15:1.2f}_lmax{lmax}.png')
plt.show()


fig = plt.figure()
plt.title(f'mean over channels, B:{B_15:0.2f}, jmax:{jmax}, lmax:{lmax}')
#plt.semilogy(np.arange(1,jmax+1), fg_input_need_est.mean(axis=0)[1:], label = f'Foreground')
plt.semilogy(np.arange(1,jmax+1), fg_lkg_need_est.mean(axis=0)[1:],label = f'Foreground leakage')#c= '#3ba3ec', label = f'Cosmo')
plt.semilogy(np.arange(1,jmax+1), HI_lkg_need_est.mean(axis=0)[1:],label = f'HI leakage')#c= '#3ba3ec', label = f'Cosmo')
plt.xlabel('j')
plt.ylabel(r'$\langle \beta_j \rangle_{\rm ch} $')
plt.xticks(np.arange(1,jmax+1))
plt.legend()
plt.savefig(out_dir_plot+f'need_est_leak_fg_HI_Nfg3_jmax{jmax}_B{B_15:1.2f}_lmax{lmax}.png')
plt.show()

ratio_j15_leak_fg = (fg_lkg_need_est/fg_input_need_est)
ratio_j15_leak_HI = (HI_lkg_need_est/HI_cosmo_need_est)

fig=plt.figure()
plt.title(f'mean over channels, B:{B_15:0.2f}, jmax:{jmax}, lmax:{lmax}')
plt.semilogy(np.arange(1,jmax+1),ratio_j15_leak_fg.mean(axis=0)[1:], label = 'Fg leak') 
plt.semilogy(np.arange(1,jmax+1),ratio_j15_leak_HI.mean(axis=0)[1:], label = 'HI leak') 
plt.xlabel('j')
plt.ylabel(r'$\langle \beta_j^{\rm Leak}/\beta_j^{\rm Input}\rangle_{\rm ch} $')
plt.xticks(np.arange(1,jmax+1))
plt.legend()
#plt.savefig(out_dir_plot+f'ratio_need_est_leak_fg_HI_input_Nfg3_jmax{jmax}_B{B_15:1.2f}_lmax{lmax}.png')
plt.show()

fig = plt.figure()
plt.title(f'mean over channels, B:{B_15:0.2f}, jmax:{jmax}, lmax:{lmax}')
#plt.semilogy(np.arange(1,jmax+1), fg_input_need_est.mean(axis=0)[1:], label = f'Foreground')
plt.semilogy(np.arange(1,jmax+1), HI_cosmo_need_est.mean(axis=0)[1:],c=colors[1],ls='-',label = f'HI input')#c= '#3ba3ec', label = f'Cosmo')
plt.semilogy(np.arange(1,jmax+1), fg_input_need_est.mean(axis=0)[1:],c=colors[0],ls='-',label = f'Foreground input')#c= '#3ba3ec', label = f'Cosmo')
plt.semilogy(np.arange(1,jmax+1), HI_lkg_need_est.mean(axis=0)[1:],c=colors[1],ls='--',label = f'HI leakage')#c= '#3ba3ec', label = f'Cosmo')
plt.semilogy(np.arange(1,jmax+1), fg_lkg_need_est.mean(axis=0)[1:],c=colors[0],ls='--',label = f'Foreground leakage')#c= '#3ba3ec', label = f'Cosmo')
plt.xlabel('j')
plt.ylabel(r'$\langle \beta_j \rangle_{\rm ch} $')
plt.ylim([1e-7,1e7])
plt.xlim([0,15+1])
plt.xticks(np.arange(1,15+1))
plt.legend()
plt.savefig(out_dir_plot+f'need_est_leak_input_fg_HI_Nfg3_jmax{jmax}_B{B_15:1.2f}_lmax{lmax}.png')
plt.show()

HI_cosmo_need_est_jmax15 = HI_cosmo_need_est
fg_input_need_est_jmax15 = fg_input_need_est

del HI_PCA_need_est; del HI_cosmo_need_est; del fg_lkg_need_est; del HI_lkg_need_est; del fg_input_need_est
#################################################################################################
################################### jmax=12 #####################################################

jmax=12
B_12 = 1.59#1.64
HI_PCA_need_est=np.loadtxt(dir_est+f'need_est_ps_HI_PCA_Nfg3_jmax{jmax}_B{B_12:0.2f}_lmax{lmax}_nside{nside}.dat' )
HI_cosmo_need_est=np.loadtxt(dir_est+f'need_est_ps_cosmo_HI_jmax{jmax}_B{B_12:0.2f}_lmax{lmax}_nside{nside}.dat' )
fg_input_need_est=np.loadtxt(dir_est+f'need_est_ps_fg_input_jmax{jmax}_B{B_12:0.2f}_lmax{lmax}_nside{nside}.dat' )
fg_lkg_need_est=np.loadtxt(dir_est+f'need_est_ps_fg_leak_PCA_Nfg3_jmax{jmax}_B{B_12:0.2f}_lmax{lmax}_nside{nside}.dat' )
HI_lkg_need_est=np.loadtxt(dir_est+f'need_est_ps_HI_leak_PCA_Nfg3_jmax{jmax}_B{B_12:0.2f}_lmax{lmax}_nside{nside}.dat' )


fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'mean over channels, B:{B_12:0.2f}, jmax:{jmax}, lmax:{lmax}')
plt.plot(np.arange(1,jmax+1), HI_PCA_need_est.mean(axis=0)[1:], label = f'PCA')
plt.plot(np.arange(1,jmax+1), HI_cosmo_need_est.mean(axis=0)[1:],label = f'Cosmo')#c= '#3ba3ec', label = f'Cosmo')
plt.legend()
frame1.set_ylabel(r'$\langle \beta_j^{\rm HI} \rangle_{\rm ch} $')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,jmax+1))


diff_j12 = HI_PCA_need_est/HI_cosmo_need_est-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(np.arange(1,jmax+1), diff_j12.mean(axis=0)[1:]*100)
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel('j')
frame2.set_xticks(np.arange(1,jmax+1))
plt.savefig(out_dir_plot+f'need_est_PCA_cosmo_HI_jmax{jmax}_B{B_12:1.2f}_lmax{lmax}.png')
plt.show()

fig = plt.figure()
plt.title(f'mean over channels, B:{B_12:0.2f}, jmax:{jmax}, lmax:{lmax}')
#plt.semilogy(np.arange(1,jmax+1), fg_input_need_est.mean(axis=0)[1:], label = f'Foreground')
plt.semilogy(np.arange(1,jmax+1), fg_lkg_need_est.mean(axis=0)[1:],label = f'Foreground leakage')#c= '#3ba3ec', label = f'Cosmo')
plt.semilogy(np.arange(1,jmax+1), HI_lkg_need_est.mean(axis=0)[1:],label = f'HI leakage')#c= '#3ba3ec', label = f'Cosmo')
plt.xlabel('j')
plt.ylabel(r'$\langle \beta_j \rangle_{\rm ch} $')
plt.xticks(np.arange(1,jmax+1))
plt.legend()
plt.savefig(out_dir_plot+f'need_est_leak_fg_HI_Nfg3_jmax{jmax}_B{B_12:1.2f}_lmax{lmax}.png')
plt.show()


ratio_j12_leak_fg = (fg_lkg_need_est/fg_input_need_est)
ratio_j12_leak_HI = (HI_lkg_need_est/HI_cosmo_need_est)

fig=plt.figure()
plt.title(f'mean over channels, B:{B_15:0.2f}, jmax:{jmax}, lmax:{lmax}')
plt.semilogy(np.arange(1,jmax+1),ratio_j12_leak_fg.mean(axis=0)[1:], label = 'Fg leak') 
plt.semilogy(np.arange(1,jmax+1),ratio_j12_leak_HI.mean(axis=0)[1:], label = 'HI leak') 
plt.xlabel('j')
plt.ylabel(r'$\langle \beta_j^{\rm Leak}/\beta_j^{\rm Input}\rangle_{\rm ch} $')
plt.xticks(np.arange(1,jmax+1))
plt.legend()
#plt.savefig(out_dir_plot+f'ratio_need_est_leak_fg_HI_input_Nfg3_jmax{jmax}_B{B_12:1.2f}_lmax{lmax}.png')
plt.show()

fig = plt.figure()
plt.title(f'mean over channels, B:{B_12:0.2f}, jmax:{jmax}, lmax:{lmax}')
#plt.semilogy(np.arange(1,jmax+1), fg_input_need_est.mean(axis=0)[1:], label = f'Foreground')
plt.semilogy(np.arange(1,jmax+1), HI_cosmo_need_est.mean(axis=0)[1:],c=colors[1],ls='-',label = f'HI input')#c= '#3ba3ec', label = f'Cosmo')
plt.semilogy(np.arange(1,jmax+1), fg_input_need_est.mean(axis=0)[1:],c=colors[0],ls='-',label = f'Foreground input')#c= '#3ba3ec', label = f'Cosmo')
plt.semilogy(np.arange(1,jmax+1), HI_lkg_need_est.mean(axis=0)[1:],c=colors[1],ls='--',label = f'HI leakage')#c= '#3ba3ec', label = f'Cosmo')
plt.semilogy(np.arange(1,jmax+1), fg_lkg_need_est.mean(axis=0)[1:],c=colors[0],ls='--',label = f'Foreground leakage')#c= '#3ba3ec', label = f'Cosmo')
plt.xlim([0,15+1])
plt.xticks(np.arange(1,15+1))
plt.ylim([1e-7,1e7])
plt.xlabel('j')
plt.ylabel(r'$\langle \beta_j \rangle_{\rm ch} $')
plt.legend()
plt.savefig(out_dir_plot+f'need_est_leak_fg_input_HI_Nfg3_jmax{jmax}_B{B_12:1.2f}_lmax{lmax}.png')
plt.show()

HI_cosmo_need_est_jmax12 = HI_cosmo_need_est
fg_input_need_est_jmax12 = fg_input_need_est

del HI_PCA_need_est; del HI_cosmo_need_est; del fg_lkg_need_est; del HI_lkg_need_est; del fg_input_need_est
#################################################################################################
################################### jmax=8 #####################################################

jmax=8
B_8 = 2#2.10
HI_PCA_need_est=np.loadtxt(dir_est+f'need_est_ps_HI_PCA_Nfg3_jmax{jmax}_B{B_8:0.2f}_lmax{lmax}_nside{nside}.dat' )
HI_cosmo_need_est=np.loadtxt(dir_est+f'need_est_ps_cosmo_HI_jmax{jmax}_B{B_8:0.2f}_lmax{lmax}_nside{nside}.dat' )
fg_input_need_est=np.loadtxt(dir_est+f'need_est_ps_fg_input_jmax{jmax}_B{B_8:0.2f}_lmax{lmax}_nside{nside}.dat' )
fg_lkg_need_est=np.loadtxt(dir_est+f'need_est_ps_fg_leak_PCA_Nfg3_jmax{jmax}_B{B_8:0.2f}_lmax{lmax}_nside{nside}.dat' )
HI_lkg_need_est=np.loadtxt(dir_est+f'need_est_ps_HI_leak_PCA_Nfg3_jmax{jmax}_B{B_8:0.2f}_lmax{lmax}_nside{nside}.dat' )


fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'mean over channels, B:{B_8:0.2f}, jmax:{jmax}, lmax:{lmax}')
plt.plot(np.arange(1,jmax+1), HI_PCA_need_est.mean(axis=0)[1:], label = f'PCA')
plt.plot(np.arange(1,jmax+1), HI_cosmo_need_est.mean(axis=0)[1:],label = f'Cosmo')#c= '#3ba3ec', label = f'Cosmo')
plt.legend()
frame1.set_ylabel(r'$\langle \beta_j^{\rm HI} \rangle_{\rm ch} $')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,jmax+1))


diff_j8 = HI_PCA_need_est/HI_cosmo_need_est-1
frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(np.arange(1,jmax+1), diff_j8.mean(axis=0)[1:]*100)
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel('j')
frame2.set_xticks(np.arange(1,jmax+1))
plt.savefig(out_dir_plot+f'need_est_PCA_cosmo_HI_jmax{jmax}_B{B_8:1.2f}_lmax{lmax}.png')
plt.show()


fig = plt.figure()
plt.title(f'mean over channels, B:{B_8:0.2f}, jmax:{jmax}, lmax:{lmax}')
#plt.semilogy(np.arange(1,jmax+1), fg_input_need_est.mean(axis=0)[1:], label = f'Foreground')
plt.semilogy(np.arange(1,jmax+1), fg_lkg_need_est.mean(axis=0)[1:],label = f'Foreground leakage')#c= '#3ba3ec', label = f'Cosmo')
plt.semilogy(np.arange(1,jmax+1), HI_lkg_need_est.mean(axis=0)[1:],label = f'HI leakage')#c= '#3ba3ec', label = f'Cosmo')
plt.xlabel('j')
plt.ylabel(r'$\langle \beta_j \rangle_{\rm ch} $')
plt.xticks(np.arange(1,jmax+1))
plt.legend()
plt.savefig(out_dir_plot+f'need_est_leak_fg_HI_Nfg3_jmax{jmax}_B{B_8:1.2f}_lmax{lmax}.png')
plt.show()



ratio_j8_leak_fg = (fg_lkg_need_est/fg_input_need_est)
ratio_j8_leak_HI = (HI_lkg_need_est/HI_cosmo_need_est)

fig=plt.figure()
plt.title(f'mean over channels, B:{B_15:0.2f}, jmax:{jmax}, lmax:{lmax}')
plt.semilogy(np.arange(1,jmax+1),ratio_j8_leak_fg.mean(axis=0)[1:], label = 'Fg leak') 
plt.semilogy(np.arange(1,jmax+1),ratio_j8_leak_HI.mean(axis=0)[1:], label = 'HI leak') 
plt.xlabel('j')
plt.ylabel(r'$\langle \beta_j^{\rm Leak}/\beta_j^{\rm Input}\rangle_{\rm ch} $')
plt.xticks(np.arange(1,jmax+1))
plt.legend()
#plt.savefig(out_dir_plot+f'ratio_need_est_leak_fg_HI_input_Nfg3_jmax{jmax}_B{B_8:1.2f}_lmax{lmax}.png')
plt.show()


fig = plt.figure()
plt.title(f'mean over channels, B:{B_8:0.2f}, jmax:{jmax}, lmax:{lmax}')
#plt.semilogy(np.arange(1,jmax+1), fg_input_need_est.mean(axis=0)[1:], label = f'Foreground')
plt.semilogy(np.arange(1,jmax+1), HI_cosmo_need_est.mean(axis=0)[1:],c=colors[1],ls='-',label = f'HI input')#c= '#3ba3ec', label = f'Cosmo')
plt.semilogy(np.arange(1,jmax+1), fg_input_need_est.mean(axis=0)[1:],c=colors[0],ls='-',label = f'Foreground input')#c= '#3ba3ec', label = f'Cosmo')
plt.semilogy(np.arange(1,jmax+1), HI_lkg_need_est.mean(axis=0)[1:],c=colors[1],ls='--',label = f'HI leakage')#c= '#3ba3ec', label = f'Cosmo')
plt.semilogy(np.arange(1,jmax+1), fg_lkg_need_est.mean(axis=0)[1:],c=colors[0],ls='--',label = f'Foreground leakage')#c= '#3ba3ec', label = f'Cosmo')
plt.xlabel('j')
plt.ylabel(r'$\langle \beta_j \rangle_{\rm ch} $')
plt.ylim([1e-7,1e7])
plt.xlim([0,15+1])
plt.xticks(np.arange(1,15+1))
plt.legend()
plt.savefig(out_dir_plot+f'need_est_leak_fg_input_HI_Nfg3_jmax{jmax}_B{B_8:1.2f}_lmax{lmax}.png')
plt.show()

HI_cosmo_need_est_jmax8 = HI_cosmo_need_est
fg_input_need_est_jmax8 = fg_input_need_est

del HI_PCA_need_est; del HI_cosmo_need_est; del fg_lkg_need_est; del HI_lkg_need_est; del fg_input_need_est
##############################################################################################################

jmax=4
B_4 = 4#4.42
HI_PCA_need_est=np.loadtxt(dir_est+f'need_est_ps_HI_PCA_Nfg3_jmax{jmax}_B{B_4:0.2f}_lmax{lmax}_nside{nside}.dat' )
HI_cosmo_need_est=np.loadtxt(dir_est+f'need_est_ps_cosmo_HI_jmax{jmax}_B{B_4:0.2f}_lmax{lmax}_nside{nside}.dat' )
fg_input_need_est=np.loadtxt(dir_est+f'need_est_ps_fg_input_jmax{jmax}_B{B_4:0.2f}_lmax{lmax}_nside{nside}.dat' )
fg_lkg_need_est=np.loadtxt(dir_est+f'need_est_ps_fg_leak_PCA_Nfg3_jmax{jmax}_B{B_4:0.2f}_lmax{lmax}_nside{nside}.dat' )
HI_lkg_need_est=np.loadtxt(dir_est+f'need_est_ps_HI_leak_PCA_Nfg3_jmax{jmax}_B{B_4:0.2f}_lmax{lmax}_nside{nside}.dat' )


fig = plt.figure(figsize=(10,7))
frame1=fig.add_axes((.1,.3,.8,.6))
plt.title(f'mean over channels, B:{B_4:0.2f}, jmax:{jmax}, lmax:{lmax}')
plt.plot(np.arange(1,jmax+1), HI_PCA_need_est.mean(axis=0)[1:], label = f'PCA')
plt.plot(np.arange(1,jmax+1), HI_cosmo_need_est.mean(axis=0)[1:],label = f'Cosmo')#c= '#3ba3ec', label = f'Cosmo')
plt.legend()
frame1.set_ylabel(r'$\langle \beta_j^{\rm HI} \rangle_{\rm ch} $')
frame1.set_xlabel([])
frame1.set_xticks(np.arange(1,jmax+1))


diff_j4 = HI_PCA_need_est/HI_cosmo_need_est-1

frame2=fig.add_axes((.1,.1,.8,.2))
plt.plot(np.arange(1,jmax+1), diff_j4.mean(axis=0)[1:]*100)
frame2.axhline(ls='--', c= 'k', alpha=0.3)
frame2.set_ylabel(r'%$ \langle diff \rangle_{\rm ch}$')
frame2.set_xlabel('j')
frame2.set_xticks(np.arange(1,jmax+1))
plt.savefig(out_dir_plot+f'need_est_PCA_cosmo_HI_jmax{jmax}_B{B_4:1.2f}_lmax{lmax}.png')
plt.show()


fig = plt.figure()
plt.title(f'mean over channels, B:{B_4:0.2f}, jmax:{jmax}, lmax:{lmax}')
#plt.semilogy(np.arange(1,jmax+1), fg_input_need_est.mean(axis=0)[1:], label = f'Foreground')
plt.semilogy(np.arange(1,jmax+1), fg_lkg_need_est.mean(axis=0)[1:],label = f'Foreground leakage')#c= '#3ba3ec', label = f'Cosmo')
plt.semilogy(np.arange(1,jmax+1), HI_lkg_need_est.mean(axis=0)[1:],label = f'HI leakage')#c= '#3ba3ec', label = f'Cosmo')
plt.xlabel('j')
plt.ylabel(r'$\langle \beta_j \rangle_{\rm ch} $')
plt.xticks(np.arange(1,jmax+1))
plt.legend()
plt.savefig(out_dir_plot+f'need_est_leak_fg_HI_Nfg3_jmax{jmax}_B{B_4:1.2f}_lmax{lmax}.png')
plt.show()



ratio_j4_leak_fg = (fg_lkg_need_est/fg_input_need_est)
ratio_j4_leak_HI = (HI_lkg_need_est/HI_cosmo_need_est)

fig=plt.figure()
plt.title(f'mean over channels, B:{B_15:0.2f}, jmax:{jmax}, lmax:{lmax}')
plt.semilogy(np.arange(1,jmax+1),ratio_j4_leak_fg.mean(axis=0)[1:], label = 'Fg leak') 
plt.semilogy(np.arange(1,jmax+1),ratio_j4_leak_HI.mean(axis=0)[1:], label = 'HI leak') 
plt.xlabel('j')
plt.ylabel(r'$\langle \beta_j^{\rm Leak}/\beta_j^{\rm Input}\rangle_{\rm ch} $')
plt.xticks(np.arange(1,jmax+1))
plt.legend()
#plt.savefig(out_dir_plot+f'ratio_need_est_leak_fg_HI_input_Nfg3_jmax{jmax}_B{B_4:1.2f}_lmax{lmax}.png')
plt.show()


fig = plt.figure()
plt.title(f'mean over channels, B:{B_4:0.2f}, jmax:{jmax}, lmax:{lmax}')
plt.semilogy(np.arange(1,jmax+1), HI_cosmo_need_est.mean(axis=0)[1:],c=colors[1],ls='-',label = f'HI input')#c= '#3ba3ec', label = f'Cosmo')
plt.semilogy(np.arange(1,jmax+1), fg_input_need_est.mean(axis=0)[1:],c=colors[0],ls='-',label = f'Foreground input')#c= '#3ba3ec', label = f'Cosmo')
plt.semilogy(np.arange(1,jmax+1), HI_lkg_need_est.mean(axis=0)[1:],c=colors[1],ls='--',label = f'HI leakage')#c= '#3ba3ec', label = f'Cosmo')
plt.semilogy(np.arange(1,jmax+1), fg_lkg_need_est.mean(axis=0)[1:],c=colors[0],ls='--',label = f'Foreground leakage')#c= '#3ba3ec', label = f'Cosmo')
plt.xlabel('j')
plt.ylabel(r'$\langle \beta_j \rangle_{\rm ch} $')
plt.ylim([1e-7,1e7])
plt.xlim([0,15+1])
plt.xticks(np.arange(1,15+1))
plt.legend()
plt.savefig(out_dir_plot+f'need_est_leak_fg_input_HI_Nfg3_jmax{jmax}_B{B_4:1.2f}_lmax{lmax}.png')
plt.show()


HI_cosmo_need_est_jmax4 = HI_cosmo_need_est
fg_input_need_est_jmax4 = fg_input_need_est

del HI_PCA_need_est; del HI_cosmo_need_est; del fg_lkg_need_est; del HI_lkg_need_est; del fg_input_need_est
################################################################################################################
fig = plt.figure()
plt.title(r'Ratio Leak$_{\rm HI}$ over HI$_{\rm input}$, mean over channels, lmax:%d'%lmax)
plt.plot(np.arange(1,5),ratio_j4_leak_HI.mean(axis=0)[1:], c=colors[0], label = 'jmax = 4')
plt.plot(np.arange(1,9),ratio_j8_leak_HI.mean(axis=0)[1:], c=colors[1], label = 'jmax = 8')
plt.plot(np.arange(1,13),ratio_j12_leak_HI.mean(axis=0)[1:], c=colors[2], label = 'jmax = 12')
plt.plot(np.arange(1,16),ratio_j15_leak_HI.mean(axis=0)[1:], c=colors[3], label = 'jmax = 15')
#plt.axhline(ls='--', c= 'k', alpha=0.3)
plt.xlabel('j')
plt.xticks(np.arange(1, 16))
plt.ylabel(r'$\langle \beta_j^{\rm Leak_{HI}}/\beta_j^{\rm HI_{input}}\rangle_{\rm ch} $')
plt.legend()
plt.savefig(out_dir_plot+f'ratio_HI_leak_input_jmax_need_est_PCA_cosmo_HI_lmax{lmax}.png')
plt.show()


fig = plt.figure()
plt.title(r'Ratio Leak$_{\rm fg}$ over Fg$_{\rm input}$, mean over channels, lmax:%d'%lmax)
plt.plot(np.arange(1,5),ratio_j4_leak_fg.mean(axis=0)[1:], c=colors[0], ls='-', label = 'jmax = 4')
plt.plot(np.arange(1,9),ratio_j8_leak_fg.mean(axis=0)[1:], c=colors[1], ls='-', label = 'jmax = 8')
plt.plot(np.arange(1,13),ratio_j12_leak_fg.mean(axis=0)[1:], c=colors[2], ls='-', label = 'jmax = 12')
plt.plot(np.arange(1,16),ratio_j15_leak_fg.mean(axis=0)[1:], c=colors[3], ls='-', label = 'jmax = 15')
plt.xlabel('j')
plt.xticks(np.arange(1, 16))
plt.ylabel(r'$\langle \beta_j^{\rm Leak_{fg}}/\beta_j^{\rm Fg_{input}}\rangle_{\rm ch} $')
plt.legend()
plt.savefig(out_dir_plot+f'ratio_fg_leak_input_jmax_need_est_PCA_cosmo_HI_lmax{lmax}.png')
plt.show()


fig = plt.figure()
plt.title(r'Percentage relative diff Res$_{\rm HI}$ and HI$_{\rm input}$, mean over channels, lmax:%d'%lmax)
plt.plot(np.arange(1,4+1),diff_j4.mean(axis=0)[1:]*100, label = 'jmax = 4')
plt.plot(np.arange(1,8+1),diff_j8.mean(axis=0)[1:]*100, label = 'jmax = 8')
plt.plot(np.arange(1,13),diff_j12.mean(axis=0)[1:]*100, label = 'jmax = 12')
plt.plot(np.arange(1,16),diff_j15.mean(axis=0)[1:]*100, label = 'jmax = 15')
plt.axhline(ls='--', c= 'k', alpha=0.3)
plt.xlabel('j')
plt.xticks(np.arange(1, 16))
plt.ylabel(r'%$ \langle \beta_j^{\rm Res_{HI}}/\beta_j^{\rm HI_{input}} -1 \rangle_{\rm ch}$')
plt.legend()
plt.savefig(out_dir_plot+f'diff_jmax_need_est_PCA_cosmo_HI_lmax{lmax}.png')
plt.show()


fig = plt.figure()
plt.title('Input HI')
plt.semilogy(np.arange(1,5),HI_cosmo_need_est_jmax4.mean(axis=0)[1:]*100, label = 'jmax = 4')
plt.semilogy(np.arange(1,9),HI_cosmo_need_est_jmax8.mean(axis=0)[1:]*100, label = 'jmax = 8')
plt.semilogy(np.arange(1,13),HI_cosmo_need_est_jmax12.mean(axis=0)[1:]*100, label = 'jmax = 12')
plt.semilogy(np.arange(1,16),HI_cosmo_need_est_jmax15.mean(axis=0)[1:]*100, label = 'jmax = 15')
plt.xlabel('j')
plt.ylabel(r'$\langle \beta_j \rangle_{\rm ch} $')
plt.ylim([1e-7,1e7])
plt.xlim([0,15+1])
plt.xticks(np.arange(1,15+1))
plt.legend()
plt.savefig(out_dir_plot+f'need_est_input_HI_Nfg3_jmax_lmax{lmax}.png')
plt.show()


fig = plt.figure()
plt.title('Input Fg')
plt.semilogy(np.arange(1,5),fg_input_need_est_jmax4.mean(axis=0)[1:]*100, label = 'jmax = 4')
plt.semilogy(np.arange(1,9),fg_input_need_est_jmax8.mean(axis=0)[1:]*100, label = 'jmax = 8')
plt.semilogy(np.arange(1,13),fg_input_need_est_jmax12.mean(axis=0)[1:]*100, label = 'jmax = 12')
plt.semilogy(np.arange(1,16),fg_input_need_est_jmax15.mean(axis=0)[1:]*100, label = 'jmax = 15')
plt.xlabel('j')
plt.ylabel(r'$\langle \beta_j \rangle_{\rm ch} $')
#plt.ylim([1e-7,1e7])
plt.xlim([0,15+1])
plt.xticks(np.arange(1,15+1))
plt.legend()
plt.savefig(out_dir_plot+f'need_est_input_fg_Nfg3_jmax_lmax{lmax}.png')
plt.show()