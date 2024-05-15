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

from needlets_analysis import analysis
import cython_mylibc as pippo


nu_ch = np.linspace(901.0, 1299.0, 200)
num_freq = len(nu_ch)

ich=45
HI_lkg = np.load('HI_leak_200_901.0_1299.0MHz_Nfg3.npy')

Nfg=3
nside=128
npix=hp.nside2npix(nside)
lmax=256#3*nside-1
betajk_dir = './Need_betajk_PCA/'
out_dir = './Maps_betajk2harm/'

jmax=8

need_analysis_HI_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_lkg)

B = need_analysis_HI_lkg.B
del need_analysis_HI_lkg; 

map_leak_HI_need2harm = np.load(out_dir+f'maps_reconstructed_PCA_leak_HI_jmax{jmax}_Nfg{Nfg}_lmax{lmax}_nside{nside}.npy')

fig = plt.figure(figsize=(10, 7))
plt.suptitle(f'Need2harm, jmax:{jmax}, mean over channel',fontsize=20)
fig.add_subplot(211) 
hp.mollview(map_leak_HI_need2harm.mean(axis=0),min=0, max=0.1, title= 'Leakage HI',cmap='viridis', hold= True)
fig.add_subplot(212)
hp.mollview(HI_lkg.mean(axis=0)-map_leak_HI_need2harm.mean(axis=0),min=0, max=0.1, title= 'HI Healpix - HI recons',cmap='viridis', hold=True)
plt.show()



####################################################################################

lmax_cl = 256

cl_HI_leak_Nfg=np.zeros((num_freq, lmax_cl+1))
cl_leak_HI_need2harm_jmax8 = np.zeros((num_freq, lmax_cl+1))

for i in range(num_freq):
    cl_leak_HI_need2harm_jmax8[i] = hp.anafast(map_leak_HI_need2harm[i], lmax=lmax_cl,use_pixel_weights=True, iter=3)
    cl_HI_leak_Nfg[i]=hp.anafast(HI_lkg[i], lmax=lmax_cl,use_pixel_weights=True, iter=3)

del map_leak_HI_need2harm; del HI_lkg

fig = plt.figure()
plt.suptitle(f'Channel:{nu_ch[ich]}')
plt.semilogy(cl_leak_HI_need2harm_jmax8[ich][1:], label = 'Need2Harm')
plt.semilogy(cl_HI_leak_Nfg[ich][1:], label = 'Carucci')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}$')
plt.legend()
plt.tight_layout()
plt.show()

fig = plt.figure()
plt.plot(cl_leak_HI_need2harm_jmax8[ich][1:]/cl_HI_leak_Nfg[ich][1:]-1)
plt.xlabel(r'$\ell$')
plt.ylabel(f'% rel diff')
plt.axhline(y=0, ls='--', color='grey')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.tight_layout()
plt.show()

rs = np.random.randint(543543,size=4)
print(rs)
cls2map_need2harm = np.zeros((len(rs), npix))
cls2map = np.zeros((len(rs), npix))
for n in range(len(rs)):
    np.random.seed(rs[n])
    print(rs[n])
    cls2map_need2harm[n] = hp.synfast(cls=cl_leak_HI_need2harm_jmax8[ich], nside=nside)
    np.random.seed(rs[n])
    print(rs[n])
    cls2map[n] = hp.synfast(cls=cl_HI_leak_Nfg[ich], nside=nside)

fig = plt.figure()
fig.suptitle(f'Need2harm, jmax=8, channel={nu_ch[ich]} MHz')
fig.add_subplot(221) 
hp.mollview(cls2map_need2harm[0], title=f'seed={rs[0]}',  cmap='viridis', min=0, max=0.1, hold=True  )
fig.add_subplot(222) 
hp.mollview(cls2map_need2harm[1], title=f'seed={rs[1]}',  cmap='viridis', min=0, max=0.1, hold=True  )
fig.add_subplot(223) 
hp.mollview(cls2map_need2harm[2], title=f'seed={rs[2]}',  cmap='viridis', min=0, max=0.1, hold=True  )
fig.add_subplot(224) 
hp.mollview(cls2map_need2harm[3], title=f'seed={rs[3]}',  cmap='viridis', min=0, max=0.1, hold=True  )
#plt.savefig(f'maps_need2harm_HI_leak_seeds_ch{nu_ch[ich]}_Nfg3_jmax8_lmax383_nside128.png')

fig = plt.figure()
fig.suptitle(f'Carucci s map, Channel={nu_ch[ich]} MHz')
fig.add_subplot(221) 
hp.mollview(cls2map[0], title=f'seed={rs[0]}',  cmap='viridis', min=0, max=0.1, hold=True  )
fig.add_subplot(222) 
hp.mollview(cls2map[1], title=f'seed={rs[1]}',  cmap='viridis', min=0, max=0.1, hold=True  )
fig.add_subplot(223) 
hp.mollview(cls2map[2], title=f'seed={rs[2]}',  cmap='viridis', min=0, max=0.1, hold=True  )
fig.add_subplot(224) 
hp.mollview(cls2map[3], title=f'seed={rs[3]}',  cmap='viridis', min=0, max=0.1, hold=True  )
#plt.savefig(f'maps_PCA_HI_leak_seeds_ch{nu_ch[ich]}_Nfg3_lmax383_nside128.png')
plt.show()

###Check veloce dei momenti come descritti da healpix###

min_cls2map = np.min(cls2map[0])
max_cls2map = np.max(cls2map[0])
mean_cls2map = np.min(cls2map[0])
std_cls2map = np.std(cls2map[0])
var_cls2map = np.var(cls2map[0])
print(f'Carucci, ch:{nu_ch[ich]}:\nmin:{min_cls2map:1.5f}, max:{max_cls2map:1.5f}, mean:{mean_cls2map:1.5f}, std:{std_cls2map:1.5f}, var:{var_cls2map:1.5f}')

min_cls2map_need2harm = np.min(cls2map_need2harm[0])
max_cls2map_need2harm = np.max(cls2map_need2harm[0])
mean_cls2map_need2harm = np.min(cls2map_need2harm[0])
std_cls2map_need2harm = np.std(cls2map_need2harm[0])
var_cls2map_need2harm = np.var(cls2map_need2harm[0])
print(f'Need2harm,jmax:{jmax},ch:{nu_ch[ich]}:\nmin:{min_cls2map_need2harm:1.5f}, max:{max_cls2map_need2harm:1.5f}, mean:{mean_cls2map_need2harm:1.5f}, std:{std_cls2map_need2harm:1.5f}, var:{var_cls2map_need2harm:1.5f}')