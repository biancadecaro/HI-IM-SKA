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



nu_ch = np.linspace(901.0, 1299.0, 200)
num_freq = len(nu_ch)

ich=100


#res_HI = np.load('res_PCA_HI_200_901.0_1299.0MHz_Nfg3.npy')
#cosmo_HI = np.load('cosmo_HI_200_901.0_1299.0MHz.npy')
#fg_input = np.load('fg_input_200_901.0_1299.0MHz.npy')
fg_lkg = np.load('fg_leak_200_901.0_1299.0MHz_Nfg3.npy')
HI_lkg = np.load('HI_leak_200_901.0_1299.0MHz_Nfg3.npy')

hp.mollview(HI_lkg[100],min=0, max=0.1,cmap='viridis', hold=True)
plt.show()

Nfg=3
nside=128
lmax=256#3*nside-1
out_dir = './Need_betajk_PCA/'
######################################################################################################
################################ jmax=15  ############################################################

jmax=15

need_analysis_HI_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_lkg)
need_analysis_fg_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_lkg)

B = need_analysis_fg_lkg.B
print(B)
fname_leak_HI=f'bjk_maps_leak_HI_jmax{jmax}_lmax{lmax}_B{B:0.2f}_Nfg{Nfg}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_nside{nside}'
fname_leak_fg=f'bjk_maps_leak_fg_jmax{jmax}_lmax{lmax}_B{B:0.2f}_Nfg{Nfg}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_nside{nside}'

map_leak_HI_need = need_analysis_HI_lkg.GetBetajkSims(map_input=HI_lkg, nfreq=num_freq, fname=fname_leak_HI)
map_leak_fg_need = need_analysis_fg_lkg.GetBetajkSims(map_input=fg_lkg, nfreq=num_freq, fname=fname_leak_fg)


del map_leak_HI_need; del map_leak_fg_need; del need_analysis_HI_lkg; del need_analysis_fg_lkg

######################################################################################################
################################ jmax=12  ############################################################

jmax=12

need_analysis_HI_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_lkg)
need_analysis_fg_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_lkg)

B = need_analysis_fg_lkg.B
print(B)
fname_leak_HI=f'bjk_maps_leak_HI_jmax{jmax}_lmax{lmax}_B{B:0.2f}_Nfg{Nfg}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_nside{nside}'
fname_leak_fg=f'bjk_maps_leak_fg_jmax{jmax}_lmax{lmax}_B{B:0.2f}_Nfg{Nfg}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_nside{nside}'

map_leak_HI_need = need_analysis_HI_lkg.GetBetajkSims(map_input=HI_lkg, nfreq=num_freq, fname=fname_leak_HI)
map_leak_fg_need = need_analysis_fg_lkg.GetBetajkSims(map_input=fg_lkg, nfreq=num_freq, fname=fname_leak_fg)


del map_leak_HI_need; del map_leak_fg_need; del need_analysis_HI_lkg; del need_analysis_fg_lkg

######################################################################################################
################################ jmax=8  ############################################################
jmax=8

need_analysis_HI_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_lkg)
need_analysis_fg_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_lkg)

B = need_analysis_fg_lkg.B
print(B)
fname_leak_HI=f'bjk_maps_leak_HI_jmax{jmax}_lmax{lmax}_B{B:0.2f}_Nfg{Nfg}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_nside{nside}'
fname_leak_fg=f'bjk_maps_leak_fg_jmax{jmax}_lmax{lmax}_B{B:0.2f}_Nfg{Nfg}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_nside{nside}'

map_leak_HI_need = need_analysis_HI_lkg.GetBetajkSims(map_input=HI_lkg, nfreq=num_freq, fname=fname_leak_HI)
map_leak_fg_need = need_analysis_fg_lkg.GetBetajkSims(map_input=fg_lkg, nfreq=num_freq, fname=fname_leak_fg)


del map_leak_HI_need; del map_leak_fg_need; del need_analysis_HI_lkg; del need_analysis_fg_lkg

######################################################################################################
################################ jmax=4  ############################################################
jmax=4

need_analysis_HI_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_lkg)
need_analysis_fg_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_lkg)

B = need_analysis_fg_lkg.B
print(B)
fname_leak_HI=f'bjk_maps_leak_HI_jmax{jmax}_lmax{lmax}_B{B:0.2f}_Nfg{Nfg}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_nside{nside}'
fname_leak_fg=f'bjk_maps_leak_fg_jmax{jmax}_lmax{lmax}_B{B:0.2f}_Nfg{Nfg}_{num_freq}freq_{min(nu_ch)}_{max(nu_ch)}MHz_nside{nside}'

map_leak_HI_need = need_analysis_HI_lkg.GetBetajkSims(map_input=HI_lkg, nfreq=num_freq, fname=fname_leak_HI)
map_leak_fg_need = need_analysis_fg_lkg.GetBetajkSims(map_input=fg_lkg, nfreq=num_freq, fname=fname_leak_fg)


del map_leak_HI_need; del map_leak_fg_need; del need_analysis_HI_lkg; del need_analysis_fg_lkg