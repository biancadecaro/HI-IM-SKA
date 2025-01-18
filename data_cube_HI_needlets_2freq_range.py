import healpy as hp
from astropy import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
from needlets_analysis import analysis

path_data_sims_tot_1 = 'foregrounds+HI_sims_100freq_901.0_1099.0MHz_nside128'
with open(path_data_sims_tot_1+'.pkl', 'rb') as f:
        file1 = pickle.load(f)
        f.close()
path_data_sims_tot_2 = 'foregrounds+HI_sims_100freq_1101.0_1299.0MHz_nside128'
with open(path_data_sims_tot_2+'.pkl', 'rb') as ff:
        file2 = pickle.load(ff)
        ff.close()

nu_ch1 = file1['freq']
nu_ch2 = file2['freq']
num_freq = len(nu_ch1)

#print('freq file1:',min(file1['freq']), max(file1['freq']), file1['freq'][1]-file1['freq'][0], len(file1['freq']) )
#print('freq file2:',min(file2['freq']), max(file2['freq']), file2['freq'][1]-file2['freq'][0], len(file2['freq']) )

HI_maps_freq1 = file1['maps_sims_HI']
HI_maps_freq2 = file2['maps_sims_HI']

fg_maps_freq1 = file1['maps_sims_fg']
fg_maps_freq2 = file2['maps_sims_fg']

full_maps_freq1 = file1['maps_sims_tot']
full_maps_freq2 = file2['maps_sims_tot']
del file1
del file2

npix = np.shape(HI_maps_freq1)[1]
nside = hp.get_nside(HI_maps_freq1[0])
lmax=256
jmax=12
out_dir = './Maps_test_1/'

need_analysis = analysis.NeedAnalysis(jmax, lmax, out_dir, full_maps_freq1)
need_analysis_HI = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_maps_freq1)
need_analysis_fg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_maps_freq1)

B=need_analysis.B

fname_obs_tot=f'bjk_maps_obs_{num_freq}freq_{min(nu_ch1)}_{max(nu_ch1)}MHz_jmax{jmax}_B{B:0.2f}_nside{nside}'
fname_HI=f'bjk_maps_HI_{num_freq}freq_{min(nu_ch1)}_{max(nu_ch1)}MHz_jmax{jmax}_B{B:0.2f}_nside{nside}'
fname_fg=f'bjk_maps_fg_{num_freq}freq_{min(nu_ch1)}_{max(nu_ch1)}MHz_jmax{jmax}_B{B:0.2f}_nside{nside}'

map_need_output = need_analysis.GetBetajkSims(map_input=full_maps_freq1, nfreq=num_freq, fname=fname_obs_tot)
map_HI_need_output = need_analysis_HI.GetBetajkSims(map_input=HI_maps_freq1, nfreq=num_freq, fname=fname_HI)
map_fg_need_output = need_analysis_fg.GetBetajkSims(map_input=fg_maps_freq1, nfreq=num_freq, fname=fname_fg)

del need_analysis; del need_analysis_HI; del need_analysis_fg
del HI_maps_freq1;del full_maps_freq1; del fg_maps_freq1

ich=10
j_test=7
fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch1[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(map_need_output[ich,j_test], cmap='viridis', title=f'Observation, j={j_test}, freq={nu_ch1[ich]}', hold=True)
fig.add_subplot(222) 
hp.mollview(map_HI_need_output[ich,j_test], cmap='viridis', title=f'HI signal, j={j_test}, freq={nu_ch1[ich]}',hold=True)
fig.add_subplot(223)
hp.mollview(map_fg_need_output[ich, j_test], title=f'Fg signal, j={j_test}, freq={nu_ch1[ich]}',cmap='viridis', hold=True)
fig.add_subplot(224)
hp.mollview(map_need_output[ich,j_test]-map_fg_need_output[ich, j_test], title=f'Observtion - Fg, j={j_test}, freq={nu_ch1[ich]}',cmap='viridis', hold=True)
plt.show()


del map_need_output; del map_HI_need_output; del map_fg_need_output

need_analysis = analysis.NeedAnalysis(jmax, lmax, out_dir, full_maps_freq2)
need_analysis_HI = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_maps_freq2)
need_analysis_fg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_maps_freq2)

B=need_analysis.B

fname_obs_tot=f'bjk_maps_obs_{num_freq}freq_{min(nu_ch2)}_{max(nu_ch2)}MHz_jmax{jmax}_B{B:0.2f}_nside{nside}'
fname_HI=f'bjk_maps_HI_{num_freq}freq_{min(nu_ch2)}_{max(nu_ch2)}MHz_jmax{jmax}_B{B:0.2f}_nside{nside}'
fname_fg=f'bjk_maps_fg_{num_freq}freq_{min(nu_ch2)}_{max(nu_ch2)}MHz_jmax{jmax}_B{B:0.2f}_nside{nside}'

map_need_output = need_analysis.GetBetajkSims(map_input=full_maps_freq2, nfreq=num_freq, fname=fname_obs_tot)
map_HI_need_output = need_analysis_HI.GetBetajkSims(map_input=HI_maps_freq2, nfreq=num_freq, fname=fname_HI)
map_fg_need_output = need_analysis_fg.GetBetajkSims(map_input=fg_maps_freq2, nfreq=num_freq, fname=fname_fg)

del need_analysis; del need_analysis_HI; del need_analysis_fg
del HI_maps_freq2;del full_maps_freq2; del fg_maps_freq2

ich=10
j_test=7
fig = plt.figure(figsize=(10, 7))
fig.suptitle(f'channel {ich}: {nu_ch2[ich]} MHz',fontsize=20)
fig.add_subplot(221) 
hp.mollview(map_need_output[ich,j_test], cmap='viridis', title=f'Observation, j={j_test}, freq={nu_ch2[ich]}', hold=True)
fig.add_subplot(222) 
hp.mollview(map_HI_need_output[ich,j_test], cmap='viridis', title=f'HI signal, j={j_test}, freq={nu_ch2[ich]}',hold=True)
fig.add_subplot(223)
hp.mollview(map_fg_need_output[ich, j_test], title=f'Fg signal, j={j_test}, freq={nu_ch2[ich]}',cmap='viridis', hold=True)
fig.add_subplot(224)
hp.mollview(map_need_output[ich,j_test]-map_fg_need_output[ich, j_test], title=f'Observtion - Fg, j={j_test}, freq={nu_ch2[ich]}',cmap='viridis', hold=True)
plt.show()



