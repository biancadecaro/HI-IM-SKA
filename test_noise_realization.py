import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import pymaster as nm
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

########################################
path_data_SKA_AA4 = f'Sims/beam_SKA_AA4_no_mean_sims_synch_ff_ps_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax767_nside256'

with open(path_data_SKA_AA4+'.pkl', 'rb') as f:
        data_SKA_AA4 = pickle.load(f)
        f.close()
del f
path_data_SKA_AA4_pol = f'Sims/beam_SKA_AA4_no_mean_sims_synch_ff_ps_pol_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax767_nside256'

with open(path_data_SKA_AA4_pol+'.pkl', 'rb') as f:
        data_SKA_AA4_pol = pickle.load(f)
        f.close()
del f

path_data_SKA_1p3deg = f'Sims/beam_1.3deg_SKA_AA4_no_mean_sims_synch_ff_ps_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax767_nside256'

with open(path_data_SKA_1p3deg+'.pkl', 'rb') as f:
        data_SKA_1p3deg = pickle.load(f)
        f.close()
del f
path_data_SKA_1p3deg_pol = f'Sims/beam_1.3deg_SKA_AA4_no_mean_sims_synch_ff_ps_pol_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax767_nside256'

with open(path_data_SKA_1p3deg_pol+'.pkl', 'rb') as f:
        data_SKA_1p3deg_pol = pickle.load(f)
        f.close()

del f
path_data = f'Sims/no_mean_sims_synch_ff_ps_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax767_nside256'

with open(path_data+'.pkl', 'rb') as f:
        data_SKA = pickle.load(f)
        f.close()
del f
path_data_pol = f'Sims/no_mean_sims_synch_ff_ps_pol_noise_105freq_900.5_1004.5MHz_thick1.0MHz_lmax767_nside256'

with open(path_data_pol+'.pkl', 'rb') as f:
        data_SKA_pol = pickle.load(f)
        f.close()


noise_data_SKA_AA4 = data_SKA_AA4['maps_sims_noise']
noise_data_SKA_AA4_pol   = data_SKA_AA4_pol['maps_sims_noise']
noise_data_SKA_1p3deg= data_SKA_1p3deg['maps_sims_noise']
noise_data_SKA_1p3deg_pol= data_SKA_1p3deg_pol['maps_sims_noise']

noise_data= data_SKA['maps_sims_noise']
noise_data_pol= data_SKA_pol['maps_sims_noise']

ich = int(105./2.)


HI_noise_data_SKA_AA4 = data_SKA_AA4['maps_sims_HI'] + noise_data_SKA_AA4
HI_noise_data_SKA_AA4_pol   = data_SKA_AA4_pol['maps_sims_HI']+ noise_data_SKA_AA4_pol
HI_noise_data_SKA_1p3deg= data_SKA_1p3deg['maps_sims_HI']+ noise_data_SKA_1p3deg
HI_noise_data_SKA_1p3deg_pol= data_SKA_1p3deg_pol['maps_sims_HI']+ noise_data_SKA_1p3deg_pol


HI_data_SKA_AA4 = data_SKA_AA4['maps_sims_HI'] 
HI_data_SKA_AA4_pol   = data_SKA_AA4_pol['maps_sims_HI']
HI_data_SKA_1p3deg= data_SKA_1p3deg['maps_sims_HI']
HI_data_SKA_1p3deg_pol= data_SKA_1p3deg_pol['maps_sims_HI']

################################################
cl_noise_data_SKA_AA4 = hp.anafast(data_SKA_AA4['maps_sims_noise'][ich], lmax=2*256)
cl_noise_data_SKA_AA4_pol   = hp.anafast(data_SKA_AA4_pol['maps_sims_noise'][ich], lmax=2*256)
cl_noise_data_SKA_1p3deg= hp.anafast(data_SKA_1p3deg['maps_sims_noise'][ich], lmax=2*256)
cl_noise_data_SKA_1p3deg_pol= hp.anafast(data_SKA_1p3deg_pol['maps_sims_noise'][ich], lmax=2*256)


cl_HI_noise_data_SKA_AA4 = hp.anafast(HI_noise_data_SKA_AA4[ich], lmax=2*256)
cl_HI_noise_data_SKA_AA4_pol   = hp.anafast(HI_noise_data_SKA_AA4_pol[ich], lmax=2*256)
cl_HI_noise_data_SKA_1p3deg= hp.anafast(HI_noise_data_SKA_1p3deg[ich], lmax=2*256)
cl_HI_noise_data_SKA_1p3deg_pol= hp.anafast(HI_noise_data_SKA_1p3deg_pol[ich], lmax=2*256)

cl_HI_data_SKA_AA4 = hp.anafast(HI_data_SKA_AA4[ich], lmax=2*256)
cl_HI_data_SKA_AA4_pol   = hp.anafast(HI_data_SKA_AA4_pol[ich], lmax=2*256)
cl_HI_data_SKA_1p3deg= hp.anafast(HI_data_SKA_1p3deg[ich], lmax=2*256)
cl_HI_data_SKA_1p3deg_pol= hp.anafast(HI_data_SKA_1p3deg_pol[ich], lmax=2*256)

cl_noise_data= hp.anafast(data_SKA['maps_sims_noise'][ich], lmax=2*256)
cl_noise_data_pol= hp.anafast(data_SKA_pol['maps_sims_noise'][ich], lmax=2*256)


#plt.figure
#fig = plt.figure()
#plt.plot(cl_noise_data, label='no pol')
#plt.plot(cl_noise_data_pol, ls='--', label='pol')
#plt.xlim([-1, 2*256+1])
#plt.legend()


plt.figure
fig = plt.figure()
plt.plot(cl_noise_data/cl_noise_data_pol, label='no pol')
plt.xlim([-1, 2*256+1])
plt.legend()


fig = plt.figure()
plt.plot(cl_noise_data_SKA_AA4, label='SKA AA4')
plt.plot(cl_noise_data_SKA_AA4_pol, ls='--', label='SKA AA4 pol')
plt.plot(cl_noise_data_SKA_1p3deg,   label='SKA 1p3deg')
plt.plot(cl_noise_data_SKA_1p3deg_pol, ls='--',label='SKA 1.3deg pol')
plt.xlim([-1, 2*256+1])
plt.legend()


diff_AA4_SKA=cl_HI_noise_data_SKA_AA4/cl_HI_data_SKA_AA4-1
diff_SKA_AA4_pol=cl_HI_noise_data_SKA_AA4_pol/cl_HI_data_SKA_AA4_pol-1
diff_1p3_SKA=cl_HI_noise_data_SKA_1p3deg/cl_HI_data_SKA_1p3deg-1
diff_1p3_SKA_pol=cl_HI_noise_data_SKA_1p3deg_pol/cl_HI_data_SKA_1p3deg_pol-1


fig = plt.figure()
plt.plot(diff_AA4_SKA, label='SKA AA4')
plt.plot(diff_SKA_AA4_pol, ls='--', label='SKA AA4 pol')
plt.plot(diff_1p3_SKA,   label='SKA 1p3deg')
plt.plot(diff_1p3_SKA_pol, ls='--',label='SKA 1.3deg pol')
plt.xlim([-1, 2*256+1])
plt.legend()

plt.show()