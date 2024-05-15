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

from needlets_analysis import analysis, theory
import cython_mylibc as pippo

out_dir_est = './Need_est_ps/'
out_dir_plot= './Plots_need_est/'
#########################
####### jmax=15  ########

nside=128
lmax=256#3*nside-1
jmax=15

out_dir = './'

res_HI = np.load('res_PCA_HI_200_901.0_1299.0MHz_Nfg3.npy')
cosmo_HI = np.load('cosmo_HI_200_901.0_1299.0MHz.npy')
fg_lkg = np.load('fg_leak_200_901.0_1299.0MHz_Nfg3.npy')
HI_lkg = np.load('HI_leak_200_901.0_1299.0MHz_Nfg3.npy')
fg_input = np.load('fg_input_200_901.0_1299.0MHz.npy')
#diff_cosmo_PCA_HI = np.load('diff_cosmo_PCA_HI_200_901.0_1299.0MHz_Nfg3.npy')
#diff_HI_fg_leak = np.load('diff_HI_fg_leak_200_901.0_1299.0MHz_Nfg3.npy')

need_analysis_res = analysis.NeedAnalysis(jmax, lmax, out_dir, res_HI)
need_analysis_HI = analysis.NeedAnalysis(jmax, lmax, out_dir, cosmo_HI)
need_analysis_fg_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_lkg)
need_analysis_HI_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_lkg)
need_analysis_fg_input = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_input)
B_res = need_analysis_res.B
B_HI = need_analysis_HI.B
need_theory = theory.NeedletTheory(B_HI)

print(B_res, B_HI)


b_values_HI = pippo.mylibpy_needlets_std_init_b_values(B_HI, jmax,lmax)

fig = plt.figure(figsize=(7,5)) 
plt.title(f'Needlet window function, B:{B_HI:1.2f}, jmax:{jmax}, lmax: {lmax}')

for i in range(1,b_values_HI.shape[0]):
    plt.semilogx(b_values_HI[i], label = 'j='+str(i) )
#plt.xscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$b^{2}(\frac{\ell}{D^{j}})$')
plt.legend(loc='right')
plt.tight_layout()
plt.savefig(out_dir_plot+f'b2_jmax{jmax}_lmax{lmax}_B{B_HI:1.2f}.png')
plt.show()

ell_binning=need_theory.ell_binning(jmax, lmax)

fig = plt.figure()
plt.title(f'Multipole binning, B:{B_HI:1.2f}, jmax:{jmax}, lmax: {lmax}')
#ax = fig.add_subplot(1, 1, 1)
for i in range(1,jmax+1):
    ell_range = ell_binning[i][ell_binning[i]!=0]
    plt.plot(ell_range, i*ell_range/ell_range, label= f'j={i}')
    plt.text(ell_range[0], i, r'$\ell_{min}=%d,\,\ell_{max}=%d$'%(ell_range[0],ell_range[-1]))

plt.xlabel(r'$\ell$')
plt.legend(loc='right', ncol=2)
plt.tight_layout()
plt.savefig(out_dir_plot+f'ell_binning_jmax{jmax}_lmax{lmax}_B{B_HI:1.2f}.png')
plt.show()


nu_ch = np.linspace(901.0, 1299.0, 200)

ich=100

#ottenere spettro betaj
HI_PCA_need_est = np.zeros((len(nu_ch), jmax+1))
HI_cosmo_need_est = np.zeros((len(nu_ch), jmax+1))
fg_lkg_need_est = np.zeros((len(nu_ch), jmax+1))
HI_lkg_need_est = np.zeros((len(nu_ch), jmax+1))
fg_input_need_est = np.zeros((len(nu_ch), jmax+1))

for ch in range(len(nu_ch)):
    #HI_PCA_need_est[ch] =  need_analysis_res.Betajk2Betaj(betajk1=res_HI_need_output[ch])
    #HI_cosmo_need_est[ch] =  need_analysis_HI.Betajk2Betaj(betajk1=cosmo_HI_need_output[ch])
    HI_PCA_need_est[ch] =  need_analysis_res.Map2Betaj(map1=res_HI[ch])
    HI_cosmo_need_est[ch] =  need_analysis_HI.Map2Betaj(map1=cosmo_HI[ch])
    fg_lkg_need_est[ch] =  need_analysis_fg_lkg.Map2Betaj(map1=fg_lkg[ch])
    HI_lkg_need_est[ch] =  need_analysis_HI_lkg.Map2Betaj(map1=HI_lkg[ch])
    fg_input_need_est[ch] =  need_analysis_fg_input.Map2Betaj(map1=fg_input[ch])
np.savetxt(out_dir_est+f'need_est_ps_HI_PCA_Nfg3_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',HI_PCA_need_est )
np.savetxt(out_dir_est+f'need_est_ps_cosmo_HI_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',HI_cosmo_need_est )
np.savetxt(out_dir_est+f'need_est_ps_fg_input_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',fg_input_need_est )
np.savetxt(out_dir_est+f'need_est_ps_fg_leak_PCA_Nfg3_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',fg_lkg_need_est )
np.savetxt(out_dir_est+f'need_est_ps_HI_leak_PCA_Nfg3_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',HI_lkg_need_est )

del need_analysis_res; del need_analysis_HI; del need_analysis_fg_lkg; del need_analysis_HI_lkg; del need_analysis_fg_input

del HI_PCA_need_est; del HI_cosmo_need_est;del fg_lkg_need_est; del HI_lkg_need_est; del fg_input_need_est; del b_values_HI


#########################
####### jmax=12  ########
jmax=12

out_dir = './'

need_analysis_res = analysis.NeedAnalysis(jmax, lmax, out_dir, res_HI)
need_analysis_HI = analysis.NeedAnalysis(jmax, lmax, out_dir, cosmo_HI)
need_analysis_fg_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_lkg)
need_analysis_HI_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_lkg)
need_analysis_fg_input = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_input)

B_res = need_analysis_res.B
B_HI = need_analysis_HI.B
need_theory = theory.NeedletTheory(B_HI)

print(B_res, B_HI)


b_values_HI = pippo.mylibpy_needlets_std_init_b_values(B_HI, jmax,lmax)
fig = plt.figure(figsize=(7,5)) 
plt.title(f'Needlet window function, B:{B_HI:1.2f}, jmax:{jmax}, lmax: {lmax}')

for i in range(1,b_values_HI.shape[0]):
    plt.semilogx(b_values_HI[i], label = 'j='+str(i) )
#plt.xscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$b^{2}(\frac{\ell}{D^{j}})$')
plt.legend(loc='right')
plt.tight_layout()
plt.savefig(out_dir_plot+f'b2_jmax{jmax}_lmax{lmax}_B{B_HI:1.2f}.png')
plt.show()

ell_binning=need_theory.ell_binning(jmax, lmax)

fig = plt.figure()
plt.title(f'Multipole binning, B:{B_HI:1.2f}, jmax:{jmax}, lmax: {lmax}')
#ax = fig.add_subplot(1, 1, 1)
for i in range(1,jmax+1):
    ell_range = ell_binning[i][ell_binning[i]!=0]
    plt.plot(ell_range, i*ell_range/ell_range, label= f'j={i}')
    plt.text(ell_range[0], i, r'$\ell_{min}=%d,\,\ell_{max}=%d$'%(ell_range[0],ell_range[-1]))

plt.xlabel(r'$\ell$')
plt.legend(loc='right', ncol=2)
plt.tight_layout()
plt.savefig(out_dir_plot+f'ell_binning_jmax{jmax}_lmax{lmax}_B{B_HI:1.2f}.png')
plt.show()

nu_ch = np.linspace(901.0, 1299.0, 200)

ich=100


#ottenere spettro betaj
HI_PCA_need_est = np.zeros((len(nu_ch), jmax+1))
HI_cosmo_need_est = np.zeros((len(nu_ch), jmax+1))
fg_lkg_need_est = np.zeros((len(nu_ch), jmax+1))
HI_lkg_need_est = np.zeros((len(nu_ch), jmax+1))
fg_input_need_est = np.zeros((len(nu_ch), jmax+1))
for ch in range(len(nu_ch)):
    #HI_PCA_need_est[ch] =  need_analysis_res.Betajk2Betaj(betajk1=res_HI_need_output[ch])
    #HI_cosmo_need_est[ch] =  need_analysis_HI.Betajk2Betaj(betajk1=cosmo_HI_need_output[ch])
    HI_PCA_need_est[ch] =  need_analysis_res.Map2Betaj(map1=res_HI[ch])
    HI_cosmo_need_est[ch] =  need_analysis_HI.Map2Betaj(map1=cosmo_HI[ch])
    fg_lkg_need_est[ch] =  need_analysis_fg_lkg.Map2Betaj(map1=fg_lkg[ch])
    HI_lkg_need_est[ch] =  need_analysis_HI_lkg.Map2Betaj(map1=HI_lkg[ch])
    fg_input_need_est[ch] =  need_analysis_fg_input.Map2Betaj(map1=fg_input[ch])
np.savetxt(out_dir_est+f'need_est_ps_HI_PCA_Nfg3_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',HI_PCA_need_est )
np.savetxt(out_dir_est+f'need_est_ps_cosmo_HI_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',HI_cosmo_need_est )
np.savetxt(out_dir_est+f'need_est_ps_fg_input_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',fg_input_need_est )
np.savetxt(out_dir_est+f'need_est_ps_fg_leak_PCA_Nfg3_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',fg_lkg_need_est )
np.savetxt(out_dir_est+f'need_est_ps_HI_leak_PCA_Nfg3_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',HI_lkg_need_est )

del need_analysis_res; del need_analysis_HI; del need_analysis_fg_lkg; del need_analysis_HI_lkg; del need_analysis_fg_input

del HI_PCA_need_est; del HI_cosmo_need_est;del fg_lkg_need_est; del HI_lkg_need_est; del fg_input_need_est; del b_values_HI



########################
###### jmax=8  ########

jmax=8

out_dir = './'

need_analysis_res = analysis.NeedAnalysis(jmax, lmax, out_dir, res_HI)
need_analysis_HI = analysis.NeedAnalysis(jmax, lmax, out_dir, cosmo_HI)
need_analysis_fg_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_lkg)
need_analysis_HI_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_lkg)
need_analysis_fg_input = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_input)

B_res = need_analysis_res.B
B_HI = need_analysis_HI.B
need_theory = theory.NeedletTheory(B_HI)

print(B_res, B_HI)


b_values_HI = pippo.mylibpy_needlets_std_init_b_values(B_HI, jmax,lmax)
fig = plt.figure(figsize=(7,5)) 
plt.title(f'Needlet window function, B:{B_HI:1.2f}, jmax:{jmax}, lmax: {lmax}')

for i in range(1,b_values_HI.shape[0]):
    plt.semilogx(b_values_HI[i], label = 'j='+str(i) )
#plt.xscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$b^{2}(\frac{\ell}{D^{j}})$')
plt.legend(loc='right')
plt.tight_layout()
plt.savefig(out_dir_plot+f'b2_jmax{jmax}_lmax{lmax}_B{B_HI:1.2f}.png')
plt.show()

ell_binning=need_theory.ell_binning(jmax, lmax)

fig = plt.figure()
plt.title(f'Multipole binning, B:{B_HI:1.2f}, jmax:{jmax}, lmax: {lmax}')
#ax = fig.add_subplot(1, 1, 1)
for i in range(1,jmax+1):
    ell_range = ell_binning[i][ell_binning[i]!=0]
    plt.plot(ell_range, i*ell_range/ell_range, label= f'j={i}')
    plt.text(ell_range[0], i, r'$\ell_{min}=%d,\,\ell_{max}=%d$'%(ell_range[0],ell_range[-1]))

plt.xlabel(r'$\ell$')
plt.legend(loc='right', ncol=2)
plt.tight_layout()
plt.savefig(out_dir_plot+f'ell_binning_jmax{jmax}_lmax{lmax}_B{B_HI:1.2f}.png')
plt.show()
nu_ch = np.linspace(901.0, 1299.0, 200)

ich=100

#ottenere spettro betaj
HI_PCA_need_est = np.zeros((len(nu_ch), jmax+1))
HI_cosmo_need_est = np.zeros((len(nu_ch), jmax+1))
fg_lkg_need_est = np.zeros((len(nu_ch), jmax+1))
HI_lkg_need_est = np.zeros((len(nu_ch), jmax+1))
fg_input_need_est = np.zeros((len(nu_ch), jmax+1))

for ch in range(len(nu_ch)):
    HI_PCA_need_est[ch] =  need_analysis_res.Map2Betaj(map1=res_HI[ch])
    HI_cosmo_need_est[ch] =  need_analysis_HI.Map2Betaj(map1=cosmo_HI[ch])
    fg_lkg_need_est[ch] =  need_analysis_fg_lkg.Map2Betaj(map1=fg_lkg[ch])
    HI_lkg_need_est[ch] =  need_analysis_HI_lkg.Map2Betaj(map1=HI_lkg[ch])
    fg_input_need_est[ch] =  need_analysis_fg_input.Map2Betaj(map1=fg_input[ch])
np.savetxt(out_dir_est+f'need_est_ps_HI_PCA_Nfg3_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',HI_PCA_need_est )
np.savetxt(out_dir_est+f'need_est_ps_cosmo_HI_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',HI_cosmo_need_est )
np.savetxt(out_dir_est+f'need_est_ps_fg_input_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',fg_input_need_est )
np.savetxt(out_dir_est+f'need_est_ps_fg_leak_PCA_Nfg3_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',fg_lkg_need_est )
np.savetxt(out_dir_est+f'need_est_ps_HI_leak_PCA_Nfg3_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',HI_lkg_need_est )

del need_analysis_res; del need_analysis_HI; del need_analysis_fg_lkg; del need_analysis_HI_lkg; del need_analysis_fg_input

del HI_PCA_need_est; del HI_cosmo_need_est;del fg_lkg_need_est; del HI_lkg_need_est; del fg_input_need_est; del b_values_HI


########################
###### jmax=4  ########

jmax=4

out_dir = './'

need_analysis_res = analysis.NeedAnalysis(jmax, lmax, out_dir, res_HI)
need_analysis_HI = analysis.NeedAnalysis(jmax, lmax, out_dir, cosmo_HI)
need_analysis_fg_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_lkg)
need_analysis_HI_lkg = analysis.NeedAnalysis(jmax, lmax, out_dir, HI_lkg)
need_analysis_fg_input = analysis.NeedAnalysis(jmax, lmax, out_dir, fg_input)

B_res = need_analysis_res.B
B_HI = need_analysis_HI.B
need_theory = theory.NeedletTheory(B_HI)

print(B_res, B_HI)


b_values_HI = pippo.mylibpy_needlets_std_init_b_values(B_HI, jmax,lmax)
fig = plt.figure(figsize=(7,5)) 
plt.title(f'Needlet window function, B:{B_HI:1.2f}, jmax:{jmax}, lmax: {lmax}')

for i in range(1,b_values_HI.shape[0]):
    plt.semilogx(b_values_HI[i], label = 'j='+str(i) )
#plt.xscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$b^{2}(\frac{\ell}{D^{j}})$')
plt.legend(loc='right')
plt.tight_layout()
plt.savefig(out_dir_plot+f'b2_jmax{jmax}_lmax{lmax}_B{B_HI:1.2f}.png')
plt.show()

ell_binning=need_theory.ell_binning(jmax, lmax)

fig = plt.figure()
plt.title(f'Multipole binning, B:{B_HI:1.2f}, jmax:{jmax}, lmax: {lmax}')
#ax = fig.add_subplot(1, 1, 1)
for i in range(1,jmax+1):
    ell_range = ell_binning[i][ell_binning[i]!=0]
    plt.plot(ell_range, i*ell_range/ell_range, label= f'j={i}')
    plt.text(ell_range[0], i, r'$\ell_{min}=%d,\,\ell_{max}=%d$'%(ell_range[0],ell_range[-1]))

plt.xlabel(r'$\ell$')
plt.legend(loc='right', ncol=2)
plt.tight_layout()
plt.savefig(out_dir_plot+f'ell_binning_jmax{jmax}_lmax{lmax}_B{B_HI:1.2f}.png')
plt.show()

nu_ch = np.linspace(901.0, 1299.0, 200)

ich=100

#ottenere spettro betaj
HI_PCA_need_est = np.zeros((len(nu_ch), jmax+1))
HI_cosmo_need_est = np.zeros((len(nu_ch), jmax+1))
fg_lkg_need_est = np.zeros((len(nu_ch), jmax+1))
HI_lkg_need_est = np.zeros((len(nu_ch), jmax+1))
fg_input_need_est = np.zeros((len(nu_ch), jmax+1))

for ch in range(len(nu_ch)):
    HI_PCA_need_est[ch] =  need_analysis_res.Map2Betaj(map1=res_HI[ch])
    HI_cosmo_need_est[ch] =  need_analysis_HI.Map2Betaj(map1=cosmo_HI[ch])
    fg_lkg_need_est[ch] =  need_analysis_fg_lkg.Map2Betaj(map1=fg_lkg[ch])
    HI_lkg_need_est[ch] =  need_analysis_HI_lkg.Map2Betaj(map1=HI_lkg[ch])
    fg_input_need_est[ch] =  need_analysis_fg_input.Map2Betaj(map1=fg_input[ch])
np.savetxt(out_dir_est+f'need_est_ps_HI_PCA_Nfg3_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',HI_PCA_need_est )
np.savetxt(out_dir_est+f'need_est_ps_cosmo_HI_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',HI_cosmo_need_est )
np.savetxt(out_dir_est+f'need_est_ps_fg_input_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',fg_input_need_est )
np.savetxt(out_dir_est+f'need_est_ps_fg_leak_PCA_Nfg3_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',fg_lkg_need_est )
np.savetxt(out_dir_est+f'need_est_ps_HI_leak_PCA_Nfg3_jmax{jmax}_B{B_HI:0.2f}_lmax{lmax}_nside{nside}.dat',HI_lkg_need_est )

del need_analysis_res; del need_analysis_HI; del need_analysis_fg_lkg; del need_analysis_HI_lkg; del need_analysis_fg_input

del HI_PCA_need_est; del HI_cosmo_need_est;del fg_lkg_need_est; del HI_lkg_need_est; del fg_input_need_est; del b_values_HI

