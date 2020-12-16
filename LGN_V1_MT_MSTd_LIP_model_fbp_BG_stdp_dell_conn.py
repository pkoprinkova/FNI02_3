import nest
import numpy as np
import nest.raster_plot
import matplotlib.pyplot as plt
import cPickle
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.lines as mlines

from random import seed
from random import random

from func_fig_nonZero_ID import figure1, figure2, func_conn, func_non_zero, function_rectangle


seed(1234)

nest.ResetKernel()
nest.SetKernelStatus({"resolution": 0.1})

seed = 1055
np.random.seed(seed)

lamb=1.2 #float(raw_input("lambda: "))
lamb_MT=1.2 #float(raw_input("lambda: "))
ww=0.5 #float(raw_input("ww: "))
sigma=1.0 #float(raw_input("sigma: "))
#sigma_MT=3.0 #float(raw_input("sigma: "))

LGN_size='20x20'
EI_size='20x10'
MT_size='20x10'
MSTd_size='20x20'
LIP_size='20x20'
layer_size='10x10'

scale_fb_MT=0.05
scale_fb_V1=0.05

ns=10
d=0.05
sigma_ts=10.0
sigma_I=3.0
nfp=3
sigma_lip=3.0
sigma_bg=3.0
#scale=1.0

#LGN
fr_data_folder='/home/simona/V1NetworkModels/PyNN/20x20_lgn_fr_bin_150_5_eye_last_ml200_1670/'# '/home/simona/model/bin_Version_1/20x20_lgn_fr_bin_150_5_eye_last/'
#fr_data_folder='/home/nest/my_work/new_screen_stimuli/'
params_folder='/home/simona/model/bin_Version_1/Parameters/'
results_folder="/home/simona/model/bin_Version_1/result/" #result_20x20_lgn_fr_bin_150_5_eye_last_ml200_1670_BG/" #'/home/simona/model/bin_Version_1/20x20_lgn_fr_bin_150_5_eye_last/'

#fr_data_folder='/home/nest/my_work/bin_Version_1/20x20_lgn_fr_bin_150_5_eye_last/'
#fr_data_folder='/home/nest/my_work/new_screen_stimuli/'
#params_folder='/home/nest/my_work/bin_Version_1/Parameters/'
#results_folder='/home/nest/my_work/bin_Version_1/20x20_lgn_fr_bin_150_5_eye_last/'
#charts_folder='/home/nest/my_work/BG_sim_leftIe200_rightIe10/'
charts_folder="/home/simona/model/bin_Version_1/result/" #result_20x20_lgn_fr_bin_150_5_eye_last_ml200_1670_BG/"

f = open(fr_data_folder+LGN_size+"_firing_rates_ON_11.cpickle", "rb")
scaled_firing_rateON1 = cPickle.load(f)
scaled_firing_rateON2 = cPickle.load(f)
f.close()

f = open(fr_data_folder+LGN_size+"_firing_rates_OFF_11.cpickle", "rb")
scaled_firing_rateOFF1 = cPickle.load(f)
scaled_firing_rateOFF2 = cPickle.load(f)
f.close()

dimension = len(scaled_firing_rateON1[0,:])

#screen version!
#f=open(fr_data_folder+LGN_size+'_LGN_positions_in_pixels_screen.cpickle','rb')
f = open(params_folder+LGN_size+"_LGN_positions_in_pixels.cpickle", "rb")
xc_OFF1 = cPickle.load(f)
yc_OFF1 = cPickle.load(f)
xc_ON1 = cPickle.load(f)
yc_ON1 = cPickle.load(f)
xc_OFF2 = cPickle.load(f)
yc_OFF2 = cPickle.load(f)
xc_ON2 = cPickle.load(f)
yc_ON2 = cPickle.load(f)
f.close()
print 'LGN positions: ', xc_ON1
Nx = len(xc_ON1)
Ny = len(yc_ON1)
n_LGN=4*4*Nx*Ny

t_sim = len(scaled_firing_rateON1[:,0])
sim_time=0.0+t_sim
t = np.arange(0, t_sim).astype(int)
at=t+0.1

params_ON = {"V_th" : -59.0} #, "V_m" : 0.0 
params_OFF = {"V_th" : -59.0} # ,"V_m" : 0.0 
interneurons_params = {"V_th": -67.6} #{"V_th": -69.992,"V_reset": -80.0}

neuronON1 = nest.Create("iaf_chxk_2008", dimension)
nest.SetStatus(neuronON1, params_ON)

neuronOFF1 = nest.Create("iaf_chxk_2008", dimension)
nest.SetStatus(neuronOFF1, params_OFF)

neuronON2 = nest.Create("iaf_chxk_2008", dimension)
nest.SetStatus(neuronON2, params_ON)

neuronOFF2 = nest.Create("iaf_chxk_2008", dimension)
nest.SetStatus(neuronOFF2, params_OFF)

interneuron1ON1 = nest.Create("iaf_psc_exp", dimension, interneurons_params)
interneuron2ON1 = nest.Create("iaf_psc_exp", dimension, interneurons_params)

interneuron1OFF1 = nest.Create("iaf_psc_exp", dimension, interneurons_params)
interneuron2OFF1 = nest.Create("iaf_psc_exp", dimension, interneurons_params)

interneuron1ON2 = nest.Create("iaf_psc_exp", dimension, interneurons_params)
interneuron2ON2 = nest.Create("iaf_psc_exp", dimension, interneurons_params)

interneuron1OFF2 = nest.Create("iaf_psc_exp", dimension, interneurons_params)
interneuron2OFF2 = nest.Create("iaf_psc_exp", dimension, interneurons_params)

scg_ON1 = nest.Create("step_current_generator", dimension)
sd_ON1 = nest.Create("spike_detector", dimension, params = {"to_memory": True, "to_file": False, "withtime": True, "withgid": True})

scg_OFF1 = nest.Create("step_current_generator", dimension)
sd_OFF1 = nest.Create("spike_detector", dimension, params = {"to_memory": True, "to_file": False, "withtime": True, "withgid": True})

scg_ON2 = nest.Create("step_current_generator", dimension)
sd_ON2 = nest.Create("spike_detector", dimension, params = {"to_memory": True, "to_file": False, "withtime": True, "withgid": True})

scg_OFF2 = nest.Create("step_current_generator", dimension)
sd_OFF2 = nest.Create("spike_detector", dimension, params = {"to_memory": True, "to_file": False, "withtime": True, "withgid": True})

'''
bias_frON=70.0
bias_frOFF=70.0
for xi in np.arange(0,dimension,1):
	nest.SetStatus([scg_ON1[xi]],{'amplitude_values':scaled_firing_rateON1[:,xi]+bias_frON, 'amplitude_times':at})
	nest.Connect([scg_ON1[xi]], [neuronON1[xi]])
	nest.Connect([neuronON1[xi]], [sd_ON1[xi]])
	
	nest.SetStatus([scg_OFF1[xi]],{'amplitude_values':scaled_firing_rateOFF1[:,xi]+bias_frOFF, 'amplitude_times':at})
	nest.Connect([scg_OFF1[xi]], [neuronOFF1[xi]])
	nest.Connect([neuronOFF1[xi]], [sd_OFF1[xi]])

	nest.SetStatus([scg_OFF2[xi]],{'amplitude_values':scaled_firing_rateOFF2[:,xi]+bias_frOFF, 'amplitude_times':at})	
	nest.Connect([scg_OFF2[xi]], [neuronOFF2[xi]])
	nest.Connect([neuronOFF2[xi]], [sd_OFF2[xi]])

	nest.SetStatus([scg_ON2[xi]],{'amplitude_values':scaled_firing_rateON2[:,xi]+bias_frON, 'amplitude_times':at})
	nest.Connect([scg_ON2[xi]], [neuronON2[xi]])
	nest.Connect([neuronON2[xi]], [sd_ON2[xi]])
		
	nest.Connect([scg_ON1[xi]], [interneuron1ON1[xi]])	
	nest.Connect([interneuron1ON1[xi]], [neuronON1[xi]],syn_spec={'weight': -1.0, 'delay': 10.0})
	nest.Connect([interneuron2ON1[xi]], [neuronON1[xi]],syn_spec={'weight': -1.0, 'delay': 10.0})
	nest.Connect([neuronON1[xi]], [interneuron2ON1[xi]],syn_spec={'weight': 1.0, 'delay': 3.0})

	nest.Connect([scg_OFF1[xi]], [interneuron1OFF1[xi]])	
	nest.Connect([interneuron1OFF1[xi]], [neuronOFF1[xi]],syn_spec={'weight': -1.0, 'delay': 10.0})
	nest.Connect([interneuron2OFF1[xi]], [neuronOFF1[xi]],syn_spec={'weight': -1.0, 'delay': 10.0})
	nest.Connect([neuronOFF1[xi]], [interneuron2OFF1[xi]],syn_spec={'weight': 1.0, 'delay': 3.0})
	
	nest.Connect([scg_ON2[xi]], [interneuron1ON2[xi]])	
	nest.Connect([interneuron1ON2[xi]], [neuronON2[xi]],syn_spec={'weight': -1.0, 'delay': 10.0})
	nest.Connect([interneuron2ON2[xi]], [neuronON2[xi]],syn_spec={'weight': -1.0, 'delay': 10.0})
	nest.Connect([neuronON2[xi]], [interneuron2ON2[xi]],syn_spec={'weight': 1.0, 'delay': 3.0})
	
	nest.Connect([scg_OFF2[xi]], [interneuron1OFF2[xi]])
	nest.Connect([interneuron1OFF2[xi]], [neuronOFF2[xi]],syn_spec={'weight': -1.0, 'delay': 10.0})
	nest.Connect([interneuron2OFF2[xi]], [neuronOFF2[xi]],syn_spec={'weight': -1.0, 'delay': 10.0})
	nest.Connect([neuronOFF2[xi]], [interneuron2OFF2[xi]],syn_spec={'weight': 1.0, 'delay': 3.0})
'''

#dynamic synapses 
wr_LGN = nest.Create('weight_recorder',1)
lambda_LGN=0.1
glgn=2.0
wlgn=1.0
nest.CopyModel('stdp_synapse', 'stdp_synapse_LGN', {'mu_plus': 1.0, 'mu_minus': 1.0, 'lambda': lambda_LGN}) #'weight_recorder': wr_LGN[0], 
####

bias_frON=70.0
bias_frOFF=70.0
for xi in np.arange(0,dimension,1):
	nest.SetStatus([scg_ON1[xi]],{'amplitude_values':scaled_firing_rateON1[:,xi]+bias_frON, 'amplitude_times':at})
	nest.Connect([scg_ON1[xi]], [neuronON1[xi]])
	nest.Connect([neuronON1[xi]], [sd_ON1[xi]])
	
	nest.SetStatus([scg_OFF1[xi]],{'amplitude_values':scaled_firing_rateOFF1[:,xi]+bias_frOFF, 'amplitude_times':at})
	nest.Connect([scg_OFF1[xi]], [neuronOFF1[xi]])
	nest.Connect([neuronOFF1[xi]], [sd_OFF1[xi]])

	nest.SetStatus([scg_OFF2[xi]],{'amplitude_values':scaled_firing_rateOFF2[:,xi]+bias_frOFF, 'amplitude_times':at})	
	nest.Connect([scg_OFF2[xi]], [neuronOFF2[xi]])
	nest.Connect([neuronOFF2[xi]], [sd_OFF2[xi]])

	nest.SetStatus([scg_ON2[xi]],{'amplitude_values':scaled_firing_rateON2[:,xi]+bias_frON, 'amplitude_times':at})
	nest.Connect([scg_ON2[xi]], [neuronON2[xi]])
	nest.Connect([neuronON2[xi]], [sd_ON2[xi]])
		
	nest.Connect([scg_ON1[xi]], [interneuron1ON1[xi]])	
	nest.Connect([interneuron1ON1[xi]], [neuronON1[xi]],syn_spec={'model': 'stdp_synapse_LGN', 'weight': -wlgn, 'Wmax': -glgn*wlgn ,'delay': 10.0})
	nest.Connect([interneuron2ON1[xi]], [neuronON1[xi]],syn_spec={'model': 'stdp_synapse_LGN', 'weight': -wlgn, 'Wmax': -glgn*wlgn, 'delay': 10.0})
	nest.Connect([neuronON1[xi]], [interneuron2ON1[xi]],syn_spec={'model': 'stdp_synapse_LGN', 'weight': wlgn, 'Wmax': glgn*wlgn, 'delay': 3.0})

	nest.Connect([scg_OFF1[xi]], [interneuron1OFF1[xi]])	
	nest.Connect([interneuron1OFF1[xi]], [neuronOFF1[xi]],syn_spec={'model': 'stdp_synapse_LGN', 'weight': -wlgn, 'Wmax': -glgn*wlgn, 'delay': 10.0})
	nest.Connect([interneuron2OFF1[xi]], [neuronOFF1[xi]],syn_spec={'model': 'stdp_synapse_LGN', 'weight': -wlgn, 'Wmax': -glgn*wlgn, 'delay': 10.0})
	nest.Connect([neuronOFF1[xi]], [interneuron2OFF1[xi]],syn_spec={'model': 'stdp_synapse_LGN', 'weight': wlgn, 'Wmax': glgn*wlgn, 'delay': 3.0})
	
	nest.Connect([scg_ON2[xi]], [interneuron1ON2[xi]])	
	nest.Connect([interneuron1ON2[xi]], [neuronON2[xi]],syn_spec={'model': 'stdp_synapse_LGN', 'weight': -wlgn, 'Wmax': -glgn*wlgn, 'delay': 10.0})
	nest.Connect([interneuron2ON2[xi]], [neuronON2[xi]],syn_spec={'model': 'stdp_synapse_LGN', 'weight': -wlgn, 'Wmax': -glgn*wlgn, 'delay': 10.0})
	nest.Connect([neuronON2[xi]], [interneuron2ON2[xi]],syn_spec={'model': 'stdp_synapse_LGN', 'weight': wlgn, 'Wmax': glgn*wlgn, 'delay': 3.0})
	
	nest.Connect([scg_OFF2[xi]], [interneuron1OFF2[xi]])
	nest.Connect([interneuron1OFF2[xi]], [neuronOFF2[xi]],syn_spec={'model': 'stdp_synapse_LGN', 'weight': -wlgn, 'Wmax': -glgn*wlgn, 'delay': 10.0})
	nest.Connect([interneuron2OFF2[xi]], [neuronOFF2[xi]],syn_spec={'model': 'stdp_synapse_LGN', 'weight': -wlgn, 'Wmax': -glgn*wlgn, 'delay': 10.0})
	nest.Connect([neuronOFF2[xi]], [interneuron2OFF2[xi]],syn_spec={'model': 'stdp_synapse_LGN', 'weight': wlgn, 'Wmax': glgn*wlgn, 'delay': 3.0})

#end LGN

epop_params = {"V_th": -69.992,"V_reset": -80.0}#,, "I_e": 0.50 "C_m": 290.0, "tau_syn_ex": 3.0, "tau_syn_in": 10.0, , "V_m":-70.0}
ipop_params = {"V_th": -69.992,"V_reset": -80.0}#,, "I_e": 0.50 "C_m": 141.0, "tau_syn_ex": 3.0, "tau_syn_in": 10.0, "E_L": -55.0, "V_m":-70.0}

#V1 layer
f=open(params_folder+EI_size+'_EI_positions_bin.cpickle',"rb") 
e_xy1=cPickle.load(f)
e_xy2=cPickle.load(f)
i_xy1=cPickle.load(f)
i_xy2=cPickle.load(f)
xe1=cPickle.load(f)
ye1=cPickle.load(f)
xe2=cPickle.load(f)
ye2=cPickle.load(f)
xi1=cPickle.load(f)
yi1=cPickle.load(f)
xi2=cPickle.load(f)
yi2=cPickle.load(f)
f.close()
'''
np.savetxt(fr_data_folder+'V1_positions_E1.txt',(xe1,ye1),fmt='%f')
np.savetxt(fr_data_folder+'V1_positions_E2.txt',(xe2,ye2),fmt='%f')
np.savetxt(fr_data_folder+'V1_positions_I1.txt',(xi1,yi1),fmt='%f')
np.savetxt(fr_data_folder+'V1_positions_I2.txt',(xi2,yi2),fmt='%f')'''
print 'V1 positions: ', xe1
Nxc_exc = len(xe1)
Nyc_exc = len(ye1)
Nxc_inh = len(xi1)
Nyc_inh = len(yi1)

Ne=Nxc_exc*Nyc_exc
Ni=Nxc_inh*Nyc_inh

epop1 = nest.Create("iaf_psc_exp", Ne, epop_params)
epop2 = nest.Create("iaf_psc_exp", Ne, epop_params)
ipop1 = nest.Create("iaf_psc_exp", Ni, ipop_params)
ipop2 = nest.Create("iaf_psc_exp", Ni, ipop_params)

f = open(params_folder+'V1_orientations_'+EI_size+'_bins_l'+str(lamb)+'.cpickle', "rb")
phases=cPickle.load(f)
orientations=cPickle.load(f)

thetae1=cPickle.load(f)
phie1=cPickle.load(f)
thetae2=cPickle.load(f)
phie2=cPickle.load(f)

thetai1=cPickle.load(f)
phii1=cPickle.load(f)
thetai2=cPickle.load(f)
phii2=cPickle.load(f)
f.close()

#scr - screen version!
f = open(params_folder+'V1par_fbp'+str(scale_fb_V1)+'_LGN_'+LGN_size+'_EI'+EI_size+'_scale100_1_l'+str(lamb)+'_w'+str(ww)+'_s'+str(sigma)+'_we0.1_se0.1_wi0.1_si0.1.cpickle', "rb")
#f = open(fr_data_folder + 'V1screen_fbp0.05_LGN_20x20_EI20x10_scale200_1_l1.2_w0.5_s5.425_we0.1_se0.535526315789_wi0.1_si0.505774853801.cpickle', 'rb')
W_excON1 = cPickle.load(f)
W_excOFF1 = cPickle.load(f)
W_excON2 = cPickle.load(f)
W_excOFF2 = cPickle.load(f)
W_inhON1 = cPickle.load(f)
W_inhOFF1 = cPickle.load(f)
W_inhON2 = cPickle.load(f)
W_inhOFF2 = cPickle.load(f)

Wfb_excON1 = cPickle.load(f)
Wfb_excOFF1 = cPickle.load(f)
Wfb_excON2 = cPickle.load(f)
Wfb_excOFF2 = cPickle.load(f)
Wfb_inhON1 = cPickle.load(f)
Wfb_inhOFF1 = cPickle.load(f)
Wfb_inhON2 = cPickle.load(f)
Wfb_inhOFF2 = cPickle.load(f)
'''
D_eON1 = cPickle.load(f)
D_eON2 = cPickle.load(f)
D_eOFF2 = cPickle.load(f)
D_eOFF1 = cPickle.load(f)
D_iON1 = cPickle.load(f)
D_iON2 = cPickle.load(f)
D_iOFF2 = cPickle.load(f)
D_iOFF1 = cPickle.load(f)
'''
WE1E1 = cPickle.load(f)
#D_e1e1 = cPickle.load(f)
WE1I1 = cPickle.load(f)
#D_e1i1 = cPickle.load(f)
WE2E2 = cPickle.load(f)
#D_e2e2 = cPickle.load(f)
WE2I2 = cPickle.load(f)
#D_e2i2 = cPickle.load(f)
WI1I2 = cPickle.load(f)
#D_i1i2 = cPickle.load(f)
WI1E2 = cPickle.load(f)
#D_i1e2 = cPickle.load(f)
WI2I1 = cPickle.load(f)
#D_i2i1 = cPickle.load(f)
WI2E1 = cPickle.load(f)
#D_i2e1 = cPickle.load(f)

f.close()

#remove ID
epop1_ID, epop1_ID_removed = function_rectangle(e_xy1, "V1 E1", epop1, results_folder) 
ipop1_ID, ipop1_ID_removed = function_rectangle(i_xy1, "V1 I1", ipop1, results_folder) 
epop2_ID, epop2_ID_removed = function_rectangle(e_xy2, "V1 E2", epop2, results_folder) 
ipop2_ID, ipop2_ID_removed = function_rectangle(i_xy2, "V1 I2", ipop2, results_folder)  


conn_W_excON1 = func_conn(W_excON1, neuronON1, epop1, [],  epop1_ID_removed)
conn_W_excON2 = func_conn(W_excON2, neuronON2, epop2, [], epop2_ID_removed)
conn_W_excOFF2 = func_conn(W_excOFF2, neuronOFF2, epop2, [], epop2_ID_removed)
conn_W_excOFF1 = func_conn(W_excOFF1, neuronOFF1, epop1, [], epop1_ID_removed)
conn_W_inhON1 = func_conn(W_inhON1, neuronON1, ipop1, [], ipop1_ID_removed)
conn_W_inhON2 = func_conn(W_inhON2, neuronON2, ipop2, [], ipop2_ID_removed )
conn_W_inhOFF2 = func_conn(W_inhOFF2, neuronOFF2, ipop2, [], ipop2_ID_removed)
conn_W_inhOFF1 = func_conn(W_inhOFF1, neuronOFF1, ipop1, [], ipop1_ID_removed)

conn_Wfb_excON1 = func_conn(Wfb_excON1, epop1, neuronON1, epop1_ID_removed, [])
conn_Wfb_excON2 = func_conn(Wfb_excON2, epop2, neuronON2, epop2_ID_removed, [])
conn_Wfb_excOFF2 = func_conn(Wfb_excOFF2, epop2, neuronOFF2, epop2_ID_removed, [])
conn_Wfb_excOFF1 = func_conn(Wfb_excOFF1, epop1, neuronOFF1, epop1_ID_removed, [])
conn_Wfb_inhON1 = func_conn(Wfb_inhON1, ipop1, neuronON1, ipop1_ID_removed, [])
conn_Wfb_inhON2 = func_conn(Wfb_inhON2, ipop2, neuronON2, ipop2_ID_removed, [])
conn_Wfb_inhOFF2 = func_conn(Wfb_inhOFF2, ipop2, neuronOFF2, ipop2_ID_removed, [])
conn_Wfb_inhOFF1 = func_conn(Wfb_inhOFF1, ipop1, neuronOFF1, ipop1_ID_removed, [])

conn_Wfb_excON1 = func_conn(Wfb_excON1, epop1, interneuron1ON1, epop1_ID_removed, [])
conn_Wfb_excON2 = func_conn(Wfb_excON2, epop2, interneuron1ON2, epop2_ID_removed, [])
conn_Wfb_excOFF2 = func_conn(Wfb_excOFF2, epop2, interneuron1OFF2, epop2_ID_removed, [])
conn_Wfb_excOFF1 = func_conn(Wfb_excOFF1, epop1, interneuron1OFF1, epop1_ID_removed, [])
conn_Wfb_inhON1 = func_conn(Wfb_inhON1, ipop1, interneuron1ON1, ipop1_ID_removed, [])
conn_Wfb_inhON2 = func_conn(Wfb_inhON2, ipop2, interneuron1ON2, ipop2_ID_removed, [])
conn_Wfb_inhOFF2 = func_conn(Wfb_inhOFF2, ipop2, interneuron1OFF2, ipop2_ID_removed, [])
conn_Wfb_inhOFF1 = func_conn(Wfb_inhOFF1, ipop1, interneuron1OFF1, ipop1_ID_removed, [])

conn_Wfb_excON1 = func_conn(Wfb_excON1, epop1, interneuron2ON1, epop1_ID_removed, [])
conn_Wfb_excON2 = func_conn(Wfb_excON2, epop2, interneuron2ON2, epop2_ID_removed, [])
conn_Wfb_excOFF2 = func_conn(Wfb_excOFF2, epop2, interneuron2OFF2, epop2_ID_removed, [])
conn_Wfb_excOFF1 = func_conn(Wfb_excOFF1, epop1, interneuron2OFF1, epop1_ID_removed, [])
conn_Wfb_inhON1 = func_conn(Wfb_inhON1, ipop1, interneuron2ON1, ipop1_ID_removed, [])
conn_Wfb_inhON2 = func_conn(Wfb_inhON2, ipop2, interneuron2ON2, ipop2_ID_removed, [])
conn_Wfb_inhOFF2 = func_conn(Wfb_inhOFF2, ipop2, interneuron2OFF2, ipop2_ID_removed, [])
conn_Wfb_inhOFF1 = func_conn(Wfb_inhOFF1, ipop1, interneuron2OFF1, ipop1_ID_removed, [])

conn_WE1E1 = func_conn(WE1E1, epop1, epop1, epop1_ID_removed, epop1_ID_removed)
conn_WE1I1 = func_conn(WE1I1, epop1, ipop1, epop1_ID_removed, ipop1_ID_removed)
conn_WI1I2 = func_conn(WI1I2, ipop1, ipop2, ipop1_ID_removed, ipop2_ID_removed)
conn_WI1E2 = func_conn(WI1E2, ipop1, epop2, ipop1_ID_removed, epop2_ID_removed)
conn_WE2E2 = func_conn(WE2E2, epop2, epop2, epop2_ID_removed, epop2_ID_removed)
conn_WI2E1 = func_conn(WI2E1, ipop2, epop1, ipop2_ID_removed, epop1_ID_removed)
conn_WI2I1 = func_conn(WI2I1, ipop2, ipop1, ipop2_ID_removed, ipop1_ID_removed)
conn_WE2I2 = func_conn(WE2I2, epop2, ipop2, epop2_ID_removed, ipop2_ID_removed)


spd_V1= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})

nest.Connect(neuronON1, epop1, syn_spec={'weight': conn_W_excON1, "delay": 3.0}) #D_eON1})
nest.Connect(neuronON2, epop2, syn_spec={'weight': conn_W_excON2, "delay": 3.0}) #D_eON2})
nest.Connect(neuronOFF2, epop2, syn_spec={'weight': conn_W_excOFF2, "delay": 3.0}) #D_eOFF2})
nest.Connect(neuronOFF1, epop1, syn_spec={'weight': conn_W_excOFF1, "delay": 3.0}) #D_eOFF1})
nest.Connect(neuronON1, ipop1, syn_spec={'weight': conn_W_inhON1, "delay": 3.0}) #D_iON1})
nest.Connect(neuronON2, ipop2, syn_spec={'weight': conn_W_inhON2, "delay": 3.0}) #D_iON2})
nest.Connect(neuronOFF2, ipop2, syn_spec={'weight': conn_W_inhOFF2, "delay": 3.0}) #D_iOFF2})
nest.Connect(neuronOFF1, ipop1, syn_spec={'weight': conn_W_inhOFF1, "delay": 3.0}) #D_iOFF1})

#feedback to LGN!!!
nest.Connect(epop1, neuronON1, syn_spec={'weight': conn_Wfb_excON1, "delay": 0.1}) 
nest.Connect(epop2, neuronON2, syn_spec={'weight': conn_Wfb_excON2, "delay": 0.1}) 
nest.Connect(epop2, neuronOFF2, syn_spec={'weight':conn_Wfb_excOFF2, "delay": 0.1}) 
nest.Connect(epop1, neuronOFF1, syn_spec={'weight':conn_Wfb_excOFF1, "delay": 0.1}) 
nest.Connect(ipop1, neuronON1, syn_spec={'weight': conn_Wfb_inhON1, "delay": 0.1}) 
nest.Connect(ipop2, neuronON2, syn_spec={'weight': conn_Wfb_inhON2, "delay": 0.1}) 
nest.Connect(ipop2, neuronOFF2, syn_spec={'weight':conn_Wfb_inhOFF2, "delay": 0.1}) 
nest.Connect(ipop1, neuronOFF1, syn_spec={'weight':conn_Wfb_inhOFF1, "delay": 0.1}) 

nest.Connect(epop1, interneuron1ON1, syn_spec={'weight': conn_Wfb_excON1, "delay": 0.1}) 
nest.Connect(epop2, interneuron1ON2, syn_spec={'weight': conn_Wfb_excON2, "delay": 0.1}) 
nest.Connect(epop2, interneuron1OFF2, syn_spec={'weight':conn_Wfb_excOFF2, "delay": 0.1}) 
nest.Connect(epop1, interneuron1OFF1, syn_spec={'weight':conn_Wfb_excOFF1, "delay": 0.1}) 
nest.Connect(ipop1, interneuron1ON1, syn_spec={'weight': conn_Wfb_inhON1, "delay": 0.1}) 
nest.Connect(ipop2, interneuron1ON2, syn_spec={'weight': conn_Wfb_inhON2, "delay": 0.1}) 
nest.Connect(ipop2, interneuron1OFF2, syn_spec={'weight':conn_Wfb_inhOFF2, "delay": 0.1}) 
nest.Connect(ipop1, interneuron1OFF1, syn_spec={'weight':conn_Wfb_inhOFF1, "delay": 0.1}) 

nest.Connect(epop1, interneuron2ON1, syn_spec={'weight': conn_Wfb_excON1, "delay": 0.1}) 
nest.Connect(epop2, interneuron2ON2, syn_spec={'weight': conn_Wfb_excON2, "delay": 0.1}) 
nest.Connect(epop2, interneuron2OFF2, syn_spec={'weight':conn_Wfb_excOFF2, "delay": 0.1}) 
nest.Connect(epop1, interneuron2OFF1, syn_spec={'weight':conn_Wfb_excOFF1, "delay": 0.1}) 
nest.Connect(ipop1, interneuron2ON1, syn_spec={'weight': conn_Wfb_inhON1, "delay": 0.1}) 
nest.Connect(ipop2, interneuron2ON2, syn_spec={'weight': conn_Wfb_inhON2, "delay": 0.1}) 
nest.Connect(ipop2, interneuron2OFF2, syn_spec={'weight':conn_Wfb_inhOFF2, "delay": 0.1}) 
nest.Connect(ipop1, interneuron2OFF1, syn_spec={'weight':conn_Wfb_inhOFF1, "delay": 0.1}) 
#####

spdON1= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdON1", "withtime": True, "withgid": True}) #, "precise_times": True})
spdOFF1= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdOFF1", "withtime": True, "withgid": True}) #, "precise_times": True})
spdON2= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdON2", "withtime": True, "withgid": True}) #, "precise_times": True})
spdOFF2= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdOFF2", "withtime": True, "withgid": True}) #, "precise_times": True})

nest.Connect(neuronON1, spdON1)
nest.Connect(neuronOFF1, spdOFF1)
nest.Connect(neuronON2, spdON2)
nest.Connect(neuronOFF2, spdOFF2)

#Connecting populations
#================== epop1 connectinos begin ==================#
nest.Connect(epop1, epop1, syn_spec={'weight': conn_WE1E1, "delay": 3.0}) #D_e1e1})
nest.Connect(epop1, ipop1, syn_spec={'weight': conn_WE1I1, "delay": 3.0}) #D_e1i1})
#================== epop1 connectinos end ==================#

#================== epop2 connectinos begin ===============#
nest.Connect(epop2, epop2, syn_spec={'weight': conn_WE2E2, "delay": 3.0}) #D_e2e2})
nest.Connect(epop2, ipop2, syn_spec={'weight': conn_WE2I2, "delay": 3.0}) #D_e2i2})
#================== epop2 connectinos end ==================#

#================== ipop1 connectinos begin ===============#
nest.Connect(ipop1, ipop2, syn_spec={'weight': conn_WI1I2, "delay": 10.0}) #D_i1i2})
nest.Connect(ipop1, epop2, syn_spec={'weight': conn_WI1E2, "delay": 10.0}) #D_i1e2})
#================== ipop1 connectinos end ==================#

#================== ipop2 connectinos begin ===============#
nest.Connect(ipop2, ipop1, syn_spec={'weight': conn_WI2I1, "delay": 10.0}) #D_i2i1})
nest.Connect(ipop2, epop1, syn_spec={'weight': conn_WI2E1, "delay": 10.0}) #D_i2e1})
#================== ipop2 connectinos end ==================#
'''
pg = nest.Create("poisson_generator", params = {"start": 0.0, "stop": sim_time, "rate": 5.0})
n_LGN+=1
nest.Connect(pg, neuronON1)
nest.Connect(pg, neuronOFF1)
nest.Connect(pg, neuronON2)
nest.Connect(pg, neuronOFF2)
nest.Connect(pg, epop1)
nest.Connect(pg, epop2)
nest.Connect(pg, ipop1)
nest.Connect(pg, ipop2)'''

nest.Connect(epop1,spd_V1)
nest.Connect(epop2,spd_V1)
nest.Connect(ipop1,spd_V1)
nest.Connect(ipop2,spd_V1)

n_spd=np.zeros((len(orientations)))
spd_orientations= nest.Create("spike_detector", len(orientations), params = {"to_memory": True, "to_file": False, "label": "spd_orientations", "withtime": True, "withgid": True}) #, "precise_times": True})
for i in np.arange(0, len(orientations), 1):
	for j in np.arange(0,Ne,1):
		if (thetae1[j]==orientations[i]):
			nest.Connect([epop1[j]], [spd_orientations[i]])
			#print 'E1 ',j,' to spd ',i
                        n_spd[i]+=1
			
		if (thetae2[j]==orientations[i]):
			nest.Connect([epop2[j]], [spd_orientations[i]])
			#print 'E2 ',j,' to spd ',i
                        n_spd[i]+=1
			
	for j in np.arange(0,Ni,1):
		if (thetai1[j]==orientations[i]):
			nest.Connect([ipop1[j]], [spd_orientations[i]])
			#print 'I1 ',j,' to spd ',i
                        n_spd[i]+=1
			
		if (thetai2[j]==orientations[i]):
			nest.Connect([ipop2[j]], [spd_orientations[i]])
			#print 'I2 ',j,' to spd ',i
                        n_spd[i]+=1
### end V1 layer

#MT layer
f=open(params_folder+MT_size+'_MT_positions_bin.cpickle',"rb") 
MT_e_xy1=cPickle.load(f)
MT_e_xy2=cPickle.load(f)
MT_i_xy1=cPickle.load(f)
MT_i_xy2=cPickle.load(f)
MT_xe1=cPickle.load(f)
MT_ye1=cPickle.load(f)
MT_xe2=cPickle.load(f)
MT_ye2=cPickle.load(f)
MT_xi1=cPickle.load(f)
MT_yi1=cPickle.load(f)
MT_xi2=cPickle.load(f)
MT_yi2=cPickle.load(f)
f.close()
print 'MT positions: ', MT_xe1
'''
np.savetxt(fr_data_folder+'MT_positions_E1.txt',(MT_xe1,MT_ye1),fmt='%f')
np.savetxt(fr_data_folder+'MT_positions_E2.txt',(MT_xe2,MT_ye2),fmt='%f')
np.savetxt(fr_data_folder+'MT_positions_I1.txt',(MT_xi1,MT_yi1),fmt='%f')
np.savetxt(fr_data_folder+'MT_positions_I2.txt',(MT_xi2,MT_yi2),fmt='%f')'''

MT_Nxc_exc = len(MT_xe1)
MT_Nyc_exc = len(MT_ye1)
MT_Nxc_inh = len(MT_xi1)
MT_Nyc_inh = len(MT_yi1)

MT_Ne=MT_Nxc_exc*MT_Nyc_exc
MT_Ni=MT_Nxc_inh*MT_Nyc_inh

MT_epop1 = nest.Create("iaf_psc_exp", MT_Ne, epop_params)
MT_epop2 = nest.Create("iaf_psc_exp", MT_Ne, epop_params)
MT_ipop1 = nest.Create("iaf_psc_exp", MT_Ni, ipop_params)
MT_ipop2 = nest.Create("iaf_psc_exp", MT_Ni, ipop_params)

f = open(params_folder+'MT_orientations10pi_'+MT_size+'_bins_l'+str(lamb_MT)+'.cpickle', "rb")
MT_phases=cPickle.load(f)
MT_orientations=cPickle.load(f)

MT_thetae1=cPickle.load(f)
MT_phie1=cPickle.load(f)
MT_thetae2=cPickle.load(f)
MT_phie2=cPickle.load(f)

MT_thetai1=cPickle.load(f)
MT_phii1=cPickle.load(f)
MT_thetai2=cPickle.load(f)
MT_phii2=cPickle.load(f)
f.close()

f = open(params_folder+'MT10par_fbp'+str(scale_fb_MT)+'_V1_'+EI_size+'_MT'+MT_size+'_scale100_1_l'+str(lamb_MT)+'_we0.1_se0.1_wi0.1_si0.1.cpickle', "rb")
#f = open(fr_data_folder + 'MT10screen_fbp0.05_V1_20x10_MT20x10_scale1_1_l1.2_we0.1_se0.535526315789_wi0.1_si0.505774853801.cpickle', "rb")
W_ME1_VE1=cPickle.load(f)
W_ME1_VE2=cPickle.load(f)
W_ME1_VI1=cPickle.load(f)
W_ME1_VI2=cPickle.load(f)

W_ME2_VE1=cPickle.load(f)
W_ME2_VE2=cPickle.load(f)
W_ME2_VI1=cPickle.load(f)
W_ME2_VI2=cPickle.load(f)

W_MI1_VE1=cPickle.load(f)
W_MI1_VE2=cPickle.load(f)
W_MI1_VI1=cPickle.load(f)
W_MI1_VI2=cPickle.load(f)

W_MI2_VE1=cPickle.load(f)
W_MI2_VE2=cPickle.load(f)
W_MI2_VI1=cPickle.load(f)
W_MI2_VI2=cPickle.load(f)

Wfb_ME1_VE1=cPickle.load(f)
Wfb_ME1_VE2=cPickle.load(f)
Wfb_ME1_VI1=cPickle.load(f)
Wfb_ME1_VI2=cPickle.load(f)

Wfb_ME2_VE1=cPickle.load(f)
Wfb_ME2_VE2=cPickle.load(f)
Wfb_ME2_VI1=cPickle.load(f)
Wfb_ME2_VI2=cPickle.load(f)

Wfb_MI1_VE1=cPickle.load(f)
Wfb_MI1_VE2=cPickle.load(f)
Wfb_MI1_VI1=cPickle.load(f)
Wfb_MI1_VI2=cPickle.load(f)

Wfb_MI2_VE1=cPickle.load(f)
Wfb_MI2_VE2=cPickle.load(f)
Wfb_MI2_VI1=cPickle.load(f)
Wfb_MI2_VI2=cPickle.load(f)

'''
D_eON1 = cPickle.load(f)
D_eON2 = cPickle.load(f)
D_eOFF2 = cPickle.load(f)
D_eOFF1 = cPickle.load(f)
D_iON1 = cPickle.load(f)
D_iON2 = cPickle.load(f)
D_iOFF2 = cPickle.load(f)
D_iOFF1 = cPickle.load(f)
'''
MT_WE1E1 = cPickle.load(f)
#D_e1e1 = cPickle.load(f)
MT_WE1I1 = cPickle.load(f)
#D_e1i1 = cPickle.load(f)
MT_WE2E2 = cPickle.load(f)
#D_e2e2 = cPickle.load(f)
MT_WE2I2 = cPickle.load(f)
#D_e2i2 = cPickle.load(f)
MT_WI1I2 = cPickle.load(f)
#D_i1i2 = cPickle.load(f)
MT_WI1E2 = cPickle.load(f)
#D_i1e2 = cPickle.load(f)
MT_WI2I1 = cPickle.load(f)
#D_i2i1 = cPickle.load(f)
MT_WI2E1 = cPickle.load(f)
#D_i2e1 = cPickle.load(f)
f.close()

#remove ID
MT_epop1_ID, MT_epop1_ID_removed = function_rectangle(MT_e_xy1, "MT E1", MT_epop1, results_folder) 
MT_ipop1_ID, MT_ipop1_ID_removed = function_rectangle(MT_i_xy1, "MT I1", MT_ipop1, results_folder) 
MT_epop2_ID, MT_epop2_ID_removed = function_rectangle(MT_e_xy2, "MT E2", MT_epop2, results_folder) 
MT_ipop2_ID, MT_ipop2_ID_removed = function_rectangle(MT_i_xy2, "MT I2", MT_ipop2, results_folder)  

conn_W_ME1_VE1 = func_conn(W_ME1_VE1, epop1, MT_epop1, epop1_ID_removed, MT_epop1_ID_removed)
conn_W_ME1_VE2 = func_conn(W_ME1_VE2, epop2, MT_epop1, epop2_ID_removed, MT_epop1_ID_removed)
conn_W_ME1_VI1 = func_conn(W_ME1_VI1, ipop1, MT_epop1, ipop1_ID_removed, MT_epop1_ID_removed)
conn_W_ME1_VI2 = func_conn(W_ME1_VI2, ipop2, MT_epop1, ipop2_ID_removed, MT_epop1_ID_removed)

conn_W_ME2_VE1 = func_conn(W_ME2_VE1, epop1, MT_epop2, epop1_ID_removed, MT_epop2_ID_removed)
conn_W_ME2_VE2 = func_conn(W_ME2_VE2, epop2, MT_epop2, epop2_ID_removed, MT_epop2_ID_removed)
conn_W_ME2_VI1 = func_conn(W_ME2_VI1, ipop1, MT_epop2, ipop1_ID_removed, MT_epop2_ID_removed)
conn_W_ME2_VI2 = func_conn(W_ME2_VI2, ipop2, MT_epop2, ipop2_ID_removed, MT_epop2_ID_removed)

conn_W_MI1_VE1 = func_conn(W_MI1_VE1, epop1, MT_ipop1, epop1_ID_removed, MT_ipop1_ID_removed)
conn_W_MI1_VE2 = func_conn(W_MI1_VE2, epop2, MT_ipop1, epop2_ID_removed, MT_ipop1_ID_removed)
conn_W_MI1_VI1 = func_conn(W_MI1_VI1, ipop1, MT_ipop1, ipop1_ID_removed, MT_ipop1_ID_removed)
conn_W_MI1_VI2 = func_conn(W_MI1_VI2, ipop2, MT_ipop1, ipop2_ID_removed, MT_ipop1_ID_removed)

conn_W_MI2_VE1 = func_conn(W_MI2_VE1, epop1, MT_ipop2, epop1_ID_removed, MT_ipop2_ID_removed)
conn_W_MI2_VE2 = func_conn(W_MI2_VE2, epop2, MT_ipop2, epop2_ID_removed, MT_ipop2_ID_removed)
conn_W_MI2_VI1 = func_conn(W_MI2_VI1, ipop1, MT_ipop2, ipop1_ID_removed, MT_ipop2_ID_removed)
conn_W_MI2_VI2 = func_conn(W_MI2_VI2, ipop2, MT_ipop2, ipop2_ID_removed, MT_ipop2_ID_removed)



conn_Wfb_ME1_VE1 = func_conn(Wfb_ME1_VE1, MT_epop1, epop1, MT_epop1_ID_removed, epop1_ID_removed)
conn_Wfb_ME1_VE2 = func_conn(Wfb_ME1_VE2, MT_epop1, epop2, MT_epop1_ID_removed, epop2_ID_removed)
conn_Wfb_ME1_VI1 = func_conn(Wfb_ME1_VI1, MT_epop1, ipop1, MT_epop1_ID_removed, ipop1_ID_removed)
conn_Wfb_ME1_VI2 = func_conn(Wfb_ME1_VI2, MT_epop1, ipop2, MT_epop1_ID_removed, ipop2_ID_removed)

conn_Wfb_ME2_VE1 = func_conn(Wfb_ME2_VE1, MT_epop2, epop1, MT_epop2_ID_removed, epop1_ID_removed)
conn_Wfb_ME2_VE2 = func_conn(Wfb_ME2_VE2, MT_epop2, epop2, MT_epop2_ID_removed, epop2_ID_removed)
conn_Wfb_ME2_VI1 = func_conn(Wfb_ME2_VI1, MT_epop2, ipop1, MT_epop2_ID_removed, ipop1_ID_removed)
conn_Wfb_ME2_VI2 = func_conn(Wfb_ME2_VI2, MT_epop2, ipop2, MT_epop2_ID_removed, ipop2_ID_removed)

conn_Wfb_MI1_VE1 = func_conn(Wfb_MI1_VE1, MT_ipop1, epop1, MT_ipop1_ID_removed, epop1_ID_removed)
conn_Wfb_MI1_VE2 = func_conn(Wfb_MI1_VE2, MT_ipop1, epop2, MT_ipop1_ID_removed, epop2_ID_removed)
conn_Wfb_MI1_VI1 = func_conn(Wfb_MI1_VI1, MT_ipop1, ipop1, MT_ipop1_ID_removed, ipop1_ID_removed)
conn_Wfb_MI1_VI2 = func_conn(Wfb_MI1_VI2, MT_ipop1, ipop2, MT_ipop1_ID_removed, ipop2_ID_removed)

conn_Wfb_MI2_VE1 = func_conn(Wfb_MI2_VE1, MT_ipop2, epop1, MT_ipop2_ID_removed, epop1_ID_removed)
conn_Wfb_MI2_VE2 = func_conn(Wfb_MI2_VE2, MT_ipop2, epop2, MT_ipop2_ID_removed, epop2_ID_removed)
conn_Wfb_MI2_VI1 = func_conn(Wfb_MI2_VI1, MT_ipop2, ipop1, MT_ipop2_ID_removed, ipop1_ID_removed)
conn_Wfb_MI2_VI2 = func_conn(Wfb_MI2_VI2, MT_ipop2, ipop2, MT_ipop2_ID_removed, ipop2_ID_removed)

conn_MT_WE1E1 = func_conn(MT_WE1E1, MT_epop1, MT_epop1, MT_epop1_ID_removed, MT_epop1_ID_removed)
conn_MT_WE1I1 = func_conn(MT_WE1I1, MT_epop1, MT_ipop1, MT_epop1_ID_removed, MT_ipop1_ID_removed)
conn_MT_WI1I2 = func_conn(MT_WI1I2, MT_ipop1, MT_ipop2, MT_ipop1_ID_removed, MT_ipop2_ID_removed)
conn_MT_WI1E2 = func_conn(MT_WI1E2, MT_ipop1, MT_epop2, MT_ipop1_ID_removed, MT_epop2_ID_removed)
conn_MT_WE2E2 = func_conn(MT_WE2E2, MT_epop2, MT_epop2, MT_epop2_ID_removed, MT_epop2_ID_removed)
conn_MT_WI2E1 = func_conn(MT_WI2E1, MT_ipop2, MT_epop1, MT_ipop2_ID_removed, MT_epop1_ID_removed)
conn_MT_WI2I1 = func_conn(MT_WI2I1, MT_ipop2, MT_ipop1, MT_ipop2_ID_removed, MT_ipop1_ID_removed)
conn_MT_WE2I2 = func_conn(MT_WE2I2, MT_epop2, MT_ipop2, MT_epop2_ID_removed, MT_ipop2_ID_removed)


spd_MT= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})

nest.Connect(epop1, MT_epop1, syn_spec={'weight': conn_W_ME1_VE1, "delay": 3.0}) #D_eON1})
nest.Connect(epop2, MT_epop1, syn_spec={'weight': conn_W_ME1_VE2, "delay": 3.0}) #D_eON1})
nest.Connect(ipop1, MT_epop1, syn_spec={'weight': conn_W_ME1_VI1, "delay": 3.0}) #D_eON1})
nest.Connect(ipop2, MT_epop1, syn_spec={'weight': conn_W_ME1_VI2, "delay": 3.0}) #D_eON1})

nest.Connect(epop1, MT_epop2, syn_spec={'weight': conn_W_ME2_VE1, "delay": 3.0}) #D_eON1})
nest.Connect(epop2, MT_epop2, syn_spec={'weight': conn_W_ME2_VE2, "delay": 3.0}) #D_eON1})
nest.Connect(ipop1, MT_epop2, syn_spec={'weight': conn_W_ME2_VI1, "delay": 3.0}) #D_eON1})
nest.Connect(ipop2, MT_epop2, syn_spec={'weight': conn_W_ME2_VI2, "delay": 3.0}) #D_eON1})

nest.Connect(epop1, MT_ipop1, syn_spec={'weight': conn_W_MI1_VE1, "delay": 3.0}) #D_eON1})
nest.Connect(epop2, MT_ipop1, syn_spec={'weight': conn_W_MI1_VE2, "delay": 3.0}) #D_eON1})
nest.Connect(ipop1, MT_ipop1, syn_spec={'weight': conn_W_MI1_VI1, "delay": 3.0}) #D_eON1})
nest.Connect(ipop2, MT_ipop1, syn_spec={'weight': conn_W_MI1_VI2, "delay": 3.0}) #D_eON1})

nest.Connect(epop1, MT_ipop2, syn_spec={'weight': conn_W_MI2_VE1, "delay": 3.0}) #D_eON1})
nest.Connect(epop2, MT_ipop2, syn_spec={'weight': conn_W_MI2_VE2, "delay": 3.0}) #D_eON1})
nest.Connect(ipop1, MT_ipop2, syn_spec={'weight': conn_W_MI2_VI1, "delay": 3.0}) #D_eON1})
nest.Connect(ipop2, MT_ipop2, syn_spec={'weight': conn_W_MI2_VI2, "delay": 3.0}) #D_eON1})

#feedback connections
nest.Connect(MT_epop1, epop1, syn_spec={'weight': conn_Wfb_ME1_VE1, "delay": 0.1})
nest.Connect(MT_epop1, epop2, syn_spec={'weight': conn_Wfb_ME1_VE2, "delay": 0.1})
nest.Connect(MT_epop1, ipop1, syn_spec={'weight': conn_Wfb_ME1_VI1, "delay": 0.1})
nest.Connect(MT_epop1, ipop2, syn_spec={'weight': conn_Wfb_ME1_VI2, "delay": 0.1})

nest.Connect(MT_epop2, epop1, syn_spec={'weight': conn_Wfb_ME2_VE1, "delay": 0.1})
nest.Connect(MT_epop2, epop2, syn_spec={'weight': conn_Wfb_ME2_VE2, "delay": 0.1})
nest.Connect(MT_epop2, ipop1, syn_spec={'weight': conn_Wfb_ME2_VI1, "delay": 0.1})
nest.Connect(MT_epop2, ipop2, syn_spec={'weight': conn_Wfb_ME2_VI2, "delay": 0.1})

nest.Connect(MT_ipop1, epop1, syn_spec={'weight': conn_Wfb_MI1_VE1, "delay": 0.1})
nest.Connect(MT_ipop1, epop2, syn_spec={'weight': conn_Wfb_MI1_VE2, "delay": 0.1})
nest.Connect(MT_ipop1, ipop1, syn_spec={'weight': conn_Wfb_MI1_VI1, "delay": 0.1})
nest.Connect(MT_ipop1, ipop2, syn_spec={'weight': conn_Wfb_MI1_VI2, "delay": 0.1})

nest.Connect(MT_ipop2, epop1, syn_spec={'weight': conn_Wfb_MI2_VE1, "delay": 0.1})
nest.Connect(MT_ipop2, epop2, syn_spec={'weight': conn_Wfb_MI2_VE2, "delay": 0.1})
nest.Connect(MT_ipop2, ipop1, syn_spec={'weight': conn_Wfb_MI2_VI1, "delay": 0.1})
nest.Connect(MT_ipop2, ipop2, syn_spec={'weight': conn_Wfb_MI2_VI2, "delay": 0.1})


#Connecting populations MT
#================== epop1 connectinos begin ==================#
nest.Connect(MT_epop1, MT_epop1, syn_spec={'weight': conn_MT_WE1E1, "delay": 3.0}) #D_e1e1})
nest.Connect(MT_epop1, MT_ipop1, syn_spec={'weight': conn_MT_WE1I1, "delay": 3.0}) #D_e1i1})
#================== epop1 connectinos end ==================#

#================== epop2 connectinos begin ===============#
nest.Connect(MT_epop2, MT_epop2, syn_spec={'weight': conn_MT_WE2E2, "delay": 3.0}) #D_e2e2})
nest.Connect(MT_epop2, MT_ipop2, syn_spec={'weight': conn_MT_WE2I2, "delay": 3.0}) #D_e2i2})
#================== epop2 connectinos end ==================#

#================== ipop1 connectinos begin ===============#
nest.Connect(MT_ipop1, MT_ipop2, syn_spec={'weight': conn_MT_WI1I2, "delay": 10.0}) #D_i1i2})
nest.Connect(MT_ipop1, MT_epop2, syn_spec={'weight': conn_MT_WI1E2, "delay": 10.0}) #D_i1e2})
#================== ipop1 connectinos end ==================#

#================== ipop2 connectinos begin ===============#
nest.Connect(MT_ipop2, MT_ipop1, syn_spec={'weight': conn_MT_WI2I1, "delay": 10.0}) #D_i2i1})
nest.Connect(MT_ipop2, MT_epop1, syn_spec={'weight': conn_MT_WI2E1, "delay": 10.0}) #D_i2e1})
#================== ipop2 connectinos end ==================#

nest.Connect(MT_epop1,spd_MT)
nest.Connect(MT_epop2,spd_MT)
nest.Connect(MT_ipop1,spd_MT)
nest.Connect(MT_ipop2,spd_MT)

n_spd_MT=np.zeros((len(MT_orientations)))
spd_orientations_MT= nest.Create("spike_detector", len(MT_orientations), params = {"to_memory": True, "to_file": False, "label": "spd_orientations", "withtime": True, "withgid": True}) #, "precise_times": True})
for i in np.arange(0, len(MT_orientations), 1):
	for j in np.arange(0,MT_Ne,1):
		if (MT_thetae1[j]==MT_orientations[i]):
			nest.Connect([MT_epop1[j]], [spd_orientations_MT[i]])
			#print 'E1 ',j,' to spd ',i
                        n_spd_MT[i]+=1
			
		if (MT_thetae2[j]==MT_orientations[i]):
			nest.Connect([MT_epop2[j]], [spd_orientations_MT[i]])
			#print 'E2 ',j,' to spd ',i
                        n_spd_MT[i]+=1
			
	for j in np.arange(0,MT_Ni,1):
		if (MT_thetai1[j]==MT_orientations[i]):
			nest.Connect([MT_ipop1[j]], [spd_orientations_MT[i]])
			#print 'I1 ',j,' to spd ',i
                        n_spd_MT[i]+=1
			
		if (MT_thetai2[j]==MT_orientations[i]):
			nest.Connect([MT_ipop2[j]], [spd_orientations_MT[i]])
			#print 'I2 ',j,' to spd ',i
                        n_spd_MT[i]+=1
### end MT layer

### MSTd layer
f=open(params_folder+MSTd_size+'_MSTd_positions_bin.cpickle',"rb") 
MSTd_e_xy=cPickle.load(f)
MSTd_c_xy=cPickle.load(f)
MSTd_xe=cPickle.load(f)
MSTd_ye=cPickle.load(f)
MSTd_xc=cPickle.load(f)
MSTd_yc=cPickle.load(f)
f.close()
print 'MST positions: ', MSTd_xe
MSTd_exp_x = len(MSTd_xe)
MSTd_exp_y = len(MSTd_ye)
MSTd_cont_x = len(MSTd_xc)
MSTd_cont_y = len(MSTd_yc)

MSTd_Ne=MSTd_exp_x*MSTd_exp_y
MSTd_Nc=MSTd_cont_x*MSTd_cont_y

MSTd_exppop = nest.Create("iaf_psc_exp", MSTd_Ne, epop_params)
MSTd_conpop = nest.Create("iaf_psc_exp", MSTd_Nc, epop_params)

MST_params_filename = params_folder+'MSTd_'+MSTd_size+'_MT' + MT_size+ '_nfp' + str(nfp)+ '_d' + str(d)+'_sts'+str(sigma_ts)+'_sI' + str(sigma_I)+'.cpickle'
f = open(MST_params_filename, "rb") 
cx=cPickle.load(f)
cy=cPickle.load(f)
focal_points=cPickle.load(f)

MSTc_fpn=cPickle.load(f)
MSTc_fp=cPickle.load(f)
W_MTe1_MSTc=cPickle.load(f)
W_MTe2_MSTc=cPickle.load(f)
W_MTi1_MSTc=cPickle.load(f)
W_MTi2_MSTc=cPickle.load(f)

MSTe_fpn=cPickle.load(f)
MSTe_fp=cPickle.load(f)
W_MTe1_MSTe=cPickle.load(f)
W_MTe2_MSTe=cPickle.load(f)
W_MTi1_MSTe=cPickle.load(f)
W_MTi2_MSTe=cPickle.load(f)

W_MSTe_i=cPickle.load(f)
W_MSTe_e=cPickle.load(f)

W_MSTc_i=cPickle.load(f)
W_MSTc_e=cPickle.load(f)

W_MST_ec=cPickle.load(f)
W_MST_ce=cPickle.load(f)

f.close()

#remove ID
MSTd_exppop_ID, MSTd_exppop_ID_removed = function_rectangle(MSTd_e_xy, "MSTe", MSTd_exppop, results_folder) 
MSTd_conpop_ID, MSTd_conpop_ID_removed = function_rectangle(MSTd_c_xy, "MSTc", MSTd_conpop, results_folder)

conn_W_MTe1_MSTe = func_conn(W_MTe1_MSTe, MT_epop1, MSTd_exppop, MT_epop1_ID_removed, MSTd_exppop_ID_removed)
conn_W_MTe1_MSTc = func_conn(W_MTe1_MSTc, MT_epop1, MSTd_conpop, MT_epop1_ID_removed, MSTd_conpop_ID_removed)
conn_W_MTe2_MSTe = func_conn(W_MTe2_MSTe, MT_epop2, MSTd_exppop, MT_epop2_ID_removed, MSTd_exppop_ID_removed)
conn_W_MTe2_MSTc = func_conn(W_MTe2_MSTc, MT_epop2, MSTd_conpop, MT_epop2_ID_removed, MSTd_conpop_ID_removed)

conn_W_MTi1_MSTe = func_conn(W_MTi1_MSTe, MT_ipop1, MSTd_exppop, MT_ipop1_ID_removed, MSTd_exppop_ID_removed)
conn_W_MTi1_MSTc = func_conn(W_MTi1_MSTc, MT_ipop1, MSTd_conpop, MT_ipop1_ID_removed, MSTd_conpop_ID_removed)
conn_W_MTi2_MSTe = func_conn(W_MTi2_MSTe, MT_ipop2, MSTd_exppop, MT_ipop2_ID_removed, MSTd_exppop_ID_removed)
conn_W_MTi2_MSTc = func_conn(W_MTi2_MSTc, MT_ipop2, MSTd_conpop, MT_ipop2_ID_removed, MSTd_conpop_ID_removed)

conn_W_MSTe_i = func_conn(W_MSTe_i, MSTd_exppop, MSTd_exppop, MSTd_exppop_ID_removed, MSTd_exppop_ID_removed)
conn_W_MSTe_e = func_conn(W_MSTe_e, MSTd_exppop, MSTd_exppop, MSTd_exppop_ID_removed, MSTd_exppop_ID_removed)
conn_W_MSTc_i = func_conn(W_MSTc_i, MSTd_conpop, MSTd_conpop, MSTd_conpop_ID_removed, MSTd_conpop_ID_removed)
conn_W_MSTc_e = func_conn(W_MSTc_e, MSTd_conpop, MSTd_conpop, MSTd_conpop_ID_removed, MSTd_conpop_ID_removed)
conn_W_MST_ce = func_conn(W_MST_ce, MSTd_conpop, MSTd_exppop, MSTd_conpop_ID_removed, MSTd_exppop_ID_removed)
conn_W_MST_ec = func_conn(W_MST_ec, MSTd_exppop, MSTd_conpop, MSTd_exppop_ID_removed, MSTd_conpop_ID_removed)



nest.Connect(MT_epop1, MSTd_exppop, syn_spec={'weight': conn_W_MTe1_MSTe, "delay": 3.0})
nest.Connect(MT_epop1, MSTd_conpop, syn_spec={'weight': conn_W_MTe1_MSTc, "delay": 3.0})
nest.Connect(MT_epop2, MSTd_exppop, syn_spec={'weight': conn_W_MTe2_MSTe, "delay": 3.0})
nest.Connect(MT_epop2, MSTd_conpop, syn_spec={'weight': conn_W_MTe2_MSTc, "delay": 3.0})

nest.Connect(MT_ipop1, MSTd_exppop, syn_spec={'weight': conn_W_MTi1_MSTe, "delay": 3.0})
nest.Connect(MT_ipop1, MSTd_conpop, syn_spec={'weight': conn_W_MTi1_MSTc, "delay": 3.0})
nest.Connect(MT_ipop2, MSTd_exppop, syn_spec={'weight': conn_W_MTi2_MSTe, "delay": 3.0})
nest.Connect(MT_ipop2, MSTd_conpop, syn_spec={'weight': conn_W_MTi2_MSTc, "delay": 3.0})

nest.Connect(MSTd_exppop, MSTd_exppop, syn_spec={'weight': conn_W_MSTe_i, "delay": 3.0})
nest.Connect(MSTd_exppop, MSTd_exppop, syn_spec={'weight': conn_W_MSTe_e, "delay": 3.0})

nest.Connect(MSTd_conpop, MSTd_conpop, syn_spec={'weight': conn_W_MSTc_i, "delay": 3.0})
nest.Connect(MSTd_conpop, MSTd_conpop, syn_spec={'weight': conn_W_MSTc_e, "delay": 3.0})

nest.Connect(MSTd_conpop, MSTd_exppop, syn_spec={'weight': conn_W_MST_ce, "delay": 3.0})
nest.Connect(MSTd_exppop, MSTd_conpop, syn_spec={'weight': conn_W_MST_ec, "delay": 3.0})

#feedback connections MSTd to MT
scale_fb_MST=0.05

nest.Connect(MSTd_exppop, MT_epop1, syn_spec={'weight': np.transpose(conn_W_MTe1_MSTe)*scale_fb_MST, "delay": 0.1})
nest.Connect(MSTd_conpop, MT_epop1, syn_spec={'weight': np.transpose(conn_W_MTe1_MSTc)*scale_fb_MST, "delay": 0.1})
nest.Connect(MSTd_exppop, MT_epop2, syn_spec={'weight': np.transpose(conn_W_MTe2_MSTe)*scale_fb_MST, "delay": 0.1})
nest.Connect(MSTd_conpop, MT_epop2, syn_spec={'weight': np.transpose(conn_W_MTe2_MSTc)*scale_fb_MST, "delay": 0.1})

nest.Connect(MSTd_exppop, MT_ipop1, syn_spec={'weight': np.transpose(conn_W_MTi1_MSTe)*scale_fb_MST, "delay": 0.1})
nest.Connect(MSTd_conpop, MT_ipop1, syn_spec={'weight': np.transpose(conn_W_MTi1_MSTc)*scale_fb_MST, "delay": 0.1})
nest.Connect(MSTd_exppop, MT_ipop2, syn_spec={'weight': np.transpose(conn_W_MTi2_MSTe)*scale_fb_MST, "delay": 0.1})
nest.Connect(MSTd_conpop, MT_ipop2, syn_spec={'weight': np.transpose(conn_W_MTi2_MSTc)*scale_fb_MST, "delay": 0.1})

spd_MSTc= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})
spd_MSTe= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})
nest.Connect(MSTd_conpop,spd_MSTc)
nest.Connect(MSTd_exppop,spd_MSTe)

n_spd_MST_fpe=np.zeros((len(focal_points)))
spd_fpe_MST= nest.Create("spike_detector", len(focal_points), params = {"to_memory": True, "to_file": False, "label": "spd_orientations", "withtime": True, "withgid": True}) #, "precise_times": True})

for i in np.arange(0, MSTd_Ne,1):
	for j in np.arange(0, len(n_spd_MST_fpe), 1):
		if MSTe_fpn[i]==j:
			nest.Connect([MSTd_exppop[i]], [spd_fpe_MST[j]])
			n_spd_MST_fpe[j]+=1	

n_spd_MST_fpc=np.zeros((len(focal_points)))
spd_fpc_MST= nest.Create("spike_detector", len(focal_points), params = {"to_memory": True, "to_file": False, "label": "spd_orientations", "withtime": True, "withgid": True}) #, "precise_times": True})

for i in np.arange(0, MSTd_Nc,1):
	for j in np.arange(0, len(n_spd_MST_fpc), 1):
		if MSTe_fpn[i]==j:
			nest.Connect([MSTd_conpop[i]], [spd_fpc_MST[j]])
			n_spd_MST_fpc[j]+=1
### end MSTd layer

### LIP layer
f=open(params_folder+LIP_size+'_LIP_positions_bin.cpickle',"rb") 
LIP_l_xy=cPickle.load(f)
LIP_r_xy=cPickle.load(f)
LIP_xl=cPickle.load(f)
LIP_yl=cPickle.load(f)
LIP_xr=cPickle.load(f)
LIP_yr=cPickle.load(f)
f.close()
print 'LIP positions: ', LIP_xl
LIP_left_x = len(LIP_xl)
LIP_left_y = len(LIP_yl)
LIP_right_x = len(LIP_xr)
LIP_right_y = len(LIP_yr)

LIP_Nl=LIP_left_x*LIP_left_y
LIP_Nr=LIP_right_x*LIP_right_y

lip_params={"V_th": -69.992,"V_reset": -80.0,"I_e": 200.0}
LIP_left = nest.Create("iaf_psc_exp", LIP_Nl, lip_params)
LIP_right = nest.Create("iaf_psc_exp", LIP_Nr, lip_params)
#wlip=10000.0

LIP_params_filename = params_folder+'LIP_'+LIP_size+ '_MSTd'+MSTd_size+'_nfp'+str(nfp)+'_s'+str(sigma_lip)+'.cpickle'
#print 'LIP_params_filename: ', LIP_params_filename
f = open(LIP_params_filename, "rb") 
WLIPl_MSTdc=cPickle.load(f)
WLIPl_MSTde=cPickle.load(f)
WLIPr_MSTdc=cPickle.load(f)
WLIPr_MSTde=cPickle.load(f)
WLIP_LR=cPickle.load(f)
WLIP_L=cPickle.load(f)
WLIP_R=cPickle.load(f)
f.close()

nest.Connect(LIP_left, LIP_right, syn_spec={'weight': WLIP_LR, "delay": 1.0})
nest.Connect(LIP_right, LIP_left, syn_spec={'weight': np.transpose(WLIP_LR), "delay": 1.0})

'''nest.Connect(LIP_left, LIP_left, syn_spec={'weight': WLIP_L, "delay": 1.0})
nest.Connect(LIP_right, LIP_right, syn_spec={'weight': WLIP_R, "delay": 1.0})'''

glip=2.0
lambda_lip=0.1

#remove ID

LIP_left_ID, LIP_left_ID_removed = function_rectangle(LIP_l_xy, "LIP left", LIP_left, results_folder)
LIP_right_ID,LIP_right_ID_removed = function_rectangle(LIP_r_xy, "LIP right", LIP_right, results_folder)


conn_WLIPl_MSTde = func_conn(WLIPl_MSTde, MSTd_exppop, LIP_left, MSTd_exppop_ID_removed, LIP_left_ID_removed)
conn_WLIPr_MSTde = func_conn(WLIPr_MSTde, MSTd_exppop, LIP_right, MSTd_exppop_ID_removed, LIP_right_ID_removed)
conn_WLIPl_MSTdc = func_conn(WLIPl_MSTdc, MSTd_conpop, LIP_left, MSTd_conpop_ID_removed, LIP_left_ID_removed)
conn_WLIPr_MSTdc = func_conn(WLIPr_MSTdc, MSTd_conpop, LIP_right, MSTd_conpop_ID_removed, LIP_right_ID_removed)


nest.Connect(MSTd_exppop, LIP_left, syn_spec={'model': 'stdp_synapse', 'weight': conn_WLIPl_MSTde, "Wmax": np.array(conn_WLIPl_MSTde)*glip , "mu_plus": 1.0, "mu_minus": 1.0, "lambda": lambda_lip, "delay": 1.0})
nest.Connect(MSTd_exppop, LIP_right, syn_spec={'model': 'stdp_synapse', 'weight':conn_WLIPr_MSTde, "Wmax": np.array(conn_WLIPr_MSTde)*glip , "mu_plus": 1.0, "mu_minus": 1.0, "lambda": lambda_lip, "delay": 1.0})
nest.Connect(MSTd_conpop, LIP_left, syn_spec={'model': 'stdp_synapse', 'weight': conn_WLIPl_MSTdc, "Wmax": np.array(conn_WLIPl_MSTdc)*glip , "mu_plus": 1.0, "mu_minus": 1.0, "lambda": lambda_lip, "delay": 1.0})
nest.Connect(MSTd_conpop, LIP_right, syn_spec={'model': 'stdp_synapse', 'weight':conn_WLIPr_MSTdc, "Wmax": np.array(conn_WLIPr_MSTdc)*glip , "mu_plus": 1.0, "mu_minus": 1.0, "lambda": lambda_lip, "delay": 1.0})

#feedback LIP to MST
scale_fb_LIP_MST=0.01

nest.Connect(LIP_left, MSTd_exppop, syn_spec={'weight': np.transpose(conn_WLIPl_MSTde)*scale_fb_LIP_MST, "delay": 1.0})
nest.Connect(LIP_right, MSTd_exppop, syn_spec={'weight': np.transpose(conn_WLIPr_MSTde)*scale_fb_LIP_MST, "delay": 1.0})
nest.Connect(LIP_left, MSTd_conpop, syn_spec={'weight': np.transpose(conn_WLIPl_MSTdc)*scale_fb_LIP_MST, "delay": 1.0})
nest.Connect(LIP_right, MSTd_conpop, syn_spec={'weight': np.transpose(conn_WLIPr_MSTdc)*scale_fb_LIP_MST, "delay": 1.0})

spd_LIPl= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})
spd_LIPr= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})
'''
mm_LIPl= nest.Create("multimeter", params = {"to_memory": True, "to_file": False, "withtime": True, "withgid": True, "record_from": ["Ca"]})
mm_LIPr= nest.Create("multimeter", params = {"to_memory": True, "to_file": False, "withtime": True, "withgid": True, "record_from": ["Ca"]})'''

nest.Connect(LIP_left,spd_LIPl)
nest.Connect(LIP_right,spd_LIPr)
'''
nest.Connect(mm_LIPl,LIP_left)
nest.Connect(mm_LIPr,LIP_right)'''
			
### end LIP layer

## BG
f = open(params_folder+layer_size+"_layer_positions_bin.cpickle", "rb")
l_xy=cPickle.load(f)
r_xy=cPickle.load(f)
l_x=cPickle.load(f)
l_y=cPickle.load(f)
r_x=cPickle.load(f)
r_y=cPickle.load(f)
f.close()

left_x = len(l_x)
left_y = len(l_y)
right_x = len(r_x)
right_y = len(r_y)

BG_Nl=left_x*left_y
BG_Nr=right_x*right_y

bg_params={"V_th": -69.992,"V_reset": -80.0,"I_e": 100.0}

D1_left = nest.Create("iaf_psc_exp", BG_Nl, bg_params)
D1_right = nest.Create("iaf_psc_exp", BG_Nr, bg_params)

D2_left = nest.Create("iaf_psc_exp", BG_Nl, bg_params)
D2_right = nest.Create("iaf_psc_exp", BG_Nr, bg_params)

GPe_left = nest.Create("iaf_psc_exp", BG_Nl, bg_params)
GPe_right = nest.Create("iaf_psc_exp", BG_Nr, bg_params)

STN_left = nest.Create("iaf_psc_exp", BG_Nl, bg_params)
STN_right = nest.Create("iaf_psc_exp", BG_Nr, bg_params)

SNr_left = nest.Create("iaf_psc_exp", BG_Nl, bg_params)
SNr_right = nest.Create("iaf_psc_exp", BG_Nr, bg_params)

SNc_left = nest.Create("iaf_psc_exp", BG_Nl, bg_params)
SNc_right = nest.Create("iaf_psc_exp", BG_Nr, bg_params)

# reinforcement signal
nest.SetStatus(SNc_left, {'I_e': 100.0})
nest.SetStatus(SNc_right, {'I_e': 100.0})

SC_left = nest.Create("iaf_psc_exp", BG_Nl, bg_params)
SC_right = nest.Create("iaf_psc_exp", BG_Nr, bg_params)

BG_params_filename = params_folder+'BG_conn_LIP_'+LIP_size+ '_layer'+layer_size+'_s'+str(sigma_bg)+'.cpickle'
f = open(BG_params_filename, "rb") 
WLIP_D1_l=cPickle.load(f)
WLIP_D1_r=cPickle.load(f)
WLIP_D2_l=cPickle.load(f)
WLIP_D2_r=cPickle.load(f)
WD1l_lateral=cPickle.load(f)
WD1r_lateral=cPickle.load(f)
WD2l_lateral=cPickle.load(f)
WD2r_lateral=cPickle.load(f)
WGPe_GPe_l=cPickle.load(f)
WGPe_GPe_r=cPickle.load(f)
WSNr_l=cPickle.load(f)
WSNr_r=cPickle.load(f)
WSNr_SC_l=cPickle.load(f)
WSNr_SC_r=cPickle.load(f)
WLIP_SC_l=cPickle.load(f)
WLIP_SC_r=cPickle.load(f)
'''WD1_SNr_l=cPickle.load(f)
WD1_SNr_r=cPickle.load(f)
WD1_SNc_l=cPickle.load(f)
WD1_SNc_r=cPickle.load(f)'''
WSTNl_SNrr=cPickle.load(f)
WSTNr_SNrl=cPickle.load(f)

f.close()

SNc_vtl = nest.Create('volume_transmitter')
SNc_vtr = nest.Create('volume_transmitter')
nest.Connect(SNc_left, SNc_vtl,'all_to_all')
nest.Connect(SNc_right, SNc_vtr,'all_to_all')

nest.CopyModel('stdp_dopamine_synapse', 'dopsyn_l', {'vt': SNc_vtl[0], 'Wmin':-2000.0, 'Wmax':2000.0})
nest.CopyModel('stdp_dopamine_synapse', 'dopsyn_r', {'vt': SNc_vtr[0], 'Wmin':-2000.0, 'Wmax':2000.0})

nest.CopyModel('stdp_dopamine_synapse', 'dopsyn_lm', {'vt': SNc_vtl[0], 'A_plus': -1.0, 'A_minus': -1.5, 'Wmin':-2000.0, 'Wmax':2000.0})
nest.CopyModel('stdp_dopamine_synapse', 'dopsyn_rm', {'vt': SNc_vtr[0], 'A_plus': -1.0, 'A_minus': -1.5, 'Wmin':-2000.0, 'Wmax':2000.0})

'''
nest.Connect(LIP_left, D1_left, syn_spec={'weight': WLIP_D1_l, "delay": 1.0, 'model': 'dopsyn_l'})
nest.Connect(LIP_right, D1_right, syn_spec={'weight': WLIP_D1_r, "delay": 1.0, 'model': 'dopsyn_r'})

nest.Connect(LIP_left, D2_left, syn_spec={'weight': WLIP_D1_l, "delay": 1.0, 'model': 'dopsyn_lm'})
nest.Connect(LIP_right, D2_right, syn_spec={'weight': WLIP_D1_r, "delay": 1.0, 'model': 'dopsyn_rm'})'''

nest.Connect(LIP_left, D1_left, 'all_to_all', syn_spec={'weight': 100.0, "delay": 1.0, 'model': 'dopsyn_l'})
nest.Connect(LIP_right, D1_right, 'all_to_all', syn_spec={'weight': 100.0, "delay": 1.0, 'model': 'dopsyn_r'})

nest.Connect(LIP_left, D2_left, 'all_to_all', syn_spec={'weight': 100.0, "delay": 1.0, 'model': 'dopsyn_lm'})
nest.Connect(LIP_right, D2_right, 'all_to_all', syn_spec={'weight': 100.0, "delay": 1.0, 'model': 'dopsyn_rm'})

nest.Connect(D1_left, D1_left, syn_spec={'weight': WD1l_lateral, "delay": 1.0})
nest.Connect(D1_right, D1_right, syn_spec={'weight': WD1r_lateral, "delay": 1.0})
nest.Connect(D2_left, D2_left, syn_spec={'weight': WD2l_lateral, "delay": 1.0})
nest.Connect(D2_right, D2_right, syn_spec={'weight': WD2r_lateral, "delay": 1.0})

nest.Connect(D2_left, GPe_left, 'one_to_one', syn_spec={'weight': -100.0, "delay": 1.0}) #-10000.0
nest.Connect(D2_right, GPe_right, 'one_to_one', syn_spec={'weight': -100.0, "delay": 1.0}) #-10000.0

Gama=0.9
# Value function V(t)
nest.Connect(D1_left, SNc_left, 'one_to_one', syn_spec={'weight': -1000.0, "delay": 1.0, 'model': 'dopsyn_l'}) #-100.0
nest.Connect(D1_right, SNc_right, 'one_to_one', syn_spec={'weight': -1000.0, "delay": 1.0, 'model': 'dopsyn_r'}) #-100.0
# Value function V(t+1)
nest.Connect(D1_left, SNc_left, 'one_to_one', syn_spec={'weight': Gama*1000.0, "delay": 2.0, 'model': 'dopsyn_l'}) #100.0
nest.Connect(D1_right, SNc_right, 'one_to_one', syn_spec={'weight': Gama*1000.0, "delay": 2.0, 'model': 'dopsyn_r'}) #100.0

wrand=-100.0*random()
nest.Connect(D1_left, SNr_left, 'one_to_one',syn_spec={'weight': wrand, "delay": 1.0, 'model': 'dopsyn_l'})
nest.Connect(D1_right, SNr_right, 'one_to_one',syn_spec={'weight': wrand, "delay": 1.0, 'model': 'dopsyn_r'})

# recurrent SNc ---
'''nest.Connect(SNc_left, SNc_left, 'one_to_one',syn_spec={'weight': -100.0, "delay": 1.0})
nest.Connect(SNc_right, SNc_right, 'one_to_one',syn_spec={'weight': -100.0, "delay": 1.0})'''

nest.Connect(GPe_left, STN_left, 'one_to_one', syn_spec={'weight': -100.0, "delay": 1.0}) #100.0!
nest.Connect(GPe_right, STN_right, 'one_to_one', syn_spec={'weight': -100.0, "delay": 1.0}) #100.0!

nest.Connect(STN_left, GPe_left, 'one_to_one', syn_spec={'weight': 100.0, "delay": 1.0}) #100.0!
nest.Connect(STN_right, GPe_right, 'one_to_one', syn_spec={'weight': 100.0, "delay": 1.0}) #100.0!

nest.Connect(GPe_left, GPe_left, syn_spec={'weight': WGPe_GPe_l, "delay": 1.0})
nest.Connect(GPe_right, GPe_right, syn_spec={'weight': WGPe_GPe_r, "delay": 1.0})

nest.Connect(GPe_left, GPe_right, 'one_to_one',syn_spec={'weight': -5000.0, "delay": 1.0}) #-2000.0
nest.Connect(GPe_right, GPe_left, 'one_to_one',syn_spec={'weight': -5000.0, "delay": 1.0}) #-2000.0

##switched left-right!!!
'''
nest.Connect(STN_left, SNr_right, syn_spec={'weight': WSTNl_SNrr, "delay": 1.0, 'model': 'dopsyn_l'}) ###
nest.Connect(STN_right, SNr_left, syn_spec={'weight': WSTNr_SNrl, "delay": 1.0, 'model': 'dopsyn_r'}) ###
'''
nest.Connect(STN_left, SNr_right, 'all_to_all', syn_spec={'weight': 100.0, "delay": 1.0, 'model': 'dopsyn_l'})
nest.Connect(STN_right, SNr_left, 'all_to_all', syn_spec={'weight': 100.0, "delay": 1.0, 'model': 'dopsyn_r'})

g_SNr_SC=1.0
nest.Connect(SNr_left, SC_left, syn_spec={'weight': np.array(WSNr_SC_l)*g_SNr_SC, "delay": 1.0})
nest.Connect(SNr_right, SC_right, syn_spec={'weight': np.array(WSNr_SC_r)*g_SNr_SC, "delay": 1.0})

g_LIP_SC=0.25
nest.Connect(LIP_left, SC_left, syn_spec={'weight': np.array(WLIP_SC_l)*g_LIP_SC, "delay": 1.0})
nest.Connect(LIP_right, SC_right, syn_spec={'weight': np.array(WLIP_SC_r)*g_LIP_SC, "delay": 1.0})

w_SC_D=100.0
nest.Connect(SC_left, D1_left, 'all_to_all', syn_spec={'weight': w_SC_D, "delay": 1.0})
nest.Connect(SC_right, D1_right, 'all_to_all', syn_spec={'weight': w_SC_D, "delay": 1.0})
nest.Connect(SC_left, D2_left, 'all_to_all', syn_spec={'weight': w_SC_D, "delay": 1.0})
nest.Connect(SC_right, D2_right, 'all_to_all', syn_spec={'weight': w_SC_D, "delay": 1.0})

w_SC_lr=-100.0
nest.Connect(SC_left, SC_right, 'all_to_all', syn_spec={'weight': w_SC_lr, "delay": 1.0})
nest.Connect(SC_right, SC_left, 'all_to_all', syn_spec={'weight': w_SC_lr, "delay": 1.0})

gSC_LIP=0.2
nest.Connect(SC_left, LIP_left, syn_spec={'weight': np.transpose(WLIP_SC_l)*gSC_LIP, "delay": 1.0}) ###
nest.Connect(SC_right, LIP_right, syn_spec={'weight': np.transpose(WLIP_SC_r)*gSC_LIP, "delay": 1.0}) ###

spd_SCl= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})
spd_SCr= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})

spd_D1l= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})
spd_D1r= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})

spd_D2l= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})
spd_D2r= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})

spd_GPel= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})
spd_GPer= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})

spd_STNl= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})
spd_STNr= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})

spd_SNrl= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})
spd_SNrr= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})

spd_SNcl= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})
spd_SNcr= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})

nest.Connect(SC_left,spd_SCl)
nest.Connect(SC_right,spd_SCr)

nest.Connect(D1_left,spd_D1l)
nest.Connect(D1_right,spd_D1r)

nest.Connect(D2_left,spd_D2l)
nest.Connect(D2_right,spd_D2r)

nest.Connect(GPe_left,spd_GPel)
nest.Connect(GPe_right,spd_GPer)

nest.Connect(STN_left,spd_STNl)
nest.Connect(STN_right,spd_STNr)

nest.Connect(SNr_left,spd_SNrl)
nest.Connect(SNr_right,spd_SNrr)

nest.Connect(SNc_left,spd_SNcl)
nest.Connect(SNc_right,spd_SNcr)

### end BG

ampl_values_I_left = []
ampl_values_I_right = []
ampl_times = np.arange(0.1, sim_time, 0.1)
ampl_values_zero=[]
i_zero=0.0

A = 200.0
k = 0.02
for t in np.arange(0.1, sim_time, 0.1):
	I_left =  A*(1/(1+np.exp(-k*t)))
	ampl_values_I_left.append(I_left)

	I_right = (A/2)*(1/(1+np.exp(k*t)))
	ampl_values_I_right.append(I_right)

	ampl_values_zero.append(i_zero)

ampl_values_I_left = np.array(ampl_values_I_left)
ampl_values_I_right = np.array(ampl_values_I_right)
ampl_values_zero = np.array(ampl_values_zero)

cg_left = nest.Create("step_current_generator",1, params={"amplitude_values": ampl_values_I_left , "amplitude_times": ampl_times})#, "start": 0.0, "stop": sim_time})
cg_right = nest.Create("step_current_generator",1, params={"amplitude_values": ampl_values_I_right, "amplitude_times": ampl_times})#, "start": 0.0, "stop": sim_time})

ri_left = nest.Create("iaf_psc_exp", 1, bg_params)
ri_right = nest.Create("iaf_psc_exp", 1, bg_params)

r_left = nest.Create("iaf_psc_exp", 1, bg_params)
r_right = nest.Create("iaf_psc_exp", 1, bg_params)

nest.SetStatus(ri_left, {'I_e': 0.0})
nest.SetStatus(ri_right, {'I_e': 0.0})
nest.SetStatus(r_left, {'I_e': 0.0})
nest.SetStatus(r_right, {'I_e': 0.0})
'''
nest.Connect(cg_left, [ri_left[0]]) 
nest.Connect(cg_right, [ri_right[0]])'''
'''
nest.Connect(ri_left, r_left, 'all_to_all', syn_spec={'weight': 10000.0, "delay": 1.0})
nest.Connect(ri_right, r_right, 'all_to_all', syn_spec={'weight': 10000.0, "delay": 1.0})'''

nest.Connect(SC_left, ri_left, 'all_to_all', syn_spec={'weight': 1.0}) #, "delay": 1.0})
nest.Connect(SC_right, ri_right, 'all_to_all', syn_spec={'weight': 1.0}) #, "delay": 1.0})
'''
nest.Connect([ri_left[0]], r_left, syn_spec={'weight': -1.0, "delay": 1.0})
nest.Connect([ri_right[0]], r_right, syn_spec={'weight': -1.0, "delay": 1.0})'''
####!
nest.Connect(cg_left, r_left) #, syn_spec={'weight': 100.0, "delay": 1.0}
nest.Connect(cg_right, r_right) #, syn_spec={'weight': 100.0, "delay": 1.0}

nest.Connect(ri_left, r_left, syn_spec={'weight': -100.0}) #, "delay": 1.0})
nest.Connect(ri_right, r_right, syn_spec={'weight': -100.}) #, "delay": 1.0})

'''nest.Connect(cg_left, LIP_left)
nest.Connect(cg_right, LIP_right)'''

spd_rl= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})
spd_rr= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})

spd_ril= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})
spd_rir= nest.Create("spike_detector", params = {"to_memory": True, "to_file": False, "label": "spdALL", "withtime": True, "withgid": True}) #, "precise_times": True})

nest.Connect(r_left,spd_rl)
nest.Connect(r_right,spd_rr)

nest.Connect(ri_left,spd_ril)
nest.Connect(ri_right,spd_rir)
# ! rf
#nest.Connect(r_left, SNc_left, 'all_to_all', syn_spec={'weight': -100.0, "delay": 0.1})
#nest.Connect(r_right, SNc_right, 'all_to_all', syn_spec={'weight': -100.0, "delay": 0.1})
'''
nest.Connect(SC_left, SNc_left, 'all_to_all', syn_spec={'weight': -1.0, "delay": 1.0})
nest.Connect(SC_right, SNc_right, 'all_to_all', syn_spec={'weight': -1.0, "delay": 1.0})'''

nest.Simulate(sim_time)

spikes_rl = nest.GetStatus(spd_rl, "events")[0]
if len(spikes_rl["times"])>1:
	nest.raster_plot.from_device(spd_rl, hist = True, title = 'left reinforcement signal')
	plt.savefig(charts_folder+"r_left"+".png")
	np.savetxt(charts_folder+'r_left.txt',(spikes_rl["times"],spikes_rl["senders"]),fmt='%f')
spikes_rr = nest.GetStatus(spd_rr, "events")[0]
if len(spikes_rr["times"])>1:
	nest.raster_plot.from_device(spd_rr, hist = True, title = 'right reinforcement signal')
	plt.savefig(charts_folder+"r_right"+".png")
	np.savetxt(charts_folder+'r_right.txt',(spikes_rr["times"],spikes_rr["senders"]),fmt='%f')
'''
spikes_ril = nest.GetStatus(spd_ril, "events")[0]
if len(spikes_ril["times"])>1:
	nest.raster_plot.from_device(spd_ril, hist = True, title = 'left reinforcement signal')
	plt.savefig(charts_folder+"ri_left"+".png")
	np.savetxt(charts_folder+'ri_left.txt',(spikes_ril["times"],spikes_ril["senders"]),fmt='%f')
spikes_rir = nest.GetStatus(spd_rir, "events")[0]
if len(spikes_rir["times"])>1:
	nest.raster_plot.from_device(spd_rir, hist = True, title = 'right reinforcement signal')
	plt.savefig(charts_folder+"ri_right"+".png")
	np.savetxt(charts_folder+'ri_right.txt',(spikes_rir["times"],spikes_rir["senders"]),fmt='%f')'''
#plt.show()

print "V1 orientations numbers: ", n_spd
print "MT orientations numbers: ", n_spd_MT
print "MSTc orientations numbers: ", n_spd_MST_fpc
print "MSTe orientations numbers: ", n_spd_MST_fpe

#nest.raster_plot.from_device(spdALL, hist = True, title = 'All V1 neurons spikes')
#plt.show()
colors_MT=['r', 'g', 'c', 'm', 'k', 'y', 'm', 'c', 'g', 'r']
colors=['r', 'g', 'c', 'm', 'r']
colors_MST=['k', 'g', 'c', 'b', 'm', 'r', 'y', 'k', 'g', 'c', ]
#spikes = nest.GetStatus(spd_V1, "events")[0]
#ts=spikes["times"]
#ns=spikes["senders"]
#if len(spikes["times"])>0:
	#plt.figure('All spikes')
	#plt.plot(spikes["times"],spikes["senders"],'.')
	#plt.show()

'''spikes_ON1 = nest.GetStatus(spdON1, "events")[0]
if len(spikes_ON1["times"])>0:
	plt.figure('ON1 spikes')
	plt.plot(spikes_ON1["times"],spikes_ON1["senders"],'r.')
	plt.show()

spikes_OFF2 = nest.GetStatus(spdOFF2, "events")[0]
if len(spikes_OFF2["times"])>0:
	plt.figure('OFF2 spikes')
	plt.plot(spikes_OFF2["times"],spikes_OFF2["senders"],'b.')
	plt.show()

spikes_ON2 = nest.GetStatus(spdON2, "events")[0]
if len(spikes_ON2["times"])>0:
	plt.figure('ON2 spikes')
	plt.plot(spikes_ON2["times"],spikes_ON2["senders"],'r.')
	plt.show()

spikes_OFF1 = nest.GetStatus(spdOFF1, "events")[0]
if len(spikes_OFF1["times"])>0:
	plt.figure('OFF1 spikes')
	plt.plot(spikes_OFF1["times"],spikes_OFF1["senders"],'b.')
	plt.show()'''

'''
#extract spikes per time
N=len(orientations)
ctx_x=np.concatenate((e_xy1[:,0],e_xy2[:,0],i_xy1[:,0],i_xy2[:,0]))
ctx_y=np.concatenate((e_xy1[:,1],e_xy2[:,1],i_xy1[:,1],i_xy2[:,1]))
all_o=np. concatenate((thetae1,thetae2,thetai1,thetai2))
z=np.zeros((len(all_o))).astype(int)
for i in np.arange(0, len(all_o), 1):
	for j in np.arange(0, N, 1):
		if all_o[i]==orientations[j]:
			z[i]=j
			
moments_of_spike=[] 
list_of_neurons=[]
#list_of_orientations=[]

ns_temp=[]
i=0
while (i < len(ts)-2):	
	ns_temp.append(ns[i])
	#z_temp.append(z[i])
	moments_of_spike.append(ts[i])
	while (ts[i]==ts[i+1] and i<len(ts)-2):	
		ns_temp.append(ns[i+1])
		#z_temp.append(z[i+1])
		i+=1
	list_of_neurons.append(ns_temp)
	#list_of_orientations.append(z_temp)
	#z_temp=[]
	ns_temp=[]
	i+=1

#extract spieks coordinates
x_spikes=[]
y_spikes=[]
z_spikes=[]
x_tmp=[]
y_tmp=[]
z_tmp=[]
spikes_per_orientation=np.zeros((len(moments_of_spike),len(orientations)))
for i in np.arange(0,len(moments_of_spike),1):
	for j in np.arange(0,len(list_of_neurons[i]),1):
		if (list_of_neurons[i][j]<Ne+n_LGN):  # 400
			x_tmp.append(e_xy1[list_of_neurons[i][j]-1-n_LGN,0])
			y_tmp.append(e_xy1[list_of_neurons[i][j]-1-n_LGN,1])
			z_tmp.append(z[list_of_neurons[i][j]-1-n_LGN])
							
		elif list_of_neurons[i][j]>=(Ne+n_LGN+1) and list_of_neurons[i][j]<=(2*Ne+n_LGN):  # 401  800
			x_tmp.append(e_xy2[list_of_neurons[i][j]-Ne-n_LGN-1,0])
			y_tmp.append(e_xy2[list_of_neurons[i][j]-Ne-n_LGN-1,1])
			z_tmp.append(z[list_of_neurons[i][j]-1-n_LGN])
			
		elif list_of_neurons[i][j]>=(2*Ne+n_LGN+1) and list_of_neurons[i][j]<=(2*Ne+Ni+n_LGN):  #801  900
			x_tmp.append(i_xy1[list_of_neurons[i][j]-(2*Ne+n_LGN)-1,0])
			y_tmp.append(i_xy1[list_of_neurons[i][j]-(2*Ne+n_LGN)-1,1])
			z_tmp.append(z[list_of_neurons[i][j]-1-n_LGN])
			
		elif list_of_neurons[i][j]>(2*Ne+Ni+n_LGN):
			x_tmp.append(i_xy2[list_of_neurons[i][j]-(2*Ne+Ni+n_LGN)-1,0])
			y_tmp.append(i_xy2[list_of_neurons[i][j]-(2*Ne+Ni+n_LGN)-1,1])
			z_tmp.append(z[list_of_neurons[i][j]-1-n_LGN])
		
	for k in np.arange(0,len(z_tmp),1):
		for l in np.arange(0,len(orientations),1):
			if (z_tmp[k]==l):
				spikes_per_orientation[i,l]+=1	
	x_spikes.append(x_tmp)
	y_spikes.append(y_tmp)
	z_spikes.append(z_tmp)
	x_tmp=[]
	y_tmp=[]
	z_tmp=[]

legend_marks=["orientation=-90 deg", "orientation=-45 deg", "orientation=0 deg", "orientation=+45 deg", "orientation=+90 deg"]
legend_handle_m90=mlines.Line2D([],[],color=colors[0],marker='.',label=legend_marks[0])
legend_handle_m45=mlines.Line2D([],[],color=colors[1],marker='.',label=legend_marks[1])
legend_handle_0=mlines.Line2D([],[],color=colors[2],marker='.',label=legend_marks[2])
legend_handle_p45=mlines.Line2D([],[],color=colors[3],marker='.',label=legend_marks[3])
legend_handle_p90=mlines.Line2D([],[],color=colors[4],marker='.',label=legend_marks[4])

spikes_frequencies=np.zeros((len(moments_of_spike),len(orientations)))
max_spikes_frequencies=np.zeros((len(moments_of_spike)))
orient_max_spikes_frequencies=np.zeros((len(moments_of_spike))).astype(int)
for i in np.arange(0,N,1):
	spikes_frequencies[:,i]=spikes_per_orientation[:,i]/n_spd[i]
for i in np.arange(0,len(moments_of_spike),1):
	max_spikes_frequencies[i]=np.max(spikes_frequencies[i,:])
	orient_max_spikes_frequencies[i]=np.argmax(spikes_frequencies[i,:])

plt.figure('Maximum spikes frequencies')
#plt.hold('on')
for i in np.arange(0,len(moments_of_spike),1):
	plt.plot(moments_of_spike[i],max_spikes_frequencies[i]*100,'.',color=colors[orient_max_spikes_frequencies[i]],label=legend_marks[orient_max_spikes_frequencies[i]])
plt.legend(handles=[legend_handle_m90,legend_handle_m45,legend_handle_0,legend_handle_p45,legend_handle_p90])
plt.title('Maximal percent of spiking neurons',horizontalalignment='center', fontsize=16, color='b')
plt.xlabel('Simulation time in miliseconds')
plt.ylabel('%')
#plt.savefig(data_folder+'Maximum spikes frequencies_scale100_01-l'+str(lamb)+'_w'+str(ww)+'-s'+str(sigma)+'_we0.1_se0.1_wi0.1_si0.1.png')
plt.show()

plt.figure('orientation frequencies')
for i in np.arange(0,N,1):
	plt.subplot(N,1,i+1)
	plt.plot(moments_of_spike,spikes_frequencies[:,i]*100,'.',color=colors[i]) #,label=legend_marks[i])
	if i==0:
		plt.title('Percent of spiking neurons per orientation',horizontalalignment='center', fontsize=16, color='b')
	plt.ylim([0.0,np.max(max_spikes_frequencies)*100])	
	plt.ylabel('%')
	plt.legend([legend_marks[i]]) #,bbox_to_anchor=(0.75,0.55)),
plt.xlabel('Simulation time in miliseconds')
#plt.savefig(data_folder+'orientation frequencies_scale100_01-l'+str(lamb)+'_w'+str(ww)+'-s'+str(sigma)+'_we0.1_se0.1_wi0.1_si0.1.png')
plt.show()
'''

total_firing_rate_per_orientation=np.zeros((len(orientations)))
to=np.zeros((len(orientations),1))
nsto=np.zeros((len(orientations),1))
num_events=np.zeros((len(orientations)))

total_firing_rate_per_orientation_MT=np.zeros((len(MT_orientations)))
to_MT=np.zeros((len(MT_orientations),1))
nsto_MT=np.zeros((len(MT_orientations),1))
num_events_MT=np.zeros((len(MT_orientations)))

total_firing_rate_per_fpc_MST=np.zeros((len(focal_points)))
to_MSTc=np.zeros((len(focal_points),1))
nsto_MSTc=np.zeros((len(focal_points),1))
num_events_MSTc=np.zeros((len(focal_points)))

total_firing_rate_per_fpe_MST=np.zeros((len(focal_points)))
to_MSTe=np.zeros((len(focal_points),1))
nsto_MSTe=np.zeros((len(focal_points),1))
num_events_MSTe=np.zeros((len(focal_points)))

#print file_name, ' case ', case_name
#tt=input("time of simulation: ")
#plt.figure('all')
#plt.hold('on')
#spikes=[]

'''f = open(results_folder+'Spike_trains_fbp2.cpickle', "wb")
cPickle.dump(spikes, f, protocol=2)
cPickle.dump(spikes_ON1, f, protocol=2)
cPickle.dump(spikes_ON2, f, protocol=2)
cPickle.dump(spikes_OFF1, f, protocol=2)
cPickle.dump(spikes_OFF2, f, protocol=2)'''
'''
### V1
#plt.figure('Orientation Spikes')
nest.raster_plot.from_device(spd_V1, hist = True, title = 'All V1 neurons spikes')
plt.figure('All V1 neurons spikes')
plt.xlim([0.0,sim_time])
plt.ylim([1,2*(Ne+Ni)])
plt.hold('on')
for i in np.arange(0,len(orientations),1):
	spikes_or = nest.GetStatus([spd_orientations[i]], "events")[0]
	#ts=spikes["times"]
	#ns=spikes["senders"]
	#cPickle.dump(spikes_or, f, protocol=2)
	#np.savetxt(fr_data_folder+'V1_spikes_for_'+str(orientations[i])+'.txt',(spikes_or['times'],spikes_or['senders']),fmt='%f')
	num_events[i] = len(spikes_or["times"])
	total_firing_rate_per_orientation[i]=num_events[i]/n_spd[i]
	#print 'orientation ', orientations[i] , ' frequency :', total_firing_rate_per_orientation[i]
	print 'orientation: ', orientations[i] , ' n spikes: ', num_events[i], ' n neurons: ', n_spd[i]
	#print 'ts: ', ts
        if len(spikes_or["times"])>0:        
		#nest.raster_plot.from_device([spd_orientations[i]], hist = True, title = 'orientation' + str(orientations[i]))
		#plt.figure(i)
		plt.plot(spikes_or["times"],spikes_or["senders"]-np.min(spikes_or["senders"])+1,'.',color=colors[i])
		#plt.show()
	''''''to[i,0]=ts[0]
	nsto[i,0]=ns[0]
	for j in np.arange(1,num_events[i],1):
		if ts[j]>ts[j-1]:
			to[i,0].append(ts[j])
		else:
			nsto[i].append(ns[j])''''''			
	#spikes=[]		
	#ts=[]
	#ns=[]
#plt.savefig("zero_in_V1_all_new_1"+".png")
#plt.savefig(data_folder+'All V1 neurons spikes_scale100_01-l'+str(lamb)+'_w'+str(ww)+'-s'+str(sigma)+'_we0.1_se0.1_wi0.1_si0.1.png')
#plt.show()
#f.close()
#np.savetxt(results_folder+'Firig_orientations_stats_fbp2.txt',(total_firing_rate_per_orientation,num_events,n_spd),fmt='%f')

plt.savefig(charts_folder+"LIP_left"+".png")
np.savetxt(charts_folder+'LIP_left.txt',(LIPl_spikes["times"],LIPl_spikes["senders"]),fmt='%f')


###MT
nest.raster_plot.from_device(spd_MT, hist = True, title = 'All MT neurons spikes')
plt.figure('All MT neurons spikes')
plt.xlim([0.0,sim_time])
plt.ylim([1,2*(MT_Ne+MT_Ni)])
plt.hold('on')
for i in np.arange(0,len(MT_orientations),1):
	spikes_or_MT = nest.GetStatus([spd_orientations_MT[i]], "events")[0]
	#ts=spikes["times"]
	#ns=spikes["senders"]
	#cPickle.dump(spikes_or, f, protocol=2)
	#np.savetxt(fr_data_folder+'MT_spikes_for_'+str(MT_orientations[i])+'.txt',(spikes_or_MT['times'],spikes_or_MT['senders']),fmt='%f')
	num_events_MT[i] = len(spikes_or_MT["times"])
	total_firing_rate_per_orientation_MT[i]=num_events_MT[i]/n_spd_MT[i]
	#print 'orientation MT ', MT_orientations[i] , ' frequency :', total_firing_rate_per_orientation_MT[i]
	print 'orientation MT: ', MT_orientations[i] , 'n spikes:', num_events_MT[i], ' n neurons: ',n_spd_MT[i]
	#print 'ts: ', ts
        if len(spikes_or_MT["times"])>0:        
		#nest.raster_plot.from_device([spd_orientations[i]], hist = True, title = 'orientation' + str(orientations[i]))
		#plt.figure(i)
		plt.plot(spikes_or_MT["times"],spikes_or_MT["senders"]-np.min(spikes_or_MT["senders"])+1,'.',color=colors_MT[i])
		#plt.show()
	''''''to[i,0]=ts[0]
	nsto[i,0]=ns[0]
	for j in np.arange(1,num_events[i],1):
		if ts[j]>ts[j-1]:
			to[i,0].append(ts[j])
		else:
			nsto[i].append(ns[j])''''''			
	#spikes=[]		
	#ts=[]
	#ns=[]
#plt.savefig("zero_in_V1_all_new_1"+".png")
#plt.savefig(data_folder+'All V1 neurons spikes_scale100_01-l'+str(lamb)+'_w'+str(ww)+'-s'+str(sigma)+'_we0.1_se0.1_wi0.1_si0.1.png')
#plt.show()
#f.close()
#np.savetxt(results_folder+'Firig_orientations_stats_fbp2.txt',(total_firing_rate_per_orientation,num_events,n_spd),fmt='%f')

###MSTc
MSTc_spikes=nest.GetStatus(spd_MSTc, "events")[0]
if len(MSTc_spikes["times"])>0:
	plt.figure('All MSTd contraction neurons spikes')
	plt.xlim([0.0,sim_time])
	plt.ylim([1,MSTd_Nc])
	plt.hold('on')
	#nest.raster_plot.from_device(spd_MSTc, hist = True, title = 'All MSTd contraction neurons spikes')
	for i in np.arange(0,len(focal_points),1):
		spikes_fpc_MST = nest.GetStatus([spd_fpc_MST[i]], "events")[0]
		#ts=spikes["times"]
		#ns=spikes["senders"]
		#cPickle.dump(spikes_or, f, protocol=2)
		#np.savetxt(fr_data_folder+'MT_spikes_for_'+str(MT_orientations[i])+'.txt',(spikes_or_MT['times'],spikes_or_MT['senders']),fmt='%f')
		num_events_MSTc[i] = len(spikes_fpc_MST["times"])
		total_firing_rate_per_fpc_MST[i]=num_events_MSTc[i]/n_spd_MST_fpc[i]
		#print 'orientation MT ', MT_orientations[i] , ' frequency :', total_firing_rate_per_orientation_MT[i]
		print 'focal point MSTc: ', focal_points[i] , 'n spikes:', num_events_MSTc[i], ' n neurons: ',n_spd_MST_fpc[i]
		#print 'ts: ', ts
	        if len(spikes_fpc_MST["times"])>0:        
			#nest.raster_plot.from_device([spd_orientations[i]], hist = True, title = 'orientation' + str(orientations[i]))
			#plt.figure(i)
			plt.plot(spikes_fpc_MST["times"],spikes_fpc_MST["senders"]-np.min(spikes_fpc_MST["senders"])+1,'.',color=colors_MST[i])
			#plt.show()
		''''''to[i,0]=ts[0]
		nsto[i,0]=ns[0]
		for j in np.arange(1,num_events[i],1):
			if ts[j]>ts[j-1]:
				to[i,0].append(ts[j])
			else:
				nsto[i].append(ns[j])''''''			
		#spikes=[]		
		#ts=[]
		#ns=[]
	#plt.savefig("zero_in_V1_all_new_1"+".png")
	#plt.savefig(data_folder+'All V1 neurons spikes_scale100_01-l'+str(lamb)+'_w'+str(ww)+'-s'+str	(sigma)+'_we0.1_se0.1_wi0.1_si0.1.png')
	#plt.show()
	#f.close()
	#np.savetxt(results_folder+'Firig_orientations_stats_fbp2.txt',(total_firing_rate_per_orientation,num_events,n_spd),fmt='%f')

###MSTe
MSTe_spikes=nest.GetStatus(spd_MSTe, "events")[0]
if len(MSTe_spikes["times"])>0:
	#nest.raster_plot.from_device(spd_MSTe, hist = True, title = 'All MSTd expansion neurons spikes')
	plt.figure('All MSTd expansion neurons spikes')
	plt.xlim([0.0,sim_time])
	plt.ylim([1,MSTd_Ne])
	plt.hold('on')
	for i in np.arange(0,len(focal_points),1):
		spikes_fpe_MST = nest.GetStatus([spd_fpe_MST[i]], "events")[0]
		#ts=spikes["times"]
		#ns=spikes["senders"]
		#cPickle.dump(spikes_or, f, protocol=2)
		#np.savetxt(fr_data_folder+'MT_spikes_for_'+str(MT_orientations[i])+'.txt',(spikes_or_MT['times'],spikes_or_MT['senders']),fmt='%f')
		num_events_MSTe[i] = len(spikes_fpe_MST["times"])
		total_firing_rate_per_fpe_MST[i]=num_events_MSTe[i]/n_spd_MST_fpe[i]
		#print 'orientation MT ', MT_orientations[i] , ' frequency :', total_firing_rate_per_orientation_MT[i]
		print 'focal point MSTe: ', focal_points[i] , 'n spikes:', num_events_MSTe[i], ' n neurons: ',n_spd_MST_fpe[i]
		#print 'ts: ', ts
        	if len(spikes_fpe_MST["times"])>0:        
			#nest.raster_plot.from_device([spd_orientations[i]], hist = True, title = 'orientation' + str(orientations[i]))
			#plt.figure(i)
			plt.plot(spikes_fpe_MST["times"],spikes_fpe_MST["senders"]-np.min(spikes_fpe_MST["senders"])+1,'.',color=colors_MST[i])
			#plt.show()
		''''''to[i,0]=ts[0]
		nsto[i,0]=ns[0]
		for j in np.arange(1,num_events[i],1):
			if ts[j]>ts[j-1]:
				to[i,0].append(ts[j])
			else:
				nsto[i].append(ns[j])''''''			
		#spikes=[]		
		#ts=[]
		#ns=[]
	#plt.savefig("zero_in_V1_all_new_1"+".png")
	#plt.savefig(data_folder+'All V1 neurons spikes_scale100_01-l'+str(lamb)+'_w'+str(ww)+'-s'+str(sigma)+'_we0.1_se0.1_wi0.1_si0.1.png')

	#f.close()
	#np.savetxt(results_folder+'Firig_orientations_stats_fbp2.txt',(total_firing_rate_per_orientation,num_events,n_spd),fmt='%f')
plt.show()'''

#V1
V1_spikes = nest.GetStatus(spd_V1, "events")[0]
if len(V1_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_V1, hist = True, title = 'V1 neurons spikes')
	plt.savefig(charts_folder+"V1_neurons_spikes"+".png")
	np.savetxt(charts_folder+'V1_neurons_spikes.txt',(V1_spikes["times"],V1_spikes["senders"]),fmt='%f')


#MT
MT_spikes = nest.GetStatus(spd_MT, "events")[0]
if len(MT_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_MT, hist = True, title = 'MT neurons spikes')
	plt.savefig(charts_folder+"MT_neurons_spikes"+".png")
	np.savetxt(charts_folder+'MT_neurons_spikes.txt',(MT_spikes["times"],MT_spikes["senders"]),fmt='%f')


#MSTc
MSTc_spikes=nest.GetStatus(spd_MSTc, "events")[0]
if len(MSTc_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_MSTc, hist = True, title = 'MSTd contraction neurons spikes')
	plt.savefig(charts_folder+"MSTc_neurons_spikes"+".png")
	np.savetxt(charts_folder+'MSTc_neurons_spikes.txt',(MSTc_spikes["times"],MSTc_spikes["senders"]),fmt='%f')


#MSTe
MSTe_spikes=nest.GetStatus(spd_MSTe, "events")[0]
if len(MSTe_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_MSTe, hist = True, title = 'MSTd expansion neurons spikes')
	plt.savefig(charts_folder+"MSTe_neurons_spikes"+".png")
	np.savetxt(charts_folder+'MSTe_neurons_spikes.txt',(MSTe_spikes["times"],MSTe_spikes["senders"]),fmt='%f')

#plt.show()

#LIP
LIPl_spikes=nest.GetStatus(spd_LIPl, "events")[0]
nest.raster_plot.from_device(spd_LIPl, hist = True, title = 'LIP neurons left')
plt.savefig(charts_folder+"LIP_left"+".png")
np.savetxt(charts_folder+'LIP_left.txt',(LIPl_spikes["times"],LIPl_spikes["senders"]),fmt='%f')

LIPr_spikes=nest.GetStatus(spd_LIPr, "events")[0]
nest.raster_plot.from_device(spd_LIPr, hist = True, title = 'LIP neurons right')
plt.savefig(charts_folder+"LIP_right"+".png")
np.savetxt(charts_folder+'LIP_right.txt',(LIPr_spikes["times"],LIPr_spikes["senders"]),fmt='%f')

#SC
SCl_spikes=nest.GetStatus(spd_SCl, "events")[0]
if len(SCl_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_SCl, hist = True, title = 'SC neurons left')	
	plt.savefig(charts_folder+"SC_left"+".png")
	np.savetxt(charts_folder+'SC_left.txt',(SCl_spikes["times"],SCl_spikes["senders"]),fmt='%f')
SCr_spikes=nest.GetStatus(spd_SCr, "events")[0]
if len(SCr_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_SCr, hist = True, title = 'SC neurons right')
	plt.savefig(charts_folder+"SC_right"+".png")
	np.savetxt(charts_folder+'SC_right.txt',(SCr_spikes["times"],SCr_spikes["senders"]),fmt='%f')

#BG layers
D1l_spikes=nest.GetStatus(spd_D1l, "events")[0]
if len(D1l_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_D1l, hist = True, title = 'D1 neurons left')
	plt.savefig(charts_folder+"D1_left"+".png")
	np.savetxt(charts_folder+'D1_left.txt',(D1l_spikes["times"],D1l_spikes["senders"]),fmt='%f')
D1r_spikes=nest.GetStatus(spd_D1r, "events")[0]
if len(D1r_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_D1r, hist = True, title = 'D1 neurons right')
	plt.savefig(charts_folder+"D1_right"+".png")
	np.savetxt(charts_folder+'D1_right.txt',(D1r_spikes["times"],D1r_spikes["senders"]),fmt='%f')

D2l_spikes=nest.GetStatus(spd_D2l, "events")[0]
if len(D2l_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_D2l, hist = True, title = 'D2 neurons left')
	plt.savefig(charts_folder+"D2_left"+".png")
	np.savetxt(charts_folder+'D2_left.txt',(D2l_spikes["times"],D2l_spikes["senders"]),fmt='%f')
D2r_spikes=nest.GetStatus(spd_D2r, "events")[0]
if len(D2r_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_D2r, hist = True, title = 'D2 neurons right')
	plt.savefig(charts_folder+"D2_right"+".png")
	np.savetxt(charts_folder+'D2_right.txt',(D2r_spikes["times"],D2r_spikes["senders"]),fmt='%f')

GPel_spikes=nest.GetStatus(spd_GPel, "events")[0]
if len(GPel_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_GPel, hist = True, title = 'GPe neurons left')
	plt.savefig(charts_folder+"GPe_left"+".png")
	np.savetxt(charts_folder+'GPe_left.txt',(GPel_spikes["times"],GPel_spikes["senders"]),fmt='%f')
GPer_spikes=nest.GetStatus(spd_GPer, "events")[0]
if len(GPer_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_GPer, hist = True, title = 'GPe neurons right')
	plt.savefig(charts_folder+"GPe_right"+".png")
	np.savetxt(charts_folder+'GPe_right.txt',(GPer_spikes["times"],GPer_spikes["senders"]),fmt='%f')

STNl_spikes=nest.GetStatus(spd_STNl, "events")[0]
if len(STNl_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_STNl, hist = True, title = 'STN neurons left')
	plt.savefig(charts_folder+"STN_left"+".png")
	np.savetxt(charts_folder+'STN_left.txt',(STNl_spikes["times"],STNl_spikes["senders"]),fmt='%f')
STNr_spikes=nest.GetStatus(spd_STNr, "events")[0]
if len(STNr_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_STNr, hist = True, title = 'STN neurons right')
	plt.savefig(charts_folder+"STN_right"+".png")
	np.savetxt(charts_folder+'STN_right.txt',(STNr_spikes["times"],STNr_spikes["senders"]),fmt='%f')

SNrl_spikes=nest.GetStatus(spd_SNrl, "events")[0]
if len(SNrl_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_SNrl, hist = True, title = 'SNr neurons left')
	plt.savefig(charts_folder+"SNr_left"+".png")
	np.savetxt(charts_folder+'SNr_left.txt',(SNrl_spikes["times"],SNrl_spikes["senders"]),fmt='%f')
SNrr_spikes=nest.GetStatus(spd_SNrr, "events")[0]
if len(SNrr_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_SNrr, hist = True, title = 'SNr neurons right')
	plt.savefig(charts_folder+"SNr_right"+".png")
	np.savetxt(charts_folder+'SNr_right.txt',(SNrr_spikes["times"],SNrr_spikes["senders"]),fmt='%f')

SNcl_spikes=nest.GetStatus(spd_SNcl, "events")[0]
if len(SNcl_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_SNcl, hist = True, title = 'SNc neurons left')
	plt.savefig(charts_folder+"SNc_left"+".png")
	np.savetxt(charts_folder+'SNc_left.txt',(SNcl_spikes["times"],SNcl_spikes["senders"]),fmt='%f')
SNcr_spikes=nest.GetStatus(spd_SNcr, "events")[0]
if len(SNcr_spikes["senders"])>0:
	nest.raster_plot.from_device(spd_SNcr, hist = True, title = 'SNc neurons right')
	plt.savefig(charts_folder+"SNc_right"+".png")
	np.savetxt(charts_folder+'SNc_right.txt',(SNcr_spikes["times"],SNcr_spikes["senders"]),fmt='%f')
#plt.show()	

'''
LIPl_spikes=nest.GetStatus(spd_LIPl, "events")[0]
LIPr_spikes=nest.GetStatus(spd_LIPr, "events")[0]

LIPl_sp=np.zeros((len(LIPl_spikes['times'])))+0.95
LIPr_sp=np.zeros((len(LIPr_spikes['times'])))+1.05
plt.figure('LIP reactions')
plt.plot(LIPl_spikes['times'],LIPl_sp,'<b',markersize=25, markeredgewidth=0.0)
plt.xlim([0, sim_time])
plt.ylim([0.0, 2.0])
plt.hold('on')
plt.plot(LIPr_spikes['times'],LIPr_sp,'>r',markersize=25, markeredgewidth=0.0)
plt.xlabel('Time, ms')
plt.ylabel('LIP spikes')'''

'''
print 'LIP left spikes number = ', len(LIPl_spikes["times"]) #/len(LIPl_spikes["senders"])
print 'LIP right spikes number = ', len(LIPr_spikes["times"]) #/len(LIPr_spikes["senders"])
'''




