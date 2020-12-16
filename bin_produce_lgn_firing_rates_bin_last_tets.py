import numpy as np
import cPickle
from misc_functions import create_standar_indexes, create_extra_indexes
from stimuli_functions import sine_grating, ternary_noise 
from moving_shape_fun import moving_shape
from kernel_functions import create_kernel, temporal_kernel
from analysis_functions import produce_spikes, convolution
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.animation as animation
from plot_functions import visualize_firing_rate
from bin_dot_function import bin_dot
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() 
print 'size',size, rank ,"\n"

#f1=np.loadtxt("ah1_m1_eye.txt").astype(int)

# Time scales   
dt = 1
dt_kernel = 5.0  # ms
dt_stimuli = 33.4  # ms
kernel_duration = 150  # ms
kernel_size = int(kernel_duration / dt_kernel)

# Simulation time duration
T_simulation = 1670.0 #len(f1) # 1 * 10 ** 3.0  # ms
remove_start = int(kernel_size * dt_kernel)
#T_simulation += remove_start  # Add the size of the kernel
Nt_simulation = int(T_simulation / dt)  # Number of simulation points 
N_stimuli = int(T_simulation / dt_stimuli)  # Number of stimuli points 

## Space parameters
# visual space resolution and size  
dx = 1.0
dy = 1.0
lx = 900.0  # In degrees
ly = 900.0  # In degrees

LX=int(lx/8)
LY=int(ly/8)

# center-surround parameters 
factor = 1 # Controls the overall size of the center-surround pattern
sigma_center = 6.0 * factor  # Corresponds to 15'
sigma_surround = sigma_center*1.5 # 8.0 * factor  # Corresponds to 1 degree

# sine grating spatial parameters 
K = 0.8  # Cycles per degree 
Phi = 0 * np.pi
Theta = 0 * np.pi
max_contrast = 2.4 * 2
contrast = 0.5  # Percentage
A = contrast * max_contrast 
# Temporal frequency of sine grating 
w = 3  # Hz

# Set the random set for reproducibility
seed = 31255433
np.random.seed(seed)

screen_color=0.5

# Chose the particular cell
Nx = 20
Ny = 20
n=0

number = 11

numDataPerRank = 20/size

# Initialize the signal array to be filled 
firing_rate_ON1 = np.zeros((Nt_simulation, numDataPerRank*Nx))
firing_rate_OFF1 = np.zeros((Nt_simulation, numDataPerRank*Nx))
firing_rate_ON2 = np.zeros((Nt_simulation, numDataPerRank*Nx))
firing_rate_OFF2 = np.zeros((Nt_simulation, numDataPerRank*Nx))

kernelON1=np.zeros((kernel_size,int(2*LX/dx),int(2*LY/dy)))
kernelOFF1=np.zeros((kernel_size,int(2*LX/dx),int(2*LY/dy)))
kernelON2=np.zeros((kernel_size,int(2*LX/dx),int(2*LY/dy)))
kernelOFF2=np.zeros((kernel_size,int(2*LX/dx),int(2*LY/dy)))


if rank == 0:
	# Create indexes 
	signal_indexes, delay_indexes, stimuli_indexes = create_standar_indexes(dt, dt_kernel, dt_stimuli, kernel_size, Nt_simulation)
	working_indexes, kernel_times = create_extra_indexes(kernel_size, Nt_simulation)

	#read bin file
	f = open("mr1200.bin", "rb")
	data = np.fromfile(f, dtype=np.int16)
	data_color=data[10000:]
	data_xy=data[:10000]
	data_shape=np.reshape(data_xy,(2,5000))
	f.close()
	# Create the stimuli 
	#stimuli = sine_grating(dx, lx, dy, ly, A, K, Phi, Theta, dt_stimuli, N_stimuli, w)
	#stimuli = moving_dot(lx, ly, dx, dy, Nt_simulation, dt_stimuli, 2.0, -2.0, dd, rd, 0.0, screen_color)
	#stimuli_f1=np.zeros((T_simulation, 2*LX, 2*LY))

	stimuli = bin_dot(lx, ly, dx, dy, 50.0, 1.0, data_shape, data_color, screen_color)
	'''for i in np.arange(0,50,1):
		stimuli_f1[i, ...]=stimuli[i, lx/2-LX+f1[i,0]: lx/2+LX+f1[i,0] , ly/2-LY+f1[i,1] : ly/2+LY+f1[i,1]]
	'''
	
	f=open('20x20_LGN_positions_in_pixels.cpickle',"rb")
	xc_OFF1 =  cPickle.load(f)
	yc_OFF1 =  cPickle.load(f)
	xc_ON1 =  cPickle.load(f)
	yc_ON1 =  cPickle.load(f)
	xc_OFF2 =  cPickle.load(f)
	yc_OFF2 =  cPickle.load(f)
	xc_ON2 =  cPickle.load(f)
	yc_ON2 =  cPickle.load(f)
	f.close()

	print xc_OFF1
	print yc_OFF1
	print xc_ON1
	print yc_ON1

	print xc_OFF2
	print yc_OFF2
	print xc_ON2
	print yc_ON2

	print len(xc_OFF1), len(yc_OFF1)
	i=np.arange(0, len(xc_OFF1), 1).astype(int)
	j=np.arange(0, len(yc_OFF1), 1).astype(int)
	
else:
	signal_indexes =  np.empty(Nt_simulation-kernel_duration).astype(int)
	delay_indexes =  np.empty(kernel_size).astype(int)
	stimuli_indexes =  np.empty(Nt_simulation).astype(int)
	kernel_times =  np.empty(kernel_size).astype(int)
	working_indexes =  np.empty(Nt_simulation).astype(int)
	
	#data_shape = np.empty((2, 5000)).astype(int)
	stimuli = np.empty((50,900,900), dtype=float)
	#stimuli_v = np.empty((50,224,224), dtype=float)

	xc_OFF1 = np.empty(Nx).astype(float)
	yc_OFF1 = np.empty(Nx).astype(float)
	xc_ON1 = np.empty(Nx).astype(float)
	yc_ON1 = np.empty(Nx).astype(float)
	xc_OFF2 = np.empty(Nx).astype(float)
	yc_OFF2 = np.empty(Nx).astype(float)
	xc_ON2 = np.empty(Nx).astype(float)
	yc_ON2 = np.empty(Nx).astype(float)

	i=np.empty(Nx).astype(int)
	j=np.empty(Nx).astype(int)
	

recvbuf = np.empty(numDataPerRank).astype(int) # allocate space for recvbuf

comm.Scatter(j, recvbuf,root=0)

comm.Bcast(signal_indexes ,root=0)
comm.Bcast(delay_indexes ,root=0)
comm.Bcast(stimuli_indexes ,root=0)
comm.Bcast(kernel_times ,root=0)
#comm.Bcast(working_indexes ,root=0)

comm.Bcast(stimuli, root=0)

comm.Bcast(i ,root=0)
comm.Bcast(xc_OFF1 ,root=0)
comm.Bcast(yc_OFF1 ,root=0)
comm.Bcast(xc_ON1 ,root=0)
comm.Bcast(yc_ON1 ,root=0)
comm.Bcast(xc_OFF2 ,root=0)
comm.Bcast(yc_OFF2 ,root=0)
comm.Bcast(xc_ON2 ,root=0)
comm.Bcast(yc_ON2 ,root=0)

stimuli_v=stimuli[:, int(lx/2-LX): int(lx/2+LX) , int(ly/2-LY) : int(ly/2+LY)]



'''
xc_OFF1 = np.arange(-lx/4+sigma_surround/2, lx/4-sigma_surround/2+dx, (lx/2-(2*sigma_surround/2))/(Nx-1))
yc_OFF1 = np.arange(-ly/4+sigma_surround/2, ly/4-sigma_surround/2+dy, (ly/2-(2*sigma_surround/2))/(Ny-1))
xc_ON1 = xc_OFF1+(lx/2-(2*sigma_surround/2))/(2*(Nx-1)) #dx*2
yc_ON1 = yc_OFF1+(ly/2-(2*sigma_surround/2))/(2*(Ny-1)) #dy*2

xc_ON2 = np.arange(-lx/4+sigma_surround/2, lx/4-sigma_surround/2+dx, (lx/2-(2*sigma_surround/2))/(Nx-1))
yc_ON2 = np.arange(-ly/4+sigma_surround/2, ly/4-sigma_surround/2+dy, (ly/2-(2*sigma_surround/2))/(Ny-1))
xc_OFF2 = xc_ON2+(lx/2-(2*sigma_surround/2))/(2*(Nx-1)) #dx*2
yc_OFF2 = yc_ON2+(ly/2-(2*sigma_surround/2))/(2*(Ny-1)) #dy*2
'''
'''
f1=open(folder + str(Nx) + 'x' + str(Ny) + 'kernel_firing_rates_OFF1_' + str(number) + '.cpickle',"wab")
f2=open(folder + str(Nx) + 'x' + str(Ny) + 'kernel_firing_rates_ON1_' + str(number) + '.cpickle',"wab")
f3=open(folder + str(Nx) + 'x' + str(Ny) + 'kernel_firing_rates_OFF2_' + str(number) + '.cpickle',"wab")
f4=open(folder + str(Nx) + 'x' + str(Ny) + 'kernel_firing_rates_ON2_' + str(number) + '.cpickle',"wab")'''
'''
stim_lgn=np.zeros((len(stimuli[:,0,0]), LX, LY), dtype=float)
print len(stimuli[:,0,0])
stim_lgn+=stimuli[:, LX: 2*LX , LY : 2*LY]'''
'''
fig = plt.figure()
ax1=fig.add_subplot(1,1,1)
#ax2=fig.add_subplot(1,2,2)

def update(i):	
	#print lx/2-LX+f1[i,0], lx/2+LX+f1[i,0] , ly/2-LY+f1[i,1] , ly/2+LY+f1[i,1], f1[i,:] , i
	im1=ax1.imshow(stimuli[i,...], extent=[-lx/2, lx/2, ly/2, -ly/2], cmap='gray') 
	#plt.hold('on')
	#im=ax1.imshow(stimuli[i, lx/2-LX+f1[i,0]: lx/2+LX+f1[i,0] , ly/2-LY+f1[i,1] : ly/2+LY+f1[i,1]], cmap='gray', extent=[-LX+f1[i,0],LX+f1[i,0],LY+f1[i,1],-LY+f1[i,1]] )
	return im1
ani = animation.FuncAnimation(fig, update, frames=50, interval=1  )
#plt.show()
ani.save(folder + 'animation_1_' + str(Nx) + 'x' + str(Ny) + '_' + str(number) + '.gif', writer="imagemagick")		
'''
'''
ax1.imshow(stimuli[19, lx/2-LX: lx/2+LX , ly/2-LY : ly/2+LY], extent=[-LX,LX,LY,-LY])
ax2.imshow(stimuli[19,...], extent=[-lx/2, lx/2, ly/2, -ly/2]) 
plt.show()'''

#t = np.arange(0, kernel_size * dt_kernel, dt_kernel)
#f_t = temporal_kernel(t)

for xi in i:
	print "rank", rank, 'kernel at', xi
	for yj in recvbuf:
		

		# Create the kernel 
		kernelOFF1 = create_kernel(dx, 2*LX, dy, 2*LY, sigma_surround, sigma_center, 
			dt_kernel, kernel_size, inverse=-1, x_tra=xc_OFF1[xi], y_tra=yc_OFF1[yj])
		kernelON1 = create_kernel(dx, 2*LX, dy, 2*LY, sigma_surround, sigma_center, 
			dt_kernel, kernel_size, inverse=1, x_tra=xc_ON1[xi], y_tra=yc_ON1[yj])

		kernelOFF2 = create_kernel(dx, 2*LX, dy, 2*LY, sigma_surround, sigma_center, 
			dt_kernel, kernel_size, inverse=-1, x_tra=xc_OFF2[xi], y_tra=yc_OFF2[yj])
		kernelON2 = create_kernel(dx, 2*LX, dy, 2*LY, sigma_surround, sigma_center, 
			dt_kernel, kernel_size, inverse=1, x_tra=xc_ON2[xi], y_tra=yc_ON2[yj])
		
		'''plt.figure(5)
		plt.subplot(2,2,1)	
		plt.imshow(kernelON[n,0,...],extent=[-lx/2,lx/2,ly/2,-ly/2])
		plt.subplot(2,2,2)
		plt.imshow(stimuli[0,...],extent=[-lx/2,lx/2,ly/2,-ly/2])
		plt.subplot(2,2,3)
		plt.imshow(kernelOFF[n,0,...],extent=[-lx/2,lx/2,ly/2,-ly/2])
		plt.show()'''
		
		# Calculate the firing rate 
		for index in signal_indexes:
		    
		    firing_rate_ON1[index,n] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernelON1, stimuli_v)
		    firing_rate_OFF1[index,n] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernelOFF1, stimuli_v)

		    firing_rate_ON2[index,n] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernelON2, stimuli_v)
		    firing_rate_OFF2[index,n] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernelOFF2, stimuli_v)
		
		'''firing_rate_ON1[:, n] += 10 # Add background noise 
		# Rectify the firing rate
		firing_rate_ON1[firing_rate_ON1[:, n] < 0, n] = 0
		firing_rate_OFF1[:, n] += 15 # Add background noise 
		# Rectify the firing rate
		firing_rate_OFF1[firing_rate_OFF1[:, n] < 0, n] = 0

		firing_rate_ON2[:, n] += 10 # Add background noise 
		# Rectify the firing rate
		firing_rate_ON2[firing_rate_ON2[:, n] < 0, n] = 0
		firing_rate_OFF2[:, n] += 15 # Add background noise 
		# Rectify the firing rate
		firing_rate_OFF2[firing_rate_OFF2[:, n] < 0, n] = 0'''

		n+=1
comm.Barrier
firing_rate_ON1_new = comm.gather(firing_rate_ON1, root=0)
firing_rate_OFF1_new = comm.gather(firing_rate_OFF1, root=0)
firing_rate_ON2_new = comm.gather(firing_rate_ON2, root=0)
firing_rate_OFF2_new = comm.gather(firing_rate_OFF2, root=0)	
#plt.figure(1)
#visualize_firing_rate(firing_rate[signal_indexes], dt, T_simulation - remove_start, label='Firing rate')
#visualize_firing_rate(firing_rate / np.max(firing_rate), dt, T_simulation, label='Firing rate')

# Produce spikes with the signal
#spike_times_thin = produce_spikes(firing_rate, dt, T_simulation, remove_start)
#spike_times_thin -= remove_start 

#y = np.ones_like(spike_times_thin) * np.max(firing_rate)

# Scale firing rate
'''
aON1=np.min(firing_rate_ON1[signal_indexes, :])
bON1=np.max(firing_rate_ON1[signal_indexes, :])
scaled_firing_rate_ON1=np.zeros((Nt_simulation-remove_start, len(i)*len(j)))
scaled_firing_rate_ON1=firing_rate_ON1[signal_indexes,:]*100.0/(bON1-aON1)-100*aON1/(bON1-aON1)

aOFF1=np.min(firing_rate_OFF1[signal_indexes, :])
bOFF1=np.max(firing_rate_OFF1[signal_indexes, :])
scaled_firing_rate_OFF1=np.zeros((Nt_simulation-remove_start, len(i)*len(j)))
scaled_firing_rate_OFF1=firing_rate_OFF1[signal_indexes,:]*100.0/(bOFF1-aOFF1)-100*aOFF1/(bOFF1-aOFF1)

print 'minON1=', aON1, 'maxON1=', bON1
print 'minOFF1=', aOFF1, 'maxOFF1=', bOFF1

aON2=np.min(firing_rate_ON2[signal_indexes, :])
bON2=np.max(firing_rate_ON2[signal_indexes, :])
scaled_firing_rate_ON2=np.zeros((Nt_simulation-remove_start, len(i)*len(j)))
scaled_firing_rate_ON2=firing_rate_ON2[signal_indexes,:]*100.0/(bON2-aON2)-100*aON2/(bON2-aON2)

aOFF2=np.min(firing_rate_OFF2[signal_indexes, :])
bOFF2=np.max(firing_rate_OFF2[signal_indexes, :])
scaled_firing_rate_OFF2=np.zeros((Nt_simulation-remove_start, len(i)*len(j)))
scaled_firing_rate_OFF2=firing_rate_OFF2[signal_indexes,:]*100.0/(bOFF2-aOFF2)-100*aOFF2/(bOFF2-aOFF2)
'''

if rank==0:
	#Zapisva v papka, koqto e suzdadena predvaritelno
	folder = './test_20x20_lgn_fr_bin_150_5_eye_last_mr1200_1670/'

	f=open(folder + str(Nx) + 'x' + str(Ny) + '_firing_rates_ON_' + '.cpickle',"wb")
	cPickle.dump(firing_rate_ON1_new, f, protocol=2)
	cPickle.dump(firing_rate_ON2_new, f, protocol=2)
	f.close()

	f1=open(folder + str(Nx) + 'x' + str(Ny) + '_firing_rates_OFF_' + '.cpickle',"wb")
	cPickle.dump(firing_rate_OFF1_new, f1, protocol=2)
	cPickle.dump(firing_rate_OFF2_new, f1, protocol=2)
	f1.close()

'''
print 'minON2=', aON2, 'maxON2=', bON2
print 'minOFF2=', aOFF2, 'maxOFF2=', bOFF2

plt.figure(1)
for i in np.arange(0,Nx*Ny,1):
	plt.subplot(Nx,Ny,i+1)
	plt.plot(scaled_firing_rate_ON1[:,i])
	plt.ylim([0.0, 100.0])
plt.savefig(folder + 'fig ON1_' + str(Nx) + 'x' + str(Ny) + '_' + str(number) + '.png')		

plt.figure(2)
for i in np.arange(0,Nx*Ny,1):
	plt.subplot(Nx,Ny,i+1)
	plt.plot(scaled_firing_rate_OFF1[:,i])
	plt.ylim([0.0, 100.0])
plt.savefig(folder + 'fig OFF1_' + str(Nx) + 'x' + str(Ny) + '_' + str(number) + '.png')

plt.figure(3)
for i in np.arange(0,Nx*Ny,1):
	plt.subplot(Nx,Ny,i+1)
	plt.plot(scaled_firing_rate_ON2[:,i])
	plt.ylim([0.0, 100.0])
plt.savefig(folder + 'fig ON2_' + str(Nx) + 'x' + str(Ny) + '_' + str(number) + '.png')		

plt.figure(4)
for i in np.arange(0,Nx*Ny,1):
	plt.subplot(Nx,Ny,i+1)
	plt.plot(scaled_firing_rate_OFF2[:,i])
	plt.ylim([0.0, 100.0])
plt.savefig(folder + 'fig OFF2_' + str(Nx) + 'x' + str(Ny) + '_' + str(number) + '.png')

plt.figure(5)
plt.imshow(stimuli[0,...],extent=[-lx/2,lx/2,ly/2,-ly/2])
plt.savefig(folder + 'fig stimuli_1_' + str(Nx) + 'x' + str(Ny) + '_' + str(number) + '.png')		

plt.figure(6)
plt.imshow(stimuli[len(stimuli)-1, ...], extent=[-lx/2, lx/2, ly/2, -ly/2])
plt.savefig(folder + 'fig stimuli_end_' + str(Nx) + 'x' + str(Ny) + '_' + str(number) + '.png')

#plt.figure(4)
#plt.plot(firing_rate[remove_start:])
#plt.hold('on')
#plt.figure(5)
#plt.plot(spike_times_thin, y, '*', label='spikes')
#plt.legend()
#plt.ylim([9.5,9.8])
plt.show()
'''

