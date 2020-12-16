import numpy as np
from math import cos, sin


def bin_dot(lx, ly, dx, dy, screen_stimuli=50.0, dt_stimuli=1.0, array=[], dot_color=[], screen_color=0.5):

	Nx = int(lx / dx)
	Ny = int(ly / dy)
	x = np.arange(-lx/2, lx/2, dx)
	y = np.arange(-ly/2, ly/2, dy)
	X, Y = np.meshgrid(x, y)
	Z = np.zeros((Nx, Ny), dtype=float)
	Z += screen_color
	#t = np.arange(0, N_stimuli * dt_stimuli, dt_stimuli)
	stimuli = np.zeros((int(screen_stimuli*dt_stimuli*1), Nx, Ny), dtype=float)
	 

	stim_step = int(dt_stimuli*1)  
	stim_t = np.arange(0, int(screen_stimuli*dt_stimuli), stim_step).astype(int)

	'''fi = np.linspace(0, 2 * np.pi, 100)
	r = np.linspace(0, 3, 4) #circles radius
	rk = np.arange(0, len(r), 1).astype(int)
	fik=np.arange(0, len(fi), 1).astype(int)'''
	#ns=0
	nc=0
	for k in stim_t: 	
		for i in np.arange(0,50,1):

		       			dpx = X[0,:] - array[0,nc] #- r[ri] * sin(fi[kfi])
		    			dix = np.argmin(abs(dpx))
		       			dpy = Y[:,0] - array[1,nc] #- r[ri] * cos(fi[kfi])
					diy = np.argmin(abs(dpy))
					#Z[dix, diy] = dot_color[nc]
	
					'''for xi in np.arange(dix-3, dix+2, 1):
						for yi in np.arange(diy-3, diy+2, 1):				
				   			Z[xi, yi] = dot_color[nc] '''

					'''def dist(x1, y1, x2, y2):
		   				 return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
					for xi in np.arange(dix-3,  dix+2, 1):
						for yi in np.arange(diy-3,diy+2, 1):
						    if dist(dix, diy, xi, yi) <= 2:
							Z[xi,yi] = dot_color[nc]'''

					'''for xi in np.arange(dix-3, dix, 1):
						for yi in np.arange(diy-3, diy, 1):
							s=(dix-1)
							if (s>=(dix-3) and s==xi and ) :
								Z[xi,yi] = dot_color[nc]
							s+=1'''


					Z[dix-2, diy-3] = dot_color[i]	
					Z[dix-1, diy-3] = dot_color[i]
					Z[dix, diy-3] = dot_color[i]
					Z[dix+1, diy-3] = dot_color[i]	

					Z[dix-3, diy-2] = dot_color[i]	
					Z[dix-2, diy-2] = dot_color[i]
					Z[dix-1, diy-2] = dot_color[i]
					Z[dix, diy-2] = dot_color[i]
					Z[dix+1, diy-2] = dot_color[i]
					Z[dix+2, diy-2] = dot_color[i]

					Z[dix-3, diy-1] = dot_color[i]	
					Z[dix-2, diy-1] = dot_color[i]	
					Z[dix-1, diy-1] = dot_color[i]	
					Z[dix, diy-1] = dot_color[i]	
					Z[dix+1, diy-1] = dot_color[i]	
					Z[dix+2, diy-1] = dot_color[i]	

					Z[dix-3, diy] = dot_color[i]	
					Z[dix-2, diy] = dot_color[i]	
					Z[dix-1, diy] = dot_color[i]	
					Z[dix, diy] = dot_color[i]	
					Z[dix+1, diy] = dot_color[i]	
					Z[dix+2, diy] = dot_color[i]

					Z[dix-3, diy+1] = dot_color[i]	
					Z[dix-2, diy+1] = dot_color[i]	
					Z[dix-1, diy+1] = dot_color[i]	
					Z[dix, diy+1] = dot_color[i]	
					Z[dix+1, diy+1] = dot_color[i]	
					Z[dix+2, diy+1] = dot_color[i]	

					Z[dix-2, diy+2] = dot_color[i]	
					Z[dix-1, diy+2] = dot_color[i]	
					Z[dix, diy+2] = dot_color[i]	
					Z[dix+1, diy+2] = dot_color[i]	


					nc+=1
		#for i in np.arange(0, 1,1):
		stimuli[k, ...] = Z
		#ns+=1
	      	Z[0:, 0:] = screen_color



	return stimuli

