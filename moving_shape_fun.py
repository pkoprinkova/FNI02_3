import numpy as np
from math import cos, sin, sqrt,degrees,radians

def moving_shape(lx, ly, dx, dy, N_stimuli, dt_stimuli, xc0 = -2.0, yc0 = -2.0, dd = 0.015, hs=11, ws=5, alfa=0, dot_color=1.0, screen_color=0.5):

	#moving dot stimuli
	Nx = int(lx / dx)
	Ny = int(ly / dy)
	x = np.arange(-lx/2, lx/2, dx)
	y = np.arange(-ly/2, ly/2, dy)
	X, Y = np.meshgrid(x, y)
	Z = np.zeros((Nx, Ny), dtype=float)
	Z += screen_color
#	t = np.arange(0, N_stimuli * dt_stimuli, dt_stimuli)
	stimuli = np.zeros((int(N_stimuli*dt_stimuli), Nx, Ny), dtype=float)

	#xc0 = -2.0
	#yc0 = -2.0
	#dd = 0.015
	#rd = 0.1
	#fi = np.linspace(0, 2 * np.pi, 100)
	#r = np.linspace(0, rd, 100) #circles radius
	xc = np.zeros(int(N_stimuli*dt_stimuli), dtype=float) #dot center x
	yc = np.zeros(int(N_stimuli*dt_stimuli), dtype=float) #dot center y
	xc[0:] = xc0
	yc[0:] = yc0
	h=np.arange(-hs/2,hs/2,0.5)*dy
	w=np.arange(-ws/2,ws/2,0.5)*dx

	

	stim_step = int(dt_stimuli*5)
	stim_t = np.arange(0, int(N_stimuli*dt_stimuli), stim_step).astype(int)
	#rk = np.arange(0, len(r), 1).astype(int)
	#fik=np.arange(0, len(fi), 1).astype(int)
	ns = 0
	for k in stim_t: #1:stim_step:N_stimuli*dt_stimuli
		for kr in w: 
      			for kfi in h: 
				#x1=xc[ns] - kr
				#y1=yc[ns] + kfi
				x1=(-kr)*cos(alfa)-kfi*sin(alfa)
				y1=(-kr)*sin(alfa)+kfi*cos(alfa)
				#a=xc[ns]-x1
				#b=y1-yc[ns]
				#gama=radians(180-alfa-degrees(np.arctan(b/a)))
				#c=sqrt((a**2 + b**2)/(1+(np.tan(gama))**2))
				#d=c*np.tan(gama) - b
				#x2=c + xc[ns]
				#y2=d + y1
         			dix = np.argmin(abs(X[0,:] - x1-xc[ns]))
	       			diy = np.argmin(abs(Y[:,0] - y1-yc[ns]))
				
				#x11=xc[ns] + kr
				#y11=yc[ns] + kfi
				x11=kr*cos(alfa)-kfi*sin(alfa)
				y11=kr*sin(alfa)+kfi*cos(alfa)
				'''a1=y11-yc[ns]
				b1=x11-xc[ns]
				gama1=radians(180-alfa-degrees(np.arctan(b1/a1)))	
				c1=sqrt((a1**2 + b1**2)/(1+(np.tan(gama1))**2))
				d1=c1*np.tan(gama1) - b1
            			y21=c1 + xc[ns]
				x21=d1 + y11'''
				dix1 = np.argmin(abs(X[0,:] - x11-xc[ns]))
	       			diy1 = np.argmin(abs(Y[:,0] -  y11-yc[ns]))


				#x12=xc[ns] - kr
				#y12=yc[ns] - kfi 
				x12=(-kr)*cos(alfa)+kfi*sin(alfa)
				y12=(-kr)*sin(alfa)-kfi*cos(alfa)
				'''a2=yc[ns]-y12
				b2=xc[ns]-x12
				gama2=radians(180-alfa-degrees(np.arctan(b2/a2)))	
				c2=sqrt((a2**2 + b2**2)/(1+(np.tan(gama2))**2))
				d2=c2*np.tan(gama2) - b2
            			y22=c2 + xc[ns]
				x22=y12 - d2'''
            			dix2 = np.argmin(abs(X[0,:] - x12-xc[ns]))
	       			diy2 = np.argmin(abs(Y[:,0] - y12-yc[ns]))

				#x13=xc[ns] + kr 
				#y13=yc[ns] - kfi
				x13=kr*cos(alfa)+kfi*sin(alfa)
				y13=kr*sin(alfa)-kfi*cos(alfa)
				'''a3=xc[ns]-x13
				b3=yc[ns]-y13
				gama3=radians(180-alfa-degrees(np.arctan(b3/a3)))	
				c3=sqrt((a3**2 + b3**2)/(1+(np.tan(gama3))**2))
				d3=c3*np.tan(gama3) - b3
				x23=c3 + xc[ns]
				y23=d3 + y13'''
            			dix3 = np.argmin(abs(X[0,:] - x13-xc[ns]))
	       			diy3 = np.argmin(abs(Y[:,0] - y13-yc[ns]))
				

	       			Z[dix, diy] = dot_color 
				Z[dix1, diy1] = dot_color
				Z[dix2, diy2] = dot_color
				Z[dix3, diy3] = dot_color
				
				
		xc[ns+1] = xc[ns] - np.sign(xc0)* dd
        	yc[ns+1] = yc[ns] - np.sign(yc0)* dd
        	ns += 1
            	s = np.arange(k, k+stim_step, 1).astype(int)
            	for index in s:
                	stimuli[index, ...] = Z
            	Z[0:, 0:] = screen_color

	
	return stimuli
