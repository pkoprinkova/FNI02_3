import numpy as np
import nest.raster_plot
import matplotlib.pyplot as plt
import cPickle
import matplotlib.cm as cm
import matplotlib as mpl

params_folder='/home/simona/model/bin_Version_1/Parameters/'

LGN_size='20x20'
EI_size='20x10'
MT_size='20x10'
MSTd_size='20x20'
LIP_size='20x20'
layer_size='10x10'
ns=10
d=0.05
sigma_ts=10.0
sigma_I=3.0
nfp=3
sigma_lip=3.0
sigma_bg=3.0

f=open(params_folder+MSTd_size+'_MSTd_positions_bin.cpickle',"rb") 
MSTd_e_xy=cPickle.load(f)
MSTd_c_xy=cPickle.load(f)
MSTd_xe=cPickle.load(f)
MSTd_ye=cPickle.load(f)
MSTd_xc=cPickle.load(f)
MSTd_yc=cPickle.load(f)
f.close()

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

fig = plt.figure(1)
plt.title("Focal points of MSTe layer")
col=[]

size=[10]
z=np.zeros((len(MSTd_e_xy))).astype(int)

for i in np.arange(0, len(MSTc_fp)):
    if MSTc_fp[i,0]<0:
        col.append(mpl.colors.to_rgb("red"))
    elif MSTc_fp[i,0]>0:
        col.append(mpl.colors.to_rgb("blue"))
    else:
        col.append(mpl.colors.to_rgb("green"))

scat=plt.scatter(MSTd_e_xy[:,0], MSTd_e_xy[:,1], 10, color=col)

colorbar_ax = fig.add_axes([0.92, 0.11, 0.021, 0.77])
colorbar_ax.set_yticklabels(["left","right","center"])
#colorbar_ax.set_ylabel( "left\n\n\n\n right\n\n\n\n center\n\n\n\n",rotation=1)# position=(2,0.25))
colorbar_ax.set_yticks(range(2, 14,4))
colorbar_ax.set_xticks([])
colorbar_ax.tick_params(direction='in', length=3, width=1, colors='black', right=True, left=False, labelleft="off", labelright='on')
colorbar_ax.axhspan(0, 4, facecolor='red')
colorbar_ax.axhspan(4, 9, facecolor='blue')
colorbar_ax.axhspan(9, 12, facecolor='green')
plt.ylim(bottom=0)
plt.ylim(top=12)
plt.show()

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
cm=plt.cm.jet

xy = range(400)
z = xy
sc=plt.scatter(MT_e_xy1[:,0], MT_e_xy1[:,1],c=z, s=35, cmap=cm)


plt.colorbar(sc)

plt.show()