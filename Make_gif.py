import imageio
import numpy as np

time=np.arange(0,100.1,0.2)
Re=100
i=0
fps=25
filenames=[]
for t in time:
    savepath='/mnt/d/dropbox_data/figures/cylinder_omega103_Re'+str(Re)+'_'+str(i)+'.png'
    i=i+2
    filenames.append(savepath)
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('/mnt/d/dropbox_data/figures/cylinder_omega103_Re'+str(Re)+'.gif', images,fps=fps)



#time=np.arange(76,151,1)
#i=76
#filenames=[]
#for t in time:
#    savepath='./figures/zero_trace_'+str(i)+'.png'
#    i=i+1
#    filenames.append(savepath)
#images = []
#for filename in filenames:
#    images.append(imageio.imread(filename))
#imageio.mimsave('./figures/zero_trace.gif', images)