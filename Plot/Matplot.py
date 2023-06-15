from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as mplc
from matplotlib.colors import LogNorm
import math
import numpy as np


"""This module provides the functions that plot flow fields with triangular mesh
"""

def contourf_cylinder(x_ori,y_ori,z_ori,xlim=(-5,5),ylim=(-5,5),radius=0.5, level=100, vminf=None,vmaxf=None,normf=None,figname=None,savepath=None,
                      colormap='jet',axis='on',colorbar='on',figsize=(8,4),bluff_body={'type': 'circle','radius':[0.5],'facecolor':'none','edgecolor':'k','location_x':[0.0],'location_y':[0.0],'linewidth':1.0},
                      patch=None,colorbar_location={}, contour_line=False,linespace_type='linear',contour_line_wd=1.5, line_level=10, line_range=[None,None]):
    if colorbar == 'on':
        colorbar_loc={'orientation': 'vertical', 'fraction': 0.07, 'pad': 0.0, 'aspect': 20, 'shrink': 0.95}
        for key in colorbar_location.keys():
            colorbar_loc[key]=colorbar_location[key]
    
    if patch is None:
        patches_list=[bluff_body]
    elif type(patch) is list:
        patches_list=[bluff_body]+patch
    elif type(patch) is dict:
        patches_list=[bluff_body]+[patch]
                          
    x=x_ori*1.0
    y=y_ori*1.0
    z=z_ori*1.0
    triang = tri.Triangulation(x,y)
    # Mask off unwanted triangles.
    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    mask = np.where(xmid*xmid + ymid*ymid < radius*radius, 1, 0)
    triang.set_mask(mask)
    # pcolor plot.
    plt.figure(figsize=(figsize[0]*1.07,figsize[1]))
    plt.gca().set_aspect('equal')
    plt.rcParams['image.cmap'] = colormap
    if vmaxf is not None and vminf is not None:
        max = np.where(z>0.999*vmaxf,1,0)
        min = np.where(z<0.999*vminf,1,0)
        z[max==1]=0.999*vmaxf
        z[min==1]=0.999*vminf
        corange = np.linspace(vminf,vmaxf,10, endpoint=True)
        plt.tricontourf(triang, z,np.linspace(vminf,vmaxf,level, endpoint=True),vmin=vminf, vmax=vmaxf,norm=normf)
        if colorbar == 'on':
            bar = plt.colorbar(ticks=corange,orientation=colorbar_loc['orientation'], fraction=colorbar_loc['fraction'],pad=colorbar_loc['pad'],aspect=colorbar_loc['aspect'],shrink=colorbar_loc['shrink'])
            print(bar)
    else:
        plt.tricontourf(triang, z, level, vmin=vminf, vmax=vmaxf, norm=normf)
        if colorbar == 'on':
            plt.colorbar(orientation=colorbar_loc['orientation'], fraction=colorbar_loc['fraction'],pad=colorbar_loc['pad'],aspect=colorbar_loc['aspect'],shrink=colorbar_loc['shrink'])
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.axis(axis)
    plt.subplots_adjust(left=0, bottom=0, right=1.0, top=1.0, wspace=0, hspace=0)
    
    for j in range(len(patches_list)):
        patches=patches_list[j]
        if patches is not None and type(patches) is dict:
            if len(patches['location_x'])!=len(patches['location_y']):
                raise Exception("Coordinate array not equal length")
            if patches['type'] == 'circle':
                for i in range(len(patches['location_x'])):            
                    circle=plt.Circle((patches['location_x'][i],patches['location_y'][i]),patches['radius'][0],facecolor=patches['facecolor'],edgecolor=patches['edgecolor'],lw=patches['linewidth'])
                    plt.gca().add_artist(circle)
                    try:
                        patches['symmetric axis']
                    except:
                        pass
                    else:
                        if patches['symmetric axis'] is not None:
                            axis_vec=patches['symmetric axis']
                            mark_vec=(patches['location_x'][i],patches['location_y'][i])
                            project_vec=(axis_vec[0]*mark_vec[0]+axis_vec[1]*mark_vec[1])/abs(axis_vec[0]+axis_vec[1]*1j)
                            mark_new=(2.0*project_vec*axis_vec[0]-mark_vec[0],2.0*project_vec*axis_vec[1]-mark_vec[1])
                            circle=plt.Circle((mark_new[0],mark_new[1]),patches['radius'][0],facecolor=patches['facecolor'],edgecolor=patches['edgecolor'])
                            plt.gca().add_artist(circle)
    if contour_line==True:
        if linespace_type=='linear':
            plt.tricontour(triang, z, np.linspace(line_range[0],line_range[1],line_level, endpoint=True), colors='k', linewidths=contour_line_wd)
        elif linespace_type=='log10':
            plt.tricontour(triang, z, np.logspace(np.log10(line_range[0]),np.log10(line_range[1]),line_level, endpoint=True, base=10), colors='k', linewidths=contour_line_wd)
            plt.tricontour(triang, z, -np.logspace(np.log10(line_range[0]),np.log10(line_range[1]),line_level, endpoint=True, base=10)[::-1], colors='k', linewidths=contour_line_wd)
    if figname is not None:
        plt.title(figname, loc='left')
    if savepath is None:
        plt.show()
        return
    else:
        plt.savefig(savepath, dpi=128,pad_inches=0.0)  # save the figure to file
        plt.close()
        plt.clf()
        return

