# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:19:29 2021

@author: Matteo
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special


def BeadRayTrace(n1,n2,x,plot = False ):
    
    a_0 = np.array( [x[0], np.sqrt( 1 - x[0]**2 - x[1]**2), x[1]] )
    r_0 = np.array( [0, 0, 0] )
    
    a = np.array( [0, 1, 0] )
    r = np.array( [x[0], np.sqrt( 1 - x[0]**2 - x[1]**2), x[1]] )
    
    theta2 = np.arccos( np.dot(a,r) )
    theta1 = np.arcsin( n2 / n1 * np.sin( theta2 ) )
    gamma = theta2 - theta1 
    
    beta = ( np.cos( theta2 )**2 - np.cos( theta1 ) ) / (np.cos( theta2 )**2 - 1)      #np.cos( theta1 ) * ( np.cos( gamma ) - 1 ) / ( np.cos( theta1 ) * np.cos( theta2 ) - 1 )
    alpha = np.cos( theta2 ) * ( np.cos( theta1 ) - 1 ) / (np.cos( theta2 )**2 - 1)    #1 - ( np.cos( gamma ) - 1 ) / ( np.cos( theta1 ) * np.cos( theta2 ) - 1 )
    
    b = alpha * a + beta * r
    b_0 =  r - b 

    
    if plot == True:
        X = np.array( [a_0[0],b_0[0],r_0[0]])
        Y = np.array( [a_0[1],b_0[1],r_0[1]])
        Z = np.array( [a_0[2],b_0[2],r_0[2]])
        
        U = np.array( [a[0],b[0],r[0]])
        V = np.array( [a[1],b[1],r[1]])
        W = np.array( [a[2],b[2],r[2]])
        
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(X, Y, Z, U, V, W, length=1, normalize=True,arrow_length_ratio=0.2)
        
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.view_init(20, 120)
        
        ax.scatter(a_0[0], a_0[1], a_0[2], color="g", s=50)
        
        # draw sphere
        u, v = np.mgrid[0:2*np.pi:80j, 0:np.pi:80j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="g",linewidth = 0.1)
        
        
        ax.grid(True)
        
        plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\Ooptical_Path.png', dpi=300)
        return a_0,a,b_0,b,r_0,r
    else:
        print('no plot has been saved')
    
    return a_0,a,b_0,b,r_0,r

def LineSphereIntersection( a_0 , b_0 , Sc = [0,0,0], R =1 ):
    '''
    pass the two points that identify the line
    
    '''
    x1 = a_0[0]
    y1 = a_0[1]
    z1 = a_0[2]
    
    x2 = b_0[0]
    y2 = b_0[1]
    z2 = b_0[2]
    
    xc = Sc[0]
    yc = Sc[1]
    zc = Sc[2]
    
    alpha = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2
    beta = - 2 * ((x2 - x1) * (xc - x1) + (y2 - y1) * (yc - y1) + (zc - z1) * (z2 - z1))
    gamma = (xc - x1)**2 + (yc - y1)**2 + (zc - z1)**2 - R**2
    
    t1 = ( - beta + np.sqrt( beta**2 - 4 * alpha * gamma ) ) / ( 2 * alpha )
    t2 = ( - beta - np.sqrt( beta**2 - 4 * alpha * gamma ) ) / ( 2 * alpha )
    
    if t1 > t2:
        tmax = t1
        t0 = t2
    else:
        tmax = t2
        t0 = t1
    
    return t0,tmax

def IntensityProfile( n1 , n2, x , beam_waist , plot = False):
    
    A = np.array( [-1 , 0 , 0] )
    l = np.array( [1 , 0 , 0] )
    p = np.array( [0 , 0 , -1] )
    
    df = 2
    sigma = beam_waist
    
    a_0,a,b_0,b,r_0,r = BeadRayTrace(n1, n2, [x[0],x[1]] ,plot=True)
    
    theta_scattering = np.arccos( np.dot(l,b) )
    q_scattering = 4 * np.pi * n1 * np.sin( theta_scattering / 2 )
    phi_angle = np.arccos( np.dot(p,b) )
    
    Form_factor = 1 / (q_scattering)**df
    
    t0,tmax = LineSphereIntersection(a_0, b_0)
    
    func = lambda t:  1 / ( sigma *np.sqrt( 2 * np.pi ) ) * np.exp(- np.linalg.norm(np.cross( (a_0 - t * b - A),l )) **2 / sigma**2) * np.sin( phi_angle )**2  
    I = integrate.quad(func , t0, tmax)
    
    return I

I = IntensityProfile(n1 = 1.33, n2 = 1, x = [-0.7,0.0], beam_waist=0.1)
