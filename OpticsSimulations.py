# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:42:18 2021

@author: Matteo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:19:29 2021

@author: Matteo
"""

import numpy as np
import scipy as sc
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
    
    beta = ( np.cos( theta1 ) -  np.cos( gamma ) * np.cos( theta2 ) ) / ( 1 - np.cos( theta2 )**2 )
    alpha = np.cos( gamma ) - beta * np.cos( theta2 )
    
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
        ax.view_init(10, 120)
        
        ax.scatter(a_0[0], a_0[1], a_0[2], color="g", s=50)
        
        # draw sphere
        u, v = np.mgrid[0:2*np.pi:80j, 0:np.pi:80j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="g",linewidth = 0.1)
        
        
        ax.grid(True)
        
        plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\RayTracing\\Ooptical_Path.png', dpi=300)
        return a_0,a,b_0,b,r_0,r
    else:
        m=3#print('no plot has been saved')
    
    return a_0,a,b_0,b,r_0,r



def LaserTilt(n1,n2,x,plot = False ):
    
    
    r_0 = np.array( [0, 0, 0] )
    
    if x[0]>0:
        a = np.array( [-1, 0, 0] )
        
    else:
        a = np.array( [1, 0, 0] )
        
    a_0 = np.array( [x[0], x[1], x[2]] ) - a
        
    r = np.array( [x[0], x[1], x[2]] )#np.array( [x[0], np.sqrt( 1 - x[0]**2 - x[1]**2), x[1]] )
    
    theta2 = np.pi - np.arccos(  np.dot(a,r) )
    theta1 = np.arcsin( n2 / n1 * np.sin( theta2 ) )
    gamma = theta2 - theta1
    '''
    print(theta2 * 360 / np.pi)
    print(theta1 * 360 / np.pi)
    print(gamma * 360 / np.pi)
    '''
    beta = (  - np.cos( theta1 ) +  np.cos( gamma ) * np.cos( theta2 ) ) / ( 1 - np.cos( theta2 )**2 )
    alpha = np.cos( gamma ) + beta * np.cos( theta2 )
    
    if x[2]==0:
        b = -a
    else:    
        b = (alpha * a + beta * r)
    b_0 =  np.array( [x[0], x[1], x[2]] )
    #print(b)
    
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
        ax.view_init(1, 90)
        
        ax.scatter(a_0[0], a_0[1], a_0[2], color="g", s=50)
        
        # draw sphere
        u, v = np.mgrid[0:2*np.pi:80j, 0:np.pi:80j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="g",linewidth = 0.1)
        
        
        ax.grid(True)
        
        plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\RayTracing\\Ooptical_Path.png', dpi=300)
        return b
    else:
         m=3#print('no plot has been saved')
    
    return b


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
    
    zl=0.0
    xl=np.sqrt( 1 - zl**2 )
    A = np.array( [xl , 0 , zl] )
    #l = np.array( [1 , 0 , 0] )
    
    l = LaserTilt(n1,n2,A,plot = False )
    p = np.array( [0 , 0 , 1] )
    
    df = 2
    sigma = beam_waist
    particle_diameter = 7*10**-6 
    volume_fraction = 0.05
    eta = particle_diameter * volume_fraction ** (1 /( df  - 3 ))
    
    a_0,a,b_0,b,r_0,r = BeadRayTrace(n1, n2, [x[0],x[1]] ,plot = False)
    
    theta_scattering = np.arccos( np.dot(l,b) )
    q_scattering = 4 * np.pi * n1 * np.sin( theta_scattering / 2 ) / (532*10**-6)
    phi_angle = np.arccos( np.dot(p,b) )
    
    Form_factor = 1 / ( 1 + (eta * q_scattering)**df)
    
    t0,tmax = LineSphereIntersection(a_0, b_0)
    
    func_isotropic = lambda t:  1 / ( sigma *np.sqrt( 2 * np.pi ) ) * np.exp(- np.linalg.norm(np.cross( (a_0 - t * b - A),l )) **2 / sigma**2) * np.sin( phi_angle )**2 
    I_0 = integrate.quad(func_isotropic , t0, tmax)
    
    func = lambda t:  1 / ( sigma *np.sqrt( 2 * np.pi ) ) * np.exp(- np.linalg.norm(np.cross( (a_0 - t * b - A),l )) **2 / sigma**2) * np.sin( phi_angle )**2 * Form_factor 
    I_gel = integrate.quad(func , t0, tmax)
    
    I = I_gel[0]/I_0[0]
    
    return I,theta_scattering,I_gel[0]



def LinePlaneIntersection(l_0,l):
    normal_plane = np.array([0,1,0])
    p_0 = np.array([1,0,0])
    d =np.dot( (p_0 - l_0) , normal_plane ) / np.dot( l,normal_plane )
    
    interseption_point = l_0 + l * d
    
    return interseption_point

def DifferentialPath(n1 , n2, x, plot = False):
    a_0,a,b_0,b,r_0,r = BeadRayTrace(n1, n2, [x[0],x[1]] ,plot )
    interseption = LinePlaneIntersection(b_0,b)
    
    Dx = np.abs(x[0]-interseption[0])
    Dz = np.abs(x[1]-interseption[2])
    D = np.sqrt( Dx**2 + Dz**2 )
    return Dx,Dz,D



def g2PlasticDeformation(n1,n2,x,beam_waist):
    
    a_0,a,b_0,b,r_0,r = BeadRayTrace(n1, n2, [x[0],x[1]] ,plot = False)
    zl=0.0
    xl=np.sqrt( 1 - zl**2 )
    A = np.array( [xl , 0 , zl] )

    
    l = LaserTilt(n1,n2,A,plot = False )
    p = np.array( [0 , 0 , 1] )
    
    sigma = beam_waist
    
    phi_angle = np.arccos( np.dot(p,b) )
    
    theta_scattering = np.arccos( np.dot(l,b) )
    
    q = 4 * np.pi * n1 * np.sin( theta_scattering / 2 ) / (532*10**-6) * (-l + b) 
     
    
    u_x,u_y,u_z = ParametricDisplacement(a_0,x)
    
    t0,tmax = LineSphereIntersection(a_0, b_0)
    
    
    
    df = 2
    sigma = beam_waist
    particle_diameter = 7*10**-6 
    volume_fraction = 0.05
    q_scattering = 4 * np.pi * n1 * np.sin( theta_scattering / 2 ) / (532*10**-6)
    eta = particle_diameter * volume_fraction ** (1 /( df  - 3 ))
    Form_factor = 1 / ( 1 + (eta * q_scattering)**df)
    
    

    func = lambda t:  1 / ( sigma *np.sqrt( 2 * np.pi ) )* np.exp( -1j * ( q[0] * u_x(t) + q[1] * u_y(t) + q[2] * u_z(t)) ) * np.exp(- np.linalg.norm(np.cross( (a_0 - t * b - A),l )) **2 / sigma**2) * np.sin( phi_angle )**2 * Form_factor  
    
    # 
    
    real_f = lambda t: sc.real(func(t))
    imag_f = lambda t: sc.imag(func(t))
    integ_r,bho = integrate.quad(real_f , t0, tmax)
    integ_i,bho = integrate.quad(imag_f , t0, tmax)
    
    
    I,theta_scattering,I_gel = IntensityProfile(n1, n2, x, beam_waist,plot=False)
    
    g2 = [integ_r,integ_i]
    

    return g2,I_gel

def ParametricDisplacement(a_0,x):
    
    Ri = 1.00
    ri = 1e-4
    dr = 0.72*10**-3
    Rf = Ri - dr
    O = np.asarray([0,0,0])
    
    C =  1 / ( Ri + ri**3 / Ri**2 ) * ( Rf - Ri )
    E = - C * ri**3

    theta = np.arctan( np.sqrt( a_0[0]**2 + a_0[1]**2 ) / a_0[2])
    phi = np.arctan( a_0[1] / a_0[0] )
    
    if a_0[0] !=  11111110 :
        
        d = lambda t: np.linalg.norm( t * (a_0 - O)   )
        u_r = lambda t: C * d(t) + E / d(t)**2
    else:
        u_r = lambda t: t * 0
        
    
    u_x = lambda t: u_r(t) * np.sin(theta) * np.cos(phi)
    u_y = lambda t: u_r(t) * np.sin(theta) * np.sin(phi)
    u_z = lambda t: u_r(t) * np.cos(theta)
        
        
    
    
    return u_x,u_y,u_z

def g2Simulation(n1,n2,x_0 ,beam_waist):
    
    speckle_size_x = 0.03
    speckle_size_z = 0.01
    
    step_x = 5
    step_z = 5 
        
    dx = speckle_size_x / step_x
    dz = speckle_size_z / step_z
           
    x_spec = np.linspace(x_0[0],x_0[0]+speckle_size_x,step_x)
    z_spec = np.linspace(x_0[1],x_0[1]+speckle_size_z,step_z)
    
    g2real= []
    g2imag= []
    Int = []
    
    for i in range(len(x_spec)):
        for j in range(len(z_spec)):
            g2,I = g2PlasticDeformation(n1 = 1.33,n2 = 1,x = [x_spec[i],z_spec[j]],beam_waist=0.1)
            g2real.append(g2[0]*dx*dz)
            g2imag.append(g2[1]*dx*dz)
            Int.append(I*dz*dx)
            
    integral_r = np.sum( np.asarray(g2real)) 
    integral_i = np.sum( np.asarray(g2imag)) 
    normalization = np.sum( np.asarray(Int) )
    
    G2 = ( integral_r**2 + integral_i**2 )  / (normalization**2 )
    print('n :' + str(integral_r**2 + integral_i**2))
    print('d :' + str(normalization**2))
    
    return G2


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    
    try:
        n_ = int(n * multiplier) / multiplier
        
    except ValueError:
        n_ = n
    return n_

########## CHECK AT Z = 0 ###########

line = np.linspace(-1,1,50)
point = []


for j in range(len(line)):
    I,theta_scattering,Ig = IntensityProfile(n1 = 1.33, n2 = 1, x = [line[j],0], beam_waist=0.05,plot=True)
    q_scattering = 4 * np.pi * 1.33 * np.sin( theta_scattering / 2 ) / (532*10**-6)
    point.append(np.array([line[j],0,I,theta_scattering*180/np.pi,q_scattering]))
    
    
x_ = []
z_ = []
Intesity_ = []
theta_s_ = []
q_ = []

for i in range(len(point)):
    x_.append(point[i][0])
    z_.append(point[i][1])
    Intesity_.append(point[i][2])
    theta_s_.append(point[i][3])
    q_.append(point[i][4]) 

plt.figure()
plt.plot(x_[1:-1],Intesity_[1:-1],'o')
plt.xlabel("x [mm]")
plt.ylabel("intasity [a.u.]")
plt.legend(loc='upper left')
plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\Fig_simulations\\Intesity_plot_.png', dpi=300)

plt.figure()
plt.loglog(q_[1:-1],Intesity_[1:-1],'o')
plt.xlabel("q [m-1]")
plt.ylabel("intasity [a.u.]")
plt.legend(loc='upper left')
plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\Fig_simulations\\Intesity_plot_q_fun.png', dpi=300)

plt.figure()
plt.plot(x_,theta_s_,'o')
plt.xlabel("x")
plt.ylabel("scattering angle [grad]")
plt.legend(loc='upper left')
plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\Fig_simulations\\Scatt_ang_plot_.png', dpi=300)

########## COMPLETE PROGRAM ###########
'''
raw = np.linspace(-1,1,70)
columns = np.linspace(-1,1,70)
points = []
dpoints = []

for i in range(len(columns)):
    for j in range(len(raw)):
        I,theta_scattering,Ig = IntensityProfile(n1 = 1.36, n2 = 1, x = [raw[j],columns[i]], beam_waist=0.15,plot=False)
        dx,dz,d = DifferentialPath(n1 = 1.36, n2 = 1, x = [raw[j],columns[i]],plot=False)
        points.append(np.array([raw[j],columns[i],I,theta_scattering*180/np.pi]))
        dpoints.append(np.array([dx,dz,d]))
    

x=[]
z=[]
Intesity=[]
theta_s=[]
Dx = []
Dz = []
D = []

for i in range(len(points)):
    x.append(points[i][0])
    z.append(points[i][1])
    Intesity.append(points[i][2])
    theta_s.append(points[i][3])
    Dx.append(dpoints[i][0])
    Dz.append(dpoints[i][1])
    D.append(dpoints[i][2])

plt.figure()
plt.scatter(x, z, c=Intesity)
plt.xlabel("x [mm]")
plt.ylabel("z [mm]")
plt.legend(loc='upper left')
theta = np.linspace(0, 2 * np.pi ,1000)
plt.plot( np.cos(theta),  np.sin(theta))
clb = plt.colorbar()
clb.set_label('Intensity', labelpad=10, rotation=270)
plt.axes().set_aspect('equal', 'datalim')
plt.title('Intesity plot')
plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\Fig_simulations\\Intesity_plot.png', dpi=300)

plt.figure()
plt.scatter(x, z, c=theta_s)
plt.xlabel("x [mm]")
plt.ylabel("z [mm]")
plt.legend(loc='upper left')
theta = np.linspace(0, 2 * np.pi ,1000)
plt.plot( np.cos(theta),  np.sin(theta))
clb = plt.colorbar()
clb.set_label('Scattering angle ', labelpad=10, rotation=270)
plt.axes().set_aspect('equal', 'datalim')
plt.title('Scattering angle plot')
plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\Fig_simulations\\scattang_plot.png', dpi=300)

plt.figure()
plt.scatter(x, z, c=Dx)
plt.xlabel("x [mm]")
plt.ylabel("z [mm]")
plt.legend(loc='upper left')
theta = np.linspace(0, 2 * np.pi ,1000)
plt.plot( np.cos(theta),  np.sin(theta))
clb = plt.colorbar()
clb.set_label('Dx ', labelpad=10, rotation=270)
plt.axes().set_aspect('equal', 'datalim')
plt.title('Dx')
plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\Fig_simulations\\Dx.png', dpi=300)

plt.figure()
plt.scatter(x, z, c=Dz)
plt.xlabel("x [mm]")
plt.ylabel("z [mm]")
plt.legend(loc='upper left')
theta = np.linspace(0, 2 * np.pi ,1000)
plt.plot( np.cos(theta),  np.sin(theta))
clb = plt.colorbar()
clb.set_label('Dz ', labelpad=10, rotation=270)
plt.axes().set_aspect('equal', 'datalim')
plt.title('Dz plot')
plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\Fig_simulations\\Dz.png', dpi=300)

plt.figure()
plt.scatter(x, z, c=D)
plt.xlabel("x [mm]")
plt.ylabel("z [mm]")
plt.legend(loc='upper left')
theta = np.linspace(0, 2 * np.pi ,1000)
plt.plot( np.cos(theta),  np.sin(theta))
clb = plt.colorbar()
clb.set_label('Dz ', labelpad=10, rotation=270)
plt.axes().set_aspect('equal', 'datalim')
plt.title('Dz plot')
plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\Fig_simulations\\D.png', dpi=300)


############################################    DISPLACEMENT FIELD SIMULATION BLOCK #################################
'''
#g2real3 = g2Simulation(n1 = 1.33,n2 = 1,x_0 = [-0.01,0.7],beam_waist=0.1)


g2real = g2Simulation(n1 = 1.33,n2 = 1,x_0 = [0.9,0.0],beam_waist=0.1)


raw = np.linspace(-1,1,50)
columns = np.linspace(-0.8,0.8,1)
points = []
dpoints = []

for i in range(len(columns)):
    for j in range(len(raw)):
        g2real = g2Simulation(n1 = 1.33,n2 = 1,x_0 = [raw[j],0],beam_waist=0.1)
        points.append(np.array([raw[j],0,g2real]))
    

x=[]
z=[]
G2=[]
Qtau=[]

for i in range(len(points)):
    x.append(points[i][0])
    z.append(points[i][1])
    G2.append(points[i][2])
    Qtau.append(q_[i] * -20/np.log(np.asarray(points[i][2])) )



plt.figure()
plt.scatter(x, z, c=G2)
plt.xlabel("x [mm]")
plt.ylabel("z [mm]")
plt.legend(loc='upper left')
theta = np.linspace(0, 2 * np.pi ,1000)
plt.plot( np.cos(theta),  np.sin(theta))
clb = plt.colorbar()
clb.set_label('g2-1', labelpad=10, rotation=270)
plt.axes().set_aspect('equal', 'datalim')
plt.title('g2-1 plot')
plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\Fig_simulations\\G2_plot_.png', dpi=300)


tau = -20/np.log(np.asarray(G2))
plt.figure()
plt.plot(x,G2,'o')
plt.xlabel("x [mm]")
plt.ylabel("tau [s]")
plt.legend(loc='upper left')
plt.title('tau plot')
plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\Fig_simulations\\g2_plot_LEFT.png', dpi=300)

plt.figure()
plt.semilogy(x,tau,'o')
plt.xlabel("x [mm]")
plt.ylabel("tau [s]")
plt.legend(loc='upper left')
plt.title('tau plot')
plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\Fig_simulations\\tau_plot_LEFT.png', dpi=300)


plt.figure()
plt.semilogy(x,Qtau,'o')
plt.xlabel("x [mm]")
plt.ylabel("tau [s]")
plt.legend(loc='upper left')
plt.title('tau plot')
plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\Fig_simulations\\tau_plot_LEFT.png', dpi=300)
