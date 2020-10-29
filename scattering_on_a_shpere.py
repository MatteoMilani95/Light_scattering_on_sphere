import numpy as np
import matplotlib.pyplot as plt
import math
from pynverse import inversefunc

####################################### configurations settings #########################

_config = {
        'sphere_calculation' : True, #True only if Matlab Engine is installed
        'laser_waist' : False,
        'brownian_motion' : False,
        }

_config_laser = {
        'beam waist (mm)' : 0.80851925,
        'wavelength (nm)' : 550,
        }

_config_optic_setup = {
        'bead_position (cm)' : 1, #position of the bead in therms of distance from the focus of the lens
        'focus_1 (cm)' : 6.6,
        }

_config_brownian = {
        'wavelength' : 550, # in nm
        'refraction_index' : 1.33,
        'scattering_angle' : math.pi/2, # in radiants
        'temperature' :293.15, # in K
        'viscosity' : 0.01, # Dynamic viscosity in Ns/m2
        'particles_radius' : 25, # in nm
        }
############################################# functions###################################

def rayleigh_rang(w_0,wave_lenght):
    #returns the rayleigh range of a laser beam give its waist and wavelenght
    #Require: wave_length in nm and w_0 in mm
    #Give: zr is in mm
    zr=math.pi*((w_0*10**-3))**2/(wave_lenght*10**-9)*10**3
    return zr

def beam_waist(w_0,wave_lenght,f):
    #returns modified beam waist and the new rayleigh range of a laser beam give its waist and wavelenght
        #and the focus of the length 
    #Require: wave_length in nm, w_0 in mm, f in cm
    #Return: w_f in mm, zr_f in mm
    w_f=(wave_lenght*10**-9)*(f*10**-2)/(math.pi*w_0*10**-3)*10**3; #not sure of this formula
    zr_f=rayleigh_rang(w_f,wave_lenght);

    return (w_f,zr_f)

def spot_dymension(w_0,d,wave_lenght,f):
    #function that returns the dimension of the laser spotgive at a distance z 
        #from the focus of the lenght f of the lens.
    #Require: w_0 mm,f cm, z cm
    #Return: D_f in mm, A_0 in mm^2
    w_r,zr_f=beam_waist(w_0,wave_lenght,f)
    D_f=2*(w_r*10**-3*np.sqrt(1.0+(wave_lenght*10**-9*d*10**-2/(math.pi*(w_r*10**-3)**2))**2))*10**3
    #D_2=2*(w_r*10**-3*np.sqrt(1.0+(d*10**-2/(zr_f*10**-3))**2))*10**3
    return D_f

def optic_path(R,n1,n2,D):
    H=np.linspace(-np.sqrt((R-0.0001)**2-D**2), np.sqrt((R-0.0001)**2-D**2), num=1000);
    h=[];
    scatt_angle=[];
    for i in range(len(H)):
        inner_h,scattering_angle=theta1_func(H[i],R,n1,n2,D)
        h.append(inner_h)
        scatt_angle.append(scattering_angle)
    
    return (h,H,scatt_angle)

def theta1_func(H_value,R,n1,n2,D):
    tc=np.arcsin(n2/n1)
    H=lambda theta1 :R*np.sin(theta1)/np.cos(np.arcsin(n1/n2*np.sin(theta1))-theta1)*1/(1-np.tan(np.arcsin(n1/n2*np.sin(theta1))-theta1)/np.tan(np.arcsin(n1/n2*np.sin(theta1))))
    theta=inversefunc(H,y_values=H_value,domain=[-tc, tc])
    h=R*np.sin(theta)/np.cos(np.arcsin(n1/n2*np.sin(theta))-theta)+D*np.tan(np.arcsin(n1/n2*np.sin(theta))-theta)
    if H_value>=0:
        theta_scattering=np.arcsin(R*np.sin(theta)/h)
    else:
        theta_scattering=math.pi-np.arcsin(R*np.sin(theta)/h)
    return h,theta_scattering

def q_vector(wave_lenght,theta,n):
    q=4*math.pi*n*np.sin(theta/2);
    l=1.0/q;
    return q,l

def f_brownian(n,wavelength,theta,T,nu,r):
    kb=1.38064852*10**-23
    tau=np.logspace(-6, 6, num=200)
    f=np.exp(-(4*math.pi*n/(wavelength*10**-9)*np.sin(theta))**2*kb*T/(6*math.pi*nu*(r**10**-9))*tau);
    return tau,f


#################################      the program starts here     #############################

if _config['laser_waist'] == True:
    
    LaserRaylieghRange=rayleigh_rang(_config_laser['beam waist (mm)'],_config_laser['wavelength (nm)'])
    #DiameterLaserOnBead=spot_dymension(_config_laser['beam waist (mm)'],_config_optic_setup['bead_position (cm)'],_config_laser['wavelength (nm)'],_config_optic_setup['focus_1 (cm)'])
    DiameterLaserOnBead=[];
    bead_position=np.linspace(-5, 5, num=100);
    for i in range(len(bead_position)):
        DiameterLaserOnBead.append(spot_dymension(_config_laser['beam waist (mm)'],bead_position[i],_config_laser['wavelength (nm)'],_config_optic_setup['focus_1 (cm)']));
        
    plt.plot(bead_position,DiameterLaserOnBead)
    plt.xlabel('Position respedt to the focus of the lens   (cm)')
    plt.ylabel('Diameter of the laser spot (mm)')
    plt.title('Beam waist')
    plt.grid(True)
    plt.savefig('C:\\Users\\hp\\Desktop\\PHD\\LIGHT_SCATTERING_EXPERIMENTS\\Light_scattering_on_a_sphere\\Spot_dimension_on_the_drop.png')


        

if _config['sphere_calculation'] == True:
    
    h,H,sc_ang=optic_path(R=1,n1=1.33,n2=1.00,D=0);
    h_01,H_01,sc_ang_01=optic_path(R=1,n1=1.33,n2=1.00,D=0.1);
    h_05,H_05,sc_ang_05=optic_path(R=1,n1=1.33,n2=1.00,D=0.5);
    h_07,H_07,sc_ang_07=optic_path(R=1,n1=1.33,n2=1.00,D=0.7);
    plt.figure() 
    plt.plot(H,sc_ang,label='D=0')
    plt.plot(H_01,sc_ang_01,label='D=0.1')
    plt.plot(H_05,sc_ang_05,label='D=0.5')
    plt.plot(H_07,sc_ang_07,label='D=0.7')
    plt.xlabel('Ray position outside the sphere  H (mm)')
    plt.ylabel('Scattering angle (rad)')
    plt.title('Optical Path R=1mm')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.figure() 
    plt.plot(H,h,label='D=0')
    plt.plot(H_01,h_01,label='D=0.1')
    plt.plot(H_05,h_05,label='D=0.5')
    plt.plot(H_07,h_07,label='D=0.7')
    plt.xlabel('Ray position outside the sphere  H (mm)')
    plt.ylabel('Ray position inside the sphere  h (mm)')
    plt.title('Optical Path R=1mm')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('C:\\Users\\hp\\Desktop\\PHD\\LIGHT_SCATTERING_EXPERIMENTS\\Light_scattering_on_a_sphere\\Optical_Path.png')

if _config['brownian_motion'] == True:
    f,tau=f_brownian(_config_brownian['refraction_index'],_config_brownian['wavelength'],_config_brownian['scattering_angle'],_config_brownian['temperature'],_config_brownian['viscosity'],_config_brownian['particles_radius'])
    
    plt.plot(tau,f)
    plt.xlabel('tau   (s)')
    plt.ylabel('g2')
    plt.title('g2 brownian motion')