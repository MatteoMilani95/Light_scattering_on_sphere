import numpy as np
import matplotlib.pyplot as plt
import math
from pynverse import inversefunc
from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import leastsq, least_squares
import easygui as g
from lmfit import Minimizer, Parameters, fit_report
import os
from sys import exit

####################################### configurations settings #########################

_config = {
        'sphere_calculation' : False, #True only if Matlab Engine is installed
        'laser_waist' : False,
        'brownian_motion' : False,
        'data_analysis' : True,
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
_config_data_analysis = {
        'cI_file_nber' : 2,  #Number of cI files to process
        'cI_file_folder_path' : r"E:\20201029_gelation2_silicagel_0036M_15VF_15ul\out\all_12x5_ROI",  #Path of the folder where the cI files are stored
        'out_folder_path': r"E:\20201029_gelation2_silicagel_0036M_15VF_15ul\out\all_12x5_ROI",  #Path of the folder where the cI files are stored
        'cI_file_name' : r'ROI',     #Name of the CI files (without numbers)
        'cI_file_digits' : 4,    #Number of digits in the CI file name
        'cI_file_first_number' : 1,  #ID of the first cI file
        'cI_file_extension' : r'ci.dat',   #End of the file name
        'lag_time' : 0.1,    #Lag time between 2 successive images (in s)

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


def determine_ROI_file_ID(cI_file_digits, file_number):
    """
    Calculates the ID of a file

    Parameters
    ----------
    cI_file_digits : int
                     Number of digits following the file name
    file_number : int
                  ID of the file number

    Returns
    -------
    str
        ID of the file number including zeros

    """
    file_ID=[]
    for i in range(cI_file_digits):
        if len(file_ID) < cI_file_digits:
            if len(str(file_number)) < (cI_file_digits - len(file_ID)):
                file_ID.append('0')
            else:
                file_ID.append(str(file_number))
    return "".join(file_ID) 

def get_tau_value(dID, lag_time):
    """
    Determines tau from dID

    Parameters
    ----------
    dID : str
          String starting by 'd' and followed by tau
    lag_time : float
               Delay time between 2 pictures

    Returns
    int
          tau

    """
    decomposition=list(dID)
    for i in range(len(dID)):
        if decomposition[0].isalpha():
            decomposition.pop(0)
    return float(''.join(decomposition))*lag_time
                    
def get_tau_list(df, lag_time):
    """
    Gets list of lag times

    Parameters
    ----------
    df : DataFrame
         Contains ROI000x.dat data
    lag_time : float
               Delay time between 2 pictures

    Returns
    -------
    tau_list : list
               Contains lag times

    """
    tau_list=[]
    for i in range(2,len(df.columns) - 1):
        tau_list.append(get_tau_value(df.columns[i], lag_time))
    return tau_list

def f(var,xs):
    """
    Single exponential with beta = 1 and a baseline

    Parameters
    ----------
    var : list
          [amplitude, 1/tau, baseline]
    xs : Array of float64
         x values

    Returns
    -------
    Array of float64
        y values

    """
    return (var[0]*np.exp(-((xs/var[1])**var[2])))**2 +var[3]

def func(params, xs, data=None):
    """
    Determines the residuals

    Parameters
    ----------
    params : lmfit.parameter.Parameters
             Values of the parameters
    xs : Array of float64
         x values of the points to be fitted
    ys : Array of float64
         y values of the points to be fitted

    Returns
    -------
    Array of float64
        Residuals, i.e. y(calculated values) - y(experimental values)

    """
    A = params['A']
    tauc = params['tauc']
    beta = params['beta']
    y0 = params['y0']
    
    if data is None:
        return (A * np.exp(-((xs/tauc)**beta)))**2 + y0
    else:
        return(A * np.exp(-((xs/tauc)**beta)))**2 + y0 - data

def jacob(var,xs,ys):
    """
    Returns the derivatives of the fitting function with respect to the fitting parameters. Eg.: d0 is the derivative of f with respect to var[0]

    Parameters
    ----------
    var : list
          List of variables needed to get f to work
    xs : Array of float64
         x values of the points to be fitted
    ys : Array of float64
         y values of the points to be fitted

    Returns
    -------
    Array of float64
        Derivatives of the fitting function
    """
    d0 = np.exp(-((var[1]*xs)**var[2]))
    d1 = (-(xs**var[2]) * var[2] * (var[1]**(var[2]-1))) * d0
    d2 = (np.log(var[1]*xs) * np.exp(var[2]*np.log(var[1]*xs))) * d0
    d3 = np.ones(len(xs))
    return np.column_stack((d0,d1,d2,d3))

def fitting_data_in_series(processed_data, I_df, results, tau_min, tau_max, series_name, twi=None, twf=None, Npts=None):
    """
    Saves fitting data in a Series

    Parameters
    ----------
    processed_data : str
                     Range of processed data
    twi : float, optional
         Ageing time. The Default is None.
    twf : float, optional
          Last ageing time considered to compute g2 - 1. The Default is None.
    I_df : DataFrame
          Contains the Iav and SE_Iav
    results : lmfit.minimizer.MinimizerResult
              Results from fitting with a stretched exponential with a baseline
    tau_min: int or float
             Minimum value of lag time on which the fit is performed
    tau_max: int or float
             Maximum value of lag time on which the fit is performed
    series_name : str
                  Name to give to the created series
    Npts: int, optional
          Number of points per decade, when data are fillted for logspacec values of tw. The Default is None.

    Returns
    -------
    fitting_results_ser : pd.Series
                          Series containing the results of the fitting

    """
    fitting_results_ser=pd.Series(index=['processed data', 'twi', 'twf', 'Nptspdecade', 'Iav_value', 'Iav_SE', 'A_value', 'A_SE', 'tauc_value', 'tauc_SE', 'beta_value', 'beta_SE', 'y0_value', 'y0_SE','tau_min', 'tau_max', 'iter_nber', 'chi2', 'red_chi2', 'success', 'message'], dtype=float)
    
    fitting_results_ser['processed data']=processed_data
    
    if twi is not None:
        fitting_results_ser['twi']=twi
        fitting_results_ser['twf']=twf
    
    if Npts is not None:
        fitting_results_ser['Nptspdecade']=Npts
    
    fitting_results_ser['Iav_value']=I_df.iloc[1, len(I_df.columns)-1]
    fitting_results_ser['Iav_SE']=I_df.loc['SE_Iav', series_name]
    
    counter=6

    for key in results.params:
        fitting_results_ser.iloc[counter]=results.params[key].value
        counter+=1
        fitting_results_ser.iloc[counter]=results.params[key].stderr
        counter+=1
    fitting_results_ser['tau_min']=tau_min
    fitting_results_ser['tau_max']=tau_max
    fitting_results_ser['iter_nber']=results.nfev
    fitting_results_ser['chi2']=results.chisqr
    fitting_results_ser['red_chi2']=results.redchi
    fitting_results_ser['success']=results.success
    fitting_results_ser['message']=results.message
    fitting_results_ser.rename(series_name, inplace=True)
    
    return fitting_results_ser

def g2_fitting_data_in_df(full_g2_df, df2fill, column_name, fitting_results=None, fitted_df=None):
    """
    Saves tau, g2-1 and fitted g2-1 values

    Parameters
    ----------
    full_g2_df: DataFrame
                Contains the raw g2 - 1 data
    fitted_df : DataFrame, optional
                Contains the data used for fitting
    column_name: str
                 Names of the data to save
    df2fill : DataFrame
              DataFrame to fill up
    fitting_results : lmfit.minimizer.MinimizerResult, optional
                      Results from fitting

    Returns
    -------
    DataFrame
        Filled DataFrame

    """
    name=column_name
    
    if fitting_results is not None:
        fitted_df[name+'fitted_g2']=func(fitting_results.params, fitted_df.tau)
        fitted_df=fitted_df.reset_index(drop=True)
        fitted_df=fitted_df.set_index('tau', drop=True)
        fitted_df=fitted_df.drop(columns=[name])
        
    df2fill[name+'g2-1']=''
    for i in range(len(full_g2_df)):
        df2fill.loc[df2fill.index[i], name+'g2-1']=full_g2_df.loc[full_g2_df.index[i], name]
      
    if fitting_results is None:
        return df2fill
    else:
        return pd.concat([df2fill, fitted_df], axis=1)

def save_cI_data_in_single_df(cI_data_df, ROI_name, all_cI_data_df=None):
    """
    Saves all the cI data in a DataFrame

    Parameters
    ----------
    cI_data_df : DataFrame
                 cI data from the text file provided by Analysis
    ROI_name : str
               ROI name
    all_cI_data_df : DataFrame, optional
                     DataFrame with all cI data. The default is None.

    Returns
    -------
    all_cI_data_df : str
                     DataFrame with all cI data

    """
    ROI_array=[]
    cI_index=[]

    for i in range(len(cI_data_df)):
        ROI_array.append(ROI_name)
        cI_index.append(cI_data_df.index[i])

    combined_arrays=[ROI_array, cI_index]
    tup4multindex=list(zip(*combined_arrays))    
    index = pd.MultiIndex.from_tuples(tup4multindex, names=['ROI', 'ind'])
    
    cI_data_df.set_index(index, drop=True, inplace=True)
    
    if all_cI_data_df is None:
        all_cI_data_df=cI_data_df
        return all_cI_data_df
    else:
        all_cI_data_df=pd.concat([all_cI_data_df, cI_data_df], axis=0)
        return all_cI_data_df
    
def plot_cI(cI_data_df, x, filepath, title):
    """
    Plots and saves cI vs n or tau graphs

    Parameters
    ----------
    cI_data_df : DataFrame
                 Contains data presented as in the text files provided by Analysis
    x : str
        Variable cIs needs to be plotted against
    filepath : str
               Filepath to save the graphs
    title : str
            Graph title

    Returns
    -------
    None.

    """
    cI_vs_n =cI_data_df.plot(kind='line', x=x, y=cI_data_df.columns[2:len(cI_data_df.columns) - 1])
    if x=='n':
        cI_vs_n.set(ylabel='cI', title=title, ylim=[0,1])
    else:
        cI_vs_n.set(xlabel='tw (s)', ylabel='cI', title=title, ylim=[0,1])
    plt.savefig(filepath)

def plot_g2_1(g2_for_calc_df, data_name, title, filepath, data2fit=None, fit1=None):
    """
    Plots g2(tau) - 1 data and saves the figure

    Parameters
    ----------
    g2_for_calc_df : DataFrame
                     Contains all calculated g2 - 1 data
    data_name : str
                Name of the dataset in the DataFrame
    title : str
            Graph title
    filepath : str
               Filepath to save the image
    data2fit : DataFrame, optional
               Contains the data to fit. The default is None.
    fit1 : lmfit.minimizer.MinimizerResult, optional
           Results from fitting the data with a single stretched exp that has a baseline. The default is None.

    Returns
    -------
    None.

    """
    g2_vs_tau=g2_for_calc_df.plot(kind='scatter', x='tau', y=data_name)
            
    if data2fit is None:
        g2_vs_tau.set(xlabel = 'tau (s)', ylabel = 'g(2)(tau) - 1', title=title, xscale='log', xlim=[_config_data_analysis['lag_time']/2, max(g2_for_calc_df.tau)*2])
    else:       
        g2_vs_tau.plot(data2fit.tau, fit1)
        g2_vs_tau.set(xlabel = 'tau (s)', ylabel = 'g(2)(tau) - 1', title=title, xscale='log', xlim=[_config_data_analysis['lag_time']/2, max(g2_for_calc_df.tau)*2])
            
    plt.savefig(filepath)

def fit_data(params, data2fit, data_name):
    """
    Fits with func. The method allows the user to choose which parameters are free or fixed and involves the calculation of a Jacobian.

    Parameters
    ----------
    params : lmfit.parameter.Parameters
             Fitting parameters
    data2fit : DataFrame
               Contains the data to fit
    data_name : str
                Name of the column that contains the g2 - 1 data

    Returns
    -------
    out1 : lmfit.minimizer.MinimizerResult
           Fitting results
    fit1 : numpy.ndarray
           Y values from the model

    """
    init_params = Parameters()
    init_params.add('A', value=data2fit.iloc[0, len(data2fit.columns)-1])
    init_params.add('tauc', value=params['tauc'])
    init_params.add('beta', value=1, vary=False)
    init_params.add('y0',  value=0, vary=False)
    
    min0 = Minimizer(func, init_params, fcn_args=(np.float64(data2fit.tau),), fcn_kws={'data': np.float64(data2fit[data_name])})

    err1=False

    try:
        out0=min0.leastsq()
        print('Initial fit for ', data_name)
        print(fit_report(out0))
    except ValueError:
        err1=True 

    params4fitting = Parameters()
    if params['A'].vary == True and err1 == False:
        params4fitting.add('A', value=out0.params['A'].value)
    else:
        params4fitting.add('A', value=params['A'], vary=params['A'].vary)
        
    if params['tauc'].vary == True and err1 == False:
        params4fitting.add('tauc', value=out0.params['tauc'].value)
    else:
        params4fitting.add('tauc', value=params['tauc'], vary=params['tauc'].vary)
        
    if params['beta'].vary == True and err1 == False:
        params4fitting.add('beta', value=out0.params['beta'].value)
    else:
        params4fitting.add('beta', value=params['beta'], vary=params['beta'].vary)
        
    if params['y0'].vary == True and err1 == False:
        params4fitting.add('y0', value=out0.params['y0'].value)
    else:
        params4fitting.add('y0', value=params['y0'], vary=params['y0'].vary)
    
    min1 = Minimizer(func, params4fitting, fcn_args=(np.float64(data2fit.tau),), fcn_kws={'data': np.float64(data2fit[data_name])})
    
    try:
        out1=min1.leastsq()
        fit1 = func(out1.params, np.array(data2fit.tau))
    
        print('Obtained fitting parameters for ', data_name)
        print('Best fit: ')
        print(fit_report(out1))
                
        return out1, fit1
    except ValueError:
        out1='Issue when computing the residuals'
        fit1=0
        return out1, fit1

def get_g2_for_diff_tw(cI_data_df, ROI_tw_name, tw, delta_tw, g2_tw_df=None):
    """
    Calculates g2 - 1 for a given list of tw

    Parameters
    ----------
    cI_data_df : DataFrame
                 Contains the cI data as provided in the Analysis text file.
    ROI_tw_name : str
                  ROI000x_twy, where y is the value of tw
    tw : float
         tw value
    delta_tw : float
               Range of tw over which calculations are to be performed
    g2_tw_df : DataFrame, optional
               Contains the calculated g2 - 1 data. The default is None.

    Returns
    -------
    g2_tw_df : DataFrame
               Contains the calculated g2 - 1 data.

    """
    if g2_tw_df is None:
        temp_mean=cI_data_df.loc[tw: delta_tw + tw].mean(axis=0)
        temp_mean.rename(ROI_tw_name, inplace=True)
        g2_tw_df=pd.DataFrame(index=temp_mean.index)
        g2_tw_df=pd.concat([g2_tw_df, temp_mean], axis=1)
        return g2_tw_df
    else:
        temp_mean=cI_data_df.loc[tw: delta_tw + tw].mean(axis=0)
        temp_mean.rename(ROI_tw_name, inplace=True)
        g2_tw_df=pd.concat([g2_tw_df, temp_mean], axis=1)
        return g2_tw_df

def replace_data_in_df(full_df, data2replace):
    """
    Replaces the values in a column of a dataframe

    Parameters
    ----------
    full_df : DataFrame
              df where the values of a column needs to be replaced
    data2replace : Series
                   New values to put in the df. The name of the Series needs to be the same as that of the column where the values need replacing.

    Returns
    -------
    full_df : DataFrame
              Modified df

    """
    init_column_list=list(full_df.columns)
    full_df=full_df.drop([data2replace.name], axis=1)
    full_df=pd.concat([full_df, data2replace], axis=1)
    full_df=full_df[init_column_list]
    return full_df
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
 ###############################################   data analysis part  ###########################################################à   
if _config['data_analysis'] == True: 
    #Creates a list with the file names
    try :
        os.mkdir(_config_data_analysis['out_folder_path']+ '\\out\\')
    except FileExistsError:
        print('out folder already existing, do you want to overwrite?');
        answer = input("Enter y/n: ") 
        if answer == "y": 
            print("results will be overwritten");
            pass
        elif answer == "n": 
            print("change out folder path, restart the program");
            exit()
        else: 
            print("Please y or n") 
       
    file_name_list=list()
    ROI_name_list=list()
    ROI_list=[]
    for i in range(_config_data_analysis['cI_file_nber']):
        ROI_name_list.append(str(_config_data_analysis['cI_file_first_number'] + i).zfill(_config_data_analysis['cI_file_digits']))
        file_name_list.append(_config_data_analysis['cI_file_folder_path'] + '\\' + _config_data_analysis['cI_file_name'] + ROI_name_list[i] + _config_data_analysis['cI_file_extension'])
        ROI_list.append('ROI' + ROI_name_list[i])
  
#Creates a DataFrame to store file names
    process_index=[]
    for i in range(len(ROI_list)):
        process_index.append('all')
    combined_indexes=[ROI_list, process_index]
    tup4multindex=list(zip(*combined_indexes))    
    index = pd.MultiIndex.from_tuples(tup4multindex, names=['ROI', 'processed data'])
    image_names_df=pd.DataFrame(index=index, columns=['cI_vs_n', 'cI_vs_t', 'g2_1', 'fitted_g2_1', 'status'])
  
#Plots raw data and calculates g2(tau) - 1

    g2_df=pd.DataFrame(dtype=float)
 
    for i in range(_config_data_analysis['cI_file_nber']):
        ##Loads the data of ROI cI_file_first_number + i
        cI_data_df=pd.read_csv(file_name_list[i], sep='\\t', engine='python')
        #Calculates tau
        if list(cI_data_df.columns)[0] == 'tsec':
            cI_data_df['tau'] = list(cI_data_df.tsec)
            n_exist = False
        else:
            cI_data_df['tau']=cI_data_df.n * _config_data_analysis['lag_time']
            n_exist = True 
        #Plots cI vs n    
        if n_exist == True:
            image_names_df.iloc[(i,0)]=_config_data_analysis['out_folder_path']+ '\\out\\' + _config_data_analysis['cI_file_name'] + ROI_name_list[i] + '_cI_vs_n.png'    
            plot_cI(cI_data_df, 'n', image_names_df.iloc[(i,0)], ROI_list[i])    
    
        #Plots cI vs tau
        image_names_df.iloc[(i,1)]=_config_data_analysis['out_folder_path']+ '\\out\\' + _config_data_analysis['cI_file_name'] + ROI_name_list[i] + '_cI_vs_tw.png'
        plot_cI(cI_data_df, 'tau', image_names_df.iloc[(i,1)], ROI_list[i])
   
        #Calculates g2(tau) - 1, averages I and computes SE(I)      
        g2_series=cI_data_df.mean(axis=0)
    
        g2_series['SE_Iav']=cI_data_df.iloc[:, 1].sem()
    
        g2_series=g2_series.rename(ROI_list[i])
    
        g2_df=pd.concat([g2_df, g2_series], axis=1)
        print(i)
        if i ==0:
            all_cI_data_df=save_cI_data_in_single_df(cI_data_df, ROI_list[i])
        else:
            all_cI_data_df=save_cI_data_in_single_df(cI_data_df, ROI_list[i], all_cI_data_df)
    
        if i != _config_data_analysis['cI_file_nber'] - 1:
            del cI_data_df
    
        del g2_series

    #Adds a column with the tau values in the DataFrame containing g2 - 1 data
    g2_for_calc_df=g2_df[2:len(g2_df)-2]
    tau_list_df=pd.DataFrame(get_tau_list(cI_data_df, _config_data_analysis['lag_time']), g2_for_calc_df.index, columns = ['tau'])
    g2_for_calc_df=pd.concat([g2_for_calc_df, tau_list_df], axis=1)
    g2_for_calc_df.drop(['d0'], inplace=True)
    del tau_list_df

    new_indexes=[]
    new_indexes.append('tau')
    for i in range(len(g2_df.columns)):
        new_indexes.append(ROI_list[i])
    
    g2_for_calc_df=g2_for_calc_df.reindex(columns=new_indexes)

    del cI_data_df

    #Plots and fits g2(tau) - 1 data
    fit_method = 'dogleg'
    fitting_param_names=['A','tauc','beta','y0']

    #Creates an empty DataFrame to save the fitting data
    fitting_results_df=pd.DataFrame(index=['processed data', 'twi', 'twf', 'Nptspdecade', 'Iav_value', 'Iav_SE', 'A_value', 'A_SE', 'tauc_value', 'tauc_SE', 'beta_value', 'beta_SE', 'y0_value', 'y0_SE','tau_min', 'tau_max', 'iter_nber', 'chi2', 'red_chi2', 'success', 'message'])

    #Creates and empty DataFrame to save the data corresponding to g2 - 1 fitted data
    g2_fitting_data=pd.DataFrame(index=g2_for_calc_df.tau)

    #Initialises parameters for later fitting
    params = Parameters()
    params.add('A', value=0.7)
    params.add('tauc', value=200)
    params.add('beta', value=0.7)
    params.add('y0',  value=0)
    for i in range(_config_data_analysis['cI_file_nber']):
        #Checks whether the user wants to process the data
        if g.buttonbox('Do you want to process the data?', image=image_names_df.iloc[(i,1)], choices=['Yes', 'No']) == 'No':
            image_names_df.iloc[(i,4)]='cI data not suitable for further calculations'
            continue
            
        #Checks how the user wants to process the data
        fitting_method = g.buttonbox('How do you want to process the g2 - 1 data?', image=image_names_df.iloc[(i,1)], choices=['Across all lag times', 'Across a range of lag times', 'For several log-spaced ageing times'])
        
        if fitting_method=='Across all lag times':    
            #Plots the data and saves the obtained figure
            image_names_df.iloc[(i,2)]=_config_data_analysis['out_folder_path'] + '\\out\\' + cI_file_name + ROI_name_list[i] + '_g2-1_vs_tau.png'
            plot_g2_1(g2_for_calc_df, ROI_list[i], ROI_list[i], image_names_df.iloc[(i,2)])
            
            
            #Checks whether the user wants to fit the g2 - 1 data
            if g.buttonbox('Do you want to fit g2 - 1 data?', image=image_names_df.iloc[(i,2)], choices=['Yes', 'No']) == 'No':
                image_names_df.iloc[(i,4)]='reason for not fitting the data' + g.enterbox('Why do you not want to fit the g2 - 2 data.')
                g2_fitting_data=g2_fitting_data_in_df(g2_for_calc_df, g2_fitting_data, ROI_list[i])
                continue   
                
            keep_fitting=True
            
            tau_fitting_range=[]
            tau_fitting_range.append(0)
            tau_fitting_range.append(len(g2_for_calc_df)-1)
            
            #Fits the data until the user is happy
            while keep_fitting==True:
                #Set-up a DataFrame with the data to fit
                data2fit=g2_for_calc_df.iloc[tau_fitting_range[0]:tau_fitting_range[1]+1, [0, i+1]]
                
                #Uses a Jacobian that is numerically determined to fit the data
                [out1, fit1]=fit_data(params, data2fit, ROI_list[i])
                
                if type(fit1) == np.ndarray:
                    image_names_df.iloc[(i,3)]=_config_data_analysis['out_folder_path'] + '\\out\\' + _config_data_analysis['cI_file_name'] + ROI_name_list[i] + '_fitted_g2-1_vs_tau.png'
                    plot_g2_1(g2_for_calc_df, ROI_list[i], ROI_list[i], image_names_df.iloc[(i,3)], data2fit, fit1)
                 
                    text=out1.message + ' Are you happy with the fit?'
                    choice = g.buttonbox(text, image=image_names_df.iloc[(i,3)], choices=['Yes', 'No'])
                        
                    if choice == 'Yes':
                        fitting_data_ser=fitting_data_in_series('all', g2_df, out1, min(data2fit.tau), max(data2fit.tau), ROI_list[i])
                        fitting_results_df=pd.concat([fitting_results_df, fitting_data_ser], axis=1)
                        g2_fitting_data=g2_fitting_data_in_df(g2_for_calc_df, g2_fitting_data, ROI_list[i], out1, data2fit)
                        image_names_df.iloc[(i,4)]='data fitted'
                        del out1, fit1, data2fit, fitting_data_ser            
                        break
                    elif choice == 'No':
                        further_fitting=g.buttonbox('Do you want to keep fitting the data?', choices=['Yes, change initialisation values', 'Yes, change tau fitting range', 'Yes, change both', 'No'])
                elif type(out1) == str:
                    text=out1 + '. Do you want to try fitting the data again?'
                    further_fitting=g.buttonbox(text, choices=['Yes, change initialisation values', 'Yes, change tau fitting range', 'Yes, change both', 'No'])            
                
                if further_fitting=='No':    
                    image_names_df.iloc[(i,4)]='reason for not being happy with the fit: ' + g.enterbox('Why do you not want to fit the g2 - 2 data.')
                    g2_fitting_data=g2_fitting_data_in_df(g2_for_calc_df, g2_fitting_data, ROI_list[i])
                    del out1, fit1, data2fit
                    break
                
                if further_fitting=='Yes, change initialisation values' or further_fitting=='Yes, change both':
                    msg = "Fitting function: (A exp(-(tau/tc)^beta)^2 + y0"
                    title = "Initialisation of the variables for fitting"
                    fieldNames = ['A', 'Fixed? (Y/N)', 'tc', 'Fixed? (Y/N)', 'beta', 'Fixed? (Y/N)', 'y0', 'Fixed? (Y/N)']
                    fieldValues = g.multenterbox(msg, title, fieldNames)
                
                    params = Parameters()
                
                    for j in range(0,len(fieldValues),2):
                        if fieldValues[j+1]=='Y':
                            params.add(fitting_param_names[int(j/2)], value=float(fieldValues[j]), vary=False)
                        else:
                            params.add(fitting_param_names[int(j/2)], value=float(fieldValues[j]))
            
                if further_fitting=='Yes, change tau fitting range' or further_fitting=='Yes, change both':
                    msg1 = 'Enter the indexes of the first and the last points to be in included in the fit below (N.B: very first and very last point indexes are 0 and ' + str(len(g2_for_calc_df) - 1) + ', respectively)'
                    title1 = "tau fitting range"
                    fieldNames1 = ['first point', 'last point']
                    fieldValues1 = g.multenterbox(msg1, title1, fieldNames1)
                    
                    for j in range(2):
                        tau_fitting_range[j]=int(fieldValues1[j])
    
        if fitting_method=='Across a range of lag times':
            keep_asking_for_tau_range=True
            g2_fitting_data=g2_fitting_data_in_df(g2_for_calc_df, g2_fitting_data, ROI_list[i])
            
            while keep_asking_for_tau_range==True:
                msg2 = ('Enter the lowest and highest values for n (N.B: min and max n values are 1 and ' + str(max(all_cI_data_df.n)) + ',respectively). If you do not want to fit another range, press Cancel.')
                title2 = "n processing range"
                fieldNames2 = ['Lowest n', 'Highest n']
                fieldValues2 = g.multenterbox(msg2, title2, fieldNames2)
                
                if fieldValues2 is None:
                    break
                
                #Stores the data to process in a DataFrame called cI_data_df
                n_range=[]
                for j in range(2):
                    n_range.append(int(fieldValues2[j]))
                    
                cI_data_df=all_cI_data_df[ROI_list[i]:ROI_list[i]]
                cI_data_df=cI_data_df.reset_index(drop=True)
                
                cI_data_df=cI_data_df.iloc[n_range[0]-1:n_range[1]]
                cI_data_df.reset_index(drop=True, inplace=True)
                tw=cI_data_df.tau[0]
                ROI_tw_name=ROI_list[i] + '_tw' + str(tw)
                
                #Creates a temporary DataFrame to store the names of the images
                tup=[(ROI_list[i], 'tw' + str(tw))]
                index_temp = pd.MultiIndex.from_tuples(tup, names=['ROI', 'processed data'])
                image_names_temp_df=pd.DataFrame(index=index_temp, columns=['cI_vs_n', 'cI_vs_t', 'g2_1', 'fitted_g2_1', 'status'])
                
                #Plots and saves the cI vs n data
                image_names_temp_df.iloc[(0, 0)]=_config_data_analysis['out_folder_path']  + '\\out\\' + _config_data_analysis['cI_file_name'] + ROI_name_list[i] + '_cI_vs_n_tw' + str(tw) + '.png'
                plot_cI(cI_data_df, 'n', image_names_temp_df.iloc[(0,0)], ROI_tw_name)
                
                #Plots and saves the cI vs tw data
                image_names_temp_df.iloc[(0, 1)]=_config_data_analysis['out_folder_path'] + '\\out\\' + _config_data_analysis['cI_file_name']  + ROI_name_list[i] + '_cI_vs_tw_tw' + str(tw) + '.png'
                plot_cI(cI_data_df, 'tau', image_names_temp_df.iloc[(0,1)], ROI_tw_name)            
                
                #Calculates g2(tau) - 1, averages I and computes SE(I)  
                g2_series=cI_data_df.mean(axis=0)
                g2_series['SE_Iav']=cI_data_df.Iav.sem()
                g2_series=g2_series.rename(ROI_tw_name)
        
                g2_df=pd.concat([g2_df, g2_series], axis=1)
                
                g2_for_calc_df=pd.concat([g2_for_calc_df, g2_series.iloc[3:len(g2_series)-2]], axis=1)
                
                del g2_series
                
                #Plots the data and saves the obtained figure
                image_names_temp_df.iloc[(0,2)]=_config_data_analysis['out_folder_path']  + '\\out\\' + _config_data_analysis['cI_file_name']  + ROI_name_list[i] + '_g2-1_vs_tau_tw' + str(tw) + '.png'
                plot_g2_1(g2_for_calc_df, ROI_tw_name, ROI_tw_name, image_names_temp_df.iloc[(0,2)])
            
            
                #Checks whether the user wants to fit the g2 - 1 data
                if g.buttonbox('Do you want to fit g2 - 1 data?', image=image_names_temp_df.iloc[(0,2)], choices=['Yes', 'No']) == 'No':
                    image_names_temp_df.iloc[(0,4)]='reason for not fitting the data: ' + g.enterbox('Why do you not want to fit the g2 - 2 data.')
                    g2_fitting_data=g2_fitting_data_in_df(g2_for_calc_df, g2_fitting_data, ROI_tw_name)
                    image_names_df=pd.concat([image_names_df, image_names_temp_df])
                    del image_names_temp_df
                    continue
                
                keep_fitting=True
                
                tau_fitting_range=[]
                tau_fitting_range.append(0)
                tau_fitting_range.append(len(g2_for_calc_df)-1)
                
                #Fits the data until the user is happy
                while keep_fitting==True:
                    #Set-up a DataFrame with the data to fit
                    data2fit=g2_for_calc_df.iloc[tau_fitting_range[0]:tau_fitting_range[1]+1, [0,len(g2_for_calc_df.columns)-1]]
                    data2fit.dropna(inplace=True)
                    
                    #Uses a Jacobian that is numerically determined to fit the data
                    [out1, fit1]=fit_data(params, data2fit, ROI_tw_name)
                    
                    if type(fit1) == np.ndarray:
                        image_names_temp_df.iloc[(0,3)]=_config_data_analysis['out_folder_path'] + '\\out\\' + _config_data_analysis['cI_file_name']  + ROI_name_list[i] + '_fitted_g2-1_vs_tau_tw' + str(tw) + '.png'
                        plot_g2_1(g2_for_calc_df, ROI_tw_name, ROI_tw_name, image_names_temp_df.iloc[(0,3)], data2fit, fit1)
                        
                        text=out1.message + ' Are you happy with the fit?'
                        choice = g.buttonbox(text, image=image_names_temp_df.iloc[(0,3)], choices=['Yes', 'No'])
                        
                        if choice == 'Yes':
                            fitting_data_ser=fitting_data_in_series('across a range of lag times', g2_df, out1, min(data2fit.tau), max(data2fit.tau), ROI_tw_name, tw, max(cI_data_df.tau))
                            fitting_results_df=pd.concat([fitting_results_df, fitting_data_ser], axis=1)
                            g2_fitting_data=g2_fitting_data_in_df(g2_for_calc_df, g2_fitting_data, ROI_tw_name, out1, data2fit)
                            image_names_temp_df.iloc[(0,4)]='data fitted'
                            image_names_df=pd.concat([image_names_df, image_names_temp_df])
                            del out1, fit1, data2fit, fitting_data_ser, image_names_temp_df           
                            break
                        elif choice == 'No':
                            further_fitting=g.buttonbox('Do you want to keep fitting the data?', choices=['Yes, change initialisation values', 'Yes, change tau fitting range', 'Yes, change both', 'No'])
                    elif type(out1) == str:
                        text=out1 + '. Do you want to try fitting the data again?'
                        further_fitting=g.buttonbox(text, choices=['Yes, change initialisation values', 'Yes, change tau fitting range', 'Yes, change both', 'No'])
                    
                    if further_fitting=='No':    
                        image_names_temp_df.iloc[(0,4)]='reason for not being happy with the fit: ' + g.enterbox('Why do you not want to fit the g2 - 2 data.')
                        image_names_df=pd.concat([image_names_df, image_names_temp_df])
                        g2_fitting_data=g2_fitting_data_in_df(g2_for_calc_df, g2_fitting_data, ROI_tw_name)
                        del out1, fit1, data2fit, image_names_temp_df
                        break
                    
                    if further_fitting=='Yes, change initialisation values' or further_fitting=='Yes, change both':
                        msg = "Fitting function: (A exp(-(tau/tc)^beta)^2 + y0"
                        title = "Initialisation of the variables for fitting"
                        fieldNames = ['A', 'Fixed? (Y/N)', 'tc', 'Fixed? (Y/N)', 'beta', 'Fixed? (Y/N)', 'y0', 'Fixed? (Y/N)']
                        fieldValues = g.multenterbox(msg, title, fieldNames)
                    
                        params = Parameters()
                    
                        for j in range(0,len(fieldValues),2):
                            if fieldValues[j+1]=='Y':
                                params.add(fitting_param_names[int(j/2)], value=float(fieldValues[j]), vary=False)
                            else:
                                params.add(fitting_param_names[int(j/2)], value=float(fieldValues[j]))
                
                    if further_fitting=='Yes, change tau fitting range' or further_fitting=='Yes, change both':
                        msg1 = 'Enter the indexes of the first and the last points to be in included in the fit below (N.B: very first and very last point indexes are 0 and ' + str(len(g2_for_calc_df) - 1) + ', respectively)'
                        title1 = "tau fitting range"
                        fieldNames1 = ['first point', 'last point']
                        fieldValues1 = g.multenterbox(msg1, title1, fieldNames1)
                        
                        for j in range(2):
                            tau_fitting_range[j]=int(fieldValues1[j])
                            
        if fitting_method=='For several log-spaced ageing times':
            g2_fitting_data=g2_fitting_data_in_df(g2_for_calc_df, g2_fitting_data, ROI_list[i])
            
            #Asks the user to enter the desired parameters to calculate the log-spaced tw values
            msg3 = ('Enter the values of the requested parameters below (N.B: min and max tw values are ' + str(all_cI_data_df.tau.iloc[0]) +' and ' + str(max(all_cI_data_df.tau)) + ',respectively).')
            title3 = "Parameters for log-spaced tw calculations"
            fieldNames3 = ['Lowest tw', 'delta tw', 'N(tw)/decade']
            fieldValues3 = g.multenterbox(msg3, title3, fieldNames3)
            
            #Transforms the collected data into usable data
            tw=float(fieldValues3[0])
            delta_tw=float(fieldValues3[1])
            N_tw_p_dec=float(fieldValues3[2])
            
            max_tw=max(all_cI_data_df.tau)
            
            #Determines the number of points corresponding to that requested by the
            #user. This number of points corresponds to the number of points required
            #to build an array starting from 10^x, where 10^x is the next inferior
            #10^x to tw, and ending at 10^y, where 10^y is the next superior 10^y
            #to max_tw
            Npts = int((((int(math.log(max_tw, 10)) + 1) - int(math.log(tw, 10)))) * N_tw_p_dec + 1)
                                
            #Gets the cI data
            cI_data_df=all_cI_data_df[ROI_list[i]:ROI_list[i]]
            cI_data_df=cI_data_df.reset_index(drop=True)
            
            #Builds a logspaced array containing Npts, starting at 10^x and finishing
            #at 10^y (see previous comment)
            logspaced_tw=np.geomspace(10**int(math.log(tw, 10)), 10**(int(math.log(max_tw, 10)) + 1), Npts)
            
            #Adds tw if not in the list of logspaced tw values
            if tw not in logspaced_tw:
                logspaced_tw=np.insert(logspaced_tw, obj=[0], values = [tw])
            
            #Creates a list with "logspaced" tw values. It starts from tw. For the
            #lowest values, only the values allowing the distance between two successive
            #tw values to be at least delta_tw are kept.
            logspaced_tw_list=[]
            logspaced_tw_list.append(tw)
            
            for j in range(len(logspaced_tw)):
                if logspaced_tw[j] <= tw:
                    pass
                elif logspaced_tw[j] - max(logspaced_tw_list) > delta_tw:
                    if logspaced_tw[j] + delta_tw > max_tw:
                        break
                    else:
                        logspaced_tw_list.append(cI_data_df.tau.iloc[int(logspaced_tw[j]//lag_time)]-1)
    
            
            #Creates a temporary DataFrame to store the names of the images
            ROI_ind=[]
            tw_ind=[]
            ROI_tw_list=[]
            
            for j in range(len(logspaced_tw_list)):
                ROI_ind.append(ROI_list[i])
                tw_ind.append(logspaced_tw_list[j])
                ROI_tw_list.append(ROI_list[i] + '_tw' + str(logspaced_tw_list[j]))
                
            combined_arrays=[ROI_ind, tw_ind]
            tup4multindex=list(zip(*combined_arrays))    
            index = pd.MultiIndex.from_tuples(tup4multindex, names=['ROI', 'tw'])
            
            image_names_temp_df=pd.DataFrame(index=index, columns=['cI_vs_n', 'cI_vs_t', 'g2_1', 'fitted_g2_1', 'status'])
            
            for j in range(len(image_names_temp_df)):
                image_names_temp_df.iloc[j,0]=cI_file_folder_path + '\\out\\' + cI_file_name + ROI_name_list[i] + '_cI_vs_tw_list_logspaced.png'
                image_names_temp_df.iloc[j,1]=cI_file_folder_path + '\\out\\' + cI_file_name + ROI_name_list[i] + '_cI_vs_tw_list_linspaced.png'
            
            #Creates a colour array to plot the vertical lines with different colours
            #as well as a linestyle array for plotting the raw data and their fits
            #at the end of the fitting process
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            linestyles=['-', '--', '-.', ':']
            counter=0
            if len(colors) < len(logspaced_tw_list):
                temp_colors=[]
                for j in range(len(logspaced_tw_list)):
                    temp_colors.append(colors[counter])
                    if counter < len(colors) - 1:
                        counter +=1
                    else:
                        counter=0
                colors=temp_colors
            
            counter=0
            if len(linestyles) < len(logspaced_tw_list):
                temp_linestyles=[]
                for j in range(len(logspaced_tw_list)):
                    temp_linestyles.append(linestyles[counter])
                    if counter < len(linestyles) - 1:
                        counter +=1
                    else:
                        counter=0
                linestyles=temp_linestyles
            
            #Plots the cI data on a logscale and represents tw values and the ranges
            #over which g2 - 1 data are going to be computed for each tw
            cI_vs_tw =cI_data_df.plot(kind='line', x='tau', y=cI_data_df.columns[2:len(cI_data_df.columns) - 1])
            
            title=ROI_list[i] + '_selected_tw'
            cI_vs_tw.set(xlabel='tw (s)', ylabel='cI', title=title, ylim=[0,1], xscale='log', xlim=[lag_time/2, max(g2_for_calc_df.tau)*2])
    
            for j in range(len(logspaced_tw_list)):
                cI_vs_tw.axvline(logspaced_tw_list[j], ls='--', color=colors[j])
                cI_vs_tw.axvline(logspaced_tw_list[j] + delta_tw , ls='--', color=colors[j])
                
            plt.savefig(image_names_temp_df.iloc[j,0])
            
            #Plots the cI data on a linscale and represents tw values and the ranges
            #over which g2 - 1 data are going to be computed for each tw
            cI_vs_tw =cI_data_df.plot(kind='line', x='tau', y=cI_data_df.columns[2:len(cI_data_df.columns) - 1])
            
            title=ROI_list[i] + '_selected_tw'
            cI_vs_tw.set(xlabel='tw (s)', ylabel='cI', title=title, ylim=[0,1])
    
            for j in range(len(logspaced_tw_list)):
                cI_vs_tw.axvline(logspaced_tw_list[j], ls='--', color=colors[j])
                cI_vs_tw.axvline(logspaced_tw_list[j] + delta_tw, ls='--', color=colors[j])
                
            plt.savefig(image_names_temp_df.iloc[j,1])
    
            cI_data_df=cI_data_df.set_index(cI_data_df.tau, drop=True)
    
            #Calculates the values of g2(tau) - 1 for each tw and stores them in the
            #dataframe g2_tw_df; SE(Iav) is also computed and stored in the DataFrame
            SE_Iav_df=pd.DataFrame(index=['SE_Iav'], columns=ROI_tw_list)        
            for k in range(len(logspaced_tw_list)):
                if k == 0:
                    g2_tw_df = get_g2_for_diff_tw(cI_data_df, ROI_tw_list[k], logspaced_tw_list[k], delta_tw)
                    SE_Iav_df.loc['SE_Iav', ROI_tw_list[k]]=cI_data_df.Iav.loc[tw: delta_tw + tw].sem(axis=0)
                else:
                    g2_tw_df = get_g2_for_diff_tw(cI_data_df, ROI_tw_list[k], logspaced_tw_list[k], delta_tw, g2_tw_df)
                    SE_Iav_df.loc['SE_Iav', ROI_tw_list[k]]=cI_data_df.Iav.loc[tw: delta_tw + tw].sem(axis=0)
    
            g2_tw_df=pd.concat([g2_tw_df, SE_Iav_df], axis=0)        
            
            #Adds a column with the tau values in the DataFrame containing g2 - 1 data
            g2_tw_for_calc_df=g2_tw_df[2:len(g2_df)-2]
            tau_list_df=pd.DataFrame(get_tau_list(cI_data_df, lag_time), g2_tw_for_calc_df.index, columns = ['tau'])
            g2_tw_for_calc_df=pd.concat([g2_tw_for_calc_df, tau_list_df], axis=1)
            g2_tw_for_calc_df.drop(['d0'], inplace=True)
            del tau_list_df
    
            new_indexes=[]
            new_indexes.append('tau')
            for k in range(len(g2_tw_df.columns)):
                new_indexes.append(ROI_tw_list[k])
        
            g2_tw_for_calc_df=g2_tw_for_calc_df.reindex(columns=new_indexes)
            refit_list=[]
            
            #Fits the g2(tau) - 1 data according to the user's indications
            for k in range(len(logspaced_tw_list)):
                #Plots the data and saves the obtained figure
                image_names_temp_df.iloc[(k,2)]=cI_file_folder_path + '\\out\\' + cI_file_name + ROI_name_list[i] + '_g2-1_vs_tau_tw' + str(logspaced_tw_list[k]) + '.png'
                plot_g2_1(g2_tw_for_calc_df, ROI_tw_list[k], ROI_tw_list[k], image_names_temp_df.iloc[(k,2)])
                refit_list.append('')
            
            one_data_set_fitted=False
            refit=True
            
            while refit == True:
                for k in range(len(logspaced_tw_list)):
                    #Checks whether the user wants to fit the g2 - 1 data
                    if one_data_set_fitted == False or refit_list[k] == 'yes':
                        if g.buttonbox('Do you want to fit g2 - 1 data?', image=image_names_temp_df.iloc[(k,2)], choices=['Yes', 'No']) == 'No':
                            image_names_temp_df.iloc[(k,4)]='reason for not fitting the data: ' + g.enterbox('Why do you not want to fit the g2 - 2 data.')
                            g2_fitting_data=g2_fitting_data_in_df(g2_tw_for_calc_df, g2_fitting_data, ROI_tw_list[k])
                            continue
                        
                        keep_fitting=True
                        
                        tau_fitting_range=[]
                        tau_fitting_range.append(0)
                        tau_fitting_range.append(len(g2_tw_for_calc_df)-1)
                        
                        #Fits the data until the user is happy
                        while keep_fitting==True:
                            #Set-up a DataFrame with the data to fit
                            data2fit=g2_tw_for_calc_df.iloc[tau_fitting_range[0]:tau_fitting_range[1]+1, [0,k+1]]
                            data2fit.dropna(inplace=True)
                            
                            #Uses a Jacobian that is numerically determined to fit the data
                            [out1, fit1]=fit_data(params, data2fit, ROI_tw_list[k])
                            if type(fit1) == np.ndarray:
                                image_names_temp_df.iloc[(k,3)]=cI_file_folder_path + '\\out\\' + cI_file_name + ROI_name_list[i] + '_fitted_g2-1_vs_tau_tw' + str(logspaced_tw_list[k]) + '.png'
                                plot_g2_1(g2_tw_for_calc_df, ROI_tw_list[k], ROI_tw_list[k], image_names_temp_df.iloc[(k,3)], data2fit, fit1)
                            
                                text=out1.message + ' Are you happy with the fit?'
                                choice = g.buttonbox(text, image=image_names_temp_df.iloc[(k,3)], choices=['Yes', 'No'])
                                
                                if choice == 'Yes':
                                    fitting_data_ser=fitting_data_in_series('logspaced tw', g2_tw_df, out1, min(data2fit.tau), max(data2fit.tau), ROI_tw_list[k], logspaced_tw_list[k], logspaced_tw_list[k] + delta_tw, N_tw_p_dec)
                                    if ROI_tw_list[k] in fitting_results_df.columns:
                                        fitting_results_df=replace_data_in_df(fitting_results_df, fitting_data_ser)
                                        g2_fitted_data_ser=pd.Series(fit1, index=data2fit.tau)
                                        g2_fitted_data_ser=g2_fitted_data_ser.rename(ROI_tw_list[k] + 'fitted_g2')
                                        g2_fitting_data=replace_data_in_df(g2_fitting_data, g2_fitted_data_ser)
                                    else:
                                        fitting_results_df=pd.concat([fitting_results_df, fitting_data_ser], axis=1)
                                        g2_fitting_data=g2_fitting_data_in_df(g2_tw_for_calc_df, g2_fitting_data, ROI_tw_list[k], out1, data2fit)
                                    image_names_temp_df.iloc[(k,4)]='data fitted'
                                    one_data_set_fitted=True
                                    del out1, fit1, data2fit, fitting_data_ser
                                    break
                                elif choice == 'No':
                                    further_fitting=g.buttonbox('Do you want to keep fitting the data?', choices=['Yes, change initialisation values', 'Yes, change tau fitting range', 'Yes, change both', 'No'])
                                    
                            elif type(out1) == str:
                                text=out1 + '. Do you want to try fitting the data again?'
                                further_fitting=g.buttonbox(text, choices=['Yes, change initialisation values', 'Yes, change tau fitting range', 'Yes, change both', 'No'])
                            
                            if further_fitting=='No':    
                                image_names_temp_df.iloc[(k,4)]='reason for not being happy with the fit: ' + g.enterbox('Why do you not want to fit the g2 - 2 data.')
                                g2_fitting_data=g2_fitting_data_in_df(g2_tw_for_calc_df, g2_fitting_data, ROI_tw_list[k])
                                del out1, fit1, data2fit
                                break
                            
                            if further_fitting=='Yes, change initialisation values' or further_fitting=='Yes, change both':
                                msg = "Fitting function: (A exp(-(tau/tc)^beta)^2 + y0"
                                title = "Initialisation of the variables for fitting"
                                fieldNames = ['A', 'Fixed? (Y/N)', 'tc', 'Fixed? (Y/N)', 'beta', 'Fixed? (Y/N)', 'y0', 'Fixed? (Y/N)']
                                fieldValues = g.multenterbox(msg, title, fieldNames)
                            
                                params = Parameters()
                            
                                for j in range(0,len(fieldValues),2):
                                    if fieldValues[j+1]=='Y':
                                        params.add(fitting_param_names[int(j/2)], value=float(fieldValues[j]), vary=False)
                                    else:
                                        params.add(fitting_param_names[int(j/2)], value=float(fieldValues[j]))
                        
                            if further_fitting=='Yes, change tau fitting range' or further_fitting=='Yes, change both':
                                msg1 = 'Enter the indexes of the first and the last points to be in included in the fit below (N.B: very first and very last point indexes are 0 and ' + str(len(g2_tw_for_calc_df) - 1) + ', respectively)'
                                title1 = "tau fitting range"
                                fieldNames1 = ['first point', 'last point']
                                fieldValues1 = g.multenterbox(msg1, title1, fieldNames1)
                                
                                for j in range(2):
                                    tau_fitting_range[j]=int(fieldValues1[j])
                    elif refit_list[k] == '':
                        tau_fitting_range=[]
                        tau_fitting_range.append(0)
                        tau_fitting_range.append(len(g2_tw_for_calc_df)-1)
                        
                        #Set-up a DataFrame with the data to fit
                        data2fit=g2_tw_for_calc_df.iloc[tau_fitting_range[0]:tau_fitting_range[1]+1, [0,k+1]]
                        data2fit.dropna(inplace=True)
                            
                        #Uses a Jacobian that is numerically determined to fit the data
                        [out1, fit1]=fit_data(params, data2fit, ROI_tw_list[k])
                        if type(fit1) == np.ndarray:
                            image_names_temp_df.iloc[(k,3)]=cI_file_folder_path + '\\out\\' + cI_file_name + ROI_name_list[i] + '_fitted_g2-1_vs_tau_tw' + str(logspaced_tw_list[k]) + '.png'
                            plot_g2_1(g2_tw_for_calc_df, ROI_tw_list[k], ROI_tw_list[k], image_names_temp_df.iloc[(k,3)], data2fit, fit1)
                            fitting_data_ser=fitting_data_in_series('logspaced tw', g2_tw_df, out1, min(data2fit.tau), max(data2fit.tau), ROI_tw_list[k], logspaced_tw_list[k], logspaced_tw_list[k] + delta_tw, N_tw_p_dec)
                            fitting_results_df=pd.concat([fitting_results_df, fitting_data_ser], axis=1)
                            g2_fitting_data=g2_fitting_data_in_df(g2_tw_for_calc_df, g2_fitting_data, ROI_tw_list[k], out1, data2fit)
                            image_names_temp_df.iloc[(k,4)]='data fitted'
                            one_data_set_fitted=True
                            del out1, fit1, data2fit, fitting_data_ser
                            continue           
                        
                        elif type(out1) == str:
                            text=out1 + '. Do you want to try fitting the data again?'
                            further_fitting=g.buttonbox(text, choices=['Yes, change initialisation values', 'Yes, change tau fitting range', 'Yes, change both', 'No'])
                            
                        if further_fitting=='No':    
                            image_names_temp_df.iloc[(k,4)]='reason for not being happy with the fit: ' + g.enterbox('Why do you not want to fit the g2 - 2 data.')
                            g2_fitting_data=g2_fitting_data_in_df(g2_tw_for_calc_df, g2_fitting_data, ROI_tw_list[k])
                            del out1, fit1, data2fit
                            continue
                            
                        if further_fitting=='Yes, change initialisation values' or further_fitting=='Yes, change both':
                            msg = "Fitting function: (A exp(-(tau/tc)^beta)^2 + y0"
                            title = "Initialisation of the variables for fitting"
                            fieldNames = ['A', 'Fixed? (Y/N)', 'tc', 'Fixed? (Y/N)', 'beta', 'Fixed? (Y/N)', 'y0', 'Fixed? (Y/N)']
                            fieldValues = g.multenterbox(msg, title, fieldNames)
                            
                            params = Parameters()
                            
                            for j in range(0,len(fieldValues),2):
                                if fieldValues[j+1]=='Y':
                                    params.add(fitting_param_names[int(j/2)], value=float(fieldValues[j]), vary=False)
                                else:
                                    params.add(fitting_param_names[int(j/2)], value=float(fieldValues[j]))
                        
                        if further_fitting=='Yes, change tau fitting range' or further_fitting=='Yes, change both':
                            msg1 = 'Enter the indexes of the first and the last points to be in included in the fit below (N.B: very first and very last point indexes are 0 and ' + str(len(g2_tw_for_calc_df) - 1) + ', respectively)'
                            title1 = "tau fitting range"
                            fieldNames1 = ['first point', 'last point']
                            fieldValues1 = g.multenterbox(msg1, title1, fieldNames1)
                                
                            for j in range(2):
                                tau_fitting_range[j]=int(fieldValues1[j])
                                
                #Plots both the raw and the fitted data                
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                legend=[]
        
                for j in range(len(logspaced_tw_list)):
                    ax.plot(g2_fitting_data.index, g2_fitting_data[ROI_tw_list[j] + 'g2-1'], 'o', color=colors[j])
                    legend.append('tw' + str(logspaced_tw_list[j]) + 'raw')
                    
                    if ROI_tw_list[j] + 'fitted_g2' in g2_fitting_data.columns:       
                        ax.plot(g2_fitting_data.index, g2_fitting_data[ROI_tw_list[j] + 'fitted_g2'], color=colors[j], linestyle=linestyles[j])
                        if fitting_results_df.loc['success', ROI_tw_list[j]] == True:
                            legend.append('tw' + str(logspaced_tw_list[j]) + 'sf')
                            refit_list[j]='no'
                        else:
                            legend.append('tw' + str(logspaced_tw_list[j]) + 'uf *')
                
                plt.xscale('log')
                plt.xlabel('tau (s)')
                plt.ylabel('g(2)(tau) - 1')
                plt.title(ROI_list[i] + ' all log_tw_spaced IC data')
                
                ax.set_xlim(lag_time/2, max(g2_for_calc_df.tau)*20)
                ax.legend(legend, loc=1)
                    
                figpath=cI_file_folder_path + '\\out\\' + ROI_list[i] + '_log_spaced_tw_IC.png'
                fig.savefig(figpath)
                
                print('Legend: ' + ROI_list[i] + ' all log_tw_spaced IC data')
                print(legend)
                
                #Asks whether some datasets need to be refitted
                if g.buttonbox('Do you want to refit some of the ICs?', image=figpath, choices=['Yes', 'No']) == 'No':
                    image_names_df=pd.concat([image_names_df, image_names_temp_df])
                    del image_names_temp_df                
                    break
                else:
                    msg ='Which datasets do you need to refit?'
                    title = 'Selection of datasets to refit'
                    choice = g.multchoicebox(msg, title, ROI_tw_list)
                    
                    for l in range(len(choice)):
                        refit_list[ROI_tw_list.index(choice[l])]='yes'
    
    #Saves the data in an Excel file
    out_Excel_file_path = _config_data_analysis['out_folder_path'] + r'\out\fitting_outputs.xlsx'
    with pd.ExcelWriter(out_Excel_file_path) as writer:
        image_names_df.to_excel(writer, sheet_name = 'image_names')
        g2_fitting_data.to_excel(writer, sheet_name = 'g2_1_data')
        fitting_results_df.to_excel(writer, sheet_name = 'fitting_outputs')
    
    #Saves the output DataFrames in compressed files
    image_names_df.to_pickle(_config_data_analysis['out_folder_path'] + r'\out\image_names_df.pkl')
    g2_fitting_data.to_pickle(_config_data_analysis['out_folder_path'] + r'\out\g2_fitting_data.pkl')
    fitting_results_df.to_pickle(_config_data_analysis['out_folder_path'] + r'\out\fitting_results_df.pkl')