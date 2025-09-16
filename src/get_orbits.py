#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:40:53 2022

@author: sasson

Function df = get_orbits(year,month,day,hour)

V1 : ne marche que pour les dates > 08 Sep. 2021 et <=30 Sep. 2021 (dev en cours)

"""

import numpy as np 
import pandas as pd
import datetime 
import epygram
epygram.init_env()



def get_orbits(year,month,day,hour):
    
    
    
   
    #Mise en forme de la date :
        
    orbits_file   = "/cnrm/obs/data1/borderiesm/WIVERN/orbit/Wivern_prop_5_days_octobre.txt"

    day_begin_orbit = 8 # beginning day of orbits file
    
    #==============================================================================
    #== Gestion des dates
    
    base = datetime.datetime(year, month, day, hour,0,0) 
    base_deb = datetime.datetime(2021, 9, day_begin_orbit, 00, 00, 00) #1ère ligne du fichier d'orbite
    
    delta = base-base_deb #difference between obsoul required time and begininning time of orbit file
    
    #==============================================================================
    # Définition de i et R,  dépendant de la date et de l'heure du réseau, et
    # nécessaires pour naviguer dans le fichier d'orbites
    #==============================================================================
    if hour == 0:
        i=0;
    elif hour == 6:
        i=1;
    elif hour == 12:
        i=2 
    elif hour ==18:
        i=3    
    
    q, R=divmod(delta.days,5)
    
    
    # Défintion du coefficient k : ligne_begin = 135000 + k * 270000
    if R==0:
        k= i-1
    else:
        k = 4*R + i - 1
    
    rows_begin=135000 #début du réseau 20210908T06
    nrows=270000
    
    skiprows = rows_begin + k*nrows #ligne de début pour le réseau en question
    
    #==============================================================================
    # Récupération des données orbitales de WIVERN
    #==============================================================================
    if (R==0) & (hour==0): # Exception sur le tout premier réseau (20210908T00)
    
        df1 = pd.read_csv(orbits_file,sep=';',header=0,
                             names = ["dates","longitude",'latitude',"azimuth",'Scanline_id','Scanline_position_id'],
                             parse_dates = ['dates'],skiprows =0 , nrows=nrows/2)
        df2 = pd.read_csv(orbits_file,sep=';',header=0,
                             names = ["dates","longitude",'latitude',"azimuth",'Scanline_id','Scanline_position_id'],
                             parse_dates = ['dates'],skiprows =5265000 , nrows=nrows/2)
        df=pd.concat([df2,df1],ignore_index=True)
        del df1,df2
        
    else:
        df = pd.read_csv(orbits_file,sep=';',header=0,
                             names = ["dates","longitude",'latitude',"azimuth",'Scanline_id','Scanline_position_id'],
                             parse_dates = ['dates'],skiprows =skiprows , nrows=nrows)
        
        
    df.longitude = np.mod(df.longitude+180,360) - 180 
    
    #Différence entre le début des df.dates et le début du fichier d'orbite
    delta2= df.dates[0] - base_deb + datetime.timedelta(0,0,0,0,0,3) # +3h pour retomber sur le réseau d'assim(-3h:+3h)
    
    if q>0:
    
        if (R==0) & (hour==0):
        
            df[0:int(nrows/2)].dates=df[0:int(nrows/2)].dates + delta - delta2
            df[int(nrows/2):].dates=df[int(nrows/2):].dates + delta
        
        else:
            df.dates=df.dates+delta -delta2 #Adjust dates in df   
    
    print(df)   
    

    print("\n\n#=======================================================================")
    print('#== Orbit file reading : resume')
    print('\nDesired assim time : {}'.format(base))
    print('Beginning of orbit file : {}'.format(base_deb))
    
    print('\nDelta1 = {}'.format(delta))
    print('Delta2 = {}'.format(delta2))
    
    print('\nDelta days = %i'%(delta.days))
    print('Quotient = %i | Reste = %i'%(q,R))
    
    print('\ndf.dates.min() = {}  |  df.dates.max() = {}'.format(df.dates.min(),df.dates.max()) )
    
    print('\nMiddle of df.dates : check pd.concat() ')
    print(df[int(nrows/2-4):int(nrows/2+4)])
    
    print('#========================================================================')
    
    return df
    