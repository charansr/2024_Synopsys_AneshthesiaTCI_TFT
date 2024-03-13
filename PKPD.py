import pandas as pd
import numpy as np
from itertools import groupby
from PyTCI.models import propofol

# Gender mapping dictionary to convert to lowercase
gender_mapping = {"M": 'm', "F": 'f'}

# Helper fucntion to split data into segments of unique target concentration values
def split_data(original_series):
    segemented_data = [(value, len(list(group))) for value, group in groupby(original_series)]
    return segemented_data

# Reading dataframe
df = pd.read_csv("first_processed_data.csv")

# Creating Empty columns
df["Marsh"]=pd.Series([None]) 
df["Schnider"]=pd.Series([None])
df["Eleveld"]=pd.Series([None])

# Iterating over every patient
for cas in df["caseid"].unique():
    print("caseid: ",cas)
    cur=df[df["caseid"]==cas]
    cur.reset_index(inplace=True,drop=True)
    splitup = split_data(cur["Orchestra/PPF20_CT"])
    weight=cur["weight"][0]
    age=cur["age"][0]
    height=cur["height"][0]
    gender=gender_mapping[cur["sex"][0]]
    
    # Creating Marsh Forecast
    tot=[]
    patient = propofol.Marsh(weight)

    for val,freq in splitup:
        infuse=patient.plasma_infusion(val, freq, 6)
        if(len(infuse)>0):
            div=int(freq/len(infuse))
            newinf=[]
            for i in infuse:
                newinf+=[i]
                newinf+= np.zeros(div).tolist()
        else:
            newinf = np.zeros(6).tolist()
        err=len(newinf)-freq
        del(newinf[-err:])
        tot+=newinf
    
    if(cas==3):
        df["Marsh"]=pd.Series(tot)
    else:
        df["Marsh"]=pd.concat([df[df["Marsh"].notnull()]["Marsh"],pd.Series(tot)],ignore_index=True)
    
    # Creating Schnider Forecast
    tot=[]
    patient = propofol.Schnider(age,weight,height,gender)

    for val,freq in splitup:
        infuse=patient.plasma_infusion(val, freq, 6)
        if(len(infuse)>0):
            div=int(freq/len(infuse))
            newinf=[]
            for i in infuse:
                newinf+=[i]
                newinf+= np.zeros(div).tolist()
        else:
            newinf = np.zeros(6).tolist()
        err=len(newinf)-freq
        del(newinf[-err:])
        tot+=newinf
    
    if(cas==3):
        df["Schnider"]=pd.Series(tot)
    else:
        df["Schnider"]=pd.concat([df[df["Schnider"].notnull()]["Schnider"],pd.Series(tot)],ignore_index=True)

    # Creating Eleveld Forecast
    tot=[]
    patient = propofol.Eleveld(age,weight,height,gender)

    for val,freq in splitup:
        infuse=patient.plasma_infusion(val, freq, 6)
        if(len(infuse)>0):
            div=int(freq/len(infuse))
            newinf=[]
            for i in infuse:
                newinf+=[i]
                newinf+= np.zeros(div).tolist()
        else:
            newinf = np.zeros(6).tolist()
        err=len(newinf)-freq
        del(newinf[-err:])
        tot+=newinf
    
    if(cas==3):
        df["Eleveld"]=pd.Series(tot)
    else:
        df["Eleveld"]=pd.concat([df[df["Eleveld"].notnull()]["Eleveld"],pd.Series(tot)],ignore_index=True)


df.to_csv("pkpd_data.csv")