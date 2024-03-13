import pandas as pd
import csv
import numpy as np

# Obtaining Data and API codes from VitalDB
df_trks = pd.read_csv("https://api.vitaldb.net/trks")  # track list
df_labs = pd.read_csv('https://api.vitaldb.net/labs')  # laboratory results
df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # clinical information

# Obtaining list of Valid Case IDs
valid_case_ids=[]
with open('valid_case_ids.csv', newline='') as f:
   reader = csv.reader(f)
   for row in reader:
    if(row[0]=="IDs"):
      continue
    valid_case_ids.append(int(row[0]))

# Helper function to fill gaps in time index
def fill_gaps(timeval,rates):
    df = pd.DataFrame({'x': timeval, 'y': rates})
    filled_timeval = pd.DataFrame({'x': range(df['x'].min(), df['x'].max() + 1)})
    df['x'] = df['x'].astype(str)
    filled_timeval['x'] = filled_timeval['x'].astype(str)
    df_filled = pd.merge(filled_timeval, df, on='x', how='left')
    df_filled['y'] = df_filled['y'].where(pd.notna(df_filled['y']), np.nan)
    return df_filled["x"],df_filled["y"]

# Helper function to align time at a frequency of 1 second and interpolate missing values
def align_time_and_interpolate(timeval, rates):
    timeval = timeval.round().astype(int)
    timeval,rates = fill_gaps(timeval,rates)
    rates=rates.interpolate(limit_direction='both')
    return timeval.astype(int),rates.astype(float)
    
# Helping function to smooth data with a moving median algorithm
def moving_median(data, window_size):
    return data.rolling(window=window_size,min_periods=1).median()

# Add a 0 to the beginning to standardize input
def pad_start(timeval, rates):
  if(timeval[0]>=0.5):
    timeval=pd.concat([pd.Series([0]),timeval],ignore_index=True)
    rates=pd.concat([pd.Series([np.nan]),rates],ignore_index=True)
  else:
    if(rates[0]==0):
      rates[0]=np.nan
  return timeval, rates

# Helper function to remove duplicate x values
def remove_duplicate_x(timeval,rates):
  df = pd.DataFrame({'x': timeval, 'y': rates})
  filtered_df = df.loc[df.groupby('x')['y'].idxmax()]
  return filtered_df["x"],filtered_df["y"]

# Helper function to get a certain type of data for a certain patient
def get_data(caseid,code):
    tid=df_trks[df_trks['tname']==code]
    tid=tid[tid["caseid"]==caseid]
    tid=tid["tid"].iloc[0]
    data=pd.read_csv("https://api.vitaldb.net/"+str(tid))
    return data["Time"], data[code]

# Helper function to add a 0 to the end to standardize input
def pad_end(timeval,rates,lasttime):
    if(timeval[len(timeval)-1]<(lasttime-0.5)):
        timeval=pd.concat([timeval,pd.Series([lasttime])],ignore_index=True)
        rates=pd.concat([rates,pd.Series([np.nan])],ignore_index=True)
    return timeval, rates

# Helper function to clean infusion rate data
def clean_infusion_target(timeval,rates,window_coeff=0.02):
  timeval,rates = remove_duplicate_x(timeval,rates)
  timeval,rates=pad_start(timeval,rates)
  timeval,rates=align_time_and_interpolate(timeval,rates)
  timeval,rates = remove_duplicate_x(timeval,rates)
  rates.reset_index(drop=True,inplace=True)
  timeval.reset_index(drop=True,inplace=True)

  valid=True
  if(rates.max()==0 or timeval[0]>100):
    valid=False

  return timeval, rates, valid

# Helper function with smoothing to clean non-infusion rate data
def clean_nt_data(timeval,rates,lasttime, window_coeff=0.02):
  timeval,rates = remove_duplicate_x(timeval,rates)
  timeval,rates=pad_start(timeval,rates)
  timeval,rates=pad_end(timeval,rates,lasttime)
  timeval,rates=align_time_and_interpolate(timeval,rates)
  timeval,rates = remove_duplicate_x(timeval,rates)
  rates=moving_median(rates,int(window_coeff*len(rates)))
  rates.reset_index(drop=True,inplace=True)
  timeval.reset_index(drop=True,inplace=True)

  valid=True
  if(rates.max()==0 or timeval[0]>200):
    valid=False

  return timeval, rates, valid



columns=["time","caseid","Orchestra/PPF20_CP","Orchestra/PPF20_CT","Orchestra/PPF20_RATE","Orchestra/PPF20_CE","Orchestra/RFTN20_CP","Orchestra/RFTN20_CT","Orchestra/RFTN20_RATE","Orchestra/RFTN20_CE","BIS/BIS","Solar8000/HR","Solar8000/PLETH_SPO2","Solar8000/ART_MBP","Solar8000/ART_SBP","Solar8000/BT","casestart","anestart","opstart","age","sex","height","weight","bmi","asa","optype","dx","opname","ane_type","preop_dm"]
static_columns=["casestart","anestart","opstart","age","sex","height","weight","bmi","asa","optype","dx","opname","ane_type","preop_dm"]
normal_track_data=["Orchestra/PPF20_CP","Orchestra/PPF20_CT","Orchestra/PPF20_CE","Orchestra/RFTN20_CP","Orchestra/RFTN20_CT","Orchestra/RFTN20_CE","Solar8000/HR","Solar8000/PLETH_SPO2","Solar8000/ART_MBP","Solar8000/ART_SBP","Solar8000/BT","BIS/BIS"]
complete_data=pd.DataFrame(columns=columns)


for i in range(len(valid_case_ids)):
    current_df=pd.DataFrame(columns=columns)
    caseid=valid_case_ids[i]
    print("case: ",caseid," i: ",i)
    #Working with Target
    target_data_time,target_data_rate=get_data(caseid,"Orchestra/PPF20_RATE")
    target_data_time,target_data_rate,target_valid=clean_infusion_target(target_data_time,target_data_rate)
    current_df["time"]=target_data_time
    current_df["caseid"]= pd.Series(caseid, index=range(len(target_data_time)))
    current_df["Orchestra/PPF20_RATE"]=target_data_rate

    #Working with Remifentanil
    remi_data_time,remi_data_rate=get_data(caseid,"Orchestra/RFTN20_RATE")
    remi_data_time,remi_data_rate,remi_valid=clean_infusion_target(remi_data_time,remi_data_rate)
    current_df["Orchestra/RFTN20_RATE"]=remi_data_rate

    # Cleaning all non-infusion data
    for j in normal_track_data:
        nt_data_time,nt_data_rate=get_data(caseid,j)
        lastval=target_data_time[len(target_data_time)-1]
        nt_data_time,nt_data_rate,nt_valid=clean_nt_data(nt_data_time,nt_data_rate,lastval)
        current_df[j]=nt_data_rate
    
    cur_cases_row=df_cases[df_cases["caseid"]==caseid]

    for j in static_columns:
        current_df[j]= pd.Series(cur_cases_row[j].iloc[0], index=range(len(target_data_time)))
    
    if(i==0):
        current_df.to_csv("first_processed_data.csv",index=False,mode='a', header=True)
    current_df.to_csv("first_processed_data.csv",index=False,mode='a', header=False)

    