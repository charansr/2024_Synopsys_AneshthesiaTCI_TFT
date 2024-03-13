import pandas as pd
import torch
import torch.nn as nn
from typing import Any,Dict
import lightning.pytorch as pl
from pytorch_forecasting.utils import to_list
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import CrossEntropy, TimeSeriesDataSet, GroupNormalizer, TemporalFusionTransformer, NaNLabelEncoder
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_forecasting.metrics import (
    MAE,
    MASE,
    SMAPE,
    RMSE,
    MAPE,
    MultiLoss,
    QuantileLoss,
)

import pickle
from itertools import groupby



#Data Constants
workernum = 8
batch_size = 12
brave_workers = False
pinmem=False
marsh_coeff=50
schnider_coeff=200
eleveld_coeff=100


# Helper function to add strings to a file
def append_string_to_txt(string_to_append, file_path):
    try:
        with open(file_path, 'a') as file:
            file.write(string_to_append)
    except Exception as e:
        pass


# Reading CSV with data
final_df = pd.read_csv("alldata_pkpd_spiky1719.csv")

#Compressing or processing the file creates Unnamed columns so these lines clean them out
if("Unnamed: 0" in final_df.columns):
    final_df=final_df.drop("Unnamed: 0",axis=1)

if("Unnamed: 0.1" in final_df.columns):
    final_df=final_df.drop("Unnamed: 0.1",axis=1)

if("Unnamed: 0.2" in final_df.columns):
    final_df=final_df.drop("Unnamed: 0.2",axis=1)

# Use Mean deviance to create another input column which identifies spike locations
tar_mean=(final_df["Orchestra/PPF20_RATE"]).mean()
tar_std=(final_df["Orchestra/PPF20_RATE"]).std()
spike_detector = final_df["Orchestra/PPF20_RATE"] > (tar_mean + (0.5 * tar_std))
final_df["InfusionSpikes"]=spike_detector.apply(lambda x: 1 if x else 0)

# Lists of all data inputs and their categories
columns=["time","caseid","Orchestra/PPF20_CP","Orchestra/PPF20_CT","Orchestra/PPF20_RATE","Orchestra/PPF20_CE","Orchestra/RFTN20_CP","Orchestra/RFTN20_CT","Orchestra/RFTN20_RATE","Orchestra/RFTN20_CE","BIS/BIS","Solar8000/HR","Solar8000/PLETH_SPO2","Solar8000/ART_MBP","Solar8000/ART_SBP","Solar8000/BT","casestart","anestart","opstart","age","sex","height","weight","bmi","asa","optype","dx","opname","ane_type","preop_dm","Marsh","Schnider","Eleveld","InfusionSpikes"]
static_tracks=["casestart","anestart","opstart","age","height","weight","bmi","asa","preop_dm"]
static_cat=["sex","optype","dx","opname","ane_type"]
time_varying_known=["Orchestra/PPF20_CT","Orchestra/RFTN20_CT","Marsh","Schnider","Eleveld"]
time_varying_notknown=["Orchestra/PPF20_RATE","Orchestra/PPF20_CP","Orchestra/PPF20_CE","Orchestra/RFTN20_CP","Orchestra/RFTN20_CE","Orchestra/RFTN20_RATE","Solar8000/HR","Solar8000/PLETH_SPO2","Solar8000/ART_MBP","Solar8000/ART_SBP","Solar8000/BT","BIS/BIS"]

# Last cleaning and retyping of data
final_df[static_tracks]=final_df[static_tracks].astype("float")
final_df[time_varying_known]=final_df[time_varying_known].astype("float")
final_df[time_varying_notknown]=final_df[time_varying_notknown].astype("float")
final_df["time"]=final_df["time"].astype(int)
final_df["InfusionSpikes"]=final_df["InfusionSpikes"].astype(int)
final_df[static_cat]=final_df[static_cat].astype(str)
final_df=final_df.dropna()
final_df.reset_index(drop=True,inplace=True)

# Scaling PK/PD model data
final_df["Marsh"]=final_df["Marsh"]*marsh_coeff
final_df["Schnider"]=final_df["Schnider"]*schnider_coeff
final_df["Eleveld"]=final_df["Eleveld"]*eleveld_coeff

# splitting data into train and validate sectinons
with open("train_caseids.pkl", 'rb') as file:
    traincas = pickle.load(file)
train_df=final_df[final_df["caseid"].isin(traincas)]

with open("val_caseids.pkl", 'rb') as file:
    valcas = pickle.load(file)
val_df=final_df[final_df["caseid"].isin(valcas)]

# Creating train dataset
max_prediction_len = 1000
max_encoder_len = 1800
min_encoder_len = int(max_encoder_len/2)

training = TimeSeriesDataSet(
    train_df,
    time_idx="time",
    target="InfusionSpikes",
    group_ids=["caseid"],
    min_encoder_length=min_encoder_len, 
    max_encoder_length=max_encoder_len,
    min_prediction_length=1,
    max_prediction_length=max_prediction_len,
    static_categoricals=static_cat,
    static_reals=static_tracks,
    time_varying_known_reals=time_varying_known,
    time_varying_unknown_reals=time_varying_notknown,
    target_normalizer=NaNLabelEncoder(),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True
)
        
# Creating subclass of TFT to add extra logging. This code is mostly from Pytorch with my edits meant to save extra metrics to an output.txt file
class custom_tft(TemporalFusionTransformer):
    def log_metrics(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        out: Dict[str, torch.Tensor],
        prediction_kwargs: Dict[str, Any] = None,
    ) -> None:
        """
        Log metrics every training/validation step.

        Args:
            x (Dict[str, torch.Tensor]): x as passed to the network by the dataloader
            y (torch.Tensor): y as passed to the loss function by the dataloader
            out (Dict[str, torch.Tensor]): output of the network
            prediction_kwargs (Dict[str, Any]): parameters for ``to_prediction()`` of the loss metric.
        """
        # logging losses - for each target
        if prediction_kwargs is None:
            prediction_kwargs = {}
        y_hat_point = self.to_prediction(out, **prediction_kwargs)
        if isinstance(self.loss, MultiLoss):
            y_hat_point_detached = [p.detach() for p in y_hat_point]
        else:
            y_hat_point_detached = [y_hat_point.detach()]

        for metric in self.logging_metrics:
            for idx, y_point, y_part, encoder_target in zip(
                list(range(len(y_hat_point_detached))),
                y_hat_point_detached,
                to_list(y[0]),
                to_list(x["encoder_target"]),
            ):
                y_true = (y_part, y[1])
                if isinstance(metric, MASE):
                    loss_value = metric(
                        y_point, y_true, encoder_target=encoder_target, encoder_lengths=x["encoder_lengths"]
                    )
                else:
                    loss_value = metric(y_point, y_true)
                if len(y_hat_point_detached) > 1:
                    target_tag = self.target_names[idx] + " "
                else:
                    target_tag = ""
                    
                    #CUSTOM PART START
                    save_string = "{}{}_{}: {}".format(target_tag, self.current_stage, metric.name, loss_value)
                    save_string+="\n"
                    append_string_to_txt(save_string,"/home/charansr/out_dev1/output.txt")
                    # CUSTOM PART END
                    
                    self.log(
                    f"{target_tag}{self.current_stage}_{metric.name}",
                    loss_value,
                    on_step=self.training,
                    on_epoch=True,
                    batch_size=len(x["decoder_target"]),
                    sync_dist=True
                )

# Creating the actual model and setting hyperparemeters
tft = custom_tft.from_dataset(
    training,
    learning_rate=0.01,
    hidden_size=160,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=160,
    output_size=2,  
    loss=CrossEntropy(),
    log_interval=10, 
    reduce_on_plateau_patience=4,
    logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])
)


path = "binary_best_model-epoch=43-val_loss=0.09.ckpt"
checkpoint=torch.load(path)
tft.load_state_dict(state_dict=checkpoint["state_dict"],strict=True)



for cas in train_df["caseid"].unique():
    print("caseid: ",cas)
    cur=train_df[train_df["caseid"]==cas]
    cur.reset_index(inplace=True,drop=True)
    slices_needed=int((cur.shape[0]/1000)+0.5)

    case_dataset=TimeSeriesDataSet(
        cur,
        time_idx="time",
        target="InfusionSpikes",
        group_ids=["caseid"],
        min_encoder_length=min_encoder_len, 
        max_encoder_length=max_encoder_len,
        min_prediction_length=1,
        max_prediction_length=max_prediction_len,
        static_categoricals=static_cat,
        static_reals=static_tracks,
        time_varying_known_reals=time_varying_known,
        time_varying_unknown_reals=time_varying_notknown,
        target_normalizer=NaNLabelEncoder(),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    tot=[]

    for i in range(slices_needed):
        preds=tft.predict(case_dataset.filter(lambda x: x.time_idx_first_prediction == (i*1000)))
        preds=preds.view(-1).tolist()
        tot+=to_list
    
    if(cas==3):
        final_df["InfusionSpikes"]=pd.Series(tot)
    else:
        final_df["InfusionSpikes"]=pd.concat([final_df[final_df["InfusionSpikes"].notnull()]["InfusionSpikes"],pd.Series(tot)],ignore_index=True)


final_df.to_csv("fully_cleaned_data.csv")

