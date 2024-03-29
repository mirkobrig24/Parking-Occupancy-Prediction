import argparse
import traceback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import utils.callbacks
import utils.data
#import utils.email
import utils.logging
import torch
from bayes_opt import BayesianOptimization
import numpy as np
import json
import csv
import os
import h5py

# datapath for features matrix and adj matrix
#DATA_PATHS = {
#    "data": {"feat": "../results/feat_1h.csv", "adj": "../results/MEAN_transition_matrix_1H.pickle"},
#    }

DATA_PATHS = {
    "data": {"feat": "../results/feat_GCN_1h.h5", "adj": "../results/MEAN_transition_matrix_1H.pickle"},
     }

T = 24  # number of time intervals in one day (For 8H is 3 and for 1H is 24)

seq_len = 12  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 0  # length of trend dependent sequence
nb_flow = 1  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test,
days_test = 122
len_test = int((T*days_test) * 0.2)

def get_model(hidden, dm):
    model = None
    if args.model_name == "GCN":
        model = models.GCN(adj=dm.adj, input_dim=args.seq_len, output_dim=args.hidden_dim)
    if args.model_name == "GRU":
        model = models.GRU(input_dim=dm.adj.shape[0], hidden_dim=args.hidden_dim)
    if args.model_name == "TGCN":
        model = models.TGCN(adj=dm.adj, hidden_dim=hidden)
    return model


def get_task(args, model, dm):
    task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
        model=model, feat_max_val=dm.feat_max_val, **vars(args)
    )
    return task

def get_callbacks(args):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='RES', monitor="RMSE", filename='{epoch}-{val_loss:.2f}', 
    verbose = True, mode = 'min')
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=25)
    #plot_validation_predictions_callback = utils.callbacks.PlotValidationPredictionsCallback(monitor="train_loss")
    callbacks = [
        checkpoint_callback,
        early_stopping
        #plot_validation_predictions_callback,
    ]
    return callbacks

# Only closeness
#def main_supervised(batch_s, seq_l, hidden_d, i):
#    print('dimensioni del test', len_test)
#    batch_s = 585 #2 ** int(batch_s)
#    seq_l = int(seq_l)
#    hidden_d = 2 ** int(hidden_d)

def main_supervised(batch_s, seq_l, len_period, len_trend, hidden_d, i):
    batch_s = batch_s #2 ** int(batch_s)
    seq_l = int(seq_l)
    len_period = int(len_period)
    len_trend = int(len_trend)
    hidden_d = 2 ** int(hidden_d)
    np.random.seed(i*18)
    torch.manual_seed(i*18)

    #dm = utils.data.SpatioTemporalCSVDataModule(
    #    feat_path=DATA_PATHS[args.data]["feat"], adj_path=DATA_PATHS[args.data]["adj"],
    #    batch_size=batch_s, seq_len=seq_l)

    dm = utils.data.SpatioTemporalCSVDataModule_2(
        feat_path=DATA_PATHS[args.data]["feat"], adj_path=DATA_PATHS[args.data]["adj"],
        batch_size=batch_s, seq_len=seq_l, nb_flow=nb_flow, T=T, len_period=len_period, 
        len_trend=len_trend, len_test=len_test)   


    print('dimensioni della batch', batch_s)

    model = get_model(hidden_d, dm)
    task = get_task(args, model, dm)
    callbacks = get_callbacks(args)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, limit_train_batches=0)
    trainer.fit(task, dm)
    results = trainer.validate(datamodule=dm, ckpt_path='/home/gpu2/Documenti/Parking-Occupancy-Prediction/T-GCN-Mean-ADJ/RES/epoch=190-RMSE=1.08.ckpt')
    
    print(results)


def main():
    rank_zero_info(vars(args))
    params_fname = 'opt_1H.json'
    with open(os.path.join('results', params_fname), 'r') as f:
        opt = json.load(f)
    
    # Only closeness
    #results = main_supervised(batch_s=585,#opt['batch_s'],
    #                              seq_l=opt['seq_l'],
    #                              hidden_d=opt['hidden_d'],
    #                              i=1)

    
    results = main_supervised(batch_s=585,#opt['batch_s'],
                                  seq_l=opt['seq_l'],
                                  len_period=opt['len_period'],
                                  len_trend=opt['len_trend'],
                                  hidden_d=opt['hidden_d'],
                                  i=0)           
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--data", type=str, help="The name of the dataset", choices=("shenzhen", "losloop"), default="data"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("GCN", "GRU", "TGCN"),
        default="TGCN",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised",),
        default="supervised",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")
    parser.add_argument("--send_email", "--email", action="store_true", help="Send email when finished")

    temp_args, _ = parser.parse_known_args()

    parser = getattr(utils.data, temp_args.settings.capitalize() + "DataModule").add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.capitalize() + "ForecastTask").add_task_specific_arguments(parser)

    global args
    args = parser.parse_args()
    utils.logging.format_logger(pl._logger)
    if args.log_path is not None:
        utils.logging.output_logger_to_file(pl._logger, args.log_path)

    try:
        results = main()
    except:  # noqa: E722
        traceback.print_exc()
        if args.send_email:
            tb = traceback.format_exc()
            subject = "[Email Bot][] " + "-".join([args.settings, args.model_name, args.data])
            utils.email.send_email(tb, subject)
        exit(-1)

    if args.send_email:
        subject = "[Email Bot][] " + "-".join([args.settings, args.model_name, args.data])
        utils.email.send_experiment_results_email(args, results, subject=subject)
