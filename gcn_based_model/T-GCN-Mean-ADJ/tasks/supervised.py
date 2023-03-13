import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import utils.losses
import torch.nn.functional as F
import h5py

class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        regressor="linear",
        loss="mse",
        pre_len: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.regressor = (
            nn.Linear(
                self.model.hyperparameters.get("hidden_dim")
                or self.model.hyperparameters.get("output_dim"),
                self.hparams.pre_len,
            )
            if regressor == "linear"
            else regressor
        )
        #self.linear = (
        #    nn.Linear(
        #        self.model.hyperparameters.get("hidden_dim")
        #        or self.model.hyperparameters.get("output_dim"),
        #        self.model.hyperparameters.get("hidden_dim")
        #        or self.model.hyperparameters.get("output_dim")
        #    )
        #    if regressor == "linear"
        #    else regressor
        #)
        self._loss = loss
        self.feat_max_val = feat_max_val
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        # (batch_size, seq_len, num_nodes)
        batch_size, _, num_nodes = x.size()
        # (batch_size, num_nodes, hidden_dim)
        hidden = self.model(x)
        # (batch_size * num_nodes, hidden_dim)
        hidden = hidden.reshape((-1, hidden.size(2)))
        #hidden = F.dropout(hidden, 0.5, training=self.training)
        #x = self.linear(hidden)
        #x = self.tanh(x)
        # (batch_size * num_nodes, pre_len)
        if self.regressor is not None:
            predictions = self.regressor(hidden) #(x)
        else:
            predictions = x
        predictions = self.tanh(predictions)
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions

    def shared_step(self, batch, batch_idx):
        # (batch_size, seq_len/pre_len, num_nodes)
        #print('---SHARED-STEP---')
        x, y = batch
        num_nodes = x.size(2)
        predictions = self(x)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        y = y.reshape((-1, y.size(2)))
        return predictions, y

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch, batch_idx):
        #print('---TRAINING-STEP---')
        predictions, y = self.shared_step(batch, batch_idx)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        #predictions = predictions* self.feat_max_val  # Normalized
        #y = y * self.feat_max_val
        loss = self.loss(predictions, y)
        rmse = utils.metrics.rmse(y, predictions, self.feat_max_val) #torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        mae = utils.metrics.mape(y, predictions, self.feat_max_val) #torchmetrics.functional.mean_absolute_error(predictions, y)
        accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        explained_variance = utils.metrics.explained_variance(predictions, y)
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            "accuracy": accuracy,
            "R2": r2,
            "ExplainedVar": explained_variance,
        }
        self.log_dict(metrics)
        # se devo prendere i pesi
        #fname = 'TGCN_parking_realVSpred.h5'
        #h5 = h5py.File(fname, 'w')
        #h5.create_dataset('Y_pred', data=predictions.reshape(batch[1].size()))
        #h5.create_dataset('Y_real', data= y.reshape(batch[1].size()))
        #h5.close()
        #f = h5py.File(fname, 'r')
        #print( [key for key in f.keys()])

        return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())

    def test_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())


    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1e-4)
        parser.add_argument("--loss", type=str, default="mse_with_regularizer")
        return parser