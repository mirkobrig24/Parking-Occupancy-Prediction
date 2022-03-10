import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import utils.losses


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
        self._loss = loss
        self.feat_max_val = feat_max_val

    def forward(self, x, adj):
        #print('---FORWARD---', x.size())
        # (batch_size, seq_len, num_nodes)
        batch_size, _, num_nodes = x.size()
        # (batch_size, num_nodes, hidden_dim)
        hidden = self.model(x, adj)
        # (batch_size * num_nodes, hidden_dim)
        hidden = hidden.reshape((-1, hidden.size(2)))
        # (batch_size * num_nodes, pre_len)
        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions

    def shared_step(self, batch, batch_idx):
        
        # (batch_size, seq_len/pre_len, num_nodes)
        #print('---SHARED-STEP---', type(batch))
        #print('---SHARED-STEP---', len(batch))
        #print('---SHARED-STEP[0]---', type(batch[0]))
        #print('---SHARED-STEP[0]---', batch[0].size())
        #print('---SHARED-STEP[1]---', batch[1].size())
        #print('---IDX---', batch_idx)
        
        x, adj, y = batch
        num_nodes = x.size(2)
        #print('---SHARED---')
        predictions = self(x, adj)
        #print('---CIAO---')
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
        #print('---TRAINING-STEP---', batch[0].size())
        predictions, y = self.shared_step(batch, batch_idx)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        #print('---VAL---')
        #batch[1] = batch[1].to_dense()
        predictions, y = self.shared_step(batch, batch_idx)
        #print('---PRED---', predictions.size())
        #print('---Y---', y.size())
        #print('---BATCH[1]---', batch[2].size())
        #exit()
        predictions = predictions * self.feat_max_val
        y = y * self.feat_max_val
        loss = self.loss(predictions, y)
        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        mae = torchmetrics.functional.mean_absolute_error(predictions, y)
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
        return predictions.reshape(batch[2].size()), y.reshape(batch[2].size())

    def test_step(self, batch, batch_idx):
        pass

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
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--loss", type=str, default="mse")
        return parser
