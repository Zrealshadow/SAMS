
import model
import time
import utils
from torch import optim
from torch.utils.data import DataLoader
from singleton import logger
from model_utils import *


class CombinedModel:

    def __init__(self, args, total_columns: int, col_cardinality_sum: int):
        """
        :param args: all parameters
        :param total_columns:  # of columns
        """
        self.args = args

        # declarition
        self.hyper_net = None
        self.moe_net = None

        # total number of columns, each column is one feature field.
        self.total_columns = total_columns
        # sume of cardinality  +1 of each column
        self.col_cardinality_sum = col_cardinality_sum

        self.construct_model()

    def construct_model(self):
        self.hyper_net = model.HyperNet(
            nfield=self.total_columns,
            nfeat=self.col_cardinality_sum,
            nemb=self.args.sql_nemb,
            nlayers=self.args.num_layers,
            hid_layer_len=self.args.hid_layer_len,
            dropout=self.args.dropout,
            L=self.args.moe_num_layers,
            K=self.args.K,
        )

        self.moe_net = model.MOENet(
            nfield=self.args.nfield,
            nfeat=self.args.nfeat,
            nemb=self.args.data_nemb,
            moe_num_layers=self.args.moe_num_layers,
            moe_hid_layer_len=self.args.moe_hid_layer_len,
            dropout=self.args.dropout,
        )

    def trian(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, use_test_acc=True):

        start_time, best_valid_auc = time.time(), 0.

        # define the loss function, for two class classification
        opt_metric = nn.BCEWithLogitsLoss(reduction='mean').to(self.args.device)

        # define the parameter
        params = list(self.hyper_net.parameters()) + list(self.moe_net.parameters())
        optimizer = torch.optim.Adam(params, lr=self.args.lr)

        # scheduler to update the learning learning rate
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.epoch_num,  # Maximum number of iterations.
            eta_min=1e-4)  # Minimum learning rate.

        # gradient clipping, set the gradient value to be -1 - 1 to prevent exploding gradients
        for p in params:
            p.register_hook(lambda grad: torch.clamp(grad, -1., 1.))

        info_dic = {}
        valid_auc = -1
        test_auc = -1
        # train for epoches.
        for epoch in range(self.args.epoch_num):
            logger.info(f'Epoch [{epoch:3d}/{self.args.epoch_num:3d}]')

            train_auc, train_loss = self.run(epoch, train_loader, opt_metric, optimizer=optimizer, namespace='train')
            scheduler.step()

            valid_auc, valid_loss = self.run(epoch, val_loader, opt_metric, namespace='val')

            if use_test_acc:
                test_auc, test_loss = self.run(epoch, test_loader, opt_metric, namespace='test')
            else:
                test_auc = -1

            info_dic[epoch] = {
                "train_auc": train_auc,
                "valid_auc": valid_auc,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_val_total_time": time.time() - start_time}

        if valid_auc >= best_valid_auc:
            best_valid_auc, best_test_auc = valid_auc, test_auc
            logger.info(f'best valid auc: valid {valid_auc:.4f}, test {test_auc:.4f}')
        else:
            logger.info(f'valid {valid_auc:.4f}, test {test_auc:.4f}')

    #  train one epoch of train/val/test
    def run(self, epoch, data_loader, opt_metric, optimizer=None, namespace='train'):
        if optimizer:
            self.hyper_net.train()
            self.moe_net.train()
        else:
            self.hyper_net.eval()
            self.moe_net.eval()

        time_avg, timestamp = utils.AvgrageMeter(), time.time()
        loss_avg, auc_avg = utils.AvgrageMeter(), utils.AvgrageMeter()

        if namespace == 'train':
            for batch_idx in self.args.iter_per_epoch:
                # randomly sample one batch
                sql_batch, data_batch = data_loader.sample_batch_sql_and_data()

                sql_batch = sql_batch.to(self.args.device)
                target = data_batch['y'].to(self.args.device)
                data_batch['id'] = data_batch['id'].to(self.args.device)
                data_batch['value'] = data_batch['value'].to(self.args.device)

                # 1. get the arch_advisor B*L*K
                arch_advisor = self.hyper_net(sql_batch)
                assert arch_advisor.size() == (self.args.batch_size, self.args.moe_num_layers * self.args.K)

                # reshape it to (B, L, K)
                arch_advisor = arch_advisor.reshape(self.args.batch_size, self.args.moe_num_layers, self.args.K)

                # todo: conduct sparse softmax
                arch_advisor = arch_advisor
                if self.args.alpha == 1.:
                    self.sparsemax = nn.Softmax(dim=-1)
                else:
                    self.sparsemax = EntmaxBisect(self.args.alpha, dim=-1)

                # calculate y and loss
                y = self.moe_net.forward(data_batch, arch_advisor)
                loss = opt_metric(y, target)
                optimizer.zero_grad()

                # one step to update both hypernet and moenet
                loss.backward()
                optimizer.step()

                # logging
                auc = utils.roc_auc_compute_fn(y, target)
                loss_avg.update(loss.item(), target.size(0))
                auc_avg.update(auc, target.size(0))

                time_avg.update(time.time() - timestamp)
                if batch_idx % self.args.report_freq == 0:
                    logger.info(f'Epoch [{epoch:3d}/{self.args.epoch_num}][{batch_idx:3d}/{len(data_loader)}]\t'
                                f'{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) '
                                f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')
        else:

            sql_all, data_all = data_loader.sample_all_data_by_sql()
            mini_batches = [(sql_all[i:i + self.args.batch_size], data_all[i:i + self.args.batch_size]) for i in
                            range(0, len(sql_all), self.args.batch_size)]

            for batch_idx, (sql_batch, data_batch) in enumerate(mini_batches):
                sql_batch = sql_batch.to(self.args.device)
                target = data_batch['y'].to(self.args.device)
                data_batch['id'] = data_batch['id'].to(self.args.device)
                data_batch['value'] = data_batch['value'].to(self.args.device)

                with torch.no_grad():
                    arch_advisor = self.hyper_net(sql_batch)
                    assert arch_advisor.size() == (self.args.batch_size, self.args.moe_num_layers * self.args.K)
                    # reshape it to (B, L, K)
                    arch_advisor = arch_advisor.reshape(self.args.batch_size, self.args.moe_num_layers, self.args.K)
                    # calculate y and loss
                    y = self.moe_net.forward(data_batch, arch_advisor)
                    loss = opt_metric(y, target)

                # logging
                auc = utils.roc_auc_compute_fn(y, target)
                loss_avg.update(loss.item(), target.size(0))
                auc_avg.update(auc, target.size(0))

                time_avg.update(time.time() - timestamp)
                if batch_idx % self.args.report_freq == 0:
                    logger.info(f'Epoch [{epoch:3d}/{self.args.epoch_num}][{batch_idx:3d}/{len(data_loader)}]\t'
                                f'{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) '
                                f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

        logger.info(f'{namespace}\tTime {utils.timeSince(s=time_avg.sum):>12s} '
                    f'AUC {auc_avg.avg:8.4f} Loss {loss_avg.avg:8.4f}')
        return auc_avg.avg, loss_avg.avg




