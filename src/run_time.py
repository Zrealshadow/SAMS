
import json
import time
import os
from copy import deepcopy
from typing import Union

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.data_loader import Workload


import third_party.utils.func_utils as utils
from src.singleton import logger

from src.model import initialize_model


class Wrapper(object):

    def __init__(self, args, writer: SummaryWriter):
        """
        :param args: all parameters
        """
        self.args = args
        self.writer = writer
        self.net = initialize_model(args)
        self.save_best_model = args.save_best_model

    def train(self,
              train_loader: DataLoader, val_loader: DataLoader, test_loader: Workload,
              use_test_acc=True):
        """
        :param train_loader: data loaer
        :param val_loader: data loaer
        :param test_loader: data loaer
        :param use_test_acc: if use test dataset to valid during trianing.
        :return:
        """

        start_time, best_valid_auc, best_test_auc = time.time(), 0., 0.

        # define the loss function, for two class classification
        opt_metric = nn.BCEWithLogitsLoss(
            reduction='mean').to(self.args.device)

        # define the parameter, which includes both networks
        params = self.net.parameters()
        optimizer = torch.optim.Adam(params, lr=self.args.lr)

        # scheduler to update the learning rate
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.epoch,  # Maximum number of iterations.
            eta_min=1e-4)  # Minimum learning rate.

        # gradient clipping, set the gradient value to be -1 - 1 to prevent exploding gradients
        for p in params:
            p.register_hook(lambda grad: torch.clamp(grad, -1., 1.))

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        # records running info
        info_dic = {}

        best_net = self.net
        best_epoch = 0
        epochs_no_improve = 0
        alpha_list = []

        # train for epoches.
        for epoch in range(self.args.epoch):
            print(f'Epoch [{epoch:3d}/{self.args.epoch:3d}]')

            # 1. train
            train_auc, train_pr, train_loss = self.run(
                epoch, train_loader, opt_metric, optimizer=optimizer, namespace='train')
            scheduler.step()

            # 2. valid with valid
            # update val_loader's history
            # print(f'Training sql histry = {len(train_loader.dataset.sql_history)}')
            # val_loader.sql_history = train_loader.dataset.sql_history
            valid_auc, valid_pr, valid_loss = self.run(
                epoch, val_loader, opt_metric, namespace='val')

            # 3. valid with test
            if use_test_acc:
                # test_loader.sql_history = train_loader.dataset.sql_history
                (test_micro_auc, test_micro_auc_, test_macro_auc), (test_micro_pr, test_micro_pr_,
                                                                    test_macro_pr), test_loss = self.run(epoch, test_loader, opt_metric, namespace='test')
            else:
                test_micro_auc = -1

            # valid_auc = test_micro_auc_
            # if self.args.net == "sparsemax_vertical_sams":
            #     print(f"sparsemax alpha update", self.net.sparsemax.alpha)
            #     alpha_list.append(self.net.sparsemax.alpha.item())
            # set to empty for the next epoch
            # train_loader.dataset.reset_sql_his()

            info_dic[epoch] = {
                "train_aucroc": train_auc,
                "train_aucpr": train_pr,
                "train_loss": train_loss,

                "valid_aucroc": valid_auc,
                "valid_aucpr": valid_pr,
                "valid_loss": valid_loss,

                "test_micro_roc": test_micro_auc,
                "test_macro_roc": test_macro_auc,
                "test_micro_pr": test_micro_pr,
                "test_macro_pr": test_macro_pr,
                "test_micro_roc_": test_micro_auc_,
                "test_micro_pr_": test_micro_pr_,
                "test_loss": test_loss,

                "train_val_total_time": time.time() - start_time
            }

            self.writer.add_scalar(
                'Loss/Training_Epoch_Ave_Loss', train_loss, epoch)
            self.writer.add_scalar(
                'Loss/Valid_Epoch_Ave_Loss', valid_loss, epoch)
            self.writer.add_scalar(
                'Loss/Test_Epoch_Ave_Loss', test_loss, epoch)

            self.writer.add_scalar('AUCROC/Train_Ave', train_auc, epoch)
            self.writer.add_scalar('AUCROC/Valid_Ave', valid_auc, epoch)
            self.writer.add_scalar(
                'AUCROC/Workload_Micro', test_micro_auc, epoch)
            self.writer.add_scalar(
                'AUCROC/Workload_Macro', test_macro_auc, epoch)
            self.writer.add_scalar(
                'AUCROC/Workload_Micro_P', test_micro_auc_, epoch)

            self.writer.add_scalar('AUCPR/Train_Ave', train_pr, epoch)
            self.writer.add_scalar('AUCPR/Valid_Ave', valid_pr, epoch)
            self.writer.add_scalar(
                'AUCPR/Workload_Micro', test_micro_pr, epoch)
            self.writer.add_scalar(
                'AUCPR/Workload_Macro', test_macro_pr, epoch)
            self.writer.add_scalar(
                'AUCPR/Workload_Micro_P', test_micro_pr_, epoch)

            self.writer.flush()

            if valid_auc >= best_valid_auc:
                best_epoch = epoch
                best_valid_auc = valid_auc
                best_test_auc = test_micro_auc_
                epochs_no_improve = 0
                print(
                    f'best valid auc: valid {valid_auc:.4f}, test {test_micro_auc_:.4f}')
                # update best model
                if self.save_best_model:
                    best_net = deepcopy(self.net)

            else:
                epochs_no_improve += 1
                print(f"epochs no improves {epochs_no_improve}")
                print(
                    f'valid {valid_auc:.4f}, test {test_micro_auc_:.4f} Best epoch {best_epoch}, Test AUC {best_test_auc:.4f}')

            # early stopping mechanism
            if epochs_no_improve > self.args.patience:
                print(
                    f"Early stopping at epoch {epoch}, since there is no improving between {self.args.patience} epoch")
                break
        if self.save_best_model:
            dir_path = os.path.dirname(self.writer.log_dir)
            torch.save(best_net.state_dict(), os.path.join(
                dir_path, "best_model.pth"))
            print(
                f"Successfully save best model in {best_epoch}, Test Pefromance AUC:{info_dic[best_epoch]['test_micro_roc']:.4f} into {dir_path}")

        # print(f"best model alpha ", best_net.sparsemax.alpha)

        best_dic = info_dic[best_epoch]
        msg = (f"\n--------------------Final Result -----------------\n"
               + f"Get Best Model in epoch {best_epoch}\n"
               + f"Validation Performance:\n"
               + f"AUCROC:{best_dic['valid_aucroc']:.4f} AUCPR:{best_dic['valid_aucpr']:.4f}\n"
               + f"Workload Pefromance:\n"
               + f"Micro-AUCROC-P:{best_dic['test_micro_roc_']:.4f} Micro-AUCROC:{best_dic['test_micro_roc']:.4f} Macro-AUCROC:{best_dic['test_macro_roc']:.4f}\n"
               + f"Micro-AUCPR-P:{best_dic['test_micro_pr_']:.4f} Micro-AUCPR:{best_dic['test_micro_pr']:.4f} Macro-AUCPR:{best_dic['test_macro_pr']:.4f}\n"
               )

        print(msg)
        print(alpha_list)
        self.writer.add_text("Performance", msg)

        with open(os.path.join(dir_path, 'exp_info.json'), 'w', encoding='utf-8') as f:
            json.dump(info_dic, f)

        # save info dict
        dir_path = os.path.dirname(self.writer.log_dir)

    #  train one epoch of train/val/test
    def run(self, epoch, data_loader: Union[DataLoader, Workload], opt_metric, optimizer=None, namespace='train'):
        """
        :param epoch:
        :param data_loader:
        :param opt_metric: the loss function
        :param optimizer:
        :param namespace: train | valid | test
        :return:
        """

        self.net.to(self.args.device)
        if optimizer:
            self.net.train()
        else:
            self.net.eval()

        time_avg, timestamp = utils.AvgrageMeter(), time.time()
        loss_avg, auc_avg, pr_avg = utils.AvgrageMeter(
        ), utils.AvgrageMeter(), utils.AvgrageMeter()

        if namespace == 'train':

            # for batch_idx in range(self.args.iter_per_epoch):
            # todo: how to ensure the epoch traverse all combinations? re-use dataloader with sql aware?
            # randomly sample one batch
            # sql_batch, data_batch = data_loader.sample_batch_sql_and_data(self.args.batch_size)
            # sql_batch_tensor = torch.tensor(sql_batch).to(self.args.device)

            for batch_idx, data_batch in enumerate(data_loader):
                if batch_idx > self.args.iter_per_epoch:
                    break
                sql = data_batch["sql"].to(self.args.device)
                target = data_batch['y'].to(self.args.device)
                x_id = data_batch['id'].to(self.args.device)
                # x_value = data_batch['value'].to(self.args.device)

                # y = self.net((x_id, x_value), sql)
                y = self.net(x_id, sql)
                aux_loss = 0
                spa_loss = 0
                if isinstance(y, tuple):
                    y, (aux_loss, spa_loss) = y
                    a, b = self.args.beta * aux_loss, self.args.gamma * spa_loss
                    # print(f"aux_loss and sparse_loss: {a}, {b}")

                loss = opt_metric(y, target)
                loss += self.args.beta * aux_loss - self.args.gamma * spa_loss
                optimizer.zero_grad()

                # one step to update both hypernet and moenet
                loss.backward()
                optimizer.step()

                # if epoch == 1:
                #     print(f"alpha ", self.net.sparsemax.alpha)

                # logging
                # print(y)
                auc = utils.roc_auc_compute_fn(y, target)
                pr = utils.pr_auc_compute_fn(y, target)

                loss_avg.update(loss.item(), target.size(0))
                auc_avg.update(auc, target.size(0))
                pr_avg.update(pr, target.size(0))

                time_avg.update(time.time() - timestamp)
                if batch_idx % self.args.report_freq == 0:
                    print(f'Epoch [{epoch:3d}/{self.args.epoch}][{batch_idx:3d}/{len(data_loader)}]\t'
                          f'Time {time_avg.val} AUC {auc_avg.avg:.4f} PR {pr_avg.avg:.4f} '
                          f'Loss {loss_avg.avg:.4f}')

                step = epoch * len(data_loader) + batch_idx

        elif namespace == 'val':
            for batch_idx, data_batch in enumerate(data_loader):
                # sql = data_batch['default_sql'].to(self.args.device)
                sql = data_batch['sql'].to(self.args.device)
                target = data_batch['y'].to(self.args.device)
                x_id = data_batch['id'].to(self.args.device)
                # x_value = data_batch['value'].to(self.args.device)

                with torch.no_grad():
                    # y = self.net((x_id, x_value), sql)
                    y = self.net(x_id, sql)

                    aux_loss = 0
                    if isinstance(y, tuple):
                        y, (aux_loss, spa_loss) = y
                    loss = opt_metric(y, target) + self.args.beta * aux_loss

                # logging
                auc = utils.roc_auc_compute_fn(y, target)
                pr = utils.pr_auc_compute_fn(y, target)

                loss_avg.update(loss.item(), target.size(0))
                auc_avg.update(auc, target.size(0))
                pr_avg.update(pr, target.size(0))

                time_avg.update(time.time() - timestamp)

        else:
            # test
            workload: Workload = data_loader
            n = len(workload)
            y_workload_list = []
            target_workload_list = []
            auc_avg_, pr_avg_ = utils.AvgrageMeter(), utils.AvgrageMeter()
            for i in range(n):
                dataset, sql = workload[i]
                data_loader = DataLoader(
                    dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True)
                sql = sql.to(self.args.device)

                target_list = []
                y_list = []
                loss_list = []
                for batch_idx, data_batch in enumerate(data_loader):
                    target = data_batch['y'].to(self.args.device)
                    target_list.append(target)

                    x_id = data_batch['id'].to(self.args.device)
                    # x_value = data_batch['value'].to(self.args.device)

                    B = target.shape[0]
                    # print(f"x_id shape{x_id.shape}")
                    sql_ = sql.unsqueeze(0).expand(B, -1)  # [B, nfield]
                    with torch.no_grad():
                        # y = self.net((x_id, x_value), sql_)
                        y = self.net(x_id, sql_)
                        aux_loss = 0

                        if isinstance(y, tuple):
                            y, (aux_loss, spa_loss) = y
                        loss = opt_metric(y, target) + \
                            self.args.beta * aux_loss

                        y_list.append(y)
                        loss_list.append(loss.item())

                target = torch.cat(target_list, dim=0)  # [#n]
                y = torch.cat(y_list, dim=0)

                loss = sum(loss_list)
                loss_avg.update(loss, target.size(0))

                y_workload_list.append(y.detach().cpu())
                target_workload_list.append(target.detach().cpu())

                try:
                    auc = utils.roc_auc_compute_fn(y, target)
                    pr = utils.pr_auc_compute_fn(y, target)

                    auc_avg.update(auc, y.shape[0])
                    pr_avg.update(pr, y.shape[0])

                    auc_avg_.update(auc, 1)
                    pr_avg_.update(pr,  1)
                except ValueError:
                    # print("Skip this workload micro-evaluation")
                    pass

            y_workload = torch.cat(y_workload_list, dim=0)
            target_workload = torch.cat(target_workload_list, dim=0)

            auc = utils.roc_auc_compute_fn(y_workload, target_workload)
            pr = utils.pr_auc_compute_fn(y_workload, target_workload)

            print(f'\n--------------------Workload Test----------------------\n'
                  f'PureMICRO AUC-ROC {auc_avg_.avg:8.4f} Micro AUC-ROC {auc_avg.avg:8.4f} Macro AUC-ROC {auc:8.4f}\n'
                  f'PureMICRO AUC-PR {pr_avg_.avg:8.4f} Micro AUC-PR {pr_avg.avg:8.4f} Macro AUC-PR{pr:8.4f}\n'
                  f'-------------------------------------------------------\n'
                  )
            return (auc_avg.avg, auc_avg_.avg, auc), (pr_avg.avg, pr_avg_.avg, pr), loss_avg.avg

        print(f'{namespace}\tTime {utils.timeSince(s=time_avg.sum):>12s} '
              f'AUC {auc_avg.avg:8.4f} Loss {loss_avg.avg:8.4f}')

        return auc_avg.avg, pr_avg.avg, loss_avg.avg

    def close(self):
        self.writer.close()
