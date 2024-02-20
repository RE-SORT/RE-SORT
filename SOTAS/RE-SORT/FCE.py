import os
import sys
import math
import numpy as np
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from re_sortctr.metrics import evaluate_metrics
from re_sortctr.pytorch.torch_utils import get_device, get_optimizer, get_loss, get_regularizer
from re_sortctr.utils import Monitor


def sd(x):
    return np.std(x, axis=0, ddof=1)


def sd_gpu(x):
    return torch.std(x, dim=0)


def normalize_gpu(x):
    x = F.normalize(x, p=1, dim=1)
    return x


def normalize(x):
    mean = np.mean(x, axis=0)
    std = sd(x)
    std[std == 0] = 1
    x = (x - mean) / std
    return x


def random_fourier_features_gpu(x, w=None, b=None, num_f=None, sum=True, sigma=None, seed=None):
    if num_f is None:
        num_f = 1
    n = x.size(0)
    r = x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:
        w = 1 / sigma * (torch.randn(size=(num_f, c)))
        b = 2 * np.pi * torch.rand(size=(r, num_f))
        b = b.repeat((n, 1, 1))
    Z = torch.sqrt(torch.tensor(2.0 / num_f).cuda())
    mid = torch.matmul(x.cuda(), w.t().cuda())
    mid = mid + b.cuda()
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0].cuda()
    mid *= np.pi / 2.0
    if sum:
        Z = Z * (torch.cos(mid).cuda() + torch.sin(mid).cuda())
    else:
        Z = Z * torch.cat((torch.cos(mid).cuda(), torch.sin(mid).cuda()), dim=-1)
    return Z


def lossc(inputs, target, weight):
    loss = nn.NLLLoss(reduce=False)
    return loss(inputs, target).view(1, -1).mm(weight).view(1)


def cov(x, w=None):
    if w is None:
        n = x.shape[0]
        cov = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        w = w.view(-1, 1)
        cov = torch.matmul((w * x).t(), x)
        e = torch.sum(w * x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    return res


def lossb_expect(cfeaturec, weight, num_f, sum=True):
    cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=num_f, sum=sum).cuda()
    loss = Variable(torch.FloatTensor([0]).cuda())
    weight = weight.cuda()
    for i in range(cfeaturecs.size()[-1]):
        cfeaturec = cfeaturecs[:, :, i]
        cov1 = cov(cfeaturec, weight)
        cov_matrix = cov1 * cov1
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)
    return loss


def lossq(cfeatures, cfs):
    return - cfeatures.pow(2).sum(1).mean(0).view(1) / cfs


def lossn(cfeatures):
    return cfeatures.mean(0).pow(2).mean(0).view(1)


def lr_setter(optimizer, epoch, args, bl=False):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args['lr']
    if bl:
        lr = args['lrbl'] * (0.1 ** (epoch // (args['epochb'] * 0.5)))
    else:
        if args['cos']:
            lr *= ((0.01 + math.cos(0.5 * (math.pi * epoch / args['epochs']))) / 1.01)
        else:
            if epoch >= args['epochs_decay'][0]:
                lr *= 0.1
            if epoch >= args['epochs_decay'][1]:
                lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weight_learner(cfeatures, pre_features, pre_weight1, args, global_epoch=0, iter=0):
    softmax = nn.Softmax(0)
    weight = Variable(torch.ones(cfeatures.size()[0], 1).cuda())
    weight.requires_grad = True
    cfeaturec = Variable(torch.FloatTensor(cfeatures.size()).cuda())
    cfeaturec.data.copy_(cfeatures.data)
    all_feature = torch.cat([cfeaturec, pre_features.detach()], dim=0)
    optimizerbl = torch.optim.SGD([weight], lr=args['lrbl'], momentum=args['momentum'])
    for epoch in range(args['epochb']):
        lr_setter(optimizerbl, epoch, args, bl=True)
        all_weight = torch.cat((weight, pre_weight1.detach()), dim=0)
        optimizerbl.zero_grad()
        lossb = lossb_expect(all_feature, softmax(all_weight), args['num_f'], args[
            'sum'])
        lossp = softmax(weight).pow(args['decay_pow']).sum()
        lambdap = args['lambdap'] * max((args['lambda_decay_rate'] ** (global_epoch // args['lambda_decay_epoch'])),
                                        args['min_lambda_times'])
        lossg = lossb / lambdap + lossp
        if global_epoch == 0:
            lossg = lossg * args['first_step_cons']
        lossg.backward(retain_graph=True)
        optimizerbl.step()
    if global_epoch == 0 and iter < 10:
        pre_features = (pre_features * iter + cfeatures) / (iter + 1).cuda()
        pre_weight1 = (pre_weight1 * iter + weight) / (iter + 1)
        # print("first")
    elif cfeatures.size()[0] < pre_features.size()[0]:
        pre_features[:cfeatures.size()[0]] = pre_features[:cfeatures.size()[0]].clone() * args[
            'presave_ratio'] + cfeatures.clone() * (
                                                     1 - args['presave_ratio'])
        pre_weight1[:cfeatures.size()[0]] = pre_weight1[:cfeatures.size()[0]].clone() * args[
            'presave_ratio'] + weight.clone() * (
                                                    1 - args['presave_ratio'])
    else:

        pre_features = pre_features * args['presave_ratio'] + (cfeatures * (1 - args['presave_ratio'])).cuda()
        pre_weight1 = pre_weight1 * args['presave_ratio'] + weight * (1 - args['presave_ratio'])

    softmax_weight = softmax(weight)
    return softmax_weight, pre_features, pre_weight1


class FCEModel(nn.Module):
    def __init__(self,
                 feature_map,
                 model_id="FCEModel",
                 task="binary_classification",
                 gpu=-1,
                 monitor="AUC",
                 save_best_only=True,
                 monitor_mode="max",
                 early_stop_patience=2,
                 eval_steps=None,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 reduce_lr_on_plateau=True,
                 **kwargs):
        super(FCEModel, self).__init__()
        self.device = get_device(gpu)
        self.gpu = gpu
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._early_stop_patience = early_stop_patience
        self._eval_steps = eval_steps  # None default, that is evaluating every epoch
        self._save_best_only = save_best_only
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._verbose = kwargs["verbose"]
        self.feature_map = feature_map
        self.output_activation = self.get_output_activation(task)
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + ".model"))
        self.validation_metrics = kwargs["metrics"]
        self.feat1 = None
        self.feat2 = None
        self.flat_emb = None
        self.weight1 = 1
        self.args = kwargs['args']
        self.epoch = kwargs['epochs']
        self.count = 0

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        self.loss_fn = get_loss(loss)

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        conf = self.flat_emb

        if (self.pre_features != None):
            pre_features = torch.tensor(self.pre_features).cuda()
        else:
            pre_features = torch.tensor(self.flat_emb).cuda()

        if (self.pre_weight1 != None):
            pre_weight1 = torch.tensor(self.pre_weight1).cuda()
        else:
            pre_weight1 = torch.tensor(torch.zeros((conf.size(0), 1))).cuda()
        self.weight1, self.pre_features, self.pre_weight1 = weight_learner(conf, pre_features, pre_weight1, self.args,
                                                                           self.epoch, self.count)

        loss = (self.loss_fn(return_dict["y_pred"], y_true, reduction='none')).cuda().view(1, -1).mm(
            self.weight1.cuda()).view(1).cuda()
        return loss

    def add_regularization(self):
        reg_term = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if "embedding_layer" in name:
                        if self._embedding_regularizer:
                            for emb_p, emb_lambda in emb_reg:
                                reg_term += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
                    else:
                        if self._net_regularizer:
                            for net_p, net_lambda in net_reg:
                                reg_term += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_term

    def get_total_loss(self, inputs):
        total_loss = self.add_loss(inputs) + self.add_regularization()
        return total_loss

    def reset_parameters(self):
        def reset_default_params(m):
            # initialize nn.Linear/nn.Conv1d layers by default
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        def reset_custom_params(m):
            # initialize layers with customized reset_parameters
            pass

        self.apply(reset_default_params)
        self.apply(reset_custom_params)

    def get_inputs(self, inputs, feature_source=None):
        if feature_source and type(feature_source) == str:
            feature_source = [feature_source]
        X_dict = dict()
        for feature, spec in self.feature_map.features.items():
            if (feature_source is not None) and (spec["source"] not in feature_source):
                continue
            if spec["type"] == "meta":
                continue
            X_dict[feature] = inputs[:, self.feature_map.get_column_index(feature)].to(self.device)
        return X_dict

    def get_labels(self, inputs):
        labels = self.feature_map.labels
        assert len(labels) == 1, "Please override get_labels(), add_loss(), evaluate() when using multiple labels!"
        y = inputs[:, self.feature_map.get_column_index(labels[0])].to(self.device)
        return y.float().view(-1, 1)

    def get_group_id(self, inputs):
        return inputs[:, self.feature_map.get_column_index(self.feature_map.group_id)]

    def model_to_device(self):
        self.to(device=self.device)

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            reduced_lr = max(param_group["lr"] * factor, min_lr)
            param_group["lr"] = reduced_lr
        return reduced_lr

    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self.train_epoch(data_generator)
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

    def checkpoint_and_earlystop(self, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
                (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({})={:.6f} STOP!".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.lr_decay()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({})={:.6f}" \
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps >= self._early_stop_patience:
            self._stop_training = True
            logging.info("********* Epoch=={} early stop *********".format(self._epoch_index + 1))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)

    def eval_step(self):
        logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())
        self.checkpoint_and_earlystop(val_logs)
        self.train()

    def score(self, X, y):
        return self.get_total_loss(X)

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        loss = self.get_total_loss(batch_data)
        # print("loss=",loss)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0
        self.train()
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)

        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss = self.train_step(batch_data)
            train_loss += loss.item()
            if self._total_steps % self._eval_steps == 0:
                logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                train_loss = 0
                self.eval_step()
            if self._stop_training:
                break
        self.count += 1
        print("self.count=", self.count)

    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            group_id = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None
            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            return val_logs

    def predict(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            return y_pred

    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None):
        return evaluate_metrics(y_true, y_pred, metrics, group_id)

    def save_weights(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)

    def get_output_activation(self, task):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "regression":
            return nn.Identity()
        else:
            raise NotImplementedError("task={} is not supported.".format(task))

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters():
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))
