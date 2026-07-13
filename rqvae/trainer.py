import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils import ensure_dir,set_color,get_local_time,delete_file
import os

import heapq
class Trainer(object):

    def __init__(self, args, model, data_num, start_epoch=1): # data_num: batch数量
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type

        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.start_epoch = start_epoch
        self.warmup_steps = args.warmup_epochs * data_num
        # Adjust warmup for resumed training: skip already-completed steps
        if start_epoch > 1:
            self.warmup_steps = max(0, self.warmup_steps - (start_epoch - 1) * data_num)
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = [] # 小顶堆 —— 保留碰撞率最低的 save_limit 个
        self.newest_save_queue = [] # FIFO 队列 —— 保留最新的 save_limit 个
        self.eval_step = min(args.eval_step, self.epochs) # 每eval_step轮eval一次
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir, saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_steps,
                                                           num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")


    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()
        has_collab = self.model.collab_weight > 0

        total_loss = 0.0
        total_recon = 0.0
        total_quant = 0.0
        total_cl = [0.0, 0.0, 0.0]
        total_div = [0.0, 0.0, 0.0]
        total_collab = 0.0
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )

        for _, batch in enumerate(iter_data):
            if has_collab:
                item_ids, data = batch
                item_ids = item_ids.to(self.device)
            else:
                data = batch
                item_ids = None
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, _, _, _, x_q_raw_list, cl_list, div_list = self.model(data)
            result = self.model.compute_loss(
                out, rq_loss, xs=data, cl_list=cl_list, div_list=div_list,
                x_q_raw_list=x_q_raw_list, item_ids=item_ids)
            (loss, loss_recon, loss_quant,
             _, cl_l1, cl_l2, cl_l3,
             _, div_l1, div_l2, div_l3,
             loss_collab) = result
            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_quant += loss_quant.item()
            total_cl[0] += cl_l1.item()
            total_cl[1] += cl_l2.item()
            total_cl[2] += cl_l3.item()
            total_div[0] += div_l1.item()
            total_div[1] += div_l2.item()
            total_div[2] += div_l3.item()
            total_collab += loss_collab.item()

        return total_loss, total_recon, total_quant, total_cl, total_div, total_collab

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()
        has_collab = self.model.collab_weight > 0

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )

        indices_set = set()
        num_sample = 0
        for _, batch in enumerate(iter_data):
            if has_collab:
                _, data = batch
            else:
                data = batch
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(list(indices_set)))/num_sample

        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

        return ckpt_path

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time,
                                     loss, recon_loss, quant_loss,
                                     cl_list, div_list, collab_loss):
        m = self.model
        parts = [
            set_color("epoch %d training" % epoch_idx, "green") + " [",
            set_color("time", "blue") + ": %.2fs" % (e_time - s_time),
        ]
        parts.append(set_color("train loss", "blue") + ": %.4f" % loss)
        if m.recon_weight > 0:
            parts.append(set_color("recon", "blue") + ": %.4f" % recon_loss)
        if m.quant_weight > 0:
            parts.append(set_color("quant", "blue") + ": %.4f" % quant_loss)
        for lvl, w in enumerate(m.cl_weights):
            if w > 0:
                parts.append(set_color("cl_L%d" % (lvl + 1), "blue")
                             + ": %.4f" % cl_list[lvl])
        for lvl, w in enumerate(m.div_weights):
            if w > 0:
                parts.append(set_color("div_L%d" % (lvl + 1), "blue")
                             + ": %.4f" % div_list[lvl])
        if m.collab_weight > 0:
            parts.append(set_color("collab", "blue")
                         + ": %.4f" % collab_loss)
        return ", ".join(parts) + "]"


    def fit(self, data):

        cur_eval_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs + 1):
            # train
            training_start_time = time()
            (train_loss, train_recon, train_quant,
             train_cl_list, train_div_list, train_collab) = \
                self._train_epoch(data, epoch_idx)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time,
                train_loss, train_recon, train_quant,
                train_cl_list, train_div_list, train_collab)
            self.logger.info(train_loss_output)


            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0 # 归零
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1


                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)
                ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
                now_save = (-collision_rate, ckpt_path)
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save) 
                else:
                    old_save = self.newest_save_queue.pop(0) # 移除最早保存的那个
                    self.newest_save_queue.append(now_save)
                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap) # 移除碰撞率最最高的那个
                        heapq.heappush(self.best_save_heap, now_save)

                        if bad_save not in self.newest_save_queue: # 如果不在FIFO中再删除文件
                            delete_file(bad_save[1])

                    if old_save not in self.best_save_heap: # 如果不在小顶堆中再删除文件
                        delete_file(old_save[1])

        # Save final model
        self._save_checkpoint(epoch=self.epochs - 1,
                              ckpt_file="last_epoch_model.pth")
        return self.best_loss, self.best_collision_rate




