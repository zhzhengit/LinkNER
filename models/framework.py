from ast import arg
from re import A
import torch
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim import SGD
import time
import prettytable as pt
import torch.optim as optim
import numpy as np
from metrics.function_metrics import span_f1_prune, ECE_Scores, get_predict_prune

class FewShotNERFramework:

    def __init__(self, args, logger, task_idx2label, train_data_loader, val_data_loader, test_data_loader, edl, seed_num, num_labels):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.logger = logger
        # self.gpu = True
        self.seed = seed_num
        self.args = args
        self.eps = 1e-10
        self.learning_rate = args.lr
        self.load_ckpt=args.load_ckpt
        self.optimizer = args.optimizer
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.loss = args.loss
        self.annealing_start = 1e-6
        self.epoch_num = args.iteration
        self.edl = edl
        self.num_labels = num_labels
        self.task_idx2label = task_idx2label
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()
            
    def metric(self,
                model,
                eval_dataset,
                mode): 
            '''
            model: a FewShotREModel instance
            B: Batch size
            N: Num of classes for each batch
            K: Num of instances for each class in the support set
            Q: Num of instances for each class in the query set
            eval_iter: Num of iterations
            ckpt: Checkpoint path. Set as None if using current model parameters.
            return: Accuracy
            '''
            pred_cnt = 0 # pred entity cnt
            label_cnt = 0 # true label entity cnt
            correct_cnt = 0 # correct predicted entity cnt
            context_results = []
            predict_results = []
            prob_results = []
            uncertainty_results = []
            
            with torch.no_grad():
                for it, data in enumerate(eval_dataset):
                    
                    gold_tokens_list = []
                    pred_scores_list = []
                    pred_list = []
                    batch_soft = []
                    
                    if self.args.paradigm == 'span':
                        tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,real_span_mask_ltoken,words,all_span_word,all_span_idxs =  data
                        loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,
                                real_span_mask_ltoken, words, all_span_word, all_span_idxs]
                        attention_mask = (tokens != 0).long()
                        logits = model(loadall,all_span_lens,all_span_idxs_ltoken,tokens, attention_mask, token_type_ids)
                        predicts, uncertainty = self.edl.pred(logits)
                        correct, tmp_pred_cnt, tmp_label_cnt = span_f1_prune(predicts, span_label_ltoken, real_span_mask_ltoken)
                        pred_cls, pred_scores, tgt_cls = self.edl.ece_value(logits, span_label_ltoken, real_span_mask_ltoken)

                        prob, pred_id = torch.max(predicts, 2)
                        batch_results = get_predict_prune(self.args.label2idx_list, all_span_word, words, pred_id, span_label_ltoken, all_span_idxs, prob, uncertainty)
                        pred_cnt += tmp_pred_cnt
                        label_cnt += tmp_label_cnt
                        correct_cnt += correct
                        batch_soft += span_label_ltoken
                        prob_results += prob

                        predict_results += pred_id
                        uncertainty_results += uncertainty
                        
                        context_results += batch_results

                    else:
                        return ValueError

                    pred_list.append(pred_cls)
                    pred_scores_list.append(pred_scores)
                    gold_tokens_list.append(tgt_cls)
                    
                    
                gold_tokens_cat = torch.cat(gold_tokens_list, dim=0)
                pred_scores_cat = torch.cat(pred_scores_list, dim=0)
                pred_cat = torch.cat(pred_list, dim=0)
                
                ece = ECE_Scores(pred_cat, gold_tokens_cat, pred_scores_cat)
                precision = correct_cnt / (pred_cnt + 0.0)
                recall = correct_cnt / (label_cnt + 0.0)
                f1 = 2 * precision * recall / (precision + recall+float("1e-8"))
                if mode == 'test' and self.args.save_result:
                    results_dir = self.args.results_dir + self.args.dataname + 'local_model.txt'
                    sent_num = len(context_results)
                    fout = open(results_dir, 'w', encoding='utf-8')
                    for idx in range(sent_num):
                        fout.write("".join(context_results[idx]))
                        fout.write('\n')
                    fout.close()
                
                return precision, recall, f1, ece
    
    def eval(self,
            model,
            mode=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        if mode == 'dev':
            self.logger.info("Use val dataset")
            precision, recall, f1, ece = self.metric(model, self.val_data_loader, mode='dev')
            self.logger.info('{} Label F1 {}'.format("dev", f1))
            table = pt.PrettyTable(["{}".format("dev"), "Precision", "Recall", 'F1', 'ECE'])
        
        elif mode == 'test':
            self.logger.info("Use " + str(self.args.test_mode)+ " test dataset")
            precision, recall, f1, ece = self.metric(model, self.test_data_loader, mode='test')
            self.logger.info('{} Label F1 {}'.format("test", f1))
            table = pt.PrettyTable(["{}".format("test"), "Precision", "Recall", 'F1', 'ECE'])
            
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [precision, recall, f1, ece]])
        self.logger.info("\n{}".format(table))
        return f1, ece
                
    def train(self,
              model
              ): 
        self.logger.info("Start training...")

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=self.learning_rate,
                              eps=self.args.adam_epsilon,)

        elif self.optimizer == "sgd":
            optimizer = SGD(optimizer_grouped_parameters, self.learning_rate, momentum=0.9)
        
        elif self.optimizer == "torch.adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                        lr=self.args.lr,
                                        eps=self.args.adam_epsilon,
                                        weight_decay=self.args.weight_decay)

        t_total = len(self.train_data_loader) * self.args.iteration
        warmup_steps = int(self.args.warmup_proportion * t_total)

        if self.args.lr_scheulder == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total) 
        elif self.args.lr_scheulder == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 1e-8)
        elif self.args.lr_scheulder == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.learning_rate, pct_start=float(self.args.warmup_steps/t_total),
                final_div_factor=self.args.final_div_factor,
                total_steps=t_total, anneal_strategy='linear'
            )
        elif self.args.lr_scheulder == "polydecay":
            if self.args.lr_mini == -1:
                lr_mini = self.args.lr / self.args.polydecay_ratio
            else:
                lr_mini = self.args.lr_mini
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, t_total, lr_end=lr_mini)
        else:
            raise ValueError

        model.train()
        # Training
        best_f1 = 0.0
        best_step = 0
        iter_loss = 0.0

        for idx in range(self.args.iteration):
            pred_cnt = 0
            label_cnt = 0
            correct_cnt = 0
            epoch_start = time.time()
            self.logger.info("training...")

            for it in range(len(self.train_data_loader)):
                loss = 0
                if self.args.paradigm == 'span':
                    tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,real_span_mask_ltoken,words,all_span_word,all_span_idxs =  next(iter(self.train_data_loader))
                    loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,
                            real_span_mask_ltoken, words, all_span_word, all_span_idxs]
                    attention_mask = (tokens != 0).long()
                    logits = model(loadall,all_span_lens,all_span_idxs_ltoken,tokens, attention_mask, token_type_ids)
                    loss, pred = self.edl.loss(logits, loadall, span_label_ltoken, real_span_mask_ltoken, idx)
                    correct, tmp_pred_cnt, tmp_label_cnt = span_f1_prune(pred, span_label_ltoken, real_span_mask_ltoken)
                    
                    pred_cnt += tmp_pred_cnt
                    label_cnt += tmp_label_cnt
                    correct_cnt += correct
                    
                else:
                    return ValueError
                
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                scheduler.step()
                iter_loss += self.item(loss.data)

            epoch_finish = time.time()
            epoch_cost = epoch_finish - epoch_start
            
            precision = correct_cnt / pred_cnt + 0.
            recall = correct_cnt / label_cnt + 0.
            f1 = 2 * precision * recall / (precision + recall+float("1e-8"))

            self.logger.info("Time '%.2f's" % epoch_cost)

            self.logger.info('step: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                .format(idx+1, iter_loss, precision, recall, f1) + '\r')

            if (idx + 1) % 1 == 0:
                f1, ece = self.eval(model, mode = 'dev')
                self.inference(model)
                if f1>best_f1:
                    best_step = idx + 1
                    best_f1 = f1
                    if self.args.load_ckpt:
                        torch.save(model, self.args.results_dir + str(self.args.seed)+'notel5model.pkl')

                if (idx+1) > best_step + self.args.early_stop:
                    self.logger.info('earlt stop!')
                    return

            iter_loss = 0.
            pred_cnt = 0
            label_cnt = 0
            correct_cnt = 0

    def inference(self, model):
        f1, ece = self.eval(model, mode = 'test')