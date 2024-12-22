# encoding: utf-8
import argparse
from ast import arg
import os
from re import X
from typing import Dict
from models.framework import FewShotNERFramework
from dataloaders.spanner_dataset import get_span_labels, get_loader
from models.bert_model_spanner import BertNER
from transformers import AutoTokenizer
from models.config_spanner import BertNerConfig
import random
import logging
import torch
from models.Evidential_woker import Span_Evidence
logger = logging.getLogger(__name__)
import numpy as np
import time
from run_llm import *
from metrics.mtrics_LinkResult import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    torch.random.manual_seed(seed)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_logger(args, seed):
    path =args.logger_dir + args.etrans_func + "{}_{}.txt"
    pathname = path.format(seed, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def main():

    args: argparse.Namespace
    parser = argparse.ArgumentParser(description="Training")
    # fine-tuning local model basic argument&value

    parser.add_argument("--paradigm", default="span", type=str, help="data dir")
    parser.add_argument('--seed', default=123123, type=int, help='random seed')
    parser.add_argument("--test_mode", default="ori", type=str, help='Except conll can choose <ori>, <typos>, <oov>, <ood>, others can only choose the original setting <ori>')
    parser.add_argument("--dataname", choices=["conll03", "notev4", "wnut17", "twitter", "bioner"], default="notev4", type=str, help="the name of a dataset")
    parser.add_argument("--n_class", type=int, default=19, help="conll03: 5, notev4:19, wnut17: 7,twitter: 5, bioner: 6")
    parser.add_argument("--data_dir", default="data/notev4", type=str, help='The file path where the dataset is located')
    parser.add_argument("--state", choices=["train", "inference", "link"], default="train", type=str, help="Train or Inference or Link Models")

    parser.add_argument("--data_dir_typos", default="data/conll03/typos", type=str)
    parser.add_argument("--data_dir_oov", default="data/conll03/oov", type=str)
    parser.add_argument("--data_dir_ood", default="data/conll03/ood", type=str)
    parser.add_argument("--results_dir", default="results/", type=str)
    parser.add_argument("--logger_dir", default="log/", type=str)
    
    parser.add_argument('--gpu', type=str2bool, default=True, help='gpu')
    parser.add_argument('--iteration', default=30, type=int, help='num of iteration')
    parser.add_argument('--etrans_func', default='exp', type=str, help='type of evidence')
    parser.add_argument("--loss", default='edl', type=str, help='train cost function')
    parser.add_argument('--lr_scheulder', default='linear', type=str, help='(linear,StepLR,OneCycleLR,polydecay)')
    parser.add_argument('--with_uc',type=str2bool, default=True, help='whether the loss function uses regularization')
    parser.add_argument('--with_iw', type=str2bool, default=True, help='whether the loss function uses reweighting')
    parser.add_argument('--with_kl', type=str2bool, default=True, help='whether the loss function uses kl')
    parser.add_argument("--lr_mini", type=float, default=-1)
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--polydecay_ratio", type=float, default=4, help="ratio for polydecay learing rate scheduler.")
    parser.add_argument('--annealing_start', default=1e-5, type=float, help='Regularization settings')
    parser.add_argument('--annealing_step', default=70, type=float, help='Regularization settings')
    parser.add_argument('--regr', default=2, type=float, help='Regularization settings')
    parser.add_argument('--early_stop', default=5, type=float, help='early stop')
    parser.add_argument('--clip_grad', type=str2bool, default=False, help='clip grad')
    parser.add_argument('--load_ckpt', type=str2bool, default=True, help='save model')
    parser.add_argument('--save_result', type=str2bool, default=True, help='save model')
    parser.add_argument("--bert_config_dir", default="bert_large_uncased", type=str, help="bert config dir")
    parser.add_argument("--bert_max_length", default=128, type=int, help="max length of dataset")
    parser.add_argument("--batch_size", default=10, type=int, help="batch size")
    parser.add_argument("--lr", default=4e-6, type=float,help="learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps used for scheduler.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--model_dropout", type=float, default=0.2, help="model dropout rate")
    parser.add_argument("--bert_dropout", type=float, default=0.15, help="bert dropout rate")
    parser.add_argument("--final_div_factor", type=float, default=1e4, help="final div factor of linear decay scheduler")
    parser.add_argument("--optimizer", choices=["adamw", "sgd", "torch.adam"], default="adamw")

    parser.add_argument("--max_spanLen", type=int, default=5, help="max span length")
    parser.add_argument("--data_sign", type=str, default="en_conll", help="data signature for the dataset.")
    parser.add_argument("--classifier_sign", type=str, default="multi_nonlinear")
    parser.add_argument("--classifier_act_func", type=str, default="gelu")
    
    parser.add_argument('--ignore_index', type=int, default=-1,help='label index to ignore when calculating loss and metrics')
    parser.add_argument('--use_tokenLen', default=True, type=str2bool, help='use the token length (after the bert tokenizer process) as a feature',nargs='?',choices=['yes (default)', True, 'no', False])
    parser.add_argument("--tokenLen_emb_dim", default=60, type=int, help="the embedding dim of a span")
    parser.add_argument('--span_combination_mode', default='x,y', help='Train data in format defined by --data-io param.')
    parser.add_argument('--use_spanLen', type=str2bool, default=True, help='use the span length as a feature', nargs='?',choices=['yes (default)', True, 'no', False])
    parser.add_argument("--spanLen_emb_dim", type=int, default=100, help="the embedding dim of a span length")
    parser.add_argument('--use_morph', type=str2bool, default=True, help='use the span length as a feature', nargs='?',choices=['yes (default)', True, 'no', False])
    parser.add_argument("--morph_emb_dim", default=100, type=int,  help="the embedding dim of the morphology feature.")
    parser.add_argument('--morph2idx_list', type=list, help='a list to store a pair of (morph, index).', )
    parser.add_argument('--label2idx_list', type=list, help='a list to store a pair of (label, index).',)
    parser.add_argument('--param_name', type=str, default='param_name', help='a prexfix for a param file name', )
    parser.add_argument('--best_dev_f1', type=float, default=0.0, help='best_dev_f1 value', )
    parser.add_argument("--use_span_weight", type=str2bool, default=True
                        , help="range: [0,1.0], the weight of negative span for the loss.")
    parser.add_argument("--neg_span_weight", type=float,default=0.6, help="range: [0,1.0], the weight of negative span for the loss.")
    
    # linking to gpt3.5 basic argument&value
    parser.add_argument("--selectShot_dir", type=str, default="data/conll03/spanSelect.dev", help="few-shot data files, such as: data/conll03/spanSelect.dev.")
    parser.add_argument("--linkSave_dir", type=str, default="results/", help="After the results of gpt3.5 classification, the path to save the file")
    parser.add_argument("--threshold", type=float, default="0.4", help="Uncertainty threshold, used to confirm the uncertainty interval of local model and gpt3.5 classification.")

    args = parser.parse_args()

    if args.seed == -1:
        seed = '%08d' % (random.randint(0, 100000000))
        seed_num = int(seed)
    else:
        seed_num = int(args.seed)

    print('random_int:', seed_num)
    print("Seed num:", seed_num)
    setup_seed(seed_num)

    logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))

    num_labels = args.n_class
    task_idx2label = None
    args.label2idx_list, args.morph2idx_list = get_span_labels(args)
    
    bert_config = BertNerConfig.from_pretrained(args.bert_config_dir,
                                                        hidden_dropout_prob=args.bert_dropout,
                                                        attention_probs_dropout_prob=args.bert_dropout,
                                                        model_dropout=args.model_dropout)
    model = BertNER.from_pretrained(args.bert_config_dir,
                                                    config=bert_config,
                                                    args=args)
    model.cuda()
    
    train_data_loader = get_loader(args, args.data_dir, "train", True)
    dev_data_loader = get_loader(args, args.data_dir, "dev", False)
    if args.test_mode == 'ori':
        test_data_loader = get_loader(args, args.data_dir, "test", False)
    elif args.test_mode == 'typos' and args.dataname == 'conll03':
        test_data_loader = get_loader(args, args.data_dir_typos, "test", False)
    elif args.test_mode == 'oov' and args.dataname == 'conll03':
        test_data_loader = get_loader(args, args.data_dir_oov, "test", False)
    elif args.test_mode == 'ood' and args.dataname == 'conll03':
        test_data_loader = get_loader(args, args.data_dir_ood, "test", False)
    else:
        raise Exception("Invalid dataname or test_mode! Please check")

    edl = Span_Evidence(args, num_labels)
    logger = get_logger(args, seed_num)
    framework = FewShotNERFramework(args, 
                                    logger, 
                                    task_idx2label, 
                                    train_data_loader, 
                                    dev_data_loader, 
                                    test_data_loader, 
                                    edl, 
                                    seed_num, 
                                    num_labels=num_labels)
    
    # train
    if args.state == 'train':
        framework.train(model)
        logger.info("training is ended! 🎉")

    # inference
    if args.state == 'inference':
        model = torch.load(args.results_dir + args.dataname + str(seed_num) + 'model.pkl')
        framework.inference(model)
        logger.info("inference is ended!! 🎉")

    #link to llm
    if args.state == 'link':
        selectShot_dir =args.selectShot_dir
        save_dir = args.linkSave_dir + "link_result.txt"
        file = open(selectShot_dir, 'r')
        js = file.read()
        threshold = 0.4
        dic = json.loads(js)
        shot = 1
        GptForLink(args.results_dir+'conll03local_model.txt', dic, save_dir, threshold, shot, args.dataname)
        logger.info("linking is ended! 🎉")

if __name__ == '__main__':
    main()