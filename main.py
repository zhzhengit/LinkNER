# encoding: utf-8
import argparse
import os
import random
import logging
import torch
import json
from src.framework import FewShotNERFramework
from dataloaders.spanner_dataset import get_span_labels, get_loader
from src.bert_model_spanner import BertNER
from transformers import AutoTokenizer
from src.config_spanner import BertNerConfig
from src.Evidential_woker import Span_Evidence
from metrics.mtrics_LinkResult import *
from args_config import get_args
from run_llm import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(args, seed):
    path = os.path.join(args.logger_dir, f"{args.etrans_func}{seed}_{time.strftime('%m-%d_%H-%M-%S')}.txt")
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def main():
    args = get_args()
    
    if args.seed == -1:
        seed_num = random.randint(0, 100000000)
    else:
        seed_num = int(args.seed)

    print('random_int:', seed_num)
    print("Seed num:", seed_num)
    setup_seed(seed_num)

    logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))

    if args.state == 'llm_classify':
        dic = json.load(open(args.selectShot_dir)) if args.selectShot_dir and args.selectShot_dir != 'None' else None
        linkToLLM(args.input_file, args.save_file, dic, args)
        return

    num_labels = args.n_class
    task_idx2label = None
    args.label2idx_list, args.morph2idx_list = get_span_labels(args)
    
    bert_config = BertNerConfig.from_pretrained(
        args.bert_config_dir,
        hidden_dropout_prob=args.bert_dropout,
        attention_probs_dropout_prob=args.bert_dropout,
        model_dropout=args.model_dropout
    )
    model = BertNER.from_pretrained(args.bert_config_dir, config=bert_config, args=args)
    model.cuda()

    train_data_loader = get_loader(args, args.data_dir, "train", True)
    dev_data_loader = get_loader(args, args.data_dir, "dev", False)
    test_data_loader = get_loader(args, args.data_dir, "test", False)
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
    framework = FewShotNERFramework(
        args, 
        logger, 
        task_idx2label, 
        train_data_loader, 
        dev_data_loader, 
        test_data_loader, 
        edl, 
        seed_num, 
        num_labels=num_labels
    )

    if args.state == 'train':
        framework.train(model)
        logger.info("training is ended! 🎉")

    if args.state == 'inference':
        model = torch.load(args.inference_model)
        framework.inference(model)
        logger.info("inference is ended!! 🎉")

if __name__ == '__main__':
    main()
