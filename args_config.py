# args_config.py
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser(description="Training")

    # --------------------- Basic Model Arguments ------------------------
    # 数据处理与模型选择相关的基本参数
    parser.add_argument('--seed', default=123123, type=int, help='random seed')
    parser.add_argument("--test_mode", default="ori", type=str, 
                        help='Except conll can choose <ori>, <typos>, <oov>, <ood>, others can only choose the original setting <ori>')

    # --------------------- Dataset Arguments ----------------------------
    # 数据集相关参数
    parser.add_argument("--dataname", choices=["conll03", "notev4", "wnut17", "twitter", "bioner"], 
                        default="conll03", type=str, help="the name of a dataset")
    parser.add_argument("--data_dir", default="data/conll03", type=str, help='The file path where the dataset is located')
    parser.add_argument("--data_dir_typos", default="data/conll03/typos", type=str)
    parser.add_argument("--data_dir_oov", default="data/conll03/oov", type=str)
    parser.add_argument("--data_dir_ood", default="data/conll03/ood", type=str)
    parser.add_argument('--n_class', type=int, default=5, help='Label numbers')

    # --------------------- Training State Arguments ---------------------------
    # 训练或推理状态相关参数
    parser.add_argument("--state", choices=["train", "inference", "llm_classify"], 
                        default="train", type=str, help="Train or Inference or Link Models")
    parser.add_argument("--inference_model", type=str, help="BERT configuration directory")
    parser.add_argument("--uncertainty_type", type=str, default="confidence")

    # --------------------- Logging and Result Directories -----------------------
    # 日志记录和结果保存相关参数
    parser.add_argument("--model_save_dir", default="results/", type=str)
    parser.add_argument("--logger_dir", default="log/", type=str)

    # --------------------- General Training Parameters ---------------------------
    # 一般训练参数
    parser.add_argument('--gpu', type=str2bool, default=True, help='Use GPU for training')
    parser.add_argument('--iteration', default=30, type=int, help='Number of iterations')
    parser.add_argument('--etrans_func', default='exp', type=str, help='Type of evidence used in training, choice in [softplus, exp, relu, softmax]')
    parser.add_argument("--loss", default='edl', type=str, help='Training cost function, choice in [ce, edl_mse, edl]')

    # Regularization and Weight Options if Loss == 'edl'
    parser.add_argument('--with_uc', type=str2bool, default=True, help='Whether to use regularization in the loss function')
    parser.add_argument('--with_iw', type=str2bool, default=True, help='Whether to use reweighting in the loss function')
    parser.add_argument('--with_kl', type=str2bool, default=True, help='Whether to use Kullback-Leibler divergence in the loss function')

    # Learning Rate Parameters
    parser.add_argument("--lr_mini", type=float, default=-1)
    parser.add_argument("--warmup_proportion", default=0.1, type=float, 
                        help="Proportion of training for linear learning rate warmup")
    parser.add_argument("--polydecay_ratio", type=float, default=4, help="Ratio for polydecay learning rate scheduler.")
    
    # Regularization Settings
    parser.add_argument('--annealing_start', default=1e-5, type=float, help='Regularization settings')
    parser.add_argument('--annealing_step', default=100, type=float, help='Regularization settings')
    parser.add_argument('--regr', default=2, type=float, help='Regularization settings')
    parser.add_argument('--early_stop', default=5, type=float, help='Early stopping strategy')
    parser.add_argument('--clip_grad', type=str2bool, default=False, help='Gradient clipping option')
    
    # Checkpoint and Result Saving
    parser.add_argument('--load_ckpt', type=str2bool, default=True, help='Load model checkpoint option')
    parser.add_argument('--results_dir', default="test_res/", type=str)

    # --------------------- BERT Model Configuration -------------------------
    # BERT模型相关参数
    parser.add_argument("--bert_config_dir", type=str,default=None,  help="BERT configuration directory")
    parser.add_argument("--bert_max_length", default=128, type=int, help="Maximum length of sequences")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training")
    parser.add_argument("--lr", default=3e-5, type=float, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if applied")
    parser.add_argument("--warmup_steps", default=100, type=int, help="Warmup steps used for learning rate scheduler")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer")
    parser.add_argument("--model_dropout", type=float, default=0.2, help="Dropout rate for the model")
    parser.add_argument("--bert_dropout", type=float, default=0.15, help="Dropout rate for the BERT model")
    parser.add_argument("--final_div_factor", type=float, default=1e4, help="Final division factor for linear decay scheduler")
    parser.add_argument("--optimizer", choices=["adamw", "sgd", "torch.adam"], default="adamw")

    # --------------------- Span Length and Classifier Settings ---------------
    # 与span长度和分类器相关的设置
    parser.add_argument("--max_spanLen", type=int, default=5, help="Maximum span length")
    parser.add_argument("--data_sign", type=str, default="en_conll", help="Data signature for the dataset")
    parser.add_argument("--classifier_sign", type=str, default="multi_nonlinear")
    parser.add_argument("--classifier_act_func", type=str, default="gelu")


    # --------------------- Additional Features and Settings -------------------
    # 其他功能和设置
    parser.add_argument('--ignore_index', type=int, default=-1, help='Label index to ignore when calculating loss and metrics')
    parser.add_argument('--use_tokenLen', default=True, type=str2bool, 
                        help='Use token length after BERT tokenizer process as a feature', nargs='?', choices=['yes (default)', True, 'no', False])
    parser.add_argument("--tokenLen_emb_dim", default=60, type=int, help="Embedding dimension of span length token")
    
    parser.add_argument('--span_combination_mode', default='x,y', 
                        help='Train data format defined by --data-io param.')
    
    parser.add_argument('--use_spanLen', type=str2bool, default=True, 
                        help='Use span length as a feature', nargs='?', choices=['yes (default)', True, 'no', False])
    parser.add_argument("--spanLen_emb_dim", type=int, default=100, help="Embedding dimension of span length")
    
    parser.add_argument('--use_morph', type=str2bool, default=True, 
                        help='Use morphology as a feature', nargs='?', choices=['yes (default)', True, 'no', False])
    parser.add_argument("--morph_emb_dim", default=100, type=int,  help="Embedding dimension of morphology feature")
    
    parser.add_argument('--morph2idx_list', type=list, help='List to store pairs of (morph, index)')
    parser.add_argument('--label2idx_list', type=list, help='List to store pairs of (label, index)')
    
    parser.add_argument('--param_name', type=str, default='param_name', help='Prefix for parameter filename')
    parser.add_argument('--best_dev_f1', type=float, default=0.0, help='Best development F1 score')
    
    parser.add_argument("--use_span_weight", type=str2bool, default=True, 
                        help="Weight of negative span for loss function")
    parser.add_argument("--neg_span_weight", type=float, default=0.6, help="Weight of negative span for loss function")

    # --------------------- Few-Shot Learning and Linking --------------------
    # LLM Few-shot学习和链接相关参数
    parser.add_argument("--selectShot_dir", type=str, default="data/conll03/spanSelect.dev", 
                        help="Few-shot data files, e.g., data/conll03/spanSelect.dev.")
    parser.add_argument("--llm_ckpt", type=str, help="llm hf ckpt")
    parser.add_argument("--llm_name", type=str, help="llm name")
    parser.add_argument("--linkDataName", type=str, help="link llm data name")

    parser.add_argument("--shot", default=0, type=int, help="Few-shot numbers")
    parser.add_argument("--input_file", type=str,  
                        help="Path to input files for GPT-3.5 classification.")
    parser.add_argument("--save_file", type=str, 
                        help="Path to save files after results from GPT-3.5 classification.")
    parser.add_argument("--threshold", type=float, default=0.4, help="Uncertainty threshold.")

    return parser.parse_args()
