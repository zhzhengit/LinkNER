import os
import re
# import openai
from tqdm import tqdm
import json
'''
After getting the LinkNER result, 
evaluate it through the metrics("linkner_result.dir") function in this file,
'''

# def link(result):
#     tp = 0
#     tmp_pred_cnt = 0
#     tmp_label_cnt = 0

#     trup_sent = len(result)
#     if trup_sent > 1:
#         for idx in(result[1:]):
#             if '<Context>' not in idx[0]:
#                 idx = idx[0].split(' ')
#                 if str(idx[-7]) == str(idx[-6]):
#                     tp += 1
#                 if str(idx[-7]) != 'O':
#                     tmp_label_cnt += 1
#                 if str(idx[-6]) != 'O':
#                     tmp_pred_cnt += 1
#                 else:
#                     pass
#     return tp, tmp_pred_cnt, tmp_label_cnt     

# def link_metrics(file_dir):
#     print('Precision ','Recall ','F1 ')

#     stage2_result = []
#     # Open the file in read mode
#     with open(file_dir, 'r') as f:
#         TP, TPC, TLC = 0, 0, 0
#         count = 0
#         for line in f.readlines():
#             line = line.strip('\t\n')
#             line = str(line)
#             sentence = re.split(':: |\t|"', line)
#             ## 判断是否是句子
#             if '<Context>' in line:
#                 count += 1
#                 if count > 1:
#                     tp, tmp_pred_cnt, tmp_label_cnt  = link(stage2_result)
#                     TP += tp
#                     TPC += tmp_pred_cnt
#                     TLC += tmp_label_cnt
#                     stage2_result = []
#                     stage2_result.append(sentence)
#                 else:
#                     stage2_result.append(sentence)
#             else:
#                 stage2_result.append(sentence) 
#         tp, tmp_pred_cnt, tmp_label_cnt = link(stage2_result)
#         TP += tp
#         TPC += tmp_pred_cnt
#         TLC += tmp_label_cnt
#         precision = TP/(TPC + 1e-5)
#         recall = TP/(TLC+1e-5)
#         f1 = 2*precision*recall/(precision+recall+1e-5)
#         print(precision, recall, f1)

        
def link_metrics(res_dict):
    stage2_result = []
    tp = 0
    tmp_pred_cnt = 0
    tmp_label_cnt = 0
    for res in res_dict:
        for entity in res['entity']:
            if entity['llm_pred'] == entity['answer']:
                tp += 1
            if entity['answer'] != 'O':
                tmp_label_cnt += 1
            if entity['llm_pred'] != 'O':
                tmp_pred_cnt += 1
    precision = tp/(tmp_pred_cnt+1e-5)
    recall = tp/(tmp_label_cnt+1e-5)
    f1 = 2*precision*recall/(precision+recall+1e-5)
    print(f"Precision: {round(precision, 4)}\nRecall: {round(recall, 4)}\nF1: {round(f1, 4)}")