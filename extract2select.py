import json
from collections import defaultdict
import re
from tqdm import tqdm
import os
import random
import argparse

# # conll
# label_map = {"LOC": "Loction", "ORG": "Organization", "PER": "Person", "MISC": "Miscellaneous"}
# label_select = "\n Location\n Person\n Organization\n Miscellaneous\n Non-entity"
# label_list = ['LOC','PER','ORG','MISC','Non']


# # Ontonote 5 labels
# label_map = {"PERSON": "PERSON", "ORG": "ORGANIZATION", "GPE": "GPE", "DATE": "DATE", "NORP": "NORP", "CARDINAL": "CARDINAL", "TIME":"TIME", "LOC": "LOCTION", "FAC":"FACILITY", "PRODUCT":"PRODUCT", "WORK_OF_ART":"WORK_OF_ART", "MONEY":"MONEY", "ORDINAL":"ORDINAL", "QUANTITY":"QUANTITY", "EVENT":"EVENT", "PERCENT":"PERCENT", "LAW":"LAW", "LANGUAGE":"LANGUAGE"}
# label_select = "\n PERSON\n ORGANIZATION\n GPE\n DATE\n NORP\n CARDINAL\n TIME\n LOCTION\n FACILITY \n PRODUCT\n WORK_OF_ART\n MONEY\n ORDINAL\n QUANTITY\n EVENT\n PERCENT\n LAW\n LANGUAGE\n Non-entity"
# label_list = ['PERSON', 'ORG', 'GPE', 'DATE', 'NORP', 'CARDINAL', 'TIME', 'LOC', 'FAC', 'PRODUCT', 'WORK_OF_ART', 'MONEY', 'ORDINAL', 'QUANTITY', 'EVENT', 'PERCENT', 'LAW', 'LANGUAGE', 'Non']

# # wnut17 labels
# label_map = {"person": "Person", "creative_work": "Creative_work", "product": "Product", "location": "Location", "group": "Group", "corporation":"Corporation"}
# label_select = "\n Location\n Group\n Corporation\n Person\n Creative_work\n Product\n Non-entity"
# label_list = ['location', 'group','corporation','person','creative_work','product','Non']

# # tweet labels
# label_map = {"LOC": "Loction", "ORG": "Organization", "PER": "Person", "OTHER": "Other"}
# label = "\n Location\n Person\n Organization\n Other\n Non-entity"
# label_list = ['PER', 'LOC', 'OTHER', 'ORG', 'Non']

# Bioner
label_map = {"protein": "Protein", "DNA": "DNA", "cell_type": "Cell_type", "cell_line": "Cell_line", "RNA": "RNA"}
label = "\n Protein\n DNA\n Cell_type\n Cell_line\n RNA\n Non-entity"
label_list = ['protein', 'DNA', 'cell_type', 'cell_line', 'RNA', 'Non']
def ext2sel(input_dir, output_dir):
    sentence_dic = defaultdict(list)

    file = open(input_dir, 'r').read()
    dic = json.loads(file)
    data_length = (len(dic))
    context_len = 0
    all_datas = []

    for i in range(data_length):
        data = dic[i]
        context = data['context']
        label = data['Label']
        for idx in label.keys():
            query = "<Context>:" + context + "\n Select the entity type of "+str(idx)+" in this context, and only need to output the entity type."
            answer = "\nAnswer:" + label_map[label[idx]]
            example = {
                "question": query,
                "answer": answer
                }
            all_datas.append(example)

    with open(output_dir, "w") as f:
        json.dump(all_datas, f, ensure_ascii=False, indent=2)

def sample_example(dic, shot):
    data_length = (len(dic))
    example = []

    count = 0
    if shot<=0:
        return []
    else:
        set_example = (len(label_list) - 1)

    for i in range (shot):
        label = []
        while len(label)<set_example:
            idx = random.randint(0,data_length)
            data = dic[idx]
            if data['answer'] not in label:
                count += 1
                example.append("Example"+":\n"+data['question']+data['answer']+"\n")
                label.append(data['answer'])
            else:
                pass
    return("".join(example))

ext2sel("Bio-NER/Bio-NERNewspanner.dev", "Bio-NER/select.dev")