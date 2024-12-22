import json
import os
import torch
import time
from tqdm import tqdm
from dataprocess.context import sample_example, labr
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from metrics.mtrics_LinkResult import link_metrics
# OpenAI配置

def completion_with_backoff(in_data, retries=5, backoff_factor=0.5):
    for attempt in range(retries):
        try:
            return openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": in_data}],
                temperature=0.3,
            )
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
    return None

def call_qwen(prompt, tokenizer, model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            temperature=0.6,
            top_p=0.75,
            max_new_tokens=10
        )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def call_llama(prompt, tokenizer, model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    
    return tokenizer.decode(response, skip_special_tokens=True)

def process_entity(entity, threshold):
    if 'uncertainty' in entity and entity['uncertainty'] > threshold:
        name = list(entity.keys())[0]
        pred = entity['entity']
        return name, pred
    return None, None

def chatRes(content):
    response = completion_with_backoff(content)
    return response['choices'][0]['message']['content']

def worker(tokenizer, model, input_file, save_file, threshold, dic, data_name, shot, llm_name):
    results = []
    f = open(input_file).readlines()
   
    for line in tqdm(f):
        data = json.loads(line)
        sentence = data['sentence']
        entities = data['entity']
        updated_entities = []

        for entity in entities:
            attempts = 0
            name, pred = process_entity(entity, threshold)
            if name is not None:
                label_describe, label, label_set, label_num = labr(data_name)
                if shot == 0 or dic==None:
                    example = ''
                else:
                    example = sample_example(dic, shot, label_num)
                prompt = f"{label_describe}{example}{sentence}\nSelect the entity type of {pred} in this context, and only need to output the entity type: {label}\nAnswer:"
                llm_pred = None
                while attempts < 5 and not llm_pred:
                    # res = chatRes(query)
                    if llm_name == 'qwen':
                        res = call_qwen(prompt, tokenizer, model)
                    elif llm_name == 'llama':
                        res = call_llama(prompt, tokenizer, model)
                    elif llm_name == 'gpt-3.5':
                        res = chatRes(prompt)
                    else:
                        raise ValueError("llm name error!")
                    attempts += 1
                    for key in label_set:
                        if key.lower() == res.lower():
                            llm_pred = label_set[key]
                            break
                if not llm_pred:
                    llm_pred = ['O']
                entity['llm_pred'] = llm_pred
            else:
                entity['llm_pred'] = entity['pred']
            updated_entities.append(entity)

        data["entity"] = updated_entities
        results.append(data)
    link_metrics(results)
    with open(save_file, 'w') as outfile:
        for item in results:
            json.dump(item, outfile)
            outfile.write('\n')

    print(f"Done!🎆 Saved in {save_file}")
 

def linkToLLM(input_file, save_file, dic, args):
    if args.llm_name == 'qwen' or args.llm_name == 'llama':
        ckpt = args.llm_ckpt
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(ckpt, do_sample=True).to('cuda')
        worker(tokenizer, model, input_file, save_file, args.threshold, dic, data_name=args.linkDataName, shot=args.shot,llm_name=args.llm_name)
    else:
        import openai
        import api_config_example
        openai.api_type = api_config_example.API_TYPE
        openai.api_base = api_config_example.API_BASE
        openai.api_version = api_config_example.API_VERSION
        openai.api_key = api_config_example.API_KEY
        tokenizer = None
        model = None
        worker(tokenizer, model, input_file, save_file, args.threshold, dic, data_name=args.linkDataName, shot=args.shot,llm_name=args.llm_name)