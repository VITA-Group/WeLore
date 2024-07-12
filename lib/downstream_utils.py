import re
import torch
import json
import random
import numpy as np
import multiprocessing
from loguru import logger
from statistics import mean
from functools import partial
from torch.utils.data import Dataset
from .data_utils import fix_seed, shuffleDict
from peft_pretraining import args_utils

class MyDataset(Dataset):
    def __init__(self, args, eval=False):
        super().__init__()
        self.questions, self.answers, self.contexts = data_reader(args, eval)
        self.len = len(self.questions)
        

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        context = self.contexts[index]
        return input, output, context

def set_dataset_path(args):
    if args.dataset == "aqua":
        args.dataset_path = "./data/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./data/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./data/CommonsenseQA/train_rand_split.jsonl"
        args.val_dataset_path = "./data/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./data/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./data/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./data/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "svamp":
        args.dataset_path = "./data/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./data/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./data/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./data/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./data/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./data/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "boolq":
        args.dataset_path = "./data/boolq/boolq_train.jsonl"
        args.val_dataset_path = "./data/boolq/boolq_dev.jsonl"
        args.direct_answer_trigger = "\nTherefore, among True and False, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices True and False."
    else:
        raise ValueError("dataset is not properly defined ...")



def augument_args(args):
    if args.dataset is not None: set_dataset_path(args)
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"

    args = args_utils.check_args_torchrun_main(args)
    return args

def data_reader(args, eval=False):

    questions = []
    answers = []
    contexts = []
    decoder = json.JSONDecoder()

    dataset_path = args.dataset_path
    
    if args.dataset == "aqua":
      with open(dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "(" + "(".join(json_res["options"])
          choice = choice.replace("(", " (").replace(")", ") ")
          choice = "Answer Choices:" + choice
          questions.append(json_res["question"].strip() + " " + choice)
          answers.append(json_res["correct"])
  
    elif args.dataset == "gsm8k":
      with open(dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          questions.append(json_res["question"].strip())
          answers.append(json_res["answer"].split("#### ")[-1])
  
    elif args.dataset == "commonsensqa":
      if eval == True:
         dataset_path = args.val_dataset_path
      with open(dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "Answer Choices:"
          for c in json_res["question"]["choices"]:
              choice += " ("
              choice += c["label"]
              choice += ") "
              choice += c["text"]
          questions.append(json_res["question"]["stem"].strip() + " " + choice)
          answers.append(json_res["answerKey"])
          contexts.append("")

    elif args.dataset == "boolq":
      if eval == True:
         dataset_path = args.val_dataset_path
      with open(dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "Answer Choices: (a) True (b) False"
          questions.append(json_res["question"].strip().capitalize() + "?" + " " + choice)
          answers.append(str(json_res["answer"]))
          contexts.append(json_res["passage"].strip())

    elif args.dataset in ("addsub", "multiarith", "singleeq"):
      with open(dataset_path) as f:
        json_data = json.load(f)
        for line in json_data:
          q = line["sQuestion"].strip()
          a = str(line["lSolutions"][0])
          if a[-2:] == ".0":
              a = a[:-2]
          questions.append(q)
          answers.append(a)
        
    elif args.dataset == "strategyqa":
      with open(dataset_path) as f:
        json_data = json.load(f)["examples"]
        if eval == True: json_data = json_data[int(len(json_data) * 0.7): ]
        else: json_data = json_data[: int(len(json_data) * 0.7)]
        for line in json_data:
          q = line["input"].strip()
          a = int(line["target_scores"]["Yes"])
          if a == 1:
              a = "yes"
          else:
              a = "no"
          questions.append(q)
          answers.append(a)
          contexts.append("")
        
    elif args.dataset == "svamp":
      with open(dataset_path) as f:
        json_data = json.load(f)
        if eval == True: json_data = json_data[int(len(json_data) * 0.7): ]
        else: json_data = json_data[: int(len(json_data) * 0.7)]
        for line in json_data:
            q = line["Body"].strip() + " " + line["Question"].strip()
            a = str(line["Answer"])
            if a[-2:] == ".0":
                a = a[:-2]
            questions.append(q)
            answers.append(a)
            contexts.append("")

    elif args.dataset in ("bigbench_date", "object_tracking"):
      with open(dataset_path) as f:
        json_data = json.load(f)
        json_data = json_data["examples"]
        if args.dataset == "bigbench_date":
            choice_index = ['A','B','C','D','E','F']
        elif args.dataset in ("object_tracking"):
            choice_index = ['A','B','C']
        else:
            raise ValueError("dataset is not properly defined ...")
        if eval == True: json_data = json_data[int(len(json_data) * 0.8): ]
        else: json_data = json_data[: int(len(json_data) * 0.8)]
        for line in json_data:
          q = line["input"].strip()
          if args.dataset == "bigbench_date":
              choice = "Answer Choices:"
              # Randomly shuffle the answer choice dictionary because the original answer is always A ...
              choice_dic = shuffleDict(line["target_scores"])
          elif args.dataset == "object_tracking":
              choice = "\nWhich choice is true ? Answer Choices:"
              choice_dic = line["target_scores"]
          else:
              raise ValueError("dataset is not properly defined ...")
          for i, key_value in enumerate(choice_dic.items()):
              key, value = key_value
              choice += " ("
              choice += choice_index[i]
              choice += ") "
              choice += key
              if value == 1:
                  a = choice_index[i]
                  #a = key
          q = q + " " + choice
          questions.append(q)
          answers.append(a)
          contexts.append("")            
          
    elif args.dataset in ("coin_flip", "last_letters"):
      with open(dataset_path) as f:
        json_data = json.load(f)
        json_data = json_data["examples"]
        if eval == True: json_data = json_data[int(len(json_data) * 0.7): ]
        else: json_data = json_data[: int(len(json_data) * 0.7)]
        for line in json_data:
          q = line["question"]
          a = line["answer"]
          questions.append(q)
          answers.append(a)
          contexts.append("")
    else:
        raise ValueError("dataset is not properly defined ...")
    
    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)
    
    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))
    
    return questions, answers, contexts


def setup_data_loader(args, eval = False):

    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(args.seed)
    worker_seed = torch.initial_seed() % 2**32
    print("worker_seed : {}".format(worker_seed))
    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(worker_seed)
    
    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))
    
    dataset = MyDataset(args, eval)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                  shuffle=True,
                  batch_size=1,
                  drop_last=False,
                  num_workers=dataloader_num_workers,
                  worker_init_fn=seed_worker,
                  generator=g,
                  pin_memory=True)

    return dataloader


def dataset_generater(args, eval=False):
    dataloader = setup_data_loader(args, eval)
    question = []
    response = []
    context = []
    raw_x = []
    raw_y = []
    for i, data in enumerate(dataloader):
        x, y, z = data
        question.append("Question: " + x[0] + "\n")
        response.append(args.direct_answer_trigger_for_zeroshot + " " + y[0].strip() + ".")
        context.append(z[0])
        raw_x.append(x[0])
        raw_y.append(y[0])

    return {
            "question": question,
            "response": response,
            "context": context,
            "raw_x": raw_x,
            "raw_y": raw_y
        }


def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction', 'context', 'response')
    Then concatenate them using two newline characters 
    :param sample: Sample dictionnary
    """
    INTRO_BLURB = "Below is a question that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### "
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"
    
    blurb = f"{INTRO_BLURB}"
    if len(sample["context"]) > 1: 
       instruction = f"{INSTRUCTION_KEY}\n{sample['context']}\n\n{sample['question']}"
    else: 
       instruction = f"{INSTRUCTION_KEY}\n{sample['question']}"
    response = f"{RESPONSE_KEY}\n{sample['response']}"
    end = f"{END_KEY}"
    
    parts = [part for part in [blurb, instruction, response, end] if part]
    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt
    return sample


def create_prompt_formats_eval(sample):
    """
    Format various fields of the sample ('instruction', 'context', 'response')
    Then concatenate them using two newline characters 
    :param sample: Sample dictionnary
    """
    INTRO_BLURB = "Below is a question that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### "
    RESPONSE_KEY = "### Response:"
    END_KEY = ""
    
    blurb = f"{INTRO_BLURB}"
    if len(sample["context"]) > 1: instruction = f"{INSTRUCTION_KEY}\n{sample['context']}\n\n{sample['question']}"
    else: instruction = f"{INSTRUCTION_KEY}\n{sample['question']}"
    response = f"{RESPONSE_KEY}\n{sample['response'].split('is')[0] + 'is '}"
    end = f"{END_KEY}"
    
    parts = [part for part in [blurb, instruction, response] if part]
    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt
    return sample


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

def preprocess_dataset(tokenizer, max_length: int, seed, dataset: str):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """
    
    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)#, batched=True)
    
    logger.info(f"Prompt Sample: {dataset['text'][0]}")
    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["question", "response", "raw_x", "raw_y", "text"],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

def parse_predicted(args, prompt, outputs):
    predicted_response = ""
    if args.dataset == "commonsensqa":
        predicted_response = re.findall(r'A|B|C|D|E', outputs.replace(prompt, ""))[0]
    elif args.dataset == "boolq":
        predicted_response = re.findall(r'True|False', outputs.replace(prompt, ""))[0]
    elif args.dataset == "svamp":
        predicted_response = [s for s in re.findall(r'-?\d+\.?\d*', outputs.replace(prompt, ""))][0]
    elif args.dataset == "coin_flip":
        predicted_response = re.findall(r'yes|no', outputs.replace(prompt, ""))[0]
    elif args.dataset == "object_tracking":
        predicted_response = re.findall(r'A|B|C', outputs.replace(prompt, ""))[0]
    elif args.dataset == "bigbench_date":
        predicted_response = re.findall(r'A|B|C|D|E|F', outputs.replace(prompt, ""))[0]
    elif args.dataset == "strategyqa":
        predicted_response = re.findall(r'yes|no', outputs.replace(prompt, ""))[0]    
    return predicted_response


def match_response(args, pred, correct):
    if args.dataset == "commonsensqa":
        if pred == correct: return 1
    elif args.dataset == "boolq":
        if pred == correct: return 1
    elif args.dataset == "svamp":
        if float(pred) == float(correct): return 1
    elif args.dataset == "coin_flip":
        if pred == correct: return 1
    elif args.dataset == "object_tracking":
        if pred == correct: return 1
    elif args.dataset == "bigbench_date":
        if pred == correct: return 1
    elif args.dataset == "strategyqa":
        if pred == correct: return 1
    return 0