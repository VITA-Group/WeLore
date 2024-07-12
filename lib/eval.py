import os
import time
import torch
import torch.nn as nn
import tqdm as tqdm
from loguru import logger
# Import get_loaders function from data module within the same directory
from .data_utils import get_c4, get_wikitext2


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, device=torch.device("cuda:0"), dataset="wikitext2"):
    # Set dataset

    # Print status
    logger.info(f"Evaluating on {dataset} .....")

    if os.path.exists("./data/test_loader.pt"):
        testloader = torch.load("./data/test_loader.pt")
    else:
        # Get the test loader
        _, testloader = get_loaders(
            dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
        )
        torch.save(testloader, "./data/test_loader.pt")

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl = eval_ppl_dataset(model, testloader, 1, device)
    return ppl 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_dataset(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []

    # nsamples = 10 #Sanity check

    # Loop through each batch
    for i in range(0, nsamples, bs):

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].cuda()
        inputs = inputs.reshape(j-i, model.seqlen)

        s_time = time.time()
        # Forward pass through the model
        lm_logits = model(inputs).logits
        e_time = time.time()

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

        if i % 20 == 0: logger.info(f"Evaluated samples: {i}/{nsamples}")

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # print(ppl)
    # Empty CUDA cache to save memory
    # torch.cuda.empty_cache()

    return ppl.item()