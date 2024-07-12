import time 
import heapq 
import torch 
import torch.nn as nn 
from .data_utils import get_c4, get_wikitext2
from .LowRankLayer import LowRankLayer, LowRankLayerEval
from tqdm import tqdm
import numpy as np
import wandb
 
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    
def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res




def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def rank_analysis_weight(args, model, tokenizer, device):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers
    
    layers_singular_value = {}
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        layers_singular_value[i] = {}
        # Perform Singular Value Decomposition (SVD)
        for name in subset:
            W = subset[name].weight.data 
            _, singular_values, _ = torch.svd(W.to(torch.float32))
            layers_singular_value[i][name] = singular_values

    return layers_singular_value

def get_singular_values(args, model):
    layers = model.model.layers
    layers_singular_value = {}
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        
        # Perform Singular Value Decomposition (SVD)
        for name in subset:
            W = subset[name].weight.data 
            _, singular_values, _ = torch.svd(W.to(torch.float32))
            layers_singular_value[f"layer.{i}.{name}"] = singular_values

    return layers_singular_value


def get_grad_singular_values(args, model):
    layers = model.model.layers
    layers_singular_value = {}
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        
        # Perform Singular Value Decomposition (SVD)
        for name in subset:
            W = subset[name].weight.grad 
            _, singular_values, _ = torch.svd(W.to(torch.float32))
            layers_singular_value[f"layer.{i}.{name}"] = singular_values

    return layers_singular_value

def do_low_rank(weight, desired_rank, debug=False):

    results = torch.svd(weight)
    U = results[0][:, :desired_rank]
    S = results[1][:desired_rank]
    V = results[2][:, :desired_rank]

    loss = torch.nn.L1Loss()
    if debug:
        print(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")
    
    weight_approx = U @ torch.diag(S) @ V.T

    if debug:
        print(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]
    weight_approx = torch.nn.Parameter(weight_approx)

    with torch.no_grad():
        error = loss(weight, weight_approx)
    return weight_approx, error

def rank_reduction_weight(args, model, tokenizer, rank_pruning, device):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers
    layers_singular_value = {}

    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            k = min(W.shape[0], W.shape[1]) - rank_pruning[i][name]
            approx_w, error = do_low_rank(W.to(torch.float32), k, True)
            print(f"layer.{i}.{name} ({k}): {error}")

            subset[name].weight.data = approx_w.data.to(torch.bfloat16)

        if i == 0:
            break

    print("Pruning completed")
    return None, None

def rank_reduction_weight_wrapper(args, model, tokenizer, rank_pruning, device):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers
    layers_singular_value = {}

    for i in tqdm(range(len(layers))):
        layer = layers[i]

        attention = getattr(layer, 'self_attn')
        for key, module in attention.named_modules():
            if "proj" in key:
                name = 'self_attn.' + key
                k = min(module.weight.shape[0], module.weight.shape[1]) - rank_pruning[i][name]
                l = LowRankLayer(k, module.weight.to(torch.float32))
                setattr(attention, key, l)
                del module
        mlp = getattr(layer, 'mlp')
        for key, module in mlp.named_modules():
            if "proj" in key:
                name = 'mlp.' + key
                k = min(module.weight.shape[0], module.weight.shape[1]) - rank_pruning[i][name]
                l = LowRankLayer(k, module.weight.clone().to(torch.float32))
                setattr(mlp, key, l)
                del module
        # break
    print("Pruning completed")

def rank_reduction_weight_wrapper_selective(args, model, tokenizer, rank_pruning, device):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers
    layers_singular_value = {}
    reduced_rank, total_rank = 0, 0
    for i in tqdm(range(len(layers))):
        layer = layers[i]

        attention = getattr(layer, 'self_attn')
        for key, module in attention.named_modules():
            if "proj" in key:
                name = 'self_attn.' + key
                rank = min(module.weight.shape[0], module.weight.shape[1])
                k = rank - rank_pruning[i][name]
                if (rank_pruning[i][name] / rank) * 100 > 40:
                    l = LowRankLayer(k, module.weight.to(torch.float32), False)
                    setattr(attention, key, l)
                    del module
                    reduced_rank += rank_pruning[i][name]
                total_rank += rank
        mlp = getattr(layer, 'mlp')
        for key, module in mlp.named_modules():
            if "proj" in key:
                name = 'mlp.' + key
                rank = min(module.weight.shape[0], module.weight.shape[1])
                k = rank - rank_pruning[i][name]
                if (rank_pruning[i][name] / rank) * 100 > 40:
                    l = LowRankLayer(k, module.weight.clone().to(torch.float32), False)
                    setattr(mlp, key, l)
                    del module
                    reduced_rank += rank_pruning[i][name]
                total_rank += rank
        # break
    print(f">>>>>>>>>>>>>>> Pruning completed with Rank reduced : {(reduced_rank/total_rank) * 100}")
    return (reduced_rank/total_rank) * 100



def rank_reduction_weight_wrapper_selective_eval(args, model, tokenizer, rank_pruning, device):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers
    layers_singular_value = {}
    reduced_rank, total_rank = 0, 0
    for i in tqdm(range(len(layers))):
        layer = layers[i]

        attention = getattr(layer, 'self_attn')
        for key, module in attention.named_modules():
            if "proj" in key:
                name = 'self_attn.' + key
                rank = min(module.weight.shape[0], module.weight.shape[1])
                k = rank - rank_pruning[i][name]
                if (rank_pruning[i][name] / rank) * 100 > 40:
                    l = LowRankLayerEval(k, module.weight.to(torch.float32), False)
                    setattr(attention, key, l)
                    del module
                    reduced_rank += rank_pruning[i][name]
                total_rank += rank
        mlp = getattr(layer, 'mlp')
        for key, module in mlp.named_modules():
            if "proj" in key:
                name = 'mlp.' + key
                rank = min(module.weight.shape[0], module.weight.shape[1])
                k = rank - rank_pruning[i][name]
                if (rank_pruning[i][name] / rank) * 100 > 40:
                    l = LowRankLayerEval(k, module.weight.clone().to(torch.float32), False)
                    setattr(mlp, key, l)
                    del module
                    reduced_rank += rank_pruning[i][name]
                total_rank += rank
        # break
    print(f">>>>>>>>>>>>>>> Pruning completed with Rank reduced : {(reduced_rank/total_rank) * 100}")
    return (reduced_rank/total_rank) * 100


def rank_reduction_dynamic_pruning(args, model, device, file_name):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers

    rank_pruning = {}
    total_rank, error_thresold_att, error_thresold_ffn = 0, 5e-4, 5e-4
    pruning_bucket = [0.95, 0.9, 0.85, 0.8, 0.7, 0.75, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1]

    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        rank_pruning[i] = {}
        for name in subset:
            W = subset[name].weight.clone().data
            if "mlp" in name: error_thresold = error_thresold_ffn
            else: error_thresold = error_thresold_att
            rank_pruning[i][name] = 0
            for prune_ratio in pruning_bucket:
                desired_rank = int(min(W.shape[0], W.shape[1]) * prune_ratio)
                approx_w, error = do_low_rank(W.to(torch.float32), desired_rank, False)
                if error > error_thresold:
                    break
                else:
                    rank_pruning[i][name] = min(W.shape[0], W.shape[1]) - desired_rank
            total_rank += int(min(W.shape[0], W.shape[1]))
            print(f"layer.{i}.{name} ({rank_pruning[i][name]}): {error}")
    
    pruned_rank = 0
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            pruned_rank += rank_pruning[i][name]
    print("Pruning completed")
    torch.save(rank_pruning, "/data/adative_rank_attention_ffn.pt")
    print(f"Rank Reduction: {(pruned_rank/total_rank)* 100:.3f} %", file=file_name, flush=True)
    return rank_pruning