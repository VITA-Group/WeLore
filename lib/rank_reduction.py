import time 
import heapq 
import torch 
import torch.nn as nn 
from .data_utils import get_c4, get_wikitext2
from .LowRankLayer import LowRankLayer, LowRankLayerEval
from tqdm import tqdm
import numpy as np

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

def do_rank_reduction(args, model, tokenizer, rank_pruning, min_ratio, logger = None, load_only = False):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers
    reduced_rank, total_rank = 0, 0
    
    logger.info("*************** Pruning Model Started ***************")
    for i in range(len(layers)):
        layer = layers[i]
        attention = getattr(layer, 'self_attn')
        for key, module in attention.named_modules():
            if "proj" in key:
                name = 'self_attn.' + key
                rank = min(module.weight.shape[0], module.weight.shape[1])
                k = rank - rank_pruning[i][name]
                if (rank_pruning[i][name] / rank) > min_ratio:
                    if load_only is False:   l = LowRankLayer(k, module.weight.to(torch.float32), True)
                    else: l = LowRankLayerEval(k, module.weight.to(torch.float32), True)
                    setattr(attention, key, l)
                    del module
                    reduced_rank += rank_pruning[i][name]
                else:
                    k = rank
                    module.weight.requires_grad = False

                total_rank += rank
                logger.info(f"layer.{i}.{name:50} Desired/Total: {k}/{rank} ({(k/rank * 100):2f} %)")

        mlp = getattr(layer, 'mlp')
        
        for key, module in mlp.named_modules():
            if "proj" in key:
                name = 'mlp.' + key
                rank = min(module.weight.shape[0], module.weight.shape[1])
                k = rank - rank_pruning[i][name]
                if (rank_pruning[i][name] / rank) > min_ratio:
                    if load_only is False:   l = LowRankLayer(k, module.weight.to(torch.float32), True)
                    else: l = LowRankLayerEval(k, module.weight.to(torch.float32), True)
                    setattr(mlp, key, l)
                    del module
                    reduced_rank += rank_pruning[i][name]
                else:
                    k = rank
                    module.weight.requires_grad = False

                total_rank += rank
                logger.info(f"layer.{i}.{name:50} Desired/Total: {k}/{rank}  ({(k/rank * 100):2f} %)")

        import gc; gc.collect()
        torch.cuda.empty_cache()
    logger.info("*************** Pruning Model Completed ***************")
    return (reduced_rank, total_rank)

def do_low_rank(weight, desired_rank, debug=False):

    results = torch.svd(weight)
    U = results[0][:, :desired_rank]
    S = results[1][:desired_rank]
    V = results[2][:, :desired_rank]

    weight_approx = U @ torch.diag(S) @ V.T
    return weight_approx

def do_rank_reduction_merge(args, model, tokenizer, rank_pruning, min_ratio, logger = None):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers
    reduced_rank, total_rank = 0, 0
    
    logger.info("*************** Pruning Model Started ***************")
    for i in range(len(layers)):
        layer = layers[i]
        attention = getattr(layer, 'self_attn')
        for key, module in attention.named_modules():
            if "proj" in key:
                name = 'self_attn.' + key
                rank = min(module.weight.shape[0], module.weight.shape[1])
                k = rank - rank_pruning[i][name]

                if (rank_pruning[i][name] / rank) > min_ratio:
                    _W = module.weight.clone().data.to(torch.float32)
                    _W_approx = do_low_rank(_W, k)
                    module.weight.data = _W_approx.to(torch.bfloat16)
                    reduced_rank += rank_pruning[i][name]
                else:
                    k = rank
                    module.weight.requires_grad = False
                total_rank += rank
                logger.info(f"layer.{i}.{name:40} Desired/Total: {k}/{rank}")

        mlp = getattr(layer, 'mlp')
        
        for key, module in mlp.named_modules():
            if "proj" in key:
                name = 'mlp.' + key
                rank = min(module.weight.shape[0], module.weight.shape[1])
                k = rank - rank_pruning[i][name]
                if (rank_pruning[i][name] / rank) > min_ratio:
                    _W = module.weight.clone().data.to(torch.float32)
                    _W_approx = do_low_rank(_W, k)
                    module.weight.data = _W_approx.to(torch.bfloat16)
                    reduced_rank += rank_pruning[i][name]
                else:
                    k = rank
                    module.weight.requires_grad = False

                total_rank += rank
                logger.info(f"layer.{i}.{name:40} Desired/Total: {k}/{rank}")

    logger.info("*************** Pruning Model Completed ***************")
    return (reduced_rank, total_rank)
    

