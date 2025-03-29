import json
import math
import argparse
from scipy.optimize import minimize_scalar

def model_training_cost_analysis_llama(model_config_path):
    #TODO you code here.
    with open(model_config_path, 'r') as f:
        config = json.load(f)
    
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    intermediate_size = config['intermediate_size']
    num_hidden_layers = config['num_hidden_layers']
    num_attention_heads = config['num_attention_heads']
    max_sequence_length = config['max_sequence_length']

    embedding_params = vocab_size * hidden_size 

    layernorm_params = 2 * 2 * hidden_size
    attention_params = 4 * hidden_size * hidden_size 
    mlp_params = 3 * intermediate_size * hidden_size 
    transformer_params = num_hidden_layers * (layernorm_params  + attention_params + mlp_params)
    final_layer_params = 2 * hidden_size + hidden_size * vocab_size
    total_params = embedding_params + transformer_params + final_layer_params

    attn_matmult_flops = 3 * 2 * max_sequence_length * hidden_size * hidden_size
    attn_softmax_flops = 2 * max_sequence_length * max_sequence_length * hidden_size + 3 * max_sequence_length * max_sequence_length * num_attention_heads
    attn_weighted_sum_flops = 2 * max_sequence_length * max_sequence_length * hidden_size
    output_matmult_flops = 2 * max_sequence_length * hidden_size * hidden_size

    mlp_flops = 6 * max_sequence_length * hidden_size * 4 * hidden_size
    flops_layer_TF = (attn_matmult_flops + attn_softmax_flops + attn_weighted_sum_flops + output_matmult_flops + mlp_flops) / 1e12

    layernorm_weights = 2 * 2 * hidden_size
    attn_weights = 4 * hidden_size * hidden_size
    mlp_weights = 2 * intermediate_size * hidden_size 

    final_mlp_output = max_sequence_length * hidden_size
    peak_memory = (layernorm_weights + attn_weights + mlp_weights + final_mlp_output) * 2
    peak_memory_GB = peak_memory / (10 ** 9)

    return total_params, flops_layer_TF, peak_memory_GB



def model_training_cost_analysis_deepseek(model_config_path):
    with open(model_config_path, 'r') as f:
        config = json.load(f)
        
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    max_position_embeddings = config['max_position_embeddings']
    num_hidden_layers = config['num_hidden_layers']
    intermediate_size = config['intermediate_size']
    num_attention_heads = config['num_attention_heads']
    max_sequence_length = max_position_embeddings
    
    n_routed_experts = config['n_routed_experts']
    n_shared_experts = config['n_shared_experts']
    num_experts_per_tok = config['num_experts_per_tok']
    moe_intermediate_size = config['moe_intermediate_size']
    moe_layer_freq = config['moe_layer_freq']  
    word_embedding_params = vocab_size * hidden_size
    positional_embedding_params = max_position_embeddings * hidden_size
    
    attention_params = 4 * hidden_size * hidden_size
    mlp_params = 2 * hidden_size * intermediate_size
    layer_norm_params = 2 * hidden_size
    expert_mlp_params = 2 * hidden_size * moe_intermediate_size
    total_expert_params = n_routed_experts * expert_mlp_params
    params_per_layer = attention_params + mlp_params + layer_norm_params
    num_moe_layers = num_hidden_layers * moe_layer_freq
    transformer_params = num_hidden_layers * params_per_layer + num_moe_layers * total_expert_params
    total_params = word_embedding_params + positional_embedding_params + transformer_params
    
    N = max_sequence_length
    D = hidden_size
    H = num_attention_heads
    
    qkv_proj_flops = 3 * N * D * D
    attn_flops = N * N * D
    out_proj_flops = N * D * D
    attention_flops = qkv_proj_flops + attn_flops + out_proj_flops
    
    moe_mlp_flops = num_experts_per_tok * N * (2 * D * moe_intermediate_size)
    flops_per_layer = attention_flops + moe_mlp_flops
    flops_layer_TF = flops_per_layer / 1e12

    kv_cache = 2 * N * D * 2  
    attn_matrix = N * N * 2  
    moe_activations = num_experts_per_tok * N * D * 2  
    peak_memory_GB = (kv_cache + attn_matrix + moe_activations) / 1e9
    
    return total_params, flops_layer_TF, peak_memory_GB
        
    

def get_optimal_N_D_from_cost(cost_budget):
    """
    cost_budget:  a monetary training budget (in dollars)
    Returns:
        N: Optimal total model parameters (in absolute numbers)
        D: Optimal number of training tokens (in absolute numbers)
        training_budget_flops: Effective total training FLOPs (in FLOPs)
        best_gpu: name of the selected GPU (one of 'A100', 'V100', 'T4')
    """
    #TODO you code here
    gpus = {'A100': {'cost_per_hour': 4.0, 'flops': 312e12}, 'V100': {'cost_per_hour': 2.5, 'flops': 125e12}, 'T4': {'cost_per_hour': 1.0, 'flops': 65e12}}

    mfu = 0.4 
    total_seconds = {i: (cost_budget / gpus[i]['cost_per_hour']) * 3600 for i in gpus}
    total_flops = {i: total_seconds[i] * gpus[i]['flops'] * mfu for i in gpus}

    best_gpu = max(total_flops, key = total_flops.get)
    training_budget_flops = total_flops[best_gpu]
    a, b, c = 406.4, 410.7, 1.69

    def loss(N):
        D = training_budget_flops / (6 * N)
        return a / N ** 0.34 + b / D ** 0.29 + c
    
    sol = minimize_scalar(loss, bounds=(1e5, 1e15))
    N = int(sol.x)
    D = int(training_budget_flops / (6 * N))
    return N, D, training_budget_flops, best_gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training cost analysis')
    parser.add_argument('--model_config', type=str, help='Path to model config file')
    parser.add_argument('--training_budget', type=float, default=None, help='Training budget')
    args = parser.parse_args()

    if args.model_config:
        if 'deepseek' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_deepseek(args.model_config)
        elif 'llama' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_llama(args.model_config)
        elif 'my' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_llama(args.model_config)
        else:
            print('Unknown LLM Type!')
            exit()

        with open(args.model_config, "r") as f:
            config = json.load(f)
            
        print("Model name: {}".format(str(config["architectures"])))
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")
        print()

    if args.training_budget:    
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(args.training_budget)
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")

    