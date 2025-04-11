import time
import numpy as np
from mpi4py import MPI
from rng import get_rng
from mpiwrapper import mpi
import matplotlib.pyplot as plt
from moe import SimpleMoE, MoE_EP, MoE_TP

# Example usage
def run_moe(
    moe_type="tp", 
    batch_size=8, 
    feature_dim=32, 
    hidden_dim=128, 
    output_dim=64, 
    num_experts=None,
    topk=2
):
    """
    Unified function to run different types of MoE models
    
    Args:
        moe_type: Type of MoE to run ("simple", "ep", or "tp")
        batch_size: Number of samples in the batch
        feature_dim: Dimension of input features
        hidden_dim: Hidden dimension for experts
        output_dim: Output dimension
        topk: Number of experts to route each input to
    """
    
    num_experts = mpi.get_size()
    np.random.seed(0)
    X = np.random.randn(batch_size, feature_dim)

    if moe_type != "simple":
        if mpi.get_rank() == 0:
            X = get_rng().randn(batch_size, feature_dim)
        else:
            X = None
        X = mpi.comm.bcast(X, root=0)

    model_class = {
        "simple": SimpleMoE,
        "ep": MoE_EP,
        "tp": MoE_TP
    }.get(moe_type, MoE_TP)

    moe = model_class(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        topk=topk
    )

    _ = moe(X)
    N = 3
    forward_clock = 0
    comm_clock = 0

    for _ in range(N):
        start_forward = time.time()
        outputs = moe(X)
        end_forward = time.time()

        start_comm = time.time()
        mpi.allreduce(outputs)
        end_comm = time.time()

        forward_clock += (end_forward - start_forward)
        comm_clock += (end_comm - start_comm)

    avg_forward_ms = 1000 * forward_clock / N
    avg_comm_ms = 1000 * comm_clock / N

    if mpi.get_rank() == 0:
        print("MOE Forward Pass: {}ms".format(round(avg_forward_ms, 4)))
        print("Communication Time: {}ms".format(round(avg_comm_ms, 4)))
        print()
        
    return avg_forward_ms, avg_comm_ms

def benchmark_moe():
    batch_sizes = [8, 16, 32, 64, 128]
    sol = {"SimpleMoE": {"fw": list(), "comm": list()},
        "MoE_TP": {"fw": list(), "comm": list()},
        "MoE_EP": {"fw": list(), "comm": list()}}

    for i in batch_sizes:
        temp1 = run_moe(moe_type="simple", batch_size=i)
        sol["SimpleMoE"]["fw"].append(temp1[0])
        sol["SimpleMoE"]["comm"].append(temp1[1])

        temp2 = run_moe(moe_type="tp", batch_size=i)
        sol["MoE_TP"]["fw"].append(temp2[0])
        sol["MoE_TP"]["comm"].append(temp2[1])

        temp3 = run_moe(moe_type="ep", batch_size=i)
        sol["MoE_EP"]["fw"].append(temp3[0])
        sol["MoE_EP"]["comm"].append(temp3[1])

    return batch_sizes, sol

if __name__ == "__main__":
    bsize, results = benchmark_moe()

    if mpi.get_rank() == 0:
        plt.figure(figsize=(8, 6))
        plt.plot(bsize, results["SimpleMoE"]["fw"], marker='o', label="SimpleMoE")
        plt.plot(bsize, results["MoE_TP"]["fw"], marker='x', label="MoE_TP")
        plt.plot(bsize, results["MoE_EP"]["fw"], marker='s', label="MoE_EP")
        plt.xlabel("Batch Size")
        plt.ylabel("Avg FP Time (ms)")
        plt.title("MoE FP Benchmark")
        plt.legend()
        plt.grid(True)
        plt.savefig("moe_fwpass.png")
        plt.show()
