# cse234-w25-PA
Collection of mini-projects in LLM Systems.

# PA1<br />
Autodiff libary, implementations of operations, namely- DivOp, DivByConstOp, TransposeOp, ReLUOp, SqrtOp, PowerOp, MeanOp, MatMulOp, SoftmaxOp, LayerNormOp, and includes evaluation framework classes. Fused operators are also implemented- MatMulLayerNormOp, and MatMulSoftmaxOp. Transformer architecture implemented, with single layer ViT trained on the MNIST dataset! 

# PA2<br />
Matmul kernel optimization- with five distinct steps: Tile Assignment, Shared Memory Tiling + Cooperative Fetching, Register Tiling (Accumulator), Operator Fusion, and Write Cache/Epilogue. Implemented communication protocols for data parallel and tensor model parallel training from the ground up, using Message Passing Interface (MPI) and NumPy. **Includes a 1.25x speedup of my custom matmul kernel over generic PyTorch matmul, and a significant speedup of my custom AllReduce and AlltoAll as compared to MPI.**

# PA3<br /> 
Implemented a Mixture of Expert (MoE) model with two different variants of communication patterns for the MoE layer: tensor parallel (TP) and expert parallel (EP). Scaling law and training cost analysis code is also included, including calculation of Llama 7B and Deepseek V3. Finally, I've implemented a speculative decoder, and optimized it, and written a 500 word writeup about one topic in the future of HW and AI. **Speculative decoder includes a token acceptance rate of at least 87.5%, performance speedup ranging between 1.17x-2.08x, and latency reduction between 15-50%.**
