from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        #raise NotImplementedError

        A, B = input_values
        eps = node.attrs['eps']
        matmul_res = torch.matmul(A, B)

        avg = torch.mean(matmul_res, dim = -1, keepdim = True)
        var = torch.var(matmul_res, dim = -1, keepdim = True, unbiased = False)

        matmul_res = (matmul_res - avg)/torch.sqrt(var + eps)
        return matmul_res

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        """TODO: your code here"""
        #raise NotImplementedError

        A, B = node.inputs
        eps = node.attrs['eps']
        dim = tuple(range(-len(node.attrs['normalized_shape']), 0))

        mmres = matmul(A, B)
        avg = mean(mmres, dim = dim, keepdim = True)
        std = mean(power(sub(mmres, avg), 2), dim = dim, keepdim = True)
        length = add_by_const(zeros_like(mmres), float(torch.prod(torch.tensor(node.attrs["normalized_shape"])))) 

        x = div(sub(mmres, avg), sqrt(add_by_const(std, eps)))

        value1 = div(ones_like(mmres), sqrt(add_by_const(std, eps)))
        value2 = sub(output_grad, mul(div(ones_like(mmres), length), sum_op(output_grad, dim = -1, keepdim = True)))
        value3 = mul(div(ones_like(mmres), length), mul(x, sum_op(mul(output_grad,x), dim=-1, keepdim=True)))

        return [matmul(mul(value1, sub(value2, value3)), 
                       transpose(B, dim0 = -2, dim1 = -1)), 
                matmul(transpose(A, dim0 = -2, dim1 = -1), 
                       mul(value1, sub(value2, value3)))]


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        #raise NotImplementedError

        A, B = input_values
        matmul_result = torch.matmul(A, B)
        dim = node.attrs["dim"]
    
        max_values = torch.max(matmul_result, dim=dim, keepdim=True).values
        exp_values = torch.exp(matmul_result - max_values)
        softmax_result = exp_values / torch.sum(exp_values, dim=dim, keepdim=True)

        return softmax_result

    
    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        """TODO: your code here"""
        #raise NotImplementedError

        A, B = node.inputs
        dim = node.attrs['dim']
        mmres = matmul(A, B)

        sm = softmax(mmres)
        out = mul(sm, sub(output_grad, sum_op(mul(output_grad, sm), dim = dim, keepdim = True)))

        return [matmul(out, transpose(B, dim0 = -2, dim1 = -1)), matmul(transpose(A, dim0 = -2, dim1 = -1), out)]


# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()