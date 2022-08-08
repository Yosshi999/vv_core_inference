import re
from typing import Any, Union, Optional, List

import numpy
import torch
from torch import Tensor
from torch.onnx.symbolic_helper import _onnx_main_opset, _onnx_stable_opsets
import tltorch
import tensorly
from tensorly.decomposition import parafac
OPSET = _onnx_stable_opsets[-1]

def extract_number(f):
    s = re.findall(r"\d+", str(f))
    return int(s[-1]) if s else -1


def remove_weight_norm(m):
    try:
        torch.nn.utils.remove_weight_norm(m)
    except ValueError:
        pass


def to_tensor(array: Union[Tensor, numpy.ndarray, Any], device):
    if not isinstance(array, (Tensor, numpy.ndarray)):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        array = torch.from_numpy(array)
    return array.to(device)


class CPConv(torch.nn.Module):
    def __init__(self, cp_tensor, bias=None, stride=1, padding=0, dilation=1):
        super().__init__()
        factors = cp_tensor.factors
        self.win = factors[1].transpose(0, 1).unsqueeze(2)
        self.wker = factors[2].transpose(0, 1).unsqueeze(1)
        self.wout = factors[0].unsqueeze(2)
        self.weights = cp_tensor.weights.view(1, -1, 1)
        self.rank = cp_tensor.rank
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
    def forward(self, x):
        x = torch.nn.functional.conv1d(x, self.win)
        x = torch.nn.functional.conv1d(x, self.wker, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.rank)
        x = torch.nn.functional.conv1d(x * self.weights, self.wout, bias=self.bias)
        return x

def join_name(x: Optional[str], y: str):
    if x is None:
        return y
    else:
        return f"{x}.{y}"

def aggregate(m: torch.nn.Module, namespace=None) -> List[str]:
    target = (torch.nn.Conv1d,)
    results = []
    for name, child in m.named_children():
        path = join_name(namespace, name)
        if isinstance(child, target) and sum(child.kernel_size) >= 7:
            results.append(path)
        results += aggregate(child, path)
    return results

def decompose(m: torch.nn.Module, targets: List[str], namespace=None, decompose_weights=True):
    for name, child in list(m.named_children()):
        path = join_name(namespace, name)
        if path in targets:
            print(f"decomposing {path} ...")
            decomposition_kwargs = {
                "verbose": 1,
                "svd": "randomized_svd",
                "random_state": 0,
                "linesearch": True,
            }

            # fact_conv = tltorch.FactorizedConv.from_conv(
            #     child,
            #     rank=0.1,
            #     factorization='cp',
            #     implementation='mobilenet',
            #     decomposition_kwargs=decomposition_kwargs,
            #     dilation=child.dilation,
            #     decompose_weights=decompose_weights,
            # )
            # print("->", fact_conv.extra_repr())
            weight = child.weight.cpu().detach()
            rank = int(weight.numel() * 0.1 / sum(weight.shape))
            cp_tensor, errors = parafac(weight, rank, return_errors=True, **decomposition_kwargs)
            print("->", rank, "rank CPTensor", "err:", errors[-1].item())
            fact_conv = CPConv(cp_tensor, child.bias.cpu().detach(), child.stride, child.padding, child.dilation)

            setattr(m, name, fact_conv)
        decompose(child, targets, path, decompose_weights)
