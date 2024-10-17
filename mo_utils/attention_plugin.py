import torch
import modelopt.torch.quantization.nn as mnn
from modelopt.torch.quantization.nn import QuantModuleRegistry
import modelopt.torch.quantization as mtq

import types
import sys
import ast
import inspect
import tempfile
from modelopt.torch.quantization.nn import (
    TensorQuantizer,
)

import comfy
from comfy.ldm.modules.attention import CrossAttention, attention_pytorch
from comfy.ldm.modules.diffusionmodules.mmdit import SelfAttention as MMDITAttn
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.flux.layers import SingleStreamBlock, DoubleStreamBlock

sd_attn_cls = (CrossAttention, MMDITAttn)
flux_attn_cls = (SingleStreamBlock, DoubleStreamBlock)
attn_cls = sd_attn_cls + flux_attn_cls

def optimized_attention(
    self, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False
) -> torch.Tensor:
    scale = None
    is_causal = False

    #### Comment to export native SDPA
    if not torch.onnx.is_in_onnx_export():
        return attention_pytorch(q, k, v, heads, mask, attn_precision, skip_reshape)
    import math

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    L, S = q.size(-2), k.size(-2)
    scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
    if is_causal:
        assert mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(q.dtype)

    if mask is not None:
        if mask.dtype == torch.bool:
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
        else:
            attn_bias += mask
    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if hasattr(self, "softmax_bmm_quantizer"):
        attn_weight = self.softmax_bmm_quantizer(attn_weight)
    out = attn_weight @ v
    out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


def attention(
    self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pe: torch.Tensor
) -> torch.Tensor:
    q, k = apply_rope(q, k, pe)

    heads = q.shape[1]

    scale = None
    is_causal = False
    skip_reshape = True
    mask = None

    #### Comment to export native SDPA
    if not torch.onnx.is_in_onnx_export(): #softmax_bmm_quantizer
        return attention_pytorch(q, k, v, heads, skip_reshape=skip_reshape)
    import math

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )


    L, S = q.size(-2), k.size(-2)
    scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
    if is_causal:
        assert mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(q.dtype)

    if mask is not None:
        if mask.dtype == torch.bool:
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
        else:
            attn_bias += mask
    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if hasattr(self, "softmax_bmm_quantizer"):
        attn_weight = self.softmax_bmm_quantizer(attn_weight).to(v.dtype)
    out = attn_weight @ v
    out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


def register_attention_for_qkv_quant(attention_cls: type, sdpa_ops: tuple) -> bool:
    python_version = sys.version_info
    if not python_version >= (3, 9):
        print(
            f"Found {python_version.major}.{python_version.minor}.{python_version.micro}"
        )
        raise RuntimeError(
            "Python version >= 3.9 is required for KV Cache quantization"
        )

    source_code = inspect.getsource(attention_cls)
    model_module = inspect.getmodule(attention_cls)
    head = ast.parse(source_code)

    def is_sdpa(node):
        val = (
            isinstance(node, ast.Call)
            and hasattr(node.func, "id")
            and node.func.id in sdpa_ops
        )
        return val

    def patch(node, quantizer_names):
        node.func = ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr=node.func.id,
            ctx=ast.Load(),
        )
        for index, quantizer_name in enumerate(quantizer_names):
            if quantizer_name is None:
                continue
            arg = node.args[index]
            node.args[index] = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=quantizer_name,
                    ctx=ast.Load(),
                ),
                args=[arg],
                keywords=[],
            )

    nodes = list(ast.walk(head))
    org_class_name = nodes[1].name  # type: ignore[attr-defined]
    new_class_name = nodes[1].name = "_Quant" + nodes[1].name  # type: ignore[attr-defined]

    sdpa_nodes = []
    for node in ast.walk(head):
        if is_sdpa(node):
            sdpa_nodes.append(node)

    if len(sdpa_nodes) <= 0:
        print(f"Or expect 1 sdpa op in the {org_class_name}, found {len(sdpa_nodes)}")
        return False

    for sdpa in sdpa_nodes:
        patch(
            sdpa,
            quantizer_names=("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer"),
        )
        print("Patching 1 scaled_dot_product_attention operator with quantizers")

    head = ast.fix_missing_locations(head)
    org_class = model_module.__dict__[org_class_name]

    module_code_str = ast.unparse(head)
    # print(module_code_str)
    # exit(0)
    with tempfile.NamedTemporaryFile(
        prefix="modelopt_", suffix=".py", delete=False
    ) as temp_file:
        temp_file.write(module_code_str.encode())
        print(f"Definition of {new_class_name} saved to {temp_file.name}")

    module_code = compile(head, filename="modelopt_generated", mode="exec")
    class_code = module_code.co_consts[0]
    assert class_code.co_name == new_class_name
    method_codes = [
        const for const in class_code.co_consts if isinstance(const, types.CodeType)
    ]

    new_methods = {}
    for method_code in method_codes:
        method_name = method_code.co_name
        original_method = getattr(org_class, method_name, None)
        if not isinstance(original_method, types.FunctionType):
            continue
        # Create a new class method from bytecode
        new_methods[method_name] = types.FunctionType(
            method_code,
            globals=original_method.__globals__,
            closure=original_method.__closure__,
        )

    def setup_method(self):
        self.softmax_bmm_quantizer = TensorQuantizer()
        self.softmax_bmm_quantizer._amax = torch.ones(
            (1,), device="cuda", dtype=torch.float16
        )
        self.q_bmm_quantizer = TensorQuantizer()
        self.k_bmm_quantizer = TensorQuantizer()
        self.v_bmm_quantizer = TensorQuantizer()

    assert "_setup" not in new_methods, "Method _setup already exists"
    new_methods["_setup"] = setup_method
    new_methods[sdpa_ops[0]] = getattr(sys.modules[__name__],sdpa_ops[0])

    # Create a new subclass on the fly
    quant_class = type(new_class_name, (org_class,), new_methods)

    if org_class not in QuantModuleRegistry:
        mtq.register(original_cls=org_class, quantized_cls=quant_class)
    print(f"Successfully registered {org_class_name} for quantization")
    return True


def register_quant_modules():
    for attn in sd_attn_cls:
        register_attention_for_qkv_quant(
            attn, sdpa_ops=("optimized_attention",)
        )

    for attn in flux_attn_cls:
        register_attention_for_qkv_quant(attn, sdpa_ops=("attention",))
    for op in comfy.ops.disable_weight_init.__dict__.keys():
        if not hasattr(mnn, f"Quant{op}"):
            continue
        if getattr(comfy.ops.disable_weight_init, op) in QuantModuleRegistry:
            continue
        QuantModuleRegistry.register(
            {getattr(comfy.ops.disable_weight_init, op): f"comfy.{op}"}
        )(QuantModuleRegistry._registry[getattr(torch.nn, op)])
