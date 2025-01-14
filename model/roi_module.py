import torch
from torch.autograd import Function
import torch.nn as nn
from string import Template

CUDA_NUM_THREADS = 1024

def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
    return (N + K - 1) // K

def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = torch.utils.cpp_extension.load_inline(
        name=kernel_name,
        cpp_sources="",
        cuda_sources=code,
        verbose=True,
    )
    return kernel_code

class RoI(Function):
    def __init__(self, outh, outw, spatial_scale):
        self.forward_fn = load_kernel('roi_forward', kernel_forward)
        self.backward_fn = load_kernel('roi_backward', kernel_backward)
        self.outh, self.outw = outh, outw
        self.spatial_scale = spatial_scale

    def forward(self, x, rois):
        x = x.contiguous()
        rois = rois.contiguous()
        self.in_size = x.size()
        self.N = rois.size(0)

        output = torch.zeros((self.N, x.size(1), self.outh, self.outw), device='cuda')
        self.argmax_data = torch.zeros_like(output, dtype=torch.int32, device='cuda')
        self.rois = rois

        args = [
            x.data_ptr(), rois.data_ptr(),
            output.data_ptr(), self.argmax_data.data_ptr(),
            self.spatial_scale, x.size(1), x.size(2), x.size(3),
            self.outh, self.outw, output.numel()
        ]
        stream = torch.cuda.current_stream()
        self.forward_fn(
            args=args,
            block=(CUDA_NUM_THREADS, 1, 1),
            grid=(GET_BLOCKS(output.numel()), 1, 1),
            stream=stream
        )
        return output

    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        B, C, H, W = self.in_size
        grad_input = torch.zeros((B, C, H, W), device='cuda')

        args = [
            grad_output.data_ptr(), self.argmax_data.data_ptr(),
            self.rois.data_ptr(), grad_input.data_ptr(),
            self.N, self.spatial_scale, C, H, W,
            self.outh, self.outw, grad_input.numel()
        ]
        stream = torch.cuda.current_stream()
        self.backward_fn(
            args=args,
            block=(CUDA_NUM_THREADS, 1, 1),
            grid=(GET_BLOCKS(grad_input.numel()), 1, 1),
            stream=stream
        )
        return grad_input, None

class RoIPooling2D(nn.Module):
    def __init__(self, outh, outw, spatial_scale):
        super().__init__()
        self.RoI = RoI(outh, outw, spatial_scale)

    def forward(self, x, rois):
        return self.RoI.apply(x, rois)
