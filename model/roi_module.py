import torch as t
from torch.autograd import Function


class RoI(Function):
    def __init__(self, outh, outw, spatial_scale):
        super().__init__()
        self.outh, self.outw, self.spatial_scale = outh, outw, spatial_scale

    def forward(self, x, rois):
        # Ensure inputs are contiguous
        x = x.contiguous()
        rois = rois.contiguous()
        self.in_size = B, C, H, W = x.size()
        self.N = N = rois.size(0)

        # Prepare output tensors on CPU
        output = t.zeros(N, C, self.outh, self.outw, device=x.device)
        self.argmax_data = t.zeros(N, C, self.outh, self.outw, dtype=t.int, device=x.device)
        self.rois = rois

        # Use PyTorch grid sample for RoI pooling operation (simplified example)
        for n in range(N):
            roi = rois[n]
            batch_idx = int(roi[0])
            x1, y1, x2, y2 = (roi[1:] * self.spatial_scale).long()

            region = x[batch_idx, :, y1:y2, x1:x2]
            pooled = t.nn.functional.adaptive_max_pool2d(region, (self.outh, self.outw))
            output[n] = pooled

        return output

    def backward(self, grad_output):
        # Ensure grad_output is contiguous
        grad_output = grad_output.contiguous()
        B, C, H, W = self.in_size
        grad_input = t.zeros(self.in_size, device=grad_output.device)

        # Backpropagate gradients manually (simplified example)
        for n in range(self.N):
            roi = self.rois[n]
            batch_idx = int(roi[0])
            x1, y1, x2, y2 = (roi[1:] * self.spatial_scale).long()

            grad_region = grad_output[n]
            grad_input[batch_idx, :, y1:y2, x1:x2] += t.nn.functional.interpolate(
                grad_region.unsqueeze(0), size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False
            ).squeeze(0)

        return grad_input, None


class RoIPooling2D(t.nn.Module):
    def __init__(self, outh, outw, spatial_scale):
        super(RoIPooling2D, self).__init__()
        self.RoI = RoI(outh, outw, spatial_scale)

    def forward(self, x, rois):
        return self.RoI(x, rois)


def test_roi_module():
    # Test the RoI pooling module with fake data
    B, N, C, H, W, PH, PW = 2, 8, 4, 32, 32, 7, 7

    device = "cpu"  # Use CPU device for macOS

    bottom_data = t.randn(B, C, H, W).to(device)
    bottom_rois = t.randn(N, 5).to(device)
    bottom_rois[:int(N / 2), 0] = 0
    bottom_rois[int(N / 2):, 0] = 1
    bottom_rois[:, 1:] = (t.rand(N, 4) * 100).float().to(device)

    spatial_scale = 1. / 16
    outh, outw = PH, PW

    # PyTorch version of RoIPooling2D
    module = RoIPooling2D(outh, outw, spatial_scale)
    x = bottom_data.requires_grad_()
    rois = bottom_rois.detach()

    output = module(x, rois)
    output.sum().backward()

    print("Test passed")
