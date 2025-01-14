from __future__ import division
import numpy as np
import torch as t
try:
    from ._nms_gpu_post import _nms_gpu_post
except:
    import warnings
    warnings.warn('''
    the python code for non_maximum_suppression is about 2x slow
    It is strongly recommended to build cython code: 
    `cd model/utils/nms/; python3 build.py build_ext --inplace''')
    from ._nms_gpu_post_py import _nms_gpu_post


@t.jit.script
def _load_kernel(kernel_name, code, options=()):
    if not t.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    kernel_code = t.cuda.CUDAStream()
    def kernel_fn(*args):
        pass

    return kernel_fn


def non_maximum_suppression(bbox, thresh, score=None,
                            limit=None):
    """Suppress bounding boxes according to their IoUs.

    This method checks each bounding box sequentially and selects the bounding
    box if the Intersection over Unions (IoUs) between the bounding box and the
    previously selected bounding boxes is less than :obj:`thresh`. This method
    is mainly used as postprocessing of object detection.
    The bounding boxes are selected from ones with higher scores.
    If :obj:`score` is not provided as an argument, the bounding box
    is ordered by its index in ascending order.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    :obj:`score` is a float array of shape :math:`(R,)`. Each score indicates
    confidence of prediction.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    an input. Please note that both :obj:`bbox` and :obj:`score` need to be
    the same type.
    The type of the output is the same as the input.

    Args:
        bbox (array): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        thresh (float): Threshold of IoUs.
        score (array): An array of confidences whose shape is :math:`(R,)`.
        limit (int): The upper bound of the number of the output bounding
            boxes. If it is not specified, this method selects as many
            bounding boxes as possible.

    Returns:
        array:
        An array with indices of bounding boxes that are selected. \
        They are sorted by the scores of bounding boxes in descending \
        order. \
        The shape of this array is :math:`(K,)` and its dtype is\
        :obj:`numpy.int32`. Note that :math:`K \\leq R`.

    """

    return _non_maximum_suppression_gpu(bbox, thresh, score, limit)


def _non_maximum_suppression_gpu(bbox, thresh, score=None, limit=None):
    if len(bbox) == 0:
        return t.zeros((0,), dtype=t.int32).cpu().numpy()

    n_bbox = bbox.shape[0]

    if score is not None:
        order = t.argsort(score, descending=True).to(t.int32)
    else:
        order = t.arange(n_bbox, dtype=t.int32, device='cuda')

    sorted_bbox = bbox[order, :]
    selec, n_selec = _call_nms_kernel(
        sorted_bbox, thresh)
    selec = selec[:n_selec]
    selec = order[selec]
    if limit is not None:
        selec = selec[:limit]
    return selec.cpu().numpy()


_nms_gpu_code = '''
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__
inline float devIoU(float const *const bbox_a, float const *const bbox_b) {
  float top = max(bbox_a[0], bbox_b[0]);
  float bottom = min(bbox_a[2], bbox_b[2]);
  float left = max(bbox_a[1], bbox_b[1]);
  float right = min(bbox_a[3], bbox_b[3]);
  float height = max(bottom - top, 0.f);
  float width = max(right - left, 0.f);
  float area_i = height * width;
  float area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]);
  float area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]);
  return area_i / (area_a + area_b - area_i);
}

extern "C"
__global__
void nms_kernel(const int n_bbox, const float thresh,
                const float *dev_bbox,
                unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int row_size =
        min(n_bbox - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_bbox - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_bbox[threadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    block_bbox[threadIdx.x * 4 + 0] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_bbox[threadIdx.x * 4 + 1] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_bbox[threadIdx.x * 4 + 2] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_bbox[threadIdx.x * 4 + 3] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_bbox + cur_box_idx * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_bbox + i * 4) >= thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_bbox, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}
'''


def _call_nms_kernel(bbox, thresh):
    if not isinstance(bbox, t.Tensor):
        bbox = t.tensor(bbox, dtype=t.float32, device='cuda')
    if not bbox.is_cuda:
        bbox = bbox.cuda()

    n_bbox = bbox.shape[0]
    threads_per_block = 64
    col_blocks = int(np.ceil(n_bbox / threads_per_block))
    mask_dev = t.zeros((n_bbox * col_blocks,), dtype=t.uint64, device='cuda')
    
    def nms_kernel(n_bbox, thresh, bbox, mask_dev):
        for i in range(n_bbox):
            if mask_dev[i] != 0:
                continue

            box_i = bbox[i]

            for j in range(i + 1, n_bbox):
                if mask_dev[j] != 0:
                    continue 

                box_j = bbox[j]
                iou = compute_iou(box_i, box_j)
                
                if iou > thresh:
                    mask_dev[j] = 1
    
    nms_kernel(n_bbox, thresh, bbox, mask_dev)
    
    mask_host = mask_dev.cpu().numpy()

    selection, n_selec = _nms_gpu_post(mask_host, n_bbox, threads_per_block, col_blocks)
    return selection, n_selec

def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1 (t.Tensor): The first bounding box, shape (4,).
        box2 (t.Tensor): The second bounding box, shape (4,).
        
    Returns:
        float: The IoU between the two boxes.
    """
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])  # max(x1_1, x1_2)
    y1 = max(box1[1], box2[1])  # max(y1_1, y1_2)
    x2 = min(box1[2], box2[2])  # min(x2_1, x2_2)
    y2 = min(box1[3], box2[3])  # min(y2_1, y2_2)

    # Calculate the area of the intersection
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the area of the union
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou