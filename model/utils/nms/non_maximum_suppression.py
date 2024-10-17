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


def non_maximum_suppression(bbox, thresh, score=None, limit=None):
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

    This function accepts :obj:`numpy.ndarray` as an input. Both :obj:`bbox`
    and :obj:`score` need to be the same type.
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

    return _non_maximum_suppression_cpu(bbox, thresh, score, limit)


def _non_maximum_suppression_cpu(bbox, thresh, score=None, limit=None):
    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    n_bbox = bbox.shape[0]

    if score is not None:
        order = score.argsort()[::-1].astype(np.int32)
    else:
        order = np.arange(n_bbox, dtype=np.int32)

    sorted_bbox = bbox[order, :]
    selec, n_selec = _call_nms_cpu(sorted_bbox, thresh)
    selec = selec[:n_selec]
    selec = order[selec]
    if limit is not None:
        selec = selec[:limit]
    return selec


def _call_nms_cpu(bbox, thresh):
    """A simple CPU-based non-maximum suppression function."""
    n_bbox = bbox.shape[0]
    selected = []

    for i in range(n_bbox):
        box_a = bbox[i]
        keep = True

        for j in selected:
            box_b = bbox[j]

            # Calculate IoU between box_a and box_b
            iou = _compute_iou(box_a, box_b)

            if iou >= thresh:
                keep = False
                break

        if keep:
            selected.append(i)

    return np.array(selected, dtype=np.int32), len(selected)


def _compute_iou(box_a, box_b):
    """Compute the IoU (Intersection over Union) of two bounding boxes."""
    y_min_a, x_min_a, y_max_a, x_max_a = box_a
    y_min_b, x_min_b, y_max_b, x_max_b = box_b

    inter_y_min = max(y_min_a, y_min_b)
    inter_x_min = max(x_min_a, x_min_b)
    inter_y_max = min(y_max_a, y_max_b)
    inter_x_max = min(x_max_a, x_max_b)

    inter_area = max(0, inter_y_max - inter_y_min) * max(0, inter_x_max - inter_x_min)

    area_a = (y_max_a - y_min_a) * (x_max_a - x_min_a)
    area_b = (y_max_b - y_min_b) * (x_max_b - x_min_b)

    union_area = area_a + area_b - inter_area

    return inter_area / union_area
