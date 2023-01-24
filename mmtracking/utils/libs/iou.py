import numpy as np


def iou(rect_batch1: np.ndarray, rect_batch2: np.ndarray) -> np.ndarray:
    """Vectorized Intersection Over Union calculation.
    :param rect_batch1: first numpy array of rectangles batch (shape: [batch size, 4])
    :param rect_batch2: second numpy array of rectangles batch (shape: [batch size, 4])
    :return: numpy array with IOU values between first and second rect batches (shape: [batch size, 1])
    """
    assert rect_batch1.shape[1] == 4 and rect_batch2.shape[1] == 4, "Wrong rect size"
    ab = np.stack([rect_batch1, rect_batch2]).astype('float32')
    intersect_area = np.maximum(ab[:, :, [2, 3]].min(axis=0) - ab[:, :, [0, 1]].max(axis=0), 0).prod(axis=1)
    union_area = ((ab[:, :, 2] - ab[:, :, 0]) * (ab[:, :, 3] - ab[:, :, 1])).sum(axis=0) - intersect_area
    return intersect_area / union_area