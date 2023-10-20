"""
Metrics used for validation during training and evaluation:
Dice Score, Normalised Dice score, Lesion F1 score and nDSC R-AAC.
"""
import torch
import numpy as np
from functools import partial
from scipy import ndimage
from collections import Counter
from joblib import Parallel, delayed
from sklearn import metrics
from monai.metrics import DiceMetric


def remove_connected_components(segmentation, l_min=9):
    """
    Remove all lesions with less or equal amount of voxels than `l_min` from a
    binary segmentation mask `segmentation`.
    Args:
      segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
      l_min:  `int`, minimal amount of voxels in a lesion.
    Returns:
      Binary lesion segmentation mask (`numpy.ndarray` of shape [H, W, D])
      only with connected components that have more than `l_min` voxels.
    """
    labeled_seg, num_labels = ndimage.label(segmentation)
    label_list = np.unique(labeled_seg)
    num_elements_by_lesion = ndimage.labeled_comprehension(segmentation, labeled_seg, label_list, np.sum, float, 0)

    seg2 = np.zeros_like(segmentation)
    for i_el, n_el in enumerate(num_elements_by_lesion):
        if n_el > l_min:
            current_voxels = np.stack(np.where(labeled_seg == i_el), axis=1)
            seg2[current_voxels[:, 0],
                 current_voxels[:, 1],
                 current_voxels[:, 2]] = 1
    return seg2

def dice_metric_multiclass(ground_truth, predictions):
    """
    format [num_samples, num_classes]
    """
    dice = 0
    ground_truth = ground_truth.astype(float)
    predictions = predictions.astype(float)

    # to [num_classes, num_samples]
    ground_truth = ground_truth.transpose()
    predictions = predictions.transpose()

    intersection = np.sum(predictions * ground_truth, axis=1)

    pred_o = np.sum(predictions, axis=1)
    gt_o = np.sum(ground_truth, axis=1)
    den = pred_o + gt_o

    out = np.where(den > 0, (2.0 * intersection) / den, 1.0)
    out_mean = np.mean(out)

    return out_mean, out


def dice_metric(ground_truth, predictions):
    """
    Compute Dice coefficient for a single example.
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [W, H, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [W, H, D].
    Returns:
      Dice coefficient overlap (`float` in [0.0, 1.0])
      between `ground_truth` and `predictions`.
    """
    # Calculate intersection and union of y_true and y_predict
    intersection = np.sum(predictions * ground_truth)
    union = np.sum(predictions) + np.sum(ground_truth)

    # Calcualte dice metric
    if intersection == 0.0 and union == 0.0:
        dice = 1.0
    else:
        dice = (2. * intersection) / union

    return dice


def dice_norm_metric(ground_truth, predictions):
    """
    Compute Normalised Dice Coefficient (nDSC),
    False positive rate (FPR),
    False negative rate (FNR) for a single example.

    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """

    # Reference for normalized DSC
    r = 0.001
    # Cast to float32 type
    gt = ground_truth.astype("float32")
    seg = predictions.astype("float32")
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 1.0
    else:
        if np.sum(gt) == 0:
            k = 1.0
        else:
            k = (1 - r) * np.sum(gt) / (r * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = k * fp
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn)
        return dsc_norm


def ndsc_aac_metric(ground_truth, predictions, uncertainties, parallel_backend=None):
    """
    Compute area above Normalised Dice Coefficient (nDSC) retention curve for
    one subject. `ground_truth`, `predictions`, `uncertainties` - are flattened
    arrays of correponding 3D maps within the foreground mask only.

    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H * W * D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H * W * D].
      uncertainties:  `numpy.ndarray`, voxel-wise uncertainties,
                     with shape [H * W * D].
      parallel_backend: `joblib.Parallel`, for parallel computation
                     for different retention fractions.
    Returns:
      nDSC R-AAC (`float` in [0.0, 1.0]).
    """

    def compute_dice_norm(frac_, preds_, gts_, N_):
        pos = int(N_ * frac_)
        curr_preds = preds if pos == N_ else np.concatenate(
            (preds_[:pos], gts_[pos:]))
        return dice_norm_metric(gts_, curr_preds)

    if parallel_backend is None:
        parallel_backend = Parallel(n_jobs=1)

    ordering = uncertainties.argsort()
    gts = ground_truth[ordering].copy()
    preds = predictions[ordering].copy()
    N = len(gts)

    # # Significant class imbalance means it is important to use logspacing between values
    # # so that it is more granular for the higher retention fractions
    fracs_retained = np.log(np.arange(200 + 1)[1:])
    fracs_retained /= np.amax(fracs_retained)

    process = partial(compute_dice_norm, preds_=preds, gts_=gts, N_=N)
    dsc_norm_scores = np.asarray(
        parallel_backend(delayed(process)(frac)
                         for frac in fracs_retained)
    )

    return 1. - metrics.auc(fracs_retained, dsc_norm_scores)


def ndsc_retention_curve(ground_truth, predictions, uncertainties, best_ordering, fracs_retained, parallel_backend=None):
    """
    Compute Normalised Dice Coefficient (nDSC) retention curve.

    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H * W * D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H * W * D].
      uncertainties:  `numpy.ndarray`, voxel-wise uncertainties,
                     with shape [H * W * D].
      fracs_retained:  `numpy.ndarray`, array of increasing valies of retained
                       fractions of most certain voxels, with shape [N].
      parallel_backend: `joblib.Parallel`, for parallel computation
                     for different retention fractions.
    Returns:
      (y-axis) nDSC at each point of the retention curve (`numpy.ndarray` with shape [N]).
    """

    def compute_dice_norm(frac_, preds_, gts_, N_):
        pos = int(N_ * frac_)
        curr_preds = preds if pos == N_ else np.concatenate(
            (preds_[:pos], gts_[pos:]))
        return dice_norm_metric(gts_, curr_preds)

    if parallel_backend is None:
        parallel_backend = Parallel(n_jobs=1)
    ordering = uncertainties.argsort()
    ordering_best = best_ordering.argsort()
    gts = ground_truth[ordering].copy()
    gts_best = ground_truth[ordering_best].copy()
    preds = predictions[ordering].copy()
    preds_best = predictions[ordering_best].copy()
    N = len(gts)
    N_best = len(gts_best)
    fracs_retained_best = fracs_retained.copy()

    process = partial(compute_dice_norm, preds_=preds, gts_=gts, N_=N)
    dsc_norm_scores = np.asarray(
        parallel_backend(delayed(process)(frac)
                         for frac in fracs_retained)
    )
    process_best = partial(compute_dice_norm, preds_=preds_best, gts_=gts_best, N_=N_best)
    dsc_norm_scores_best = np.asarray(
        parallel_backend(delayed(process_best)(frac_best)
                         for frac_best in fracs_retained_best)
    )

    return dsc_norm_scores, dsc_norm_scores_best


def multi_class_dsc_retention_curve(ground_truth, predictions, uncertainties, fracs_retained, parallel_backend=None):
    """
    Compute Dice Coefficient (nDSC) retention curve.

    Args:
      ground_truth: `numpy.ndarray`, ground truth segmentation target,
                     with shape [C, H * W * D].
      predictions:  `numpy.ndarray`, segmentation predictions,
                     with shape [C, H * W * D].
      uncertainties:  `numpy.ndarray`, voxel-wise uncertainties,
                     with shape [H * W * D].
      fracs_retained:  `numpy.ndarray`, array of increasing valies of retained
                       fractions of most certain voxels, with shape [N].
      parallel_backend: `joblib.Parallel`, for parallel computation
                     for different retention fractions.
    Returns:
      (y-axis) DSC at each point of the retention curve (`numpy.ndarray` with shape [N]).
    """
    num_classes = 2

    def compute_dice_frac(frac_, preds_, gts_, N_):
        pos = int(N_ * frac_)
        curr_preds = preds_ if pos == N_ else np.concatenate(
            (preds_[:pos, :], gts_[pos:, :]))
        metric, metric_per_class = dice_metric_multiclass(predictions=curr_preds, ground_truth=gts_)

        return metric, metric_per_class

    if parallel_backend is None:
        parallel_backend = Parallel(n_jobs=1)

    ordering = uncertainties.argsort()
    gts = ground_truth[ordering].copy()
    gts_one_hot = np.zeros((gts.size, num_classes))
    gts_one_hot[np.arange(gts.size).astype(int), gts.astype(int)] = 1
    preds = predictions[ordering].copy()
    preds_one_hot = np.zeros((preds.size, num_classes))
    preds_one_hot[np.arange(preds.size).astype(int), preds.astype(int)] = 1
    N = len(gts)

    process = partial(compute_dice_frac, preds_=preds_one_hot, gts_=gts_one_hot, N_=N)
    results = parallel_backend(delayed(process)(frac) for frac in fracs_retained)
    dsc_scores, dsc_scores_per_class = zip(*results)
    dsc_scores = np.asarray(dsc_scores)
    dsc_scores_per_class = np.asarray(dsc_scores_per_class)

    return dsc_scores, dsc_scores_per_class


def intersection_over_union(mask1, mask2):
    """
    Compute IoU for 2 binary masks.

    Args:
      mask1: `numpy.ndarray`, binary mask.
      mask2:  `numpy.ndarray`, binary mask of the same shape as `mask1`.
    Returns:
      Intersection over union between `mask1` and `mask2` (`float` in [0.0, 1.0]).
    """
    return np.sum(mask1 * mask2) / np.sum(mask1 + mask2 - mask1 * mask2)


def lesion_f1_score(ground_truth, predictions, IoU_threshold=0.25, parallel_backend=None):
    """
    Compute lesion-scale F1 score.

    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
      IoU_threshold: `float` in [0.0, 1.0], IoU threshold for max IoU between
                     predicted and ground truth lesions to classify them as
                     TP, FP or FN.
      parallel_backend: `joblib.Parallel`, for parallel computation
                     for different retention fractions.
    Returns:
      Intersection over union between `mask1` and `mask2` (`float` in [0.0, 1.0]).
    """

    def get_tp_fp(label_pred, mask_multi_pred, mask_multi_gt):
        mask_label_pred = (mask_multi_pred == label_pred).astype(int)
        all_iou = [0.0]
        # iterate only intersections
        for int_label_gt in np.unique(mask_multi_gt * mask_label_pred):
            if int_label_gt != 0.0:
                mask_label_gt = (mask_multi_gt == int_label_gt).astype(int)
                all_iou.append(intersection_over_union(
                    mask_label_pred, mask_label_gt))
        max_iou = max(all_iou)
        if max_iou >= IoU_threshold:
            return 'tp'
        else:
            return 'fp'

    def get_fn(label_gt, mask_multi_pred, mask_multi_gt):
        mask_label_gt = (mask_multi_gt == label_gt).astype(int)
        all_iou = [0]
        for int_label_pred in np.unique(mask_multi_pred * mask_label_gt):
            if int_label_pred != 0.0:
                mask_label_pred = (mask_multi_pred ==
                                   int_label_pred).astype(int)
                all_iou.append(intersection_over_union(
                    mask_label_pred, mask_label_gt))
        max_iou = max(all_iou)
        if max_iou < IoU_threshold:
            return 1
        else:
            return 0

    mask_multi_pred_, n_les_pred = ndimage.label(predictions)
    mask_multi_gt_, n_les_gt = ndimage.label(ground_truth)

    if parallel_backend is None:
        parallel_backend = Parallel(n_jobs=1)

    process_fp_tp = partial(get_tp_fp, mask_multi_pred=mask_multi_pred_,
                            mask_multi_gt=mask_multi_gt_)

    tp_fp = parallel_backend(delayed(process_fp_tp)(label_pred)
                             for label_pred in np.unique(mask_multi_pred_) if label_pred != 0)
    counter = Counter(tp_fp)
    tp = float(counter['tp'])
    fp = float(counter['fp'])

    process_fn = partial(get_fn, mask_multi_pred=mask_multi_pred_,
                         mask_multi_gt=mask_multi_gt_)

    fn = parallel_backend(delayed(process_fn)(label_gt)
                          for label_gt in np.unique(mask_multi_gt_) if label_gt != 0)
    fn = float(np.sum(fn))

    f1 = 1.0 if tp + 0.5 * (fp + fn) == 0.0 else tp / (tp + 0.5 * (fp + fn))

    return f1


if __name__ == "__main__":
    import numpy as np


    def generate_data(size):
        ground_truth = np.random.randint(0, 2, size)
        predictions = np.random.randint(0, 2, size)
        uncertainties = np.random.rand(*size).flatten()
        best_ordering = np.random.rand(*size).flatten()
        fracs_retained = np.linspace(0, 1, 11)
        return ground_truth.flatten(), predictions.flatten(), uncertainties, best_ordering, fracs_retained


    ground_truth, predictions, uncertainties, best_ordering, fracs_retained = generate_data((10, 10, 10))

    dsc_norm_scores, dsc_norm_scores_best = ndsc_retention_curve(ground_truth, predictions, uncertainties,
                                                                 best_ordering, fracs_retained)
    print(fracs_retained)
    print(dsc_norm_scores[-1])
    print(dsc_norm_scores_best[-1])
