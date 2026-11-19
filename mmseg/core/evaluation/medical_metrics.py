# Copyright (c) OpenMMLab. All rights reserved.
"""Medical segmentation metrics for binary and Synapse-style evaluation."""

from __future__ import annotations

import os.path as osp
from collections import OrderedDict, defaultdict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable

EPS = 1e-8


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def softmax(x, axis=0):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + EPS)


def resize_2d(array, target_shape, mode='linear'):
    array = np.asarray(array)
    target_shape = tuple(int(v) for v in target_shape)
    if array.shape == target_shape:
        return array

    try:
        import cv2

        interp = cv2.INTER_LINEAR if mode == 'linear' else cv2.INTER_NEAREST
        return cv2.resize(array, (target_shape[1], target_shape[0]),
                          interpolation=interp)
    except ImportError:
        pass

    from PIL import Image

    resample = Image.BILINEAR if mode == 'linear' else Image.NEAREST
    img = Image.fromarray(array.astype(np.float32))
    return np.asarray(
        img.resize((target_shape[1], target_shape[0]), resample=resample))


def binary_dice_iou_standard(pred_mask, gt_mask):
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(
            f'pred and gt must have the same shape, got {pred.shape} and '
            f'{gt.shape}')

    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    pred_sum = pred.sum(dtype=np.float64)
    gt_sum = gt.sum(dtype=np.float64)
    union = np.logical_or(pred, gt).sum(dtype=np.float64)

    if pred_sum == 0 and gt_sum == 0:
        return 1.0, 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0, 0.0

    dice = 2.0 * tp / (pred_sum + gt_sum + EPS)
    iou = tp / (union + EPS)
    return float(dice), float(iou)


def binary_fscore_standard(pred_mask, gt_mask, beta=1):
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    pred_sum = pred.sum(dtype=np.float64)
    gt_sum = gt.sum(dtype=np.float64)

    if pred_sum == 0 and gt_sum == 0:
        return 1.0, 1.0, 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0, 0.0, 0.0

    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    precision = tp / (pred_sum + EPS)
    recall = tp / (gt_sum + EPS)
    fscore = ((1 + beta**2) * precision * recall /
              (beta**2 * precision + recall + EPS))
    return float(fscore), float(precision), float(recall)


def calculate_metric_percase_transunet(pred, gt):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(
            f'pred and gt must have the same shape, got {pred.shape} and '
            f'{gt.shape}')

    pred_sum = pred.sum(dtype=np.float64)
    gt_sum = gt.sum(dtype=np.float64)

    if pred_sum > 0 and gt_sum > 0:
        tp = np.logical_and(pred, gt).sum(dtype=np.float64)
        union = np.logical_or(pred, gt).sum(dtype=np.float64)
        dice = 2.0 * tp / (pred_sum + gt_sum + EPS)
        iou = tp / (union + EPS)
        return float(dice), float(iou)

    if pred_sum > 0 and gt_sum == 0:
        return 1.0, 1.0

    return 0.0, 0.0


def calculate_fscore_percase_transunet(pred, gt, beta=1):
    dice, iou = calculate_metric_percase_transunet(pred, gt)
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    pred_sum = pred.sum(dtype=np.float64)
    gt_sum = gt.sum(dtype=np.float64)

    if pred_sum > 0 and gt_sum > 0:
        tp = np.logical_and(pred, gt).sum(dtype=np.float64)
        precision = tp / (pred_sum + EPS)
        recall = tp / (gt_sum + EPS)
        fscore = ((1 + beta**2) * precision * recall /
                  (beta**2 * precision + recall + EPS))
        return dice, iou, float(fscore), float(precision), float(recall)

    if pred_sum > 0 and gt_sum == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.0

    return 0.0, 0.0, 0.0, 0.0, 0.0


def _load_seg_map(seg_map):
    if isinstance(seg_map, str):
        return mmcv.imread(seg_map, flag='unchanged', backend='pillow')
    return np.asarray(seg_map)


def _align_pred_to_gt(pred, gt):
    pred = _load_seg_map(pred)
    gt = _load_seg_map(gt)
    if pred.shape != gt.shape:
        pred = resize_2d(pred, gt.shape[:2], mode='nearest')
    return pred, gt


def _binary_foreground_masks(pred, gt, ignore_index=255):
    pred, gt = _align_pred_to_gt(pred, gt)
    gt = gt.copy()
    gt[gt == ignore_index] = 0
    pred_fg = (pred > 0).astype(np.uint8)
    gt_fg = (gt > 0).astype(np.uint8)
    return pred_fg, gt_fg


def evaluate_binary_2d_dataset(pred_masks, gt_masks, beta=1):
    if len(pred_masks) != len(gt_masks):
        raise ValueError(
            f'pred_masks and gt_masks length mismatch: {len(pred_masks)} vs '
            f'{len(gt_masks)}')
    if len(pred_masks) == 0:
        raise ValueError('No masks to evaluate.')

    dices, ious, fscores, precisions, recalls = [], [], [], [], []
    for pred, gt in zip(pred_masks, gt_masks):
        d, j = binary_dice_iou_standard(pred, gt)
        f, p, r = binary_fscore_standard(pred, gt, beta=beta)
        dices.append(d)
        ious.append(j)
        fscores.append(f)
        precisions.append(p)
        recalls.append(r)

    return {
        'mDice': float(np.mean(dices)),
        'mIoU': float(np.mean(ious)),
        'mFscore': float(np.mean(fscores)),
        'Precision': float(np.mean(precisions)),
        'Recall': float(np.mean(recalls)),
        'per_image_dice': np.asarray(dices, dtype=np.float64),
        'per_image_iou': np.asarray(ious, dtype=np.float64),
        'n_images': len(dices),
    }


def synapse_case_key(filename):
    """Return case id from 2D slice names like ``case0003_slice118.png``."""
    basename = osp.splitext(osp.basename(filename))[0]
    parts = basename.split('_')
    if parts and parts[0].startswith('case'):
        return parts[0]
    return parts[0] if parts else basename


def synapse_slice_index(filename):
    """Return slice index from 2D slice names like ``case0003_slice118.png``."""
    basename = osp.splitext(osp.basename(filename))[0]
    parts = basename.split('_')
    for token in parts[1:]:
        if token.startswith('slice'):
            try:
                return int(token[len('slice'):])
            except ValueError:
                continue
        try:
            return int(token)
        except ValueError:
            continue
    return 0


def group_synapse_volumes(results, gt_seg_maps, img_infos):
    """Group flat 2D slice predictions/GTs into 3D volumes by case id."""
    if len(results) != len(gt_seg_maps) or len(results) != len(img_infos):
        raise ValueError('results, gt_seg_maps and img_infos must align.')

    case_items = defaultdict(list)
    for idx, img_info in enumerate(img_infos):
        filename = img_info['filename']
        case_items[synapse_case_key(filename)].append(
            (synapse_slice_index(filename), idx))

    pred_volumes = []
    gt_volumes = []
    for case_key in sorted(case_items.keys()):
        indices = [idx for _, idx in sorted(case_items[case_key],
                                            key=lambda item: item[0])]
        pred_slices = [_load_seg_map(results[i]) for i in indices]
        gt_slices = [_load_seg_map(gt_seg_maps[i]) for i in indices]
        pred_volumes.append(np.stack(pred_slices, axis=0))
        gt_volumes.append(np.stack(gt_slices, axis=0))
    return pred_volumes, gt_volumes


def evaluate_synapse_volumes(pred_volumes,
                             gt_volumes,
                             num_classes,
                             class_ids=None,
                             beta=1):
    if len(pred_volumes) != len(gt_volumes):
        raise ValueError(
            f'pred_volumes and gt_volumes length mismatch: '
            f'{len(pred_volumes)} vs {len(gt_volumes)}')
    if len(pred_volumes) == 0:
        raise ValueError('No volumes to evaluate.')

    if class_ids is None:
        class_ids = list(range(1, int(num_classes)))
    else:
        class_ids = [int(c) for c in class_ids if int(c) != 0]
    if not class_ids:
        raise ValueError('No foreground class_ids to evaluate.')

    case_class_scores = []
    case_class_fscores = []
    for pred_vol, gt_vol in zip(pred_volumes, gt_volumes):
        pred_vol = np.asarray(pred_vol)
        gt_vol = np.asarray(gt_vol)
        if pred_vol.shape != gt_vol.shape:
            raise ValueError(
                f'pred and gt volume must have the same shape, got '
                f'{pred_vol.shape} and {gt_vol.shape}')

        one_case = []
        one_case_f = []
        for class_id in class_ids:
            d, j = calculate_metric_percase_transunet(
                pred_vol == class_id, gt_vol == class_id)
            _, _, f, p, r = calculate_fscore_percase_transunet(
                pred_vol == class_id, gt_vol == class_id, beta=beta)
            one_case.append([d, j])
            one_case_f.append([f, p, r])
        case_class_scores.append(one_case)
        case_class_fscores.append(one_case_f)

    scores = np.asarray(case_class_scores, dtype=np.float64)
    fscores = np.asarray(case_class_fscores, dtype=np.float64)
    per_class_dice = scores[:, :, 0].mean(axis=0)
    per_class_iou = scores[:, :, 1].mean(axis=0)
    per_class_fscore = fscores[:, :, 0].mean(axis=0)
    per_class_precision = fscores[:, :, 1].mean(axis=0)
    per_class_recall = fscores[:, :, 2].mean(axis=0)

    return {
        'mDice': float(per_class_dice.mean()),
        'mIoU': float(per_class_iou.mean()),
        'mFscore': float(per_class_fscore.mean()),
        'Precision': float(per_class_precision.mean()),
        'Recall': float(per_class_recall.mean()),
        'class_ids': np.asarray(class_ids, dtype=np.int64),
        'per_class_dice': per_class_dice,
        'per_class_iou': per_class_iou,
        'per_class_fscore': per_class_fscore,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'n_cases': len(pred_volumes),
    }


def _format_eval_results(raw_metrics,
                         metrics,
                         class_names,
                         logger=None,
                         foreground_only=True):
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError(f'metric {metrics} is not supported')

    eval_results = {}
    ret_metrics = OrderedDict()

    if 'mIoU' in metrics:
        ret_metrics['IoU'] = raw_metrics.get('per_class_iou')
        if ret_metrics['IoU'] is None:
            ret_metrics['IoU'] = np.asarray([raw_metrics['mIoU']])
    if 'mDice' in metrics:
        ret_metrics['Dice'] = raw_metrics.get('per_class_dice')
        if ret_metrics['Dice'] is None:
            ret_metrics['Dice'] = np.asarray([raw_metrics['mDice']])
    if 'mFscore' in metrics:
        ret_metrics['Fscore'] = raw_metrics.get('per_class_fscore')
        if ret_metrics['Fscore'] is None:
            ret_metrics['Fscore'] = np.asarray([raw_metrics['mFscore']])
        ret_metrics['Precision'] = raw_metrics.get('per_class_precision')
        if ret_metrics['Precision'] is None:
            ret_metrics['Precision'] = np.asarray([raw_metrics['Precision']])
        ret_metrics['Recall'] = raw_metrics.get('per_class_recall')
        if ret_metrics['Recall'] is None:
            ret_metrics['Recall'] = np.asarray([raw_metrics['Recall']])

    if foreground_only and len(class_names) > 1:
        display_class_names = class_names[1:]
        for key in list(ret_metrics.keys()):
            values = ret_metrics[key]
            if values is not None and len(values) == len(class_names):
                ret_metrics[key] = values[1:]
    else:
        display_class_names = class_names

    ret_metrics_summary = OrderedDict()
    if 'mIoU' in metrics:
        ret_metrics_summary['IoU'] = round(raw_metrics['mIoU'] * 100, 2)
    if 'mDice' in metrics:
        ret_metrics_summary['Dice'] = round(raw_metrics['mDice'] * 100, 2)
    if 'mFscore' in metrics:
        ret_metrics_summary['Fscore'] = round(raw_metrics['mFscore'] * 100, 2)

    ret_metrics_class = OrderedDict({
        key: np.round(value * 100, 2)
        for key, value in ret_metrics.items()
        if value is not None
    })
    ret_metrics_class.update({'Class': display_class_names})
    ret_metrics_class.move_to_end('Class', last=False)

    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)

    summary_table_data = PrettyTable()
    for key, val in ret_metrics_summary.items():
        summary_table_data.add_column('m' + key, [val])

    print_log('per class results:', logger)
    print_log('\n' + class_table_data.get_string(), logger=logger)
    print_log('Summary:', logger)
    print_log('\n' + summary_table_data.get_string(), logger=logger)

    if 'mIoU' in metrics:
        eval_results['mIoU'] = raw_metrics['mIoU']
    if 'mDice' in metrics:
        eval_results['mDice'] = raw_metrics['mDice']
    if 'mFscore' in metrics:
        eval_results['mFscore'] = raw_metrics['mFscore']
        eval_results['Precision'] = raw_metrics['Precision']
        eval_results['Recall'] = raw_metrics['Recall']

    ret_metrics_class.pop('Class', None)
    for key, value in ret_metrics_class.items():
        eval_results.update({
            f'{key}.{name}': value[idx] / 100.0
            for idx, name in enumerate(display_class_names)
        })

    return eval_results


def evaluate_binary_dataset(results,
                            gt_seg_maps,
                            metrics,
                            class_names,
                            logger=None,
                            ignore_index=255,
                            beta=1):
    if not (mmcv.is_list_of(results, np.ndarray)
            or mmcv.is_list_of(results, str)):
        raise NotImplementedError(
            'Binary medical datasets require hard prediction maps. '
            'Please set evaluation.pre_eval=False.')

    if gt_seg_maps is None:
        raise ValueError('gt_seg_maps must be provided for binary evaluation.')
    gt_seg_maps = list(gt_seg_maps)

    pred_masks = []
    gt_masks = []
    for pred, gt in zip(results, gt_seg_maps):
        pred_fg, gt_fg = _binary_foreground_masks(pred, gt, ignore_index)
        pred_masks.append(pred_fg)
        gt_masks.append(gt_fg)

    raw = evaluate_binary_2d_dataset(pred_masks, gt_masks, beta=beta)
    raw['per_class_dice'] = np.asarray([raw['mDice']])
    raw['per_class_iou'] = np.asarray([raw['mIoU']])
    raw['per_class_fscore'] = np.asarray([raw['mFscore']])
    raw['per_class_precision'] = np.asarray([raw['Precision']])
    raw['per_class_recall'] = np.asarray([raw['Recall']])
    return _format_eval_results(raw, metrics, class_names, logger,
                                foreground_only=True)


def evaluate_synapse_dataset(results,
                             gt_seg_maps,
                             img_infos,
                             metrics,
                             class_names,
                             logger=None,
                             num_classes=None,
                             beta=1):
    if not (mmcv.is_list_of(results, np.ndarray)
            or mmcv.is_list_of(results, str)):
        raise NotImplementedError(
            'SynapseDataset requires hard prediction maps. '
            'Please set evaluation.pre_eval=False.')

    if gt_seg_maps is None:
        raise ValueError('gt_seg_maps must be provided for Synapse evaluation.')
    gt_seg_maps = list(gt_seg_maps)

    if num_classes is None:
        num_classes = len(class_names)

    pred_volumes, gt_volumes = group_synapse_volumes(
        results, gt_seg_maps, img_infos)
    raw = evaluate_synapse_volumes(
        pred_volumes,
        gt_volumes,
        num_classes=num_classes,
        beta=beta)
    return _format_eval_results(raw, metrics, class_names, logger,
                                foreground_only=True)
