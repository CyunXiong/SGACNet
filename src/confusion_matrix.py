# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import numbers
import torch
import tensorflow as tf
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric, MetricsLambda


class ConfusionMatrixTensorflow:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        if self.n_classes < 255:
            self.dtype = tf.uint8
        else:
            self.dtype = tf.uint16

        self.overall_confusion_matrix = np.zeros((self.n_classes,
                                                  self.n_classes))
        self.cm_func = self.build_confusion_matrix_graph()

    def reset_conf_matrix(self):
        self.overall_confusion_matrix = np.zeros((self.n_classes,
                                                  self.n_classes))

    def update_conf_matrix(self, ground_truth, prediction):
        sess = tf.compat.v1.Session()

        current_confusion_matrix = \
            sess.run(self.cm_func, feed_dict={self.ph_cm_y_true: ground_truth,
                                              self.ph_cm_y_pred: prediction})

        # update or create confusion matrix
        if self.overall_confusion_matrix is not None:
            self.overall_confusion_matrix += current_confusion_matrix
        else:
            self.overall_confusion_matrix = current_confusion_matrix

    def build_confusion_matrix_graph(self):

        tf.compat.v1.disable_eager_execution() #?禁用急切执行。
        self.ph_cm_y_true = tf.compat.v1.placeholder(dtype=self.dtype,   #?为将始终输入的张量插入占位符
                                                     shape=None)
        self.ph_cm_y_pred = tf.compat.v1.placeholder(dtype=self.dtype,
                                                     shape=None)

        return tf.math.confusion_matrix(labels=self.ph_cm_y_true,
                                        predictions=self.ph_cm_y_pred,
                                        num_classes=self.n_classes)

    def compute_miou(self):
        cm = self.overall_confusion_matrix.copy()

        # sum over the ground truth, the predictions and create a mask for
        # empty classes
        gt_set = cm.sum(axis=1)#?将一个矩阵的每一行向量相加
        # print(f'gt_set:{gt_set}')
        all_set=cm.sum(axis=None) #?希望求出混淆矩阵所有元素相加
        print(f'all_set:{all_set}')
        pred_set = cm.sum(axis=0)#?将一个矩阵的每一列向量相加
        # print(f'pred_set:{pred_set}')
        # mask_gt = gt_set > 0
        #
        # # calculate intersection over union and the mean of it
        # intersection = np.diag(cm)[mask_gt]
        # union = gt_set[mask_gt] + pred_set[mask_gt] - intersection

        # calculate intersection over union and the mean of it
        intersection = np.diag(cm) #*?diag 提取对角线或构造对角线数组。
        intersect=intersection.sum(axis=None)
        print(f'intersect:{intersect}')
        union = gt_set + pred_set - intersection

        # union might be 0. Then convert nan to 0.
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = intersection / union.astype(np.float32) 
            pacc=intersect / all_set.astype(np.float32) #?pixel_acc #*pacc=(intersect /all_set.astype(np.float32))
            pacc= np.nan_to_num(pacc)
            iou = np.nan_to_num(iou) #?使用0代替数组x中的nan元素，使用有限的数字代替inf元素(默认行为)或者用户使用nan、posinf和neginf关键字来定义数字
            macc= intersection / gt_set.astype(np.float32)#?acc
            macc=np.nan_to_num(macc)
        miou = np.mean(iou)
        macc=np.mean(macc)
        

        return miou, pacc, macc,iou  #*增加了pa,acc


class ConfusionMatrixPytorch(Metric):
    def __init__(self,
                 num_classes,
                 average=None,
                 output_transform=lambda x: x):
        if average is not None and average not in ("samples", "recall",
                                                   "precision"):
            raise ValueError("Argument average can None or one of "
                             "['samples', 'recall', 'precision']")

        self.num_classes = num_classes
        if self.num_classes < np.sqrt(2**8):
            self.dtype = torch.uint8
        elif self.num_classes < np.sqrt(2**16 / 2):
            self.dtype = torch.int16
        elif self.num_classes < np.sqrt(2**32 / 2):
            self.dtype = torch.int32
        else:
            self.dtype = torch.int64
        self._num_examples = 0
        self.average = average
        self.confusion_matrix = None
        super(ConfusionMatrixPytorch, self).__init__(
            output_transform=output_transform
        )

    def reset(self):
        self.confusion_matrix = torch.zeros(self.num_classes,
                                            self.num_classes,
                                            dtype=torch.int64,
                                            device='cpu')
        self._num_examples = 0

    def update(self, y, y_pred, num_examples=1):
        assert len(y) == len(y_pred), ('label and prediction need to have the'
                                       ' same size')
        self._num_examples += num_examples

        y = y.type(self.dtype)
        y_pred = y_pred.type(self.dtype)

        indices = self.num_classes * y + y_pred
        m = torch.bincount(indices,
                           minlength=self.num_classes ** 2)
        m = m.reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += m.to(self.confusion_matrix)

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Confusion matrix must have at least one '
                                     'example before it can be computed.')
        if self.average:
            self.confusion_matrix = self.confusion_matrix.float()
            if self.average == "samples":
                return self.confusion_matrix / self._num_examples
            elif self.average == "recall":
                return self.confusion_matrix / (self.confusion_matrix.sum(dim=1) + 1e-15)
            elif self.average == "precision":
                return self.confusion_matrix / (self.confusion_matrix.sum(dim=0) + 1e-15)
        return self.confusion_matrix


def iou_pytorch(cm, ignore_index=None):
    if not isinstance(cm, ConfusionMatrixPytorch):
        raise TypeError("Argument cm should be instance of ConfusionMatrix, "
                        "but given {}".format(type(cm)))

    if ignore_index is not None:
        if (not (isinstance(ignore_index, numbers.Integral)
                 and 0 <= ignore_index < cm.num_classes)):
            raise ValueError("ignore_index should be non-negative integer, "
                             "but given {}".format(ignore_index))

    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)
    iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
    if ignore_index is not None:

        def ignore_index_fn(iou_vector):
            if ignore_index >= len(iou_vector):
                raise ValueError("ignore_index {} is larger than the length "
                                 "of IoU vector {}"
                                 .format(ignore_index, len(iou_vector)))
            indices = list(range(len(iou_vector)))
            indices.remove(ignore_index)
            return iou_vector[indices]

        return MetricsLambda(ignore_index_fn, iou)
    else:
        return iou


def miou_pytorch(cm, ignore_index=None):
    return iou_pytorch(cm=cm, ignore_index=ignore_index).mean()


if __name__ == '__main__':
    # test if pytorch confusion matrix and tensorflow confusion matrix
    # compute the same
    label = np.array([0, 0, 1, 2, 3])
    prediction = np.array([1, 1, 0, 2, 3])

    cm_tf = ConfusionMatrixTensorflow(4)
    cm_pytorch = ConfusionMatrixPytorch(4)
    miou = miou_pytorch(cm_pytorch)

    cm_tf.update_conf_matrix(label, prediction)
    cm_pytorch.update(torch.from_numpy(label), torch.from_numpy(prediction))

    print(cm_tf.overall_confusion_matrix)
    print(cm_pytorch.confusion_matrix.numpy())

    print('mIoU tensorflow:', cm_tf.compute_miou())
    print('mIoU pytorch:', miou.compute().data.numpy())
