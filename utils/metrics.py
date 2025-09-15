import threading
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np


class ImgMetric(object):
    """Evaluation Metrics for Image Classification"""

    def __init__(self, num_class=2, score_thresh=0.5):
        super(ImgMetric, self).__init__()
        self.num_class = num_class
        self.score_thresh = score_thresh
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.tol_label = np.array([], dtype=np.float64)
        self.tol_pred = np.array([], dtype=np.float64)

    def update(self, preds, labels):
        for pred, label in zip(preds, labels):
            self.tol_pred = np.append(self.tol_pred, pred)
            self.tol_label = np.append(self.tol_label, label)

    # def update(self, preds, labels):
    #     def evaluate_worker(self, pred, label):
    #         with self.lock:
    #             self.tol_pred = np.append(self.tol_pred, pred)
    #             self.tol_label = np.append(self.tol_label, label)
    #         return

    #     if isinstance(preds, np.ndarray):
    #         evaluate_worker(self, preds, labels)
    #     elif isinstance(preds, (list, tuple)):
    #         threads = [threading.Thread(target=evaluate_worker,
    #                                     args=(self, pred, label),
    #                                     )
    #                    for (pred, label) in zip(preds, labels)]
    #         for thread in threads:
    #             thread.start()
    #         for thread in threads:
    #             thread.join()
    #     else:
    #         raise NotImplemented

    def total_score(self):
        tol_auc = roc_auc_score(self.tol_label, self.tol_pred)

        tol_acc = accuracy_score(self.tol_label, np.where(self.tol_pred > self.score_thresh, 1, 0))
        
        fpr, tpr, _ = roc_curve(self.tol_label, self.tol_pred, pos_label=1)
        tol_eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        tn, fp, fn, tp = confusion_matrix(self.tol_label, np.where(self.tol_pred > self.score_thresh, 1, 0)).ravel()
        tol_f1 = (2.0 * tp / (2.0 * tp + fn + fp) + 2.0 * tn / (2.0 * tn + fn + fp)) / 2

        return tol_acc, tol_auc, tol_f1, tol_eer


class PixMetric(object):
    """Computes pix-level Acc mIoU, F1, and MCC metric scores
    refer to https://github.com/Tramac/awesome-semantic-segmentation-pytorch 
    and https://github.com/Tianfang-Zhang/AGPCNet
    and https://github.com/SegmentationBLWX/sssegmentation
    """
    def __init__(self, num_class=2):
        super(PixMetric, self).__init__()
        self.numClass = num_class

    def BatchConfusionMatrix(self, imgPredict, imgLabel): 
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

 
    def total_score(self, total_matrix):
        tp, fp, fn, tn = total_matrix.ravel()
    
        # tol_acc = (tp+tn) / (tp+fp+fn+tn)

        # tol_f1 = (2.0 * tp / (2.0 * tp + fn + fp) + 2.0 * tn / (2.0 * tn + fn + fp)) / 2
        tol_f1_fake = 2.0 * tn / (2.0 * tn + fn + fp)
        tol_f1_real = 2.0 * tp / (2.0 * tp + fn + fp)

        # tol_mIoU = (tp/(fn+fp+tp) + tn/(fn+fp+tn)) / 2
        tol_mIoU_fake = tn/(fn+fp+tn)
        tol_mIoU_real = tp/(fn+fp+tp)

        # tol_mcc = (tp*tn - fp*fn) / np.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))

        return tol_f1_fake, tol_f1_real, tol_mIoU_fake, tol_mIoU_real
    
    

class PixMetric2(object):
    def __init__(self, num_class=2):
        super(PixMetric2, self).__init__()
        self.numClass = num_class

    def BatchConfusionMatrix(self, imgPredict, imgLabel): 
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

 
    def total_score(self, total_matrix):
        tp, fp, fn, tn = total_matrix.ravel()

        tol_acc = (tp+tn) / (tp+fp+fn+tn)

        tol_f1 = (2.0 * tp / (2.0 * tp + fn + fp) + 2.0 * tn / (2.0 * tn + fn + fp)) / 2
        tol_f1_tn = 2.0 * tn / (2.0 * tn + fn + fp)
        tol_f1_tp = 2.0 * tp / (2.0 * tp + fn + fp)

        tol_mIoU = (tp/(fn+fp+tp) + tn/(fn+fp+tn)) / 2
        tol_mIoU_tn = tn/(fn+fp+tn)
        tol_mIoU_tp = tp/(fn+fp+tp)

        tol_mcc = (tp*tn - fp*fn) / np.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))

        return tol_mIoU_tn, tol_mIoU_tp, tol_mIoU, tol_f1, tol_f1_tn, tol_f1_tp
    

class PixMetricFake(object):
    def __init__(self, score_thresh=0.5):
        super(PixMetricFake, self).__init__()
        self.score_thresh = score_thresh

    def compute_score(self, pred, gt):
        """

        Args:
            pred : numpy type, shape: H*W
            gt : numpy type, shape: H*W

        Returns:
            iou : the fake class iou, np.float64
            f1 : the fake class f1, np.float64
        """
        pred = (pred > self.score_thresh).astype(np.float64)
        pred, gt = pred.flatten(), gt.flatten()
        # calculate true positives, false positives, false negatives and true negatives
        tp = np.sum(np.logical_and(pred, gt))
        fp = np.sum(np.logical_and(pred, np.logical_not(gt)))
        fn = np.sum(np.logical_and(np.logical_not(pred), gt))
        tn = np.sum(np.logical_and(np.logical_not(pred), np.logical_not(gt)))

        # calculate F1 score
        f1 = 2 * tp / (2 * tp + fn + fp)

        # calculate IoU
        iou = tp / (tp + fn + fp)

        return iou, f1
