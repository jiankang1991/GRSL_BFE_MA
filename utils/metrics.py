from torch import nn

from torch.nn import functional as F

import torch
import math

esp = 1e-5



class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



# https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
# https://github.com/ternaus/robot-surgery-segmentation/blob/master/evaluate.py
def jaccard_index(input, target):
    """IoU calculation """
    num_in_target = input.size(0)

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    # intersection = (pred*truth).long().sum(1).data.cpu()[0]
    intersection = (pred * truth).sum(1)

    # union = input.long().sum().data.cpu()[0] + target.long().sum().data.cpu()[0] - intersection
    union = pred.sum(1) + truth.sum(1) - intersection

    score = (intersection + 1e-15) / (union + 1e-15)

    return score.mean().item()


# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(input, target):
    num_in_target = input.size(0)

    smooth = 1.

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    loss = (2. * intersection + smooth) /(pred.sum(1) + truth.sum(1) + smooth)

    return loss.mean().item()


#### for evaluation
# https://github.com/huster-wgm/geoseg/blob/master/src/metrics.py

def _binarize(y_data, threshold):
    """
    args:
        y_data : [float] 4-d tensor in [batch_size, chs, img_rows, img_cols]
        threshold : [float] [0.0, 1.0]
    return 3-d binarized [int] y_data
    """
    y_data[y_data < threshold] = 0.0
    y_data[y_data >= threshold] = 1.0
    return y_data[:,0,:,:].int()


def _argmax(y_data, dim):
    """
    args:
        y_data : 4-d tensor in [batch_size, chs, img_rows, img_cols]
        dim : int
    return 3-d [int] y_data
    """
    return torch.argmax(y_data, dim).int()


def _get_tp(y_pred, y_true):
    """
    args:
        y_true : [int] 3-d in [batch_size, img_rows, img_cols]
        y_pred : [int] 3-d in [batch_size, img_rows, img_cols]
    return [float] true_positive
    """
    return torch.sum(y_true * y_pred).float()


def _get_fp(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] false_positive
    """
    return torch.sum((1 - y_true) * y_pred).float()


def _get_tn(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] true_negative
    """
    return torch.sum((1 - y_true) * (1 - y_pred)).float()


def _get_fn(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] false_negative
    """
    return torch.sum(y_true * (1 - y_pred)).float()


def _get_weights(y_true, nb_ch):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        nb_ch : int 
    return [float] weights
    """
    batch_size, img_rows, img_cols = y_true.shape
    pixels = batch_size * img_rows * img_cols
    weights = [torch.sum(y_true==ch).item() / pixels for ch in range(nb_ch)]
    return weights


class CFMatrix(object):
    def __init__(self, des=None):
        self.des = des

    def __repr__(self):
        return "ConfusionMatrix"

    def __call__(self, y_pred, y_true, threshold=0.5):

        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return confusion matrix
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fp = _get_fp(y_pred, y_true)
            nb_tn = _get_tn(y_pred, y_true)
            nb_fn = _get_fn(y_pred, y_true)
            mperforms = [nb_tp, nb_fp, nb_tn, nb_fn]
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 4).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fp = _get_fp(y_pred_ch, y_true_ch)
                nb_tn = _get_tn(y_pred_ch, y_true_ch)
                nb_fn = _get_fn(y_pred_ch, y_true_ch)
                performs[int(ch), :] = [nb_tp, nb_fp, nb_tn, nb_fn]
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class OAAcc(object):
    def __init__(self, des="Overall Accuracy"):
        self.des = des

    def __repr__(self):
        return "OAcc"

    def __call__(self, y_pred, y_true, threshold=0.5):
        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return (tp+tn)/total
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)

        nb_tp_tn = torch.sum(y_true == y_pred).float()
        mperforms = nb_tp_tn / (batch_size * img_rows * img_cols)
        performs = None
        return mperforms, performs


class Precision(object):
    def __init__(self, des="Precision"):
        self.des = des

    def __repr__(self):
        return "Prec"

    def __call__(self, y_pred, y_true, threshold=0.5):
        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return tp/(tp+fp)
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fp = _get_fp(y_pred, y_true)
            mperforms = nb_tp / (nb_tp + nb_fp + esp)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fp = _get_fp(y_pred_ch, y_true_ch)
                performs[int(ch)] = nb_tp / (nb_tp + nb_fp + esp)
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class Recall(object):
    def __init__(self, des="Recall"):
        self.des = des

    def __repr__(self):
        return "Reca"

    def __call__(self, y_pred, y_true, threshold=0.5):
        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return tp/(tp+fn)
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fn = _get_fn(y_pred, y_true)
            mperforms = nb_tp / (nb_tp + nb_fn + esp)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fn = _get_fn(y_pred_ch, y_true_ch)
                performs[int(ch)] = nb_tp / (nb_tp + nb_fn + esp)
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class F1Score(object):
    def __init__(self, des="F1Score"):
        self.des = des

    def __repr__(self):
        return "F1Sc"

    def __call__(self, y_pred, y_true, threshold=0.5):

        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return 2*precision*recall/(precision+recall)
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fp = _get_fp(y_pred, y_true)
            nb_fn = _get_fn(y_pred, y_true)
            _precision = nb_tp / (nb_tp + nb_fp + esp)
            _recall = nb_tp / (nb_tp + nb_fn + esp)
            mperforms = 2 * _precision * _recall / (_precision + _recall + esp)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fp = _get_fp(y_pred_ch, y_true_ch)
                nb_fn = _get_fn(y_pred_ch, y_true_ch)
                _precision = nb_tp / (nb_tp + nb_fp + esp)
                _recall = nb_tp / (nb_tp + nb_fn + esp)
                performs[int(ch)] = 2 * _precision * \
                    _recall / (_precision + _recall + esp)
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class Kappa(object):
    def __init__(self, des="Kappa"):
        self.des = des

    def __repr__(self):
        return "Kapp"

    def __call__(self, y_pred, y_true, threshold=0.5):

        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return (Po-Pe)/(1-Pe)
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fp = _get_fp(y_pred, y_true)
            nb_tn = _get_tn(y_pred, y_true)
            nb_fn = _get_fn(y_pred, y_true)
            nb_total = nb_tp + nb_fp + nb_tn + nb_fn
            Po = (nb_tp + nb_tn) / nb_total
            Pe = ((nb_tp + nb_fp) * (nb_tp + nb_fn) +
                  (nb_fn + nb_tn) * (nb_fp + nb_tn)) / (nb_total**2)
            mperforms = (Po - Pe) / (1 - Pe + esp)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fp = _get_fp(y_pred_ch, y_true_ch)
                nb_tn = _get_tn(y_pred_ch, y_true_ch)
                nb_fn = _get_fn(y_pred_ch, y_true_ch)
                nb_total = nb_tp + nb_fp + nb_tn + nb_fn
                Po = (nb_tp + nb_tn) / nb_total
                Pe = ((nb_tp + nb_fp) * (nb_tp + nb_fn)
                      + (nb_fn + nb_tn) * (nb_fp + nb_tn)) / (nb_total**2)
                performs[int(ch)] = (Po - Pe) / (1 - Pe + esp)
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class Jaccard(object):
    def __init__(self, des="Jaccard"):
        self.des = des

    def __repr__(self):
        return "Jacc"

    def __call__(self, y_pred, y_true, threshold=0.5):
        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return intersection / (sum-intersection)
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            _intersec = torch.sum(y_true * y_pred).float()
            _sum = torch.sum(y_true + y_pred).float()
            mperforms = _intersec / (_sum - _intersec + esp)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                _intersec = torch.sum(y_true_ch * y_pred_ch).float()
                _sum = torch.sum(y_true_ch + y_pred_ch).float()
                performs[int(ch)] = _intersec / (_sum - _intersec + esp)
            mperforms = sum([i*j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class SSIM(object):
    '''
    modified from https://github.com/jorge-pessoa/pytorch-msssim
    '''
    def __init__(self, des="structural similarity index"):
        self.des = des

    def __repr__(self):
        return "SSIM"

    def gaussian(self, w_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
        return gauss/gauss.sum()

    def create_window(self, w_size, channel=1):
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def __call__(self, y_pred, y_true, w_size=11, size_average=True, full=False):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            w_size : int, default 11
            size_average : boolean, default True
            full : boolean, default False
        return ssim
        """
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if torch.max(y_pred) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(y_pred) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val

        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel=channel).to(y_pred.device)

        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret

def generate_data(batch_size, chs, img_rows, img_cols):
    """
    args:
        batch_size : int
        chs : int
        img_rows : int
        img_cols : int
    return y_pred_fake, y_true_fake
    """
    pixel = img_cols // chs
    border = 1
    y_true = torch.zeros(batch_size, img_rows, img_cols)
    for i in range(chs):
        y_true[:, :, i*pixel:(i+1)*pixel] = i
    y_pred = y_true.clone()
    y_pred[:, :border, :] = 0
    y_pred[:, img_rows-border:, :] = 0

    y_true_fake = torch.zeros(batch_size, chs, img_rows, img_cols)
    y_pred_fake = torch.zeros(batch_size, chs, img_rows, img_cols)
    for i in range(chs):
        y_true_fake[:,i,:,:][y_true==i] = 1.0
        y_pred_fake[:,i,:,:][y_pred==i] = 1.0
    return y_pred_fake, y_true_fake


# relative volume error evaluation
def binary_relative_volume_error(s_volume, g_volume):
    """ 
    https://github.com/HiLab-git/PyMIC/blob/4046bea9371592dcc55b3a5df108482586e9450f/pymic/util/evaluation.py
    """
    s_v = float(s_volume.sum())
    g_v = float(g_volume.sum())
    assert(g_v > 0)
    rve = abs(s_v - g_v)/g_v
    return rve

if __name__ == "__main__":
    # test
    for chs in [1, 2]:
        for cuda in [True, False]:
            batch_size, img_rows, img_cols = 1, 5, 5
            y_pred, y_true = generate_data(batch_size, chs, img_rows, img_cols)
            if cuda:
                y_pred = y_pred.cuda()
                y_true = y_true.cuda()
            # print(y_true.shape, torch.argmax(y_true, 1))
            # print(y_pred.shape, torch.argmax(y_pred, 1))

            metric = OAAcc()
            maccu, accu = metric(y_pred, y_true)
            print('mAccu:', maccu, 'Accu', accu)

            metric = Precision()
            mprec, prec = metric(y_pred, y_true)
            print('mPrec:', mprec, 'Prec', prec)

            metric = Recall()
            mreca, reca = metric(y_pred, y_true)
            print('mReca:', mreca, 'Reca', reca)

            metric = F1Score()
            mf1sc, f1sc = metric(y_pred, y_true)
            print('mF1sc:', mf1sc, 'F1sc', f1sc)

            metric = Kappa()
            mkapp, kapp = metric(y_pred, y_true)
            print('mKapp:', mkapp, 'Kapp', kapp)

            metric = Jaccard()
            mjacc, jacc = metric(y_pred, y_true)
            print('mJacc:', mjacc, 'Jacc', jacc)










