#calculate the similarity of two time series with same length

"""Module exposing surface distance based measures."""


"""from __future__ import absolute_import
from __future__ import division
from __future__ import print_function"""
from scipy import ndimage
import numpy as np
import dtaidistance
from sklearn import linear_model

from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, \
    generate_binary_structure
from scipy.ndimage.measurements import label, find_objects
from scipy.stats import pearsonr
from calculate_clinical_features import *
import similaritymeasures
import medpy.metric.binary as mpy
import scipy.stats
from image_utils import *
class Similarity(object):
#s1 = Similarity(exp_data, num_data).use_method('dtw_r'), exp_data:(n,)

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def use_method(self, method, *args):
        if 'eu' in method:
            return self.euclid(args[0]) if len(args) > 0 else self.euclid()
        if 'epc' == method:#not 0-1
            return self.pearson_coefficient(self.x,self.y)
        if 'tcc' in method: #lvm wrong
            return self.temporal_correlation_coefficient()
        if 'ecc' in method: #too high
            return self.easy_cross_correlation(self.x,self.y)#too high
        if 'mcc' in method:
            return self.max_cross_correlation()
        if 'mpc' in method:# not 0-1
            return self.max_pearson_coefficient()
        if 'dtw_r' in method:#all 1
            return self.dtw_raw()
        if 'dtw_s' in method:
            return self.dtw_shift(shiftcount=int(method.strip("dtw_s")))
        if 'dtw_m_l' in method:
            return self.dtw_map_l()
        if 'dtw_m_p' in method:
            return self.dtw_map_p()
        if 'elc' in method:# low in LVM
            X=[[e] for e in self.x]
            return self.easy_linear_correlation(X,self.y)
        if 'mlc' in method:# all 1
            return self.max_linear_correlation()
        else:
            raise NotImplementedError(method)

    def euclid(self, p=1):
        return 1 / (1 + (sum([abs(x - y) ** p for x, y in zip(self.x, self.y)]) / len(self.x)) ** (1 / p))


    # 可以作为距离的系数
    def temporal_correlation_coefficient(self):
        x_diff = [t2 - t1 for t1, t2 in zip(self.x, self.x[1:])]
        y_diff = [t2 - t1 for t1, t2 in zip(self.y, self.y[1:])]
        return sum([t1 * t2 for t1, t2 in zip(x_diff, y_diff)]) / (1 + sum([t * t for t in x_diff]) ** 0.5 * sum([t * t for t in y_diff]) ** 0.5)

    def dtw_raw(self, window=20, penalty=0.01, psi=20, max_step=1, d=1):
        cx = np.power(self.x, d)
        cy = np.power(self.y, d)
        dis = dtaidistance.dtw.distance_fast(cx, cy, window=window, penalty=penalty, psi=psi, max_step=max_step)
        return 1 / (1 + dis)
    def dtw_shift(self,shiftcount=5,shiftstep=0.04, window=20, penalty=0.01, psi=20, max_step=1, d=1):
        cx=np.power(self.x,d)
        minDis=np.inf
        for i in range(2*shiftcount):
            cy = np.power(self.y+shiftstep*(i-shiftcount), d)
            dis=dtaidistance.dtw.distance_fast(cx, cy,window=window,penalty=penalty,psi=psi,max_step=max_step)
            if(minDis>dis):
                minDis=dis
                self.bestShiftY=shiftstep*(i-shiftcount)
        return 1/(1+minDis)
    def dtw_map_l(self, window=20, penalty=0.01, psi=20, max_step=1, d=1):
        cx=np.power(self.x,d)
        cy = np.power(self.y, d)
        dis, paths = dtaidistance.dtw.warping_paths_fast(cx, cy, window=window,penalty=penalty,psi=psi,max_step=max_step)
        best_path = dtaidistance.dtw.best_path(paths)
        mapped_x = np.array([cx[p[0]] for p in best_path])
        mapped_y = np.array([cy[p[1]] for p in best_path])
        X=[[e] for e in mapped_x]
        return self.easy_linear_correlation(X,mapped_y)
    def dtw_map_p(self, window=20, penalty=0.01, psi=20, max_step=1, d=1):
        cx=np.power(self.x,d)
        cy = np.power(self.y, d)
        dis, paths = dtaidistance.dtw.warping_paths_fast(cx, cy, window=window,penalty=penalty,psi=psi,max_step=max_step)
        best_path = dtaidistance.dtw.best_path(paths)
        mapped_x = np.array([cx[p[0]] for p in best_path])
        mapped_y = np.array([cy[p[1]] for p in best_path])
        X=[[e] for e in mapped_x]
        return self.pearson_coefficient(X,mapped_y)

    def easy_cross_correlation(self,x,y):

        t = zip(x, y)
        L = np.array(list(t))
        s = np.sum(L[:, 0] * L[:, 1])
        L1 = np.sum(np.square(L[:, 0]))
        L2 = np.sum(np.square(L[:, 1]))
        if s == 0:
            return 0
        else:
            return s / (L1 ** 0.5) / (L2 ** 0.5)
    def max_cross_correlation(self):
        cor=[]
        for i in range(1, 20):
            cor.append(self.easy_cross_correlation(self.x[i:], self.y))
        for i in range(0, 20):
            cor.append(self.easy_cross_correlation(self.x, self.y[i:]))
        self.bestShiftX=59-np.argmax(cor)
        return max(cor)

    def pearson_coefficient(self,x,y):
        Z=list(zip(x,y))
        x=[e[0] for e in Z]
        y=[e[1] for e in Z]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        dx=x-x_mean
        dy=y-y_mean
        x_b= np.sum(np.square(dx))**0.5
        y_b= np.sum(np.square(dy))**0.5
        cor=np.sum(dx*dy)/(1+x_b+y_b)
        return cor
    def max_pearson_coefficient(self):
        cor=[]
        for i in range(1, 20):
            cor.append(self.pearson_coefficient(self.x[i:], self.y))
        for i in range(0, 20):
            cor.append(self.pearson_coefficient(self.x, self.y[i:]))
        self.bestShiftX=20-np.argmax(cor)
        return max(cor)
    def easy_linear_correlation(self,X,y):
        Z=list(zip(X,y))
        X=[e[0] for e in Z]
        y=[e[1] for e in Z]
        reg=linear_model.LinearRegression()
        reg.fit(X,y)
        return reg.score(X,y)
    def max_linear_correlation(self):
        cor=[]
        for i in range(1, 20):
            X=[[e] for e in self.x[i:]]
            cor.append(self.easy_linear_correlation(X, self.y))
        X0=[[e] for e in self.x]
        for i in range(0, 20):
            cor.append(self.easy_linear_correlation(X0, self.y[i:]))
        self.bestShiftX=20-np.argmax(cor)
        return max(cor)

def compute_dice_coefficient(mask_gt, mask_pred):
  """Computes soerensen-dice coefficient.
  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`.
  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.
  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN.
  """
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum


def Curve_of_motion (seg, recon_batch, plot, resultpath, index):
    # LVV, LVM, RVV cureve
        #input: one-hot
    s_method ='elc'
    LVV_gt, LVM_gt, RVV_gt = calculate_FeatureCurve(seg.cpu().detach().numpy())
    LVV_pre, LVM_pre, RVV_pre = calculate_FeatureCurve(recon_batch.cpu().detach().numpy())
    time = range(0, 20)

    exp_data = np.zeros((20, 2))
    exp_data[:, 0] = time
    exp_data[:, 1] = LVV_gt

    num_data = np.zeros((20, 2))
    num_data[:, 0] = time
    num_data[:, 1] = LVV_pre
    similarity_LVV, d = similaritymeasures.dtw(exp_data, num_data)
    dis_LVV = np.mean(abs(np.array(LVV_gt)-np.array(LVV_pre)))
    #dtw_LVV = Similarity(exp_data[:, 1], num_data[:, 1]).use_method(s_method)

    exp_data[:, 1] = LVM_gt
    num_data[:, 1] = LVM_pre
    similarity_LVM, d = similaritymeasures.dtw(exp_data, num_data)
    dis_LVM = np.mean(abs(np.array(LVM_gt) - np.array(LVM_pre)))
    #dtw_LVM = Similarity(exp_data[:, 1], num_data[:, 1]).use_method(s_method)

    exp_data[:, 1] = RVV_gt
    num_data[:, 1] = RVV_pre
    similarity_RVV, d = similaritymeasures.dtw(exp_data, num_data)
    dis_RVV = np.mean(abs(np.array(RVV_gt) - np.array(RVV_pre)))

    #
    # names = locals()
    # if plot:
    #     type = 'LVV'
    #     plot_curve_compare(names[f'{type}_gt'], names[f'{type}_pre'], time, type = type, resultpath = resultpath, index = index)
    #     type = 'LVM'
    #     plot_curve_compare(names[f'{type}_gt'], names[f'{type}_pre'], time, type=type, resultpath = resultpath, index = index)
    #     type = 'RVV'
    #     plot_curve_compare(names[f'{type}_gt'], names[f'{type}_pre'], time, type=type, resultpath = resultpath, index = index)

    return similarity_LVV, similarity_LVM, similarity_RVV, dis_LVV, dis_LVM, dis_RVV

def Motion_smilarity_EDES (seg, recon_batch):
    features = ['LVEDV (mL)','LVM (g)','RVEDV (mL)', 'LVESV', 'LVESM', 'RVESV','LVSV','LVCO','LVEF (%)','RVSV','RVCO','RVEF (%)']

    motion_metrics = {}

    for feature in features:
        motion_metrics.update({f'{feature}': []})
    # LVV, LVM, RVV cureve
    # input-data requirement: labelmap, cpu
    Feature_curve_gt = Calculate_clinical_value(seg)
    Feature_curve_pre = Calculate_clinical_value(recon_batch)

    for feature in features:
        motion_metrics.update({feature:
            [abs(np.array(Feature_curve_gt[f'{feature}']) - np.array(Feature_curve_pre[f'{feature}']))]})

    return motion_metrics, Feature_curve_gt, Feature_curve_pre

def Distance_of_sequence(seg, recon_batch, pixdim, cal_assd=True):
    HD_seq = []
    seg = seg

    ASSD_seq = []

    for s in range(0, seg.shape[0]):
        seg_vol = seg[s]
        recon_vol = recon_batch[s]
        hd = np.zeros((3,))
        assd = np.zeros((3,))
        classes = [1, 2, 4]
        for idx, cls in enumerate(classes):
            seg_cls = np.array(seg_vol == cls)
            recon_cls = np.array(recon_vol == cls)
            if 0 == np.count_nonzero(np.array(recon_cls)):
                hd[idx] = 11.5
                if cal_assd:
                    assd[idx] = 3.5
            else:
                hd[idx] = mpy.hd(seg_cls, recon_cls, voxelspacing=pixdim, connectivity=1)
                if cal_assd:
                    assd[idx] = mpy.assd(seg_cls, recon_cls, voxelspacing=pixdim, connectivity=1)
        HD_seq.append(hd)
        ASSD_seq.append(assd)
    return np.array(HD_seq), np.array(ASSD_seq)

def calculate_motion_for_EDES(orgED, orgES, reconED, reconES):
    feature_names = ['LVEDV (mL)', 'LVM (g)', 'RVEDV (mL)', 'LVESV', 'LVESM', 'RVESV', 'LVSV', 'LVCO', 'LVEF (%)',
                     'RVSV', 'RVCO', 'RVEF (%)']
    val = {}
    pixdim = [1.9999913, 1.25, 1.25]
    volume_per_pix = pixdim[0] * pixdim[1] * pixdim[2] * 1e-3

    # 0.003124987363815308
    density = 1.05
    duration_per_cycle = 1.0
    heart_rate = 60.0 / duration_per_cycle
    Feature_curve = {}
    for metric in feature_names:
        for type in ['gt','pre']:
            Feature_curve.update({f'{type}_{metric}':[]})


    # Clinical measures
    for type in ['gt', 'pre']:
        if type =='gt':
            seg_ED = orgED
            seg_ES = orgES
        else:
            seg_ED = reconED
            seg_ES = reconES

        Feature_curve.update({f'{type}_LVEDV (mL)': [np.sum(seg_ED == 1) * volume_per_pix]})
        Feature_curve.update({f'{type}_LVM (g)':[np.sum(seg_ED == 2) * volume_per_pix * density]})
        Feature_curve.update({f'{type}_RVEDV (mL)':[np.sum(seg_ED == 4) * volume_per_pix]})
        Feature_curve.update({f'{type}_LVESV':[np.sum(seg_ES == 1) * volume_per_pix]})
        Feature_curve.update({f'{type}_LVESM':[np.sum(seg_ES == 2) * volume_per_pix * density]})
        Feature_curve.update({f'{type}_RVESV':[ np.sum(seg_ES == 4) * volume_per_pix]})

        Feature_curve.update({f'{type}_LVSV':[np.array(Feature_curve[f'{type}_LVEDV (mL)']) - np.array(Feature_curve[f'{type}_LVESV'])]})
        Feature_curve.update({f'{type}_LVCO':[np.array(Feature_curve[f'{type}_LVSV'])* heart_rate * 1e-3]})
        Feature_curve.update({f'{type}_LVEF (%)':[np.array(Feature_curve[f'{type}_LVSV']) /np.array(Feature_curve[f'{type}_LVEDV (mL)']) * 100]})

        Feature_curve.update({f'{type}_RVSV':[np.array(Feature_curve[f'{type}_RVEDV (mL)'])- np.array(Feature_curve[f'{type}_RVESV'])]})
        Feature_curve.update({f'{type}_RVCO':[np.array(Feature_curve[f'{type}_RVSV']) * heart_rate * 1e-3]})
        Feature_curve.update({f'{type}_RVEF (%)':[np.array(Feature_curve[f'{type}_RVSV']) / np.array(Feature_curve[f'{type}_RVEDV (mL)']) * 100]})
    motion_results = {}

    for metric in feature_names:
        motion_results.update({f'{metric}': []})
    for metric in feature_names:
        motion_results.update\
            ({f'{metric}':
                      [abs(np.array(Feature_curve[f'gt_{metric}']) - np.array(Feature_curve[f'pre_{metric}']))]})
    return motion_results

def Dice_of_sequence(seg, recon_batch):
    #datarequire: labelmap, cpu
    dice_seq = []

    for s in range(0, seg.shape[0]):
        seg_vol = seg[s]
        recon_vol = recon_batch[s]
        dsc = np_mean_dice(recon_vol, seg_vol)
        dice_seq.append(dsc)

    return np.array(dice_seq)

def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p,M)+0.5*scipy.stats.entropy(q, M)

def KL_divergence(p,q):
    return scipy.stats.entropy(p, q)

def Bhattacharyya_Distance(p,q):
    BC = np.sum(np.sqrt(p * q))
    return -np.log(BC)

def Wasserstein_distance(p,q):
    return scipy.stats.wasserstein_distance(p, q)