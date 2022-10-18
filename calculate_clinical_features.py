import numpy as np
from image_utils import *
import pandas as pd

def calculate_FeatureCurve(seg_all):
    Feature_curve = {'LVV': [],
                     'LVM': [],
                     'RVV': []}
    # calculate the similarity between ED and all phases of the 4D volume to find the highest one
    for t in range(0, 20):
        seg = seg_all[t]
        pixdim = [1.9999913, 1.25, 1.25]
        volume_per_pix = pixdim[0] * pixdim[1] * pixdim[2] * 1e-3  # 0.003124987363815308
        density = 1.05


        # Clinical measures
        Feature_curve['LVV'].append(np.sum(seg == 1) * volume_per_pix)
        Feature_curve['LVM'].append(np.sum(seg == 2) * volume_per_pix * density)
        Feature_curve['RVV'].append(np.sum(seg == 4) * volume_per_pix)

    return Feature_curve

def Calculate_clinical_value_batch_EDES(segED, segES):
    #val_all=[]
    lvedv=[]
    lvm=[]
    rvedv = []
    lvesv = []
    rvesv =[]
    for b in range(0,segED.shape[0]):
        val = Calculate_clinical_value_EDES(segED[b],segES[b])
        #val_all.append(val)
        lvedv.append(val["LVEDV (mL)"])
        lvm.append(val['LVM (g)'])
        rvedv.append(val['RVEDV (mL)'])
        lvesv.append(val['LVESV'])
        rvesv.append(val['RVESV'])

    results = {'LVEDV (mL)': lvedv, 'LVM (g)': lvm,
               'RVEDV (mL)': rvedv, 'LVESV (mL)': lvesv,
               'RVESV (mL)': rvesv}
    val_df = pd.DataFrame.from_dict(results)
    return val_df

def Calculate_clinical_value(seg):
    # seg: label
    val = {}
    # Clinical measures
    pixdim = [1.9999913, 1.25, 1.25]
    volume_per_pix = pixdim[0] * pixdim[1] * pixdim[2] * 1e-3
    seg_ED = seg[0,...]
    seg_ES, index = find_ES(seg)
     # 0.003124987363815308
    density = 1.05
    duration_per_cycle = 1.0
    heart_rate = 60.0 / duration_per_cycle
    val['LVEDV (mL)'] = np.sum(seg_ED == 1) * volume_per_pix
    val['LVM (g)'] = np.sum(seg_ED == 2) * volume_per_pix * density
    val['RVEDV (mL)'] = np.sum(seg_ED == 4) * volume_per_pix

    val['LVESV'] = np.sum(seg_ES == 1) * volume_per_pix
    val['LVESM'] = np.sum(seg_ES == 2) * volume_per_pix * density
    val['RVESV'] = np.sum(seg_ES == 4) * volume_per_pix

    val['LVSV'] = val['LVEDV (mL)'] - val['LVESV']
    val['LVCO'] = val['LVSV'] * heart_rate * 1e-3
    val['LVEF (%)'] = val['LVSV'] / val['LVEDV (mL)'] * 100

    val['RVSV'] = val['RVEDV (mL)'] - val['RVESV']
    val['RVCO'] = val['RVSV'] * heart_rate * 1e-3
    val['RVEF (%)'] = val['RVSV'] / val['RVEDV (mL)'] * 100
    return val


def Calculate_clinical_value_EDES(seg_ED,seg_ES):
    # seg: label
    val = {}
    # Clinical measures
    pixdim = [1.9999913, 1.25, 1.25]
    volume_per_pix = pixdim[0] * pixdim[1] * pixdim[2] * 1e-3

     # 0.003124987363815308
    density = 1.05
    duration_per_cycle = 1.0
    heart_rate = 60.0 / duration_per_cycle
    val['LVEDV (mL)'] = np.sum(seg_ED == 1) * volume_per_pix
    val['LVM (g)'] = np.sum(seg_ED == 2) * volume_per_pix * density
    val['RVEDV (mL)'] = np.sum(seg_ED == 4) * volume_per_pix

    val['LVESV'] = np.sum(seg_ES == 1) * volume_per_pix
    val['LVESM'] = np.sum(seg_ES == 2) * volume_per_pix * density
    val['RVESV'] = np.sum(seg_ES == 4) * volume_per_pix

    val['LVSV'] = val['LVEDV (mL)'] - val['LVESV']
    val['LVCO'] = val['LVSV'] * heart_rate * 1e-3
    val['LVEF (%)'] = val['LVSV'] / val['LVEDV (mL)'] * 100

    val['RVSV'] = val['RVEDV (mL)'] - val['RVESV']
    val['RVCO'] = val['RVSV'] * heart_rate * 1e-3
    val['RVEF (%)'] = val['RVSV'] / val['RVEDV (mL)'] * 100
    return val


