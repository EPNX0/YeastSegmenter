"""
Created on Thu May 16 14:28:38 2019

@author: erich
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math

        
def sort_data(gt_class_ids, class_ids, gt_bbox, rois, rtol=0, atol=10):
    '''Takes in:
    gt_class_ids: numpy array of groundtruth class IDs of the original masks
    class_ids: numpy array of result class IDs of the generated masks
    gt_bbox: numpy array of groundtruth bounding boxes of the original masks
    rois: numpy array of result bounding boxes of the generated masks
    rtol: Int to determine the relative tolrance for numpy.allclose()
    atol: Int to determine the absolut tolerance for numpy.allclose()
    Returns:
    sorted_gtr: A list of the sorted groundtruth bounding boxes of the original masks
    sorted_r: A list of the sorted result bounding boxes of the generated masks
    sorted_gtIDs: A list of the sorted groundtruth class IDs of the original masks
    sorted_IDs:  A list of the sorted result bounding boxes of the generated masks
    '''
    tmp_IDs=class_ids.tolist()
    tmp_gtIDs=gt_class_ids.tolist()
    tmp_gtr=gt_bbox.tolist()
    tmp_r=rois.tolist()
    y = len(rois)
    sorted_IDs = []
    sorted_gtIDs=[]
    sorted_gtr=[]
    sorted_r=[]

    for a in range(y):
        for b in range(len(tmp_gtr)):
            if np.allclose(tmp_r[0], tmp_gtr[b], rtol=rtol, atol=atol):
                sorted_r.append(tmp_r.pop(0))
                sorted_gtr.append(tmp_gtr.pop(b))
                sorted_IDs.append(tmp_IDs.pop(0))
                sorted_gtIDs.append(tmp_gtIDs.pop(b))
                break
            if b == len(tmp_gtr)-1:
                sorted_r.append(tmp_r.pop(0))
                sorted_gtr.append(np.nan)
                sorted_IDs.append(tmp_IDs.pop(0))
                sorted_gtIDs.append(np.nan)
                break
    while len(tmp_r) != 0:
        sorted_gtr.append(np.nan)
        sorted_r.append(tmp_r.pop(0))
        sorted_gtIDs.append(np.nan)
        sorted_IDs.append(tmp_IDs.pop(0))
        
    while len(tmp_gtr) !=0:
        sorted_r.append(np.nan)
        sorted_gtr.append(tmp_gtr.pop(0))
        sorted_IDs.append(np.nan)
        sorted_gtIDs.append(tmp_gtIDs.pop(0))

    return sorted_gtIDs, sorted_IDs, sorted_gtr, sorted_r

















































'''                
def sort_gt_bboxes(gt_bbox, rois, rtol=0, atol=10):
    Returns a sorted gt_bbox list and an updated rois list with respect to seen/unseen objects
    gt_bbox: array of ground truth bounding boxes
    rois: array of result bounding boxes
    rtol: relative tolerance for np.where
    atol: absolut tolerance for np.where
    
    tmp_gtr=gt_bbox.tolist()
    tmp_rois=rois.tolist()
    sorted_gt_bbox=[]
    sorted_rois=[]
    if len(rois)==len(gt_bbox):
        for a in range(len(rois)):
            for b in range(len(tmp_gtr)):
                if np.allclose(tmp_rois[0], tmp_gtr[b], 
                                   rtol=rtol, atol=atol):
                    tmp=tmp_gtr.pop(b)
                    sorted_gt_bbox.append(tmp)
                    tmp=tmp_rois.pop(0)
                    sorted_rois.append(tmp)
                    break
                if b==len(tmp_gtr)-1:
                    tmp=tmp_rois.pop(0)
                    sorted_rois.append(tmp)
                    sorted_gt_bbox.append(np.nan)
                    break
                
    if len(rois)>len(gt_bbox):
        for a in range(len(rois)-len(gt_bbox)):
            tmp_gtr.append(np.nan)
        for b in range(len(rois)):
            for c in range(len(tmp_gtr)):
                if np.allclose(tmp_rois[0], tmp_gtr[c], 
                               rtol=rtol, atol=atol):
                    tmp=tmp_gtr.pop(c)
                    sorted_gt_bbox.append(tmp)
                    tmp=tmp_rois.pop(0)
                    sorted_rois.append(tmp)
                    break
                if c==len(tmp_gtr)-1:
                    tmp=tmp_rois.pop(0)
                    sorted_rois.append(tmp)
                    sorted_gt_bbox.append(np.nan)
                    break
                
    if len(rois)<len(gt_bbox):
        for a in range(len(gt_bbox)-len(rois)):
            tmp_rois.append(np.nan)
        for b in range(len(tmp_rois)):
            for c in range(len(tmp_gtr)):
                if np.allclose(tmp_rois[0], tmp_gtr[c], 
                               rtol=rtol, atol=atol):
                    tmp=tmp_gtr.pop(c)
                    sorted_gt_bbox.append(tmp) 
                    tmp=tmp_rois.pop(0)
                    sorted_rois.append(tmp)
                    break
                if type(tmp_rois[0])==float:
                    tmp=tmp_rois.pop(0)
                    sorted_rois.append(tmp)
                    tmp=tmp_gtr.pop(c)
                    sorted_gt_bbox.append(tmp)
                    break
                
    return sorted_gt_bbox, sorted_rois      
    
    tmp_gtr=gt_bbox.tolist()
    tmp_rois=rois.tolist()
    sorted_gt_bbox=[]
    sorted_rois=[]
    for a in tmp_rois:
        for b in tmp_gtr:
            if np.allclose(a, b, rtol=rtol, atol=atol):
                sorted_rois.append(a)
                sorted_gt_bbox.append(b)
                break
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            sorted_rois.append(a)
            sorted_gt_bbox.append(np.nan)
            
    return sorted_gt_bbox, sorted_rois


def sort_gt_class_ids(gt_class_ids, class_ids):
    Takes in two arrays and returns two lists, so that 
    the entrances in gt_class_ids do match with those of class_ids. 
    Appends 'nan' to the arrays, if one array is longer than the other
    
    tmp1=class_ids.tolist()
    tmp2=gt_class_ids.tolist()
    sorted_gt_class_ids=[]
    if len(gt_class_ids)==len(class_ids):
        for a in range(len(class_ids)):
            for b in range(len(tmp2)):
                if tmp1[a]==tmp2[b]:
                    tmp=tmp2.pop(b)
                    sorted_gt_class_ids.append(tmp)
                    break
    
    if len(gt_class_ids)>len(class_ids):
        for a in range(len(tmp2)-len(tmp1)):
            tmp1.append('nan')
        for b in range(len(tmp1)):
            for c in range(len(tmp2)):
                if tmp1[b]==tmp2[c]:
                    tmp=tmp2.pop(c)
                    sorted_gt_class_ids.append(tmp)
                    break
        if len(tmp2)!=0:
            for d in range(len(tmp2)):
                tmp=tmp2.pop(0)
                sorted_gt_class_ids.append(tmp)
                
    if len(gt_class_ids)<len(class_ids):
        for a in range(len(class_ids)):
            for b in range(len(tmp2)):
                if tmp1[a]==tmp2[b]:
                    tmp=tmp2.pop(b)
                    sorted_gt_class_ids.append(tmp)
                    break
        for c in range(len(tmp1)):
            try:
                assert len(tmp1)==len(sorted_gt_class_ids)
            except AssertionError:
                sorted_gt_class_ids.append(np.nan)
    return sorted_gt_class_ids, tmp1   




def sort_gt_class_ids(gt_class_ids, class_ids):
    Takes in two arrays and returns two lists, so that 
    the entrances in gt_class_ids do match with those of class_ids. 
    Appends 'nan' to the arrays, if one array is longer than the other
    
    tmp1=class_ids.tolist()
    tmp2=gt_class_ids.tolist()
    sorted_IDs = []
    sorted_gtIDs=[]
    y = len(class_ids)
    for a in range(y):
        for b in range(len(tmp2)):
            if tmp1[0] == tmp2[b]:
                sorted_IDs.append(tmp1.pop(0))
                sorted_gtIDs.append(tmp2.pop(b))
                break
            if b == len(tmp2)-1:
                sorted_IDs.append(tmp1.pop(0))
                sorted_gtIDs.append(np.nan)
                break
    while len(tmp1) != 0:
        sorted_IDs.append(tmp1.pop(0))
        sorted_gtIDs.append(np.nan)
    
    while len(tmp2) != 0:
        sorted_IDs.append(np.nan)
        sorted_gtIDs.append(tmp2.pop(0))

    return sorted_gtIDs, sorted_IDs


def sort_gt_bboxes(gt_bbox, rois, rtol=0, atol=10):
    Returns a sorted gt_bbox list and an updated rois list with respect to seen/unseen objects
    gt_bbox: array of ground truth bounding boxes
    rois: array of result bounding boxes
    rtol: relative tolerance for np.where; default 0
    atol: absolut tolerance for np.where; default 10
    
    
    tmp_gtr=gt_bbox.tolist()
    tmp_r=rois.tolist()
    y = len(rois)
    sorted_gtr=[]
    sorted_r=[]
    for a in range(y):
        for b in range(len(tmp_gtr)):
            if np.allclose(tmp_r[0], tmp_gtr[b], rtol=rtol, atol=atol):
                sorted_r.append(tmp_r.pop(0))
                sorted_gtr.append(tmp_gtr.pop(b))
                break
            if b == len(tmp_gtr)-1:
                sorted_r.append(tmp_r.pop(0))
                sorted_gtr.append(np.nan)
                break
            
    while len(tmp_r) != 0:
        sorted_gtr.append(np.nan)
        sorted_r.append(tmp_r.pop(0))
            
    while len(tmp_gtr) !=0:
        sorted_r.append(np.nan)
        sorted_gtr.append(tmp_gtr.pop(0))
    
    return sorted_gtr, sorted_r
    
'''                
 
                      
                
                
                
                
                
                
                
                

































