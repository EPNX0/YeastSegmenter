"""
Created on Fri Dec 13 15:45:28 2019

@author: erich
"""
import os,sys,numpy as np, pandas as pd, time

CSV_DIR = '/home/erich/Dokumente/1ClassDetection/Detection/'
CSV_NAME = 'detected.csv'

df = pd.read_csv(os.path.join(CSV_DIR, CSV_NAME))

