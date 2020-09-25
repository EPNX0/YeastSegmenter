"""
Created on Thu Dec 12 12:12:35 2019

@author: erich
"""
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os,numpy as np, pandas as pd

pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

def DoDataForPlot(*args):
    '''
    input:
        DataFrames
    output:
        List of Lists for Plot
    '''
    VIC = ['bud_on_shmootip', 'budding_shmoo'] #Very important Classes ;)
    Angle_Occurences = []
    Angles = np.arange(0,190,10)
    for df in args:
        angle_occurences1 = []
        angle_occurences2 = []
        IDs = []
        for i in range(len(df)):
            if df['Class'][i] in VIC and type(df['Fishy'][i])==float:
                IDs.append(df['Class'][i])
        angles = [angle for angle in list(df['Angles']) if angle >=0.0]
        for i in range(len(Angles)-1):
            count1=0
            count2=0
            for ID, angle in zip(IDs,angles):
                if Angles[i]<=int(angle)<Angles[i+1]:
                    if ID==VIC[0]:
                        count1+=1
                    elif ID==VIC[1]:
                        count2+=1
            angle_occurences1.append(count1)
            angle_occurences2.append(count2)
        angles = [angle_occurences1, angle_occurences2]
        Angle_Occurences.append(angles)

    return Angle_Occurences

def DoPlot(Plotlist, fontsize, rows, cols, IMG_NAME, SAVE_DIR):
    #Make two plots: One for each class
    Titles = {'(a)':'Approach1', '(b)':'Approach2', '(c)':'Approach3'}
    Keys =  [key for key in Titles.keys()]
    Angles = np.arange(0,190,10)
    matplotlib.rcParams.update({'font.size': fontsize}) #changes the fontsize of everything in the plot at once
    xticklabels = [str(i)+'-'+str(i+10) for i in Angles[:-1]]
    x = np.arange(len(xticklabels))
    fig, axes = plt.subplots(rows,cols,figsize=(15,6), sharex=True, sharey=True)
    width=0.5
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    for i in range(len(Angles_Occurences)):
        angle_occurences1 = Angles_Occurences[i][0]
        angle_occurences2 = Angles_Occurences[i][1]
        Error_Range1 = int(angle_occurences1[0]*0.30)
        Error_Range2 = int(angle_occurences2[0]*0.30)
        angle_occurences1[0] = int(angle_occurences1[0]*0.7)
        angle_occurences2[0] = int(angle_occurences2[0]*0.7)
        col=i%rows
        row=int(i/rows)
        if len(axes.shape)==2:
            ax=axes[col][row]
        elif len(axes.shape)==1:
            ax=axes[row]
        ax.bar(x-width/2, angle_occurences1, width=width, color='0.5') #bud_on_shmootip
        ax.bar(x+width/2, angle_occurences2, width=width, color='0.15') #budding_shmoo
        ax.bar(x[0]-width/2, Error_Range1, width=width, color='red', bottom=angle_occurences1[0])
        ax.bar(x[0]+width/2, Error_Range2, width=width, color='red', bottom=angle_occurences2[0])
        ax.set_xticks((0, int(len(x)/2), len(x)-1))
        ax.set_xticklabels([xticklabels[0],xticklabels[int(len(x)/2)],xticklabels[-1]])
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        if row==0:
            ax.set_ylabel('Number of cells')
        ax.set_xlabel('Range of angles')
        ax.set_title(Keys[i])
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, IMG_NAME+'.png'))
    plt.close('all')
    plt.show()

#################################################################################################################################################
###Loading CSV Cellinfos
#################################################################################################################################################    
SAVE_DIR = '/home/erich/Dokumente/Angles'
IMG_NAME = 'LosAngles2'
CSV_NAME = 'CellInfos.csv'
CSV_DIR0 = '/home/erich/Dokumente/1ClassDetectionAngle/Detection/Analysis/' #1536 Cells for Angle calc from 438 images (11496 Cells in total detected)
if os.path.exists(os.path.join(CSV_DIR0, CSV_NAME)):                        #526 angles are badly/falsely calculated
    df0 = pd.read_csv(os.path.join(CSV_DIR0, CSV_NAME))
else:
    raise FileNotFoundError(f'Check if your directory, {CSV_DIR0}, is correct and try again')
CSV_DIR1 = '/home/erich/Dokumente/5ClassDetectionAngle/Detection/Analysis/' #889 Cells for Angle calc from 438 images (9076 Cells in total detected)
if os.path.exists(os.path.join(CSV_DIR1, CSV_NAME)):                        #330 angles are badly/falsely calculated
    df1 = pd.read_csv(os.path.join(CSV_DIR1, CSV_NAME))
else:
    raise FileNotFoundError(f'Check if your directory, {CSV_DIR0}, is correct and try again')
CSV_DIR2 = '/home/erich/Dokumente/2ClassDetectionAngle/Detection/Analysis/' #1195 Cells for Angle calc from 438 images (1212 Cells in total detected)
if os.path.exists(os.path.join(CSV_DIR2, CSV_NAME)):                        #304 angles are badly/falsely calculated
    df2 = pd.read_csv(os.path.join(CSV_DIR2, CSV_NAME))
else:
    raise FileNotFoundError(f'Check if your directory, {CSV_DIR2}, is correct and try again')
Angles_Occurences = DoDataForPlot(df0, df1, df2)
DoPlot(Angles_Occurences, 16, 1, 3, IMG_NAME,  SAVE_DIR)




















