"""
Created on Fri Jan 10 14:34:33 2020

@author: erich
To see what you have to do manually:
    Search for the keyword: changeManually
"""
import os, pandas as pd, matplotlib.pyplot as plt, numpy as np
pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)



def DoListOfLists(df):
    '''
    df = DataFrame of the csv-file generated via detection/classification
    '''
    TimeStamps = np.unique([i.split('.')[-2] for i in df['Image']]).argsort()
    
    Lists = [[] for i in range(len(TimeStamps))]
    
    
    for (label,name) in zip(df['Class'],df['Image']):
        Lists[int(name.split('.')[-2])].append(label)

        
    return Lists

def DoListsforPlots(*args):
    '''
    Input:
        DataFrames and Error-Rates
    Output:
        Two lists:
            a list of lists for the plots
            a list of lists with values for the error-bar
    '''
    Classes = ['bud_on_shmootip', 'budding_shmoo', 'inactive', 'budding', 'shmoo']
    Plotlists = []
    yErrorRates = []
    yErrors=[]
    
    for i in args:
        if type(i)==float:
            yErrorRates.append(i)
    for i in args:
        if yErrorRates!=0 and type(i)!=float:
            Lists0 = DoListOfLists(i)
            A = [List.count(Classes[2])/len(List) for List in Lists0] #fraction of inactives
            B = [(List.count(Classes[0])+List.count(Classes[1])+List.count(Classes[4]))/len(List) for List in Lists0] #fraction of shmoos (with and without buds)
            C = [(List.count(Classes[0])+List.count(Classes[1])+List.count(Classes[3]))/len(List) for List in Lists0] #fraction of all budding cells
            D = [List.count(Classes[0])/(List.count(Classes[0])+List.count(Classes[1])) for List in Lists0] #fraction of bud_on_shmootip
            E = [List.count(Classes[1])/(List.count(Classes[0])+List.count(Classes[1])) for List in Lists0] #fraction of budding_shmoo
            PlotList0 = [A,B,C,D,E]
            ErrorRate = yErrorRates.pop(0)
            yError0 = [[fraction*ErrorRate for fraction in Plot] for Plot in PlotList0]
            Plotlists.append(PlotList0)
            yErrors.append(yError0)
        
    return Plotlists, yErrors

def DoBarPlot(Plotlists, yErrors, t0, cols, rows, IMG_NAME ,SAVE_DIR):
    Titles = {'(a)':'Fraction of inactive cells','(b)':'Fraction of all shmoos','(c)':'BuddingIndex'}
    Keys =  [key for key in Titles.keys()]
    Plotlists = Plotlists[:-1]
    Plotlist0 = Plotlists[0][:3]
    Plotlist1 = Plotlists[1][:3]
    yErrors0 = yErrors[0]
    yErrors1 = yErrors[1]
    xticks = [str(i*5) for i in range(1,len(Plotlists[0][0])+1)]
    fig, axes = plt.subplots(rows,cols,figsize=(10,10), sharex=True, sharey=True)
    width = 0.5
    for i in range(len(Plotlist0)):
        col=i%rows
        row=int(i/rows)
        if len(axes.shape)==2:
            ax=axes[col][row]
        elif len(axes.shape)==1:
            ax=axes[col]
        ax.set_ylim(0,0.9)
        
        ax.bar(np.arange(len(xticks))+width/2, Plotlist0[i], width=width, color='black', yerr=yErrors0[i], ecolor='red')
        ax.bar(np.arange(len(xticks))-width/2, Plotlist1[i], width=width, color='gray', yerr=yErrors1[i], ecolor='red')
        ax.set_xticks((0,int((len(Plotlists[0][0])-1)/2),len(Plotlists[0][0])-1))
        ax.set_xticklabels([str(t0), str(int((t0+(len(Plotlists[0][0])+1)*5)/2)), str(t0+(len(Plotlists[0][0])+1)*5)], fontsize=16)
        #ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=16)
        if row==0:
            ax.set_ylabel(f'Fraction', fontsize=16)
        if col==rows-1:
            ax.set_xlabel('Time after washing [min]', fontsize=16)
        ax.set_title(Keys[i], fontsize=16)
        plt.tight_layout()  
        #print(row,col)
    #plt.show()
    plt.savefig(os.path.join(SAVE_DIR, IMG_NAME))

def DoStackedBarPlot(Plotlists, yErrors, t0, cols, rows, IMG_NAME, SAVE_DIR):
    Titles = {'(d)':'Approach1', '(e)':'Approach2', '(f)':'Approach3'}
    Keys =  [key for key in Titles.keys()]
    xticks = [str(i*5) for i in range(1,len(Plotlists[0][0])+1)]
    fig, axes = plt.subplots(rows,cols,figsize=(10,4), sharex=True, sharey=True)
    width = 1
    
    for i in range(len(Plotlists)):
        newErrors=[a+b for (a,b) in zip(yErrors[i][-2:][0],yErrors[i][-2:][1])]
        ErrorRange=[a+b for (a,b) in zip(Plotlists[i][-2:][0], newErrors)]
        brightRange=[a-(b/2) for (a,b) in zip(Plotlists[i][-2:][0], newErrors)] #bud_on_shmootip
        Marker=[a+b/4 for (a,b) in zip(Plotlists[i][-2:][0], newErrors)]
        darkRange=[a for a in Plotlists[i][-2:][1]] #budding_shmoo
        col=i%rows
        row=int(i/rows)
        if len(axes.shape)==2:
            ax=axes[col][row]
        elif len(axes.shape)==1:
            ax=axes[row]
        ax.set_ylim(0,1)
        ax.bar(np.arange(len(xticks)), ErrorRange, width=width, color='red')
        ax.bar(np.arange(len(xticks)), brightRange, width=width, color='0.5')
        ax.bar(np.arange(len(xticks)), darkRange, bottom=ErrorRange, width=width, color='0.15')
        ax.plot(Marker, '_', markersize=10, color='black')
        ax.set_xticks((0,int((len(Plotlists[0][0])-1)/2),len(Plotlists[0][0])-1))
        ax.set_xticklabels([str(t0), str(int((t0+(len(Plotlists[0][0])+1)*5)/2)), str(t0+(len(Plotlists[0][0])+1)*5)], fontsize=16)
        if row==0:
            ax.set_ylabel(f'Fraction', fontsize=16)
        ax.set_xlabel('Time after washing [min]', fontsize=16)
        ax.set_title(Keys[i], fontsize=16)
        plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(SAVE_DIR, IMG_NAME))        
    
    
Classes = ['bud_on_shmootip', 'budding_shmoo', 'inactive', 'budding', 'shmoo']

BuddingTimeDir0 = '/home/erich/Dokumente/AF120/BuddingTime7_1class/Detection/'
BuddingTimeDir1='/home/erich/Dokumente/AF120/BuddingTime7_5class/Detection/'
BuddingTimeDir2 = '/home/erich/Dokumente/AF120/BuddingTime7_2class/Detection/'

CSV_NAME = 'detected.csv'

if os.path.exists(os.path.join(BuddingTimeDir0, CSV_NAME)):
    df0 = pd.read_csv(os.path.join(BuddingTimeDir0, CSV_NAME))
else:
    raise FileNotFoundError(f'Check if your directory, {BuddingTimeDir0}, is correct and try again')
    
if os.path.exists(os.path.join(BuddingTimeDir1, CSV_NAME)):
    df1 = pd.read_csv(os.path.join(BuddingTimeDir1, CSV_NAME))
else:
    raise FileNotFoundError(f'Check if your directory, {BuddingTimeDir1}, is correct and try again')
    
if os.path.exists(os.path.join(BuddingTimeDir2, CSV_NAME)):
    df2 = pd.read_csv(os.path.join(BuddingTimeDir2, CSV_NAME))
else:
    raise FileNotFoundError(f'Check if your directory, {BuddingTimeDir2}, is correct and try again')
Error_rate0=0.116
Error_rate1=0.102
Error_rate2=0.093
Plotlists, yErrors = DoListsforPlots(df0, df1, df2, Error_rate0, Error_rate1, Error_rate2)

t0=20
cols=1
rows=3
IMG_NAME='TopDown'
SAVE_DIR = '/home/erich/Schreibtisch/BA/Total File/'
#DoBarPlot(Plotlists=Plotlists, yErrors=yErrors, t0=t0, cols=cols, rows=rows, IMG_NAME=IMG_NAME, SAVE_DIR=SAVE_DIR)
cols=3
rows=1
IMG_NAME='LeftRight'
#DoStackedBarPlot(Plotlists=Plotlists, yErrors=yErrors, t0=t0, cols=cols, rows=rows, IMG_NAME=IMG_NAME, SAVE_DIR=SAVE_DIR)




'''
Fractions = [A,B,C,D,E]

xticks = [str(i*5) for i in range(1,len(Lists)+1)]

cols=1
rows=3
fig, axes = plt.subplots(rows,cols,figsize=(10,15), sharex=True, sharey=True)
t0 = 20#  changeManually

for i in range(len(Fractions)):
    col=i%rows
    row=int(i/rows)
    ax=axes[col][row]
    ax.set_ylim(0,0.9)
    
    width=0.88 # changeManually eventually 
    ax.bar(np.arange(len(xticks)), Fractions[i], width=width, color='black')
    ax.set_xticks((0,int((len(Lists)-1)/2),len(Lists)-1))
    ax.set_xticklabels([str(t0), str(int((t0+(len(Lists)+1)*5)/2)), str(t0+(len(Lists)+1)*5)], fontsize=16)
    #ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=16)
    if row==0:
        ax.set_ylabel(f'Fraction', fontsize=16)
    if col==rows-1:
        ax.set_xlabel('Time after washing [min]', fontsize=16)
    ax.set_title(Keys[i], fontsize=16)
    plt.tight_layout()  
    #print(row,col)

IMG_NAME = 'BudTest7_1Class3' #Name depends on setup and model(s) used # changeManually
plt.show()
#plt.savefig(os.path.join(BuddingTimeDir, IMG_NAME))
'''




























