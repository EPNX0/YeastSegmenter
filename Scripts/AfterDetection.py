"""
Created on Fri Nov 29 10:15:43 2019

@author: erich
"""

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


import os
import sys
import time
sys.path.append('/home/basar/Personal/Erich/site-packages') #for tiffile
import math
import numpy as np
import numpy.linalg as la
import pandas as pd
from imageio import imread
from skimage.transform import hough_circle, hough_circle_peaks
import cv2
from scipy import signal
import tifffile
pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

##########################################################################################################################################
# FUNCTIONS
##########################################################################################################################################
##########################################################################################################################################
###Getting DateTime of Image
##########################################################################################################################################
def ExtraData(DataFrame, IMGs_DIR):
    df=DataFrame
    Time=[]
    PhysicalSizeX = []
    PhysicalSizeY = []
    Unit = []
    print('Getting extra data from the Images...')
    for i in range(len(df)):
        if os.path.exists(os.path.join(IMGs_DIR, df['Image'][i])):
            f = tifffile.TiffFile(os.path.join(IMGs_DIR, df['Image'][i]))
            tiff_tags={}
            for tag in f.pages[0].tags.values():
                name,value = tag.name, tag.value
                tiff_tags[name] = value
            f.close()
            try:
                Data = tiff_tags['IJMetadata']['Info']
                Data = Data.split('\n')
                for i in Data:
                    if i.startswith('DateTime'):
                        Time.append((i.split(' ')[-1]))
                    elif i.startswith('\t\t<OME:Pixels'):
                        Data2 = i.split(' ')
                        for j in Data2:
                            if j.startswith('PhysicalSizeX='):
                                PhysicalSizeX.append(j[-5:-1])
                            elif j.startswith('PhysicalSizeY='):
                                PhysicalSizeY.append(j[-5:-1])
                            elif j.startswith('PhysicalSizeXUnit='):
                                Unit.append(j[-3:-1])
            except KeyError:
                Time.append(tiff_tags['DateTime'])
    #print(len(Time))

    return Time, PhysicalSizeX, PhysicalSizeY, Unit
##########################################################################################################################################
# sorting masklist
##########################################################################################################################################
def SortMasks(DetectDir):
    masks = []
    masklist = os.listdir(DetectDir)
    sortlist = list(range(len(masklist)))
    for i in range(len(sortlist)):
        for j in range(len(masklist)):
            if int(masklist[j].split('_')[-1].split('.')[0]) == sortlist[i]:
                masks.append(masklist[j])
            else:
                pass
    return masks
##########################################################################################################################################
# sorting list of Maximas
##########################################################################################################################################
def SortMaxima(dists, coords):
    Dists = dists.copy()
    Coords=[]
    dists.sort(reverse=True)
    for i in dists:
        index=Dists.index(i)
        Coords.append(coords[index])
    return dists, Coords
##########################################################################################################################################
# finding contour
##########################################################################################################################################
def ContourFinder(image):
    '''
    Condition: Image must be binary, not specifically an array of 0 and 1, 
    but an array with just two different values for back- and foreground.
    '''
    
    y = image.shape[0]
    x = image.shape[1]
    contour = np.zeros((y,x))
    
    for i in range(y):
        for j in range(x):
            if i == 0 or i== y-1:
                contour[i,j]=image[i,j]
            elif j == 0 or j == x-1:
                contour[i,j]=image[i,j]
            else:
                try:
                    if image[i,j-1] == 0 or image[i,j+1] == 0 or image[i-1,j] == 0 or image[i+1,j]==0:
                        contour[i,j]=image[i,j]
                except IndexError:
                    continue
    contour = np.where(contour == 0, contour, 150)
    
    return contour
##########################################################################################################################################
# find the farthest two points; create the vectors to the two points from center; calculate the angle between them
##########################################################################################################################################

def calcDistances(contour, center, radius, Name):
    '''
    iterating over one half of image from top to bottom and the other half from bottom to top
    Problem: pixels in row 1 are nearer to other half than pixels in row0--> zigzagging
    Coordinates = []
    y = contour.shape[0]
    x = contour.shape[1]
    for i in range(y):
        for j in range(x):
            if contour[i,j]!=0:
                Coordinates.append([i,j])
    New_Coordinates=[]
    Distances=[]
    for i in Coordinates:
        if i[1]>=int(x/2):
            #print(i)
            New_Coordinates.append(i)
            distance=math.sqrt((i[0]-center[0])**2+(i[1]-center[1])**2)
            print(distance)
            Distances.append(distance/radius)
    #print('##################second half############################')
    for i in Coordinates[::-1]:
        if i[1]<int(x/2):
     #       print(i)
            New_Coordinates.append(i)
            distance=math.sqrt((i[0]-center[0])**2+(i[1]-center[1])**2)
            print(distance)
            Distances.append(distance/radius)
    Coordinates=New_Coordinates
    Distances=signal.savgol_filter(Distances, 13,4)
    
    PlaroCoords:
        Problem if theta has more than one radius-argument-->zigzagging
    Coordinates=[]
    NormCoords=[]
    y = contour.shape[0]
    x = contour.shape[1]
    for i in range(y):
        for j in range(x):
            if contour[i,j] != 0:
                Coordinates.append([i,j])
                NormCoords.append([i-center[0],j-center[1]])
    Set = []
    for coordinate in NormCoords:
        r = math.sqrt(coordinate[0]**2+coordinate[1]**2)
        t = math.atan2(coordinate[0], coordinate[1])
        Set.append([t,r])
        
    SetCopy=Set.copy()
    Set.sort()
    Coordinates2=[]
    Distances=[]
    for i in Set:
        index = SetCopy.index(i)
        Coordinates2.append(Coordinates[index])
    for i in Set:
        Distances.append(i[1]/radius)
    Distances=signal.savgol_filter(Distances, 19, 4)

    Moore Neighborhood Tracing: http://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/moore.html
    '''
    print(f'Processing contour of {Name}')
    Coordinates=[] 
    neighbours=[[0,-1], [-1,-1], [-1,0], [-1,+1], 
               [0,+1], [+1,+1], [+1,0], [+1,-1]]
    while len(Coordinates)==0:
        for i in range(len(contour[-1:][0])):
            if contour[-1:,i]!=0:
                Coordinates.append([contour.shape[0]-1,i])
                s=[contour.shape[0]-1,i]
                p=s
                break
            elif i==contour.shape[1]-1:
                print('No Pixel different from Zero found in the last row')
                print(f'Reducing shape of {Name} until pixel different from 0 is hit')
                contour=contour[:-1,:]
                break
    #Backtrack:
    c=[p[0],p[1]-1]
    #Debug for: If first encountered pixels neighbourhood is only background
    M=[[p[0]-n[0],p[1]-n[1]] for n in neighbours]
    M=M+M
    start=M.index(c)
    NeighVals=[]
    for c in M[start:len(neighbours)+start]:
        try:
            NeighVals.append(contour[c[0],c[1]])
        except IndexError:
            continue
    if np.sum(NeighVals)==0:
        raise ValueError(f'Neighbourhood consists only of background pixels, check mask: {Name}')
    repeats=0
    while c!=s:
        repeats+=1
        M=[[p[0]-n[0],p[1]-n[1]] for n in neighbours]
        M=M+M
        start=M.index(c)
        for c in M[start:len(neighbours)+start]:
            try:    
                if contour[c[0],c[1]]!=0 and c not in Coordinates and c[1]>=0 and c[0]>=0:
                    #print(c,s)
                    contour[c[0],c[1]]=100
                    Coordinates.append(c)
                    #Backtrack
                    index=M.index(c)-1
                    backtrack=[p[0]-M[index][0],p[1]-M[index][1]]
                    p=c.copy()
                    c=[p[0]-backtrack[0],p[1]-backtrack[1]]
                    break
            except IndexError:
                continue
        if repeats==contour.size:
            print('Something is fishy with the contour, got stuck in while-loop, will return empty list for Distances')
            Distances=[]
            return Coordinates, Distances
    #Idea for reducing problem with the borders of this list:
    #Adding a value of 1 to the beginning and the end of list for Distances
    #and adding [0,0] coordinated accordingly to keep index-accessibility
    Distances=[]
    for i in Coordinates:
        distance=math.sqrt((i[0]-center[0])**2+(i[1]-center[1])**2)
        Distances.append(distance/radius)
    Coordinates=Coordinates[-25:]+Coordinates[:-25]
    Distances=Distances[-25:]+Distances[:-25]
    try:
        Distances=signal.savgol_filter(Distances, 21,3)
    except ValueError:
        raise ValueError(f'Check mask: {Name}')
    
    
        
    return Coordinates, Distances

##########################################################################################################################################
###Analyze Data
##########################################################################################################################################
def Analyse(DetectDir, Mask_DIR):
    start = time.time()
    VIC = ['bud_on_shmootip', 'budding_shmoo'] #Very important Classes ;)
    IMGs_DIR = '/home/basar/Personal/Erich/Studiproject/datasets/yeast/dataset_detect/'
    csv = 'detected.csv'
    if os.path.exists(os.path.join(DetectDir, csv)):
        df = pd.read_csv(os.path.join(DetectDir, csv))
    else:
        raise FileNotFoundError(f'Check your DetectDir {DetectDir} and in case the name of the csv-file {csv} and try again')
    masks = []
    for i in range(len(df)):
        if MASK_DIR == 'Masks':
            masks.append(f'Sub_ID_{df["CellID"][i]}.{df["Image"][i]}.result.{df["Class"][i]}.tif')
        elif MASK_DIR == 'Classified':
            masks.append(f'Sub_ID_{df["CellID"][i]}.{df["Image"][i]}.result.{df["Class"][i]}.tif')
    SAVE_DIR = os.path.join(DetectDir, 'Analysis')
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
       
    count = 0
    Time, PhysicalSizeX, PhysicalSizeY, Unit=ExtraData(df, IMGs_DIR) 
    angles = [] #for CellInfoCSV-F, contains np.nans to fulfill rule of length
    location_max1 = [] #for CellInfoCSV-F, contains np.nans to fulfill rule of length
    location_max2 = [] #for CellInfoCSV-F, contains np.nans to fulfill rule of length
    distance_max1 = [] #for CellInfoCSV-F, contains np.nans to fulfill rule of length
    distance_max2 = [] #for CellInfoCSV-F, contains np.nans to fulfill rule of length
    centers = [] #for CellInfoCSV-F, contains np.nans to fulfill rule of length
    Dists2 = [] #for CellInfoCSV-F, contains np.nans to fulfill rule of length
    CellPerimeter=[] #for CellInfoCSV-F, contains np.nans to fulfill rule of length
    Fishy=[] #for CellInfoCSV-F, contains np.nans to fulfill rule of length
    CirclePerimeter=[] #debug
    AreaRatiosCellCircle=[] #debug AreaRatio between Circle and whole cell
    AreaRatiosCellImage=[] #debug
    AspectRatio=[] #debug
    masknames=[] #for ContourPlot (PDF and CSV-F)
    Dists = [] #for ContourPlot (PDF and CSV-F)
    Contours = [] #for ContourPlot (PDF and CSV-F)
    Heights = [] #for ContourPlot (PDF and CSV-F)
    Distances = [] #for ContourPlot (PDF and CSV-F)
    radius = [] #for ContourPlot (PDF and CSV-F)
    
    for maskname in masks[:]:
        if maskname.split('.')[-2] not in VIC:
            angles.append(np.nan)
            location_max1.append(np.nan)
            location_max2.append(np.nan)
            distance_max1.append(np.nan)
            distance_max2.append(np.nan)
            Dists2.append(np.nan)
            Fishy.append(np.nan)
            
            #mask = imread(os.path.join(DetectDir+'Masks', maskname)) #for Classification with Mask R-CNN
            mask = imread(os.path.join(DetectDir+MASK_DIR, maskname)) #for seperate classifier
            cnt = ContourFinder(mask)
            cellperimeter=np.sum(cnt/150)
            CellPerimeter.append(cellperimeter)
            
            cellimgratio=np.sum(mask/255)/mask.size
            aspectratio=max(mask.shape)/min(mask.shape)
            #hough transform to calculate a mean distance = radii
            if aspectratio >=1.5:
                hough_radii = np.arange(int(min(mask.shape)/2.45), math.ceil(min(mask.shape)/2), 1) 
                if aspectratio >= 1.8:
                    hough_radii = np.arange(int(min(mask.shape)/2.25), math.ceil(min(mask.shape)/2), 1)
            elif 1.0<aspectratio<1.1:
                if cellimgratio<0.65:
                    hough_radii = np.arange(int(min(mask.shape)/3), math.ceil(min(mask.shape)/2), 1) 
                else:
                    hough_radii = np.arange(int(min(mask.shape)/2.5),  math.ceil(max(mask.shape)/2)+1, 1) 
            else:
                if cellimgratio<0.68:
                    hough_radii = np.arange(int(min(mask.shape)/2.9), math.ceil(min(mask.shape)/2), 1) 
                    if cellperimeter>=220:
                        hough_radii = np.arange(int(min(mask.shape)/2.5), math.ceil(min(mask.shape)/2), 1) 
                elif cellperimeter>=160.0:
                    hough_radii = np.arange(int(min(mask.shape)/2.7),  math.ceil(max(mask.shape)/2), 1) 
                
                else:
                    hough_radii = np.arange(int(min(mask.shape)/2.5),  math.ceil(max(mask.shape)/2), 1) 
            houghSpace = hough_circle(cnt, hough_radii)
            
            accums, hough_x, hough_y, radii = hough_circle_peaks(houghSpace, hough_radii,
                                                                 total_num_peaks=1)
            center = [int(hough_y), int(hough_x)]
            centers.append(center)
            
        elif maskname.split('.')[-2] in VIC:
            masknames.append(maskname)
            print(count)
            mask = imread(os.path.join(DetectDir+MASK_DIR, maskname))
#################################################################################################################################################
###Create Thresholds for the range of hough_radii
#################################################################################################################################################    
            cnt = ContourFinder(mask)
            cellperimeter=np.sum(cnt/150)
            CellPerimeter.append(cellperimeter)
            cellimgratio=np.sum(mask/255)/mask.size
            AreaRatiosCellImage.append(cellimgratio)
            aspectratio=max(mask.shape)/min(mask.shape)
            AspectRatio.append(aspectratio)
            #hough transform to calculate a mean distance = radii
            if aspectratio >=1.5:
                hough_radii = np.arange(int(min(mask.shape)/2.45), math.ceil(min(mask.shape)/2), 1) 
                if aspectratio >= 1.8:
                    hough_radii = np.arange(int(min(mask.shape)/2.25), math.ceil(min(mask.shape)/2), 1)
            elif 1.0<aspectratio<1.1:
                if cellimgratio<0.65:
                    hough_radii = np.arange(int(min(mask.shape)/3), math.ceil(min(mask.shape)/2), 1) 
                else:
                    hough_radii = np.arange(int(min(mask.shape)/2.5),  math.ceil(max(mask.shape)/2)+1, 1) 
            else:
                if cellimgratio<0.68:
                    hough_radii = np.arange(int(min(mask.shape)/2.9), math.ceil(min(mask.shape)/2), 1) 
                    if cellperimeter>=220:
                        hough_radii = np.arange(int(min(mask.shape)/2.5), math.ceil(min(mask.shape)/2), 1) 
                elif cellperimeter>=160.0:
                    hough_radii = np.arange(int(min(mask.shape)/2.7),  math.ceil(max(mask.shape)/2), 1) 
                
                else:
                    hough_radii = np.arange(int(min(mask.shape)/2.5),  math.ceil(max(mask.shape)/2), 1) 
#################################################################################################################################################
###End of threshold for hough_radii
#################################################################################################################################################    
            houghSpace = hough_circle(cnt, hough_radii)
            
            accums, hough_x, hough_y, radii = hough_circle_peaks(houghSpace, hough_radii,
                                                                 total_num_peaks=1)
            cellcircleratio=np.sum(mask/255)/(math.pi*radii[0]**2)
            AreaRatiosCellCircle.append(cellcircleratio)
            circleperimeter=2*math.pi*radii[0]
            CirclePerimeter.append(circleperimeter)
            
            center = [int(hough_y), int(hough_x)]
            centers.append(center)
            radius = radius+radii.tolist()
            cnt = ContourFinder(mask)
            coords, dists = calcDistances(cnt, center, radii[0], maskname)
            cnt[int(hough_y), int(hough_x)]=255
            cv2.circle(cnt, (int(hough_x), int(hough_y)), radii, 255, 1)
        
#################################################################################################################################################
###Create Thresholds for detecting peaks
#################################################################################################################################################    
#Create Autothreshold if height falls below certain value
#Make it simple as possible, try to get perfect peaks from as many cells as possible instead of giving each cell a parameter
            if maskname.split('.')[-2]==VIC[0] and len(dists)!=0:
                height=max(dists)*0.8
                if height<1.1:
                    height=1.1
                    if max(dists)<height:
                        height=max(dists)
                distance=20
                Heights.append(height)
                Distances.append(distance)
            elif maskname.split('.')[-2]==VIC[1] and len(dists)!=0:
                #height=max(dists)*0.4
                #if height<1.1:
                #    height=1.1
                #To detect eben the smallest hint of a shmootip
                height=1.1 #Make higher threshold... 1.2 or so
                if max(dists)<height:
                        height=max(dists)
                distance=15
                Heights.append(height)
                Distances.append(distance)
            else:
                height=1.1
                distance=15
                Heights.append(height)
                Distances.append(distance)
        
        
#################################################################################################################################################
###End of threshold for peaks
#################################################################################################################################################            
    
            y=signal.find_peaks(dists, height=height, distance=distance)
            
            if len(y[0])==0 and len(dists)!=0:
                print('No peaks detected, switching focus of Distance-list')
                dists=np.array(dists.tolist()[-5:]+dists.tolist()[:-5])
                coords=coords[-5:]+coords[:-5]
            
            Dists.append(dists)
            y=signal.find_peaks(dists, height=height, distance=distance)
            
            if len(y[0])==0 and len(dists)==0:
                print('Empty Distance-list, so fishy mask->set maxima to [0,0]')
                max1=[0,0]
                max2=[0,0]
                max1dist=0
                max2dist=0
                location_max1.append(max1)
                distance_max1.append(max1dist)
                location_max2.append(max2)
                distance_max2.append(max2dist)
                angles.append(np.nan)
                Fishy.append('This one is fishy')
            
            else:
                MaxIDs = y[0].tolist()
                MaxVals = y[1]['peak_heights'].tolist()
                
                index=MaxVals.index(max(MaxVals))
                index=MaxIDs[index]
                max1=coords[index]
                max1dist=max(MaxVals)
                if max1dist<1.2:
                    print(f'Maximum distance of {max1dist} is too small to count as either bud- or shmootip')
                    max1dist=0
                    max2dist=max1dist
                    max1=[0,0]
                    max2=max1
                    location_max1.append(max1)
                    distance_max1.append(max1dist)
                    location_max2.append(max2)
                    distance_max2.append(max2dist)
                    angles.append(np.nan)
                    Fishy.append('Max1 too small')
    
                else:
                    location_max1.append(max1)
                    distance_max1.append(max1dist)
                    MaxIDs.remove(index)
                    MaxVals.remove(max(MaxVals))
                    Fishy.append(np.nan)
                    try:
                        index=MaxVals.index(max(MaxVals))
                        index=MaxIDs[index]
                        max2=coords[index]
                        max2dist=max(MaxVals)
                        MaxIDs.remove(index)
                        MaxVals.remove(max(MaxVals))
                        #if the maxima-coordinates too close -> maxima2=maxima1 -> angle=0
                        if np.allclose(max1, max2, atol=distance) and maskname.split('.')[-2]==VIC[0]: 
                            max2=max1
                            max2dist=max1dist
                        location_max2.append(max2)
                        distance_max2.append(max2dist)
                    except ValueError:
                        max2=max1
                        max2dist=max1dist
                        location_max2.append(max2)
                        distance_max2.append(max2dist)
            
                            
            
                
                    vec1 = np.array(max1) - np.array(center)
                    vec2 = np.array(max2) - np.array(center)
                    cosang = np.dot(vec1, vec2)
                    sinang = la.norm(np.cross(vec1, vec2))
                    angle = np.degrees(np.arctan2(sinang, cosang))
                    angles.append(angle)
                    cnt[max1[0], max1[1]]=255
                    cnt[max2[0], max2[1]]=255
            Contours.append(cnt)
            count+=1
                
    print(f'Analyzing {len(df)} cells took {time.time()-start} sec')
                
#################################################################################################################################################
###Saving everything eiter as PDF or CSV-F
#################################################################################################################################################    
     
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(SAVE_DIR, 'Plots.pdf'))
    figs = plt.figure(figsize=(10,15))
    plot_num1=321
    count=0
    fontsize=16
    print('Generating the PDFs and CSV-Files...')
    
    start=time.time()
    for i in range(len(Dists)):
        if len(Dists[i])==0:
            pass
        else:
            plt.subplot(plot_num1)
            plt.plot(Dists[i])
            maskname=masknames[i]
            #name=maskname.split(".")[-2] #name of shape
            cellID = maskname.split(".")[0] #ID-No of cell
            #imgID = maskname.split(".")[1] #Original name of bf-image
            #plt.title(f'Forbidden neighborhood:{Distances[i]}; Cell: {cellID}', fontsize=fontsize)
            peaks,_=signal.find_peaks(Dists[i], height=Heights[i], distance=Distances[i])
            plt.plot(peaks, Dists[i][peaks], 'x', markersize=10, color='black')
            plt.plot(np.zeros_like(Dists[i])+Heights[i], '--', color='red') #Height-theshold
            plt.xticks(ticks=(0, int(len(Dists[i])/2), len(Dists[i])), fontsize=fontsize)
            plt.xlabel('Pixel of Contour', fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.ylabel('Relative distance', fontsize=fontsize)
            plot_num1+=1
            count+=1
            plt.subplot(plot_num1)
            plt.imshow(Contours[i], cmap='gray_r')
            plt.title(f'Hough-circle radius: {radius[i]}, {cellID}', fontsize=fontsize)
            #plt.xticks(ticks=[''], labels=[''], fontsize=fontsize)
            #plt.yticks(ticks=[''], labels=[''], fontsize=fontsize)
            plt.axis('off')
            if count!=5:
                count+=1
                plot_num1+=1
            elif count==5:
                count=0
                plot_num1=321
                pdf.savefig(figs)
                figs = plt.figure(figsize=(10,15))
            elif Dists[i]==Dists[-1]:
                pdf.savefig(figs)
    
    pdf.close()
    plt.close('all')
    
    Keys = ['Angles', 'LocationOfMaxima1', 'LocationOfMaxima2', 'LocationOfCenter', 'DistanceToMaxima1', 
            'DistanceToMaxima2', 'CellPerimeter', 'DateOfImage', 'Fishy', 'RealSizeX', 'RealSizeY', 'Unit']
    Values = [angles, location_max1, location_max2, centers, distance_max1, distance_max2, CellPerimeter, Time, Fishy,
              PhysicalSizeX, PhysicalSizeY, Unit]
    
    for key, val in zip(Keys, Values):
        df[key]=val
    df.to_csv(os.path.join(SAVE_DIR, 'CellInfos.csv'), sep=',', index=False, float_format='%g')
    df1 = pd.DataFrame(Dists,index=masknames)
    df1.to_csv(os.path.join(SAVE_DIR, 'ContourPlotData.csv'), float_format='%g')    
    print(f'Generating Files took {time.time()-start} sec')
    print(f'{len(df1)} important Cells in total were analyzed within {len(os.listdir("/home/basar/Personal/Erich/Studiproject/datasets/yeast/dataset_detect"))} images')

##########################################################################################################################################
###Doing Stuff from here
##########################################################################################################################################

#Directory of the image detection was ran on:
IMGs_DIR = '/home/basar/Personal/Erich/Studiproject/datasets/yeast/dataset_detect/'

DetectDir = os.path.join('/home/erich/Dokumente/1ClassDetectionAngle/',  #Change only dir in this row
                         'Detection/')
#Choose one of these two Names:
MASK_DIR = 'Masks'

Analyse(DetectDir, MASK_DIR)


            








































'''
#df = pd.DataFrame({'CellCircle':AreaRatiosCellCircle, 'CellImage':AreaRatiosCellImage, 'Angles':angles, 'Aspect':AspectRatio, 
#                   'CePeri':CellPerimeter, 'CiPe':CirclePerimeter, 'max1dist':distance_max1, 'max2dist':distance_max2})
#df = pd.DataFrame({'Angles':angles, 'LocationOfMaxima1':location_max1, 'LocationOfMaxima2': location_max2, 
#                   'LocationOfCenter': centers, 'DistanceToMaxima1':distance_max1, 'DistanceToMaxima2':distance_max2})
#df.to_csv(os.path.join(SAVE_DIR, 'efd_Cells.csv'), index=False, float_format='%g')

        MaxIDs=signal.argrelmax(np.array(dists), order=2)[0].tolist() #returns list of indices for local maxima in dists4
        MaxDists=[] 
        MaxCoords=[]
        for i in range(len(dists)):
            if i in MaxIDs:
                MaxDists.append(dists[i])
                MaxCoords.append(coords[i])
            
        MaxDists, MaxCoords = SortMaxima(MaxDists, MaxCoords) #sorts MaxDists from max->min and updates MaxCoords respectivley
        index=MaxDists.index(max(MaxDists))
        max1dist=MaxDists.pop(index)
        distance_max1.append(max1dist)
        max1=MaxCoords.pop(index)
        location_max1.append(max1)
#################################################################################################################################################
###Create thresholds for nearest neighbour (atol) and minimum distance from center(threshold_dist)
#################################################################################################################################################    
        #imgs: either finetune til every single image has its own category or search for other sort-pars....
        
        atol=20
        
        threshold_dist=1.2
        if max1dist>=3.0:
            atol=max(mask.shape)
            if cellimgratio>=0.6:
                atol=math.ceil(max(mask.shape)/2)
                threshold_dist=1.5
        elif aspectratio>=1.5:
            print('problematic:50,')
            atol=25
            threshold_dist=1.5
            
        
        
        
        
        
        threshold_dist=1.1 #'Work on that further tomorrow
        
        
        
        max2Found=False
        while not max2Found:
            #print('Dists', len(MaxDists), 'Coords', len(MaxCoords))
            for dist, coord in zip(MaxDists, MaxCoords):
                #print(f'Distance: {dist}, Coordinates:{coord}')
                if not np.allclose(max1, coord, atol=atol):
                    if dist>threshold_dist:
                        print('Found max2')
                        max2=coord
                        max2Found=True
                        break
                    elif dist==MaxDists[-1]:
                        print(f'Distance {dist} is not far enough and {coord} is last possible')
                        print('No max2 Found')
                        max2=max1
                        max2Found=True
                        break
                elif np.allclose(max1, coord, atol=atol):
                    print('Maxima too close')
                    if dist==MaxDists[-1]:
                        print('No max2 Found')
                        max2=max1
                        max2Found=True
                        break
        '''

    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


