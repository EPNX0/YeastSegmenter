"""
Created on Tue Oct  8 12:24:10 2019

@author: erich
"""

'''
Usage: Run from the command line as such:

# Train a new model starting from ImageNet weights or coco weights
python3 /home/basar/Personal/Erich/Mask_RCNN/samples/yeast/yeast.py train --dataset=/path/to/datasets --weights=<imagenet or coco> <--logs=>

# Train a new model starting from specific weights file
python3 /home/basar/Personal/Erich/Mask_RCNN/samples/yeast/yeast.py train --dataset=/path/to/datasets --weights=/path/to/weights.h5 <--logs=>

# Resume training a model that you had trained last
python3 /home/basar/Personal/Erich/Mask_RCNN/samples/yeast/yeast.py train --dataset=/path/to/datasets --weights=last <--logs=>

# Validate your trained model
python3 /home/basar/Personal/Erich/Mask_RCNN/samples/yeast/yeast.py evaluate --dataset=/path/to/datasets --weights=<last or /path/to/weights.h5> <--savedir=>
    
# Run detection with your trained model
python3 /home/basar/Personal/Erich/Mask_RCNN/samples/yeast/yeast.py detect --dataset=/path/to/datasets --weights=<last or /path/to/weights.h5> <--logs=> <--savedir=>
'''

# Prevents matplotlib from displaying anything but still be able to save plots
import matplotlib
matplotlib.use('Agg')
import os, sys, pandas as pd, numpy as np
sys.path.append('/home/basar/Personal/Erich/site-packages')  #for local purposes
sys.path.remove('/usr/lib/python3/dist-packages')  #for local purposes


import imageio
import random, time #new
from imgaug import augmenters as iaa
# Root directory to the Mask_RCNN Folder:
Mask_RCNN_DIR = '/home/basar/Personal/Erich/Yeastcell_segmenter/'

# Directory to save all data generated:
DEFAULT_SAVE_DIR = '/home/basar/Personal/Erich/Yeastcell_segmenter/SAVEDIR/'


# Directory to save logs and the trained model:
DEFAULT_MODEL_DIR = '/home/basar/Personal/Erich/Yeastcell_segmenter/logs/yeast/'



sys.path.append(Mask_RCNN_DIR)
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import operations
from mrcnn import visualize
from mrcnn import sort_gt_data
from mrcnn import teaching
import mrcnn.model as modellib  # minor changes done by me(erich) (signed with #new)


##########################################################################################################################################
### Training Configuration
##########################################################################################################################################
class YeastTrainConfig(Config):
    # Name of the configuration
    NAME = 'yeast'
    
    # Batch size is GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 # Don't overdo it!
    
    # Backbone network architecture
    BACKBONE = 'resnet101'
    
    # Number of classes including Background BG
    # Three additional classes: yeast, budding-yeast, shmooing-yeast
    NUM_CLASSES = 2 #Class for Background and your added classes
    
    # non-max suppression threshold to filter RPN training; 
    # increase during training
    RPN_NMS_THRESHOLD = 0.7
    
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200#500
    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 25 #No. of Images in Val_Set
    
    # Resize instance masks to a smaller size to reduce memory load;
    # recommended for high-resolution images
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (28, 28) # (height, width) of the mini-masks
    
    IMAGE_RESIZE_MODE = 'square' # should work well in most cases
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    
    # Number of channels per image. RGB=3, grayscale=1
    IMAGE_CHANNEL_COUNT = 3
    
    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200
    
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 200
    
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.9 #0.8
    
    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.2 #0.3
    
    # Learning rate and momentum
    LEARNING_RATE = 0.01 #0.01
    LEARNING_MOMENTUM = 0.8 #0.9
    
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    
    # Train or freeze batch normalization layers
    TRAIN_BN = False # False, because batch size is small
    
    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0
    
    # New Variables for SGD
    EPSILON = 0.0 #1e-9
    DECAY = 1e-3
    NESTEROV = True
    
TrainConfig = YeastTrainConfig()
TrainConfig.display()

##########################################################################################################################################
### Validation/Detection Configuration
##########################################################################################################################################
class YeastInferenceConfig(YeastTrainConfig):
    # Batch size is GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    IMAGE_RESIZE_MODE = 'square' # should work well in most cases
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
##########################################################################################################################################
### Dataset
##########################################################################################################################################
class YeastDataset(utils.Dataset):
    def load_yeast(self, datasetdir, subset):
        '''
        datasetdir: Root directory to the datasets.
        subset: subset to use. Either 'train', 'val'
                or 'detect'.
        '''
        
        # Add Classes to the Dataset, source is yeast
        self.add_class('yeast', 1, 'yeastcell')
        #self.add_class('yeast', 1, 'bud_on_shmootip')
        #self.add_class('yeast', 2, 'budding_shmoo')
        #self.add_class('yeast', 3, 'inactive') 
        #self.add_class('yeast', 4, 'budding')
        #self.add_class('yeast', 5, 'shmoo')
        
        assert subset in ['train', 'val', 'detect', 'test']
        SUBSET_DIR = 'dataset_'+ subset
        DATASET_DIR = os.path.join(datasetdir, SUBSET_DIR)
        if subset == 'train':
            image_ids = next(os.walk(DATASET_DIR))[1]
        elif subset == 'val':
            image_ids = next(os.walk(DATASET_DIR))[1]
        elif subset == 'detect':
            image_ids = next(os.walk(DATASET_DIR))[-1]
        
        # Add images
        if subset == 'train':
            image_ids=next(os.walk(DATASET_DIR))[1]
            #config.STEPS_PER_EPOCH = int(len(image_ids)*4)
            sets=[]
            images=[]
            paths=[]
            for image_id in image_ids:
                sets.append(os.path.join(DATASET_DIR, image_id))
            for Set in sets:
                images.append(next(os.walk(Set+'/image'))[-1][-1])
                        
            for Set,image in zip(sets,images):
                paths.append(os.path.join(Set+'/image',image))
            for image,path in zip(images,paths):
                self.add_image('yeast',
                               image_id=image,
                               path=path)
                
        if subset == 'val':
            image_ids=next(os.walk(DATASET_DIR))[1]
            sets=[]
            images=[]
            paths=[]
            for image_id in image_ids:
                sets.append(os.path.join(DATASET_DIR, image_id))
            for Set in sets:
                images.append(next(os.walk(Set+'/image'))[-1][-1])
                        
            for Set,image in zip(sets,images):
                paths.append(os.path.join(Set+'/image',image))
            for image,path in zip(images,paths):
                self.add_image('yeast',
                               image_id=image,
                               path=path)
                
        if subset == 'test':
            image_ids=next(os.walk(DATASET_DIR))[1]
            sets=[]
            images=[]
            paths=[]
            for image_id in image_ids:
                sets.append(os.path.join(DATASET_DIR, image_id))
            for Set in sets:
                images.append(next(os.walk(Set+'/image'))[-1][-1])
                        
            for Set,image in zip(sets,images):
                paths.append(os.path.join(Set+'/image',image))
            for image,path in zip(images,paths):
                self.add_image('yeast',
                               image_id=image,
                               path=path)

        if subset == 'detect':
            for image_id in image_ids:
                self.add_image('yeast',
                               image_id=image_id,
                               path=os.path.join(DATASET_DIR, image_id))
            
        
    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        info=self.image_info[image_id]
        MASK_DIR = os.path.join(os.path.dirname(os.path.dirname(info['path'])), 'masks')
        masks=[]
        class_ids=[] #for multiple classes
        for f in os.listdir(MASK_DIR):
            if f.endswith(".tif"):
                if config.NUM_CLASSES == 3 and int(f.split('.')[-2]) == 1 or int(f.split('.')[-2]) ==2:
                    m = imageio.imread(os.path.join(MASK_DIR, f)).astype(np.bool)
                    masks.append(m)
                elif config.NUM_CLASSES == 6 or config.NUM_CLASSES == 2:
                    m = imageio.imread(os.path.join(MASK_DIR, f)).astype(np.bool)
                    masks.append(m)
                class_ids.append(int(f.split('.')[-2])) #for multiple classes
        if config.NUM_CLASSES == 2:
            if len(masks)!=0:
                mask=np.stack(masks, axis=-1)
                class_ids = np.ones([mask.shape[-1]], dtype=np.int32) #for single-class
            else:
                mask=np.zeros((1024,1024,0))
                class_ids = np.array([], dtype=np.int32)
        elif config.NUM_CLASSES == 6:
            if len(masks)!=0:
                mask=np.stack(masks, axis=-1)
                class_ids = np.array(class_ids, dtype=np.int32) # for five-class
            else:
                mask=np.zeros((1024,1024,0))
                class_ids = np.array([], dtype=np.int32)
        elif config.NUM_CLASSES == 3:
            if len(masks)!=0:
                mask=np.stack(masks, axis=-1)
                class_ids = [Class for Class in class_ids if Class==1 or Class==2] #for two-class
                class_ids = np.array(class_ids, dtype=np.int32)
            else:
                mask=np.zeros((1024,1024,0))
                class_ids = np.array([], dtype=np.int32)
            
        
        return mask, class_ids
        
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source']=='yeast':
            return info['id']
        else:
            super(self.__class__, self).image_reference(image_id)

##########################################################################################################################################
### Command Line
##########################################################################################################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for yeast segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'validate', 'detect'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/datasets/",
                        help='Root directory to the datasets')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Either: Path to weights .h5 file, 'coco', 'last' or 'imagenet'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_MODEL_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--savedir', required=False,
                        default=DEFAULT_SAVE_DIR,
                        metavar='path/to/savedir',
                        help='Directory to save produced data')
    args = parser.parse_args()
    if not os.path.exists(args.logs):
        os.makedirs(args.logs)
    #if not os.path.exists(args.savedir):
     #   os.makedirs(args.savedir)

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print('SAVE_DIR: ', args.savedir)

    assert args.command in ['train', 'teach', 'detect', 'evaluate']
    
##########################################################################################################################################
### Training
##########################################################################################################################################
'''
Added another modellib that contains Adam-optimizer. 
Needs to add another variable in the model.train-function.
New variable = optim=
'''
if args.command == 'train':
    random.seed(1)
    config = YeastTrainConfig()
    model = modellib.MaskRCNN(mode="training", config=config,
                        model_dir=args.logs)
    config.display()
    #Limiting GPU-Usage:
    #config2 = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config2.gpu_options.per_process_gpu_memory_fraction=0.9
    #sess = tf.Session(config=config2)
    
    # Load weights
    if args.weights == 'coco':
        COCO_DIR = os.path.join(Mask_RCNN_DIR, 'mask_rcnn_coco.h5')
        if not os.path.exists(COCO_DIR):
            utils.download_trained_weights(COCO_DIR)
        print('Loading pre-trained coco-weights:')
        model.load_weights(COCO_DIR, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    elif args.weights == 'imagenet':
        print('Loading pre-trained imagenet-weights:')
        model.load_weights(filepath = model.get_imagenet_weights(), by_name=True)
    elif args.weights == 'last':
        print('Loading your last trained model:')
        model.load_weights(filepath = model.find_last(), by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mashttps://imgaug.readthedocs.io/en/latest/source/overview_of_augmenters.htmlk"])
    else:
        print('Loading the model you wanted:')
        model.load_weights(filepath = args.weights, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    # Training Dataset
    dataset_train = YeastDataset()
    dataset_train.load_yeast(datasetdir=args.dataset, subset='train')
    dataset_train.prepare()

    # Validation Dataset
    dataset_val = YeastDataset()
    dataset_val.load_yeast(datasetdir=args.dataset, subset='val')
    dataset_val.prepare()
    
        
    aug = iaa.SomeOf((2), [iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                #iaa.JpegCompression(compression=(70,99)),
                #iaa.MedianBlur(k=(3,11)),
                #iaa.GammaContrast(gamma=(0.5,2.5)),
                iaa.SomeOf((3), [iaa.Affine(scale={'x':(0.5,2.5), 'y':(0.5,2.5)}),
                iaa.CropAndPad(percent=(-0.3, 0.3)),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270)])
                      ])
            
    
    class_weight=None
    
    print('Train head layers')
    model.train(dataset_train, dataset_val, learning_rate = config.LEARNING_RATE,
                epochs = 20, layers = 'heads', augmentation =  aug, class_weight=class_weight
                )
    
    print('Train stages 3 and up')
    model.train(dataset_train, dataset_val, learning_rate = config.LEARNING_RATE,
                epochs = 50, layers = '3+', augmentation =  aug, class_weight=class_weight
                )  
    print('Fine-tune all layers')
    config.RPN_NMS_THRESHOLD = 0.8
    model.train(dataset_train, dataset_val, learning_rate = config.LEARNING_RATE/100,
                epochs=160, layers = 'all', augmentation = aug, class_weight=class_weight
                )

    
    
    
##########################################################################################################################################
### Evaluation
##########################################################################################################################################
if args.command == 'evaluate':
    
    config = YeastInferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config,
                             model_dir=args.logs)
    config.display()
    
    # Load weights
    if args.weights == 'last':
        print('Loading your last trained model:')
        model.load_weights(filepath = model.find_last(), by_name=True)
    else:
        print('Loading the model you wanted:')
        model.load_weights(filepath = args.weights, by_name=True)
    
    
    
    # Training Dataset
    dataset_train = YeastDataset()
    dataset_train.load_yeast(datasetdir=args.dataset, subset='train')
    dataset_train.prepare()

    # Validation Dataset
    dataset_eval = YeastDataset()
    dataset_eval.load_yeast(datasetdir=args.dataset, subset='test')
    dataset_eval.prepare()
    
    DATA_SAVE_DIR = os.path.join(args.savedir, 'Eval_on_test')
    if not os.path.exists(DATA_SAVE_DIR):
        os.makedirs(DATA_SAVE_DIR)
    atol=20 #15
    IoUs = np.arange(0.5,1,0.05)
    lis=[]
    mAPs = []
    F1 = []
    sub=0
    captions = list(range(100000))
    image_ids = dataset_eval.image_ids
    for image_id in image_ids:
        begin=len(lis)
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_eval, YeastInferenceConfig(), 
                                   image_id, use_mini_mask=False)
        name = list(dataset_eval.image_info[image_id].values())[0]
        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                    dataset_eval.class_names, SAVE_DIR=os.path.join(DATA_SAVE_DIR, name))
   
##########################################################################################################################################
### Detect on random image of random Images
##########################################################################################################################################
    
        results = model.detect([original_image], verbose=1)

        r = results[0]

        operations.display_instances(image_id, original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset_eval.class_names, DATA_SAVE_DIR, name,
                                scores=r['scores'], captions=captions[begin:begin+r['rois'].shape[0]], 
                                figsize=(16,16), show_bbox=True, show_mask=True)
##########################################################################################################################################
### Sort data
##########################################################################################################################################
        gt_class_ids, class_ids, gt_bboxs, rois = sort_gt_data.sort_data(gt_class_id, r['class_ids'], gt_bbox, r['rois'],
                                                                         rtol=0, atol=atol)
    
##########################################################################################################################################
### Convert data into .csv file
##########################################################################################################################################
    
        for i in range(len(class_ids)):
            if type(class_ids[i])==type(gt_class_ids[i]):
                ar = dict(CellID=dataset_eval.class_names[class_ids[i]], gtCellID=dataset_eval.class_names[gt_class_ids[i]], 
                      bbox_y1x1y2x2=rois[i], SubID=captions[sub], Score=r['scores'][i], ImgID=name,
                      GtBbox_y1x1y2x2=gt_bboxs[i])
                lis.append(ar)
                sub+=1
            if type(class_ids[i])==float:
                ar = dict(CellID=class_ids[i], gtCellID=dataset_eval.class_names[gt_class_ids[i]], 
                      bbox_y1x1y2x2=rois[i], SubID=captions[sub], Score=np.nan, ImgID=name,
                      GtBbox_y1x1y2x2=gt_bboxs[i])
                lis.append(ar)
                sub+=1
            if type(gt_class_ids[i])==float:
                ar = dict(CellID=dataset_eval.class_names[class_ids[i]], gtCellID=gt_class_ids[i], 
                      bbox_y1x1y2x2=rois[i], SubID=captions[sub], Score=np.nan, ImgID=name,
                      GtBbox_y1x1y2x2=gt_bboxs[i])
                lis.append(ar)
                sub+=1
        for iou in IoUs:
            f1 = []
            mAP, precisions, recalls, _ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], 
                                            r['class_ids'], r['scores'], r['masks'], iou_threshold=iou)
            for j in range(len(recalls)):
                f1.append((2*(recalls[j]*precisions[j]))/(recalls[j]+precisions[j]))
            
            mAPs.append(mAP)
            F1.append(np.mean(f1))
                
    
    
                
    mAPs = np.nan_to_num(mAPs).tolist()
    F1 = np.nan_to_num(F1).tolist()
    df1 = pd.DataFrame(lis)
    df3 = pd.DataFrame({'mAP':mAPs, 'mF1-score': F1})
    df1.to_csv(os.path.join(DATA_SAVE_DIR, 'validation.csv'), sep=',', index=False, float_format='%g')
    df3.to_pickle(os.path.join(DATA_SAVE_DIR, 'Metrics.pkl'))
    
    
    DATA_SAVE_DIR = os.path.join(args.savedir, 'Eval_on_train')
    if not os.path.exists(DATA_SAVE_DIR):
        os.makedirs(DATA_SAVE_DIR)
    IoUs = np.arange(0.5,1,0.05)
    lis=[]
    mAPs = []
    F1 = []
    sub=0
    captions = list(range(100000))
    image_ids = dataset_train.image_ids
    for image_id in image_ids:
        begin=len(lis)
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_train, YeastInferenceConfig(), 
                                   image_id, use_mini_mask=False)
        name = list(dataset_train.image_info[image_id].values())[0]
        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                    dataset_train.class_names, SAVE_DIR=os.path.join(DATA_SAVE_DIR, name))
   
##########################################################################################################################################
### Detect on random image of random Images
##########################################################################################################################################
    
        results = model.detect([original_image], verbose=1)

        r = results[0]

        operations.display_instances(image_id, original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset_train.class_names, DATA_SAVE_DIR, name,
                                scores=r['scores'], captions=captions[begin:begin+r['rois'].shape[0]], 
                                figsize=(16,16), show_bbox=True, show_mask=True)
##########################################################################################################################################
### Sort data
##########################################################################################################################################
        gt_class_ids, class_ids, gt_bboxs, rois = sort_gt_data.sort_data(gt_class_id, r['class_ids'], gt_bbox, r['rois'],
                                                                         rtol=0, atol=atol)
    
##########################################################################################################################################
### Convert data into .csv file
##########################################################################################################################################
    
        for i in range(len(class_ids)):
            if type(class_ids[i])==type(gt_class_ids[i]):
                ar = dict(CellID=dataset_train.class_names[class_ids[i]], gtCellID=dataset_train.class_names[gt_class_ids[i]], 
                      bbox_y1x1y2x2=rois[i], SubID=captions[sub], Score=r['scores'][i], ImgID=name,
                      GtBbox_y1x1y2x2=gt_bboxs[i])
                lis.append(ar)
                sub+=1
            if type(class_ids[i])==float:
                ar = dict(CellID=class_ids[i], gtCellID=dataset_train.class_names[gt_class_ids[i]], 
                      bbox_y1x1y2x2=rois[i], SubID=captions[sub], Score=np.nan, ImgID=name,
                      GtBbox_y1x1y2x2=gt_bboxs[i])
                lis.append(ar)
                sub+=1
            if type(gt_class_ids[i])==float:
                ar = dict(CellID=dataset_train.class_names[class_ids[i]], gtCellID=gt_class_ids[i], 
                      bbox_y1x1y2x2=rois[i], SubID=captions[sub], Score=np.nan, ImgID=name,
                      GtBbox_y1x1y2x2=gt_bboxs[i])
                lis.append(ar)
                sub+=1
        for iou in IoUs:
            f1 = []
            mAP, precisions, recalls, _ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], 
                                            r['class_ids'], r['scores'], r['masks'], iou_threshold=iou)
            for j in range(len(recalls)):
                f1.append((2*(recalls[j]*precisions[j]))/(recalls[j]+precisions[j]))
            
            mAPs.append(mAP)
            F1.append(np.mean(f1))
                
    
    
                
    mAPs = np.nan_to_num(mAPs).tolist()
    F1 = np.nan_to_num(F1).tolist()
    df1 = pd.DataFrame(lis)
    df3 = pd.DataFrame({'mAP':mAPs, 'mF1-score': F1})
    df1.to_csv(os.path.join(DATA_SAVE_DIR, 'validation.csv'), sep=',', index=False, float_format='%g')
    df3.to_pickle(os.path.join(DATA_SAVE_DIR, 'Metrics.pkl'))
    
##########################################################################################################################################
### Detection
##########################################################################################################################################
if args.command == 'detect':
    
    config = YeastInferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=args.logs)
    config.IMAGE_RESIZE_MODE = 'square'
    config.display()
    
    # Load weights
    if args.weights == 'last':
        print('Loading your last trained model:')
        model.load_weights(filepath = model.find_last(), by_name=True)
    else:
        print('Loading the model you wanted:')
        model.load_weights(filepath = args.weights, by_name=True)
    
    
    
    # Detection Dataset
    dataset_detect = YeastDataset()
    dataset_detect.load_yeast(datasetdir=args.dataset, subset='detect')
    dataset_detect.prepare()
##########################################################################################################################################
### Preperations for the detection
##########################################################################################################################################
    DATA_SAVE_DIR = os.path.join(args.savedir, 'Detection')
    if not os.path.exists(DATA_SAVE_DIR):
        os.makedirs(DATA_SAVE_DIR)
    CLASS = []
    BBOX = []
    CellID = []
    SCORE = []
    ImgID = []
    Area=[]
    sub=0
    captions = list(range(1000000))
    image_ids = dataset_detect.image_ids
    for image_id in image_ids:
        start=time.time()
        begin=len(CLASS)
        image = dataset_detect.load_image(image_id)
        name = list(dataset_detect.image_info[image_id].values())[0]
##########################################################################################################################################
### Run the detection
##########################################################################################################################################
    
        results = model.detect([image], verbose = 0)
    
        r = results[0]
    
        operations.display_instances(image_id, image, r['rois'], r['masks'], r['class_ids'], 
                                dataset_detect.class_names, DATA_SAVE_DIR, name,
                                scores=r['scores'], captions=captions[begin:begin+r['rois'].shape[0]], 
                                show_bbox=True, show_mask=True)
        
        


##########################################################################################################################################
###Convert detected data into a .pkl file
##########################################################################################################################################
        
        for i in range(len(r['class_ids'])):
            CLASS.append(dataset_detect.class_names[r['class_ids'][i]])
            BBOX.append(r['rois'][i])
            CellID.append(captions[sub])
            SCORE.append('{:.3f}'.format(r['scores'][i]))
            ImgID.append(name)
            Area.append(np.sum(np.where(r['masks'][:,:,i].astype(np.float)!=0,1,0)))
            sub+=1
        print(f'Detection on image {name} took {time.time()-start:.2} sec')
        
            
            
    df = pd.DataFrame({'Class':CLASS, 'Box_y1x1y2x2':BBOX, 'CellID':CellID, 'Score':SCORE, 'Image':ImgID, 'Area':Area})
    df.to_csv(os.path.join(DATA_SAVE_DIR, 'detected.csv'), index=False)
    print(f'Detection finished at {time.ctime()}')
    
    
##########################################################################################################################################
### Teaching
##########################################################################################################################################


if args.command == 'teach':
    
    config = YeastInferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=args.logs)
    config.dsplay()
    
    # Load weights
    if args.weights == 'last':
        print('Loading your last trained model:')
        model.load_weights(filepath = model.find_last(), by_name=True)
    else:
        print('Loading the model you wanted:')
        model.load_weights(filepath = args.weights, by_name=True)
    
    

    # Detection Dataset
    dataset_detect = YeastDataset()
    dataset_detect.load_yeast(datasetdir=args.dataset, subset='detect')
    dataset_detect.prepare()        
        
    # Training Dataset
    dataset_train = YeastDataset()
    dataset_train.load_yeast(datasetdir=args.dataset, subset='train')
    dataset_train.prepare()
    
##########################################################################################################################################
### Preperations for the teaching
##########################################################################################################################################
    DATA_SAVE_DIR = os.path.join(args.dataset, 'train')
    image_ids = dataset_detect.image_ids
    for image_id in image_ids:
        image = dataset_detect.load_image(image_id)
##########################################################################################################################################
### Run the teaching
##########################################################################################################################################
    
        results = model.detect([image], verbose = 0)
    
        r = results[0]
    
        teaching.display_instances(image_id, image, r['rois'], r['masks'], r['class_ids'], 
                                dataset_detect.class_names, DATA_SAVE_DIR, 
                                scores=r['scores'], captions=None, 
                                show_bbox=False, show_mask=False)
        
    
    os.popen('python3 python3 /home/basar/Personal/Erich/Mask_RCNN/samples/yeast/yeast.py train --dataset=/home/basar/Personal/Erich/Studiproject/datasets/yeast/ --weights=last --logs=/home/erich/Schreibtisch/Pupil/')

    
    
    
        
