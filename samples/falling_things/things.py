import os
import sys
import datetime
import json
import numpy as np
import skimage.draw


import math
import random

import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
from distutils.version import LooseVersion




ROOT_DIR = os.path.abspath("../../")


sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils



COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR,"logs")





class f_things_config(Config):


    NAME  = "Falling_things"

    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 1


    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.9

    IMAGE_MIN_DIM = 540
    IMAGE_MAX_DIM = 960
   
  
    


class f_things_dataset(utils.Dataset):
    
    def load_fthings(self, dataset_dir, subset):


        self.add_class("object", 1 , "object")

        assert subset in ["train", "val"]
        

        if subset == 'train':
            annotations = json.load(open(os.path.join(dataset_dir,subset, "merged.json")))
        else:
           annotations = json.load(open(os.path.join(dataset_dir,subset, "merged_val.json")))

        
        for a in annotations:
            if type(a['objects']) is dict:
                   b_boxes = [b['bounding_box'] for b in a['objects'].values()]
            else:
                   b_boxes = [b['bounding_box'] for b in a['objects']] 

            image_path = a["image_path"]
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
               "object",
               image_id=a['filename'],  # use file name as a unique image id
               path=image_path,
               width=width, height=height,
               b_boxes = b_boxes
               )

    def image_reference(self, image_id):
       """Return the path of the image."""
       info = self.image_info[image_id]
       if info["source"] == "object":
           return info["path"]
       else:
          super(self.__class__, self).image_reference(image_id)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        info = self.image_info[image_id]
        path = info['path']
        mask_path = path.split('.jpg')
        del mask_path[-1]
        mask_path.append('seg.png')
        mask_path = '.'.join(mask_path)
       
        m = skimage.io.imread(mask_path).astype(np.bool)
        mask_shape = m.shape
       
        bbox = utils.extract_bboxes_from_labels(image_id, self, resize = False)
        
        mask = np.zeros(mask_shape + (bbox.shape[0],), dtype=bool)
        
        for i in range(bbox.shape[0]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
            instance_mask = np.zeros(mask_shape)
            y1, x1, y2, x2 = bbox[i][:4]
            instance_mask[y1:y2, x1:x2] = m[y1:y2, x1:x2]
           
      
            if instance_mask.size == 0:
                raise Exception("Invalid bounding box with area of zero")
            # Resize with bilinear interpolation
            #clip = utils.resize(clip, mask_shape)
           
            mask[:, :, i] = np.around(instance_mask).astype(np.bool)
       
      
        return mask,np.ones([mask.shape[-1]], dtype=np.int32)
        
        

############################################################
#  Training
############################################################

def train(model, dataset_dir, subset = None):
    """Train the model."""
    # Training dataset.
    dataset_train = f_things_dataset()
    dataset_train.load_fthings(dataset_dir,'train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = f_things_dataset()
    dataset_val.load_fthings(dataset_dir, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads')


############################################################
#  commandline
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/f_things/dataset/",
                        help='Directory of the f_things dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required = False,
                        metavar="dataset sub_directory",
                        help='subset of dataset to run prediction on')
    
   
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset,"Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    if args.subset:
        print("subset: ", args.subset)
    print(args.weights.lower())
    # Configurations
    if args.command == "train":
        config = f_things_config()
    else:
        class InferenceConfig(f_things_config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True,exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"] )

    # Train or evaluate
    if args.command == "train":
        train(model,args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
