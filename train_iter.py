import sys
import numpy  as np
import os, gc
import cPickle
import copy
import logging


import threading
import Queue

import collections
import os

import matplotlib
import os
matplotlib.use('Agg')
from pycocotools.coco import COCO
from PIL import Image
from tensorflow_vgg import utils

import pdb


logger = logging.getLogger(__name__)

vocab_path = "/data/lisatmp4/chitwan/mscoco/caption_processed/vocabulary.pkl"
vocabulary = cPickle.load(open(vocab_path,'r'))
STOP_TOKEN  = vocabulary['STOP_TOKEN']

def process(caption,max_length):
    caption_x  = caption[:-1]
    caption_y = caption[1:]
    if len(caption_x) > max_length:
        caption_x = caption[:max_length]
        caption_y = caption[:max_length-1]
        caption_y.append(STOP_TOKEN)
    return (caption_x,caption_y)

class SSFetcher(threading.Thread):
    def __init__(self, parent):
        threading.Thread.__init__(self)
        self.parent = parent
        self.indices = range(len(self.parent.caption_to_image_dict))
        np.random.shuffle(self.indices)


    def run(self):
        diter = self.parent
        offset = 0 
        i = 0
        while not diter.exit_flag:
            last_batch = False
            image_batch = np.zeros((diter.batch_size,224,224,3))
            caption_batch_x = np.zeros((diter.batch_size,diter.max_caption_length))
            caption_batch_y = np.zeros((diter.batch_size,diter.max_caption_length))
            mask = np.zeros((diter.batch_size,diter.max_caption_length))
            counter = 0
            while counter < diter.batch_size:
                if offset == diter.num_data_points:
                    if not diter.use_infinite_loop:
                        print("Hello")
                        last_batch = True
                        diter.queue.put(None)
                        return
                    else:
                        offset = 0
                        print("End")

                index = self.indices[offset]
                (caption_id,image_file) = diter.caption_to_image_dict[index]
                caption = diter.processed_captions[caption_id]
                image = utils.load_image(os.path.join(diter.image_path,image_file))
                caption_x,caption_y = process(caption,diter.max_caption_length)
                mask[counter,:len(caption_x)] = 1
                caption_batch_x[counter,:len(caption_x)] = caption_x
                caption_batch_y[counter,:len(caption_y)] = caption_y
                image_batch[counter,:,:,:] = image.reshape((224, 224, 3))
                counter += 1
                offset += 1

            if counter == diter.batch_size:
                batch = {}
                batch['image_batch'] = image_batch
                batch['caption_batch_x'] = caption_batch_x
                batch['caption_batch_y'] = caption_batch_y
                batch['mask'] = mask
                diter.queue.put(batch)
                i+=1

            if last_batch:
                diter.queue.put(None)
                return

class SSIterator(object):
    def __init__(self,
                 batch_size,
                 config,
                 max_caption_length,
                 seed,
                 use_infinite_loop=False,
                 dtype="int32"):

        self.batch_size = batch_size
        self.max_caption_length = max_caption_length
        self.config = config
        args = locals()
        args.pop("self")
        self.__dict__.update(args)
        self.load_files()
        self.exit_flag = False

    def load_files(self):
        config = self.config
        self.annFile = config.annFile
        self.coco = COCO(self.annFile)
        image_id_list = self.coco.getImgIds()
        self.image_dict = self.coco.loadImgs(image_id_list)
        self.image_path = config.image_path
        self.caption_path = config.caption_file
        self.processed_captions = cPickle.load(open(self.caption_path,'r'))
        self.caption_to_image_dict  = []
        for image in self.image_dict:
            for caption_id in self.coco.getAnnIds(image['id']):
                self.caption_to_image_dict.append((caption_id, image['file_name']))
        self.num_data_points = len(self.caption_to_image_dict)
        



    def start(self):
        self.exit_flag = False
        self.queue = Queue.Queue(maxsize = 100)
        self.gather = SSFetcher(self)
        self.gather.daemon = True
        # print("Hello")
        self.gather.start()

    def __del__(self):
        if hasattr(self, 'gather'):
            self.gather.exitFlag = True
            self.gather.join()

    def __iter__(self):
        return self

    def next(self):
        if self.exit_flag:
            return None
        
        batch = self.queue.get()
        if not batch:
            self.exit_flag = True
            # print("Okay")
        return batch
