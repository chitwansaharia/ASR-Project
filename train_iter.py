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

import os

import pdb


logger = logging.getLogger(__name__)

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
        last_batch = False
        while not diter.exit_flag:
            if offset == 1000:
                offset = 0
            next_batch = {}
            next_batch["input"] = diter.data[offset:offset+10,:,:,:]
            next_label["target"] = diter.label[offset:offset+10,:]
            diter.queue.put(next_batch)
            offset += 10
            if last_batch:
                diter.queue.put(None)
                return

class SSIterator(object):
    def __init__(self,
                 batch_size,
                 config,
                 seed,
                 use_infinite_loop=False,
                 dtype="int32"):

        self.batch_size = batch_size
        self.config = config
        args = locals()
        args.pop("self")
        self.__dict__.update(args)
        self.load_files()
        self.exit_flag = False

    def load_files(self):
        config = self.config
        self.data = np.zeros((1000,512,300,1))
        self.label = np.zeros((1000,1251))

    def start(self):
        self.exit_flag = False
        self.queue = Queue.Queue(maxsize = 100)
        self.gather = SSFetcher(self)
        self.gather.daemon = True
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
        return batch
