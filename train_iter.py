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

import gen


logger = logging.getLogger(__name__)

class SSFetcher(threading.Thread):
    def __init__(self, parent):
        threading.Thread.__init__(self)
        self.parent = parent


    def run(self):
        diter = self.parent
        offset = 0 
        i = 0
        last_batch = False
        if diter.mode == 'train':
            limit = 341021//diter.config.batch_size
        else:
            limit = 10000//diter.config.batch_size
        print(limit)
        while not diter.exit_flag:
            if offset > limit:
                diter.exit_flag = True
                diter.queue.put(None)
                return
            next_batch = {}
            temp_batch = next(diter.data)
            next_batch["input"] = temp_batch[0]
            next_batch["target"] = temp_batch[1]
            next_batch["num_batches"] = limit
            diter.queue.put(next_batch)
            offset += 1
            if last_batch:
                diter.queue.put(None)
                return

class SSIterator(object):
    def __init__(self,
                 batch_size,
                 config,
                 seed,
                 mode = 'train',
                 use_infinite_loop=False,
                 dtype="int32"):

        self.batch_size = batch_size
        self.config = config
        args = locals()
        args.pop("self")
        self.__dict__.update(args)
        self.load_files()
        self.exit_flag = False
        self.mode = mode

    def load_files(self):
        config = self.config
        self.data = gen.gen(self.mode)
        

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
