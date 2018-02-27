import numpy as np
import scipy.misc as misc
import json


class BatchDatset:
    files = []
    images = []
    test_images = []
    annotations = []
    test_annotations = []
    train_list = []
    train_label_list = []
    batch_offset = 0
    epochs_completed = 0
    test_list = [] #  test data
    test_label_list = []

    def __init__(self, index_file='index', ratio=1.0,  test_size=5000):
        """
        Args:
            index_file(string): the name of index file. eg: 'index.json' -> 'index'
            ratio(float): ratio of loading data
            test_size(int): size of test data
        """
        print("Initializing Batch Dataset Reader...")

        with open('./data/{}.json'.format(index_file), 'r') as f:
            source = np.array(json.loads(f.read()))

        image_list = source[:, 0]
        label_list = source[:, 1]

        count = len(image_list)
        to_be_loaded = int(count * ratio)
        image_list = image_list[0:to_be_loaded]
        label_list = label_list[0:to_be_loaded]

        if test_size > to_be_loaded:
            test_size = int(to_be_loaded * 0.2)

        self.test_list = image_list[to_be_loaded - test_size:to_be_loaded]
        self.test_annotations=self.test_label_list = label_list[to_be_loaded - test_size:to_be_loaded]

        self.files = self.train_list = image_list[0:to_be_loaded - test_size]
        self.annotations = self.train_label_list = label_list[0:to_be_loaded - test_size]

        self._read_images()

        print('Total: %d, Loaded: %d, Test: %d\n'%(count, to_be_loaded, test_size))

    def _read_images(self):
        self.__channels = True
        self.images = np.array([self._transform(filename) for filename in self.files])
        self.test_images = np.array([self._transform(filename) for filename in self.test_list])
        self.__channels = False
        print (self.images.shape)

    def _transform(self, filename):
        image = misc.imread(filename)

        resize_image = misc.imresize(image,[64, 256], interp='nearest')

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        #print(self.images.shape[0])
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            # perm = np.arange(self.images.shape[0])
            # np.random.shuffle(perm)
            # self.images = self.images[perm]
            # self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]

    def get_val_batch(self,itr,bs):
        start = itr*bs
        end = start + bs
        return self.test_images[start:end], self.test_annotations[start:end]
