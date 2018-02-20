import numpy as np
import scipy.misc as misc
import json


class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    train_list = []
    train_label_list = []
    batch_offset = 0
    epochs_completed = 0
    
    def __init__(self, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        with open('./data/index.json', 'r') as f:
            source = np.array(json.loads(f.read()))
        
        self.train_list = source[:, 0]
        self.train_label_list = source[:, 1]

        self.files = self.train_list
        self.annotations = self.train_label_list
        
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        self.__channels = True
        self.images = np.array([self._transform(filename) for filename in self.files])
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

    def get_val_batch(self,itr):
        index = [itr]
        return self.images[index], self.annotations[index]