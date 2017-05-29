import os, shutil
import urllib
import numpy as np

from nn_utils import whiten

# GE: necessary for Python 3
import urllib.request

#Basically:
#urlretrieve saves the file to a temporary file and returns a tuple (filename, headers)
#urlopen returns a Request object whose read method returns a bytestring containing the file contents


def create_binarized_mnist(tpe, dataset_dir='data'):
    print('Creating binarized %s dataset, dataset dir:  %s' % (tpe, dataset_dir))
    file_path = os.path.join(dataset_dir, 'binarized_mnist_{}.amat'.format(tpe))
    print("file_path= ", file_path)

    # Download dataset if necessary
    if not os.path.isfile(file_path):
        print("download file")
        if not os.path.exists(dataset_dir):
            print("make dataset_dir")
            os.makedirs(dataset_dir)
        url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(tpe)
        print("url= ", url)
        #urllib.urlretrieve(url, file_path)  # original (does not work with Python 3.x)
        filename, headers = urllib.request.urlretrieve(url)
        # move from filename to file_path
        os.system("mv %s %s" % (filename, file_path))
        print("filename= ", filename, ",  headers= ", headers)
        #print("return from urlretrieve: data= ", data)
        print('*Downloaded %s to %s' % (url, file_path))
        #return data

    with open(file_path) as f:
        data = [l.strip().split(' ') for l in f.readlines()]
        data = np.array(data).astype(int)
        np.save(os.path.join(dataset_dir, 'binarized_mnist_{}.npy'.format(tpe)), data)
    return data


def binarized_mnist(dataset_dir='data'):
    # Download and create datasets if necessary
    tpes = ['train', 'valid', 'test']
    for tpe in tpes:
        if not os.path.isfile(os.path.join(dataset_dir, 'binarized_mnist_{}.npy'.format(tpe))):
            create_binarized_mnist(tpe)

    print("tpes= ", tpes)
    print("dataset_dir= ", dataset_dir)
    return {tpe: UnlabelledDataSet(np.load(os.path.join(dataset_dir, 'binarized_mnist_{}.npy'.format(tpe))))
            for tpe in tpes}


class UnlabelledDataSet(object):
    def __init__(self,
                 images):
        self._num_examples = images.shape[0]
        self._images = images
        self._whitened_images = whiten(images)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def whitened_images(self):
        return self._whitened_images

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, whitened=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if whitened:
            return self.images[start:end], self._whitened_images[start:end]
        else: return self._images[start:end], self.images[start:end]
