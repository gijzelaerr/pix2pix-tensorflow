from glob import glob
from six.moves import xrange
import numpy as np
import scipy.misc
from astropy.io import fits

class ImageGenerator:
    def __init__(self, test_pattern, train_pattern, val_pattern):
        self.test_pattern = test_pattern
        self.train_pattern = train_pattern
        self.val_pattern = val_pattern
        self.is_grayscale = False

    def train_data(self):
        return glob(self.train_pattern)

    def test_data(self):
        return glob(self.test_pattern)

    def val_data(self):
        return glob(self.val_pattern)

    def train_batch_generator(self, train_size, batch_size):
        files = self.train_data()
        batch_idxs = min(len(files), train_size) // batch_size

        for idx in xrange(0, batch_idxs):
            batch_files = files[idx * batch_size:(idx + 1) * batch_size]
            batch = [self.load_data(batch_file) for batch_file in batch_files]

            if self.is_grayscale:
                batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
            else:
                batch_images = np.array(batch).astype(np.float32)
            yield batch_images, idx, batch_idxs

    def load_data(self, image_path, flip=True, is_test=False):
        img_A, img_B = self._load_image(image_path)
        img_A, img_B = self._preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

        img_A = img_A / 127.5 - 1.
        img_B = img_B / 127.5 - 1.

        img_AB = np.concatenate((img_A, img_B), axis=2)
        # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
        return img_AB

    def _load_image(self, image_path):
        input_img = self._imread(image_path)
        w = int(input_img.shape[1])
        w2 = int(w / 2)
        img_A = input_img[:, 0:w2]
        img_B = input_img[:, w2:w]

        return img_A, img_B

    def _preprocess_A_and_B(self, img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
        if is_test:
            img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
            img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
        else:
            img_A = scipy.misc.imresize(img_A, [load_size, load_size])
            img_B = scipy.misc.imresize(img_B, [load_size, load_size])

            h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
            w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
            img_A = img_A[h1:h1 + fine_size, w1:w1 + fine_size]
            img_B = img_B[h1:h1 + fine_size, w1:w1 + fine_size]

            if flip and np.random.random() > 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

        return img_A, img_B

    def _imread(self, path):
        if self.is_grayscale:
            return scipy.misc.imread(path, flatten=True).astype(np.float)
        else:
            return scipy.misc.imread(path).astype(np.float)


def dataset_chunked(start=0, end=100, split=32):
    """yields the dataset where the images are plit in chunks. split var controls the amoun of splits per axes"""
    for i in range(start, end):
        input_ = "data/meerkat_random_skies_{:04d}-dirty.fits".format(i)
        input_hdu = fits.open(input_)
        input_data = input_hdu[0].data.squeeze()
        target = "data/skymodel_{:04d}.txt.fits".format(i)
        target_hdu = fits.open(target)
        target_data = target_hdu[0].data.squeeze()
        skip = input_data.shape[0] // split
        input_list = []
        target_list = []
        for i in range(split):
            for j in range(split):
                input_chunk = input_data[i * skip:(i * skip) + skip, j * skip:(j * skip) + skip]
                target_chunk = target_data[i * skip:(i * skip) + skip, j * skip:(j * skip) + skip]
                yield input_chunk.reshape(1, input_chunk.size), target_chunk.reshape(1, target_chunk.size)
                #input_list.append(input_chunk.reshape(1, input_chunk.size))
                #target_list.append(target_chunk.reshape(1, target_chunk.size))
        #yield np.concatenate(input_list), np.concatenate(target_list)



class FitsGenerator:
    def __init__(self,
                 input_pattern="data/meerkat_random_skies_*-dirty.fits",
                 target_pattern="data/skymodel_*.txt.fits", split=16):
        self.input_files = glob(input_pattern)
        self.target_files = glob(target_pattern)
        assert len(self.input_files) == len(self.target_files)

        # find out size data
        input_ = self.input_files[0]
        input_hdu = fits.open(input_)
        input_data = input_hdu[0].data.squeeze()

        target = self.target_files[0]
        target_hdu = fits.open(target)
        target_data = target_hdu[0].data.squeeze()

        assert input_data.shape() == target_data.shape()

        self.total_data_size = self.input_files * split * split

        # split data in train and test
        border1 = self.total_data_size // 3
        border2 = (self.total_data_size // 3) * 2

        self._train_data = range(0, border1)
        self._test_data = range(border1, border2)
        self._val_data = range(border2, self.total_data_size)

    def train_data(self):
        return self._train_data

    def test_data(self):
        return self._test_data

    def val_data(self):
        return self._val_data

    def batch_generator(self, start=None, end=None):
        i_length = 20
        j_length = 20

        if start is not None:
            i_start, j_start = divmod(start, i_length)
        else:
            i_start = 0
            j_length = 0

        if end is not None:
            i_end, j_final_end = divmod(end, i_length)
        else:
            i_end = i_length

        j_end = j_length
        i_end = min(i_end, i_length)

        for i in range(i_start, i_end + 1):
            if i == i_end:
                j_end = j_final_end
            for j in range(j_start, j_end):
                yield i, j

    def train_batch_generator(self, train_size, batch_size):
        files = self.train_data()
        batch_idxs = min(len(files), train_size) // batch_size

        for idx in xrange(0, batch_idxs):
            batch_files = files[idx * batch_size:(idx + 1) * batch_size]
            batch = [self.load_data(batch_file) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
            yield batch_images, idx, batch_idxs





