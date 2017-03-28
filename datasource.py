from glob import glob
from six.moves import xrange
import numpy as np
import scipy.misc


class DataGenerator:
    def __init__(self, pattern, is_grayscale):
        self.pattern = pattern
        self.is_grayscale = is_grayscale

    def data(self):
        return glob(self.pattern)

    def batch_generator(self, train_size, batch_size):
        files = self.data()
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