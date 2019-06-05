# keras_learning
In this repository, I will make notes about keras application

## Question 1: Image augumentaton orders using Keras function ----------------------------##

Recently, I tried to train a deep learning model for tumor or non-tumor classifcation from pathology images. I used 'keras.preprocessing.image.ImageDataGenerator(...)' class to perform image augumentation. As we could find from keras website (https://keras.io/preprocessing/image/), there are many arugments that can be set for the function ImageDataGenerator(). 

Here there is a question: we could perfrom many different image augumentations together, but how about the orders of different augumentations??
To get answer for this question, I trid to read keras website and search online, but I am still not sure about it. Therefore, I did some experiments and read the original code keras_preprocessing\image.py to verify it here. 

See the following function, especially my annoated parts, we can find: transformation augumentations performed first, then standarization augumentations performed second 

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=self.dtype)
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params) # Hongming Annotations: Transforamtions (such as rotation, flip, brightness) applied first -- (1)
            x = self.image_data_generator.standardize(x)             # Hongming Annotations: standardizations (operations see the second function code I copies later ) applied second --(2)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(self.dtype)
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=self.dtype)
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

Transformation order is not very important for image augumentation, but still we may want to know the order. we can find it at function: def apply_transform(...). Basically the order is: (1) affine transform (rotation, shift, shear, zoom) (2) channel_shift_intensity (3) flip_horizontal (4) flip_vertical (5) brightness

Note that: channel_shift_range: np.clip (each of RGB channel+random intensity, min_x (e.g.,0), max_x(e.g.,255)), (1) random intensity is between -channel_shift_range and + channel_shift range, (2) all 3 channels add with the same intensity values

See the code of standardization below and my added annotations:

    def standardize(self, x):
        """Applies the normalization configuration to a batch of inputs.

        # Arguments
            x: Batch of inputs to be normalized.

        # Returns
            The inputs, normalized.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)     # customized preprocessing_function for augumentation first applied --(1) 
        if self.rescale:                           # rescaling second applied -- (2)
            x *= self.rescale
        if self.samplewise_center:                 # samplewise_center third applied -- (3)
            x -= np.mean(x, keepdims=True)         # note that: here minus the mean of whole color image (e.g., mean of 3 channels)
        if self.samplewise_std_normalization:      # samplewsie_std_normalization fourth applied -- (4)
            x /= (np.std(x, keepdims=True) + 1e-6) # note that: here divide the std of whole color image (e.g., std of 3 channels)

        if self.featurewise_center:                # featurewise_center fifth applied -- (5)
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:    # featurewsie_std_normalization sixth applied --(6)
            if self.std is not None:
                x /= (self.std + 1e-6)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, '
                              'but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:                    # zca_whitening seventh applied -- (7)
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x
