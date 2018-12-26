# keras_learning
In this repository, I will make notes about keras application

Recently, I tried to train a deep learning model for tumor or non-tumor classifcation from pathology images. I used 'keras.preprocessing.image.ImageDataGenerator(...)' class to perform image augumentation. As we could find from keras website (https://keras.io/preprocessing/image/), there are many arugments that can be set for the function ImageDataGenerator(). 

Here there is a question: we could perfrom many different image augumentations together, but how about the orders of different augumentations??
To get answer for this question, I trid to read keras website and search online, but I am still not sure about it. Therefore, I did some experiments and read the original code image.py to verify it here. See the code of standardization below and my added annotations:

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
