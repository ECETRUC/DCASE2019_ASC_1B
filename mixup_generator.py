import numpy as np

class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(y_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y

class MixupGenerator_enlarge():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None, n_enlarge=12):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(y_train)
        self.datagen = datagen
        self.n_enlarge=n_enlarge
        
    def __call__(self):
        _, h, w, c = self.X_train.shape
        _, n_labels = self.y_train.shape
        
        XX = np.array([], dtype=np.int64).reshape(0,h, w, c)
        yy = np.array([], dtype=np.int64).reshape(0,n_labels)
        print("shape of y_input: {}".format(yy.shape))
        
        for nloops in range(self.n_enlarge):
            print('enlarge {} times'.format(nloops+1))
            indexes = self.__get_exploration_order(nloops)
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                XX = np.append(XX, X, axis=0)
                yy = np.append(yy, y, axis=0)
            print("shape of y_input enlarged {nloops} times: {yy_shape}".format(nloops=nloops,yy_shape=yy.shape))
                
        return XX, yy

    def __get_exploration_order(self,seed):
        indexes = np.arange(self.sample_num)
        np.random.seed = seed
        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y

class MixupGenerator_two():
    def __init__(self, X_train1, X_train2, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train1 = X_train1
        self.X_train2 = X_train2
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(y_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X1, X2, y = self.__data_generation(batch_ids)

                yield [X1, X2], y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train1.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X11 = self.X_train1[batch_ids[:self.batch_size]]
        X12 = self.X_train1[batch_ids[self.batch_size:]]
        X1 = X11 * X_l + X12 * (1 - X_l)
        
        X21 = self.X_train2[batch_ids[:self.batch_size]]
        X22 = self.X_train2[batch_ids[self.batch_size:]]
        X2 = X21 * X_l + X22 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X1[i] = self.datagen.random_transform(X1[i])
                X1[i] = self.datagen.standardize(X1[i])
                
                X2[i] = self.datagen.random_transform(X2[i])
                X2[i] = self.datagen.standardize(X2[i])
                
        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X1, X2, y

class MixupGenerator_five():
    def __init__(self, X_train1, X_train2, X_train3, X_train4, X_train5, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train1 = X_train1
        self.X_train2 = X_train2
        self.X_train3 = X_train3
        self.X_train4 = X_train4
        self.X_train5 = X_train5
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(y_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X1, X2, X3, X4, X5, y = self.__data_generation(batch_ids)

                yield [X1, X2, X3, X4 , X5], y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train1.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X11 = self.X_train1[batch_ids[:self.batch_size]]
        X12 = self.X_train1[batch_ids[self.batch_size:]]
        X1 = X11 * X_l + X12 * (1 - X_l)
        
        X21 = self.X_train2[batch_ids[:self.batch_size]]
        X22 = self.X_train2[batch_ids[self.batch_size:]]
        X2 = X21 * X_l + X22 * (1 - X_l)
        
        X31 = self.X_train3[batch_ids[:self.batch_size]]
        X32 = self.X_train3[batch_ids[self.batch_size:]]
        X3 = X31 * X_l + X32 * (1 - X_l)
        
        X41 = self.X_train4[batch_ids[:self.batch_size]]
        X42 = self.X_train4[batch_ids[self.batch_size:]]
        X4 = X41 * X_l + X42 * (1 - X_l)
        
        X51 = self.X_train5[batch_ids[:self.batch_size]]
        X52 = self.X_train5[batch_ids[self.batch_size:]]
        X5 = X51 * X_l + X52 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X1[i] = self.datagen.random_transform(X1[i])
                X1[i] = self.datagen.standardize(X1[i])
                
                X2[i] = self.datagen.random_transform(X2[i])
                X2[i] = self.datagen.standardize(X2[i])
                
                X3[i] = self.datagen.random_transform(X3[i])
                X3[i] = self.datagen.standardize(X3[i])
                
                X4[i] = self.datagen.random_transform(X4[i])
                X4[i] = self.datagen.standardize(X4[i])
                
                X5[i] = self.datagen.random_transform(X5[i])
                X5[i] = self.datagen.standardize(X5[i])
                
        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)
#            y_tmp = np.expand_dims(y,axis=1)
#            y = np.concatenate((y_tmp,y_tmp,y_tmp,y_tmp,y_tmp),axis=1)

        return X1, X2, X3, X4, X5, y
    
    
class MixupGenerator_Snapshot():
    def __init__(self, n_cycles, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.n_cycles = n_cycles
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(y_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
#                X_list = None
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield [np.array(X) for rp in range(self.n_cycles)],y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y
    
