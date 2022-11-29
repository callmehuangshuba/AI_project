import _pickle as pickle
import numpy as np
import os
# from scipy.misc import imread


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        # 编码方式
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 2):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(cifar10_dir, num_training=500, num_validation=100, num_test=50):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    print(X_train.shape)

    # Subsample the data
    # 测试集
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    # 训练集
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]



    # Normalize the data: subtract the mean image
    # 数据标准化
    mean_image = np.mean(X_train, axis=0)
    # (500, 32, 32, 3)->(num, img_size, img_size, channels)
    X_train -= mean_image
    X_val -= mean_image

    # Transpose so that channels come first
    # 改变顺序->(num, nun_channels, img_size, img_size)
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()

    # 验证集
    if num_test == None:
        return{
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
        }
    else:
        mask = range(num_test)
        X_test = X_test[mask]
        y_test = y_test[mask]

        X_test -= mean_image
        X_test = X_test.transpose(0, 3, 1, 2).copy()

        # Package data into a dictionary
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
        }


def load_models(models_dir):
  """
  Load saved models from disk. This will attempt to unpickle all files in a
  directory; any files that give errors on unpickling (such as README.txt) will
  be skipped.

  Inputs:
  - models_dir: String giving the path to a directory containing model files.
    Each model file is a pickled dictionary with a 'model' field.

  Returns:
  A dictionary mapping model file names to models.
  """
  models = {}
  for model_file in os.listdir(models_dir):
      with open(os.path.join(models_dir, model_file), 'rb') as f:
          try:
              models[model_file] = pickle.load(f)['model']
          except pickle.UnpicklingError:
              continue
  return models
