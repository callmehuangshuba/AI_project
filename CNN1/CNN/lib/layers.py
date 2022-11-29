import numpy as np


def conv_forward_naive(x, w, b, conv_param):
    # 单个卷积层
    # 步长 和 填充
    stride, pad = conv_param['stride'], conv_param['pad']
    pad = int(pad)
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    # 数组填充 每个轴边缘需要填充的数值数目  padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    H_new = int(1 + (H + 2 * pad - HH) // stride)
    W_new = int(1 + (W + 2 * pad - WW) // stride)
    s = stride
    # 'float' object cannot be interpreted as an integer  类型错误：“float”对象不能解释为整数
    out = np.zeros((N, F, H_new, W_new))

    for i in range(N):       # ith image
        for f in range(F):   # fth filter
            for j in range(H_new):
                for k in range(W_new):
                    #print x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s].shape
                    #print w[f].shape
                    #print b.shape
                    #print np.sum((x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] * w[f]))
                    out[i, f, j, k] = np.sum(x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] * w[f]) + b[f]

    cache = (x, w, b, conv_param)

    return out, cache


def conv_backward_naive(dout, cache):
    x, w, b, conv_param = cache
    pad = int(conv_param['pad'])
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_new = 1 + (H + 2 * pad - HH) // stride
    W_new = 1 + (W + 2 * pad - WW) // stride

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    s = stride
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    for i in range(N):       # ith image
        for f in range(F):   # fth filter
            for j in range(H_new):
                for k in range(W_new):
                    window = x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s]
                    db[f] += dout[i, f, j, k]
                    dw[f] += window * dout[i, f, j, k]
                    dx_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] += w[f] * dout[i, f, j, k]

    # Unpad
    dx = dx_padded[:, :, pad:pad+H, pad:pad+W]

    return dx, dw, db


def relu_forward(x):   
    """    
    Computes the forward pass for a layer of rectified linear units (ReLUs).    
    Input:    
    - x: Inputs, of any shape    
    Returns a tuple of:    
    - out: Output, of the same shape as x    
    - cache: x    
    """   
    out = None    
    out = ReLU(x)    
    cache = x    

    return out, cache

def relu_backward(dout, cache):   
    """  
    Computes the backward pass for a layer of rectified linear units (ReLUs).   
    Input:    
    - dout: Upstream derivatives, of any shape    
    - cache: Input x, of same shape as dout    
    Returns:    
    - dx: Gradient with respect to x    
    """    
    dx, x = None, cache    
    dx = dout    
    dx[x <= 0] = 0    

    return dx

def BatchNorm_forward(x, gamma, beta, bn_param):
    """
    param:x    : 输入数据，设shape(B,L)
    param:gama : 缩放因子  γ  向量 可训练
    param:beta : 平移因子  β  向量 可训练
    param:bn_param   : batchnorm所需要的一些参数
    	eps      : 接近0的数，防止分母出现0
    	momentum : 动量参数，一般为0.9， 0.99， 0.999
    	running_mean ：滑动平均的方式计算新的均值，训练时计算，为测试数据做准备
    	running_var  : 滑动平均的方式计算新的方差，训练时计算，为测试数据做准备
    """
    mode = bn_param["mode"]
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    # 初始化
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))
    out, cache = None, None

    if mode == "train":
        mean = np.mean(x, axis=0) # 均值
        var = np.var(x, axis=0)  # 方差
        x_normalized = (x-mean) / np.sqrt(var + eps)  # 归一化
        results = gamma * x_normalized + beta  # 缩放平移

        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var  = momentum * running_var + (1 - momentum) * var

        inv_var = 1. / np.sqrt(var + eps)
        cache = (x, results, gamma, mean, inv_var)
    elif mode == 'test':
        mean = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * mean + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def BatchNorm_backward(dout, cache):

    dx, dgamma, dbeta = None, None, None
    x, x_hat, gamma, mu, inv_sigma = cache
    N = x.shape[0]

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_hat * dout, axis=0)
    dvar = np.sum(-0.5 * inv_sigma ** 3 * (x - mu) * gamma * dout, axis=0)
    dmu = np.sum(-1 * inv_sigma * gamma * dout, axis=0)

    dx = gamma * dout * inv_sigma + (2 / N) * (x - mu) * dvar + \
         (1 / N) * dmu
    return dx, dgamma, dbeta


def BatchNorm_backward_alt(dout,cache):
    dx,dgamma,dbeta =None,None,None
    x,gamma,beta,x_hat,sample_mean,sample_var,eps = cache
    m = dout.shape[0]
    dxhat = dout*gamma
    dvar = (dxhat*(x - sample_mean)*(-0.5)*np.power(sample_var+eps,-1.5)).sum(axis=0)
    dmean = np.sum(dxhat*(-1)*np.power(sample_var+eps,-0.5),axis=0)
    dmean += dvar * np.sum(-2*(x-sample_mean),axis=0)/m
    dx = dxhat * np.power(sample_var+eps,-0.5)+dvar*2*(x-sample_mean)/m+dmean/m
    dgamma = np.sum(dout*x_hat,axis=0)
    dbeta = np.sum(dout,axis=0)

    return dx,dgamma,dbeta




def max_pool_forward_naive(x, pool_param):
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = 1 + (H - HH) // s
    W_new = 1 + (W - WW) // s
    out = np.zeros((N, C, H_new, W_new))
    for i in range(N):    
        for j in range(C):        
            for k in range(H_new):            
                for l in range(W_new):                
                    window = x[i, j, k*s:HH+k*s, l*s:WW+l*s] 
                    out[i, j, k, l] = np.max(window)

    cache = (x, pool_param)

    return out, cache


def max_pool_backward_naive(dout, cache):
    x, pool_param = cache
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = 1 + (H - HH) // s
    W_new = 1 + (W - WW) // s
    dx = np.zeros_like(x)
    for i in range(N):    
        for j in range(C):        
            for k in range(H_new):            
                for l in range(W_new):                
                    window = x[i, j, k*s:HH+k*s, l*s:WW+l*s]                
                    m = np.max(window)               
                    dx[i, j, k*s:HH+k*s, l*s:WW+l*s] = (window == m) * dout[i, j, k, l]

    return dx



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)
    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    # 全连接层
    out = None
    # Reshape x into rows
    N = x.shape[0]
    x_row = x.reshape(N, -1)         # (N,D)
    out = np.dot(x_row, w) + b       # (N,M)
    cache = (x, w, b)

    return out, cache

def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dx = np.dot(dout, w.T)                       # (N,D)
    dx = np.reshape(dx, x.shape)                 # (N,d1,...,d_k)
    x_row = x.reshape(x.shape[0], -1)            # (N,D)
    dw = np.dot(x_row.T, dout)                   # (D,M)
    # db = np.sum(dout, axis=0, keepdims=True)     # (1,M)
    db = np.sum(dout, axis=0)

    return dx, dw, db


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
         for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
         0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N

    return loss, dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
         0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx

def ReLU(x):
    """ReLU non-linearity."""
    return np.maximum(0, x)