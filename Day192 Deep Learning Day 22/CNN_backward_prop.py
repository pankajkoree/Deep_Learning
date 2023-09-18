# CNN Backward propagation

import numpy as np
from scipy import ndimage

def conv_backward(dZ,x,w,padding="same"):
    m = x.shape[0]

    db = (1/m)*np.sum(dZ, axis=(0,1,2), keepdims=True)

    if padding=="same":
        pad = (w.shape[0]-1)//2
    else: #padding is valid - i.e no zero padding
        pad =0
    x_padded = np.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),'constant', constant_values = 0)

    #this will allow us to broadcast operations
    x_padded_bcast = np.expand_dims(x_padded, axis=-1) # shape = (m, i, j, c, 1)
    dZ_bcast = np.expand_dims(dZ, axis=-2) # shape = (m, i, j, 1, k)

    dW = np.zeros_like(w)
    f=w.shape[0]
    w_x = x_padded.shape[1]
    for a in range(f):
        for b in range(f):
            #note f-1 - a rather than f-a since indexed from 0...f-1 not 1...f
            dW[a,b,:,:] = (1/m)*np.sum(dZ_bcast*
                x_padded_bcast[:,a:w_x-(f-1 -a),b:w_x-(f-1 -b),:,:],
                axis=(0,1,2))

    dx = np.zeros_like(x_padded,dtype=float)
    Z_pad = f-1
    dZ_padded = np.pad(dZ,((0,0),(Z_pad,Z_pad),(Z_pad,Z_pad),
    (0,0)),'constant', constant_values = 0)

    for m_i in range(x.shape[0]):
        for k in range(w.shape[3]):
            for d in range(x.shape[3]):
                dx[m_i,:,:,d]+=ndimage.convolve(dZ_padded[m_i,:,:,k],
                w[:,:,d,k])[f//2:-(f//2),f//2:-(f//2)]
    dx = dx[:,pad:dx.shape[1]-pad,pad:dx.shape[2]-pad,:]
    return dx,dW,db


def relu(z, deriv = False):
        if(deriv): #this is for the partial derivatives (discussed in next blog post)
            return z>0 #Note that True=1 and False=0 when converted to float
        else:
            return np.multiply(z, z>0)


def pool_forward(x,mode="max"):
    x_patches = x.reshape(x.shape[0],x.shape[1]//2, 2,x.shape[2]//2, 2,x.shape[3])
    if mode=="max":
        out = x_patches.max(axis=2).max(axis=3)
        mask  =np.isclose(x,np.repeat(np.repeat(out,2,axis=1),2,axis=2)).astype(int)
    elif mode=="average":
        out =  x_patches.mean(axis=3).mean(axis=4)
        mask = np.ones_like(x)*0.25
    return out,mask

#backward calculation
def pool_backward(dx, mask):
    return mask*(np.repeat(np.repeat(dx,2,axis=1),2,axis=2))


def fc_backward(dA,a,x,w,b):
    m = dA.shape[1]
    dZ = dA*relu(a,deriv=True)
    dW = (1/m)*dZ.dot(x.T)
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    dx =  np.dot(w.T,dZ)
    return dx, dW,db


def softmax_backward(y_pred, y, w, b, x):
    m = y.shape[1]
    dZ = y_pred - y
    dW = (1/m)*dZ.dot(x.T)
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    dx =  np.dot(w.T,dZ)
    return dx, dW,db
