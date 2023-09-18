import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from keras.datasets import mnist

def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
	"""forward prop convolutional 3D image, RGB image - color
	Arg:
	A_prev: contains the output of prev layer (m, h_prev, w_prev, c_prev)
	W: filter for the convolution (kh, kw, c_prev, c_new)
	b: biases (1, 1, 1, c_new)
	padding: string ‘same’, or ‘valid’
	stride: tuple (sh, sw)

    Return: padded convolved images RGB np.array
	"""

	m, h_prev, w_prev, c_prev = A_prev.shape
	k_h, k_w, c_prev, c_new = W.shape
	s_h, s_w = stride
	

	if padding == 'valid':
		p_h=0
		p_w=0
	

	if padding == 'same':
		p_h = np.ceil(((s_h*h_prev) - s_h + k_h - h_prev) / 2)
		p_h = int(p_h)
		p_w = np.ceil(((s_w*w_prev) - s_w + k_w - w_prev) / 2)
		p_w = int(p_w)
	

	A_prev = np.pad(A_prev, [(0, 0), (p_h, p_h), (p_w, p_w), (0, 0)],
				mode='constant', constant_values=0)
	

	out_h = int(((h_prev - k_h + (2*p_h)) / (stride[0])) + 1)
	out_w = int(((w_prev - k_w + (2*p_w)) / (stride[1])) + 1)
	output_conv = np.zeros((m, out_h, out_w, c_new))
	m_A_prev = np.arange(0, m)
	

	for i in range(out_h):
		for j in range(out_w):
			for f in range(c_new):
				output_conv[m_A_prev, i, j, f] = activation((
					np.sum(np.multiply(
						A_prev[m_A_prev,i*(stride[0])
                        :k_h+(i*(stride[0])),j*(stride[1]):k_w+(j*(stride[1]))],W[:, :, :, f]), axis=(1, 2, 3))) + b[0, 0, 0, f])
				return output_conv


if __name__ == "__main__":
	np.random.seed(0)
	lib =(X_train,y_train),(X_test,y_test)=mnist.load_data()
	X_train = lib['X_train']
	m, h, w = X_train.shape
	X_train_c = X_train.reshape((-1, h, w, 1))
	

	W = np.random.randn(3, 3, 1, 2)
	b = np.random.randn(1, 1, 1, 2)
	

	def relu(Z):
		return np.maximum(Z, 0)
	

	plt.imshow(X_train[0])
	plt.show()
	A = conv_forward(X_train_c, W, b, relu, padding='valid')
	print(A.shape)
	plt.imshow(A[0, :, :, 0])
	plt.show()


