	数据来源：http://yann.lecun.com/exdb/mnist/
	这是关于手写数字的图像数据库。

	包含文件4个：
train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

	直接去掉.gz后缀，便可以得到数据（不能解压）。数据格式不是通用格式，需要编程来读取。

	数据说明：
The MNIST database was constructed from NIST's Special Database 3 and Special Database 1 which contain binary images of handwritten digits.
The MNIST training set is composed of 30,000 patterns from SD-3 and 30,000 patterns from SD-1. Our test set was composed of 5,000 patterns from SD-3 and 5,000 patterns from SD-1. The 60,000 pattern training set contained examples from approximately 250 writers. We made sure that the sets of writers of the training set and test set were disjoint.


陈泽晗
2016/09/17

