import readMNIST
import bayesClassfier

if __name__ == '__main__':

	train_imgs = readMNIST.loadImageSet('train')
	train_labels = readMNIST.loadLabelSet('train')
	test_imgs = readMNIST.loadImageSet('test')
	test_labels = readMNIST.loadLabelSet('test')
	print 'Reading finised'
	classNum = 10
	featureNum = 784

	print 'Training start'
	priPro, condPro = bayesClassfier.train(train_imgs, train_labels, classNum, featureNum) 

	print 'Testing start'
	pred = bayesClassfier.classfier( test_imgs, priPro, condPro, 10)
	score = bayesClassfier.accuracyRate(pred, test_labels)
	print "The accuracy socre is ", score
