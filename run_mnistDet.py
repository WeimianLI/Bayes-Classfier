import readMNIST
import bayesClassfier


train_imgs = readMNIST.loadImageSet('train')
train_labels = readMNIST.loadLabelSet('train')
test_imgs = readMNIST.loadImageSet('test')
test_labels = readMNIST.loadLabelSet('test')

bayesClassfier.imgBinaryzation(train_imgs)
bayesClassfier.imgBinaryzation(test_imgs)


print 'Reading finised'
classNum = 10
featureNum = 784
valueZone = 2
print 'Training start'
priPro, condPro = bayesClassfier.train(train_imgs, train_labels, classNum, featureNum, valueZone) 

print 'Testing start'
pred = bayesClassfier.classfier( test_imgs, priPro, condPro, classNum)
score = bayesClassfier.accuracyRate(pred, test_labels)

print "The accuracy socre is ", score
