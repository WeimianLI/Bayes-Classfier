import readLetterSet
import bayesClassfier

train_labels, train_features, test_labels, test_features = readLetterSet.loadData()
classNum = 26
featureNum = 16
valueZone = 16

print 'Training start'
priPro, condPro = bayesClassfier.train(train_features, train_labels, classNum, featureNum, valueZone)

print 'Testing start'
pred = bayesClassfier.classfier( test_features, priPro, condPro, classNum)
score = bayesClassfier.accuracyRate(pred, test_labels)

print "The accuracy socre is ", score
