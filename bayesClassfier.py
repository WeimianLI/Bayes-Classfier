import numpy as np
import cv2


def imgBinaryzation(imgs):
	for i in range(len(imgs)):
		cv_img = imgs[i].astype(np.uint8)
		cv2.threshold(cv_img, 50, 1, cv2.cv.CV_THRESH_BINARY_INV, cv_img)
		imgs[i] = cv_img
	
	return cv_img

	
def train(imgs, labels, classNum, featureNum, valueZone):
	priPro = np.zeros(classNum) #prior probability
	condPro = np.zeros((classNum, featureNum, valueZone)) #conditional probability

	#caculate the prior probability and the conditonal probability
	for i in range(len(labels)):
		img = imgs[i]
		label = labels[i][0]
		priPro[label] += 1
		
		for j in range(featureNum):
			condPro[label][j][img[j]] += 1

	for i in range(classNum):
		for j in range(featureNum):

			total = 0
			for k in range(valueZone):
				total += condPro[i][j][k]

			for k in range(valueZone):
				pro_k = (float(condPro[i][j][k]) / float(total) ) * 100000 + 1 
				condPro[i][j][k] = pro_k

	return priPro, condPro


def cal_pro(img, label, priPro, condPro):
	pro = int(priPro[label])

	for i in range(len(img)):
		pro *= int(condPro[label][i][img[i]])
	
	return pro


def classfier(imgs, priPro, condPro, classNum):
	pred = []

	for img in imgs:
		predLabel = -1
		maxPro = -1 

		for i in range(classNum):
			pro = cal_pro(img, i, priPro, condPro)
		#	print i,': ', pro
			if maxPro < pro:
				maxPro = pro
				predLabel = i

		pred.append(predLabel)

	return pred

def accuracyRate(pred, labels):
	count = 0
	for i in range(len(pred)):
		if pred[i] == int(labels[i][0]):
			count += 1
		#print 'pred: ', pred[i]
		#print 'gt: ', labels[i][0]

	acRate = float(count)/float(len(pred))

	return acRate


