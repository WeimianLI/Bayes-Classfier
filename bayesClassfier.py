import numpy as np
import cv2

def imgBinaryzation(img):
	cv_img = img.astype(np.uint8)
	cv2.threshold(cv_img, 50, 1, cv2.cv.CV_THRESH_BINARY_INV, cv_img)
	return cv_img

def train(imgs, labels, classNum, featureNum):
	priPro = np.zeros(classNum) #prior probability
	condPro = np.zeros((classNum, featureNum, 2)) #conditional probability

	#caculate the prior probability and the conditonal probability
	for i in range(len(labels)):
		img = imgBinaryzation(imgs[i])
		label = labels[i][0]
		priPro[label] += 1
		
		for j in range(featureNum):
			condPro[label][j][img[j]] += 1

	for i in range(classNum):
		for j in range(featureNum):
			pix_0 = condPro[i][j][0]
			pix_1 = condPro[i][j][1]
			pro_0 = (float(pix_0) / float(pix_0 + pix_1)) * 100000 + 1
			pro_1 = (float(pix_1) / float(pix_0 + pix_1)) * 100000 + 1
			condPro[i][j][0] = pro_0
			condPro[i][j][1] = pro_1

	return priPro, condPro

def cal_pro(img, label, priPro, condPro):
	pro = int(priPro[label])

	for i in range(len(img)):
		pro *= int(condPro[label][i][img[i]])
	
	return pro

def classfier(imgs, priPro, condPro, classNum):
	pred = []

	for img in imgs:
		img = imgBinaryzation(img)
		predLabel = -1
		maxPro = -1 

		for i in range(classNum):
			pro = cal_pro(img, i, priPro, condPro)
			if maxPro < pro:
				maxPro = pro
				predLabel = i

		pred.append(predLabel)

	return pred

def accuracyRate(pred, labels):
	count = 0
	for i in range(len(pred)):
		if pred[i] == labels[i][0]:
			count += 1
	
	acRate = float(count)/float(len(pred))

	return acRate


