from sklearn.datasets import load_digits
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
	digits = load_digits()
	img = np.reshape(digits.data[0], (8,8))

	num = len(digits.data)
	training_num = int(num*2/3)
	train_data = digits.data[:training_num]
	train_target = digits.target[:training_num]
	test_data = digits.data[training_num:]
	test_target = digits.target[training_num:]
	# サポートベクターマシンの定義
	classifier = svm.SVC(gamma=0.001)
	# サポートベクターマシンにて学習
	classifier.fit(train_data,train_target)
	# テストデータで予測
	predicted = classifier.predict(test_data)

	images_and_labels = list(zip(digits.images, digits.target))
	for index, (image, label) in enumerate(images_and_labels[:10]):
	    plt.subplot(2, 5, index + 1)
	    plt.axis('off')
	    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
	    plt.title('Training: %i' % label)
	plt.show()
	# 正解率の確認
	print(metrics.accuracy_score(test_target,predicted))
