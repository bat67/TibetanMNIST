# 数据集文档

>数据来源和使用版权详见[此网站](https://www.kesci.com/home/dataset/5bfe734a954d6e0010683839)

## 背景描述

MNIST 数据集来自美国国家标准与技术研究所, National Institute of Standards and Technology (NIST). 训练集由来自 250 个不同人手写的数字构成, 其中 50% 是高中学生, 50% 来自人口普查局 (the Census Bureau) 的工作人员。自MNIST数据集建立以来，被广泛地应用于检验各种机器学习算法，测试各种模型，为机器学习的发展做出了不可磨灭的贡献，其当之无愧为历史上最伟大的数据集之一。在一次科研部门的会议上，我无意间看到了一位藏族伙伴的笔记本上写着一些奇特的符号，好奇心驱使我去了解这些符号的意义，我的伙伴告诉我，这些是藏文当中的数字，这对于从小使用阿拉伯数字的我十分惊讶，这些奇特的符号竟有如此特殊的含义！我当即产生了一个想法，能不能让计算机也能识别这些数字呢？这个想法得到了大家的一致认可，于是我们开始模仿MNIST来制作这些数据，由于对藏文的不熟悉，一开始的工作十分艰难，直到取得了藏学研究院同学的帮助，才使得制作工作顺利完成。历时1个月，超过300次反复筛选，最终得到17768张高清藏文手写体数字图像，形成了TibetanMNIST数据集。我和我的团队为其而骄傲，因为它不仅仅是我们自行制作的第一个数据集，更是第一个藏文手写数字的图像数据集！藏文手写数字和阿拉伯数字一样，在藏文中是一个独立的个体，具有笔画简单，便于识别等优良特性。经过反复地商议，我们决定将其完全开源，供所有的开发者自由使用，使其能发挥最大的价值！为了方便大家使用，我们将数据制作成了TFRecords以及npz文件格式【文件顺序未打乱】，使其便于读取，能很好地配合现有机器学习框架使用，当然，如果你觉得它还可以做的更好，你也可以自行DIY，我们将分割后的原始图像也上传到了科赛平台上，你可以将其做成你喜欢的任何数据格式，并创建各种有趣的项目。我和我的团队衷心地希望你能在使用它的过程获得乐趣！最后，十分感谢科赛网提供的平台，为数据的维护和推广提供了极大的便利！能让更多人看到藏文数字和原创数据的美，就是我们最大的收获！

——袁明奇、才让先木、汤吉安等

中央民族大学创业团队巨神人工智能科技

2018年11月27日


## 数据说明

* 数据文化

	藏区按方言划分为卫藏、康巴、安多三大藏区，东接汉地九州。藏区有典：“法域卫藏、人域康巴、马域安多”即“卫藏的宗教、康巴的人、安多的马”。而藏文主要有楷体和形体两种文字，我们本次的TibetanMNIST正是形体藏文中的数字，也就是图片中连笔书写更加简洁的那种

	![pic1](https://github.com/bat67/TibetanMNIST/blob/master/assets/1.png)

	
* 文件列表

	i.TibetanMNIST.tfrecords（每张图像存储为28x28x3的三通道图像矩阵）

	ii.TibetanMNIST.npz（每张图像存储为28x28的单通道图像矩阵）

	iii.TibetanMNIST（原始图像文件，图像文件名的第一个数字为数字类别标签，第二个数字为数字所在纸张标签，第三个数字为纸张标签内的数字序列）

* 数据特征及属性

	![pic2](https://github.com/bat67/TibetanMNIST/blob/master/assets/2.jpg)

* 数据分布

	![pic3](https://github.com/bat67/TibetanMNIST/blob/master/assets/3.jpg)

* 藏文数字与阿拉伯数字对照表

	![pic4](https://github.com/bat67/TibetanMNIST/blob/master/assets/4.jpg)

* 数据示例

	![pic5](https://github.com/bat67/TibetanMNIST/blob/master/assets/5.jpg)

* 数据使用

	* TFReords文件使用
	
	```python
	import tensorflow as tf

	def _parse_function(example_proto):
		features = {
			'label':tf.FixedLenFeature([], tf.int64),
			'img_raw':tf.FixedLenFeature([], tf.string)
		}

		parsed_features = tf.parse_single_example(example_proto, features)
		img = tf.decode_raw(parsed_features['img_raw'], tf.uint8)
		img = tf.reshape(img, [28, 28, 3])
		# 在流中抛出img张量和label张量
		img = tf.cast(img, tf.float32) / 255
		label = tf.cast(parsed_features['label'], tf.int32)
		return img, label

	filenames = ["要读取的文件序列"]
	dataset = tf.data.TFRecordDataset(filenames)
	dataset = dataset.map(_parse_function)
	# 创建单次迭代器
	iterator = dataset.make_one_shot_iterator()
	# 读取图像数据、标签值
	image, label = iterator.get_next()
	
	```
	
	* NPZ文件读取
	
	```python
	import numpy as np

	data = np.load('文件')
	x_train = data['image'].reshape(17768, 784)
	y_train = utils.to_categorical(data['label'], 10)
	```

* 数据来源
	
	中央民族大学创业团队巨神人工智能科技
	
* 使用注意

	本数据集版权归中央民族大学所有，使用该数据请务必注明数据出处，否则我们将追究相应的法律责任！
