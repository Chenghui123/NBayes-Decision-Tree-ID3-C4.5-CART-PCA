from scipy import stats

f = open('./iris.data.set.txt', 'r')             # 读取数据
d = f.readlines()
f.close()
irisAttribute = []                               # 属性列表
irisCategory = []                                # 类型列表

for eachline in d:                               # 处理数据
	temp = eachline.strip().split(',')
	irisAttribute.append(temp[0:4])
	if temp[-1] == 'Iris-setosa':                # 用数字1-3表示三种类别
		irisCategory.append(1)
	elif temp[-1] == 'Iris-versicolor':
		irisCategory.append(2)
	else:
		irisCategory.append(3)

def cutData(irisAttribute, irisCategory):        # 切分数据，将每种花的前35项数据划分为训练数据，后15项划分为测试数据
	setosa_Data = []
	versicolor_Data = []
	virginica_Data = []
	testDataSet = []
	testCategory = []

	for i in range(0, 35):                       # 得到三种花的属性的前35条数据，分别存储
		setosa_Data.append(irisAttribute[i][:])
		versicolor_Data.append(irisAttribute[i+50][:])
		virginica_Data.append(irisAttribute[i+100][:])

	for i in range(0,15):                        # 得到三种花的后15条数据，测试数据可以存放在一起，但是属性和所属类别分开存储的
		testDataSet.append(irisAttribute[i+35][:])
		testCategory.append(irisCategory[i+35])
	for i in range(0,15):
		testDataSet.append(irisAttribute[i+85][:])
		testCategory.append(irisCategory[i+85])
	for i in range(0,15):
		testDataSet.append(irisAttribute[i+135][:])
		testCategory.append(irisCategory[i+135])

	return setosa_Data, versicolor_Data, virginica_Data, testDataSet, testCategory

def meanOfData(irisData):                       # 计算四种属性值的平均值和方差，方法比较笨，一个一个算的。
	# 训练数据每种属性值的总和
	dataSum_1 = 0.0
	dataSum_2 = 0.0
	dataSum_3 = 0.0
	dataSum_4 = 0.0
	# 训练数据每种属性值减去平均值的平方的和
	dataSum_11 = 0.0
	dataSum_22 = 0.0
	dataSum_33 = 0.0
	dataSum_44 = 0.0
	# 计算属性值总和
	for item in irisData:
		dataSum_1 += float(item[0])
		dataSum_2 += float(item[1])
		dataSum_3 += float(item[2])
		dataSum_4 += float(item[3])
	# 计算平均值
	dataMean_1 = dataSum_1/35
	dataMean_2 = dataSum_2/35
	dataMean_3 = dataSum_3/35
	dataMean_4 = dataSum_4/35
	# 计算方差
	for item in irisData:
		dataSum_11 += (float(item[0])-dataMean_1)**2
		dataSum_22 += (float(item[1])-dataMean_2)**2
		dataSum_33 += (float(item[2])-dataMean_3)**2
		dataSum_44 += (float(item[3])-dataMean_4)**2
	dataVariance_1 = dataSum_11/34
	dataVariance_2 = dataSum_22/34
	dataVariance_3 = dataSum_33/34
	dataVariance_4 = dataSum_44/34

	return dataMean_1, dataMean_2, dataMean_3, dataMean_4, dataVariance_1, dataVariance_2, dataVariance_3, dataVariance_4

def calculate(setosa_Data, versicolor_Data, virginica_Data, testData):    # 假设每个属性成正态分布，我们可以用密度函数来计算概率，这里用到了scipy库里的stats函数，代入参数即可求出概率密度
	# 计算第一种花的各个属性的概率乘积
	dataMean_1, dataMean_2, dataMean_3, dataMean_4, dataVariance_1, dataVariance_2, dataVariance_3, dataVariance_4 = meanOfData(setosa_Data)
	# result_1 = ((1/sqrt(2*3.1415926*dataVariance_1**2))*(e**((-(float(testData[0])-dataMean_1)**2)/(2*dataVariance_1**2))))*((1/sqrt(2*3.1415926*dataVariance_2**2))*(e**((-(float(testData[1])-dataMean_2)**2)/(2*dataVariance_2**2))))*((1/sqrt(2*3.1415926*dataVariance_3**2))*(e**((-(float(testData[2])-dataMean_3)**2)/(2*dataVariance_3**2))))*((1/sqrt(2*3.1415926*dataVariance_4**2))*(e**((-(float(testData[3])-dataMean_4)**2)/(2*dataVariance_4**2))))
	res_1 = stats.norm.pdf(float(testData[0]), dataMean_1, dataVariance_1)*stats.norm.pdf(float(testData[1]), dataMean_2, dataVariance_2)*stats.norm.pdf(float(testData[2]), dataMean_3, dataVariance_3)*stats.norm.pdf(float(testData[3]), dataMean_4, dataVariance_4)
	# 计算第二种花的各个属性的概率乘积
	dataMean_1, dataMean_2, dataMean_3, dataMean_4, dataVariance_1, dataVariance_2, dataVariance_3, dataVariance_4 = meanOfData(versicolor_Data)
	res_2 = stats.norm.pdf(float(testData[0]), dataMean_1, dataVariance_1)*stats.norm.pdf(float(testData[1]), dataMean_2, dataVariance_2)*stats.norm.pdf(float(testData[2]), dataMean_3, dataVariance_3)*stats.norm.pdf(float(testData[3]), dataMean_4, dataVariance_4)
	# 计算第三种花的各个属性的概率乘积
	dataMean_1, dataMean_2, dataMean_3, dataMean_4, dataVariance_1, dataVariance_2, dataVariance_3, dataVariance_4 = meanOfData(virginica_Data)
	res_3 = stats.norm.pdf(float(testData[0]), dataMean_1, dataVariance_1)*stats.norm.pdf(float(testData[1]), dataMean_2, dataVariance_2)*stats.norm.pdf(float(testData[2]), dataMean_3, dataVariance_3)*stats.norm.pdf(float(testData[3]), dataMean_4, dataVariance_4)
	# 根据概率的大小，判断属于哪一类
	if res_1 >= res_2 and res_1 >= res_3:
		print(testData, "为第一种类型")
		return 1
	if res_2 >= res_1 and res_2 >= res_3:
		print(testData, "为第二种类型")
		return 2
	if res_3 >= res_1 and res_3 >= res_2:
		print(testData, "为第三种类型")
		return 3

if __name__ == "__main__":
	setosa_Data, versicolor_Data, virginica_Data, testDataSet, testCategory = cutData(irisAttribute, irisCategory)
	i = 0
	count = 0
	for testData in testDataSet:
		res = calculate(setosa_Data, versicolor_Data, virginica_Data, testData)
		if res == testCategory[i]:
			count+=1
		i+=1

	print("训练数据为35*3，测试数据为15*3时的准确率为：%f" %(count/45))