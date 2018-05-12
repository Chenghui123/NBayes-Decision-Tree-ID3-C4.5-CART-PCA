from math import log
import operator
import treePlot
def calcShannonEnt(dataSet):  # 计算数据的熵(entropy)
    numEntries=len(dataSet)  # 数据条数
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1] # 每行数据的最后一个字（类别）
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1  # 统计有多少个类以及每个类的数量
    shannonEnt=0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries # 计算单个类的熵值
        shannonEnt-=prob*log(prob,2) # 累加每个类的熵值
    return shannonEnt

def createDataSet1():    # 创造示例数据
    dataSet = [['长', '粗', '男'],
               ['短', '粗', '男'],
               ['短', '粗', '男'],
               ['长', '细', '女'],
               ['短', '细', '女'],
               ['短', '粗', '女'],
               ['长', '粗', '女'],
               ['长', '粗', '女']]
    labels = ['头发','声音']  #两个特征
    return dataSet,labels

def splitDataSet(dataSet,axis,value): # 按某个特征分类后的数据
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec =featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):  # 选择最优的分类特征
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)  # 原始的熵
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob =len(subDataSet)/float(len(dataSet))
            newEntropy +=prob*calcShannonEnt(subDataSet)  # 按特征分类后的熵
        infoGain = baseEntropy - newEntropy  # 原始熵与按特征分类后的熵的差值
        print(u"ID3中第%d个特征的信息增益为：%.3f"%(i,infoGain))
        if (infoGain>bestInfoGain):   # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
            bestInfoGain=infoGain
            bestFeature = i
    return bestFeature

def C45_chooseBestFeatureToSplit(dataset):
    numFeatures=len(dataset[0])-1
    baseEnt=calcShannonEnt(dataset)
    bestInfoGain_ratio=0.0
    bestFeature=-1
    for i in range(numFeatures): #遍历所有特征
        featList=[example[i]for example in dataset]  
        uniqueVals=set(featList) #将特征列表创建成为set集合，元素不可重复。创建唯一的分类标签列表
        newEnt=0.0
        IV=0.0
        for value in uniqueVals:     #计算每种划分方式的信息熵
            subdataset=splitDataSet(dataset,i,value)
            p=len(subdataset)/float(len(dataset))
            newEnt+=p*calcShannonEnt(subdataset)
            IV=IV-p*log(p,2)
        infoGain=baseEnt-newEnt
        if (IV == 0): 
            continue
        infoGain_ratio = infoGain / IV                   #这个feature的infoGain_ratio    
        print(u"C4.5中第%d个特征的信息增益率为：%.3f"%(i,infoGain_ratio))
        if (infoGain_ratio >bestInfoGain_ratio):          #选择最大的gain ratio
            bestInfoGain_ratio = infoGain_ratio
            bestFeature = i                              #选择最大的gain ratio对应的feature
    return bestFeature

def CART_chooseBestFeatureToSplit(dataset):

    numFeatures = len(dataset[0]) - 1
    bestGini = 999999.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        gini = 0.0
        for value in uniqueVals:
            subdataset=splitDataSet(dataset,i,value)
            p=len(subdataset)/float(len(dataset))
            subp = len(splitDataSet(subdataset, -1, '0')) / float(len(subdataset))
        gini += p * (1.0 - pow(subp, 2) - pow(1 - subp, 2))
        print(u"CART中第%d个特征的基尼值为：%.3f"%(i,gini))
        if (gini < bestGini):
            bestGini = gini
            bestFeature = i
    return bestFeature

def majorityCnt(classList):    #按分类后类别数量排序，比如：最后分类为2男1女，则判定为男；
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def ID3_createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]  # 类别：男或女
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet) #选择最优特征
    bestFeatLabel=labels[bestFeat]
    print(u"此时最优索引为："+(bestFeatLabel))
    myTree={bestFeatLabel:{}} #分类结果以字典形式保存
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=ID3_createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

def C45_createTree(dataset,labels):
    classList=[example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = C45_chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为："+(bestFeatLabel))
    C45Tree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        C45Tree[bestFeatLabel][value] = C45_createTree(splitDataSet(dataset, bestFeat, value), subLabels)
    return C45Tree

def CART_createTree(dataset,labels):
    classList=[example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = CART_chooseBestFeatureToSplit(dataset)
    #print(u"此时最优索引为："+str(bestFeat))
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为："+(bestFeatLabel))
    CARTTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        CARTTree[bestFeatLabel][value] = CART_createTree(splitDataSet(dataset, bestFeat, value), subLabels)
    return CARTTree

if __name__=='__main__':
    dataSet, labels=createDataSet1()  # 创造示列数据
    ID3_Tree = ID3_createTree(dataSet, labels)  # 输出决策树模型结果
    treePlot.ID3_Tree(ID3_Tree)
    print("------------------------------------")

    dataSet, labels=createDataSet1()
    C45_Tree = C45_createTree(dataSet, labels)
    treePlot.C45_Tree(C45_Tree)
    print("------------------------------------")

    dataSet, labels=createDataSet1()
    CART_Tree = CART_createTree(dataSet, labels)
    treePlot.CART_Tree(CART_Tree)
