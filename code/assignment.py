import numpy as np
import pandas as pd

# 数据可视化代码
from titanic_visualizations import survival_stats
from IPython.display import display

# 加载数据集
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

# 显示数据列表中的前几项乘客数据
# display(full_data.head())

# 从数据集中移除 'Survived' 这个特征，并将它存储在一个新的变量中。它也做为我们要预测的目标。
outcomes = full_data['Survived']
data = full_data.drop('Survived', axis=1)


# 显示已移除 'Survived' 特征的数据集
# display(data.head())

def accuracy_score(truth, pred):
    """
    Returns accuracy score for input truth and predictions.
    计算预测的正确率
    """

    # 确保预测的数量与结果的数量一致
    if len(truth) == len(pred):

        # 计算预测准确率（百分比）
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean() * 100)

    else:
        return "Number of predictions does not match number of outcomes!"


# 测试 'accuracy_score' 函数
predictions = pd.Series(np.ones(5, dtype=int))
print(accuracy_score(outcomes[:5], predictions))


def predictions_0(data):
    """
    Model with no features. Always predicts a passenger did not survive.
    predictions_0 函数就预测船上的乘客全部遇难，全部为 0
    Survived：是否存活（0代表否，1代表是）
    """

    predictions = []
    for _, passenger in data.iterrows():
        # 预测 'passenger' 的生还率
        predictions.append(0)

    # 返回预测结果
    return pd.Series(predictions)


# 进行预测
predictions = predictions_0(data)
print("predictions_0 " + accuracy_score(outcomes, predictions))


# 传递给函数的前两个参数分别是泰坦尼克号的乘客数据和乘客的生还结果。第三个参数表明我们会依据哪个特征来绘制图形。
# survival_stats(data, outcomes, 'Sex')
# 观察泰坦尼克号上乘客存活的数据统计，我们可以发现大部分男性乘客在船沉没的时候都遇难了。相反的，大部分女性乘客都在事故中生还。


def predictions_1(data):
    """
    Model with one feature:
            - Predict a passenger survived if they are female.
    如果乘客是男性，那么我们就预测他们遇难；如果乘客是女性，那么我们预测他们在事故中活了下来。
    提示：您可以用访问 dictionary（字典）的方法来访问船上乘客的每个特征对应的值。例如， passenger['Sex'] 返回乘客的性别。
    """

    predictions = []
    for _, passenger in data.iterrows():
        # 移除下方的 'pass' 声明
        # 输入你自己的预测条件
        if 'female' == passenger['Sex']:
            predictions.append(1)
        else:
            predictions.append(0)

    # 返回预测结果
    return pd.Series(predictions)


# 进行预测
predictions = predictions_1(data)
print("predictions_1 " + accuracy_score(outcomes, predictions))


# 综合考虑所有在泰坦尼克号上的男性乘客：我们是否找到这些乘客中的一个子集，他们的存活概率较高。
# survival_stats(data, outcomes, 'Age', ["Sex == 'male'"])
# 仔细观察泰坦尼克号存活的数据统计，在船沉没的时候，大部分小于10岁的男孩都活着，而大多数10岁以上的男性都随着船的沉没而遇难。


def predictions_2(data):
    """
    Model with two features:
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10.
    如果乘客是女性，那么我们就预测她们全部存活；
    如果乘客是男性并且小于10岁，我们也会预测他们全部存活；所有其它我们就预测他们都没有幸存。
    """

    predictions = []
    for _, passenger in data.iterrows():
        # 移除下方的 'pass' 声明
        # 输入你自己的预测条件
        if 'female' == passenger['Sex']:
            predictions.append(1)
        else:
            if passenger['Age'] <= 10:
                predictions.append(1)
            else:
                predictions.append(0)

    # 返回预测结果
    return pd.Series(predictions)


# 进行预测
predictions = predictions_2(data)
print("predictions_2 " + accuracy_score(outcomes, predictions))


# survival_stats(data, outcomes, 'Fare', ["Sex == 'female'", "Pclass == 3"])
# 三等舱的女乘客，票价大于等于20的全部遇难了

def predictions_3(data):
    """
    Model with multiple features. Makes a prediction with an accuracy of at least 80%.
    """

    predictions = []
    for _, passenger in data.iterrows():
        if 'female' == passenger['Sex']:
            if passenger['Pclass'] == 3:  # 三等舱
                if passenger['Fare'] >= 20:  # 票价大于等于20
                    predictions.append(0)
                else:
                    predictions.append(1)
            else:
                predictions.append(1)
        else:
            if passenger['Age'] <= 10:
                predictions.append(1)
            else:
                predictions.append(0)

    return pd.Series(predictions)


predictions = predictions_3(data)
print("predictions_3 " + accuracy_score(outcomes, predictions))
