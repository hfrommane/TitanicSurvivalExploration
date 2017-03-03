import pandas as pd

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

# Survived：是否存活（0代表否，1代表是）
# Pclass：社会阶级（1代表上层阶级，2代表中层阶级，3代表底层阶级）
# Name：船上乘客的名字
# Sex：船上乘客的性别（male、female）
# Age:船上乘客的年龄（可能存在 NaN）
# SibSp：乘客在船上的兄弟姐妹和配偶的数量（0 1 2 。。）
# Parch：乘客在船上的父母以及小孩的数量（0 1 。。）
# Ticket：乘客船票的编号
# Fare：乘客为船票支付的费用
# Cabin：乘客所在船舱的编号（可能存在 NaN）
# Embarked：乘客上船的港口（C 代表从 Cherbourg 登船，Q 代表从 Queenstown 登船，S 代表从 Southampton 登船）

# survival_stats(data, outcomes, 'Sex')
# survival_stats(data, outcomes, 'Age', ["Sex == 'male'"])
# survival_stats(data, outcomes, 'Age', ["Sex == 'male'", "Age < 18"])
survival_stats(data, outcomes, 'Fare', ["Sex == 'female'", "Pclass == 3"])
# survival_stats(data, outcomes, 'Embarked', ["Sex == 'female'"])
# survival_stats(data, outcomes, 'Fare', ["Sex == 'female'", "Embarked == 'S'"])
