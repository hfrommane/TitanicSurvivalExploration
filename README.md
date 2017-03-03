# TitanicSurvivalExploration

## 问题1
![](https://raw.githubusercontent.com/hfrommane/TitanicSurvivalExploration/master/figure/figure_1.png)

观察泰坦尼克号上乘客存活的数据统计，我们可以发现大部分男性乘客在船沉没的时候都遇难了。相反的，大部分女性乘客都在事故中生还。

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

**准确率为： 78.68%.**

## 问题2
![](https://raw.githubusercontent.com/hfrommane/TitanicSurvivalExploration/master/figure/figure_2.png)

仔细观察泰坦尼克号存活的数据统计，在船沉没的时候，大部分小于10岁的男孩都活着，而大多数10岁以上的男性都随着船的沉没而遇难。

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

**准确率为： 79.24%.**

## 问题3
探索数据：

	survival_stats(data, outcomes, 'Fare', ["Sex == 'female'", "Pclass == 3"])

![](https://raw.githubusercontent.com/hfrommane/TitanicSurvivalExploration/master/figure/figure_3.png)

三等舱的女乘客，票价大于等于20的全部遇难了。

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

**准确率为： 81.37%.**

**达到要求！**