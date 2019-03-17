import pandas as pd
from sklearn.ensemble import RandomForestClassifier #classification problem

testing_txt = 'Testing/testRF_S5.csv'
training_txt = 'Training/trainRF_S5_Merger.csv'
output_txt = 'Output/outputRF_S5_err1_Merger.csv'

#Load trainData
trainData = pd.read_csv(training_txt, sep = ',', header=None)
trainData.head()

rows, cols = trainData.shape

x = trainData.iloc[:, 0:cols-1].values  #attributes
y = trainData.iloc[:, cols-1].values    #labels

#Grown a forest and fit it to the trainData
print('Forest !')
forest = RandomForestClassifier(n_estimators = 100, max_depth = 15, random_state = 1)
forest.fit(x, y)

#Load testData
testData = pd.read_csv(testing_txt, sep = ',', header=None)
testData.head()

#Predict labels
prediction = forest.predict(testData.as_matrix()) #output lables

print(forest.classes_)

#Export result
f = open(output_txt,'w')
for ea_prediction in prediction:
    f.write(str(ea_prediction))
    f.write('\n')
f.close()

print('Done !')