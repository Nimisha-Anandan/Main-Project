
# Credit card customer churn prediction
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
dt1 = pd.read_csv("bank_churner_updated.csv")

# Fitting of Random Forest Model
x=dt1.drop(['Attrition_Flag'],axis=1)
y=pd.DataFrame(dt1['Attrition_Flag'])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
# Since the data set is too small we are taking the whole data to train model
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
#Fitting the model
m=rf.fit(x_train,y_train)
#Saving the model to disk
pickle.dump(rf,open('Pmodel.pkl','wb') )
ypred = rf.predict(x_test)
print(accuracy_score(y_test,ypred))
print(rf.predict([[43,6,2,10388,1961,0.703,61,.649]]))
print(x.columns)