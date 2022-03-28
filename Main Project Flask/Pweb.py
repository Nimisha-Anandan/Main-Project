from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
rf_model=pickle.load(open('Pmodel.pkl','rb'))
@app.route('/')
def home():
    return render_template('Phome.html')
@app.route('/predict',methods=['POST'])
def predict():
    a= request.values["Age"]
    trc= request.values["Total Relationship Count"]
    minm= request.values["Months Inactive 12 months"]
    cl= request.values["Credit Limit"]
    tmc= request.values["Total_Amt_Chng_Q4_Q1"]
    trb= request.values["Total Revolving Balance"]
    ttc= request.values["Total_Trans_Ct"]
    tcc= request.values["Total_Ct_Chng_Q4_Q1"]
    churn=[[a,trc,minm,cl,trb,tmc,tcc,ttc]]
    output=rf_model.predict(churn).item()
    print(output)
    return render_template ('Presult.html',prediction_text="Customer status in future : {}".format(output))
if __name__=='__main__':
    
    app.run()

