import csv
from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV




# Create your views here.
def EmployeeAttrition(fil):   
    df = pd.read_csv(fil)  
    df['Attrition'] = [1  if i == 'Yes' else 0 for i in df['Attrition']]
    
    changes = []
    for column in df.columns:
        fields = []
        if df[column].dtype == 'object':
            value = 0
            catDict = {}
            for key in df[column].unique():
                catDict.update({key:value})
                value +=1
            changes.append([column,catDict])
            for record in df[column]:
                fields.append(catDict.get(record))
            df[column] = fields    
    df = df.drop(columns=['EmployeeCount', 'Over18','StandardHours'])
    leftLength = len(df[df['Attrition'] == 1])
    stayedLength = list(range(0,len(df[df['Attrition'] == 0])))

    randomSelection = []
    for i in range(leftLength):
        choice = random.choice(stayedLength)
        stayedLength.remove(choice)
        randomSelection.append(choice)

    randomSelection.sort()

    tempDF = df[df['Attrition'] == 0]

    tempDF = tempDF.reset_index(drop=True)

    equalSample = tempDF.loc[randomSelection]

    result = pd.concat([equalSample,df[df['Attrition'] == 1]])

    result = result.reset_index(drop=True)     





    balance = 0
    balances = []
    stays = []
    lefts = []
    overalls = []
    for num in range(0,20):

        leftLength = len(df[df['Attrition'] == 1]) + balance
        stayedLength = list(range(0,len(df[df['Attrition'] == 0])))

        randomSelection = []
        for i in range(leftLength):
            choice = random.choice(stayedLength)
            stayedLength.remove(choice)
            randomSelection.append(choice)
            
        randomSelection.sort()

        tempDF = df[df['Attrition'] == 0]

        tempDF = tempDF.reset_index(drop=True)

        equalSample = tempDF.loc[randomSelection]

        result = pd.concat([equalSample,df[df['Attrition'] == 1]])

        result = result.reset_index(drop=True)

        features = [i for i in result.columns if i != 'Attrition']

        X = result[features]
        y = result['Attrition']

        train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42, test_size=0.2)

        model = RandomForestClassifier(random_state=42)

        model.fit(train_X,train_y)

        report = classification_report(test_y, model.predict(test_X),output_dict=True)
        
        balances.append(balance)
        stays.append(report['0']['f1-score'])
        lefts.append(report['1']['f1-score'])
        overalls.append(report['macro avg']['f1-score'])
        
        balance += 50     
    bestBalance = balances[overalls.index(max(overalls))]   




    leftLength = len(df[df['Attrition'] == 1]) + bestBalance
    stayedLength = list(range(0,len(df[df['Attrition'] == 0])))

    randomSelection = []
    for i in range(leftLength):
        choice = random.choice(stayedLength)
        stayedLength.remove(choice)
        randomSelection.append(choice)

    randomSelection.sort()

    tempDF = df[df['Attrition'] == 0]

    tempDF = tempDF.reset_index(drop=True)

    equalSample = tempDF.loc[randomSelection]

    result = pd.concat([equalSample,df[df['Attrition'] == 1]])

    result = result.reset_index(drop=True)

    features = [i for i in result.columns if i != 'Attrition']

    X = result[features]
    y = result['Attrition']

    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42, test_size=0.2)

    model = RandomForestClassifier(random_state=42)

    model.fit(train_X,train_y)
    out=model.predict(test_X)
    out=model.predict(test_X)
    out = pd.DataFrame(out, columns = ['output'])
    df = pd.concat([test_X,out],axis=1)
    df.to_csv('media/output.csv')
    # print(accuracy_score(test_y,model.predict(test_X)))
    # print(classification_report(test_y, model.predict(test_X)))                                                                                                                                        

import os
from django.conf import settings
def home(request):
    output=""
    if request.method=='POST':
        fil=request.FILES.getlist('file')
        im=fil[0]._name[:-4]
        fs=FileSystemStorage()
        fs.save(im+".csv",fil[0])
        # return Response("/media/")
        EmployeeAttrition("media/"+im+".csv")
        output="Check Out the Text File"
        # data = open('media/output.csv','r').read()
        # resp = HttpResponse(data, mimetype='application/x-download')
        # resp['Content-Disposition'] = 'attachment;filename=media/output.csv'
    return render(request,'front.html',{'output':output})