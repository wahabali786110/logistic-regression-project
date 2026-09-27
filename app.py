from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.templating import Jinja2Templates

model=joblib.load('loan_model.pkl')
preprocessor=joblib.load('preprocessor.pkl')

class LoanApplication(BaseModel):
    Gender:str
    Married:str
    Dependents:str
    Education:str
    Self_Employed:str
    ApplicantIncome:float
    CoapplicantIncome:float
    LoanAmount:float
    Loan_Amount_Term:float
    Credit_History:float
    Property_Area:str

app=FastAPI()

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

@app.get("/")
def serve_home(request: Request):
    
    return templates.TemplateResponse(request=request, name="index.html")

@app.post('/predict')
def predict_loan(application:LoanApplication):
    model_input={
        'ApplicantIncome':application.ApplicantIncome,
        'CoapplicantIncome':application.CoapplicantIncome,
        'LoanAmount':application.LoanAmount,
        'Loan_Amount_Term':application.Loan_Amount_Term,
        'Credit_History':application.Credit_History,

        'Gender_Male':0,
        'Married_Yes':0,
        'Dependents_1':0,
        'Dependents_2':0,
        'Dependents_3+':0,
        'Education_Not Graduate':0,
        'Self_Employed_Yes':0,
        'Property_Area_Semiurban':0,
        'Property_Area_Urban':0
    }

    if application.Gender=='Male':
        model_input['Gender_Male']=1

    if application.Married=='Yes':
        model_input['Married_Yes']=1

    if application.Dependents=='1':
        model_input['Dependents_1']=1

    elif application.Dependents=='2':
        model_input['Dependents_2']=1

    elif application.Dependents=='3+':
        model_input['Dependents_3+']=1

    if application.Education=='Not Graduate':
        model_input['Education_Not Graduate']=1

    if application.Self_Employed=='Yes':
        model_input['Self_Employed_Yes']=1

    if application.Property_Area=='Semiurban':
        model_input['Property_Area_Semiurban']=1

    elif application.Property_Area=='Urban':
        model_input['Property_Area_Urban']=1

    model_df=pd.DataFrame([model_input])

    model_df=preprocessor.transform(model_df)

    model_prob=model.predict_proba(model_df)[:,1].item()

    is_approved=model_prob>0.55

    if is_approved:
        decision_confidence=model_prob*100
    else:
        decision_confidence=(1-model_prob)*100

    return {
            'Loan Status':is_approved,
            'Confident': decision_confidence       
            }