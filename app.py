from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictemployment',methods=['GET','POST'])
def predict_employment():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            CGPA=float(request.form.get('CGPA')),
            Internships=int(request.form.get('Internships')),
            Projects=int(request.form.get('Projects')),
            Certifications=int(request.form.get('Certifications')),
            Aptitude_Test_Score=float(request.form.get('Aptitude_Test_Score')),
            Soft_Skills_Rating=float(request.form.get('Soft_Skills_Rating')),
            Extracurricular_Activities=request.form.get('Extracurricular_Activities'),
            Placement_Training=request.form.get('Placement_Training'),
            SSC_Marks=float(request.form.get('SSC_Marks')),
            HSC_Marks=float(request.form.get('HSC_Marks'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")
        print(pred_df)
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        return render_template('home.html', results=results[0])
    
if __name__=="__main__":        
    app.run(host="0.0.0.0",debug=True) 




