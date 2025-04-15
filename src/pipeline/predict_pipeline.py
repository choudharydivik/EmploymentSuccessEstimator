import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            labelencoder_path = os.path.join("artifacts", "label_encoder.pkl")

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            label_encoder = load_object(file_path=labelencoder_path)
            print("After Loading")


            print(features)

            data_scaled = preprocessor.transform(features)
            preds_encoded = model.predict(data_scaled)
            preds_encoded = preds_encoded.astype(int).ravel()
            preds = label_encoder.inverse_transform(preds_encoded)
            return preds

        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(
        self,
        CGPA: float,
        Internships: int,
        Projects: int,
        Certifications: int,
        Aptitude_Test_Score: float,
        Soft_Skills_Rating: float,
        Extracurricular_Activities: str,
        Placement_Training: str,
        SSC_Marks: float,
        HSC_Marks: float
    ):
        self.CGPA = CGPA
        self.Internships = Internships
        self.Projects = Projects
        self.Certifications = Certifications
        self.Aptitude_Test_Score = Aptitude_Test_Score
        self.Soft_Skills_Rating = Soft_Skills_Rating
        self.Extracurricular_Activities = Extracurricular_Activities
        self.Placement_Training = Placement_Training
        self.SSC_Marks = SSC_Marks
        self.HSC_Marks = HSC_Marks

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "CGPA": [self.CGPA],
                "Internships": [self.Internships],
                "Projects": [self.Projects],
                "Certifications": [self.Certifications],
                "Aptitude_Test_Score": [self.Aptitude_Test_Score],
                "Soft_Skills_Rating": [self.Soft_Skills_Rating],
                "Extracurricular_Activities": [self.Extracurricular_Activities],
                "Placement_Training": [self.Placement_Training],
                "SSC_Marks": [self.SSC_Marks],
                "HSC_Marks": [self.HSC_Marks],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
