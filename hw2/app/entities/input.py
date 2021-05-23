from pydantic import BaseModel, validator
from fastapi import HTTPException
import pandas as pd


class InputModelData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

    def to_pandas(self) -> pd.DataFrame:
        """Transform BaseModel to row of pandas.DataFrame"""
        data = pd.DataFrame.from_dict([self.dict()], orient='columns')
        return data

    @validator('sex', 'fbs', 'exang')
    def binary_features_values(cls, binary_f):
        if binary_f not in (0, 1):
            raise HTTPException(
                status_code=400,
                detail="sex, fbs, exang features must be 1 or 0",
            )
        return binary_f
