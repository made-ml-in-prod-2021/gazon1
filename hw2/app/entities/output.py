from pydantic import BaseModel, validator
from fastapi import HTTPException


class OutputModelData(BaseModel):
    preds: float

    @validator("preds")
    def range_of_probability(cls, preds):
        if not (0 <= preds <= 1):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Model's output should be in interval [0, 1]. "
                )
            )
        return preds
