from pydantic import BaseModel
from typing import List

# request
class Text(BaseModel):
    text: str

class Data(BaseModel):
    data: List[Text]

# response
class Output(BaseModel):
    original: str
    generated: List[str]

class Pred(BaseModel):
    prediction: List[Output]