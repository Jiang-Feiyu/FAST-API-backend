from pydantic import BaseModel


class SamplePostRequest(BaseModel):
    message: str
    user_id: int
    sample_id: int
