from pydantic import BaseModel


class InterpretPostRequest(BaseModel):
    message: str
    user_id: int
    interpret_id: int
