from pydantic import BaseModel


class MessagePostRequest(BaseModel):
    message: str