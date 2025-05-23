from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
import uvicorn
from typing import List
from langchain_core.messages import HumanMessage
from Agent.buildagents import run_graph
app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def home():
    return "Đây là hệ thống ai multi-agent hỗ trợ dự đoán bệnh tiểu đường và bệnh đột quỵ"

#Khởi tạo biến lưu lịch sử trò chuyện
conversation_history: List[str]=[]
class MessageDTO(BaseModel):
    message:str
@app.post("/chat")
async def chat(data:MessageDTO):
    global conversation_history
    try:
        print("user:" + data.message)
        conversation_history.append(data.message)
        history=[HumanMessage(content=m) for m in conversation_history]
        print("Hệ thống đang trả lời......")
        reply=run_graph(history)
        print("Hệ thông đã có câu trả lời....")
        #Cập nhật lại lịch sử
        updated=[m.content for m in reply["messages"]]
        conversation_history=updated
        return{
            "result": updated[-1],#Câu trả lời cuối cùng
            "is_end":reply.get("next")=="END"
        }  
    except Exception as e:
        return {"error": str(e)}
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
    #Muốn truy cập từ trình duyệt khác/ máy khác trong cùng mạng đổi host
    #uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    #Và đổi fetch bên html lại ví dụ fetch('http://192.168.1.10:8000/chat')