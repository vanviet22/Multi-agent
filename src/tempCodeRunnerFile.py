@app.post("/chat")
async def chat(data:MessageDTO):
    global conversation_history
    try:
        print("user:" + data.message)
        # conversation_history.append(data.message)
        # history=[HumanMessage(content=m) for m in conversation_history]
        # reply=run_graph(history)
        # #Cập nhật lại lịch sử
        # updated=[m.content for m in reply["messages"]]
        # conversation_history=updated
        # return{
        #     "result": updated[-1],#Câu trả lời cuối cùng
        #     "is_end":reply.get("next")=="END"
        # }   
        return{"result": "đã nhận được"}
    except Exception as e:
        return {"error": str(e)}