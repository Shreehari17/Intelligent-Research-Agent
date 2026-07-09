from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from agent.graph import agent
app=FastAPI(title="Intelligent Research Agent")

class ChatRequest(BaseModel):
    session_id:str
    message:str

class ChatResponse(BaseModel):
    session_id:str
    response:str


@app.post("/chat",response_model=ChatResponse)
def chat(req:ChatRequest):
    try:
        config={"configurable":{"thread_id":req.session_id}}
        result=agent.invoke(
            {"messages":[{"role":"user","content":req.message}]},
            config=config,
        )
        final_message=result["messages"][-1].content
        return ChatResponse(session_id=req.session_id,response=final_message)
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@app.get("/health")
def health():
    return {"status":"ok"}
                   