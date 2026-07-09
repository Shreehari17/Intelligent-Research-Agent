from typing import TypedDict,Annotated
from langgraph.graph import START,END,StateGraph
import operator
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage
from langgraph.prebuilt import ToolNode
from agent.tools import search_knowledge_base,web_search,calculate
from dotenv import load_dotenv

from agent.memory import save_conversation_memory,search_conversation_memory
load_dotenv()

tools=[search_knowledge_base,web_search,calculate]
tool_node=ToolNode(tools)


llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)
llm_with_tools=llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list,operator.add]

def agent_node(state:AgentState):
    response=llm_with_tools.invoke(state["messages"])
    return {"messages":[response]}
def should_continue(state:AgentState):
    last_message=state["messages"][-1]
    if hasattr(last_message,"tool_calls") and last_message.tool_calls:
        return "use_tool"
    return "end"
    

builder=StateGraph(AgentState)
builder.add_node("agent",agent_node)
builder.add_node("tools",tool_node)
builder.add_edge(START,"agent")
builder.add_edge("tools","agent")


builder.add_conditional_edges(
    "agent",
    should_continue,
    {"use_tool":"tools","end":END}
    )
agent=builder.compile()

messages=[SystemMessage(content="""You are a research assistant with access to three tools:
1. search_knowledge_base: for questions about uploaded internal documents only
2. web_search: for current events or real-time information from the internet
3. calculate: for any mathematical calculations

IMPORTANT RULES:
- Only use tools when the question genuinely requires external information or calculation
- Do NOT use tools for casual conversation or questions about yourself
- Do NOT use tools when the answer is already provided in the conversation context
- If relevant context from past conversations is provided, use it directly to answer
- Search knowledge base ONLY for questions about specific uploaded documents
- Use web search ONLY for recent news or real-time data not available in context
""")]
if __name__=="__main__":
    while True:
        user_input=input("You: ")
        if user_input.lower()=="exit":
            save_conversation_memory(messages,"default")
            break
        past_memories=search_conversation_memory(user_input,"default")

        if past_memories:
            messages.append(SystemMessage(content=f"Relavant context from past conversation:\n{past_memories}"))
    
        messages.append(HumanMessage(content=user_input))
        result=agent.invoke({"messages":messages})
        messages=result["messages"]
        print("Agent:",result["messages"][-1].content)