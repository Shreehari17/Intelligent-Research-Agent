import psycopg2
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage,AIMessage

load_dotenv()
embedder=SentenceTransformer("all-MiniLM-L6-v2")
llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")
    )

def save_conversation_memory(messages:list,tenant_id:str):
    role_map = {
    "human": "Human",
    "ai": "AI",
    }
    
    conversation = []

    for message in messages:
        role = role_map.get(message.type)

        if role:
            conversation.append(f"{role}: {message.content}")

    conversation_text = "\n".join(conversation)
    if not conversation:
        return
    ai_summary= llm.invoke(f"""
    Summarize the following conversation in 2-3 sentences.
    Keep only the important context, decisions, and user preferences.
    Conversation:{conversation_text}
    """)
    summary=ai_summary.content

    embedding=embedder.encode(summary).tolist()
    vector_str = "[" + ",".join(str(x) for x in embedding) + "]"
    conn=get_db_connection()
    cursor=conn.cursor()
    cursor.execute("""
    INSERT into memories(tenant_id,summary,embedding)
    values(%s,%s,%s::vector)""",(tenant_id,summary,vector_str))
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Memory saved for tenant: {tenant_id}")

def search_conversation_memory(query:str,tenant_id:str)->str:
    
    try:
        embedding=embedder.encode(query).tolist()
        vector_str = "[" + ",".join(str(x) for x in embedding) + "]"
        conn=get_db_connection()
        cursor=conn.cursor()
        cursor.execute("""
        SELECT summary, embedding <=> %s::vector as distance
        FROM memories
        WHERE tenant_id = %s
        ORDER BY distance
        LIMIT 3
        """, (vector_str, tenant_id))
        rows = cursor.fetchall()

        cursor.close()
        conn.close()
        if rows:
            return "\n".join([r[0]for r in rows])
        else: 
            return ""
    except Exception as e:
        return ""
    
