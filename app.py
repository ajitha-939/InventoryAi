import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# --- PAGE SETUP ---
st.set_page_config(page_title="Inventory Management AI", page_icon="📦", layout="centered")

# Try fetching from Streamlit config first (for Cloud Deployment)
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    # Fallback to local .env file
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found. Please configure it in Streamlit Secrets or your .env file.")
    st.stop()

@st.cache_resource
def get_agent():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=api_key
    )

    from langchain_core.tools import tool

    @tool
    def Inventory_Calculator(stock: int, demand: int) -> str:
        """Calculates restocking needs. You must pass in the stock and demand numbers."""
        if demand > stock:
            return f"Restock needed: {demand - stock} units"
        else:
            return "Stock is sufficient"
            
    tool_list = [Inventory_Calculator]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful inventory assistant. Your job is to help the user manage their inventory. Always be friendly and helpful. When a user asks about restocking, gather their current stock and demand, and use the Inventory Calculator tool."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tool_list, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tool_list, memory=memory, verbose=False)
    return agent_executor

agent_executor = get_agent()

# --- MAIN UI ---
st.title("📦 Inventory AI")
st.write("Welcome to your intelligent inventory assistant!")
st.write("This tool uses AI to analyze your stock and demand levels and determine if you need to order more items.")
st.divider()

st.header("Inventory Check")
st.write("Please enter your current inventory levels below:")

col1, col2 = st.columns(2)
with col1:
    stock_input = st.number_input("Current Stock Amount", min_value=0, value=10, step=1)
with col2:
    demand_input = st.number_input("Current Demand", min_value=0, value=40, step=1)

if st.button("Check Inventory", type="primary"):
    with st.spinner("Analyzing data..."):
        # We format the numbers exactly as the agent expects
        query = f"stock={stock_input},demand={demand_input}"
        
        try:
            response = agent_executor.invoke({"input": query})
            output = response["output"]
            st.success(output)
        except Exception as e:
            st.error(f"I encountered an error: {str(e)}")
