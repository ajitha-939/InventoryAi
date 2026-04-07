import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv(dotenv_path=".env")

# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file")

# Initialize LLM (latest working model)
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

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful inventory assistant. When a user asks about restocking, gather their current stock and demand, and use the Inventory Calculator tool."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Agent
agent = create_tool_calling_agent(llm, tool_list, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tool_list, memory=memory, verbose=True)

# Run program
print("Inventory AI Assistant Started (type 'exit' to quit)")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    if not user_input:
        print("Please enter valid input")
        continue

    try:
        response = agent_executor.invoke({"input": user_input})
        print("AI:", response["output"])
    except Exception as e:
        print("Error:", e)
