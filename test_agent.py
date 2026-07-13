import os
from dotenv import load_dotenv
load_dotenv()
from src.agent import build_agent
agent = build_agent(model_type="open-source", model_name="openai/gpt-oss-120b")
inputs = {"messages": [("user", "how to treat stomach pain during night at sleep?")]}
response = agent.invoke(inputs)
for i, msg in enumerate(response["messages"]):
    print(f"--- Message {i} ---")
    print(f"Type: {type(msg)}")
    print(f"Content: {repr(msg.content)}")
    if hasattr(msg, 'tool_calls'):
        print(f"Tool Calls: {msg.tool_calls}")
