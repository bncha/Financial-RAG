from langchain_core.messages import HumanMessage, AIMessage
from llm_project.graph_engine import app
import gradio as gr
import json 

async def stream_chat(message, history):

    msg_history = []
    for m in history:
        if m["role"] == "user":
            msg_history.append(HumanMessage(content=m["content"]))
        else:
            msg_history.append(AIMessage(content=m["content"]))

    inputs = {"messages": msg_history + [HumanMessage(content=message)]}
    config = {"configurable": {"thread_id": "temp_session"}}
    
    # Visual "thinking" for comfort of user
    yield "### 🔍 Searching and Analyzing..."

    # Run the graph
    async for event in app.astream_events(inputs, config=config, version="v2"):
        kind = event["event"]
        node = event.get("metadata", {}).get("langgraph_node", "")

        # Update status based on which node is working
        if node == "retrieve_node":
            yield "### 📂 Extracting information from financial reports..."
        elif node == "generate":
            yield "### ✍️ Synthesizing final answer..."

    # Final state retrieval
    state = await app.aget_state(config)
    res = state.values.get("final_output")

    if res:
        answer = res.generated_response
        sources = ", ".join(res.sources) if res.sources else "None"
        summary = res.summary_sources

        # Final output formatted nicely
        full_response = (
            f"🎯 Answer\n{answer}\n"
            f"\--- \n"
            f"Sources: {sources}"
        )
        #res_dict = res.model_dump() 
        #json_string = json.dumps(res_dict, indent=2)
        yield full_response
    else:
        yield "⚠️ I processed the request but couldn't generate a structured answer. Please try again."


def main():
    demo = gr.ChatInterface(fn=stream_chat, type="messages")
    demo.launch()

if __name__ == "__main__":
    main()