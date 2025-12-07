# --- Imports ---
import gradio as gr
from openai import OpenAI
from ddgs import DDGS
import json

# --- Constants and Configuration ---

# Configure the OpenAI client to connect to the local Ollama server.
# The API key is not needed for local instances but the library requires it to be set.
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# The model to be used for generating responses.
LLM_MODEL = "qwen3:8b" # As per specification

# --- Core Logic ---

def web_search(query: str) -> str:
    """
    Performs a web search using the DuckDuckGo search library.

    Args:
        query: The search query.

    Returns:
        A formatted string containing the search results, or an error message.
    """
    try:
        print(f"--- Performing web search for: {query} ---")
        with DDGS() as ddgs:
            # We will take the top 3 results to keep the context concise for the LLM
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No results found."
            
            # Format the results into a single string
            return "\n".join([f"Title: {r['title']}\nSnippet: {r['body']}" for r in results])
    except Exception as e:
        print(f"--- Error during web search: {e} ---")
        return f"An error occurred during search: {e}"

def get_answer(question: str) -> str:
    """
    This is the main function that orchestrates the process of getting an answer.
    It involves sending the initial question to the LLM, handling potential tool calls (like web search),
    and generating the final response.

    Args:
        question: The user's question.

    Returns:
        The final answer from the LLM.
    """
    print(f"--- User Question: {question} ---")
    
    # Define the tool(s) the LLM can use
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for recent information or topics the model doesn't know about.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to use.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    
    # Start the conversation with the user's question
    messages = [{"role": "user", "content": question}]

    try:
        # First call to the LLM to see if it uses a tool or answers directly
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # Check if the LLM decided to use a tool
        if tool_calls:
            print("--- LLM decided to use a tool ---")
            messages.append(response_message)  # Append the assistant's decision to messages

            # Currently, we only have one tool defined, so we'll just handle the first one.
            # In a more complex app, you might loop through tool_calls.
            tool_call = tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "web_search":
                # The LLM wants to search, so we execute our search function
                search_results = web_search(query=function_args.get("query"))
                
                # Append the search results to the conversation history as a 'tool' role message
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": search_results,
                    }
                )

                # Second call to the LLM, this time with the search results included
                print("--- Getting final answer from LLM using search results ---")
                final_response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                )
                return final_response.choices[0].message.content
            else:
                # The LLM tried to call a function we don't know about
                return f"Error: LLM tried to call unknown function '{function_name}'."

        else:
            # The LLM answered directly without using a tool
            print("--- LLM answered directly ---")
            return response_message.content

    except Exception as e:
        print(f"--- An error occurred: {e} ---")
        return f"An error occurred while communicating with the LLM: {e}"



# --- Gradio User Interface ---

def create_ui():
    """
    Creates and launches the Gradio web interface.
    """
    with gr.Blocks(title="Local LLM with Web Search") as app:
        gr.Markdown("# Local LLM with Web Search")
        gr.Markdown(
            "Ask a question. The LLM will answer it. If it needs to search the web, it will do so automatically."
        )
        
        with gr.Row():
            text_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What is the latest news about Gemini model?",
                lines=3,
            )
        
        submit_button = gr.Button("Submit")
        
        with gr.Row():
            text_output = gr.Textbox(
                label="Answer",
                lines=10,
                interactive=False,  # The user should not be able to edit the answer
            )

        submit_button.click(
            fn=get_answer,
            inputs=text_input,
            outputs=text_output,
        )
    
    return app

# --- Main Execution ---

if __name__ == "__main__":
    ui = create_ui()
    # To allow sharing, set share=True. For local use, it's safer to keep it False.
    ui.launch()
