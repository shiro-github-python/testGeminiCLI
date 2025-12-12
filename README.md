# testGeminiCLI
Create a chatbot using gemini CLI. Use local LLM to search the web and answer questions.

Virtual environment construction  
$ uv venv  
$ source .venv/bin/activate  
$ uv pip install -r requirements.txt

Local LLM execution  
$ ollama run qwen3:8b

Application execution  
$ python app.py

