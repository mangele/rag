import requests
from langchain_core.output_parsers import StrOutputParser

#question = "What you can say about the Technical Report No.: 0713234993-00"
#question = "What's the maximum number of charging stations that can be connected to a PKM150 rectification unit?"
question = input()
#question = "What's the maximum number of charging stations that can be connected to a PKM150 rectification unit?"
response = requests.post('http://localhost:5000/ask', json={'query': question})
#print(response)
context = response.json()["results"][0]
# RAG prompt
template = f"""
Based on the following context(code):
Context: "{context}"
Answer the following Question: "{question}"
"""


url = "http://localhost:8080/completion"
headers = {"Content-Type": "application/json"}
data = {"prompt": template, "n_predict":120, 'cache_prompt':False}

response = requests.post(url, headers=headers, json=data)

# Create a parser instance
parser = StrOutputParser()


output = parser.parse(response.json()['content'])

print(output)
