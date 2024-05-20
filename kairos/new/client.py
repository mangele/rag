import requests
from langchain_core.output_parsers import StrOutputParser

def get_context(question: str) -> str:
    response = requests.post('http://localhost:5000', json={'query': question})
    response.raise_for_status()
    return response.json()["results"][0]

def get_completion(prompt: str, n_predict: int = 120) -> str:
    url = "http://localhost:5000"
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt, "n_predict": n_predict}

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    parser = StrOutputParser()
    return parser.parse(response.json()['content'])

def main():
    question = input("Enter your question: ")
    context = get_context(question)
    prompt = f"""
    Based on the following context(code):
    Context: "{context}"
    Answer the following Question: "{question}"
    """
    answer = get_completion(prompt)
    print(answer)

if __name__ == "__main__":
    main()

