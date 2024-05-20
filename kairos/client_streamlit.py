import streamlit as st
import requests
import json
import time

# Function to get a response from the server
def get_response(query):
    url = "http://localhost:5000/"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        'query': query
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        answers = response.json()
        answer_with_rag = answers.get('with_RAG', 'No RAG answer found')
        answer_without_rag = answers.get('without_RAG', 'No non-RAG answer found')
        return answer_with_rag, answer_without_rag
        #return response.json().get('answer', 'No answer found')
    else:
        return "Error: Unable to get response from the server."

def stream_data(gen_answer):
    for word in gen_answer.split(" "):
        yield word + " "
        time.sleep(0.02)

    for word in gen_answer.split(" "):
        yield word + " "
        time.sleep(0.02)

st.title("MediCAN 🚑")

######
with st.chat_message("ai"):
    uploaded_file = st.file_uploader("Choose a CAN data file")
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
    #st.write(bytes_data)


######
prompt = st.chat_input("Pass your prompt here")

# If user submits a query
if prompt:
    st.chat_message("user").markdown(prompt)
    # Display the user's message immediately

    with st.chat_message("ai"):
        with st.status("Analysing data...", expanded=True) as status:
            st.write("Generating Feature Importance and Anomalies")
            time.sleep(3)
            st.write("Projecting CAN data into Feature Importance Space")
            time.sleep(2.5)
            st.write("Matching Feature I space and semantic space")
            time.sleep(2.3)
            status.update(label="✅ Analysis completed", state="complete", expanded=False)

    with st.status("Processing..", expanded=True):
        arag, allm  = get_response(prompt)

        with st.chat_message("ai"):
            st.write("Answers with no context")
            st.write_stream(stream_data(allm))
            st.write("Answers with context")
            st.write_stream(stream_data(arag))
        # Get the response from the server
        



# Debug information (optional)
#if st.session_state.messages:
#    st.write(st.session_state.messages)
