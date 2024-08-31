import streamlit as st

# Streamlit app configuration - MUST be the first Streamlit command
st.set_page_config(page_title="Generate Blogs", page_icon='🤖', layout='centered')

from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import time

@st.cache_resource
def load_llama_model():
    return CTransformers(model='TheBloke/Llama-2-7B-Chat-GGML')

llm = load_llama_model()

# Function to get response from LLama 2 model
@st.cache_data(show_spinner=False)
def getLLamaresponse(input_text, no_words, blog_style):
    try:
        template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
        """
        prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'], template=template)

        # Generate the response from the LLama 2 model
        response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))

        return response
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

st.header("Generate Blogs 🤖")

# Initialize session state to keep track of responses
if 'responses' not in st.session_state:
    st.session_state['responses'] = []

# Input fields for the user
input_text = st.text_input("Enter the Blog Topic")

# Creating two columns for additional fields
col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input('No of Words', value="300")  # Default value to reduce input time
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)

# Generate button
if st.button("Generate"):
    with st.spinner("Generating response..."):
        start_time = time.time()
        response = getLLamaresponse(input_text, no_words, blog_style)
        if response:
            st.session_state['responses'].append(response)
        st.write(f"Generation time: {time.time() - start_time:.2f} seconds")

# Display previous responses
st.subheader("Generated Blogs")
for idx, res in enumerate(st.session_state['responses']):
    st.write(f"**Response {idx + 1}:** {res}")