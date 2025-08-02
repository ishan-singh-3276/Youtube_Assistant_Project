from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
import os

# genai.configure(api_key="AIzaSyBng-yI_H1tNNNU9B3crrZyfQLOVc6b-4w")

load_dotenv()

google_api_key = os.getenv('GOOGLE_API_KEY')
hf_token = os.getenv('HF_TOKEN')

video_url = input("Enter the YouTube video URL: ")

def get_video_id(url):
  index = video_url.find("?v=") + 3
  video_id = ""
  while (index < len(video_url) and video_url[index] != '&'):
    video_id = video_id + video_url[index]
    index = index + 1
  return video_id

video_id = get_video_id(video_url)

try:
    video_id = get_video_id(video_url)
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
    transcript = " ".join([item.text for item in transcript_list.snippets])
except TranscriptsDisabled:
    print("No Captions Available For This Video")

  # st.write("Transcript of the video: " + transcript)

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = splitter.create_documents([transcript])

embeddings = HuggingFaceEmbeddings()

vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

llm = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

prompt = PromptTemplate(
      template="""
        You are a helpful assistant
        Answer ONLY from the provided transcript context.
        If the context if insufficient, just say you don't know

        {context}
        Question: {question}
      """,
      input_variables = ['context', 'question']
  )

question = input("Enter your question: ")

retrieved_chunk = retriever.invoke(question)

context = ""
for chunk in retrieved_chunk:
  context = context + " " + chunk.page_content

final_prompt = prompt.invoke({'context': context, 'question': question})

answer = llm.invoke(final_prompt)   #The LLM doesn't remember context from previous questions
# answer = llm.invoke(st.session_state.messages)   ##The LLM does remember context from previous questions

print("Answer: " + answer.content)