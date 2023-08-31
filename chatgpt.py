import os
import sys
import speech_recognition as sr

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import pyttsx3

import constants

os.environ["OPENAI_API_KEY"] = "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"

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False
recognizer = sr.Recognizer()

def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

def teste():
  with sr.Microphone() as source:
    print("Adjusting noise ")
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Recording for 4 seconds")
    recorded_audio = recognizer.listen(source, timeout=4)
    print("Done recording")
    print("Recognizing the text")
    text = recognizer.recognize_google(
            recorded_audio, 
            language="pt-BR"
        )
    return text


query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
  #loader = DirectoryLoader("data/")
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
  if not query:
    text = teste()
    query = text
  if query in ['quit', 'q', 'exit']:
    sys.exit()

  if text:
    result = chain({"question": query, "chat_history": chat_history})
    SpeakText(result['answer'])
    print(result['answer'])
    chat_history.append((query, result['answer']))
  query = None

