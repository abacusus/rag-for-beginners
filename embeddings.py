from google import genai
from google.genai import types
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

client = genai.Client(api_key="AIzaSyDuk1gDTGii64IIiLyQGoAteq3GCfBapwI")

texts = [
    "What is the meaning of life?",
    "ha vicky is smart and intelligent",
    "How do I bake a cake?",
]#ab dekhna results 

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=texts,
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)

# Create a 3x3 table to show the similarity matrix
df = pd.DataFrame(
    cosine_similarity([e.values for e in result.embeddings]),
    index=texts,
    columns=texts,
)
#yes sahi kaha 
#dekh me bata agar tu question dhyan se dekhega to isme similarty najar aayega dekh what is the  similarity hai aur how wale me less similarity 
print(df)
#this is vector        ha             vector is the conversion of any data into numerical format so that machine learning algorithms can process and analyze it effectively. Vectors are essentially arrays of numbers that represent various features or attributes of the data. In the context of machine learning and data science, vectors are used to represent different types of data, such as text, images, audio, and more.