from langchain_huggingface import HuggingFaceEmbeddings

model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
result = model.embed_query("hello")
print(str(result))