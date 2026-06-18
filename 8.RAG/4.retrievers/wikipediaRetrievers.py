from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=1, lang='en')

query = 'what is age of pakistan'

result = retriever.invoke(query)
print(result[0].page_content)