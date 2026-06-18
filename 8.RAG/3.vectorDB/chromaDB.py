from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
load_dotenv()

model = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')

#Vector DB creation
vectorStoree = Chroma(
    embedding_function= model ,
    persist_directory= 'chromaDDB',
    collection_name='ipl-teams-data'
)


doc1 = Document(
    page_content="Virat Kohli scored a शानदार century in an IPL match leading Royal Challengers Bangalore to victory. His batting style is aggressive and consistent.",
    metadata={"team": "RCB"}
)

doc2 = Document(
    page_content="MS Dhoni is known for his calm leadership. Under his captaincy, Chennai Super Kings have won multiple IPL titles.",
    metadata={"team": "CSK"}
)

doc3 = Document(
    page_content="Mumbai Indians is one of the most successful IPL teams with players like Rohit Sharma and Jasprit Bumrah dominating the league.",
    metadata={"team": "MI"}
)

doc4 = Document(
    page_content="Kolkata Knight Riders rely on strong all-rounders and explosive batting. Their performance has been unpredictable but exciting.",
    metadata={"team": "KKR"}
)

doc5 = Document(
    page_content="Rajasthan Royals focus on young talent and smart strategies. They were the first winners of the IPL in 2008.",
    metadata={"team": "RR"}
)

documents=[doc1, doc2, doc3, doc4, doc5]


#Adding documents to vector DB
vectors = vectorStoree.add_documents(documents)
print(vectors)


# print(vectorStoree.get(include=['embeddings' , 'documents' , 'metadatas']))

# r1 = vectorStoree.similarity_search_with_score(
#     query='who are the best bowler',
#     k=2
# )
# print(r1)
r2 = vectorStoree.similarity_search_with_score(
    query='Who is in CSK team',
    # k=1,
    filter={'team':'CSK'}
)
# print(r2)

vectorStoree.delete(ids=['f0ffdd78-30cf-4a97-93b2-133bcb5a7861'])