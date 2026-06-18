from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

splitter = CharacterTextSplitter(
    chunk_size=100,
    separator='',
    chunk_overlap=0
)


text_result = splitter.split_text("Under golden sun and floodlit skies, A game of passion gently lies,Where leather meets the willows grace,And dreams are chased across the space.The bowler runs with steady stride,A storm of focus locked inside,He leaps, he turns, the ball takes flight,A blazing spark in pure delight.The batter stands with watchful eye,Bat raised against the azure sky,A heartbeat still, the moment tight,Then cracks the ball with all his might.")
print(text_result[0])




loader = PyPDFLoader(
    file_path="F:\LangChain Models\8.RAG\TextSplitter\elect.pdf"
    )
document = loader.load()
doc_result = splitter.split_documents(document)

print(doc_result[0].page_content)
