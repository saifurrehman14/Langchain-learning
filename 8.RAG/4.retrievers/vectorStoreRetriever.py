from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

model = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')

vectorStore = Chroma(
    embedding_function=model,
    persist_directory='chrom',
    collection_name='demo_1'
)

doc1 = Document(
    page_content="""
Karachi is the largest city in Pakistan and serves as the country’s economic hub. Located along the Arabian Sea, it is home to major ports that handle a significant portion of Pakistan’s trade. The city is known for its diverse population, vibrant culture, and fast-paced lifestyle.

Karachi offers a mix of modern infrastructure and historical landmarks. Popular areas include Clifton Beach, Saddar, and DHA. Despite facing challenges such as traffic congestion and pollution, Karachi remains a center for business, finance, and education.
"""
)

doc2 = Document(
    page_content="""
Lahore is the cultural capital of Pakistan and is famous for its rich history, architecture, and cuisine. It is the capital of Punjab province and attracts tourists from across the country and beyond.

The city is home to historical landmarks such as the Badshahi Mosque, Lahore Fort, and Shalimar Gardens. Lahore is also known for its lively food scene, especially in areas like Food Street and Anarkali. The people of Lahore are known for their hospitality and love for festivals.
"""
)

doc3 = Document(
    page_content="""
Islamabad is the capital city of Pakistan and is known for its modern design and peaceful environment. Unlike other major cities, Islamabad is well-planned with organized sectors and wide roads.

The city is surrounded by natural beauty, including the Margalla Hills, making it a popular place for outdoor activities like hiking. Islamabad houses important government institutions, foreign embassies, and educational centers. It is considered one of the cleanest and safest cities in the country.
"""
)

doc4 = Document(
    page_content="""
Peshawar is one of the oldest cities in South Asia and has a rich cultural and historical heritage. It is the capital of Khyber Pakhtunkhwa province and has long served as a gateway to Central Asia.

The city is known for its traditional bazaars, such as Qissa Khwani Bazaar, and its unique Pashtun culture. Peshawar offers a glimpse into ancient traditions while also adapting to modern developments. Its food, especially kebabs and traditional dishes, is widely appreciated.
"""
)

doc5 = Document(
    page_content="""
Quetta is the capital of Balochistan province and is known for its scenic beauty and cool climate. It is surrounded by mountains, which give it a unique landscape compared to other cities in Pakistan.

The city is famous for its fruits, especially apples, grapes, and cherries. Quetta has a calm and quiet lifestyle, making it different from more crowded urban centers. It also holds strategic importance due to its proximity to international borders.
"""
)

docc = [doc1 , doc2 , doc3 , doc4 , doc5]

vectorStore.add_documents(docc)

retrieverr = vectorStore.as_retriever(search_kwargs={'k':1})

query = "which is the historical heritage hub city of pakistan"

r = retrieverr.invoke(query)
print(r[0].page_content)
# print(r[1].page_content)