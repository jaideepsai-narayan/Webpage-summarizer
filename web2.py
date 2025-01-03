from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
# from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline,TextStreamer

from langchain import PromptTemplate

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def out(urls):
# urls = ["https://en.wikipedia.org/wiki/Rahul_Dravid"]
    loader = WebBaseLoader(urls)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    # print(docs_transformed)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = [Document(page_content=x) for x in text_splitter.split_text(docs_transformed[0].page_content)]
    # print(docs)

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embedding_function)
    q0 = "about whom this article is about?"
    q1 = "what is this about?"
    q2 = "why is this?"
    q3 = "what is the strongest point?"
    q4=str((docs_transformed[0].page_content)[:512])

    text=""

    for i in (q4,q0,q1,q2,q3):
        docs = db.similarity_search(i)
        text+=str(docs[0].page_content)+" "
        
    # print(docs_transformed[0].page_content[0:500])

    # doc=docs_transformed[0].page_content[:511]
    # print(context)


    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    prompt_tmp="""
    You are a helpful assistant, helping students to prepare for their exams by summarizing the below context.
    {context}
    detailed summary:
    """
    prompt=PromptTemplate(template=prompt_tmp,input_variables=['context'])
    # # streamer=TextStreamer(skip_prompt=True, skip_special_tokens=False)
    # model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin"
    llm = LlamaCpp(
        model_path="../mistral-7b-instruct-v0.1.Q4_0.gguf",
        temperature=0.75,
        max_tokens=200,
        top_p=1,
        # callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
        # streamer=streamer
    )
    out=llm.invoke(prompt.format(context=text[:511]))
    print(out)
    return out
