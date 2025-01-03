from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
# from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline,TextStreamer

from langchain import PromptTemplate

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

def out(urls):
    # urls = ["https://blog.gopenai.com/unveiling-retrieval-qa-and-load-qa-chain-for-langchain-question-answering-42de13c1de84"]
    loader = WebBaseLoader(urls)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    # print(docs_transformed[0].page_content[0:500])
    doc=docs_transformed[0].page_content[:511]
    # print(doc)


    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    prompt_tmp="""
    You are a helpful assistant, helping students prepare for their exam by summarizing the below context.
    {context}
    detailed summary:
    """
    prompt=PromptTemplate(template=prompt_tmp,input_variables=['context'])
    # streamer=TextStreamer(skip_prompt=True, skip_special_tokens=False)
    # model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin"
    llm = LlamaCpp(
        model_path="../zephyr-7b-beta-pl.Q4_K_S.gguf",
        temperature=0.75,
        max_tokens=120,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
        # streamer=streamer
    )
    out=llm.invoke(prompt.format(context=doc))
    # print("hi alekhya: ",out)
    return out
