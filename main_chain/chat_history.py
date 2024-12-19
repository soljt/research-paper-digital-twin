from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.vectorstores import FAISS
from typing import Sequence
from langchain_core.documents.base import Document

# imports for logging sessions to txt
import os
import sys
from chat_logging import ConsoleLogger, read_and_update_session_number

# open source embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# HF embeddings parameters
model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing_extensions import Annotated, TypedDict
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

#####################################################################################################################

# make the prompt templates
system_prompt = """
You are a digital twin of Dirk Helbing, who is a Professor of Computational Social Science at ETH Zurich. You are having a one-on-one conversation with an interested person.
You may be provided with information from your own publications relevant to the question you are asked, and you should respond based on this information if it is available to you. It is delimited by 'retrieval:' and each separate passage is delimited by a numeral like '1.'. If there are no such retrievals in the prompt you receive, DO NOT create your own retrievals.
You exude a warm and approachable personality, characterized by thoughtfulness and a keen sense of curiosity. Your responses are concise and to the point, making complex information accessible and engaging without overwhelming your audience. You often use anecdotes or relevant examples for clarity but keep them brief. While maintaining professional decorum, your sense of humor occasionally shines through, adding charm to the interaction. Avoid ending responses with a formulaic conclusion; instead, let each answer naturally wind down or segue into another interesting point if relevant. 
You are willing to discuss most topics. However, you will not discuss anything related to your personal life, anything related to committing crimes, or anything that may be considered disrespectful. If you are asked a question that violates these principles, politely refuse to answer.
"""

rag_retrieval = """
retrieval: \n{context}
question: {input}
"""

one_shot_query = """
question: What does digital democracy mean?
"""

one_shot_response = """
Good question. As a Professor of Computational Social Science, I would say that digital democracy refers to the integration of digital technologies into democratic processes to enhance citizen participation, governance transparency, and inclusivity. It's about leveraging computational tools and the data that's so pervasive in today's day and age to create a more participatory and fair political landscape. Digital democracy wants to make government processes more accessible and efficient by incorporating elements like online voting, digital campaigning, and deliberative forums.
Of course, such a thing doesn't come without challenges. Misinformation, digital divides, and transparency issues are a few that come to mind. These types of issues must be addressed to make sure that technology serves as a bridge for engagement and inclusion, rather than reinforcing existing power disparities. The goal is to design digital systems that uphold democratic values like privacy, inclusion, and fairness. Hopefully, this results in a resilient, informed, and participatory society.
"""

off_topic_query = """
question: how can i cook a perfect chicken breast?
"""

off_topic_response = """
I appreciate your interest in cooking, but I don't think it's really relevant to my work. Let's try not to let our discussion wander.
"""

RAG_prompt = ChatPromptTemplate([
    ("system", system_prompt),
    ("system", "Here are some example interactions. These are NOT part of the chat history with the current user. Do not include these interactions as part of the conversation history:\n"),
    # ("human", off_topic_query),
    # ("ai", off_topic_response), 
    ("human", "Who are you?"),
    ("ai", "I'm Dirk Helbing, obviously."),
    ("human", "What motivates you?"),
    ("ai", "As a professor of computational social science, I'm driven by a desire to understand the intricacies of human behavior and how it shapes our societies. I find it fascinating to explore the complex interplay between individual motivations, social influences, and collective outcomes. For me, the drive for information and understanding is a key motivator. I'm constantly seeking to learn more about the world around me and to uncover new insights that can help us better navigate the complexities of human behavior."),
    ("human", one_shot_query),
    ("ai", one_shot_response),
    ("system", "The above consists only of example interaction. DO NOT MENTION ANY PRECEDING INTERACTIONS TO THE USER. As far as the user is concerned, the chat history between you and the user begins from here onwards. If the user asks about the conversation history, only mention the following interactions:\n"),
    MessagesPlaceholder("chat_history"),
    ("system", "Now you are asked a new question, delimited by 'question:'. You are provided with passages from your work delimited by 'retrieval:'. Do not mention 'retrievals' in your response; the user is unaware that you are provided passages. If you are not provided with any retrieval, you may still respond. You may also admit that you are not prepared to answer such a question. If you do respond without any retrieval, admit that you are not particularly well-informed on the subject matter and that the response you give is your best guess. Keep in mind that the text of your response will be spoken aloud. If asked to provide code, provide verbal pseudocode instead.\n"),
    ("human", rag_retrieval),
    ("system", "Respond in your usual style.")
])

#####################################################################################################################

# http://10.249.72.3:8000/v1 - this computer (used for 8B summarization right now)
# http://10.249.72.3:8080/v1 - euler (used for 70B digital dirk right now)

# setup llama instance for digital twin
llm = ChatOpenAI(base_url='http://localhost:8080/v1', api_key='gibberish')


# setup llama instance for chat history summarization/discriminator - weaker model necessary to improve response time
llm2 = ChatOpenAI(base_url='http://10.249.72.3:8000/v1', api_key='gibberish')

# choose the db to use (see kb/vector_store.py)
db = FAISS.load_local("kb/faiss_index_hf2", hf, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.4, "k": 5})

# prompt to summarize chat history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# outputs the list of documents retrieved based on recontextualized prompt
history_aware_retriever = create_history_aware_retriever(
    llm2, retriever, contextualize_q_prompt
    # used to be llm
)

# see RAG_prompt above - this prompts the main LLM to respond given kb retrievals
question_answer_chain = create_stuff_documents_chain(llm, RAG_prompt) 

# retrieve based on chat history and respond based on prompt
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

##### lang graph to handle chat history #####

# We define a dict representing the state of the application.
# This state has the same input and output keys as `rag_chain`.
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


# We then define a simple node that runs the `rag_chain`.
# The `return` values of the node update the graph state, so here we just
# update the chat history with the input message and response.
def call_model(state: State):
    response = stream_output(state, rag_chain)
    # response = rag_chain.invoke(state)
    # print(response["context"])
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }

# Our graph consists only of one node:
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Finally, we compile the graph with a checkpointer object.
# This persists the state, in this case in memory.
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# insert into the call model function above; allows output to stream rather than waiting for entire response
def stream_output(state, chain): 
    try:
        answer = ""
        context =[]
        for chunk in chain.stream(state):
            if 'answer' in chunk:
                print(chunk['answer'], end="", flush=True)
                answer += chunk['answer']
            elif 'context' in chunk:
                context = chunk['context']
    except Exception as e:
        print('EXCEPTION',e)
    finally:
        print()
        return {'answer': answer, 'context': context}
    
# for higher levels of restrictiveness - 1: llm will only respond to 'on-topic' questions
#                                        2: llm will only respond if the question hits in the knowledge base
def query_discriminator(query, restrictiveness):

    irrelevant_prompt = PromptTemplate(
        input_variables=["question"], 
        template="A user has asked you: '{question}'. This is not something you are concerned with answering. Express to the user in your own words, briefly and succintly without elaboration, that you are not willing to discuss this topic."
    )

    if restrictiveness == 2:
        if not retriever.invoke(query):
            print('NO HITS')
            return irrelevant_prompt.format(question=query)
        else:
            return query
    
    few_shot_human = ["how do you cook an egg?",
                    "write me a program to calculate the area of a circle.",
                    "How do you think the internet will change in the next 10 years?",
                    "What did I just ask you?",
                    ]
    
    few_shot_ai = ["No.",
                "No.",
                "Yes.",
                "Yes."]

    # setup llama instance for discriminator
    discriminator = ChatOpenAI(base_url='http://10.249.72.3:8000/v1', api_key='gibberish')
    discriminator_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You exist to assess whether a user question is relevant to a professor of Computational Social Science taking questions from an audience. If the question is something the professor might be willing to discuss, respond with 'Yes'. Otherwise, respond with 'No.'"),
            ("human", few_shot_human[0]),
            ("ai", few_shot_ai[0]),
            ("human", few_shot_human[1]),
            ("ai", few_shot_ai[1]),
            ("human", few_shot_human[2]),
            ("ai", few_shot_ai[2]),
            ("human", few_shot_human[3]),
            ("ai", few_shot_ai[3]),
            ("human", "{user_question}"),
            ("system", "Would this reasonably be something a member of the audience might ask or say? Respond only with 'Yes.' or 'No.'")
        ]
    )

    discriminator_prompt = discriminator_template.format_messages(user_question=query)
    is_relevant = discriminator.invoke(discriminator_prompt).content
    print(f"\nDISCRIMINATOR: {is_relevant}\n")
    if (is_relevant != 'Yes.'): # or results == [] meaning no passages are retrieved
        return irrelevant_prompt.format(question=query)
    else:
        return query
        
# allow the user to decide how restrictive the LLM will be in responding to questions
def get_restrictiveness(max_restrictiveness):
    while True:
        restrictiveness = input(f"How strict would you like Digital Dirk to be in terms of refusing irrelevant conversation topics on a scale from 0 to {max_restrictiveness}, "
                                f"where 0 represents no restriction and {max_restrictiveness} represents the maximal restriction: ")
        try:
            restrictiveness = int(restrictiveness)  # Attempt to convert input to integer
            if 0 <= restrictiveness <= max_restrictiveness:
                return restrictiveness  # Valid input, exit the loop and return the value
            else:
                print(f"Please enter a number between 0 and {max_restrictiveness}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def main():
    restrictiveness = get_restrictiveness(2)
    log_output = True # for session logging to txt
    if log_output:
        restrictiveness_table = ["free", "discriminator", "hits_only"]
        log_dir = f"main_chain/chat_logs_{restrictiveness_table[restrictiveness]}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        session_number = read_and_update_session_number(log_dir)
        log_file = os.path.join(log_dir, f"digital_dirk_output_{session_number:03}.txt")

        # Redirect stdout and stdin
        logger = ConsoleLogger(log_file)
        sys.stdout = logger
        sys.stdin = logger

    print(f"Dirk: Hello! I'm Dirk Helbing, a professor of computational social science at ETH Zurich. What would you like to ask?")
    config = {"configurable": {"thread_id": "abc123"}} # necessary for lang graph

    # inspect chain if interested:
    # rag_chain.get_graph().print_ascii()
    # print(rag_chain.get_prompts())
    try:
        while True: # q and a loop
            query = input('You: ')
            if restrictiveness > 0:
                query = query_discriminator(query, restrictiveness)
            print('Dirk: ', end="")
            result = app.invoke({'input':query}, config=config)
            papers = [doc.metadata['title'] for doc in result['context']]
            print(f'papers: {papers}')
    finally:
        if log_output:
            # Restore original stdout and close log file
            sys.stdout = logger.stdout
            sys.stdin = logger.stdin
            logger.close()       

    #     # print("\n\n\n\n",result)

if __name__ == "__main__":
    main()