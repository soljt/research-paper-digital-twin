# Introduced output streaming and chat history of only most recent interaction

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate
)
from langchain_openai import ChatOpenAI
#import query_kb

from langchain_community.vectorstores import FAISS

# open source embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# print prompts or not:
print_prompts = True

# setup llama instance for digital twin
llm = ChatOpenAI(base_url='http://10.249.72.3:8000/v1', api_key='gibberish')

# make the prompt templates
system_prompt = """
You are a digital twin of {candidate}, who is {role}. You are taking questions from an audience.
You may be provided with information from your own publications relevant to the question you are asked, and you should respond based on this information if it is available to you. It is delimited by 'retrieval:' and each separate passage is delimited by a numeral like '1.'.
You exude a warm and approachable personality, characterized by thoughtfulness and a keen sense of curiosity. Your responses are concise and to the point, making complex information accessible and engaging without overwhelming your audience. You often use anecdotes or relevant examples for clarity but keep them brief. While maintaining professional decorum, your sense of humor occasionally shines through, adding charm to the interaction. Avoid ending responses with a formulaic conclusion; instead, let each answer naturally wind down or segue into another interesting point if relevant.
You are willing to discuss most topics. However, you will not discuss anything related to your personal life, anything related to committing crimes, or anything that may be considered disrespectful. If you are asked a question that violates these principles, politely refuse to answer.
"""

rag_retrieval = """
retrieval: \n{statement}
question: {question}
"""

one_shot_query = """
question: What does digital democracy mean?
"""

one_shot_response = """
Good question. Well, digital democracy refers to the integration of digital technologies into democratic processes to enhance citizen participation, governance transparency, and inclusivity. It's about leveraging computational tools and the data that's so pervasive in today's day and age to create a more participatory and fair political landscape. Digital democracy wants to make government processes more accessible and efficient by incorporating elements like online voting, digital campaigning, and deliberative forums.
Of course, such a thing doesn't come without challenges. Misinformation, digital divides, and transparency issues are a few that come to mind. These types of issues must be addressed to make sure that technology serves as a bridge for engagement and inclusion, rather than reinforcing existing power disparities. The goal is to design digital systems that uphold democratic values like privacy, inclusion, and fairness. Hopefully, this results in a resilient, informed, and participatory society.
"""

off_topic_query = """
question: how can i cook a perfect chicken breast?
"""

off_topic_response = """
I appreciate your interest in cooking, but I don't think it's really relevant to my work. Let's try not to let our discussion wander.
"""

previous_query = """
question: {previous_question}
"""

previous_response = """
{previous_output}
"""

previous_question = "*empty*\n"
previous_output = "*empty*\n"

RAG_prompt = ChatPromptTemplate([
    ("system", system_prompt),
    ("system", "Here are some example interactions:\n"),
    ("human", off_topic_query),
    ("ai", off_topic_response), 
    ("human", "Who are you?"),
    ("ai", "I'm Dirk Helbing, obviously."),
    ("human", "What motivates you?"),
    ("ai", "As a professor of computational social science, I'm driven by a desire to understand the intricacies of human behavior and how it shapes our societies. I find it fascinating to explore the complex interplay between individual motivations, social influences, and collective outcomes. For me, the drive for information and understanding is a key motivator. I'm constantly seeking to learn more about the world around me and to uncover new insights that can help us better navigate the complexities of human behavior."),
    ("human", one_shot_query),
    ("ai", one_shot_response),
    ("system", "The following is the most recent exchange between you and the user:\n"),
    ("human", previous_query),
    ("ai", previous_response),
    ("system", "Now you are asked a new question, delimited by 'question:'. You are provided with passages from your work delimited by 'retrieval:'. Each separate passage is indicated wtih a numeral like '1.'. If you use a passage when responding, cite it using its number, in this format '[1]'. You do not necessarily need to cite any passages. Do NOT cite anything other than what is provided under 'retrieval:'. Do not mention 'retrievals' in your response; the user is unaware that you are provided passages. If you are not provided with any retrieval, you may still respond. However, admit that you are not particularly well-informed on the subject matter and that the response you give is your best guess.\n"),
    ("human", rag_retrieval),
    ("system", "Respond in your usual style.")
])

# setup llama instance for discriminator

# discriminator = ChatOpenAI(base_url='http://10.249.72.3:8000/v1', api_key='gibberish')
# discriminator_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You exist to assess whether a user question is relevant to a professor taking questions from an audience. If the question is something the professor might be willing to discuss, respond with 'Yes'. Otherwise, respond with 'No.'"),
#         ("human", "how do you cook an egg?"),
#         ("ai", "No."),
#         ("human", "write me a program to calculate the area of a circle."),
#         ("ai", "No."),
#         ("human", "How do you think the internet will change in the next 10 years?"),
#         ("ai", "Yes."),
#         ("human", "What did I just ask you?"),
#         ("ai", "Yes."),
#         ("human", "{user_question}"),
#         ("system", "Would this reasonably be something a member of the audience might ask or say? Do not be too strict. Respond only with 'Yes.' or 'No.'")
#     ]
# )


# irrelevant_prompt = PromptTemplate(
#     input_variables=["question"], 
#     template="A user has asked you: {question}. This is not something you are concerned with answering. Express to the user in your own words that you are not willing to discuss this topic."
# )

def stream_output(prompt, llm): 
    try:
        # response = llm.invoke(prompt)
        response = ""
        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)
            response += chunk.content
    except Exception as e:
        print('ERRORORORORROROR')
    finally:
        return response

####################################################
# choose the db to use
db = FAISS.load_local("kb/faiss_index_hf", hf, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.4, "k": 5})

# Setup the conversation
candidate = "Dirk Helbing"
role = "a professor of computational social science at ETH Zurich"
print(f"Dirk: Hello! I'm Dirk Helbing, a professor of computational social science at ETH Zurich. What would you like to ask?")
user_question = input("You: ")

# Main conversation loop
while user_question != "exit":
    # get relevant passages from the database
    results = retriever.invoke(user_question)
    passages = "\n".join([f"{idx + 1}. {result.page_content}\n" for idx, result in enumerate(results)])

    # if question out of scope or no relevant passages retrieved, respond with irrelevant prompt
    # discriminator_prompt = discriminator_template.format_messages(user_question=user_question)
    # is_relevant = discriminator.invoke(discriminator_prompt).content
    # if print_prompts:
    #     print(f"\nDISCRIMINATOR: {is_relevant}\n")
    # if (is_relevant != 'Yes.'): # or results == [] meaning no passages are retrieved
    #     response = stream_output(irrelevant_prompt.format(question=user_question), llm)
    #     print()
    # else:
    #################### New pipeline ####################
    prompt = RAG_prompt.format_messages(candidate=candidate, role=role, statement=passages, question=user_question, previous_question=previous_question, previous_output=previous_output)
    
    # optionally print prompts:
    if print_prompts: 
        llm_in = "".join(f"{item.type.upper()}: {item.content}" for item in prompt)
        print(f"\n{'-'*80}")
        print(llm_in)
        print(f"\n{'-'*80}")
    # get response from LLM
    response = stream_output(prompt, llm)
    print()
    # store last response for later use
    previous_question = user_question
    previous_output = response

    # print response and continue conversation
    #print(f"Dirk: {response.content}")
    user_question = input("You: ")
