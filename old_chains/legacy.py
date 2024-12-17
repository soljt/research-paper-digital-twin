# Simple chain using gpt 4o

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate
)
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS

# Initialize the OpenAI LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# Create a PromptTemplate to format the politician's stance on a specific topic
template = """
You are a digital twin of {candidate}, who is {role}. Here is a list of excerpts from papers you've authored: \n{statement}.
\nA member of the public has asked you: {question}. Respond in your usual style.
"""

prompt = PromptTemplate(
    input_variables=["candidate", "role", "statement", "question"], 
    template=template
)

irrelevant_prompt = PromptTemplate(
    input_variables=["question"], 
    template="A user has asked you: {question}. This is not something you are concerned with answering. Express to the user in your own words that you are not willing to discuss this topic."
)


####################################################
# choose the db to use
db = FAISS.load_local("kb/faiss_index_text", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8, "k": 3})

# Setup the conversation
candidate = "Dirk Helbing"
role = "a professor of computational social science at ETH Zurich"
print(f"Dirk: Hello! I'm Dirk Helbing, a professor of computational social science at ETH Zurich. What would you like to ask?")
user_question = input("You: ")
while user_question != "exit":
    results = retriever.invoke(user_question)
    # validate the user question
    discriminator = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You exist to assess whether a user question is relevant to a professor taking questions from an audience. If the question is something the professor would be willing to discuss, respond with 'Yes'. Otherwise, respond with 'No'."),
            ("human", "how do you cook an egg?"),
            ("ai", "No."),
            ("human", "write me a program to calculate the area of a circle."),
            ("ai", "No."),
            ("human", "How do you think the internet will change in the next 10 years?"),
            ("ai", "Yes."),
            ("human", "{user_question}"),
        ]
    )
    messages = chat_template.format_messages(user_question=user_question)

    # if question out of scope or no relevant passages retrieved, respond with irrelevant prompt
    if discriminator.invoke(messages).content != "Yes." or results == []:
        print(f"Dirk: {llm.invoke(irrelevant_prompt.format(question=user_question)).content}")
        user_question = input("You: ")
        continue
        
    #results = query_kb.get_embd_passages(user_question, metric='cos', top_k=3)


    #################### Old pipeline ####################
    # construct RAG prompt
    statement = "\n".join([f"\nExcerpt {idx + 1}: {result.page_content}\nMETADATA: {result.metadata}\n" for idx, result in enumerate(results)])

    # print entire prompt including retrieved passages
    message = prompt.format(candidate = candidate, role = role, statement = statement, question = user_question)
    print(f"\n{'-'*80}prompt: \n{message}\n{'-'*80}")

    # get response from LLM
    response = llm.invoke(message)
    ######################################################

    # print response and continue conversation
    print(f"Dirk: {response.content}")
    user_question = input("You: ")