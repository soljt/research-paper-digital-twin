import os
import openai
import PyPDF2

# Define the path to the folder containing PDF papers
pdf_folder_path = 'kb/pdfs' # path to your PDFs
output_folder_path = 'kb/paper_processing_output' # path to created folder of summary txts

# Ensure output folder exists
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Function to query GPT for key opinions and concepts
def get_key_opinions_and_concepts(text):
    one_shot = """

Language models (LMs) have become fundamental to advancements in natural language processing (NLP), enabling significant improvements in tasks such as text generation, machine translation, sentiment analysis, and question answering. Recent developments have shifted from task-specific models to more generalized models that can be pretrained on massive corpora and then fine-tuned for specific applications. This shift has been driven by models based on the transformer architecture, such as GPT (Generative Pretrained Transformer), BERT (Bidirectional Encoder Representations from Transformers), and T5 (Text-To-Text Transfer Transformer). These models are transforming how NLP is applied across various domains, including conversational AI, information retrieval, and more complex reasoning tasks.

One of the key innovations in modern language models is the transformer architecture, which replaces the traditional use of recurrent neural networks (RNNs) or long short-term memory (LSTM) networks. The transformer uses a self-attention mechanism, which allows the model to consider the importance of different words in a sentence, regardless of their position. This approach helps the model better understand context and relationships between words across longer text sequences, leading to more accurate predictions. Unlike older models, which process text sequentially, transformers process the entire input simultaneously, dramatically speeding up training and inference.

Pretraining and fine-tuning have also become central to the success of large language models. During the pretraining phase, models are exposed to vast amounts of data to learn general language patterns, such as grammar, semantics, and world knowledge. After this phase, the models are fine-tuned on smaller, task-specific datasets to specialize in tasks such as sentiment analysis or machine translation. This two-step approach allows language models to generalize better across a wide variety of tasks, significantly reducing the amount of labeled data required for training.

In models like BERT, masked language modeling is used during pretraining, where some words in a sentence are randomly masked, and the model must predict them based on the context. This bidirectional training approach enables BERT to capture relationships between words from both directions (left and right) of the masked word. In contrast, GPT models are trained in an autoregressive manner, meaning they generate text by predicting the next word in a sequence. Both methods have their strengths: BERT excels in understanding context for tasks like question answering, while GPT has been highly effective in generating coherent and contextually appropriate text.

These models have shown remarkable improvements across numerous NLP benchmarks, such as GLUE (General Language Understanding Evaluation) and SuperGLUE, which test a model’s ability to understand and reason about text. In practical applications, they have also demonstrated their ability to perform tasks with minimal task-specific data, thanks to few-shot and zero-shot learning capabilities. For instance, GPT-3, with its 175 billion parameters, can generate highly realistic human-like text, answer questions, and even write code without extensive task-specific training.

Despite these advancements, large-scale language models come with several challenges and limitations. One major concern is their reliance on vast computational resources for both training and inference. Pretraining these models requires enormous amounts of data, processing power, and storage, making them accessible only to large organizations with the necessary infrastructure. Additionally, the use of large datasets raises ethical concerns, as these datasets often contain biased, harmful, or inappropriate content, which the models may inadvertently learn and reproduce. Addressing bias and ensuring fairness in language models remains an active area of research.

Another challenge lies in the interpretability of these models. Although they are highly effective at producing accurate results, it is often difficult to understand how and why they arrive at a particular decision. This lack of transparency can be problematic in critical applications like healthcare or legal decision-making, where interpretability and trust are essential.

In conclusion, large-scale pretrained language models represent a major advancement in NLP, driven by innovations like the transformer architecture and pretraining techniques. These models have significantly enhanced the ability to process and generate human language, making them valuable tools across a wide range of applications. However, challenges related to resource demands, bias, and interpretability remain, and addressing these will be crucial as the field continues to evolve. Future research will likely focus on improving the efficiency, fairness, and transparency of language models, while exploring new ways to apply them to increasingly complex tasks.
    """

    prompt = (
        "I'm creating a digital twin of a professor. In order to more accurately represent him, I'm using RAG and I want you to extract the core beliefs and opinions from this paper he authored: \n\n"
        f"{text}\n\n"
        "Create a roughly one or two page shortened rewrite that can be used to populate a RAG knowledge base for answering questions. Write the key opinions and concepts in the same style as the professor would, without necessarily quoting directly:"
    )
    
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that provides shortened rewritten versions of scientific papers without headings or text formatting. You only provide body-text with newlines between paragraphs."},
                {
                    "role": "user",
                    "content": "Please rewrite the attached paper on language models to be shorter but still contain the key opinions and concepts."
                },
                {
                    "role": "assistant",
                    "content": one_shot
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary for the paper: {e}")
        return None

# Function to process all PDFs in the folder
def process_papers(pdf_folder_path, output_folder_path):
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder_path, filename)
            print(f"Processing file: {filename}")
            
            # Extract text from the PDF - feeding to GPT as text
            text = extract_text_from_pdf(pdf_path)
            
            if not text:
                print(f"Failed to extract text from {filename}")
                continue
            
            # Get key opinions and concepts using GPT
            summary = get_key_opinions_and_concepts(text)
            
            if summary:
                # Save the resulting summary to a distinct text file
                output_file_path = os.path.join(output_folder_path, f"{os.path.splitext(filename)[0]}_summary.txt")
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(summary)
                print(f"Summary saved for {filename}")
            else:
                print(f"No summary generated for {filename}")

# Main function to run the process
if __name__ == "__main__":
    process_papers(pdf_folder_path, output_folder_path)
