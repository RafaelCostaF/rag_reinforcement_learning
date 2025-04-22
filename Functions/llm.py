from openai import OpenAI
import os
import re
# from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from datetime import datetime
import json



# Setup DeepSeek API
os.environ["OPENAI_API_KEY"] = "sk-chave_qualquer"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.deepseek.com")

def get_evaluation_from_llm(query: str, selected_chunks: list[str], answer: str) -> int:
    # AVALIA A QUALIDADE DAS CHUNKS COM BASE NUMA RESPOSTA ESPERADA
    
    
    
    # Format selected chunks for clarity
    formatted_chunks = "\n\n".join([f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(selected_chunks)])

    # Build the prompt
    prompt = (
        "Você é um avaliador de qualidade de resposta.\n\n"
        f"Consulta do usuário:\n{query}\n\n"
        f"Resposta gerada:\n{answer}\n\n"
        f"Fontes utilizadas (chunks):\n{formatted_chunks if selected_chunks else 'Nenhuma'}\n\n"
        "Com base apenas nas informações acima, avalie o quão boa é a resposta à consulta original, "
        "considerando se ela é útil, correta e relevante, usando uma escala de 0 a 10:\n"
        "- 0: Totalmente errada ou sem sentido\n"
        "- 5: Mediana, útil mas incompleta ou ambígua\n"
        "- 10: Muito boa, precisa e completa\n\n"
        "Responda com **apenas** um número inteiro entre 0 e 10."
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Você é um avaliador que responde apenas com um número inteiro de 0 a 10."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
            stream=False
        )

        score_text = response.choices[0].message.content.strip()

        try:
            score = int(score_text)
            return max(0, min(10, score))  # Clamp to [0, 10]
        except ValueError:
            print(f"[Parsing Error] Could not parse response '{score_text}' as an integer.")
            return 5

    except Exception as e:
        print(f"[LLM Error] {e}")
        return 5


def get_response_from_llm(query: str, chunks: list[str]) -> tuple[str, int, int]:
    # GERE UMA RESPOSTA COM BASE NAS CHUNKS SELECIONADAS
    
    
    # Format the chunks in a readable way
    formatted_chunks = "\n\n".join([f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(chunks)])

    # Create a prompt to help the LLM generate a response based on the query and chunks
    prompt = (
        "Você é um assistente inteligente que responde perguntas com base exclusivamente nas informações fornecidas abaixo.\n\n"
        f"Consulta do usuário:\n{query}\n\n"
        f"Fontes disponíveis (chunks):\n{formatted_chunks if chunks else 'Nenhuma'}\n\n"
        "Responda de forma clara, objetiva e apenas com base nas fontes."
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Você é um assistente útil e conciso."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        content = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        return content, input_tokens, output_tokens

    except Exception as e:
        print(f"[LLM Error] {e}")
        return "Erro ao gerar resposta.", 0, 0


from joblib import Memory

# Set up a cache directory
memory = Memory(location="./llm_cache", verbose=0)

@memory.cache
def cached_evaluation_from_llm(query: str, chunks: list[str], answer: str) -> float:
    return get_evaluation_from_llm(query, chunks, answer)


def create_evaluation_prompt(query: str, ground_truth: str, generated_answer: str) -> str:
    return f"""You are a helpful evaluator. Given a user question, a correct answer, and a model-generated answer, your task is to judge whether the generated answer is correct.

    Here are the possible judgments:

    - "the answer is correct": if the generated answer correctly matches the ground truth in factual content and answers the question.
    - "the answer is wrong": if the generated answer contains factual errors or contradicts the ground truth.
    - "information not provided": if the generated answer is vague, unrelated, or avoids answering the question.

    Respond with exactly one of the options above. Do not explain.

    Question: {query}

    Correct Answer: {ground_truth}

    Generated Answer: {generated_answer}

    Your judgment:"""


def call_llm(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an evaluator that only responds with one of the following: 'the answer is correct', 'the answer is wrong', or 'information not provided'."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            stream=False
        )

        result = response.choices[0].message.content.strip().lower()

        # Normalize the response
        valid_responses = {
            "the answer is correct",
            "the answer is wrong",
            "information not provided"
        }

        if result in valid_responses:
            return result
        else:
            print(f"[Warning] Unexpected response: {result}")
            return "information not provided"

    except Exception as e:
        print(f"[LLM Error] {e}")
        return "information not provided"
