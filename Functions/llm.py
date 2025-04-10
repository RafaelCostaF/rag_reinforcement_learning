from openai import OpenAI
import os
import re
# from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from datetime import datetime
import json



# Setup DeepSeek API
os.environ["OPENAI_API_KEY"] = "sk-00691242394046909d398f327080db20"
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


# def get_eval_from_llm_response(query: str, ground_truth: str, prediction: str, model_name: str = "gpt-4-0125-preview") -> tuple[str, int]:
#     """
#     Consulta uma LLM para avaliar se a `prediction` está correta em relação ao `ground_truth`.

#     Retorna uma tupla:
#         (explanation: str, score: int)
#         Onde score ∈ {0, 1} e -1 para erro de parsing.
#     """
#     client = OpenAI()
#     system_message = INSTRUCTIONS + "\n" + IN_CONTEXT_EXAMPLES

#     # Cria mensagem para o modelo
#     messages = [
#         {"role": "system", "content": system_message},
#         {
#             "role": "user",
#             "content": f"Question: {query}\nGround truth: {ground_truth}\nPrediction: {prediction}"
#         }
#     ]

#     # Chamada à API
#     try:
#         response = client.chat.completions.create(
#             model=model_name,
#             messages=messages,
#             response_format={"type": "json_object"},
#             temperature=0.0,
#         )
#         llm_output = response.choices[0].message.content
#     except Exception as e:
#         print(f"Erro ao chamar LLM: {e}")
#         return "Erro ao consultar LLM", -1

#     # (opcional) salvar log da resposta
#     os.makedirs("api_responses", exist_ok=True)
#     file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
#     with open(os.path.join("api_responses", file_name), "w") as f:
#         json.dump({"messages": messages, "response": llm_output}, f)

#     # Parsing da resposta da LLM
#     matches = re.findall(r"{([^}]*)}", llm_output)
#     text = ""
#     for match in matches:
#         text = "{" + match + "}"

#     try:
#         score_match = re.search(r'"score"\s*:\s*(\d+)', text)
#         if not score_match:
#             return "Erro ao extrair score", -1

#         score = int(score_match.group(1))
#         if score not in [0, 1]:
#             raise ValueError("Score inválido: " + str(score))

#         explanation_match = re.search(r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
#         explanation = explanation_match.group(1) if explanation_match else text

#         return explanation, score

#     except Exception as e:
#         print(f"[Parsing Error] Resposta bruta: {llm_output}")
#         print(f"Erro: {e}")
#         return llm_output, -1


from joblib import Memory

# Set up a cache directory
memory = Memory(location="./llm_cache", verbose=0)

@memory.cache
def cached_evaluation_from_llm(query: str, chunks: list[str], answer: str) -> float:
    return get_evaluation_from_llm(query, chunks, answer)
