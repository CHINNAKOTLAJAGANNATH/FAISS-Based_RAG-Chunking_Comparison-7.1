import openai

def generate_answer(context, query):
    openai.api_key = "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"  # Replace with your key

    prompt = f"""Use the context below to answer the question.
Context: {context}
Q: {query}
A:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response['choices'][0]['message']['content']
