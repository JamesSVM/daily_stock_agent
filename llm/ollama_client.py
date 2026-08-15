import ollama

def analyze(prompt):
    response = ollama.chat(
        model='qwen3:7b',
        messages=[
            {'role': 'system', 'content': '你是一位台股短波段交易研究員'},
            {'role': 'user', 'content': prompt}
        ]
    )

    return response['message']['content']

print(analyze('分析今天的交易風險'))