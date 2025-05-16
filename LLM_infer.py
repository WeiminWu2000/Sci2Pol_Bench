import argparse
import json
import os
import time
import tqdm
from openai import OpenAI
from google import genai
import anthropic

# --- Parse CLI arguments ---
parser = argparse.ArgumentParser(description='Model inference on Sci2Pol-Bench with specified model and task.')
parser.add_argument('--model', type=str, required=True, default='google/gemma-3-1b-it:free', help='Model name (e.g., gpt-4o, o3)')
parser.add_argument('--task', type=str, required=True, help='Task name (e.g., task19)')
parser.add_argument('--dataset_folder', type=str, required=False, default='./sci2pol_data', help='Dataset folder path')
parser.add_argument('--output_folder', type=str, required=False, default='./sci2pol_results', help='Output folder path')
args = parser.parse_args()

model = args.model
task = args.task
dataset_folder = args.dataset_folder
output_dir = args.output_folder

# --- Initialize client ---
'''
We use different API for different models.

OpenRouter: 
Llama-3.1-8b-instruct, Llama-3.3-70b-instruct, Llama-4-maverick,
Mistral-large-2411, Qwen3-8b, Qwen3-235b-a22b,
Deepseek-chat-v3-0324, Deepseek-r1, 
Gemma-3-12b-it, Gemma-3-27b-it

For Grok-3-beta, GPT-4o, Gemini-2.5-pro, Claude-3-7-sonnet, we use their own specific API.
'''

# For OpenRouter API
if model in ['meta-llama/llama-3.1-8b-instruct', 'meta-llama/llama-3.3-70b-instruct', 'meta-llama/llama-4-maverick', 'mistralai/mistral-large-2411', 'qwen/qwen-3-8b', 'qwen/qwen-3-235b-a22b', 'deepseek/deepseek-chat-v3-0324', 'deepseek/deepseek-r1', 'google/gemma-3-12b-it', 'google/gemma-3-27b-it']:
    client = OpenAI(
        base_url='https://openrouter.ai/api/v1',    
        api_key='OpenRouter API key' # Replace with your OpenRouter API key
    )
# For Grok-3-beta, GPT-4o, Gemini-2.5-pro, Claude-3-7-sonnet
elif model == 'grok/grok-3-beta':
    client = OpenAI(
        base_url='https://api.x.ai/v1',
        api_key='XAI API key' # Replace with your XAI API key
    )
elif model == 'gpt/gpt-4o':
    client = OpenAI(
        api_key='OpenAI API key' # Replace with your OpenAI API key
    )
elif model == 'google/gemini-2.5-pro-preview-05-06':
    client = genai.Client(api_key='Gemini API key') # Replace with your Gemini API key
elif model == 'anthropic/claude-3-7-sonnet':
    client = anthropic.Anthropic(
        api_key='Anthropic API key' # Replace with your Anthropic API key
    )


# --- File paths ---
data_load_path = os.path.join(dataset_folder, f'{task}.jsonl')
output_dir = os.path.join(output_dir, model)
os.makedirs(output_dir, exist_ok=True)
output_save_path = os.path.join(output_dir, f'{task}_response.jsonl')

# --- Load input data ---
with open(data_load_path, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
    
# --- Model inference ---
with open(output_save_path, 'w', encoding='utf-8') as fout:
    for sample in tqdm.tqdm(data):
        try:
            if model in ['meta-llama/llama-3.1-8b-instruct', 'meta-llama/llama-3.3-70b-instruct', 'meta-llama/llama-4-maverick', 'mistralai/mistral-large-2411', 'qwen/qwen-3-8b', 'qwen/qwen-3-235b-a22b', 'deepseek/deepseek-chat-v3-0324', 'deepseek/deepseek-r1', 'google/gemma-3-12b-it', 'google/gemma-3-27b-it']:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{'role': 'user', 'content': sample['query']}]
                )
                answer = resp.choices[0].message.content.strip()
            elif model == 'grok/grok-3-beta':
                completion = client.chat.completions.create(
                    model='grok-3-beta',
                    messages=[{'role': 'user', 'content': sample['query']}],
                )
                answer = completion.choices[0].message.content
            elif model == 'gpt/gpt-4o':
                resp = client.chat.completions.create(
                    model='gpt-4o',
                    messages=[{'role': 'user', 'content': sample['query']}]
                )
                answer = resp.choices[0].message.content.strip()
            elif model == 'google/gemini-2.5-pro-preview-05-06':
                response = client.models.generate_content(
                    model="gemini-2.5-pro-preview-05-06",
                    contents=[sample["query"]],
                )
                answer = response.text
            elif model == 'anthropic/claude-3-7-sonnet':
                message = client.messages.create(
                    model='claude-3-7-sonnet-20250219',
                    max_tokens=4096,
                    messages=[{'role': 'user', 'content': sample['query']}],
                )
                answer = message.content[0].text
        except Exception as e:
            answer = f'ERROR: {e}'

        record = {
            'idx':      sample['id'],
            'expected': sample['answer'],
            'response': answer
        }
        fout.write(json.dumps(record, ensure_ascii=False) + '\n')
        fout.flush()
        time.sleep(0.5)  # Adjust sleep time as needed