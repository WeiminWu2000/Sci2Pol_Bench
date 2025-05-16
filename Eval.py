import json
from sklearn.metrics import f1_score
import argparse
import os
import re
import time
import tqdm
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from google import genai

# --- Parse CLI arguments ---
parser = argparse.ArgumentParser(description='Inference output evaluation with specified model and task.')
parser.add_argument('--model', type=str, required=True, default='google/gemma-3-1b-it:free', help='Model name (e.g., \'gpt-4o\', \'o3\')')
parser.add_argument('--dataset_folder', type=str, required=False, default='./sci2pol_data', help='Dataset folder path')
parser.add_argument('--response_folder', type=str, required=False, default='./sci2pol_results', help='Model response folder path')
parser.add_argument('--output_folder', type=str, required=False, default='./sci2pol_eval_results', help='Evaluation results folder path')
args = parser.parse_args()

model = args.model
dataset_folder = args.dataset_folder
response_folder = args.response_folder
output_folder = args.output_folder
client = genai.Client(api_key='Gemini API key')  # Replace with your Gemini API key

def LLM_prompt_sum(task_type, source_passage, summary):
    prompt = f'''You are a scientific expert evaluating a summary that restates a **{task_type}** described in a scientific paper and uses the policy-brief style sentences.
    
You are a strict and critical evaluator of summaries. Evaluate the summary on the following dimensions using a 1-5 scale (1 = very poor, 5 = excellent). Be conservative in your judgments: do not give high scores unless the summary is genuinely outstanding.

(1) Clarity: whether the summary is reader-friendly and expresses ideas clearly  
(2) Accuracy: whether the summary contains the same information as the source document  
(3) Coverage: how well the summary covers the important information from the source document  
(4) Overall quality: how good the summary is overall at representing the source document; a good summary is a shorter piece of text that has the essence of the original and tries to convey the same information as the source document

Return only a JSON object in this format:

{{
  'clarity': <1-5>,
  'accuracy': <1-5>,
  'coverage': <1-5>,
  'overall_quality': <1-5>
}}

---

Source Passage:
{source_passage}

Summary:
{summary}
'''
    return prompt

for task in ['task1', 'task2', 'task3', 'task4', 'task5', 'task6', 'task7', 'task8', 'task9', 'task10', 'task11', 'task12', 'task13', 'task14', 'task15', 'task16', 'task17', 'task18', 'task19']:
    load_path = os.path.join(response_folder, f'{model}/{task}_response.jsonl')
    with open(load_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        
    os.makedirs(os.path.join(output_folder, model), exist_ok=True)
    save_path = os.path.join(output_folder, f'{model}/{task}.jsonl')
    
    if task in ['task1', 'task2', 'task3', 'task4', 'task5', 'task6', 'task17', 'task18', 'task19']:
        
        if task == 'task18':
            verdict_pattern = re.compile(r'"verdict"\s*:\s*"([^"]+)"')
            for i in range(len(data)):
                sample = data[i]
                expected_str = sample.get('expected', '')
                m1 = verdict_pattern.search(expected_str)
                expected_verdict = m1.group(1) if m1 else ''
                response_str = sample.get('response', '')
                m2 = verdict_pattern.search(response_str)
                response_verdict = m2.group(1) if m2 else ''
                data[i] = {
                    'idx': sample.get('idx'),
                    'expected': expected_verdict,
                    'response': response_verdict
                }
            
        y_true = [d['expected'] for d in data]
        y_pred = [d['response'] for d in data]

        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({'f1_micro': f1_micro, 'f1_macro': f1_macro}, f)
            
    elif task in ['task11', 'task12', 'task13', 'task14', 'task15', 'task16']:
        
        refs = [d['expected'] for d in data[:-1]]
        preds = [d['response'] for d in data[:-1]]

        P, R, F1 = bert_score(preds, refs, lang='en', verbose=True)
        bertscore_precision = float(P.mean())
        bertscore_recall = float(R.mean())
        bertscore_f1 = float(F1.mean())

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_f, rouge2_f, rougeL_f = [], [], []

        print('\n=== ROUGE Scores (per sample) ===')
        for ref, pred in tqdm.tqdm(zip(refs, preds), total=len(data)):
            scores = scorer.score(ref, pred)
            rouge1_f.append(scores['rouge1'].fmeasure)
            rouge2_f.append(scores['rouge2'].fmeasure)
            rougeL_f.append(scores['rougeL'].fmeasure)

        rouge1_f1 = float(sum(rouge1_f) / len(rouge1_f))
        rouge2_f1 = float(sum(rouge2_f) / len(rouge2_f))
        rougeL_f1 = float(sum(rougeL_f) / len(rougeL_f))

        results = {
            'bertscore_precision': bertscore_precision,
            'bertscore_recall': bertscore_recall,
            'bertscore_f1': bertscore_f1,
            'rouge1_f1': rouge1_f1,
            'rouge2_f1': rouge2_f1,
            'rougeL_f1': rougeL_f1
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
    elif task in ['task7', 'task8', 'task9', 'task10']:
        
        initial_data_path = os.path.join(dataset_folder, f'{task}.jsonl')
        with open(initial_data_path, 'r', encoding='utf-8') as f:
            initial_data = [json.loads(line) for line in f]
        
        initial_query = {}
        for d in initial_data:
            match = re.search(r'Scientific Text:\s*(.+?)(?:\n\s*\n|\Z)', d['query'], re.DOTALL)
            initial_query[d['id']] = match.group(1).strip()
    
        LLM_judge_save_path = os.path.join(output_folder, f'{model}/{task}_LLM_judge_response.jsonl')
        
        preds_query = []
        for d in data:
            id = d['idx']
            pred = d['response']
            query = initial_query[id]
            preds_query.append({'id': id, 'query': query, 'pred': pred})
            
        res = []
        with open(LLM_judge_save_path, 'w', encoding='utf-8') as fout:
            for i in tqdm.tqdm(range(len(preds_query))):
                sample = preds_query[i]
                Id = sample['id']
                if task == 'task7':
                    task_type = 'Policy Problem'
                elif task == 'task8':
                    task_type = 'Scientific Research Findings'
                elif task == 'task9':
                    task_type = 'Scientific Research Study Methods'
                elif task == 'task10':
                    task_type = 'Policy Implications'
                request_prompt = LLM_prompt_sum(task_type, sample['query'], sample['pred']) 
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-pro-preview-05-06',
                        contents=[request_prompt],
                    )
                    answer = response.text
                except Exception as e:
                    answer = f'ERROR: {e}' 

                record = {
                    'idx':      Id,
                    'response': answer
                }
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                fout.flush()
                time.sleep(0.5)  
                
        results = []
        with open(LLM_judge_save_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                idx = sample['idx']
                content_str = sample['response']

                if content_str.startswith('```json'):
                    content_str = content_str.strip('` \n')[len('json'):].strip()

                try:
                    parsed_scores = json.loads(content_str)
                    results.append({
                        'id': idx,
                        **parsed_scores
                    })
                except json.JSONDecodeError as e:
                    print(f'[Error] Failed to parse JSON for idx {idx}: {e}')
            
        scores = []
        for item in results:
            total_score = (item['clarity'] + item['accuracy'] + item['coverage'] + item['overall_quality']) * 5
            scores.append(total_score)

        dataset_score = sum(scores) / len(scores)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({'LLM judge score': dataset_score}, f, indent=2)