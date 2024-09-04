import os, argparse, random
from tqdm import tqdm

import torch
from transformers import GemmaTokenizerFast, GemmaForCausalLM
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from utils import *
from prompting_utils import read_schema, extract_sql_query, save_logs
from load_data import load_prompting_data

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # you can add mps


def get_args():
    '''
    Arguments for prompting. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(
        description='Text-to-SQL experiments with prompting.')

    parser.add_argument('-s', '--shot', type=int, default=5,
                        help='Number of examples for k-shot learning (0 for zero-shot)')
    parser.add_argument('-p', '--ptype', type=int, default=0,
                        help='Prompt type')
    parser.add_argument('-m', '--model', type=str, default='gemma',
                        help='Model to use for prompting: gemma (gemma-1.1-2b-it) or codegemma (codegemma-7b-it)')
    parser.add_argument('-q', '--quantization', action='store_true',
                        help='Use a quantized version of the model (e.g. 4bits)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to help reproducibility')
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")
    args = parser.parse_args()
    return args


def create_prompt(sentence, k, schema, train_x, train_y):
    '''
    Function for creating a prompt for zero or few-shot prompting.

    Add/modify the arguments as needed.

    Inputs:
        * sentence (str): A text string
        * k (int): Number of examples in k-shot prompting
    '''
    prompt = f"Given the database with {', '.join(schema.keys())}, generate the SQL query for: {sentence}"
    for key in ["types", "ents", "defaults", "links"]:
        prompt += f"\nTable {key} has columns {', '.join(schema[key])}."
    # for i in range(k):
    #     prompt += f"\nExample {i+1}: Input: " + train_x[i] + ". Output: " + train_y[i]
    # prompt = "Generate a sql query given the input, your output format should be the following.\n"
    # prompt += "Output:\n{your sql query}\n"
    # prompt += "Here are some examples.\n"
    prompt = "You are a SQL expert.\n"
    prompt += "The database has the following tables.\n"
    prompt += f"{', '.join(schema.keys())}\n"
    for key in schema.keys():
        prompt += f"Table '{key}' has the following columns.\n"
        prompt += f"{', '.join(schema[key])}\n"
    prompt += "Now generate the SQL query for the given input. Here are some examples.\n"
    example_idxs = torch.randint(0, len(train_x) - 1, (k,))
    for i in example_idxs:
        prompt += f"Input:\n{train_x[i]}\nOutput:\n```{train_y[i]}```\n"
    prompt += "Now answer the following.\n"
    prompt += f"Input:\n{sentence}\nOutput:\n"
    return prompt
    # TODO


def exp_kshot(tokenizer, model, inputs, k, schema, train_x, train_y):
    '''
    k-shot prompting experiments using the provided model and tokenizer. 
    This function generates SQL queries from text prompts and evaluates their accuracy.

    Add/modify the arguments and code as needed.

    Inputs:
        * tokenizer
        * model
        * inputs (List[str]): A list of text strings
        * k (int): Number of examples in k-shot prompting
    '''
    raw_outputs = []
    extracted_queries = []

    for i, sentence in enumerate(tqdm(inputs)):
        prompt = create_prompt(sentence, k, schema=schema, train_x=train_x, train_y=train_y) # Looking at the prompt may also help
        # breakpoint()
        input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**input_ids, max_new_tokens=512) # You should set MAX_NEW_TOKENS
        response = tokenizer.decode(outputs[0]) # How does the response look like? You may need to parse it
        raw_outputs.append(response)

        # Extract the SQL query
        extracted_query = extract_sql_query(response)
        # breakpoint()
        extracted_queries.append(extracted_query)
    return raw_outputs, extracted_queries


def eval_outputs(eval_x, eval_y, gt_sql_path, model_sql_path, gt_record_path, model_record_path):
    '''
    Evaluate the outputs of the model by computing the metrics.

    Add/modify the arguments and code as needed.
    '''
    # TODO
    sql_em, record_em, record_f1, error_msgs = compute_metrics(gt_sql_path, model_sql_path, gt_record_path, model_record_path)
    error_rate = 0
    for msg in error_msgs:
        if msg:
            error_rate += 1
    error_rate = error_rate / len(error_msgs)
    return sql_em, record_em, record_f1, error_msgs, error_rate


def initialize_model_and_tokenizer(model_name, to_quantize=False):
    '''
    Args:
        * model_name (str): Model name ("gemma" or "codegemma").
        * to_quantize (bool): Use a quantized version of the model (e.g. 4bits)
    
    To access to the model on HuggingFace, you need to log in and review the 
    conditions and access the model's content.
    '''
    if model_name == "gemma":
        model_id = "google/gemma-1.1-2b-it"
        tokenizer = GemmaTokenizerFast.from_pretrained(model_id)
        # Native weights exported in bfloat16 precision, but you can use a different precision if needed
        model = GemmaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, 
        ).to(DEVICE)
    elif model_name == "codegemma":
        model_id = "google/codegemma-7b-it"
        tokenizer = GemmaTokenizer.from_pretrained(model_id)
        if to_quantize:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", # 4-bit quantization
            )
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                        torch_dtype=torch.bfloat16,
                                                        config=nf4_config).to(DEVICE)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                        torch_dtype=torch.bfloat16).to(DEVICE)
    return tokenizer, model


def main():
    '''
    Note: this code serves as a basic template for the prompting task. You can but 
    are not required to use this pipeline.
    You can design your own pipeline, and you can also modify the code below.
    '''
    args = get_args()
    shot = args.shot
    ptype = args.ptype
    model_name = args.model
    to_quantize = args.quantization
    experiment_name = args.experiment_name

    set_random_seeds(args.seed)

    data_folder = 'data'
    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data(data_folder)

    # Model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer(model_name, to_quantize)

    schema_path = 'data/flight_database.schema'
    schema = read_schema(schema_path)

    for eval_split in ["dev"]:
        eval_x, eval_y = (dev_x, dev_y) if eval_split == "dev" else (test_x, None)

        raw_outputs, extracted_queries = exp_kshot(tokenizer, model, eval_x, shot, schema, train_x, train_y)
        print('queries done')
        recs, err_msg = compute_records(extracted_queries)
        error = 0
        for msg in err_msg:
            if msg:
                error += 1
        error_rate = error / len(err_msg)
        print(f'error rate is {error_rate}')
        gt_sql_path = os.path.join(f'data/{eval_split}.sql')
        gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
        gt_qs, gt_records, _ = load_queries_and_records(gt_sql_path, gt_record_path)

        sql_em = compute_sql_exact_match(gt_qs, extracted_queries)
        record_em = compute_record_exact_match(gt_records, recs)
        record_f1 = compute_record_F1(gt_records, recs)
        print(f"Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")

    # raw_outputs_test, extracted_queries_test = exp_kshot(tokenizer, model, test_x, shot, schema, train_x, train_y)
    # recs, err_msg = compute_records(extracted_queries)
    # save_queries_and_records(extracted_queries_test, "data/llm_test.sql", "records/llm_test.pkl")
    # You can add any post-processing if needed


if __name__ == "__main__":
    main()