import os
import json
import re


def read_schema(schema_path):
    '''
    Read the .schema file
    '''
    # TODO
    schema = {}
    with open(schema_path, 'r') as file:
        content = file.read()
        schema = json.loads(content)
    return schema


def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response
    '''
    # TODO
    # pattern = r"SQL: (.*?);"
    # match = re.search(pattern, response, re.IGNORECASE)
    # if match:
    #     return match.group(1).strip()
    # else:
    #     # Handle the case where no SQL is found
    #     return ""
    try:
        code = response.split('Output:\n')[-1].split('<eos>')[0].split('```')
        if code[0] != "":
            return code[0]
        else:
            return code[1]
    except:
        return "N/A"
    # return response

def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")