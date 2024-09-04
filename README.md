# Huggingface checkpoints

Best performance checkpoints:

finetune: bs_16_lr_3e-4_wd_0_epoch_10: https://huggingface.co/JohnnaLiu/CS6741qualifier/resolve/main/ft_experiments/bs_64_lr_3e-4_wd_0_epoch_10/best_model.pt

scratch: bs_16_lr_3e-4_wd_0_epoch_10: https://huggingface.co/JohnnaLiu/CS6741qualifier/resolve/main/scr_experiments/bs_16_lr_3e-4_wd_0_epoch_10/best_model.pt

Other checkpoints (bs16/32/128, lr 3e-4/3e-5, epoch 10/20/30/50): https://huggingface.co/JohnnaLiu/CS6741qualifier/




# Assignment 4

## Environment

It's highly recommended to use a virtual environment (e.g. conda, venv) for this assignment.

Example of virtual environment creation using conda:
```
conda create -n env_name python=3.10
conda activate env_name
python -m pip install -r requirements.txt
```

## Evaluation commands

If you have saved predicted SQL queries and associated database records, you can compute F1 scores using:
```
python evaluate.py
  --predicted_sql results/t5_ft_dev.sql
  --predicted_records records/t5_ft_dev.pkl
  --development_sql data/dev.sql
  --development_records records/ground_truth_dev.pkl
```

## Submission

You need to submit your test SQL queries and their associated SQL records. Please only submit your final files corresponding to the test set.

For SQL queries, ensure that the name of the submission files (in the `results/` subfolder) are:
- `{t5_ft, ft_scr, gemma}_test.sql`

For database records, ensure that the name of the submission files (in the `records/` subfolder) are:
- `{t5_ft, ft_scr, gemma}_test.pkl`

Note that the predictions in each line of the .sql file or in each index of the list within the .pkl file must match each natural language query in 'data/test.nl' in the order they appear.

For the LLM, even if you experimented with both models, you should submit only one `.sql` file and one `.pkl` file, corresponding to the model of your choice. Do not submit separate result files for each model.
