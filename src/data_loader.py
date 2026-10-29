from datasets import load_dataset

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"

dataset = load_dataset(DATASET_ID)

train_dataset = dataset["train"]

print(len(train_dataset))
print(train_dataset.column_names)
print(train_dataset.features["annotation_id"])
print(len(train_dataset.unique("patient_id")))
print(train_dataset.unique("criterion_type"))
print(train_dataset.unique("expert_eligibility"))
print(train_dataset.unique("gpt4_eligibility"))
