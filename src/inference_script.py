from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader
import argparse
import json
import torch
from tqdm import tqdm
import pandas as pd
from typing import List

class SequenceClassificationDataset(Dataset):
	def __init__(self, x, tokenizer):
		self.examples = x
		self.tokenizer = tokenizer
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

	def __len__(self):
		return len(self.examples)
	def __getitem__(self, idx):
		return self.examples[idx]
	def collate_fn(self, batch):
		model_inputs = self.tokenizer([i for i in batch], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
		return {"model_inputs": model_inputs}


def infer(sentences: List[str], model_name='climatebert-environmental-claims'):
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# load model and tokenizer
	try:
		tokenizer = AutoTokenizer.from_pretrained(model_name)
	except:
		tokenizer = AutoTokenizer.from_pretrained("roberta-base")

	model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

				
	predict_dataset = SequenceClassificationDataset(sentences, tokenizer)

	outputs = []
	probs = []
	with torch.no_grad():
		model.eval()
		for batch in tqdm(DataLoader(predict_dataset, batch_size=32, collate_fn=predict_dataset.collate_fn)):
			output = model(**batch["model_inputs"])
			logits = output.logits
			outputs.extend(logits.argmax(dim=1).tolist())
			probs.extend(logits.softmax(dim=1)[:,1].tolist())
	

	return pd.DataFrame(list(zip(sentences, outputs, probs)), columns=["sentence", "classification", "probability"])


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', type=str, default='climatebert-environmental-claims')
	parser.add_argument('--outfile_name', type=str, default='environmental_claim_predictions.csv')
	parser.add_argument('--filename', type=str, default="data/test.jsonl")
	args = parser.parse_args()

	# load sentences to predict
	if args.filename.endswith("jsonl"):
		df = pd.read_json(args.filename, lines = True)
		if "text" in df.columns:
			sentences = df.text.tolist()
		elif "sentences" in df.columns:
			sentences = df.text.tolist()
	elif args.filename.endswith(".txt"):
		with open(args.filename) as f:
			sentences = [i for i in f]

	# Infer 
	df = infer(sentences, args.model_name)
	
	# save to outfile
	df.to_csv(args.outfile_name, index=False)

