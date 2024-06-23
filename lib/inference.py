from text_dataset import TextDataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from model import BertMixUp
from transformers import BertModel, BertTokenizer
from sklearn.metrics import f1_score
from typing import List
import dill
import torch
import argparse
import json
import pandas as pd
import os


class InferenceModel():
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.tokenizer_path = os.path.join(self.current_dir, '..', os.path.join(config['output_dir']) , 'tokenizer')
        self.model_path = os.path.join(self.current_dir, '..', os.path.join(config['output_dir']) , 'best_model.pth')	    
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path, local_files_only=True)
        self.model = self.load_model()
        self.max_len = config['max_len']

    def predict(self, text : List[str]) -> List[float]:
        self.model.eval()
        dataset = TextDataset(text, self.tokenizer, self.max_len)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for batch in  dataloader:
                input_ids, attention_mask = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
        return all_preds

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            model = dill.load(f)
        model = model.to(self.device)
        return model	
    
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixup_type", type=str, required=True)
    args = parser.parse_args()

    config_file = os.path.join("configs", f"{args.mixup_type}.json")
    print(config_file)   
    with open(config_file, 'r') as f:
        config = json.load(f)

    dataset = load_dataset(config['dataset_name'] )
    text = dataset['test']['text']
    labels = dataset['test']['label']

    inference_model = InferenceModel(config)
    preds = inference_model.predict(text)

    f1 = f1_score(labels, preds)
    print('Test F1: ', f1)

    submit_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', config['submit_file'])
    df = pd.DataFrame({'prediction': preds})
    df.to_csv(submit_path , index=False)