from text_dataset import TextDataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from model import BertMixUp

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

from sklearn.metrics import f1_score
from transformers import BertModel, BertTokenizer
import numpy as np
import os
import argparse
import json
import dill


class TrainModel():
    def __init__(self, config):
        self.num_epochs = config['num_epochs']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.output_dir = config['output_dir']
        self.num_classes = config['num_classes']
        self.mixup_type = config['mixup_type']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = config['alpha']
        self.model_path = config['model_path']
        self.dataset_name = config['dataset_name']
        self.p = config['p']
        self.max_len = config['max_len']
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', self.output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.get_dataloaders(
            config['batch_size'], self.tokenizer, config['dataset_name'], self.max_len
        )
        self.pretrained_model = BertModel.from_pretrained(self.model_path)
        self.model = BertMixUp(self.pretrained_model, self.num_classes, self.mixup_type)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.6)
        self.best_f1 = 0

    def get_dataloaders(self, batch_size, tokenizer, dataset_name, max_len=128):
        dataset = load_dataset(dataset_name)
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']

        val_texts = dataset['validation']['text']
        val_labels = dataset['validation']['label']

        test_texts = dataset['test']['text']
        test_labels = dataset['test']['label']

        train_dataset = TextDataset(train_texts, tokenizer, max_len, train_labels)
        val_dataset = TextDataset(val_texts, tokenizer, max_len, val_labels)
        test_dataset = TextDataset(test_texts, tokenizer, max_len, test_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader    
        
    def train(self):
        print(self.device)
        metrics = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
        test_f1 = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            all_preds = []
            all_labels = []

            for i, batch in enumerate(self.train_dataloader):
                #if i % 20:
                #    print(i)
                input_ids, attention_mask, labels = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device), batch['labels'].to(self.device)

                if self.mixup_type in ['embedding', 'sentences']:
                    lam = torch.tensor([np.random.beta(self.alpha, self.alpha)]).to(self.device)
                    n = len(input_ids)
                    per = torch.randperm(n - int(n*self.p)).to(self.device)
                    indices = torch.cat([per, torch.arange(n - int(n*self.p), n).to(self.device)])
                    input_ids2, attention_mask2, labels2 = input_ids[indices], attention_mask[indices], labels[indices]
                    outputs = self.model(input_ids, attention_mask, lam, input_ids2, attention_mask2)
                    loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(outputs, labels2)
                else:
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
            
            #if epoch % 2 == 0:
            #    self.scheduler.step()

            train_f1 = f1_score(all_labels, all_preds)
            val_loss, val_f1 = self.evaluate(self.val_dataloader)
            print(' Epoch: ', epoch + 1, '/', self.num_epochs, 
                  ' Loss: ', round(total_loss / len(self.train_dataloader), 5), 
                  ' Train F1: ', round(train_f1, 5),
                  ' Val F1: ', round(val_f1, 5))

            metrics['train_loss'].append(total_loss / len(self.train_dataloader))
            metrics['val_loss'].append(val_loss)
            metrics['train_f1'].append(train_f1)
            metrics['val_f1'].append(val_f1)
            
            if val_f1 > self.best_f1:
                test_f1 = self.evaluate(self.test_dataloader)[1]
                print('Update best_test_f1: ', test_f1)
                self.best_f1 = val_f1
                self.save_model()
                
            self.save_metrics(metrics)

    def evaluate(self, dataloader):
        self.model.eval()
        loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device), batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss += self.criterion(outputs, labels).item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                correct += (preds == labels).sum().item()
                total += len(labels)
        
        f1 = f1_score(all_labels, all_preds)
        return loss / len(dataloader), f1

    def save_model(self):
        self.model.to('cpu')
        model_path = os.path.join(self.output_dir, 'best_model.pth')
        with open(model_path, 'wb') as f:
            dill.dump(self.model, f)
        self.model.to(self.device)
        tokenizer_path = os.path.join(self.output_dir, 'tokenizer')
        self.tokenizer.save_pretrained(tokenizer_path)
    
    def save_metrics(self, metrics):
        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixup_type", type=str, required=True)
    args = parser.parse_args()

    config_file = os.path.join("configs", f"{args.mixup_type}.json")
    
    with open(config_file, 'r') as f:
        config = json.load(f)

    train_bert = TrainModel(config)
    train_bert.train()
