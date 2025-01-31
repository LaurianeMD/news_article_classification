import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd



class NewsDataset(Dataset):
    def __init__(self, csv_file, max_length):
        import pandas as pd
        self.df = pd.read_csv(csv_file)
        self.labels = self.df['news_category'].unique()
        self.labels_dict = {label: index for index, label in enumerate(self.labels)}
        self.df['news_category'] = self.df['news_category'].map(self.labels_dict)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        headline_text = self.df.news_headline[index]
        article_text = self.df.news_article[index]
        combined_text = headline_text + " " + article_text
        label = self.df.news_category[index]

        inputs = self.tokenizer(
            combined_text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        labels = torch.tensor(label)
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }

class CustomBert(nn.Module):
    def __init__(self, model_name_or_path="bert-base-uncased", n_classes=2):
        super(CustomBert, self).__init__()
        self.bert_pretrained = BertModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        x = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)
        x = self.classifier(x.pooler_output)
        return x

#Training function
def training_step(model, data_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0

    for data in tqdm(data_loader, total=len(data_loader)):
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        labels = data['labels']

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(output, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(data_loader.dataset)

#Evaluation
def evaluation(model, test_dataloader, loss_fn):
    model.eval()
    correct_predictions = 0
    losses = []

    for data in tqdm(test_dataloader, total=len(test_dataloader)):
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        labels = data['labels']

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        _, pred = output.max(1)
        correct_predictions += torch.sum(pred == labels)

        loss = loss_fn(output, labels)
        losses.append(loss.item())

    return correct_predictions.double() / len(test_dataloader.dataset), np.mean(losses)


#main
if __name__ == "__main__":
    dataset = NewsDataset(csv_file="./inshort_news_data.csv", max_length=100)
    num_classes = len(dataset.labels_dict)

    train_data, test_data = train_test_split(dataset, test_size=0.2)

    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False)

    model = CustomBert(n_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    num_epochs = 3  
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = training_step(model, train_dataloader, loss_fn, optimizer)
        print(f"Train Loss: {train_loss:.4f}")

        val_acc, val_loss = evaluation(model, test_dataloader, loss_fn)
        print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), '/trained_model_.pth')
