from torch.utils.data import DataLoader, Dataset, random_split
import torch
import numpy as np
from transformers import AutoTokenizer, RobertaModel, logging
import json
from torch.nn import TripletMarginLoss

logging.set_verbosity_error()
torch.manual_seed(69) # nice

class Data(Dataset):
    def __init__(self, n = None):
        ## n is a user specified length of dataset
        self.n = n

        ## data downloaded from https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases
        data_path = "quora_duplicates_triplets.jsonl"
        self.data = []
        with open(data_path, "r") as f:
          for line in f:
            triplet = json.loads(line)
            self.data.append(triplet)

    def __len__(self):
        return len(self.data) if self.n is None else self.n

    def __getitem__(self, idx):
        ## returns a tuple of (anchor, positive, negative)
        return self.data[idx]

data = Data()
train_set, test_set, val_set = random_split(data, [0.8, 0.15, 0.05])
batch_size = 32

## test_set skal bruges til at lave top k
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)

## test af dataset
print("Length: ", len(data))
print("Train: ", len(train_set))
print("Val: ", len(val_set))
print("Examples from train_set: ")
print(train_set[1])

# Create a model class with variable output dimension(output_dim)
class sRoberta(torch.nn.Module):
    def __init__(self, output_dim):
        super(sRoberta, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing model with output dimension: {output_dim} on device: {self.device}...")

        roberta_version ="roberta-base"
        self.roberta = RobertaModel.from_pretrained(roberta_version)
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_version) # load pretrained transformer and tokenizer
        self.dense = torch.nn.Linear(768, output_dim)
        self.dropout = torch.nn.Dropout(p = 0.3)
        self.output_dim = output_dim
        self.criterion = TripletMarginLoss(margin=0.05)
        self.name = f"sRoberta_{output_dim}"
        self.to(self.device)

        print("..done!\n")

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids, attention_mask)
        out = out.last_hidden_state.mean(axis=1) ## take an average of the embedding of each word
        out = self.dropout(out)
        out = self.dense(out)

        return out

    def embeddings_from_sentences(self, sentences: list, grad = True):
        ## giv en liste med sætninger og få en tensor med embeddings
        if grad:
            self.train()
            token_and_mask = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
            embedding = self(**token_and_mask)

        else:
            with torch.no_grad():
                self.eval()
                token_and_mask = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
                embedding = self(**token_and_mask).cpu()

        return embedding

    def embeddings_from_batch(self, batch):
        ## giv et batch fra dataloaderen og få embeddings for anchor, pos og neg
        anchor, positive, negative = batch
        anchor_embed = self.embeddings_from_sentences(anchor)
        positive_embed = self.embeddings_from_sentences(positive)
        negative_embed = self.embeddings_from_sentences(negative)

        return anchor_embed, positive_embed, negative_embed

    def test(self, data_loader):
        self.eval()
        loss = 0
        with torch.no_grad():
            for batch in data_loader:
                anchor_embed, positive_embed, negative_embed = self.embeddings_from_batch(batch)
                loss += self.criterion(anchor_embed, positive_embed, negative_embed)

                loss /= len(data_loader)

        self.train()
        return loss.item()

    def train_model(self, epochs, train_loader, val_loader):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-4)
        train_losses = torch.zeros(epochs)
        val_losses = torch.zeros(epochs)
        best_loss = float("inf")

        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()

                anchor_embed, positive_embed, negative_embed = self.embeddings_from_batch(batch)
                loss = self.criterion(anchor_embed, positive_embed, negative_embed)

                loss.backward()
                optimizer.step()
        
                torch.cuda.empty_cache()

            ## test på validation data
            val_loss = self.test(val_loader)

            train_losses[epoch] = loss.item()
            val_losses[epoch] = val_loss

            print(f"Epoch {epoch + 1}, train loss: {loss:.4f}, validation loss: {val_loss:.4f}\n")

            if val_loss < best_loss:
                best_loss = val_loss
                print(f"Best validation loss: {best_loss:.4f}")
                print("Saving best model...")
                self.save_model()

        path = f"data/{self.name}_train_losses.pt"
        torch.save(train_losses, path)
        torch.save(val_losses, path)
        print(f"\nFinished training. Saved train- and validation losses at {path}")

    def save_model(self):
        path = f"models/{self.name}.pt"
        torch.save(self.state_dict(), path)
        print("Model saved to {}\n".format(path))

    def load_model(self):
        path = f"models/{self.name}.pt"
        self.load_state_dict(torch.load(path))
        print("Model loaded from {}".format(path))

class sRobertaBase(sRoberta):
  def __init__(self):
    super(sRobertaBase, self).__init__(1) ## initialize random dense network since it wont be used
    self.dense = None

  def forward(self, input_ids, attention_mask):
    out = self.roberta(input_ids, attention_mask)
    out = out.last_hidden_state.mean(axis=1)

    return out

embed_sizes = [8, 64, 128, 512, 768, 1024, 2048]

num_epochs = 10
for embed_size in embed_sizes:
    print(f"Start training model with embedding size: {embed_size}\n")
    model = sRoberta(embed_size)
    model.train_model(num_epochs, train_loader, val_loader)  ## modellen bliver nu gemt i trænings-loopet, når den bedst valideringsloss bliver opnået

print(f"Start training base model\n")
model = sRobertaBase()
model.train_model(num_epochs, train_loader, val_loader)