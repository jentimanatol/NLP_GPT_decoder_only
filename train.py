
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import CharTokenizer,LMDataset
from model import GPTDecoderOnly

print("Loading dataset...")

with open("data/tinyshakespeare.txt","r",encoding="utf-8") as f:
    text=f.read()

tokenizer=CharTokenizer()
tokenizer.fit(text)

tokens=tokenizer.encode(text)

block_size=128

train_tokens=tokens[:int(len(tokens)*0.9)]
val_tokens=tokens[int(len(tokens)*0.9):]

train_ds=LMDataset(train_tokens,block_size)
train_loader=DataLoader(train_ds,batch_size=16,shuffle=True)

device="cuda" if torch.cuda.is_available() else "cpu"

model=GPTDecoderOnly(
    vocab_size=tokenizer.vocab_size,
    pad_id=0
).to(device)

opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
loss_fn=nn.CrossEntropyLoss()

epochs=5

for e in range(epochs):

    model.train()
    total=0

    for x,y in train_loader:

        x,y=x.to(device),y.to(device)

        opt.zero_grad()

        logits=model(x)

        B,T,V=logits.shape

        loss=loss_fn(logits.view(B*T,V),y.view(B*T))

        loss.backward()

        opt.step()

        total+=loss.item()

    print(f"Epoch {e+1} Loss {total/len(train_loader)}")

torch.save(model.state_dict(),"gpt_model.pt")

print("Training finished")
