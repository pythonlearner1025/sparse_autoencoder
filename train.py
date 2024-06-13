import torch
import wandb
from torch import nn, optim
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, AutoProcessor
from datasets import load_dataset
from sparse_autoencoder.model import Autoencoder, TopK   # Assuming you have a SAE module implemented in another library
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL import UnidentifiedImageError
import requests
from requests import RequestException

# Initialize tokenizer and text encoder
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

dataset = load_dataset("Thouph/Laion_aesthetics_5plus_1024_33M_csv")

# Freeze all parameters of the text encoder
for param in text_encoder.parameters():
    param.requires_grad = False

# Get the hidden dimension size
hidden_dim_size = text_encoder.config.hidden_size
print(f"The hidden dimension size of clip-vit-base-patch32 is {hidden_dim_size}")
expansion_f = 128
num_latents = hidden_dim_size * expansion_f
# Initialize SAE module
sae = Autoencoder(
    num_latents,
    hidden_dim_size, 
    activation=TopK(num_latents//10)
)

# Define optimizer and loss function
optimizer = optim.Adam(sae.parameters(), lr=1e-5)
criterion = nn.MSELoss()

# TODO SAE
# Move models to GPU to measure VRAM usage
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
text_encoder.to(device)
sae.to(device)
max_toks = 77 # the max size of positional embedder is 77 tokens

# Custom Dataset class to extract "TEXT" data
class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["TEXT"]
        img = self.dataset[idx]["URL"] 
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=max_toks)
        return inputs, img

class ImageDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.dead = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        while True:
            try:
                url = self.dataset[idx]["URL"]
                img = Image.open(requests.get(url, stream=True).raw)
                inputs = processor(images=img, return_tensor='pt')
                return inputs
            except (RequestException, UnidentifiedImageError):
                self.dead += 1
                print(f'dead urls: {self.dead}')
                print(url)
                idx = (idx + 1) % len(self.dataset)  # Move to the next sample
# Initialize dataset and dataloader

train_dataset = ImageDataset(dataset['train'], tokenizer)
test_dataset = TextDataset(dataset['train'], tokenizer)

# Collate function to stack batch data properly
def collate_fn(batch):
    input_ids = torch.stack(tuple([torch.tensor(item["input_ids"]) for item,img in batch]))
    attention_mask = torch.stack(tuple([torch.tensor(item["attention_mask"]) for item,img in batch]))
    imgs = [img for _,img in batch]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "imgs": imgs}
# VRAM per batch = (200 * bs * dim * bytes_per_dim)/(1024**2)
# DataLoader with the adjusted batch size
train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=1)
test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=1, collate_fn=collate_fn)

from tqdm import tqdm
from utils import * 

print('setting db up...')
setup()
log = 0 
if log:
    wandb.init(
        project="Stable Diffusion SAE",
        config={
        "learning_rate": 1e-5,
        "architecture": "SD",
        "dataset": "LAION_aesthetics_5plus_1024_33M_csv",
        "epochs": 10,
        "batch_size": 1024,
        "max_toks": 77
        }
    )
def validate(sae=None):
    target_layer = -2
    target_token = 0
    print("validating from ckpt...")
    if not sae:
        sae = Autoencoder.from_state_dict(
            torch.load("sae_model_num_latents=65536_k=6553_epoch0_step799.pth"), 
            strict=0
            )
        sae.to(device)
    last_latent = None
    last_cls = None
    for i, batch in enumerate(tqdm(test_dataloader)):
        input_ids = batch["input_ids"].to(device)
        imgs = batch["imgs"]
        #print(input_ids[0])
        #print(imgs[0])
        attention_mask = batch["attention_mask"].to(device)
        outputs = text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
            )
        # question: why not -1? 
        class_token = outputs.hidden_states[target_layer][:, target_token, :]  

        # Pass the last hidden state through the SAE
        print(class_token.shape)
        latents = sae.forward_encode(class_token)
        print(latents.shape)

        #assert len(latents) == hidden_dim_size * expansion_f
        if i > 0: 
            diff_resid = class_token - last_cls
            diff = last_latent - latents
            print('diff between past latents, resids')
            print(diff.sum(), diff_resid.sum())
            print('class token')
            print(class_token[0])
            
            random_indices = torch.randint(0, len(latents[-1]), (10,))
            print("Random indices:", random_indices)
            print("Current latents at random indices:")
            print(latents[-1][random_indices])
            print("Previous latents at random indices:")
            print(last_latent[-1][random_indices])
        last_latent = latents
        last_cls = class_token
        assert len(latents) == len(imgs)
        continue
        for i,url in enumerate(imgs):
            print(f'saving activations of img {i}')
            print('showing first 10:')
            print(latents[i][:10])
            save_activations(url, latents[i].tolist())

        if (i+1)%10 == 0:
            top_images = top_k(5, 10)
            print("Top images for activation index 5:", top_images)

            # log sparsity for activation index 5 with default threshold 0
            sparsity_log = log_sparsity(5)
            print("Log sparsity for activation index 5:", sparsity_log)

            # mean activation value for activation index 5
            mean_activation = mean_k(5)
            print("Mean activation for activation index 5:", mean_activation)


# Training loop
target_layer = -2
target_token = 0 # CLS
num_epochs = 10
#validate()

for epoch in range(num_epochs):
    print(f'epoch {epoch}')
    for i,batch in enumerate(tqdm(train_dataloader)):  # Assuming you have a dataloader implemented
        # Get the last hidden state from the text encoder (residual stream)
        outputs = model(
            **batch,
            output_hidden_states=True
            )
        # question: why not -1? 
        class_token = outputs.hidden_states[target_layer][:, target_token, :]  
        
        # Pass the last hidden state through the SAE
        latents_pre_act, latents, sae_recons = sae(class_token)

        # Compute loss
        loss = criterion(sae_recons, class_token)  # Assuming SAE reconstructs the last hidden state
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l = loss.item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {l:.4f}")
        wandb.log({"epoch": epoch, "step": i, "loss": l})
        if (i+1) % 100 == 0:
            torch.save(sae.state_dict(), f"sae_model_num_latents={num_latents}_k={num_latents//10}_epoch{epoch}_step{i}.pth")

# Save the trained SAE model
torch.save(sae.state_dict(), f"sae_model_epoch{epoch}.pth")
wandb.finish()
# Example usage:
# Close the connection
cursor.close()
conn.close()