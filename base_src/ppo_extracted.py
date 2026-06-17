# CELL 3
# !pip install torch transformers datasets matplotlib pandas

# CELL 5
from datasets import load_dataset

all_data = load_dataset("HumanLLMs/Human-Like-DPO-Dataset")
train_data = all_data["train"]
train_data

# CELL 7
row_num = 0
print("********** prompt **********")
print(train_data["prompt"][row_num])
print("********** chosen **********")
print(train_data["chosen"][row_num])
print("********** rejected **********")
print(train_data["rejected"][row_num])
print("********** end **********")

# CELL 9
def format_to_chatml(example):
    return {
        "chosen": f"<|im_start|>user\n{example["prompt"]}<|im_end|>\n<|im_start|>assistant\n{example["chosen"]}<|im_end|>",
        "rejected": f"<|im_start|>user\n{example["prompt"]}<|im_end|>\n<|im_start|>assistant\n{example["rejected"]}<|im_end|>",
    }

original_columns = train_data.column_names
train_data = train_data.map(format_to_chatml, remove_columns=original_columns)

# CELL 10
row_num = 0
print("********** chosen **********")
print(train_data["chosen"][row_num])
print("********** rejected **********")
print(train_data["rejected"][row_num])
print("********** end **********")

# CELL 14
import torch

device = torch.device("cuda")
torch.set_default_dtype(torch.bfloat16) # because SmolLM2-Instruct is trained on bf16

# CELL 15
from transformers import AutoModelForCausalLM, AutoConfig
from transformers import AutoTokenizer

base_model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

# download model
config = AutoConfig.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    config=config,
).to(device)

# download tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# CELL 17
from torch.utils.data import DataLoader

batch_size = 4
gradient_accumulation_steps = 8

def collate_batch(batch):
    itr_batch_size = len(batch)

    # tokenize (convert to token ids and attention mask) and convert to tensor
    token_list = [item["chosen"] for item in batch]
    token_tensor = tokenizer(
        token_list,
        padding=True,
        padding_side="right",
        return_tensors="pt").to(device)

    # generate labels for SFT
    labels = token_tensor["input_ids"][:,1:].clone()
    # generate inputs for SFT
    last_nonpad_indices = token_tensor["attention_mask"].sum(dim=1) - 1  # note: valid only in right padding
    token_tensor["input_ids"][torch.arange(itr_batch_size).to(device),last_nonpad_indices] = tokenizer.pad_token_id  # note: this is not needed, because the final token is always pad token
    token_tensor["attention_mask"][torch.arange(itr_batch_size).to(device),last_nonpad_indices] = 0
    inputs = token_tensor["input_ids"][:,:-1]
    masks = token_tensor["attention_mask"][:,:-1]

    return inputs, labels, masks

dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch
)

# CELL 19
import os, math
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
import functools

num_epochs = 1
num_steps = math.ceil(len(dataloader) / gradient_accumulation_steps)

# prepare optimizer and scheduler
optimizer = torch.optim.AdamW(
    params=base_model.parameters(),
    lr=9.0e-6,
    betas=(0.9, 0.999),
    eps=1e-08,
)

def _get_cosine_schedule(
    current_step: int,
    num_training_steps: int,
    num_warmup_steps: int=0,
    linear_warmup: bool=False,
    min_value: float=0.0,
):
    if current_step < num_warmup_steps:
        if linear_warmup:
            return min(1.0, (current_step + 1) / (num_warmup_steps + 1))  # see https://arxiv.org/abs/2410.11020
        else:
            return 1.0
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (1.0 - min_value) * scale + min_value

scheduler = LambdaLR(optimizer, lr_lambda=functools.partial(
    _get_cosine_schedule,
    num_training_steps=num_epochs*num_steps,
    min_value=0.3,
))

# remove log file if exists
log_file = "loss_sft.log"
if os.path.exists(log_file):
    os.remove(log_file)

# iterate epoch
for epoch in range(num_epochs):
    base_model.train()
    optimizer.zero_grad()
    record_loss = []

    # iterate batch
    for i, (inputs, labels, masks) in enumerate(dataloader):
        with torch.set_grad_enabled(True):
            # get logits and values to be optimized
            outputs = base_model(
                input_ids=inputs,
                attention_mask=masks,
            )

            # compute loss
            loss = F.cross_entropy(outputs.logits.transpose(1,2), labels)
            record_loss.append(loss.item())

            # optimize
            loss.backward()
            if ((i + 1) % gradient_accumulation_steps == 0) or \
               (i + 1 == len(dataloader)):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
    
            # print log
            print(f"Epoch {epoch+1} (iter{i+1}) {math.ceil((i + 1) / gradient_accumulation_steps)}/{num_steps} - loss {loss :5.4f}", end="\r")

    # save log in epoch
    with open(log_file, "a") as f:
        for l in record_loss:
            f.write("%s\n" %l)

    print("")

# save checkpoint
### torch.save(base_model.state_dict(), "llm_sft.pt")
base_model.save_pretrained("./llm_sft")

print("Done")

# CELL 21
import matplotlib.pyplot as plt
import numpy as np

log_file = "loss_sft.log"

with open(log_file, "r") as f:
    data = [float(line) for line in f]

# show plot of every 50 interval average
interval = 50
avg = []
for i in range(interval, len(data)):
    tmp_list = data[i-interval+1:i+1]
    avg.append(np.average(tmp_list))
plt.plot(np.arange(interval, len(data)), avg)
plt.show()

# CELL 23
from transformers import DynamicCache

def generate_token_by_policy(
    chat_data,
    model,
    tokenizer,
    max_seq_len,
):
    """
    Collect samples with a model (LLM) as a batch.
    To speed up generation, here we use attention cache.
    All tensors are collected with no gradient (as detached tensors).

    Parameters
    ----------
    chat_data : dic(
            input_ids: torch.tensor((batch_size, seq_len), dtype=int),
            attention_mask: torch.tensor((batch_size, seq_len), dtype=int)
        )
        Chat template data to be fed as a batch.
        The format should be left-side padding, and shouldn't include the
        final assistant's message, because it'll be generated in this function.
        (The length of input's sequence (seq_len) might differ in each call.)
    model : torch.nn.Module
        A model which is used to pick up an action (i.e., a token).
        In this function, the output is generated with no gradient.
    tokenizer : transformers.PreTrainedTokenizer
        Hugging Face tokenizer class to be used in this model.
    max_seq_len : int
        Maximum sequence length. (See above description.)

    Returns
    ----------
    completion_ids : torch.tensor((batch_size, seq_len), dtype=int)
        The array of token id for generated chat completion (including context tokens).
        The length of result's sequence (i.e., seq_len) differs depending
        on the results.
    completion_mask : torch.tensor((batch_size, seq_len), dtype=int)
        Corresponding attention mask.
    """

    # get batch size
    batch_size = chat_data["input_ids"].shape[0]

    # initialize inputs
    cur_iids = chat_data["input_ids"]
    cur_mask = chat_data["attention_mask"]

    # initialize a flag for processing/finish in a batch
    # (True: processing, False: finished)
    proceed_flag = torch.ones(batch_size, dtype=bool).to(device)

    # initialize cache parameters
    cache_position = None
    past_key_values = DynamicCache()

    # loop until all is done
    done_tokens_num = 0
    while(torch.any(proceed_flag)):
        # get current sequence length
        cur_seq_len = cur_iids.shape[1]

        # get the final non-pad token indices in sequence
        # --> shape:[batch_size]
        token_indices = torch.arange(cur_seq_len, dtype=int).to(device)
        last_nonpad_indices = (token_indices * cur_mask).argmax(-1)

        # run inference (with no gradient !)
        if cache_position is None:
            # get initial cache position
            cache_position = torch.arange(cur_seq_len, dtype=int, device=device)
            # compute logits for all input_ids
            logits = model(
                input_ids=cur_iids,
                attention_mask=cur_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
            ).logits.detach()
            # need only final output in sequence --> shape:[batch_size, vocab_size]
            logits = logits[torch.arange(batch_size).to(device), last_nonpad_indices, :]
        else:
            # compute logits only for the last input_ids
            # (others are all cached.)
            logits = model(
                input_ids=cur_iids[:,-1:],
                attention_mask=cur_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
            ).logits.detach()
            # reshape to [batch_size, vocab_size]
            logits = logits.squeeze(1)

        # select a token (i.e., take an action)
        # --> shape:[batch_size]
        probs = F.softmax(logits, dim=-1)
        selected_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # get next token indices in sequence
        # --> shape:[batch_size]
        next_token_indices = last_nonpad_indices + proceed_flag.int()

        # expand inputs when it exceeds
        # --> shape:[batch_size, cur_seq_len+1]
        if next_token_indices.max() > cur_seq_len - 1:
            cur_iids = F.pad(input=cur_iids, pad=(0, 1, 0, 0), mode="constant", value=tokenizer.pad_token_id)
            cur_mask = F.pad(input=cur_mask, pad=(0, 1, 0, 0), mode="constant", value=0)

        # store new token ids
        cur_iids[proceed_flag, next_token_indices[proceed_flag]] = selected_ids[proceed_flag]

        # store new attention mask
        cur_mask[proceed_flag, next_token_indices[proceed_flag]] = 1

        # update cache_position
        cache_position = cache_position[-1:] + 1

        # update proceed_flag
        not_lim = (cur_mask.sum(dim=1) < max_seq_len)
        is_eos = torch.logical_and((selected_ids == tokenizer.eos_token_id),proceed_flag.bool())
        not_eos = torch.logical_not(is_eos)
        proceed_flag = torch.logical_and(proceed_flag, torch.logical_and(not_lim, not_eos))

    return cur_iids, cur_mask

# CELL 24
max_seq_len = 768

#
# build a batch of questions
# (To use cache, we apply left-side padding.)
#

messages = [
    "What do you most want to do right now?",
    "What is the best gift to give a friend who loves the outdoors?",
    "How do you relax after something bad happens?",
]
inputs = [f"<|im_start|>user\n{m}<|im_end|>\n<|im_start|>assistant\n" for m in messages]
input_batch = tokenizer(
    inputs,
    padding=True,
    padding_side="left",
    return_tensors="pt").to(device)
input_seq_len = input_batch["input_ids"].shape[1]

#
# generate model's outputs
#

base_model = AutoModelForCausalLM.from_pretrained("./llm_sft").to(device)
base_model.eval()

with torch.no_grad():
    iids, mask = generate_token_by_policy(
        input_batch,
        base_model,
        tokenizer,
        max_seq_len,
    )
iids = iids[:,input_seq_len:]
outputs = tokenizer.batch_decode(iids, skip_special_tokens=True)

#
# print results
#

for i in range(len(messages)):
    print("***** Question *****")
    print(messages[i])
    print("***** Answer *****")
    print(outputs[i])
    print("")

# CELL 27
base_model = AutoModelForCausalLM.from_pretrained("./llm_sft").to(device)

# CELL 29
base_model

# CELL 31
from torch import nn

# define model class
class RewardModel(nn.Module):
    def __init__(
        self,
        base_model,
    ):
        super().__init__()

        # replace final linear layer
        self.base_model = base_model
        self.base_model.__setattr__(
            "lm_head",
            nn.Linear(576, 1, bias=False).to(device))

    def forward(self, input_ids, attention_mask):
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

        # need only final output in sequence
        batch_size = input_ids.shape[0]
        token_indices = torch.arange(input_ids.shape[-1], dtype=int).to(device)
        last_nonpad_indices = (token_indices * attention_mask).argmax(-1)
        reward_score = output[torch.arange(batch_size).to(device), last_nonpad_indices]

        return reward_score

# generate model instance
rm = RewardModel(base_model).to(device)

# CELL 32
# show how it's changes
rm

# CELL 34
# 1. add "chosen_len" and "rejected_len" column
#   (which indicates the sequence length in "chosen" and "rejected")
def add_seq_len(example):
    def get_tokenized_length(text):
        tokenized = tokenizer(text)
        return len(tokenized["input_ids"])

    chosen_len = get_tokenized_length(example["chosen"])
    reject_len = get_tokenized_length(example["rejected"])
    return {
        "chosen_len": chosen_len,
        "rejected_len": reject_len
    }

train_data = train_data.map(add_seq_len)

# 2. remove rows which exceed the maximum sequence length
train_data = train_data.filter(lambda example: example["chosen_len"] <= max_seq_len and example["rejected_len"] <= max_seq_len)
train_data = train_data.remove_columns(["chosen_len", "rejected_len"])

# show total number of filtered rows
train_data

# CELL 37
batch_size = 4
gradient_accumulation_steps = 8

def collate_batch(batch):
    # tokenize (convert to token ids and attention mask) and convert to tensor
    chosen_list = [item["chosen"] for item in batch]
    reject_list = [item["rejected"] for item in batch]
    chosen_tensor = tokenizer(
        chosen_list,
        padding=True,
        padding_side="left",  # see above description !
        return_tensors="pt").to(device)
    reject_tensor = tokenizer(
        reject_list,
        padding=True,
        padding_side="left",  # see above description !
        return_tensors="pt").to(device)
    return chosen_tensor, reject_tensor

dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch
)

# CELL 39
num_epochs = 1
num_steps = math.ceil(len(dataloader) / gradient_accumulation_steps)

optimizer = torch.optim.AdamW(
    params=rm.parameters(),
    lr=3.0e-5,
    betas=(0.9, 0.999),
    eps=1e-08,
)
scheduler = LambdaLR(optimizer, lr_lambda=functools.partial(
    _get_cosine_schedule,
    num_training_steps=num_epochs*num_steps,
    num_warmup_steps=math.ceil(num_epochs*num_steps*0.1)))

# remove log file if exists
log_file = "loss_rm.log"
if os.path.exists(log_file):
    os.remove(log_file)

# iterate epoch
for epoch in range(num_epochs):
    rm.train()
    optimizer.zero_grad()
    record_loss = []

    # iterate batch
    for i, (chosen, reject) in enumerate(dataloader):
        with torch.set_grad_enabled(True):
            # get reward score (chosen)
            rewards_chosen = rm(
                input_ids=chosen["input_ids"],
                attention_mask=chosen["attention_mask"],
            )
            # get reward score (rejected)
            rewards_reject = rm(
                input_ids=reject["input_ids"],
                attention_mask=reject["attention_mask"],
            )
            # compute loss
            loss = -F.logsigmoid(rewards_chosen - rewards_reject).mean()
            loss.backward()
            # optimization by accumulation
            if ((i + 1) % gradient_accumulation_steps == 0) or \
               (i + 1 == len(dataloader)):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            # print log
            record_loss.append(loss.item())
            print(f"Epoch {epoch+1} (iter{i+1}) {math.ceil((i + 1) / gradient_accumulation_steps)}/{num_steps} - loss {loss :5.4f}", end="\r")

    # save logging
    epoch_average_loss = sum(record_loss)/len(record_loss)
    print(f"Epoch {epoch+1} (iter{i+1}) {math.ceil((i + 1) / gradient_accumulation_steps)}/{num_steps} - loss {epoch_average_loss :5.4f}")
    with open(log_file, "a") as f:
        for l in record_loss:
            f.write("%s\n" %l)

# save checkpoint
torch.save(rm.state_dict(), "rm.pt")

print("Done")

# CELL 41
import matplotlib.pyplot as plt

log_file = "loss_rm.log"

with open(log_file, "r") as f:
    data = [float(line) for line in f]

plt.plot(np.arange(len(data)), data)
plt.show()

# CELL 43
import pandas as pd

# # uncomment when you load RM from the trained checkpoint
# rm.load_state_dict(torch.load("rm.pt", weights_only=True))

rm.eval()

test_list = [
    "<|im_start|>user\nWhat do you most want to do right now?<|im_end|>\n<|im_start|>assistant\nRight now, I most want to assist you and provide helpful, respectful, and engaging interactions. If you have any questions, need information, or just want to chat, I'm here for you!<|im_end|>",
    "<|im_start|>user\nWhat do you most want to do right now?<|im_end|>\n<|im_start|>assistant\nI'd love to go for a hike in the woods! 😍 It's such a beautiful day and I really need to get some fresh air. How about you? 😊<|im_end|>",
    "<|im_start|>user\nWhat is the best gift to give a friend who loves the outdoors?<|im_end|>\n<|im_start|>assistant\nThat's a thoughtful question! Here are a few gift ideas for a friend who loves the outdoors.: High-quality Multi-tool, Reusable Water Bottle with Filter, or Portable Hammock<|im_end|>",
    "<|im_start|>user\nWhat is the best gift to give a friend who loves the outdoors?<|im_end|>\n<|im_start|>assistant\nOoh, that's a fun question!! 🙋 If you're looking for some gift ideas for a friend who loves the outdoors, I suggest National Parks Pass, Hammock, or Portable Camping Chair! 🤪🤣<|im_end|>",
]
test_batch = tokenizer(
    test_list,
    padding=True,
    padding_side="left",
    return_tensors="pt").to(device)
test_rewards = rm(
    input_ids=test_batch["input_ids"],
    attention_mask=test_batch["attention_mask"],
)

df = pd.DataFrame({
    "text": test_list,
    "score": test_rewards.squeeze(-1).tolist()
})
df

# CELL 47
policy_model = AutoModelForCausalLM.from_pretrained("./llm_sft").to(device)

# CELL 50
class ValueModel(nn.Module):
    def __init__(
        self,
        base_model,
    ):
        super().__init__()

        # replace final linear layer
        self.base_model = base_model
        self.base_model.__setattr__(
            "lm_head",
            nn.Linear(576, 1, bias=False).to(device))

    def forward(self, input_ids, attention_mask):
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

        return output.squeeze(-1)

base_model = AutoModelForCausalLM.from_pretrained("./llm_sft").to(device)
value_model = ValueModel(base_model).to(device)

# CELL 52
# remove the final message,
# but it keeps the final "<|assistant|>\n"
def rm_fin_msg(chat_str):
    target = "<|im_start|>assistant\n"
    start_idx = chat_str.rfind(target)
    return chat_str[:(start_idx + len(target))]

# CELL 53
for row in range(3):
    print("********** original chat **********")
    print(train_data["chosen"][row])
    print("********** converted chat **********")
    print(rm_fin_msg(train_data["chosen"][row]))

# CELL 55
batch_size = 4
gradient_accumulation_steps = 8

def collate_batch(batch):
    # remove the final message
    chat_list = [rm_fin_msg(item["chosen"]) for item in batch]

    # tokenize (convert to token ids and attention mask) and convert to tensor
    chat_tensor = tokenizer(
        chat_list,
        padding=True,
        padding_side="left",
        return_tensors="pt").to(device)
    return chat_tensor

dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch
)

# CELL 57
# # uncomment when you load RM from the trained checkpoint
# rm.load_state_dict(torch.load("rm.pt", weights_only=True))

# CELL 58
# function to get advantages (see above description for GAE)
def get_advantage(delta, gamma, gae_lambda, seq_len):
    gae_params = torch.tensor([(gamma * gae_lambda)**i for i in range(seq_len)], dtype=torch.float32).to(device) # to float32 (see above)
    adv = [torch.sum(delta[:,i:] * gae_params[:(seq_len - i)], dim=-1) for i in range(seq_len)] # list of tensors
    adv = torch.stack(adv, dim=1) # shape (batch_size, seq_len)
    return adv

# CELL 59
gamma = 1.0
gae_lambda = 0.95
kl_coeff = 0.05  # beta

clip_range_policy = 0.2  # epsilon on clipped policy
clip_range_value = 0.2   # epsilon on clipped value

num_epochs = 5
num_steps = math.ceil(len(dataloader) / gradient_accumulation_steps)

# prepare optimizer and scheduler (value model)
opt1 = torch.optim.AdamW(
    params=value_model.parameters(),
    lr=3.0e-5,
    betas=(0.9, 0.999),
    eps=1e-08,
)
sch1 = LambdaLR(opt1, lr_lambda=functools.partial(
    _get_cosine_schedule,
    num_training_steps=num_epochs*num_steps,
    num_warmup_steps=math.ceil(num_epochs*num_steps*0.1)))

# prepare optimizer and scheduler (policy model)
opt2 = torch.optim.AdamW(
    params=policy_model.parameters(),
    lr=3.0e-5,
    betas=(0.9, 0.999),
    eps=1e-08,
)
sch2 = LambdaLR(opt2, lr_lambda=functools.partial(
    _get_cosine_schedule,
    num_training_steps=num_epochs*num_steps,
    num_warmup_steps=math.ceil(num_epochs*num_steps*0.1),
    linear_warmup=True))

# remove log file if exists
log_file = "reward.log"
if os.path.exists(log_file):
    os.remove(log_file)

# reward model is always for inference
rm.eval()

# iterate epoch
for epoch in range(num_epochs):
    opt1.zero_grad()
    opt2.zero_grad()
    record_reward = []

    # iterate batch
    for i, chat in enumerate(dataloader):
        itr_batch_size = chat["input_ids"].shape[0]
        input_seq_len = chat["input_ids"].shape[1]

        #####
        # Prepare data with old policy
        # - Model is used only for inference.
        #####
        policy_model.eval()
        with torch.no_grad():
            # generate tokens with current policy
            gen_iids, gen_mask = generate_token_by_policy(
                chat,
                policy_model,
                tokenizer,
                max_seq_len,
            )

            # mask only inference (pred) tokens
            seq_len = gen_iids.shape[1]
            token_indices = torch.arange(seq_len, dtype=int).to(device)
            inf_mask = (gen_mask * (token_indices >= input_seq_len).int()).bool()

            # pad left
            # (e.g., [[0,1,1,0,0,],[1,1,1,1,0]] --> [[0,0,0,1,1],[0,1,1,1,1]])
            last_nonpad_indices = (token_indices * gen_mask).argmax(-1)
            for b in range(itr_batch_size):
                gen_iids[b,:] = torch.roll(
                    gen_iids[b,:],
                    shifts=(seq_len - last_nonpad_indices[b] - 1).item()
                )
                gen_mask[b,:] = torch.roll(
                    gen_mask[b,:],
                    shifts=(seq_len - last_nonpad_indices[b] - 1).item()
                )
                inf_mask[b,:] = torch.roll(
                    inf_mask[b,:],
                    shifts=(seq_len - last_nonpad_indices[b] - 1).item()
                )

            # trim left
            # (e.g., [[0,0,0,1,1],[0,1,1,1,1]] --> [[0,0,1,1],[1,1,1,1]])
            first_nonpad_indices = (torch.flip(token_indices, dims=(0,)) * gen_mask).argmax(-1)
            start_index = first_nonpad_indices.min()
            gen_iids = gen_iids[:,start_index:]
            gen_mask = gen_mask[:,start_index:]
            inf_mask = inf_mask[:,start_index:]

            # the final state is not used for estimation
            inf_mask = inf_mask[:,:-1]

            # get rewards
            rewards = torch.zeros_like(gen_iids[:,:-1], dtype=torch.bfloat16).to(device)
            seq_rewards = rm(
                input_ids=gen_iids,
                attention_mask=gen_mask,
            ).detach().squeeze(-1)
            rewards[:,-1] = seq_rewards
            record_reward.append(seq_rewards.mean().item())

            # only the completed sequence is processed
            is_eos = (gen_iids[:,-1] == tokenizer.eos_token_id)
            is_eos_num = is_eos.int().sum()
            if is_eos_num == 0:
                continue
            elif not(is_eos_num == itr_batch_size):
                gen_iids = gen_iids[is_eos]
                gen_mask = gen_mask[is_eos]
                inf_mask = inf_mask[is_eos]
                rewards = rewards[is_eos]

        #####
        # Run training for value
        # - Model is used for training.
        # - We use float32 precision (not bfloat or float16) for value computation.
        #####

        value_model.train()
        with torch.set_grad_enabled(True):
            # get values
            values_new = value_model(
                input_ids=gen_iids[:,:-1],
                attention_mask=gen_mask[:,:-1],
            )
            values_new = values_new * gen_mask[:,:-1].float() # to float32 (see above)
            values_old = values_new.detach()

            # get next values
            values_next = values_old[:,1:]
            values_next = F.pad(input=values_next, pad=(0, 1, 0, 0), mode="constant", value=0.0)

            # get delta
            delta = rewards + values_next * gamma - values_old

            # get actual values r + \sum \gamma r (see above)
            adv = get_advantage(
                delta=delta,
                gamma=gamma,
                gae_lambda=1.0,
                seq_len=delta.shape[1])
            values_actual = adv + values_old

            # estimate a standard deviation (sigma) of values
            values_var = torch.square(values_old - values_actual)
            values_var = torch.masked_select(values_var, inf_mask).mean()
            values_stddev = torch.sqrt(values_var)

            # get value loss (maximum of unclipped and clipped)
            values_new_clipped = torch.clamp(
                values_new,
                values_old - clip_range_value * values_stddev,
                values_old + clip_range_value * values_stddev,
            )
            val_loss1 = torch.square(values_new - values_actual)
            val_loss2 = torch.square(values_new_clipped - values_actual)
            val_loss = 0.5 * torch.max(val_loss1, val_loss2) / values_var
            val_loss = torch.masked_select(val_loss, inf_mask).mean()

            # optimize value model (critic)
            val_loss.backward()
            if ((i + 1) % gradient_accumulation_steps == 0) or \
               (i + 1 == len(dataloader)):
                opt1.step()
                sch1.step()
                opt1.zero_grad()

        #####
        # Run training for policy
        # - Model is used for training.
        #####

        policy_model.train()
        with torch.set_grad_enabled(True):
            # get logits
            logits_new = policy_model(
                input_ids=gen_iids[:,:-1],
                attention_mask=gen_mask[:,:-1],
            )
            logits_new = logits_new.logits
            logits_old = logits_new.detach()

            # get propability P
            logprb_old = -F.cross_entropy(logits_old.transpose(1,2), gen_iids[:,1:], reduction="none") # get log probability (see above description)
            logprb_new = -F.cross_entropy(logits_new.transpose(1,2), gen_iids[:,1:], reduction="none") # get log probability (see above description)

            # get advantage loss with clipped objective
            prb_ratio = torch.exp(logprb_new - logprb_old) # P_new / P_old
            prb_ratio_clipped = torch.clamp(
                prb_ratio,
                1.0 - clip_range_policy,
                1.0 + clip_range_policy,
            )
            adv = get_advantage(
                delta=delta,
                gamma=gamma,
                gae_lambda=gae_lambda,  # 0.95
                seq_len=delta.shape[1])
            pg_loss1 = -adv * prb_ratio
            pg_loss2 = -adv * prb_ratio_clipped
            pg_loss = torch.max(pg_loss1, pg_loss2)
            pg_loss = torch.masked_select(pg_loss, inf_mask).mean()

            # get KL loss
            # (see https://github.com/tsmatz/reinforcement-learning-tutorials/blob/master/04-ppo.ipynb)
            l_old = logits_old - torch.amax(logits_old, dim=2, keepdim=True) # reduce quantity to avoid overflow
            l_new = logits_new - torch.amax(logits_new, dim=2, keepdim=True) # reduce quantity to avoid overflow
            e_old = torch.exp(l_old)
            e_new = torch.exp(l_new)
            e_sum_old = torch.sum(e_old, dim=2, keepdim=True)
            e_sum_new = torch.sum(e_new, dim=2, keepdim=True)
            p_old = e_old / e_sum_old
            kl_loss = torch.sum(
                p_old * (l_old - l_new + torch.log(e_sum_new) - torch.log(e_sum_old)),
                dim=2)
            kl_loss = torch.masked_select(kl_loss, inf_mask).mean()

            # get policy loss
            total_policy_loss = pg_loss + kl_loss * kl_coeff

            # optimize policy model (actor)
            total_policy_loss.backward()
            if ((i + 1) % gradient_accumulation_steps == 0) or \
               (i + 1 == len(dataloader)):
                opt2.step()
                sch2.step()
                opt2.zero_grad()
    
        # print log
        print(f"Epoch {epoch+1} (iter{i+1}) {math.ceil((i + 1) / gradient_accumulation_steps)}/{num_steps} - reward {seq_rewards.mean().item() :5.4f}", end="\r")

    # save logging
    epoch_average_reward = sum(record_reward)/len(record_reward)
    print(f"Epoch {epoch+1} (iter{i+1}) {math.ceil((i + 1) / gradient_accumulation_steps)}/{num_steps} - reward {epoch_average_reward :5.4f}")
    with open(log_file, "a") as f:
        for r in record_reward:
            f.write("%s\n" %r)

# save checkpoint
torch.save(value_model.state_dict(), "value.pt")
policy_model.save_pretrained("./llm_aligned")

print("Done")

# CELL 61
import matplotlib.pyplot as plt
import numpy as np

log_file = "reward.log"

with open(log_file, "r") as f:
    data = [float(line) for line in f]

# show plot of every 120 interval average
interval = 120
avg = []
for i in range(interval, len(data)):
    tmp_list = data[i-interval+1:i+1]
    avg.append(np.average(tmp_list))
plt.plot(np.arange(interval, len(data)), avg)
plt.show()

# CELL 63
#
# build a batch of questions
# (To use cache, we apply left-side padding.)
#

messages = [
    "What do you most want to do right now?",
    "What is the best gift to give a friend who loves the outdoors?",
    "How do you relax after something bad happens?",
]
inputs = [f"<|im_start|>user\n{m}<|im_end|>\n<|im_start|>assistant\n" for m in messages]
input_batch = tokenizer(
    inputs,
    padding=True,
    padding_side="left",
    return_tensors="pt").to(device)
input_seq_len = input_batch["input_ids"].shape[1]

#
# generate model's outputs
#

policy_model.eval()
with torch.no_grad():
    iids, mask = generate_token_by_policy(
        input_batch,
        policy_model,
        tokenizer,
        max_seq_len,
    )
iids = iids[:,input_seq_len:]
outputs = tokenizer.batch_decode(iids, skip_special_tokens=True)

#
# print results
#

for i in range(len(messages)):
    print("***** Question *****")
    print(messages[i])
    print("***** Answer *****")
    print(outputs[i])
    print("")

# CELL 65


