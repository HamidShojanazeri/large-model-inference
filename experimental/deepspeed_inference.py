import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import deepspeed

gigabyte_size = 1073741824

def format_to_gb(item, precision=4):
    """quick function to format numbers to gigabyte and round to (default) 4 digit precision"""
    metric_num = item / gigabyte_size
    metric_num = round(metric_num, ndigits=precision)
    return metric_num


MAX_NEW_TOKENS = 128
model_name = 'facebook/opt-33b'
# model_name = "bigscience/bloom-3b"

text = """
Q: On average Joe throws 25 punches per minute. A fight lasts 5 rounds of 3 minutes.
How many punches did he throw?\n
A: Let’s think step by step.\n"""
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer(text, return_tensors="pt").input_ids

free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
max_memory = f'{free_in_GB-2}GB'

n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}
start_loading = time.time()

# model_int8 = AutoModelForCausalLM.from_pretrained(
#   model_name,
#   device_map='auto',
#   load_in_8bit=True,
#   max_memory=max_memory
# )

model = AutoModelForCausalLM.from_pretrained(
  model_name
)

end_loading = time.time()
print( " model load time is  {} s ".format((end_loading-start_loading)) )

##### memory foot print on all devices

ds_engine = deepspeed.init_inference(model,
                                 mp_size=4,
                                 dtype=torch.half,
                                 checkpoint=None,
                                 replace_method='auto',
                                 replace_with_kernel_inject=True)

model = ds_engine.module

devices_list = torch.cuda.device_count()

for device in range(devices_list):
  gpu = "cuda:"+ str(device)
  print( f" memory reserved on device {gpu} is {torch.cuda.mem_get_info(device=gpu)}")

inference_latency = []

for i in range(1):
  start_inference = time.time()

  generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
# generated_ids = model_native.generate(input_ids, max_length=MAX_NEW_TOKENS)
  stop_inference = time.time()
  inference_latency.append(stop_inference-start_inference)

print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
avg_inference_latency = sum(inference_latency)/len(inference_latency)
print( " Avg time for inference is {}s ".format((avg_inference_latency)) )
