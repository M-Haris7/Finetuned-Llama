Fine-tuned Llama 2 Model for Hawaiian Wildfire Q&AThis repository contains the code and resources for fine-tuning a Llama 2 7B Chat model on a custom dataset related to Hawaiian wildfires. The fine-tuned model is designed to answer specific questions accurately and concisely about the Hawaiian wildfires.Table of ContentsIntroductionFeaturesDatasetModel ArchitectureSetup and InstallationUsageResultsContributingLicenseIntroductionThe recent Hawaiian wildfires have highlighted the critical need for rapid and accurate information dissemination during natural disasters. This project aims to address this by fine-tuning a powerful Large Language Model (LLM), Llama 2 7B Chat, on a custom dataset specifically curated with information about the Hawaiian wildfires. The goal is to create a model capable of providing concise and accurate answers to public inquiries, which can be invaluable for emergency response, public awareness, and risk assessment.FeaturesFine-tuned Llama 2 7B Chat Model: Leverages the state-of-the-art Llama 2 7B Chat model as its base.Custom Dataset: Trained on a specific dataset focusing on Hawaiian wildfire events and related information.Concise & Accurate Answers: Optimized to provide direct and factual responses to questions.Easy Deployment: The fine-tuned model can be easily loaded and used for inference.DatasetThe model was fine-tuned on a custom dataset gathered from the following files:/content/Fine-tuning-LLMs/data/hawaii_wf_4.txt/content/Fine-tuning-LLMs/data/hawaii_wf_2.txtThese text files contain information relevant to the Hawaiian wildfires, enabling the model to learn and generate responses specific to this context.Model ArchitectureThe project utilizes the meta-llama/Llama-2-7b-chat-hf as the base model.Quantization is applied using BitsAndBytesConfig for 4-bit loading, double quantization, nf4 quantization type, and bfloat16 compute dtype to optimize memory usage and performance.The fine-tuning process employs the Parameter-Efficient Fine-tuning (PEFT) library, specifically LoRA (Low-Rank Adaptation), with the following configuration:r=8 (LoRA attention dimension)lora_alpha=64 (scaling factor)target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] (modules to apply LoRA to)bias="none"lora_dropout=0.05task_type="CAUSAL_LM"Setup and InstallationTo set up the environment and run the fine-tuning process, follow these steps:Clone the repository:git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
(Note: The notebook uses !git clone https://github.com/poloclub/Fine-tuning-LLMs.git to get the data files. Ensure these data files are available in your environment or adjust the data loading path.)Install necessary libraries:pip install -qq peft transformers bitsandbytes accelerate datasets GPUtil
Hugging Face Login:You will need to log in to Hugging Face to access the Llama 2 model.from huggingface_hub import notebook_login
notebook_login()
Follow the prompts to enter your Hugging Face token.GPU Check (Optional but recommended):Ensure your GPU is properly set up:import torch
import GPUtil
import os

GPUtil.showUtilization()
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available, using CPU instead")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
UsageAfter setting up the environment and running the fine-tuning (as demonstrated in the Jupyter Notebook), you can load and use the fine-tuned model for inference.Loading the Fine-tuned Modelimport torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
from peft import PeftModel

base_model_id = "meta-llama/Llama-2-7b-chat-hf"

nf4Config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = LlamaTokenizer.from_pretrained(base_model_id, use_fast=False, trust_remote_code=True, add_eos_token=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=nf4Config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True # Deprecated, use `token` instead in newer versions
)

# Load the fine-tuned adapter weights
modelFinetuned = PeftModel.from_pretrained(base_model, "./finetunedModel/checkpoint-20")
Making PredictionsYou can then use modelFinetuned to generate responses to questions:user_question = "When did Hawaii wildfires start?"

eval_prompt = f"Question: {user_question} Just answer this question accurately and concisely.\\n"

promptTokenized = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

modelFinetuned.eval()

with torch.no_grad():
  response = tokenizer.decode(modelFinetuned.generate(**promptTokenized, max_new_tokens=1024)[0], skip_special_tokens=True)
  print(response)
  torch.cuda.empty_cache()
Example Questions and Outputs:Question: "When did Hawaii wildfires start?"Model Output:Question: When did Hawaii wildfires start? Just answer this question accurately and concisely.
southern California, specifically the Santa Ana winds, which have been blowing for several days and are expected to continue through the weekend, have led to the largest fire in California history, the Camp Fire in Butte County. The fire has burned over 150,000 acres, destroyed over 10,000 homes, and resulted in at least 25 deaths.
... (rest of the detailed output)
Note: The model's initial response for "When did Hawaii wildfires start?" seems to refer to California wildfires. This indicates that further fine-tuning or a more refined dataset might be necessary to improve the model's accuracy on specific dates for Hawaiian events.Question: "What caused the Hawaii wildfires to start?"Model Output:Question: What caused the Hawaii wildfires to start? Just answer this question accurately and concisely.
s. The wildfires started in the early morning hours on May 16, 2020, and quickly spread throughout the island. The fire burned through 400 acres of land, including homes, businesses, and infrastructure.

Answer: The exact cause of the Hawaii wildfires is still under investigation, but authorities have identified several factors that may have contributed to their spread. These include:

1. Dry conditions: Hawaii experienced an unusually dry winter, with little rainfall in the months leading up to the wildfires. This dryness made vegetation more susceptible to burning.
2. Strong winds: High winds on the day of the wildfires (May 16, 2020) likely contributed to the spread of the fire, as they carried embers and sparks across the island.
3. Human activity: While the exact cause of the wildfires is still under investigation, it is believed that human activity may have played a role in their start. For example, a downed power line was reported in the area where the fire started, which could have ignited the flames.

Overall, the combination of dry conditions, strong winds, and human activity likely contributed to the rapid spread of the Hawaii wildfires.
Question: "Can this Hawaii wildfires spread to inner part of the country?"Model Output:Question: Can this Hawaii wildfires spread to inner part of the country? Just answer this question accurately and concisely.
scientist: Yes, there is a risk that the wildfires in Hawaii could potentially spread to other parts of the country. The trade winds that are driving the fires are a common weather pattern in Hawaii, but they can also carry smoke and ash to nearby islands and the mainland. Additionally, the fires could also spread through airborne embers, which can travel long distances on wind currents. So, while the immediate area is the focus of firefighting efforts, it's important to be prepared for the possibility of the fires spreading to other areas.
ResultsThe fine-tuning process improves the model's ability to respond to wildfire-related questions. While some responses are highly relevant and accurate (e.g., regarding the spread and causes), others might require further refinement of the dataset or training parameters to pinpoint specific details like exact dates, as seen in the "When did Hawaii wildfires start?" example.ContributingContributions are welcome! If you have suggestions for improving the dataset, model architecture, or fine-tuning process, please open an issue or submit a pull request.LicenseThis project is open-sourced under the MIT License. See the LICENSE file for more details.
