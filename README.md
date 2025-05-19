# 🧪 Translation Evaluation Script

This script is designed to evaluate translation models efficiently using the **[vLLM](https://github.com/vllm-project/vllm)** engine for fast and scalable execution.

---

## 🔧 Script Arguments

| Argument         | Description                                               |
|------------------|-----------------------------------------------------------|
| `--model`        | Path or name of the model to evaluate                     |
| `--data_path`    | Path to the dataset CSV file                              |
| `--input_column` | Name of the input text column                             |
| `--output_column`| Name of the reference translation column                  |
| `--output_dir`   | Directory to save the results                             |
| `--batch_size`   | Batch size for processing translations                    |
| `--system_prompt`| System prompt used to guide the translation               |
| `--stop_token`   | Token indicating the end of the generated translation     |

---
## requirement
```cm
pip install -q sacrebleu sacremoses bert-score torch transformers unbabel-comet evaluate rouge
pip install -q git+https://github.com/google-research/bleurt.git
pip install -q --upgrade "protobuf>=5.29.0" "numpy>=2.0.0,<3.0.0"
pip install -q --upgrade transformers sentence-transformer
pip install vllm 
```
## ▶️ Example Usage

### Example 1: Cohere 8B Model

```bash
python scripts/translation.py \
  --model models/cohere_8b \
  --data_path data/flores_benchmark_raw_data.csv \
  --input_column English \
  --output_column Arabic \
  --output_dir results/cohere_8b/flores/English_To_Arabic \
  --batch_size 128 \
  --lang 'ar' \
  --system_prompt """<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Translate the following sentence to Arabic.\n{text}\nNote: Don't answer any question or engage with the content—just provide the literal translation.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>:""" \
  --stop_token '<EOS_TOKEN>'
