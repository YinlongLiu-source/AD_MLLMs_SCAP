
import os

device_ids = "0,1"  
os.environ["CUDA_VISIBLE_DEVICES"] = device_ids

import re
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score
import json

MODEL_ID = "Instruction_tuning_output/merged_directory_name"   # or MLLMs/midashenglm-7b...
test_file_path = "Dataset/test_example.jsonl"
OUTPUT_CSV = "results_test_json.csv"


from modelscope import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

print("Loading model / tokenizer / processor ...")
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

GEN_KW = dict(max_new_tokens=4, do_sample=False)

def parse_pred_label(text: str) -> str:
    """
    从模型输出中提取 AD 或 HC：
    - 优先匹配英文 AD/HC（忽略大小写）
    - 兼容中文关键词（阿尔茨海默症→AD；健康对照→HC）
    - 若都无法确定，则返回 HC（保守）
    """
    if not text:
        return "HC"
    t = text.strip()
    m = list(re.finditer(r"\b(AD|HC)\b", t, flags=re.IGNORECASE))
    if m:
        return m[-1].group(1).upper()
    # 中文关键词兜底
    t_cn = t.replace("阿尔兹海默", "阿尔茨海默")
    has_ad = ("阿尔茨海默症" in t_cn) or ("阿兹海默" in t_cn) or ("老年痴呆" in t_cn)
    has_hc = ("健康对照" in t_cn) or ("健康" in t_cn and "对照" in t_cn)
    if has_ad and not has_hc:
        return "AD"
    if has_hc and not has_ad:
        return "HC"
    # 广义兜底
    if "AD" in t.upper():
        return "AD"
    if "HC" in t.upper():
        return "HC"
    print(f"[WARN] Unable to parse AD/HC from output: {text!r}")
    return "HC"

@torch.no_grad()
def predict_single(audio_path: str, SYSTEM_PROMPT: str, USER_PROMPT_TEMPLATE: str) -> str:
    """
    对单个音频做预测，返回 'AD' 或 'HC'
    采用官方推荐的多模态 chat 模板：messages -> processor.apply_chat_template -> model.generate
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "text", "text": USER_PROMPT_TEMPLATE},
            {"type": "audio", "path": audio_path},  
        ]},
    ]

    model_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        add_special_tokens=True,
        return_dict=True,
    )

    model_inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k,v in model_inputs.items()}

    # 生成
    output_ids = model.generate(**model_inputs, **GEN_KW)
    out = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[-1]
    print(out)
    print(parse_pred_label(out))
    return parse_pred_label(out)

# -------------------------
# 遍历数据集推理
# -------------------------

test_samples = []
with open(test_file_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():  # 跳过空行
            sample = json.loads(line)
            test_samples.append(sample)

results = []
for s in tqdm(test_samples):
    audio_path = s["audios"][-1]
    SYSTEM_PROMPT = s["messages"][0]["content"]
    USER_PROMPT_TEMPLATE = s["messages"][1]["content"][7:]
    print("audio path:", audio_path)
    print("SYSTEM_PROMPT:",SYSTEM_PROMPT)
    print("USER_PROMPT_TEMPLATE:",USER_PROMPT_TEMPLATE)
    if audio_path:
        pred = predict_single(audio_path, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)
        results.append({"audio_path": audio_path, "true_label": s["messages"][2]["content"], "pred_label": pred})
        print("True label:", s["messages"][2]["content"])
        print("Predicted label:", pred)


# -------------------------
# 保存结果 & 计算指标
# -------------------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n[OK] Results saved -> {OUTPUT_CSV}")

y_true = df["true_label"].tolist()
y_pred  = df["pred_label"].tolist()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4, labels=["AD", "HC"], target_names=["AD", "HC"]))
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Macro-F1 : {f1_score(y_true, y_pred, average='macro'):.4f}")
