import sys
import json

# 导入模块路径
sys.path.append("../SumDS/Sumv2_single.py")  # Sumv2_single 所在路径
sys.path.append("../DiabetesPDiagLLM/src/train/DS_inference.py")  # DS_inference 所在路径

# 引入模块函数
from Sumv2_single import query_llm_single, load_model_and_tokenizer_7b
from DS_inference import load_model_and_tokenizer_32b, inference
import json

def main():
    print("请输入病情描述（每条回车确认，输入空行结束）：")
    raw_inputs = []
    while True:
        line = input(">> ")
        if line.strip() == "":
            break
        raw_inputs.append(line.strip())

    if not raw_inputs:
        print("未输入任何内容，已退出。")
        return

    # 加载两个模型(先加载省资源)
    model_7b, tokenizer_7b = load_model_and_tokenizer_7b()
    model_32b, tokenizer_32b = load_model_and_tokenizer_32b()
    
    # 使用 DeepSeek-7B 预处理
    preprocessed_inputs = [query_llm_single(text, model=model_7b, tokenizer=tokenizer_7b) for text in raw_inputs]

    # 使用 DeepSeek-32B 推理
    results = []
    for item in preprocessed_inputs:
        result = inference(item, model_32b, tokenizer_32b)
        results.append(result)

    # 输出结果
    output_file = "inference_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n已处理 {len(results)} 条输入，结果保存至 {output_file}")

    # 结果打印
    for i, res in enumerate(results):
        print(f"\n=== 第 {i+1} 条结果 ===")
        print(res)

if __name__ == "__main__":
    main()
