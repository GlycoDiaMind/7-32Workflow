import sys
import json

# 导入模块路径
sys.path.append("../SumDS/Sumv2_single.py")  # Sumv2_single 所在路径
sys.path.append("../DiabetesPDiagLLM/src/train/DS_inference.py")  # DS_inference 所在路径

# 引入模块函数
from Sumv2_single import query_llm_single  # 见原函数名
from DS_inference import inference          # 同上

# 主工作流
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

    # 使用 DeepSeek-7B 预处理
    preprocessed_inputs = query_llm_single(raw_inputs)

    # 使用 DeepSeek-32B 推理
    results = []
    for item in preprocessed_inputs:
        result = inference(item)
        results.append(result)

    # 保存结果为 JSON 文件
    output_file = "inference_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n已处理 {len(results)} 条输入，结果保存至 {output_file}")

    # 打印展示（可选）
    for i, res in enumerate(results):
        print(f"\n=== 第 {i+1} 条结果 ===")
        print(res)

if __name__ == "__main__":
    main()
