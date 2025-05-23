import lazyllm
from lazyllm import OnlineChatModule, pipeline, _0
from lazyllm.tools import IntentClassifier

from statistical_agent import build_statistical_agent
# from paper_rag import build_paper_rag
from rag_final import build_paper_rag

gen_prompt = """
# 图片信息提取器
 
1. 返回格式：  
   ### 提问: [用户原问题]  
   ### 提问中涉及到的图像内容描述：[客观描述，包括主体、背景、风格等]  
2. 要求：详细、中立，避免主观猜测  

**示例：**  
输入："找类似的猫图"（附橘猫图）, 
响应如下：
   ### 提问: 找类似的猫图  
   ### 提问中涉及到的图像内容描述：一只橘猫趴在沙发上，阳光从左侧照射，背景是米色窗帘  

"""

# def func(x):
#     print(">" * 50 + f"\n{x}\n")
#     return x

# 构建 rag 工作流和统计分析工作流
rag_ppl = build_paper_rag()
sql_ppl = build_statistical_agent()

# 搭建具备知识问答和统计问答能力的主工作流
def build_paper_assistant():
    llm = OnlineChatModule(source='qwen', stream=False)
    vqa = lazyllm.OnlineChatModule(source="sensenova",\
        model="SenseNova-V6-Turbo").prompt(lazyllm.ChatPrompter(gen_prompt))

    with pipeline() as ppl:
        ppl.ifvqa = lazyllm.ifs(
            lambda x: x.startswith('<lazyllm-query>'),
            lambda x: vqa(x), lambda x:x)
        with IntentClassifier(llm) as ppl.ic:
            ppl.ic.case["论文问答", rag_ppl]
            ppl.ic.case["统计问答", sql_ppl]

    return ppl

if __name__ == "__main__":
    print("Building paper assistant pipeline...")
    main_ppl = build_paper_assistant()
    print("Pipeline built. Starting WebModule...")
    lazyllm.WebModule(main_ppl, port=23459, static_paths="./images", encode_files=True).start().wait()
    print("WebModule stopped.") # 这句只有在服务停止后才会打印
