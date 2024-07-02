from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 모델 경로
model_path = "fine_tuned_qwen2-0.5B"

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 파이프라인 설정
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_length=512,
    truncation=True
)

# HuggingFace 파이프라인을 사용한 LLM 설정
hf_llm = HuggingFacePipeline(pipeline=pipe)

# 프롬프트 템플릿 설정
prompt = ChatPromptTemplate.from_messages([
    ("system", "이 AI는 게임 용어를 설명해줍니다."),
    ("user", "다음 용어를 설명해 주세요: {word}\n정의: "),
])

# 체인 설정
chain = prompt | hf_llm | StrOutputParser()

# 질문
word = "크로미시간"

formatted_prompt = prompt.format(word=word)
print("Final Prompt: ", formatted_prompt)

# 체인으로 답변 생성
response = chain.invoke({"word": word})
print(response)
