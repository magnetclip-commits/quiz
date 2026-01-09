'''
'''
import os
from typing import Any, Optional, List, Dict
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatPerplexity
from langchain_anthropic import ChatAnthropic
import openai

# 환경 변수 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# 명시적으로 export할 함수들
__all__ = ['get_llm', 'get_perplexity_search_results']

# 모듈 버전 정보
__version__ = '1.0.0'


from typing import Any, Iterator, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import openai
import os
import requests

def get_perplexity_search_results(prompt: str, api_key: str, model: str = "sonar-pro") -> List[Dict]:
    """
    Perplexity API를 직접 호출하여 search_results를 가져오는 함수
    
    Args:
        prompt: 질문 프롬프트
        api_key: Perplexity API 키
        model: 사용할 모델명 (기본값: sonar-pro)
    
    Returns:
        search_results 리스트
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 0.7,
            "stream": False
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            search_results = result.get("search_results", [])
            return search_results
        else:
            print(f"Perplexity API 오류: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"Perplexity search_results 가져오기 오류: {e}")
        return []

                
def get_llm(user_settings: dict):
    """
    사용자 설정에 따라 적절한 LLM 인스턴스를 생성하는 팩토리 함수
    
    Args:
        user_settings (dict): 사용자 설정 딕셔너리
            - model_provider: 모델 제공자 (pplx, gmn15, gmn25f, gmn25, cld4o, gpt4o, gpt4om, gpt41, gpto3, gpto3p 등)
            - temperature: 온도 설정 (기본값: 0.7)
            - frequency_penalty: 빈도 페널티 (기본값: 0.0)
    
    Returns:
        LLM 인스턴스
    """
    model_provider = user_settings.get("model_provider", "openai")
    
    if model_provider == "pplxp": # perplexity
        return ChatPerplexity(
            model="sonar-pro",
            api_key=PERPLEXITY_API_KEY,
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000,
            streaming=True
        )
    elif model_provider == "pplx": 
        return ChatPerplexity(
            model="sonar-reasoning",
            api_key=PERPLEXITY_API_KEY,
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000,
            streaming=True
        )
    elif model_provider == "gmn25f": 
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=user_settings.get("temperature", 0.7),
            max_output_tokens=5000,
            disable_streaming=False  # False로 설정하면 스트리밍 활성화
        )
    elif model_provider == "gmn25": 
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", 
            google_api_key=GOOGLE_API_KEY,
            temperature=user_settings.get("temperature", 0.7),
            max_output_tokens=5000,
            disable_streaming=False  # False로 설정하면 스트리밍 활성화
        )
    elif model_provider == "cld4o": 
        return ChatAnthropic(
            model="claude-opus-4-20250514",
            api_key=CLAUDE_API_KEY,
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000,
            streaming=True
        )
    elif model_provider == "gpt4o": # openai gpt-4o
        return ChatOpenAI(
            model="gpt-4o",
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000,
            frequency_penalty=user_settings.get("frequency_penalty", 0.0),
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )
    elif model_provider == "gpt4om": # openai gpt-o4-mini
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000,
            frequency_penalty=user_settings.get("frequency_penalty", 0.0),
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )
    elif model_provider == "gpt41": # openai gpt-4.1 
        return ChatOpenAI(
            model="gpt-4.1",
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000,
            frequency_penalty=user_settings.get("frequency_penalty", 0.0),
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )
    elif model_provider == "gpt5": # openai gpt-4.1 
        return ChatOpenAI(
            model="gpt-5",
            #temperature=user_settings.get("temperature", 0.7), temperature 지원 안함 
            max_tokens=2000,
            #frequency_penalty=user_settings.get("frequency_penalty", 0.0), frequency 지원 안함
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )
    elif model_provider == "gpto3":  # openai gpt-o3
        return ChatOpenAI(
            model="o3-2025-04-16",   # ✅ Chat 모델로 사용
            max_tokens=2000,
            temperature=1,
            openai_api_key=OPENAI_API_KEY,
            streaming=True 
        )
    elif model_provider == "gpto3p":  # openai gpt-o3-pro
        return ChatOpenAI(
            model="o3-pro-2025-06-10",
            max_tokens=2000,
            temperature=1,
            openai_api_key=OPENAI_API_KEY,
            streaming=True 
        )
    else:
        # 기본값: 아무것도 선택 안됐을 때도 ChatOpenAI 사용
        return ChatOpenAI(
            model="gpt-4o",
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000,
            frequency_penalty=user_settings.get("frequency_penalty", 0.0),
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )

