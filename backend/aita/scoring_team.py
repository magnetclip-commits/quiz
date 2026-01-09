'''
@modified: 
2025.10.27 line 43 item_type_cd 변수추가, RunScoringAgentTeam 함수를 비동기적으로 실행하기 위해 run_scoring_instance 함수 추가 
'''
from dotenv import load_dotenv
load_dotenv()

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(".."))))

import operator
from typing import List, TypedDict
from typing_extensions import Annotated
import asyncio

from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel, Field

import pandas as pd

class ScoringTeam:

    # 채점팀 state
    class ScoringTeamState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add] # 유저인풋, LLM답변들을 저장
        context : str
        problem : str
        itemtype : str
        choices : str
        answer : str
        useranswer : str
        explain : str
        query : str
        reference : str
        result : dict
        item_type_cd: str
        # docs: List[Document] # DB로부터 retrive한 문서

    class Result(BaseModel):
        correct: bool = Field(..., description="정답을 맞췄으면 True, 아니면 False")
        explain: str = Field(..., description="문제의 핵심개념 설명, 채점근거, 오답에대한 설명, 참조할 url 등의 내용")

    def __init__(self, problem_id, user_answer, trimmer=None, llm=None):

        if llm:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(model="gpt-4o")

        self.problem_id = problem_id

        self.initial_state = self.ScoringTeamState(
            messages = [],
            context = "",
            problem = "",
            itemtype = "",
            choices = "",
            answer = "",
            useranswer = user_answer,
            explain = "",
            query = "",
            reference = "",
            result = {}
            # docs = []
        )

        self.members = ["problem_retriever_agent", "generate_query_for_websearch_agent", "websearch_agent", "answer_agent"]
        self.init_scoring_team()

    # 에이전트 정의
    def problem_retriever_agent(self, state: ScoringTeamState) -> ScoringTeamState:
        print("[Agent] : problem retriever agent")

        quiz_pd = pd.read_csv("/app/tutor/aita/quiz_item_cpp.csv", encoding="utf-8")
        quiz_ids = self.problem_id
        cols = ["item_id", "item_content", "item_choices", "item_answer", "item_explain", "item_type_cd"]
        quiz = quiz_pd.loc[quiz_pd["item_id"] == quiz_ids, cols]

        return {
            "problem" : quiz["item_content"].values[0],
            "item_type_cd" : quiz["item_type_cd"].values[0],
            "choices" : quiz["item_choices"].values[0],
            "answer" : quiz["item_answer"].values[0],
            "explain" : quiz["item_explain"].values[0]
        } # type: ignore

    def generate_query_for_websearch_agent(self, state: ScoringTeamState) -> ScoringTeamState:
        print("[Agent] : generate werbseach query agent")

        class OutputQuery(BaseModel):
            query: str

        prompt = PromptTemplate(
            template=
            """
                당신은 웹에서 필요한 자료를 수집하는 정보 수집 전문가입니다.
                주어진 질문에 답변 및 채점근거를 생성 하기 위해 검색을 한다고 가정할 때, 웹으로부터 정확한 정보를 얻을 수 있도록 적절한 검색어를 한국어로 생성해 주세요.
                별도의 다른말 없이 하나의 검색어만 반환해주세요.

                질문: {question}
            """,
            input_variables=["question"]
        )

        chain = prompt | self.llm.with_structured_output(OutputQuery)
        result = chain.invoke({"question": state['problem']})

        return {
            "query" : result
        } # type: ignore

    def websearch_agent(self, state: ScoringTeamState) -> ScoringTeamState:
        print("[Agent] : websearch agent")

        websearch_tool = TavilySearch(
            max_results=3,
            topic="general",
            include_answer=True
        )

        result = websearch_tool.invoke(str(state['query']))

        return {
            "explain" : result['answer'],
            "reference" : result['results'][0]['url']
        } # type: ignore

    def answer_agent(self, state: ScoringTeamState) -> Result:
        print("[Agent] : answer")

        prompt = PromptTemplate(
            template=
            """
                당신은 시험문제를 채점하고, 채점근거 및 오답에 대한 피드백을 작성하는 조교입니다.
                다음 주어진 문제, 정답, 사용자 답안, 보기, 해설을 참조하세요.

                문제를 맞췄는지 틀렸는지 판단하고 문제의 핵심 개념, 채점 근거, 피드백에 대한 내용을 갖고있는 지식 또는 웹서치를 통해 매우 자세하게 작성해주세요.
                만약 보기가 있는 문제라면, 정답을 제외한 나머지 보기들은 왜 틀렸는지도 근거를 작성해주세요. 보기의 값이 'nan' 혹은 'null' 값 등 빈 값이라면 보기가 없는 문제이니, 보기별 해설은 작성하지 않아도 됩니다.
                마지막으로 설명의 끝에 관련 내용을 참고할 수 있는 url을 함께 넣어주세요.
                참고 url 주소 : {reference}

                문제 : {problem}
                정답 : {correct_answer}
                사용자 답안 : {user_answer}
                보기 : {choices}
                설명 : {explain}

                답변은 한국어로 생성해 주세요.
            """,
            input_variables=["question"]
        )

        chain = prompt | self.llm.with_structured_output(self.Result)
        result = chain.invoke({
            "problem": state['problem'],
            "correct_answer": state['answer'],
            "user_answer": state['useranswer'],
            "choices": state['choices'],
            "explain": state['explain'],
            "reference": state['reference']
        })

        return {
            "result" : result.model_dump()
        } # type: ignore
        
    # 그래프 구성
    def init_scoring_team(self):

        builder = StateGraph(self.ScoringTeamState)

        builder.add_node("problem_retriever_agent", self.problem_retriever_agent)
        builder.add_node("generate_query_for_websearch_agent", self.generate_query_for_websearch_agent)
        builder.add_node("websearch_agent", self.websearch_agent)
        builder.add_node("answer_agent", self.answer_agent)

        builder.set_entry_point("problem_retriever_agent")

        builder.add_edge(START, "problem_retriever_agent")
        builder.add_edge("problem_retriever_agent", "generate_query_for_websearch_agent")
        builder.add_edge("generate_query_for_websearch_agent", "websearch_agent")
        builder.add_edge("websearch_agent", "answer_agent")
        builder.add_edge("answer_agent", END)

        self.app = builder.compile()

    def run(self):

        result = self.app.invoke(self.initial_state, config={"run_name": "ScoringTeam", "verbose": True})

        return result

def run_scoring_instance(problem_id, user_answer):
    """단일 채점 인스턴스를 동기적으로 실행하는 헬퍼 함수"""
    scoring = ScoringTeam(problem_id, user_answer)
    result = scoring.run()
    return result["result"]

async def RunScoringAgentTeam(answers):
    """여러 답안을 비동기적으로 채점합니다."""
    
    tasks = []
    for info in answers.values():
        problem_id = info['problem_id']
        user_answer = info['user_answer']
        # run_in_executor를 사용하여 동기 함수를 비동기적으로 실행
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(
            None, run_scoring_instance, problem_id, user_answer
        )
        tasks.append(task)
        
    results = await asyncio.gather(*tasks)
    
    # 원래의 인덱스와 결과를 다시 매핑
    result_dict = {key: result for key, result in zip(answers.keys(), results)}
    
    return result_dict

# RunScoringAgentTeam 파라미터 예시
# {
#     1: {'problem_id' : "ITEM-506808-25b80d", 'user_answer' : '2'},
#     2: {'problem_id' : "ITEM-506808-f65a58", 'user_answer' : '2'},
#     3: {'problem_id' : "ITEM-506808-7de1a9", 'user_answer' : '2'},
#     4: {'problem_id' : "ITEM-506808-6b3f31", 'user_answer' : '2'},
#     5: {'problem_id' : "ITEM-506808-206465", 'user_answer' : '2'}
# }