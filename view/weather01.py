import os
import json
import requests
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from dotenv import load_dotenv

import re

load_dotenv() #加载环境变量
@tool
def multiply(input_str:str)->int:
    """2个数的乘法
        Args:
            input_str:输入的字符串，格式为a*b
        returns:int
    """
    match = re.findall(r'\d+', input_str)
    if len(match) == 2:
        a,b = map(int,match)
        return a*b

#天气查询
def get_weather(city:str)->str:
    """查询指定城市的当前天气信息，包括温度，天气状况，湿度，风速
    当用户查询某个城市或区域的天气、气温、冷不冷、热不热、下不下雨时，使用此工具
    Args:
        city：要查询天气的城市名称，例如“北京”、“上海”、“深圳”
    """
    try:
        url = f"https://wttr.in/{city}?format=j1&lang=zh"
        resp=requests.get(url,timeout=5,headers={'User-Agent':'curl/7.68.0'})
        resp.raise_for_status()
        data = resp.json()
        current=data["current_condition"][0]
        area=current["nearest_area"]
        return (
            f"【{area['areaName'][0]['value']}天气】\n"
            f"天气{current['weatherDesc'][0]['value']}\n"
            f"温度{current['temp_C']}.C(体感{current['FeelsLikeC']}.C\n"
            f"湿度{current['humidity']}%\n"
            f"风速{current['windspeedKmph']}km/h"
        )
    except Exception :
        return (
            f"【{city}天气】（模拟数据，网络不通时显示）\n"
            f"天气：晴.温度：28C（体感30C）湿度：65% 风速：15km/h"
        )

tools = [get_weather,multiply]

llm=ChatOpenAI(
    model=os.getenv("DATA_MODEL"),
    base_url=os.getenv("DATA_BASE_URL"),
    api_key=os.environ.get('SILICONFLOW_API_KEY'),
    temperature=0
)
print(llm)

agent = create_agent(
    model= llm,
    tools=tools,
    system_prompt="""
        你是天气小助手，专业友好
            1，问天气必须调用get_weather工具，不要编造
            2，回答简介自然，根据温度给出穿衣建议（<10度保暖，>30度防暑）
            3，用户没有说城市时礼貌询问
            4，非天气问题礼貌说明主要提供天气服务
    """
)

#记忆配置
class WeatherChat:
    """对论对话会话，自己维护消息历史"""
    def __init__(self):
        self.messages = []

    def send(self, user_input:str)->str:
        """发送用户消息，返回助手回答"""
        #1 把用户的消息叫人历史
        self.messages.append(HumanMessage(content=user_input))
        #2调用agent (传入完整历史)
        result = agent.invoke({"messages":self.messages})
        #3用agent返回的完整消息队列列表更新历史（包含工具调用过程）
        self.messages = result["messages"]
        #4 最后一条就是最终回答
        return self.messages[-1].content
    def clear(self):
        """清空对话"""
        self.messages = []

if __name__ == "__main__":
    print("="*50)
    print(" 天气助手V2 （多轮对话版本")
    print("命令：quit 退出 | clear 清空对话")

    chat = WeatherChat()

    while True:
        user_input = input("\n您：").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit","exit","q"):
            print("再见")
            break
        if user_input.lower() == "clear":
            chat.clear()
            print("对话已清空，重新开始")
            continue

        answer = chat.send(user_input)
        print(f"助手：{answer}")
