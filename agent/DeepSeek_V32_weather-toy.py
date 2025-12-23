'''
Output of this file:

Turn 1.1
reasoning_content="The user wants to know the weather in Hangzhou tomorrow. I need to get tomorrow's date and then use the get_weather function. First, I'll get the current date to calculate tomorrow's date. Let's use get_date."
content=''
tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_00_MYjXlzLn6lTXWIAuTbL2lw85', function=Function(arguments='{}', name='get_date'), type='function', index=0)]
tool result for get_date: 2025-12-01

Turn 1.2
reasoning_content='Today is December 1, 2025. Tomorrow will be December 2, 2025. So the date should be "2025-12-02". Now I need to call get_weather with location "Hangzhou" and date "2025-12-02". Let\'s do that.'
content=''
tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_00_ZN7oZwHk4y2Nax84bZ8ia8mC', function=Function(arguments='{"location": "Hangzhou", "date": "2025-12-02"}', name='get_weather'), type='function', index=0)]
tool result for get_weather: Cloudy 7~13°C

Turn 1.3
reasoning_content="I have the weather information: Cloudy with temperatures between 7 and 13°C. I'll present this to the user in a friendly manner. I'll also mention that this is for tomorrow, December 2nd. Let's respond."
content="Tomorrow (December 2, 2025) in Hangzhou, the weather will be **cloudy** with temperatures ranging from **7°C to 13°C**. It'll be a bit cool, so you might want to dress in layers."
tool_calls=None
'''

import os
import json
from openai import OpenAI

# The definition of the tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "Get the current date",
            "parameters": { "type": "object", "properties": {} },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location, the user should supply the location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": { "type": "string", "description": "The city name" },
                    "date": { "type": "string", "description": "The date in format YYYY-mm-dd" },
                },
                "required": ["location", "date"]
            },
        }
    },
]

# The mocked version of the tool calls
def get_date_mock():
    return "2025-12-01"

def get_weather_mock(location, date):
    return "Cloudy 7~13°C"

TOOL_CALL_MAP = {
    "get_date": get_date_mock,
    "get_weather": get_weather_mock
}

def clear_reasoning_content(messages):
    for message in messages:
        if hasattr(message, 'reasoning_content'):
            message.reasoning_content = None

def run_turn(turn, messages):
    sub_turn = 1
    while True:
        response = client.chat.completions.create(
            model='deepseek-chat',
            messages=messages,
            tools=tools,
            extra_body={ "thinking": { "type": "enabled" } }  # 使用 OpenAI SDK 的 thinking 功能
        )
        messages.append(response.choices[0].message)
        reasoning_content = response.choices[0].message.reasoning_content  # 思维链内容，与 content 同级
        content = response.choices[0].message.content  # 最终回答内容
        tool_calls = response.choices[0].message.tool_calls  # 模型工具调用
        print(f"Turn {turn}.{sub_turn}\n{reasoning_content=}\n{content=}\n{tool_calls=}")
        # If there is no tool calls, then the model should get a final answer and we need to stop the loop
        if tool_calls is None:
            break  # 不是 tool call 的情况解释一个完整的 turn，所以要不是 content 有内容要不就是 tool call 有内容
        for tool in tool_calls:
            tool_function = TOOL_CALL_MAP[tool.function.name]
            tool_result = tool_function(**json.loads(tool.function.arguments))
            print(f"tool result for {tool.function.name}: {tool_result}\n")
            messages.append({
                "role": "tool",
                "tool_call_id": tool.id,
                "content": tool_result,
            })
        sub_turn += 1

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url='https://api.deepseek.com/v1',
)

# The user starts a question
turn = 1
messages = [{
    "role": "user",
    "content": "How's the weather in Hangzhou Tomorrow"
}]
run_turn(turn, messages)

# The user starts a new question
turn = 2
messages.append({
    "role": "user",
    "content": "How's the weather in Hangzhou Tomorrow"
})
# We recommended to clear the reasoning_content in history messages so as to save network bandwidth
clear_reasoning_content(messages)
run_turn(turn, messages)