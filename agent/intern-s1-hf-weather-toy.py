      
from openai import OpenAI
import json


def get_current_temperature(location: str, unit: str = "celsius"):
    """Get current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, and the unit in a dict
    """
    return {
        "temperature": 26.1,
        "location": location,
        "unit": unit,
    }


def get_temperature_date(location: str, date: str, unit: str = "celsius"):
    """Get temperature at a location and date.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        date: The date to get the temperature for, in the format "Year-Month-Day".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, the date and the unit in a dict
    """
    return {
        "temperature": 25.9,
        "location": location,
        "date": date,
        "unit": unit,
    }

def get_function_by_name(name):
    tool_map = {
        "get_current_temperature": get_current_temperature,
        "get_temperature_date": get_temperature_date
    }
    return tool_map.get(name)

tools = [{
    'type': 'function',
    'function': {
        'name': 'get_current_temperature',
        'description': 'Get current temperature at a location.',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The location to get the temperature for, in the format \'City, State, Country\'.'
                },
                'unit': {
                    'type': 'string',
                    'enum': [
                        'celsius',
                        'fahrenheit'
                    ],
                    'description': 'The unit to return the temperature in. Defaults to \'celsius\'.'
                }
            },
            'required': [
                'location'
            ]
        }
    }
}, 
# {
#     'type': 'function',
#     'function': {
#         'name': 'get_temperature_date',
#         'description': 'Get temperature at a location and date.',
#         'parameters': {
#             'type': 'object',
#             'properties': {
#                 'location': {
#                     'type': 'string',
#                     'description': 'The location to get the temperature for, in the format \'City, State, Country\'.'
#                 },
#                 'date': {
#                     'type': 'string',
#                     'description': 'The date to get the temperature for, in the format \'Year-Month-Day\'.'
#                 },
#                 'unit': {
#                     'type': 'string',
#                     'enum': [
#                         'celsius',
#                         'fahrenheit'
#                     ],
#                     'description': 'The unit to return the temperature in. Defaults to \'celsius\'.'
#                 }
#             },
#             'required': [
#                 'location',
#                 'date'
#             ]
#         }
#     }
# }
]


openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:23333/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model_name = client.models.list().data[0].id

def run_turn(messages):
    sub_turn = 1
    while True:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            max_tokens=31000,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            extra_body=dict(spaces_between_special_tokens=False, enable_thinking=True)
            # extra_body={ "thinking": { "type": "enabled" } }
        )
        messages.append(response.choices[0].message)
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        tool_calls = response.choices[0].message.tool_calls
        
        print(f"Subturn {sub_turn}\n{reasoning_content=}\n{content=}\n{tool_calls=}")

        if tool_calls is None:
            break
            
        for tool_call in tool_calls:
            tool_call_args = json.loads(tool_call.function.arguments)
            tool_call_result = get_function_by_name(tool_call.function.name)(**tool_call_args)
            tool_call_result = json.dumps(tool_call_result, ensure_ascii=False)
            
            print(f"Tool result for {tool_call.function.name}: {tool_call_result}\n")
            
            messages.append({
                'role': 'tool',
                'name': tool_call.function.name,
                'content': tool_call_result,
                'tool_call_id': tool_call.id
            })
        sub_turn += 1

messages = [
    {'role': 'user', 'content': 'Today is 2024-11-14, What\'s the temperature in San Francisco now?'}
]

run_turn(messages)

