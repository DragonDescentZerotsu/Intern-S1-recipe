import dotenv
import logfire
from pydantic import BaseModel
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.settings import ModelSettings
# for local vllm host
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from vllm import LLM
from pydantic_ai.models.outlines import OutlinesModel

from rdkit import Chem
from rdkit.Chem import Descriptors



# dotenv.load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()


class MoleculeInformation(BaseModel):
    logp: float
    mass: float
    description: str


def get_logp(smiles: str) -> float:
    """Calculate the LogP of a molecule given its SMILES string.

    Args:
        smiles (str): The SMILES representation of the molecule.

    Returns:
        float: The LogP value of the molecule.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    return Descriptors.MolLogP(mol)  # ty:ignore[unresolved-attribute]


def get_mass(smiles: str) -> float:
    """Calculate the molecular weight of a molecule given its SMILES string.

    Args:
        smiles (str): The SMILES representation of the molecule.

    Returns:
        float: The molecular weight of the molecule.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    return Descriptors.MolWt(mol)  # ty:ignore[unresolved-attribute]


model = OpenRouterModel(
    "deepseek/deepseek-v3.2",
    settings=OpenRouterModelSettings(
        openrouter_reasoning={
            "effort": "high",
            "enabled": True,
        }
    ),
)

# for local vllm host
# local_model = OpenAIChatModel(
#     "local-model",
#     provider=OpenAIProvider(
#         base_url="http://127.0.0.1:8000/v1",  # vLLM 的 OpenAI 兼容地址
#         api_key="EMPTY",                      # 你 vLLM serve 里用的 --api-key EMPTY
#     ),
#     # settings=OpenRouterModelSettings(
#     #     openrouter_reasoning={
#     #         "effort": "high",
#     #         "enabled": True,
#     #     }
#     # ),
# )

# vllm 的 Outlines 模式不能使用 tool call，要用本地 model 的监控请使用 langfuse
# local_model = OutlinesModel.from_vllm_offline(
#     LLM(
#         model='checkpoints/Intern-S1-mini/full/sft-distill_DeepSeek_V32-TDC-train-set/checkpoint-30000',
#         trust_remote_code=True
#         )
# )

rdkit_agent: Agent[str, MoleculeInformation] = Agent(
    local_model,
    output_type=MoleculeInformation,
    # instructions="Given a smiles string, use your tools to find out information about it and report back what you think.",
    tools=[
        get_logp,
        get_mass,
    ],
)

run_result = rdkit_agent.run_sync(
    "Given a smiles string, use your tools to find out information about it and report back what you think: CC(=O)OC1=CC=CC=C1C(=O)O", 
    model_settings=ModelSettings(extra_body=dict(spaces_between_special_tokens=False, enable_thinking=True))
    )
print(run_result.output.logp)
print(run_result.output.mass)
print(run_result.output.description)