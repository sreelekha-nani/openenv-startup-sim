from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Action(BaseModel):
    name: str = Field(..., description="Action name: hire, fire, invest, build_feature, ignore_market")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the action (e.g., {'n': 2}, {'amount': 5000}, {'name': 'AI Chat'})")

class Observation(BaseModel):
    budget: int
    employees: int
    product_features: List[str]
    market_score: float # 0-100
    competitor_pressure: float # 0-100
    productivity: float # 0-100
    morale: float # 0-100
    funding_round: str # Seed, Series A, etc.
    active_events: List[str]
    last_action_result: str
    step: int

class State(BaseModel):
    budget: int
    initial_budget: int
    employees: int
    product_features: List[str]
    market_score: float
    competitor_pressure: float
    productivity: float
    morale: float
    funding_round: str
    steps_taken: int
    max_steps: int
    bad_actions: int

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]
