from ..env import StartupEnv

def get_task():
    config = {
        "name": "EASY — Budget Allocation",
        "description": "Objective: Survive 5 steps without losing all money. Manage your burn rate.",
        "initial_budget": 200000,
        "max_steps": 5,
        "initial_employees": 2,
        "initial_market_score": 10.0,
        "initial_competitor_pressure": 10.0
    }
    return StartupEnv(config)
