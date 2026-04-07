from ..env import StartupEnv

def get_task():
    config = {
        "name": "HARD — Market Competition Survival",
        "description": "Objective: Beat competitor pressure and reach market_score > 80 within 24 months.",
        "initial_budget": 50000,
        "max_steps": 24,
        "initial_employees": 4,
        "initial_market_score": 5.0,
        "initial_competitor_pressure": 50.0
    }
    return StartupEnv(config)
