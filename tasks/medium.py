from env import StartupEnv

def get_task():
    config = {
        "name": "MEDIUM — Feature Prioritization",
        "description": "Objective: Build best features to maximize market_score within 12 months.",
        "initial_budget": 100000,
        "max_steps": 12,
        "initial_employees": 3,
        "initial_market_score": 15.0,
        "initial_competitor_pressure": 20.0
    }
    return StartupEnv(config)
