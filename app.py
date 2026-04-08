import os
import sys
from fastapi import FastAPI, HTTPException

# Fix import paths (important for Hugging Face)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import StartupEnv
from models import Action
from tasks.easy import get_task as get_easy
from tasks.medium import get_task as get_medium
from tasks.hard import get_task as get_hard

app = FastAPI(title="Founder Brain AI Server")

# Global environment
env = StartupEnv()


# ✅ ROOT ROUTE
@app.get("/")
def home():
    return {"message": "Startup Env API is running 🚀"}


# ✅ RESET
@app.get("/reset")
def reset(task: str = "default"):
    print("START")  # Required log

    global env

    if task == "easy":
        env = get_easy()
    elif task == "medium":
        env = get_medium()
    elif task == "hard":
        env = get_hard()
    else:
        env.reset()

    return env._get_observation()


# ✅ STATE
@app.get("/state")
def get_state():
    return env.state()


# ✅ STEP
@app.post("/step")
def step(action: Action):
    print("STEP")  # Required log

    try:
        result = env.step(action)

        if result.done:
            print("END")  # Required log

        return {
            "observation": result.observation,
            "reward": result.reward,
            "done": result.done,
            "info": result.info
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ✅ REQUIRED ENTRY FOR OPENENV
def main():
    return app