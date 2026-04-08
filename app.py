import os
import sys
from fastapi import FastAPI, HTTPException

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import StartupEnv
from models import Action
from tasks.easy import get_task as get_easy
from tasks.medium import get_task as get_medium
from tasks.hard import get_task as get_hard

app = FastAPI(title="Founder Brain AI Server")

# Global environment instance
env = StartupEnv()

# ✅ ROOT ROUTE (404 problem solve avthundi)
@app.get("/")
def home():
    return {"message": "Startup Env API is running 🚀"}


@app.get("/reset")
def reset(task: str = "default"):
    """
    Resets the environment.
    """
    print("START")  # Structured Log

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


@app.get("/state")
def get_state():
    """
    Returns full state
    """
    return env.state()


@app.post("/step")
def step(action: Action):
    """
    Executes step
    """
    print("STEP")  # Structured Log

    try:
        result = env.step(action)

        if result.done:
            print("END")  # Structured Log

        return {
            "observation": result.observation,
            "reward": result.reward,
            "done": result.done,
            "info": result.info
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ✅ REQUIRED FOR OPENENV
def main():
    return app


# ✅ LOCAL RUN (HF ki effect undadu)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)