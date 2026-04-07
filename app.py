from fastapi import FastAPI, HTTPException
from startup_env.env import StartupEnv
from startup_env.models import Action
from startup_env.tasks.easy import get_task as get_easy
from startup_env.tasks.medium import get_task as get_medium
from startup_env.tasks.hard import get_task as get_hard

app = FastAPI(title="Founder Brain AI Server")

# Global environment instance
env = StartupEnv()

@app.get("/reset")
def reset(task: str = "default"):
    """
    Resets the environment. Optional 'task' query param: easy, medium, hard.
    """
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
    Returns the full internal state of the environment.
    """
    return env.state()


@app.post("/step")
def step(action: Action):
    """
    Executes a step in the environment.
    """
    try:
        result = env.step(action)
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


# ✅ REQUIRED ENTRY POINT
if __name__ == "__main__":
    main()