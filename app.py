from fastapi import FastAPI, HTTPException

from env import StartupEnv
from models import Action

from tasks.easy import get_task as get_easy
from tasks.medium import get_task as get_medium
from tasks.hard import get_task as get_hard

app = FastAPI(title="Founder Brain AI Server")

env = StartupEnv()

@app.get("/reset")
def reset(task: str = "default"):
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
    return env.state()


@app.post("/step")
def step(action: Action):
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


def main():
    return app


if __name__ == "__main__":
    main()