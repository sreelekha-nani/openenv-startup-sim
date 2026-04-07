import os
import json
from openai import OpenAI
from startup_env.tasks.easy import get_task as get_easy
from startup_env.tasks.medium import get_task as get_medium
from startup_env.tasks.hard import get_task as get_hard

from startup_env.graders.easy_grader import grade as grade_easy
from startup_env.graders.medium_grader import grade as grade_medium
from startup_env.graders.hard_grader import grade as grade_hard

from startup_env.models import Action

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-70b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def log_start(task_name, env_name, model_name):
    print(f"[START] task={task_name} env={env_name} model={model_name}")

def log_step(step, action, reward, done, error=None):
    done_str = "true" if done else "false"
    error_str = "null" if error is None else str(error)
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}")

def log_end(success, steps, score, rewards):
    success_str = "true" if success else "false"
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}")

def get_ai_action(obs):
    """
    Query the LLM to decide on the next action based on the observation.
    """
    prompt = f"""
    You are a startup founder. Current State:
    - Budget: {obs.budget}
    - Employees: {obs.employees}
    - Features: {obs.product_features}
    - Market Score: {obs.market_score}
    - Competitor Pressure: {obs.competitor_pressure}
    - Productivity: {obs.productivity}%
    - Morale: {obs.morale}%
    - Funding Round: {obs.funding_round}
    - Active Events: {obs.active_events}
    - Last Result: {obs.last_action_result}

    Available Actions:
    1. hire (args: {{"n": int}})
    2. fire (args: {{"n": int}})
    3. invest (args: {{"amount": int}})
    4. build_feature (args: {{"name": str}})
    5. pitch_investors (args: {{}})
    6. train_employees (args: {{}})
    7. team_building (args: {{}})
    8. aggressive_expansion (args: {{}})
    9. ignore_market (args: {{}})

    Respond ONLY with a JSON object: {{"name": "action_name", "args": {{...}}}}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" },
            temperature=0.1
        )
        data = json.loads(response.choices[0].message.content)
        return data
    except Exception:
        return {"name": "ignore_market", "args": {}}

def run_task(task_key, env_func, grader_func):
    try:
        env = env_func()
        log_start(task_key, "startup_env", MODEL_NAME)
        
        obs = env.reset()
        max_steps = 8
        rewards_list = []
        
        for i in range(1, max_steps + 1):
            try:
                action_data = get_ai_action(obs)
                action = Action(name=action_data["name"], args=action_data.get("args", {}))
                
                result = env.step(action)
                obs = result.observation
                rewards_list.append(result.reward)
                
                log_step(i, action.name, result.reward, result.done, error=None)
                
                if result.done:
                    break
            except Exception as e:
                log_step(i, "error", 0.0, True, error=str(e))
                break
                
        final_state = env.state()
        score = grader_func(final_state)
        success = final_state.budget > 0
        log_end(success, len(rewards_list), score, rewards_list)
    except Exception:
        pass

if __name__ == "__main__":
    tasks = [
        ("easy", get_easy, grade_easy),
        ("medium", get_medium, grade_medium),
        ("hard", get_hard, grade_hard)
    ]
    
    for key, env_f, grade_f in tasks:
        run_task(key, env_f, grade_f)
