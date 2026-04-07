from ..models import State

def grade(final_state: State) -> float:
    """
    EASY Grader: Survival
    - Survived 5 steps -> 1.0
    - Failed early -> steps_taken / max_steps (proportional penalty)
    """
    if final_state.budget > 0:
        score = 1.0
    else:
        # Penalize failing early
        score = final_state.steps_taken / final_state.max_steps
    
    # Penalize bad actions (each bad action removes 0.05 from score)
    score = max(0.0, score - (final_state.bad_actions * 0.05))
    
    return round(score, 2)
