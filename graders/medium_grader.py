from ..models import State

def grade(final_state: State) -> float:
    """
    MEDIUM Grader: Feature Prioritization
    - Based on final market_score (max score 1.0 at market_score >= 60)
    - Penalize failure to survive
    - Penalize bad actions (inefficiency)
    """
    if final_state.budget <= 0:
        return 0.0
    
    # 1. Base Score from Market Presence
    market_potential = min(1.0, final_state.market_score / 60.0)
    
    # 2. Efficiency Bonus (Budget remaining as percentage of initial)
    budget_ratio = final_state.budget / final_state.initial_budget
    efficiency_bonus = min(0.2, budget_ratio * 0.2)
    
    # 3. Penalties for bad actions (0.1 per action)
    penalty = final_state.bad_actions * 0.1
    
    score = (market_potential * 0.8) + efficiency_bonus - penalty
    
    return max(0.0, min(1.0, round(score, 2)))
