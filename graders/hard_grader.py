from ..models import State

def grade(final_state: State) -> float:
    """
    HARD Grader: Market Competition Survival
    Weighted score:
    - 60% based on market_score (max 1.0 if market_score >= 80)
    - 20% based on budget remaining
    - 20% based on competitor pressure (lower is better)
    - Penalty: bad_actions
    """
    if final_state.budget <= 0:
        return 0.0
    
    # 1. Market Component (0 to 1.0)
    market_comp = min(1.0, final_state.market_score / 80.0)
    
    # 2. Budget Component (0 to 1.0)
    # Remaining budget relative to initial (capping at 1.0 if they somehow make money)
    budget_comp = min(1.0, final_state.budget / final_state.initial_budget)
    
    # 3. Competitor Component (0 to 1.0)
    # Inverse of competitor pressure
    comp_comp = (100.0 - final_state.competitor_pressure) / 100.0
    
    # Combined weighted score
    score = (market_comp * 0.6) + (budget_comp * 0.2) + (comp_comp * 0.2)
    
    # Penalty for bad decisions (0.1 each)
    penalty = final_state.bad_actions * 0.1
    
    final_score = score - penalty
    
    return max(0.0, min(1.0, round(final_score, 2)))
