import random
from typing import List, Dict, Any, Optional
from .models import Action, Observation, State, StepResult

class StartupEnv:
    """
    Detailed OpenEnv simulation of a startup founder decision process.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_steps = self.config.get("max_steps", 24) # Default 24 months
        self.initial_budget = self.config.get("initial_budget", 100000)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            random.seed(seed)
        
        self.initial_budget = self.config.get("initial_budget", 100000)
        self.budget = self.initial_budget
        self.employees = self.config.get("initial_employees", 2)
        self.product_features: List[str] = []
        self.market_score = self.config.get("initial_market_score", 10.0)
        self.competitor_pressure = self.config.get("initial_competitor_pressure", 20.0)
        self.productivity = self.config.get("initial_productivity", 70.0)
        self.morale = self.config.get("initial_morale", 80.0)
        self.funding_round = "Pre-seed"
        self.active_events: List[str] = []
        self.steps_taken = 0
        self.bad_actions = 0
        self.last_action_msg = "Startup initiated. You are the founder."
        
        return self._get_observation()

    def state(self) -> State:
        return State(
            budget=int(self.budget),
            initial_budget=int(self.initial_budget),
            employees=self.employees,
            product_features=self.product_features,
            market_score=self.market_score,
            competitor_pressure=self.competitor_pressure,
            productivity=self.productivity,
            morale=self.morale,
            funding_round=self.funding_round,
            steps_taken=self.steps_taken,
            max_steps=self.max_steps,
            bad_actions=self.bad_actions
        )

    def _get_observation(self) -> Observation:
        return Observation(
            budget=int(self.budget),
            employees=self.employees,
            product_features=self.product_features,
            market_score=round(self.market_score, 2),
            competitor_pressure=round(self.competitor_pressure, 2),
            productivity=round(self.productivity, 2),
            morale=round(self.morale, 2),
            funding_round=self.funding_round,
            active_events=self.active_events,
            last_action_result=self.last_action_msg,
            step=self.steps_taken
        )

    def step(self, action: Action) -> StepResult:
        self.steps_taken += 1
        step_reward = 0.0
        
        # 1. Action Execution & Immediate Feedback
        action_name = action.name.lower()
        args = action.args

        handler_map = {
            "hire": self._handle_hire,
            "fire": self._handle_fire,
            "invest": self._handle_invest,
            "build_feature": self._handle_build_feature,
            "ignore_market": self._handle_ignore_market,
            "pitch_investors": self._handle_pitch_investors,
            "train_employees": self._handle_train_employees,
            "team_building": self._handle_team_building,
            "aggressive_expansion": self._handle_aggressive_expansion,
        }

        if action_name in handler_map:
            reward_inc = handler_map[action_name](args)
            step_reward += reward_inc
        else:
            self.last_action_msg = f"Invalid action: {action_name}."
            step_reward -= 0.5

        # 2. Dynamic Update (Simulation Rules)
        self._handle_market_events()
        
        # Burn rate: base + employee cost. Morale affects efficiency
        morale_factor = 1.5 - (self.morale / 100.0) # 0.5 to 1.5 multiplier
        burn_rate = ((self.employees * 4000) + 1000) * morale_factor
        self.budget -= burn_rate
        
        # Revenue: depends on market score, employee count, and productivity
        productivity_factor = self.productivity / 100.0
        revenue = (self.market_score / 100.0) * (self.employees * 2500) * productivity_factor
        self.budget += revenue
        
        # Market Score & Competitor Pressure
        decay_resistance = self.productivity / 50.0
        self.market_score = max(0.0, self.market_score - (self.competitor_pressure / (20.0 * decay_resistance)))
        self.competitor_pressure = min(100.0, self.competitor_pressure + random.uniform(0.5, 3.0))

        # Natural decay of morale and productivity
        self.morale = max(0.0, self.morale - random.uniform(1.0, 3.0))
        self.productivity = max(0.0, self.productivity - random.uniform(0.5, 2.0))

        # 3. Operational Efficiency Rewards (Scaling)
        profitability = (revenue - burn_rate) / 10000.0
        step_reward += max(-0.5, min(0.5, profitability))
        
        # Market Presence Bonus
        step_reward += (self.market_score / 200.0)

        # 4. End Condition & Final Penalties
        done = False
        if self.budget <= 0:
            done = True
            self.last_action_msg += " BANKRUPT! Game Over."
            step_reward -= 5.0
        elif self.steps_taken >= self.max_steps:
            done = True
            self.last_action_msg += " Horizon reached."

        return StepResult(
            observation=self._get_observation(),
            reward=round(step_reward, 2),
            done=done,
            info={
                "burn_rate": burn_rate,
                "revenue": revenue,
                "profitability": revenue - burn_rate
            }
        )

    def _handle_hire(self, args: Dict[str, Any]) -> float:
        n = args.get("n", 1)
        cost = n * 10000
        if self.budget >= cost:
            self.budget -= cost
            self.employees += n
            self.last_action_msg = f"Hired {n} new employees. Budget -{cost}."
            return 0.1 * n
        else:
            self.last_action_msg = "FAILED: Insufficient budget for hiring."
            self.bad_actions += 1
            return -1.0

    def _handle_fire(self, args: Dict[str, Any]) -> float:
        n = min(args.get("n", 1), self.employees)
        severance = n * 5000
        self.budget -= severance
        self.employees -= n
        self.last_action_msg = f"Fired {n} employees. Paid {severance} severance."
        return -0.5 * n

    def _handle_invest(self, args: Dict[str, Any]) -> float:
        amount = args.get("amount", 0)
        if self.budget >= amount and amount > 0:
            self.budget -= amount
            efficiency = 1.0 + (self.employees * 0.1)
            score_gain = (amount / 10000) * efficiency
            self.market_score = min(100.0, self.market_score + score_gain)
            self.last_action_msg = f"Invested {amount} into growth. Score +{score_gain:.2f}."
            return 0.3 + (score_gain * 0.1)
        else:
            self.last_action_msg = "FAILED: Invalid investment amount or budget."
            self.bad_actions += 1
            return -1.0

    def _handle_build_feature(self, args: Dict[str, Any]) -> float:
        feature_name = args.get("name", "New Feature")
        cost = 5000 + (len(self.product_features) * 2000)
        if self.budget >= cost and self.employees > 0:
            self.budget -= cost
            self.product_features.append(feature_name)
            self.market_score = min(100.0, self.market_score + 5.0)
            self.competitor_pressure = max(0.0, self.competitor_pressure - 2.0)
            self.last_action_msg = f"Built feature: {feature_name}. Cost: {cost}."
            return 0.5 
        else:
            self.last_action_msg = "FAILED: Cannot build feature (no budget or no team)."
            self.bad_actions += 1
            return -1.0

    def _handle_ignore_market(self, args: Dict[str, Any]) -> float:
        self.last_action_msg = "Ignored market. Competitors are gaining."
        self.competitor_pressure = min(100.0, self.competitor_pressure + 5.0)
        return -0.2

    def _handle_pitch_investors(self, args: Dict[str, Any]) -> float:
        success_chance = (self.market_score / 100.0) * 0.5 + (min(self.employees, 20) / 40.0)
        if random.random() < success_chance:
            funding_amounts = {
                "Pre-seed": 50000,
                "Seed": 200000,
                "Series A": 1000000,
                "Series B": 5000000
            }
            rounds = ["Pre-seed", "Seed", "Series A", "Series B", "Series C"]
            current_idx = rounds.index(self.funding_round)
            if current_idx < len(rounds) - 1:
                amount = funding_amounts.get(self.funding_round, 50000)
                self.budget += amount
                self.funding_round = rounds[current_idx + 1]
                self.last_action_msg = f"SUCCESS: Raised {amount} in {self.funding_round} round!"
                return 2.0
            else:
                self.last_action_msg = "Already reached max funding rounds."
                return -0.1
        else:
            self.last_action_msg = "FAILED: Investors were not impressed."
            self.morale -= 5.0
            return -0.5

    def _handle_train_employees(self, args: Dict[str, Any]) -> float:
        cost = self.employees * 2000
        if self.budget >= cost:
            self.budget -= cost
            gain = random.uniform(5.0, 15.0)
            self.productivity = min(100.0, self.productivity + gain)
            self.last_action_msg = f"Trained team. Productivity +{gain:.1f}%. Cost: {cost}."
            return 0.3
        else:
            self.last_action_msg = "FAILED: Insufficient budget for training."
            return -1.0

    def _handle_team_building(self, args: Dict[str, Any]) -> float:
        cost = self.employees * 1000
        if self.budget >= cost:
            self.budget -= cost
            gain = random.uniform(10.0, 20.0)
            self.morale = min(100.0, self.morale + gain)
            self.last_action_msg = f"Team building event. Morale +{gain:.1f}%. Cost: {cost}."
            return 0.2
        else:
            self.last_action_msg = "FAILED: Insufficient budget for team building."
            return -1.0

    def _handle_aggressive_expansion(self, args: Dict[str, Any]) -> float:
        cost = 50000
        if self.budget >= cost:
            self.budget -= cost
            if random.random() < 0.4:
                gain = random.uniform(15.0, 30.0)
                self.market_score = min(100.0, self.market_score + gain)
                self.last_action_msg = f"Aggressive expansion SUCCEEDED! Market Score +{gain:.1f}."
                return 3.0
            else:
                loss = random.uniform(5.0, 15.0)
                self.market_score = max(0.0, self.market_score - loss)
                self.competitor_pressure = min(100.0, self.competitor_pressure + 10.0)
                self.last_action_msg = f"Aggressive expansion FAILED. Market Score -{loss:.1f}."
                return -2.0
        else:
            self.last_action_msg = "FAILED: Insufficient budget for aggressive expansion."
            return -1.0

    def _handle_market_events(self):
        """
        Random events that can happen in the market.
        """
        self.active_events = []
        event_chance = 0.2
        
        if random.random() < event_chance:
            events = [
                ("Tech Boom", "Increased market score and investor interest.", {"market_score": 10, "morale": 5}),
                ("Economic Recession", "Decreased market score and morale.", {"market_score": -15, "morale": -10}),
                ("Viral Marketing", "Sudden boost in market score.", {"market_score": 20}),
                ("Data Breach", "Significant hit to market score and morale.", {"market_score": -20, "morale": -15}),
                ("Competitor Pivot", "Increased competitor pressure.", {"competitor_pressure": 15}),
                ("Industry Regulation", "Increased burn rate due to compliance.", {"burn_rate_mult": 1.2}),
                ("Talent War", "Decreased morale and productivity.", {"morale": -5, "productivity": -10})
            ]
            
            name, desc, effects = random.choice(events)
            self.active_events.append(name)
            self.last_action_msg += f" [EVENT: {name}] {desc}"
            
            if "market_score" in effects:
                self.market_score = max(0.0, min(100.0, self.market_score + effects["market_score"]))
            if "morale" in effects:
                self.morale = max(0.0, min(100.0, self.morale + effects["morale"]))
            if "productivity" in effects:
                self.productivity = max(0.0, min(100.0, self.productivity + effects["productivity"]))
            if "competitor_pressure" in effects:
                self.competitor_pressure = max(0.0, min(100.0, self.competitor_pressure + effects["competitor_pressure"]))
            if "burn_rate_mult" in effects:
                self.budget -= (self.employees * 1000) * (effects["burn_rate_mult"] - 1.0)
