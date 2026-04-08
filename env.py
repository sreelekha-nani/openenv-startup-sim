import random
from typing import List, Dict, Any, Optional

from models import Action, Observation, State, StepResult


class StartupEnv:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_steps = self.config.get("max_steps", 24)
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

        action_name = action.name.lower()
        args = action.args

        handler_map = {
            "hire_employee": self._handle_hire,
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
            step_reward += handler_map[action_name](args)
        else:
            self.last_action_msg = f"Invalid action: {action_name}"
            step_reward -= 5
            self.bad_actions += 1

        self._handle_market_events()

        morale_factor = 1.5 - (self.morale / 100.0)
        burn_rate = ((self.employees * 4000) + 1000) * morale_factor
        self.budget -= burn_rate

        productivity_factor = self.productivity / 100.0
        revenue = (self.market_score / 100.0) * (self.employees * 2500) * productivity_factor
        self.budget += revenue

        self.market_score = max(0.0, self.market_score - (self.competitor_pressure / 20.0))
        self.competitor_pressure = min(100.0, self.competitor_pressure + random.uniform(0.5, 3.0))

        self.morale = max(0.0, self.morale - random.uniform(1.0, 3.0))
        self.productivity = max(0.0, self.productivity - random.uniform(0.5, 2.0))

        done = self.budget <= 0 or self.steps_taken >= self.max_steps

        return StepResult(
            observation=self._get_observation(),
            reward=round(step_reward, 2),
            done=done,
            info={"burn_rate": burn_rate, "revenue": revenue}
        )

    # -------- ACTION HANDLERS -------- #

    def _handle_hire(self, args):
        if self.budget >= 50000:
            self.employees += 1
            self.budget -= 50000
            self.last_action_msg = "Hired a new employee."
            return 10
        else:
            self.last_action_msg = "Not enough budget to hire."
            return -5

    def _handle_fire(self, args):
        if self.employees > 1:
            self.employees -= 1
            self.last_action_msg = "Fired an employee."
            return -2
        else:
            self.last_action_msg = "Cannot fire more employees."
            return -5

    def _handle_build_feature(self, args):
        if self.budget >= 20000:
            feature = f"Feature_{len(self.product_features)+1}"
            self.product_features.append(feature)
            self.budget -= 20000
            self.market_score += 5
            self.last_action_msg = f"Built {feature}."
            return 15
        else:
            self.last_action_msg = "Not enough budget."
            return -5

    def _handle_invest(self, args):
        if self.budget >= 30000:
            self.budget -= 30000
            self.market_score += 10
            self.last_action_msg = "Invested in growth."
            return 12
        else:
            self.last_action_msg = "Not enough budget."
            return -5

    def _handle_ignore_market(self, args):
        self.market_score -= 5
        self.last_action_msg = "Ignored market trends."
        return -10

    def _handle_pitch_investors(self, args):
        if random.random() > 0.5:
            self.budget += 100000
            self.funding_round = "Seed"
            self.last_action_msg = "Funding raised!"
            return 25
        else:
            self.last_action_msg = "Pitch failed."
            return -5

    def _handle_train_employees(self, args):
        if self.budget >= 10000:
            self.budget -= 10000
            self.productivity += 10
            self.last_action_msg = "Employees trained."
            return 10
        else:
            self.last_action_msg = "Not enough budget."
            return -5

    def _handle_team_building(self, args):
        if self.budget >= 8000:
            self.budget -= 8000
            self.morale += 10
            self.last_action_msg = "Team morale improved."
            return 8
        else:
            self.last_action_msg = "Not enough budget."
            return -3

    def _handle_aggressive_expansion(self, args):
        if self.budget >= 70000:
            self.budget -= 70000
            self.market_score += 20
            self.competitor_pressure += 10
            self.last_action_msg = "Expansion done."
            return 20
        else:
            self.last_action_msg = "Expansion failed."
            return -10

    def _handle_market_events(self):
        if random.random() < 0.2:
            event = random.choice([
                "Market boom",
                "Competitor launch",
                "Economic slowdown"
            ])
            self.active_events.append(event)

            if event == "Market boom":
                self.market_score += 10
            elif event == "Competitor launch":
                self.competitor_pressure += 10
            elif event == "Economic slowdown":
                self.market_score -= 10