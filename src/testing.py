"""
Test memory usage on Snellius of simple pipeline.
"""

from ReProver.common import Pos

from src.utilities.dataloader import LeanDataLoader
from src.utilities.gym import LeanDojoEnv

from src.agent.premise_selection import PremiseSelector
from src.agent.tactic_generation import TacticGenerator
from src.agent.value_head import ValueHead

dataloader = LeanDataLoader()

thm_data = dataloader.train_data[0]
theorem = dataloader.extract_theorem(thm_data)
theorem_pos = Pos(*thm_data["start"])

premises = dataloader.get_premises(theorem, theorem_pos)

env = LeanDojoEnv(theorem, theorem_pos)

state = str(env.current_state)

premise_selector = PremiseSelector()

retrieved_premises = premise_selector.retrieve(state, premises, 10)

tactic_generator = TacticGenerator()

generated_tactic = tactic_generator.generate_tactics_with_probs(
    state, retrieved_premises
)

value_head = ValueHead()

value = value_head.predict(state, retrieved_premises)

print(value)
