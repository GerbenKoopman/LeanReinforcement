from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "lean_reinforcement.agent.mcts.mcts_cy.base_mcts_cy",
        ["lean_reinforcement/agent/mcts/mcts_cy/base_mcts_cy.pyx"],
        extra_compile_args=["-O3"],
    ),
    Extension(
        "lean_reinforcement.agent.mcts.mcts_cy.alphazero_cy",
        ["lean_reinforcement/agent/mcts/mcts_cy/alphazero_cy.pyx"],
        extra_compile_args=["-O3"],
    ),
    Extension(
        "lean_reinforcement.agent.mcts.mcts_cy.guidedrollout_cy",
        ["lean_reinforcement/agent/mcts/mcts_cy/guidedrollout_cy.pyx"],
        extra_compile_args=["-O3"],
    ),
    Extension(
        "ReProver.common_cy",
        ["ReProver/common_cy.pyx"],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="lean_reinforcement_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
        annotate=True,
    ),
    include_dirs=[np.get_include()],
)
