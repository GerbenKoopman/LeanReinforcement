"""
Simple testing script to load and display theorem data from a traced LeanDojo repository.
"""

from dataloader import trace_repo, load_theorems

repo = trace_repo()
tactic_theorems, statement_theorems = load_theorems(repo)

print(f"Number of tactic theorems: {len(tactic_theorems)}")
print(f"Number of statement theorems: {len(statement_theorems)}\n\n")

for thm in tactic_theorems[:5]:
    statement = thm.get_theorem_statement()
    tactic_proof = thm.get_tactic_proof()

    print(f"Theorem: {statement}\n")
    print(f"Tactic Proof: {tactic_proof}\n")
    print("-----\n\n")

for thm in statement_theorems[:5]:
    statement = thm.get_theorem_statement()

    print(f"Theorem: {statement}\n")
    print("-----\n\n")
