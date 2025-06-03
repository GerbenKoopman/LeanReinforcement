"""
This script is taken from the LeanDojo repository and is used to demonstrate how to use LeanDojo with Lean 4.
https://github.com/lean-dojo/LeanDojo/blob/main/scripts/demo-lean4.ipynb
"""

from lean_dojo import LeanGitRepo, trace


def main():
    repo = LeanGitRepo(
        "https://github.com/leanprover-community/mathlib4",
        "29dcec074de168ac2bf835a77ef68bbe069194c5",
    )

    repo.get_config("lean-toolchain")

    traced_repo = trace(repo)

    traced_file = traced_repo.get_traced_file("Mathlib/Algebra/BigOperators/Pi.lean")

    traced_theorems = traced_file.get_traced_theorems()

    thm = traced_file.get_traced_theorem("pi_eq_sum_univ")

    print(f"Premise definitions:\n{traced_file.get_premise_definitions()}")
    print(f"Number of traced files: {len(traced_repo.traced_files)}")
    print(f"Number of traced theorems: {len(traced_theorems)}")

    if thm:
        traced_tactics = thm.get_traced_tactics()
        print(f"Traced tactics for 'pi_eq_sum_univ': {traced_tactics}")
    else:
        print("Theorem 'pi_eq_sum_univ' not found in the traced file.")


if __name__ == "__main__":
    main()
