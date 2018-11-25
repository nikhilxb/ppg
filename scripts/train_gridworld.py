import argparse
import pendulum
from ppg.grammar import PolicyGrammar

# ==================================================================================================
# Command-line arguments.

parser = argparse.ArgumentParser()

# Title.
# ------

parser.add_argument(
    "--experiment_name",
    default="untitled-{}".format(pendulum.now("America/Los_Angeles").strftime("%Y-%m-%d-%H-%M-%S")),
)

parser.add_argument("--arg_name", type=int, default=1)

# ==================================================================================================
# Policy grammar definition.

pg = PolicyGrammar(
    primitives=[
        "UseToolshed",
        "UseWorkbench",
        "UseFactory",
        "GetIron",
        "GetGrass",
        "GetWood",
    ],
    goals=[
        "MakePlank",
        "MakeStick",
        "MakeCloth",
        "MakeRope",
        "MakeBridge",
        "MakeShears",
        "MakeAxe",
        "MakeBed",
        "MakeLadder",
    ],
)

pi, g = pg.get_tokens()
pg.add_productions(
    "MakePlank",
    [pi["GetWood"], pi["UseToolshed"]],
)
pg.add_productions(
    "MakeStick",
    [pi["GetWood"], pi["UseWorkbench"]],
)
pg.add_productions(
    "MakeCloth",
    [pi["GetGrass"], pi["UseFactory"]],
)
pg.add_productions(
    "MakeRope",
    [pi["GetGrass"], pi["UseToolshed"]],
)
pg.add_productions(
    "MakeBridge",
    [pi["GetIron"], pi["GetWood"], pi["UseFactory"]],
)
pg.add_productions(
    "MakeShears",
    [g["MakeStick"], pi["UseToolshed"]],
)
pg.add_productions(
    "MakeAxe",
    [g["MakeStick"], pi["GetIron"], pi["UseToolshed"]],
)
pg.add_productions(
    "MakeBed",
    [g["MakePlank"], pi["GetGrass"], pi["UseWorkbench"]],
)
pg.add_productions(
    "MakeLadder",
    [g["MakePlank"], g["MakeStick"], pi["UseFactory"]],
)

print(pg)

# ==================================================================================================
# Training functions.


def main(args):
    pass


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)