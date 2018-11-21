import argparse
import pendulum

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
# Training functions.


def main(args):
    pass


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
