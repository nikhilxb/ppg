# Create virtual environment
virtualenv -p python3.7 venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Export python path
export PYTHONPATH=./

# Display message for user
printf "\nThe virtual environment is now ready for use:\n"
printf "  $ source venv/bin/activate\n"
printf "  $ // use for a bit\n"
printf "  $ deactivate"
printf "\n"
printf "\nFormat files with the following command:\n"
printf "  yapf -r -i -e 'venv/*' ./**/*.py"
printf "\n"