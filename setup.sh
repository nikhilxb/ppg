# Create virtual environment
virtualenv -p python3.6 venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Display message for user
printf "\nThe virtual environment is now ready for use:\n"
printf "  $ source venv/bin/activate\n"
printf "  $ // use for a bit\n"
printf "  $ deactivate\n"
