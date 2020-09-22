# Lil script for setting up the virtual environment, so we upload
# all the packages to github. MAKE SURE TO RUN THIS FROM THE
# ratnet-env DIRECTORY.

python3 -m venv .
source bin/activate
pip install --upgrade pip
pip install -r requirements.txt

