# Creating conda environment
conda create --name udacity-risk-assessment python=3.7.6

# Activate environment
conda activate udacity-risk-assessment

#  through pip (conda won't install all from the requirements file)
pip install -r requirements.txt

# Deactivate environment
conda deactivate udacity-risk-assessment