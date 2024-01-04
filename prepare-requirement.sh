conda list | awk '{print $1, $2}' > requirements.txt
pip freeze | grep -v "file" >> requirements.txt
