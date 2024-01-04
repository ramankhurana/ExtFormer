# python image used for Extformer
FROM python:3.9.18

# set working dir
WORKDIR /usr/src/app

# install git
RUN apt-get update && apt-get install -y git

# Clone repository from GitHub
RUN git clone https://github.com/ramankhurana/ExtFormer.git

# Change the working directory
WORKDIR /usr/src/app/ExtFormer

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make the shell script executable
RUN echo "changing permission"
RUN chmod +x scripts/long_term_forecast/M5/*.sh

RUN echo "running the code"
# Command to execute the shell script
CMD ./scripts/long_term_forecast/M5/$SCRIPT_NAME



#RUN chmod +x scripts/long_term_forecast/M5/Autoformer_M5.sh

#CMD ["./scripts/long_term_forecast/M5/Autoformer_M5.sh"]
