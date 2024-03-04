FROM python:3.10


#Set Working Directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy packages required from local requirements file to Docker image requirements file
RUN git clone https://github.com/roshanrai1304/Real-Estate-Recommendation.git

RUN pip3 install -r requirements.txt

#Expose Port 8501 for app to be run on
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=localhost"]