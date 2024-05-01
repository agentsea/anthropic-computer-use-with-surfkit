
FROM thehale/python-poetry:1.8.2-py3.10-slim

COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install -y openssh-client ntp
RUN poetry install

EXPOSE 9090

# Run the application
CMD ["uvicorn", "surfpizza.server:app", "--host=0.0.0.0", "--port=9090", "--log-level", "debug"]
