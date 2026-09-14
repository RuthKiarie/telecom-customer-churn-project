FROM public.ecr.aws/lambda/python:3.11

# Copy requirements and install dependencies using binary-only wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

# Copy application code
COPY ./app /var/task/app

# Set Mangum / FastAPI handler for AWS Lambda
CMD ["app.main.handler"]

