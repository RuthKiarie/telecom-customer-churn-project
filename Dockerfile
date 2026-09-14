FROM public.ecr.aws/lambda/python:3.11

# Copy requirements and install dependencies using binary-only wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

# Copy application code
COPY ./app /var/task/app

# Copy model and feature columns explicitly into the working directory
COPY churn_model.pkl /var/task/app
COPY feature_columns.pkl /var/task/app

# Set Mangum / FastAPI handler for AWS Lambda
CMD ["app.main.handler"]