FROM public.ecr.aws/lambda/python:3.11

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, models, and feature configuration
COPY app ./app
COPY models ./models
COPY feature_columns.pkl .

# Set Mangum / FastAPI handler for AWS Lambda
CMD ["app.main.handler"]
