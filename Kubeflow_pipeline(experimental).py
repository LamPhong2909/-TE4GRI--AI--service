import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import (
    component, Input, Output, Model, Dataset, Artifact, Metrics,
    ClassificationMetrics, ModelUploadOp, DatasetUploadOp, CacheSettings
)

# 1. Component: Data ingestion from S3
@component(
    base_image="python:3.9",
    packages_to_install=["boto3", "pandas"]
)
def ingest_data_op(bucket: str, prefix: str, output_dataset_path: Output[Dataset]):
    import boto3, pandas as pd
    s3 = boto3.client("s3")
    # list objects
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    records = []
    for obj in objs.get("Contents", []):
        data = s3.get_object(Bucket=bucket, Key=obj['Key'])['Body'].read().decode()
        # assume JSONL
        for line in data.splitlines():
            records.append(line)
    df = pd.DataFrame({'text': records})
    df.to_parquet(output_dataset_path.path)

# 2. Component: Text preprocessing
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "nltk", "spacy"]
)
def preprocess_op(input_dataset: Input[Dataset], output_dataset: Output[Dataset]):
    import pandas as pd
    import spacy
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_parquet(input_dataset.path)
    def clean_text(txt):
        doc = nlp(txt)
        return " ".join([token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha])
    df['clean'] = df['text'].apply(clean_text)
    df.to_parquet(output_dataset.path)

# 3. Component: Embed generation
@component(
    base_image="python:3.9",
    packages_to_install=["transformers", "torch"]
)
def embed_op(input_dataset: Input[Dataset], output_dataset: Output[Dataset]):
    import pandas as pd
    from transformers import AutoTokenizer, AutoModel
    import torch
    df = pd.read_parquet(input_dataset.path)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = []
    for txt in df['clean']:
        inputs = tokenizer(txt, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(emb.tolist())
    df_emb = pd.DataFrame({'embedding': embeddings})
    df_emb.to_parquet(output_dataset.path)

# 4. Component: Upload to Vector DB
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "pinecone-client"]
)
def upload_vector_db_op(
    input_embeddings: Input[Dataset], env: str = "production"
):
    import pinecone, pandas as pd
    df = pd.read_parquet(input_embeddings.path)
    pinecone.init(api_key="${{PINECONE_API_KEY}}", environment=env)
    index = pinecone.Index("deepseek-embeddings")
    for idx, row in df.iterrows():
        index.upsert(vectors=[(str(idx), row['embedding'])])

# 5. Component: Model training (RAG retriever or fine-tune)
@component(
    base_image="python:3.9",
    packages_to_install=["transformers", "datasets", "torch", "accelerate"]
)
def train_model_op(
    input_dataset: Input[Dataset], output_model: Output[Model]
):
    from datasets import load_dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
    import torch
    ds = load_dataset('parquet', data_files={'train': input_dataset.path})
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    def preprocess(batch):
        inp = tokenizer(batch['clean'], truncation=True, padding='max_length', max_length=128)
        with tokenizer.as_target_tokenizer():
            tgt = tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)
        inp['labels'] = tgt['input_ids']
        return inp
    ds = ds['train'].map(preprocess, batched=True)
    args = TrainingArguments(
        output_dir="./model_output",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_total_limit=1,
        push_to_hub=False
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()
    model.save_pretrained(output_model.path)

# 6. Component: Model deployment
@component(
    base_image="python:3.9",
    packages_to_install=["kserve"]
)
def deploy_model_op(input_model: Input[Model]):
    from kserve import KServeClient, model
    client = KServeClient()
    inference_service = model.InferenceService(
        api_version="serving.kserve.io/v1beta1",
        kind="InferenceService",
        metadata=model.ObjectMeta(name="deepseek-model"),
        spec=model.InferenceServiceSpec(
            predictor=model.PredictorSpec(
                sklearn=model.SKLearnSpec(
                    storage_uri=input_model.path
                )
            )
        )
    )
    client.create(inference_service)

# Define the pipeline
@dsl.pipeline(
    name="deepseek-ml-pipeline",
    description="End-to-end DeepSeek ML workflow with Kubeflow Pipelines",
)
def deepseek_pipeline(
    bucket: str = "my-data-bucket",
    prefix: str = "raw/jsonl/",
    pinecone_env: str = "us-west1-gcp"
):
    # Ingestion
    ingest = ingest_data_op(bucket=bucket, prefix=prefix)
    # Preprocessing
    preprocess = preprocess_op(input_dataset=ingest.outputs['output_dataset']).after(ingest)
    # Embedding
    embed = embed_op(input_dataset=preprocess.outputs['output_dataset']).after(preprocess)
    # Upload embeddings
    upload = upload_vector_db_op(input_embeddings=embed.outputs['output_dataset'], env=pinecone_env).after(embed)
    # Train model
    train = train_model_op(input_dataset=preprocess.outputs['output_dataset']).after(upload)
    # Deploy model
    deploy = deploy_model_op(input_model=train.outputs['output_model']).after(train)

if __name__ == "__main__":
    kfp.v2.compiler.Compiler().compile(
        pipeline_func=deepseek_pipeline,
        package_path="deepseek_pipeline.json",
    )
