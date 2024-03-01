# %%
conda install snowflake-ml-python

# %%
from snowflake.ml.registry import model_registry

result = model_registry.create_model_registry(
    session=session,
    database_name="<your_database_name>",
    schema_name='MODEL_REGISTRY'
)
# document code
print(result)

# %%
registry = model_registry.ModelRegistry(
    session=session,
    database_name="<your_database_name>",
    schema_name='MODEL_REGISTRY'
)

# %%
#Getting a model from the registry always returns a specific version of the model, so it is necessary to specify the version you want when you retrieve the model.

model = model_registry.ModelReference(
            registry=registry,
            model_name="my_model",
            model_version="101")

# %%
#3 install pytorch libraries

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# %%
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Loading the model and the tokenizer
model = AutoModelForSequenceClassification.from_pretrained('Facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('Facebook/bart-large-mnli')

# Save the model locally
ARTIFACTS_DIR = "/tmp/facebook-bart-large-mnli/"
os.makedirs(os.path.join(ARTIFACTS_DIR, "model"), exist_ok=True)
os.makedirs(os.path.join(ARTIFACTS_DIR, "tokenizer"), exist_ok=True)
model.save_pretrained(os.path.join(ARTIFACTS_DIR, "model"))
tokenizer.save_pretrained(os.path.join(ARTIFACTS_DIR, "tokenizer"))


#3 document code


class FacebookBartLargeMNLICustom(custom_model.CustomModel):
   def __init__(self, context: custom_model.ModelContext) -> None:
       super().__init__(context)

       self.model = AutoModelForSequenceClassification.from_pretrained(self.context.path("model"))
       self.tokenizer = AutoTokenizer.from_pretrained(self.context.path("tokenizer"))
       self.candidate_labels = ['customer support', 'product experience', 'account issues']



   @custom_model.inference_api
   def predict(self, X: pd.DataFrame) -> pd.DataFrame:
       def _generate(input_text: str) -> str:
           classifier = pipeline(
               "zero-shot-classification",
               model=self.model,
               tokenizer=self.tokenizer
           )

           result = classifier(input_text, self.candidate_labels)
           if 'scores' in result and 'labels' in result:
               category_idx = pd.Series(result['scores']).idxmax()
               return result['labels'][category_idx]

           return None

       res_df = pd.DataFrame({"output": pd.Series.apply(X["input"], _generate)})
       return res_df

# %%
model = FacebookBartLargeMNLICustom(custom_model.ModelContext(models={}, artifacts={
   "model":os.path.join(ARTIFACTS_DIR, "model"),
   "tokenizer":os.path.join(ARTIFACTS_DIR, "tokenizer")
}))

model.predict(pd.DataFrame({"input":["The interface gets frozen very often"]}))

# %%
from snowflake.ml.model import model_signature

model_id = registry.log_model(
   model_name='Facebook/bart-large-mnli',
   model_version='100',
   model=cross_model,
   conda_dependencies=[
       "transformers==4.30.0"
   ],
   signatures={
       "predict": model_signature.ModelSignature(
           inputs=[model_signature.FeatureSpec(name="input", dtype=model_signature.DataType.STRING)],
           outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.STRING)],
       )
   }
)

# %%
reference = model_registry.ModelReference(registry=registry, model_name='Facebook/bart-large-mnli', model_version='100')
model = reference.load_model()
model.predict(pd.DataFrame({"input":["The interface gets frozen very often"]}))

# %%
model.set_metric("dataset_validation", {"accuracy": 0.9})
# Print all metrics related to the model
print(model.get_metrics())



