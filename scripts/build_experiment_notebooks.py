"""Deterministic authoring source for the two visible notebook experiments."""
from pathlib import Path
import textwrap
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
def md(value): return nb.v4.new_markdown_cell(textwrap.dedent(value).strip())
def code(value): return nb.v4.new_code_cell(textwrap.dedent(value).strip())

SETUP = code('''
import os, sys, subprocess
from pathlib import Path
# Local: start Jupyter from the cloned repo. Colab: this cell clones if needed.
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    ROOT = Path('/content/isom-group-project')
    if not ROOT.exists():
        subprocess.run(['git','clone','https://github.com/zaynfuture/isom-group-project.git',str(ROOT)],check=True)
else:
    ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'model/data/transactions.py').exists()), None)
    if ROOT is None: raise RuntimeError('Open this notebook from the cloned SpendLens repository.')
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
INSTALL_DEPENDENCIES = IN_COLAB
if INSTALL_DEPENDENCIES:
    subprocess.run([sys.executable,'-m','pip','install','-r','requirements-notebook.txt'],check=True)
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/matplotlib'))
print('Repository:', ROOT)
''')
CONFIG = code('''
import gc, json, time, hashlib, platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display
from huggingface_hub import HfApi
from transformers import set_seed
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, mean_absolute_error
from model.data.transactions import demo_transactions
from model.data.scenarios import scenario_transactions
from model.experiments import merchant_split, forecast_split
from model.inference.forecasting import numeric_examples, predict_amounts, spending_bands, BANDS

SMOKE = os.getenv('SPENDLENS_SMOKE','1') == '1'  # Set False for a full experiment.
DATASET = os.getenv('SPENDLENS_DATASET','legacy')  # legacy or lifestyle; not customer data
if DATASET not in {'legacy','lifestyle'}: raise ValueError('Choose legacy or lifestyle.')
SEED = 5240
set_seed(SEED)
torch.set_num_threads(2)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
frame = demo_transactions() if DATASET == 'legacy' else scenario_transactions()
DATA_HASH = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
print({'smoke':SMOKE,'dataset':DATASET,'rows':len(frame),'device':DEVICE,'seed':SEED,'data_sha256':DATA_HASH})
display(frame.head())
display(frame.groupby('category').amount.agg(['count','sum']))
''')
LIMITATIONS = md('''
## Interpretation and data limits
The legacy fixture has five repeated merchant-description templates. Card-held-out splits can still share merchants/templates: perfect scores are not evidence of unseen-merchant generalization. Its test targets have been inspected in previous project iterations. Lifestyle data has fourteen categories and linked refunds, but is still a designed simulation, not population evidence. Refunds are excluded from model training/forecast totals. No demographic labels or lifestyle metadata enter model inputs.

Smoke mode runs real optimization on 12 cards with one epoch. It verifies software, not model quality. Full mode uses all 120 cards and more epochs; it still requires representative, independently held-out real-world validation before business use. Candidate and epoch decisions use validation only; inspect the test set only in the final evaluation cell. Repeated reruns on the same test set are not independent validation.
''')

MERCHANT = [
md('''# 01 · Merchant classification: selection → fine-tuning → final evaluation
Run cells top-to-bottom in Jupyter or Colab (GPU recommended for full mode). This notebook contains the training and evaluation code, not shell calls to training scripts. No uploads occur unless explicitly enabled near the end.

Candidates: [Microsoft MiniLM](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased) and [BERT Tiny](https://huggingface.co/prajjwal1/bert-tiny). Both start from pretrained encoders with newly initialized classification heads. We compare validation Macro-F1 after equal epoch budgets; ties favor fewer parameters. This is a bounded comparison, not a claim of globally optimal model choice.
'''), SETUP, CONFIG, LIMITATIONS,
md('## 1. Card-held-out split and input contract'),
code('''
parts = merchant_split(frame,smoke=SMOKE)
labels = sorted(frame.loc[frame.amount.gt(0),'category'].unique().tolist())
label_to_id = {label:index for index,label in enumerate(labels)}
assert set(parts['train'].category) == set(labels), 'Training split must cover every label.'
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    assert set(parts[a].card_id).isdisjoint(parts[b].card_id)
display(pd.DataFrame({key:{'rows':len(part),'cards':part.card_id.nunique()} for key,part in parts.items()}).T)
display(parts['train'].category.value_counts())
print('Input: merchant_description only. Excluded: MCC, card ID, demographics, lifestyle and category label.')
OUT = ROOT/'outputs/notebook-runs'/f'merchant-{DATASET}-{RUN_ID}'
OUT.mkdir(parents=True,exist_ok=False)
WINNER = OUT/'selected'
'''),
md('## 2. Load candidates, tokenize, fine-tune and select on validation'),
code('''
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
CANDIDATES = ['prajjwal1/bert-tiny','microsoft/MiniLM-L12-H384-uncased']
EPOCHS = 1 if SMOKE else 3
candidate_results, histories = [], {}
best_key = None
for candidate in CANDIDATES:
    set_seed(SEED)
    revision = HfApi().model_info(candidate).sha
    tokenizer = AutoTokenizer.from_pretrained(candidate,revision=revision)
    network = AutoModelForSequenceClassification.from_pretrained(candidate,revision=revision,
        num_labels=len(labels),id2label=dict(enumerate(labels)),label2id=label_to_id)
    datasets = {}
    for split in ['train','validation']:  # No test evaluation during selection.
        part=parts[split]
        dataset=Dataset.from_dict({'text':part.merchant_description.tolist(),'label':part.category.map(label_to_id).tolist()})
        datasets[split]=dataset.map(lambda batch: tokenizer(batch['text'],truncation=True,max_length=128),batched=True,remove_columns=['text'])
    def metrics(prediction):
        guessed=prediction.predictions.argmax(-1)
        return {'macro_f1':f1_score(prediction.label_ids,guessed,labels=list(range(len(labels))),average='macro',zero_division=0),
                'accuracy':accuracy_score(prediction.label_ids,guessed)}
    folder=OUT/candidate.replace('/','--')
    trainer=Trainer(model=network,processing_class=tokenizer,
        args=TrainingArguments(output_dir=str(folder),num_train_epochs=EPOCHS,learning_rate=3e-5,
            per_device_train_batch_size=16,per_device_eval_batch_size=32,eval_strategy='epoch',save_strategy='epoch',
            load_best_model_at_end=True,metric_for_best_model='macro_f1',save_total_limit=1,
            report_to='none',seed=SEED,use_cpu=DEVICE=='cpu',dataloader_pin_memory=DEVICE=='cuda'),
        train_dataset=datasets['train'],eval_dataset=datasets['validation'],
        data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=metrics)
    started=time.perf_counter()
    trainer.train()
    val=trainer.evaluate(datasets['validation'])
    parameter_count=sum(p.numel() for p in network.parameters())
    entry={'model':candidate,'revision':revision,'validation_macro_f1':float(val['eval_macro_f1']),
        'parameters':parameter_count,'training_seconds':time.perf_counter()-started}
    candidate_results.append(entry)
    histories[candidate]=trainer.state.log_history
    key=(entry['validation_macro_f1'],-parameter_count)
    if best_key is None or key>best_key:
        best_key=key
        selected=entry.copy()
        trainer.save_model(WINNER)
        tokenizer.save_pretrained(WINNER)
    del trainer,network,datasets
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
display(pd.DataFrame(candidate_results).sort_values('validation_macro_f1',ascending=False))
print('Selected using validation only:',selected['model'])
'''),
md('## 3. Final evaluation: selected model, majority baseline and class-level errors'),
code('''
tokenizer=AutoTokenizer.from_pretrained(WINNER)
selected_model=AutoModelForSequenceClassification.from_pretrained(WINNER).to(DEVICE).eval()
def classify(texts):
    predictions=[]
    for start in range(0,len(texts),32):
        tokens=tokenizer(texts[start:start+32],padding=True,truncation=True,max_length=128,return_tensors='pt').to(DEVICE)
        with torch.no_grad(): logits=selected_model(**tokens).logits
        predictions.extend(logits.argmax(-1).cpu().tolist())
    return predictions
evaluation={}
for split,part in parts.items():
    actual=part.category.map(label_to_id).tolist()
    predicted=classify(part.merchant_description.tolist())
    evaluation[split]={f'{split}_accuracy':float(accuracy_score(actual,predicted)),
        f'{split}_macro_f1':float(f1_score(actual,predicted,labels=list(range(len(labels))),average='macro',zero_division=0))}
    if split=='test':
        test_actual,test_predicted=actual,predicted
majority=parts['train'].category.mode().iloc[0]
evaluation['baseline']={'name':'training majority class','macro_f1':float(f1_score(parts['test'].category,
    [majority]*len(parts['test']),labels=labels,average='macro',zero_division=0))}
evaluation['confusion_matrix']=confusion_matrix(test_actual,test_predicted,labels=list(range(len(labels)))).tolist()
evaluation.update(labels=labels,base_model=selected['model'],base_revision=selected['revision'],candidate_results=candidate_results,
    data_source=DATASET,data_sha256=DATA_HASH,smoke=SMOKE,seed=SEED,
    split_counts={k:len(v) for k,v in parts.items()},selection_metric='validation macro_f1; ties favor fewer parameters',
    limitations='Synthetic repeated templates, no external validation. Historical test targets previously inspected. Smoke scores are software checks.')
display(pd.DataFrame({k:evaluation[k] for k in parts}).T)
display(pd.DataFrame(classification_report(test_actual,test_predicted,labels=list(range(len(labels))),target_names=labels,output_dict=True,zero_division=0)).T)
matrix=pd.DataFrame(evaluation['confusion_matrix'],index=labels,columns=labels)
display(matrix)
fig,ax=plt.subplots(figsize=(7,5)); ax.imshow(matrix); ax.set(title='Test confusion matrix',xlabel='Predicted label index',ylabel='True label index'); plt.show()
for candidate,history in histories.items():
    points=[x for x in history if 'eval_macro_f1' in x]
    plt.plot([x['epoch'] for x in points],[x['eval_macro_f1'] for x in points],marker='o',label=candidate)
plt.xlabel('Epoch'); plt.ylabel('Validation Macro-F1'); plt.legend(); plt.show()
'''),
md('## 4. Verify saved checkpoint and write reproducible evidence'),
code('''
reloaded=AutoModelForSequenceClassification.from_pretrained(WINNER).cpu().eval()
selected_model.cpu().eval()
reload_inputs=tokenizer(parts['test'].merchant_description.iloc[:2].tolist(),padding=True,truncation=True,max_length=128,return_tensors='pt')
with torch.no_grad():
    expected_logits=selected_model(**reload_inputs).logits
    actual_logits=reloaded(**reload_inputs).logits
assert torch.allclose(expected_logits,actual_logits,atol=1e-5)
(WINNER/'reload_verification.json').write_text(json.dumps({'passed':True,'samples':2,'atol':1e-5},indent=2))
(WINNER/'evaluation.json').write_text(json.dumps(evaluation,indent=2))
(WINNER/'training_history.json').write_text(json.dumps(histories[selected['model']],indent=2))
TASK='merchant'
LIBRARY='transformers'
LICENSE='mit'
print('Save/reload passed. Artifacts:',WINNER)
''')]

FORECAST = [
md('''# 02 · Spending forecast: candidate selection → fine-tuning → final evaluation
Run all cells in local Jupyter or Colab. Compare pretrained [Chronos-Bolt Tiny](https://huggingface.co/amazon/chronos-bolt-tiny) and [Mini](https://huggingface.co/amazon/chronos-bolt-mini) on validation quantile loss, then fine-tune the selected architecture. Selection is based on pretrained validation behavior, not an exhaustive comparison of fine-tuned candidates. Test data never chooses the candidate or best epoch.
'''), SETUP, CONFIG, LIMITATIONS,
md('## 1. Chronological examples and fixed USD contract'),
code('''
parts=forecast_split(frame,smoke=SMOKE)
assert parts['train'].target_month.max()<parts['validation'].target_month.min()<parts['test'].target_month.min()
display(pd.DataFrame({k:{'rows':len(v),'months':', '.join(sorted(v.target_month.unique()))} for k,v in parts.items()}).T)
display(parts['train'].head())
print('Two complete prior-month positive purchase totals → next-month purchases. Last observed month excluded. USD bands: <200, <400, otherwise HIGH.')
OUT=ROOT/'outputs/notebook-runs'/f'forecast-{DATASET}-{RUN_ID}'
OUT.mkdir(parents=True,exist_ok=False)
WINNER=OUT/'selected'
'''),
md('## 2. Explicit quantile evaluation and validation-only model selection'),
code('''
from chronos import BaseChronosPipeline
def forecast_metrics(pipe,part):
    predicted=predict_amounts(pipe,part.context.tolist())
    actual=part.target.to_numpy()
    quantiles=predicted[['p10','median_amount','p90']].to_numpy()
    errors=actual[:,None]-quantiles
    levels=np.array([.1,.5,.9])
    metrics={'mae':float(mean_absolute_error(actual,predicted.median_amount)),
        'macro_f1':float(f1_score(part.label,predicted.prediction,labels=BANDS,average='macro',zero_division=0)),
        'accuracy':float(accuracy_score(part.label,predicted.prediction)),
        'mean_pinball_loss':float(np.maximum(levels*errors,(levels-1)*errors).mean()),
        'interval_80_coverage':float(((actual>=predicted.p10)&(actual<=predicted.p90)).mean())}
    return metrics,predicted
CANDIDATES=['amazon/chronos-bolt-tiny','amazon/chronos-bolt-mini']
candidate_results=[]
for candidate in CANDIDATES:
    revision=HfApi().model_info(candidate).sha
    pipe=BaseChronosPipeline.from_pretrained(candidate,revision=revision,device_map=DEVICE,torch_dtype=torch.float32)
    pipe.model.eval()
    metrics,_=forecast_metrics(pipe,parts['validation'])
    candidate_results.append({'model':candidate,'revision':revision,'validation_pinball_loss':metrics['mean_pinball_loss'],
        'validation_mae':metrics['mae'],'parameters':sum(p.numel() for p in pipe.model.parameters())})
    del pipe; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
selected=min(candidate_results,key=lambda x:(x['validation_pinball_loss'],x['parameters']))
display(pd.DataFrame(candidate_results)); print('Selected:',selected)
'''),
md('## 3. Fine-tune numeric histories; choose best epoch on validation'),
code('''
set_seed(SEED)
pipe=BaseChronosPipeline.from_pretrained(selected['model'],revision=selected['revision'],device_map=DEVICE,torch_dtype=torch.float32)
network=pipe.model
optimizer=torch.optim.AdamW(network.parameters(),lr=1e-4,weight_decay=.01)
context=torch.tensor(parts['train'].context.tolist(),dtype=torch.float32,device=DEVICE)
target=torch.tensor(parts['train'].target.to_numpy(),dtype=torch.float32,device=DEVICE).unsqueeze(-1)
EPOCHS=1 if SMOKE else 5
best_loss=float('inf'); history=[]; best_epoch=None
for epoch in range(1,EPOCHS+1):
    network.train(); losses=[]
    for cpu_indices in torch.randperm(len(context)).split(16):
        indices=cpu_indices.to(DEVICE)
        optimizer.zero_grad()
        loss=network(context=context[indices],target=target[indices]).loss
        if not torch.isfinite(loss): raise ValueError('Nonfinite training loss; abort.')
        loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(),1.); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    network.eval()
    metrics,_=forecast_metrics(pipe,parts['validation'])
    entry={'epoch':epoch,'train_loss':float(np.mean(losses)),'eval_loss':metrics['mean_pinball_loss'],
        'eval_macro_f1':metrics['macro_f1'],'validation_mae':metrics['mae']}
    history.append(entry); print(entry)
    if metrics['mean_pinball_loss']<best_loss:
        best_loss=metrics['mean_pinball_loss']; best_epoch=epoch
        network.save_pretrained(WINNER,safe_serialization=True)
display(pd.DataFrame(history))
pd.DataFrame(history).set_index('epoch')[['train_loss','eval_loss']].plot(title='Training and validation quantile losses'); plt.show()
'''),
md('## 4. Final held-out evaluation and simple baseline comparison'),
code('''
reloaded=BaseChronosPipeline.from_pretrained(str(WINNER),device_map=DEVICE,torch_dtype=torch.float32)
evaluation={}
for split,part in parts.items():
    metrics,predictions=forecast_metrics(reloaded,part)
    evaluation[split]={f'{split}_{key}':value for key,value in metrics.items()}
    if split=='test': test_predictions=predictions
test=parts['test']
evaluation['baseline']={'name':'previous two-month average',
    'mae':float(mean_absolute_error(test.target,test.baseline_amount)),
    'macro_f1':float(f1_score(test.label,spending_bands(test.baseline_amount),labels=BANDS,average='macro',zero_division=0))}
evaluation['confusion_matrix']=confusion_matrix(test.label,test_predictions.prediction,labels=BANDS).tolist()
evaluation.update(labels=BANDS,base_model=selected['model'],base_revision=selected['revision'],architecture='chronos-bolt',
    candidate_results=candidate_results,best_epoch=best_epoch,data_source=DATASET,data_sha256=DATA_HASH,smoke=SMOKE,seed=SEED,
    split_counts={k:len(v) for k,v in parts.items()},split_months={k:sorted(v.target_month.unique()) for k,v in parts.items()},
    selection_metric='pretrained validation pinball loss for architecture; validation pinball loss for fine-tuned epoch',
    limitations='Only two historical months per input. Synthetic data, same cards across time splits, no independent external validation. Smoke scores are software checks.')
display(pd.DataFrame({k:evaluation[k] for k in parts}).T)
display(pd.DataFrame([evaluation['baseline']]))
display(pd.DataFrame(evaluation['confusion_matrix'],index=BANDS,columns=BANDS))
display(pd.concat([test[['card_id','target_month','target']].reset_index(drop=True),test_predictions],axis=1))
plt.scatter(test.target,test_predictions.median_amount); plt.xlabel('Actual USD'); plt.ylabel('Forecast median USD'); plt.title('Held-out forecast errors'); plt.show()
print('Did the selected model beat the baseline MAE?', evaluation['test']['test_mae']<evaluation['baseline']['mae'])
'''),
md('## 5. Verify saved checkpoint and record contract'),
code('''
again=BaseChronosPipeline.from_pretrained(str(WINNER),device_map='cpu',torch_dtype=torch.float32)
samples=test.context.iloc[:2].tolist()
first=predict_amounts(reloaded,samples)
second=predict_amounts(again,samples)
assert np.allclose(first[['p10','median_amount','p90']],second[['p10','median_amount','p90']],atol=1e-3)
(WINNER/'reload_verification.json').write_text(json.dumps({'passed':True,'samples':2,'atol':1e-3},indent=2))
(WINNER/'evaluation.json').write_text(json.dumps(evaluation,indent=2))
(WINNER/'training_history.json').write_text(json.dumps(history,indent=2))
(WINNER/'forecast_contract.json').write_text(json.dumps({'context_months':2,'horizon_months':1,'currency':'USD','bands':[200,400],'amount':'positive purchases'},indent=2))
TASK='forecast'
LIBRARY='chronos'
LICENSE='apache-2.0'
print('Save/reload passed. Artifacts:',WINNER)
''')]

TAIL = [md('''## Record environment and model card
Artifacts belong to this run. Historical deployed metrics are not overwritten. Review synthetic-data limitations, baseline performance and model license before publication.
'''),code('''
import importlib.metadata
environment={name:importlib.metadata.version(name) for name in ['torch','transformers','datasets','chronos-forecasting','numpy','pandas']}
environment['python']=platform.python_version()
(OUT/'environment.json').write_text(json.dumps(environment,indent=2))
(WINNER/'README.md').write_text(
    f"---\\nlibrary_name: {LIBRARY}\\nlicense: {LICENSE}\\nbase_model: {selected['model']}\\n---\\n"
    f"# SpendLens {TASK} experiment\\n\\nDataset: {DATASET}. Smoke run: {SMOKE}.\\n"
    f"Base revision: {selected['revision']}. Labels: {evaluation['labels']}.\\n\\n"
    f"Selection: {evaluation['selection_metric']}.\\n\\n{evaluation['limitations']}\\n\\n"
    "See evaluation.json for actual train/validation/test and baseline results. Not automatically approved for serving. No demographic inference.\\n")
print(environment)
'''),md('''## Optional publication — disabled by default
Only publish after a full run and review. This cell never uploads transactions, checkpoints or credentials. Use a new experiment repository under **zhengzhihust**; do not overwrite the two currently deployed model repositories. Interactive login is used, not a token in the notebook. Publication does not change the app. A separate review must approve label-schema compatibility, evaluation and a pinned serving revision.
'''),code('''
PUBLISH = False
APPROVE_REVIEWED_FULL_RUN = False
HF_REPO = f'zhengzhihust/spendlens-{TASK}-{DATASET}-experiment'
if PUBLISH:
    if SMOKE or not APPROVE_REVIEWED_FULL_RUN:
        raise ValueError('Run full mode and explicitly approve the reviewed evaluation first.')
    if not HF_REPO.startswith('zhengzhihust/') or HF_REPO.count('/')!=1:
        raise ValueError('Repository must be under zhengzhihust.')
    protected={'zhengzhihust/spendlens-merchant-minilm','zhengzhihust/spendlens-spending-chronos-bolt-tiny'}
    if HF_REPO in protected: raise ValueError('Use a separate experiment repository; production promotion is a separate review.')
    from huggingface_hub import notebook_login
    notebook_login()
    api=HfApi()
    if api.whoami()['name'].lower()!='zhengzhihust': raise ValueError('Authenticate as zhengzhihust.')
    assert json.loads((WINNER/'reload_verification.json').read_text())['passed']
    api.create_repo(HF_REPO,repo_type='model',exist_ok=True)
    receipt=api.upload_folder(repo_id=HF_REPO,folder_path=str(WINNER),allow_patterns=[
        'model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt',
        'evaluation.json','training_history.json','forecast_contract.json','reload_verification.json','README.md'])
    published={'repo_id':HF_REPO,'revision':receipt.oid}
    if TASK=='merchant':
        remote=AutoModelForSequenceClassification.from_pretrained(HF_REPO,revision=receipt.oid).cpu().eval()
        remote_tokenizer=AutoTokenizer.from_pretrained(HF_REPO,revision=receipt.oid)
        remote_inputs=remote_tokenizer(parts['test'].merchant_description.iloc[:2].tolist(),padding=True,truncation=True,max_length=128,return_tensors='pt')
        with torch.no_grad(): remote_logits=remote(**remote_inputs).logits
        assert torch.allclose(expected_logits,remote_logits,atol=1e-5)
    else:
        remote=BaseChronosPipeline.from_pretrained(HF_REPO,revision=receipt.oid,device_map='cpu',torch_dtype=torch.float32)
        actual=predict_amounts(remote,samples)
        assert np.allclose(second[['p10','median_amount','p90']],actual[['p10','median_amount','p90']],atol=1e-3)
    published['hub_reload_verified']=True
    (OUT/'publication.json').write_text(json.dumps(published,indent=2))
    print(published)
else:
    print('Upload skipped. Serving manifest unchanged.')
'''),md('''## Next decision
Review actual results and failure cases before any business claim. For a production candidate, collect authorized representative data, define a fresh test protocol and review privacy, compute and hosting constraints. Fourteen-label classifiers require an explicit serving-schema update. No test score here certifies fairness or real-world reliability.
''')]

def build():
    folder=ROOT/'notebooks'; folder.mkdir(exist_ok=True)
    for name,cells in [('01_merchant_experiment',MERCHANT),('02_forecast_experiment',FORECAST)]:
        notebook=nb.v4.new_notebook(cells=cells+TAIL,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
        for index,cell in enumerate(notebook.cells): cell['id']=f'{name}-{index:02d}'
        nb.write(notebook,folder/f'{name}.ipynb')

if __name__=='__main__': build()
