## install
### dalex
`pip install -e .`

### evaluation - just for our development
`pip install -e evaluation`

## run scripts
```
python evaluation/src/evaluation/scripts.py run-many-experiments \
    --models xgboost,svm \
    --methods exact,kernel,unbiased \
    --datasets housing,... \
    --n_runs 5000 \
    --n_samples 30,100,200,300,400,500 
```
