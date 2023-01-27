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
    --n_runs 50,100,500,1000,5000,... \
    --n_samples ...,... 
```