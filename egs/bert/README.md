# BERT-NER Version 2


Use Google's BERT for named entity recognition （CoNLL-2003 as the dataset）. 


The original version （see old_version for more detail） contains some hard codes and lacks corresponding annotations,which is inconvenient to understand. So in this updated version,there are some new ideas and tricks （On data Preprocessing and layer design） that can help you quickly implement the fine-tuning model (you just need to try to modify crf_layer or softmax_layer).

### Folder Description:
```
BERT-NER
|____ bert                          # need git from [here](https://github.com/google-research/bert)
|____ cased_L-12_H-768_A-12	    # need download from [here](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)
|____ data		            # train data
|____ middle_data	            # middle data (label id map)
|____ output			    # output (final model, predict results)
|____ BERT_NER.py		    # mian code
|____ conlleval.pl		    # eval code
|____ run_ner.sh    		    # run model and eval result

```

**Notes**:

Replace vocab.txt in `cased_L-12_H-768_A-12` with vocab.txt (only vietnamese) (can download from [here](https://raw.githubusercontent.com/autobotasia/vibert/master/vi-vocab.txt))

### Usage:
```
bash run_ner.sh
```

### What's in run_ner.sh:
```
python BERT_NER.py\
    --task_name="NER"  \
    --do_lower_case=False \
    --crf=False \
    --do_train=True   \
    --do_eval=True   \
    --do_predict=True \
    --data_dir=data   \
    --vocab_file=cased_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=cased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=cased_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=128   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=3.0   \
    --output_dir=./output/result_dir

perl conlleval.pl -d '\t' < ./output/result_dir/label_test.txt
```

**Notice:** cased model was recommened, according to [this](https://arxiv.org/abs/1810.04805) paper. CoNLL-2003 dataset and perl Script comes from [here](https://www.clips.uantwerpen.be/conll2003/ner/)


### reference:

[1] https://arxiv.org/abs/1810.04805

[2] https://github.com/google-research/bert



