# Pretraining COCO-LM using your own data

This tutorial will walk you through pretraining COCO-LM over your own data.

### 1) Preprocess the data

Data should be preprocessed following the [language modeling format](/examples/language_model), i.e. each document should be separated by an empty line (only useful with `--sample-break-mode complete_doc`). Lines will be concatenated as a 1D text stream during training.

We'll use the [WikiText-103 dataset](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/)
to demonstrate how to preprocess raw text data with the GPT-2 BPE. Of course
this dataset is quite small, so the resulting pretrained model will perform
poorly, but it gives the general idea.

First download the dataset:
```bash
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
```

Next encode it with the GPT-2 BPE:
```bash
mkdir -p dict
cd dict
wget https://github.com/microsoft/COCO-LM/releases/download/v0.1.0/dict.tar.gz
tar -xzf dict.tar.gz
cd ..
for SPLIT in train valid test; do \
    python -m multiprocessing_sp_encoder \
        --sentencepiece_model dict/sp.model \
        --vocab dict/dict.txt \
        --inputs wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs wikitext-103-raw/wiki.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done
```

Finally preprocess/binarize the data using the GPT-2 fairseq dictionary:
```bash
python ../../fairseq_cli/preprocess.py \
    --only-source \
    --srcdict dict/dict.txt \
    --trainpref wikitext-103-raw/wiki.train.bpe \
    --validpref wikitext-103-raw/wiki.valid.bpe \
    --testpref wikitext-103-raw/wiki.test.bpe \
    --destdir data-bin/wikitext-103 \
    --workers 60
cp dict/dict.txt data-bin/wikitext-103/dict.txt
```

### 2) Train RoBERTa base
```bash
DATA_DIR=$(pwd)/data-bin/wikitext-103/

fairseq-hydra-train -m --config-dir config/ \
--config-name base task.data=$DATA_DIR
```

**Note:** You can optionally resume training the released RoBERTa base model by
adding `checkpoint.restore_file=/path/to/cocolm.base/model.pt`.

**Note:** The above command assumes training on 8x32GB V100 GPUs. Each GPU uses
a batch size of 16 sequences (`dataset.batch_size`) and accumulates gradients to
further increase the batch size by 16x (`optimization.update_freq`), for a total batch size
of 2048 sequences. If you have fewer GPUs or GPUs with less memory you may need
to reduce `dataset.batch_size` and increase dataset.update_freq to compensate.
Alternatively if you have more GPUs you can decrease `dataset.update_freq` accordingly
to increase training speed.

**Note:** The learning rate and batch size are tightly connected and need to be
adjusted together. We generally recommend increasing the learning rate as you
increase the batch size according to the following table (although it's also
dataset dependent, so don't rely on the following values too closely):

batch size | peak learning rate
---|---
256 | 0.0001
2048 | 0.0005
8192 | 0.0007

### 3) Load your pretrained model
```python
from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt', 'path/to/data')
assert isinstance(roberta.model, torch.nn.Module)
```