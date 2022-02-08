# Pretraining COCO-LM using your own data

This tutorial will walk you through pretraining COCO-LM over your own data.

### 1) Preprocess the data

Data should be preprocessed following the [language modeling format](/examples/language_model), i.e. each document should be separated by an empty line (only useful with `--sample-break-mode complete_doc`). Lines will be concatenated as a 1D text stream during training.

We'll use the [WikiText-103 dataset](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) to demonstrate how to preprocess raw text data with the GPT-2 BPE. Of course this dataset is quite small, so the resulting pretrained model will perform poorly, but it gives the general idea. You can create a new SentencePieces BPE vocabulary. 

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

### 2) Train 
Slurm distributed job script: ./train-distributed.sh ./train-runner.sh

**Note:** You can optionally resume training the released COCO-LM base model by
adding `checkpoint.restore_file=/path/to/cocolm.base/model.pt`. You can download model here: https://huggingface.co/kamalkraj/COCO-LM/tree/main

More discussion see: https://github.com/microsoft/COCO-LM/issues/2


