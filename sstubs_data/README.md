
# Download dataset

```
# in this directory
wget 'https://zenodo.org/records/7743263/files/msr_sstubs_llms.tar.gz?download=1' -O msr_sstubs_llms.tar.gz
tar -xzvf msr_sstubs_llms.tar.gz
```


Note: `msr_sstubs_llms/datasets/sstub_input_no_comments.csv`就是论文中使用的数据集.


# Process the dataset

```
python preprocess.py
```

This script will automatically transform the dataset and split the dataset into trianing/testing according to different ratio.