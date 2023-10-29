This is the code repository for the paper "Diagonal Sorting for Seq2seq Models: Enhancing Inference Performance on Matrices".

You can generate the dataset using the code below dataset or download it from [Google Cloud Drive](https://drive.google.com/file/d/1r9OVIqI5fz7m2cI5fT9DoVTW1Oe0VZl6/view?usp=drive_link).

After configuring the environment according to the provided requirements.txt and freeze.yml, you can run the experimental code.

For example, you can run the following command to try to reproduce the MIS experiment in the paper.
```
python main.py  --exp_name "mis" \
                --emb_dim 256 \
                --n_enc_layers 6 \
                --n_dec_layers 6 \
                --n_heads 8 \
                --batch_size 64 \
                --reload_data "./dataset/MIS/200000_20_20_e2632"\
                --reload_testset "{'20 nodes': './dataset/MIS/200000_20_20_e2632','19 nodes': './dataset/MIS/200000_19_19_hdzpw','18 nodes': './dataset/MIS/200000_18_18_5r9jy','17 nodes': './dataset/MIS/200000_17_17_fo19y','16 nodes': './dataset/MIS/200000_16_16_4v364'}"\
                --sort_method "SMD,SMR"
```

options of "sort_method":
 - SMR: Row-based sorting
 - SMC: Column-based sorting
 - SMD: Diagonal-based sorting (ours)
 - c-SMD: counter-Diagonal-based sorting (ours)
