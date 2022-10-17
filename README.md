
# Install environment

- Create new conda env

```conda create -n soundevent python=3.8```

```conda activate soundevent```

- Install pytorch version (for 30xx RTX GPU -> Cuda11)

```python -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html```

- Install nessesary library

```python -m pip install -r requirements.txt```

# Preprocessing Data

```bash preprocessing/normalize_audio.sh```

```python preprocessing/data_filter.py```

# Train 

- Using EfficientNet

```python train_eff.py --checkpoint_path <path_for_save_ck> --config config_efficient.json ```

- Using Wav2Vec

```python train_w2v.py --checkpoint_path <path_for_save_ck> --config config_w2v.json ```

# Inference 

```python inference.py -checkpoint_path <path_for_save_ck> --config config_efficient.json```

... Updating ...