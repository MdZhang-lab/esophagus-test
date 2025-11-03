# Heart Disease Classification with FT-Transformer

This project trains an [FT-Transformer](https://arxiv.org/abs/2106.11959) model on tabular health records to predict whether a patient suffers from heart disease.

## Dataset format

Training expects three CSV tables containing the same set of patients:

- `general.csv`
- `protein.csv`
- `blood.csv`

Each file must satisfy the following constraints:

1. **Rows** – one patient per row.
2. **Columns** –
   - Column 1: patient identifier (`label`).
   - Column 2: binary target (`0` = healthy, `1` = heart disease).
   - Columns 3+ : patient features (numeric or categorical).
3. All tables must list patients in the same order with matching labels.

The loader automatically detects numeric and categorical features, scales numeric values (using statistics computed on the training split), and encodes categorical features.

## Configuration

Model and training parameters are stored in `config.yml`:

```yaml
seed: 42

data:
  files:
    - data/general.csv
    - data/protein.csv
    - data/blood.csv
  batch_size: 64
  num_workers: 0
  val_ratio: 0.2

model:
  n_blocks: 3

optimizer:
  lr: 0.001
  weight_decay: 1.0e-05

trainer:
  num_epochs: 50
  output_dir: outputs
  gradient_accumulation_steps: 1
  mixed_precision: "no"
```

Adjust the file paths and hyper-parameters as needed.

## Training

Install the project dependencies (see `requirements.txt`), then launch training. For single-GPU or CPU runs you can execute the script directly:

```bash
python train.py --config config.yml
```

To leverage multiple GPUs (e.g., an 8×RTX 4090 server), use [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) to launch the script after running `accelerate config` once:

```bash
accelerate launch train.py --config config.yml
```

The script logs epoch-level metrics and saves the best checkpoint (based on validation accuracy) to `outputs/best_model.pt` by default.
