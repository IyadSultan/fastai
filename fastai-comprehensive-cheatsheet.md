# Ultimate Comprehensive fastai Cheat Sheet

## Data Handling and Preparation

```python
from fastai.vision.all import *
from fastai.text.all import *
from fastai.tabular.all import *

# Download and extract dataset
path = untar_data(URLs.IMAGENETTE_160)

# Vision data
dls = ImageDataLoaders.from_folder(path, valid='val', item_tfms=Resize(128), batch_tfms=aug_transforms())

# Tabular data
dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
    cont_names = ['age', 'fnlwgt', 'education-num'],
    procs = [Categorify, FillMissing, Normalize])

# Text data
dls = TextDataLoaders.from_folder(path, valid='test', vocab=vocab)

# Show a batch of data
dls.show_batch(max_n=9, figsize=(7,8))
```

### Data Augmentation

```python
# Image augmentations
tfms = aug_transforms(mult=1.0, do_flip=True, flip_vert=False, max_rotate=10.0, 
                      min_zoom=1.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, 
                      p_affine=0.75, p_lighting=0.75)

# Text augmentations
tfms = [TextBlock.categorize()]
dls = TextDataLoaders.from_folder(path, valid='test', vocab=vocab, text_col='text', 
                                  label_col='label', bs=64, seq_len=72, 
                                  num_workers=8, shuffle=True, text_vocab=None, 
                                  is_lm=False, valid_pct=0.2, item_tfms=tfms)
```

## Model Creation and Training

### Vision Models

```python
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn = vision_learner(dls, resnet50, metrics=accuracy)

# Training
learn.fit_one_cycle(5, 3e-3)
learn.fine_tune(5, freeze_epochs=3)

# Find learning rate
learn.lr_find()
```

### Tabular Models

```python
learn = tabular_learner(dls, layers=[200,100], metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)
```

### Text Models

```python
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fit_one_cycle(10, 1e-2)
```

### Language Models

```python
# Prepare data for language modeling
seqs = L((tensor(nums[i:i+sl]), tensor(nums[i+1:i+sl+1]))
         for i in range(0,len(nums)-sl-1,sl))
cut = int(len(seqs) * 0.8)
dls = DataLoaders.from_dsets(group_chunks(seqs[:cut], bs),
                             group_chunks(seqs[cut:], bs),
                             bs=bs, drop_last=True, shuffle=False)

# Create and train language model
learn = language_model_learner(dls, AWD_LSTM, drop_mult=0.3, metrics=[accuracy, Perplexity()])
learn.fit_one_cycle(1, 5e-2)
```

## Advanced Training Techniques

```python
# Mixed precision training
learn = vision_learner(dls, resnet50, metrics=accuracy).to_fp16()

# Distributed training
learn = vision_learner(dls, resnet50, metrics=accuracy).to_parallel()

# Progressive resizing (for vision models)
sizes = [128, 256, 384]
for sz in sizes:
    dls = dls.new(item_tfms=Resize(sz))
    learn.dls = dls
    learn.fit_one_cycle(5)

# Gradual unfreezing (for transfer learning)
learn.freeze()
learn.fit_one_cycle(1)
learn.unfreeze()
learn.fit_one_cycle(5, lr_max=slice(1e-6,1e-4))
```

## Callbacks and Regularization

```python
# Callbacks
cbs = [
    EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=2),
    SaveModelCallback(monitor='valid_loss', fname='best_model'),
    ReduceLROnPlateau(monitor='valid_loss', min_delta=0.01, patience=2)
]
learn.fit_one_cycle(10, 3e-3, cbs=cbs)

# Regularization techniques
learn = Learner(dls, model, loss_func=loss_func, metrics=accuracy,
                cbs=[MixUp(), RNNRegularizer(alpha=2, beta=1)])
```

## Inference and Interpretation

```python
# Get predictions
preds, targs = learn.get_preds()

# Vision model interpretation
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(9)

# Tabular model interpretation
interp = TabularInterpreter(learn)
interp.feature_importance().plot(20)

# Text model interpretation
interp = TextClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_most_confused()
```

## Model Export and Deployment

```python
# Export the model
learn.export('model.pkl')

# Load and use the model for inference
learn_inf = load_learner('model.pkl')
pred, pred_idx, probs = learn_inf.predict(img)
```

## Useful Utilities

```python
# Display model architecture
learn.summary()

# Plot learning rate vs. loss
learn.recorder.plot_lr_find()

# Plot losses
learn.recorder.plot_loss()

# Show model predictions
learn.show_results()

# Get the number of parameters in the model
total_params = sum(p.numel() for p in learn.model.parameters())

# Use fastai's built-in documentation
doc(learn.fit_one_cycle)
```

## Key Concepts

1. **Transfer Learning**: Using pre-trained models and fine-tuning them for specific tasks.
2. **Data Augmentation**: Techniques to artificially increase the diversity of training data.
3. **One-Cycle Policy**: A learning rate schedule that improves training speed and performance.
4. **Progressive Resizing**: Gradually increasing image size during training for better performance.
5. **Mixed Precision Training**: Using both float32 and float16 for faster training with less memory.
6. **Interpretation**: Analyzing model behavior and predictions to gain insights.
7. **Callbacks**: Customizable functions to extend the training loop's behavior.

## Tips for Effective Use of fastai

1. Start with pre-trained models when possible.
2. Use `learn.lr_find()` to determine a good learning rate.
3. Apply data augmentation appropriate for your dataset.
4. Utilize callbacks for early stopping, saving best models, and adjusting learning rates.
5. Experiment with different architectures and hyperparameters.
6. Use interpretation tools to understand model performance and improve it.
7. Gradually unfreeze layers when fine-tuning for transfer learning.
8. Consider using mixed precision training for faster computation.
9. Regularly validate your model on a held-out test set.
10. Keep up with fastai updates and community resources for the latest best practices.

