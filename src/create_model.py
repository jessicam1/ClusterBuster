#!/usr/local/bin/python
import sys
import json
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import keras_tuner as kt
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from contextlib import redirect_stdout


print(tf.__version__, file=sys.stderr) 

### use keras tuner to find optimized snp-specific neural network model
### this one uses features R, Theta, and one-hot encoded snpIDs

### load training data
savedir = "/data/CARD_AA/projects/2023_05_JM_gt_clusters/capstone/data"
df_train = pd.read_csv(f"{savedir}/train.csv", sep=",", header=0)
df_val = pd.read_csv(f"{savedir}/val.csv", sep=",", header=0)

### categorical and numerical features from df
features_numerical = ["R", "Theta"]
# features_onehot = [col for col in df_train.columns if col.startswith('snpID_') and col != 'snpID_ref']
features_snpid = ["snpID_cat"]

X_numerical_train = df_train[features_numerical].astype("float").to_numpy()
# X_snpid_train = df_train[features_onehot].astype("float").to_numpy()
X_snpid_train = df_train[features_snpid].astype("int").to_numpy()
y_train = df_train[["GT_AA", "GT_AB", "GT_BB"]].astype("float").to_numpy()

X_numerical_val = df_val[features_numerical].astype("float").to_numpy()
# X_snpid_val = df_val[features_onehot].astype("float").to_numpy()
X_snpid_val = df_val[features_snpid].astype("int").to_numpy()
y_val = df_val[["GT_AA", "GT_AB", "GT_BB"]].astype("float").to_numpy()


### NN Tuner Part

def model_builder(hp):
    
    num_snpids = 1051 # size of vocabulary
    num_numerical_features = 2  # Number of numerical features (R and Theta)
    
    # Define input layers 
    input_numerical = tf.keras.layers.Input(shape=(num_numerical_features,), name="input_numerical")
    input_embedded = tf.keras.layers.Input(shape=(1,), name="input_embedded")
    
    # Define embedding layer
    # embedding_size based on Jeremy Howards equation embedding_size=min(50, num_categories/2) = 532
    embedding_size=50
    embedded_features = tf.keras.layers.Embedding(input_dim=num_snpids, output_dim=embedding_size)(input_embedded)
    # Flatten the embedded features
    flattened_features = tf.keras.layers.Flatten()(embedded_features)

    # Concatenate embedded features with numerical features
    concatenated_features = tf.keras.layers.Concatenate()([flattened_features, input_numerical])

    # Define the rest of the neural network architecture
        
    # Tune the number of units in the first dense layer
    mins = 32
    maxs = 160
    steps = 32
    hp_units1 = hp.Int('units1', min_value=mins, max_value=maxs, step=steps)    
    dense1 = Dense(units=hp_units1, activation="relu")(concatenated_features)
    
    # Tune the number of units in the first dense layer
    hp_units2 = hp.Int('units2', min_value=mins, max_value=maxs, step=steps)    
    dense2 = Dense(units=hp_units2, activation="relu")(dense1)
    
    output_layer = Dense(units=3, activation="softmax")(dense2)

    # Define the model
    model = tf.keras.models.Model(inputs=[input_embedded, input_numerical], outputs=output_layer)

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.CategoricalFocalCrossentropy(),# my_categorical_focal_loss(),
                metrics=['categorical_accuracy'])

    return model

# instantiate tuner - only run once
# tuner = kt.Hyperband(model_builder,
#                      objective='val_categorical_accuracy',
#                      max_epochs=10,
#                      factor=3,
#                      directory='/data/CARD_AA/projects/2023_05_JM_gt_clusters/models/snp_specific_even_ancestry/nn_tuner',
#                      project_name='gt_snpid_r_theta_tuner')

# ### load in tuner ###
### loading in due to previous job timing out after 24 hours
tuner = kt.Hyperband(model_builder,
    objective='val_categorical_accuracy',
    overwrite=False,
    directory='/data/CARD_AA/projects/2023_05_JM_gt_clusters/capstone/model/',
    project_name="gt_snpid_r_theta_tuner",
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search([X_snpid_train, X_numerical_train], y_train, epochs=50, validation_data=([X_snpid_val, X_numerical_val], y_val), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the optimal hyperparameters
model = tuner.hypermodel.build(best_hps)

print(model.summary, file=sys.stderr)

# train the best model for 50 epochs
history = model.fit([X_snpid_train, X_numerical_train], y_train, epochs=50, validation_data=([X_snpid_val, X_numerical_val], y_val))

val_acc_per_epoch = history.history['val_categorical_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

# Re-instantiate the hypermodel and train it with the optimal number of epochs (28) from above.
hypermodel = tuner.hypermodel.build(best_hps)
 
# Retrain the model
hypermodel.fit([X_snpid_train, X_numerical_train], y_train, epochs=best_epoch, validation_data=([X_snpid_val, X_numerical_val], y_val))

# Save model
savedir = "/data/CARD_AA/projects/2023_05_JM_gt_clusters/capstone/model"
hypermodel.save(f"{savedir}/gt_model.keras")

# Create various output files about the model
with open(f'{savedir}/model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        hypermodel.summary()

config = hypermodel.get_config()                
with open(f'{savedir}/model_config.json', 'w') as f:
    json.dump(config, f, indent=4)                

optimizer_config = hypermodel.optimizer.get_config()
with open(f'{savedir}/optimizer_config.json', 'w') as f:
    json.dump(optimizer_config, f, indent=4)