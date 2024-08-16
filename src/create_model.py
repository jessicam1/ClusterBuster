#!/usr/local/bin/python
import sys
import json
import argparse
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import keras_tuner as kt
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from contextlib import redirect_stdout

def parse_arguments():
    parser= argparse.ArgumentParser(description=(
        "From a CSV or list of parquets containing SNP metrics, ",
        "get genotype predictions from Cluster Buster keras model")
    )
    parser.add_argument("-s", "--save_directory", help="path to trained model and Keras NN Tuner Project to")
    parser.add_argument("-m", "--model_name", help="name of pickled trained ML model, must end in .keras")
    parser.add_argument("-p", "--project_name", help="name of Keras NN Tuner Project")
    parser.add_argument("-t", "--training_data", help="path to training data (CSV or parquet)")
    parser.add_argument("-v", "--validation_data", help="path to validation data (CSV or parquet)")
    parser.add_argument("-i", "--save_information", action="store_true", help=(
        "If flagged, save model summary (txt), model configuration (json), and optimizer configuration (json)."
        "Saves to same directory as model.")
    )  
    return parser.parse_args()


def build_model(training_data, validation_data, save_directory, project_name):
    """
    Inputs:
    training_data - dataframe with snp metrics used to train model
    validation_data - dataframe with snp metrics used to validate model performance
    save_directory - path to directory to save Keras NN tuner project 
    project_name - name of folder Keras NN tuner creates
    Function:
    Given training and validation data, extract features for neural network, use Keras 
    NN tuner to gridsearch for optimal neural network structure and hyperparameters, train model 
    on optimized structure, hyperparameters, epochs
    Output:
    model - trained keras model with best parameters determined by Keras NN Tuner
    """
    
    def model_builder(hp):
        
        num_snpids = 1051 
        num_numerical_features = 2  
        
        input_numerical = tf.keras.layers.Input(shape=(num_numerical_features,), name="input_numerical")
        input_embedded = tf.keras.layers.Input(shape=(1,), name="input_embedded")
        
        embedding_size=50
        embedded_features = tf.keras.layers.Embedding(input_dim=num_snpids, output_dim=embedding_size)(input_embedded)
        flattened_features = tf.keras.layers.Flatten()(embedded_features)
    
        concatenated_features = tf.keras.layers.Concatenate()([flattened_features, input_numerical])
    
        mins = 32
        maxs = 160
        steps = 32
        hp_units1 = hp.Int('units1', min_value=mins, max_value=maxs, step=steps)    
        dense1 = Dense(units=hp_units1, activation="relu")(concatenated_features)
        
        hp_units2 = hp.Int('units2', min_value=mins, max_value=maxs, step=steps)    
        dense2 = Dense(units=hp_units2, activation="relu")(dense1)
        
        output_layer = Dense(units=3, activation="softmax")(dense2)
    
        model = tf.keras.models.Model(inputs=[input_embedded, input_numerical], outputs=output_layer)
    
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                    loss=tf.keras.losses.CategoricalFocalCrossentropy(),# my_categorical_focal_loss(),
                    metrics=['categorical_accuracy'])
    
        return model

    
    features_numerical = ["R", "Theta"]
    features_snpid = ["snpID_cat"]
    
    X_numerical_train = training_data[features_numerical].astype("float").to_numpy()
    X_snpid_train = training_data[features_snpid].astype("int").to_numpy()
    y_train = training_data[["GT_AA", "GT_AB", "GT_BB"]].astype("float").to_numpy()
    
    X_numerical_val = validation_data[features_numerical].astype("float").to_numpy()
    X_snpid_val = validation_data[features_snpid].astype("int").to_numpy()
    y_val = validation_data[["GT_AA", "GT_AB", "GT_BB"]].astype("float").to_numpy()

    tuner = kt.Hyperband(model_builder,
                         objective='val_categorical_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory=save_directory,
                         project_name=project_name)
    

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search([X_snpid_train, X_numerical_train], y_train, epochs=50, validation_data=([X_snpid_val, X_numerical_val], y_val), callbacks=[stop_early])
    
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    
    model = tuner.hypermodel.build(best_hps)

    history = model.fit([X_snpid_train, X_numerical_train], y_train, epochs=50, validation_data=([X_snpid_val, X_numerical_val], y_val))
    
    val_acc_per_epoch = history.history['val_categorical_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    
    hypermodel = tuner.hypermodel.build(best_hps)
     
    hypermodel.fit([X_snpid_train, X_numerical_train], y_train, epochs=best_epoch, validation_data=([X_snpid_val, X_numerical_val], y_val))

    return hypermodel 

def main():
    
    args = parse_arguments()

    if args.training_data.endswith(".parquet"):
        training_data = pd.read_parquet(args.training_data)
    elif args.training_data.endswith(".csv"):
        training_data = pd.read_csv(args.training_data, sep=",")
    else:
        print("Error: Training data file extension must be .csv or .parquet")
        sys.exit()       
    
    if args.validation_data.endswith(".parquet"):
        validation_data = pd.read_parquet(args.validation_data)
    elif args.validation_data.endswith(".csv"):
        validation_data = pd.read_csv(args.validation_data, sep=",")
    else: 
        print("Error: Validation data file extension must be .csv or .parquet")
        sys.exit()
        
    hypermodel = build_model(training_data, validation_data, args.save_directory, args.project_name)
    
    hypermodel.save(f"{args.save_directory}/{args.model_name}")
    
    if args.save_information:
        with open(f'{args.save_directory}/model_summary.txt', 'w') as f:
            with redirect_stdout(f):
                hypermodel.summary()
        
        config = hypermodel.get_config()                
        with open(f'{args.save_directory}/model_config.json', 'w') as f:
            json.dump(config, f, indent=4)                
        
        optimizer_config = hypermodel.optimizer.get_config()
        with open(f'{args.save_directory}/optimizer_config.json', 'w') as f:
            json.dump(optimizer_config, f, indent=4)

if __name__ == "__main__":
    main()