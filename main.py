import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

from ATCNet_mSEM_1 import ATCNet_
from preprocess0 import get_data

os.environ['TF_DETERMINISTIC_OPS'] = '0'


##%% Training
def train(dataset_conf, train_conf, results_path):
    # Get the current 'IN' time to calculate the overall training time
    in_exp = time.time()
    # Create a file to store the path of the best model among several runs
    best_models = open(results_path + "/best models.txt", "w")
    # Create a file to store performance during training
    log_write = open(results_path + "/log.txt", "w")

    # Get dataset paramters
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    # Get training hyperparamters
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    n_train = train_conf.get('n_train')
    model_name = train_conf.get('model')

    # Initialize variables
    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))
    recall = np.zeros((n_sub, n_train))
    precision = np.zeros((n_sub, n_train))
    f1 = np.zeros((n_sub, n_train))

    # Iteration over subjects
    # for sub in range(n_sub-1, n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
    for sub in range(n_sub):  # (num_sub): for all subjects, (i-1,i): for the ith subject.
        # Get the current 'IN' time to calculate the subject training time
        in_sub = time.time()
        print('\nTraining on subject ', sub + 1)
        log_write.write('\nTraining on subject ' + str(sub + 1) + '\n')
        # Initiating variables to save the best subject accuracy among multiple runs.
        BestSubjAcc = 0

        # Get training and test data

        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(
            data_path, sub, LOSO=LOSO, isStandard=isStandard)

        # Iteration over multiple runs
        for train in range(n_train):  # How many repetitions of training for subject i.
            tf.random.set_seed(train + 1)
            np.random.seed(train + 1)

            in_run = time.time()
            # Create folders and files to save trained models for all runs
            filepath = results_path + '/saved models/run-{}'.format(train + 1)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            filepath = filepath + '/subject-{}.weights.h5'.format(sub + 1)

            # Create the model
            model = getModel(model_name, dataset_conf)
            # Compile and train the model
            model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])

            callbacks = [
                ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0,
                                save_best_only=True, save_weights_only=True, mode='max'),

                ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=20, verbose=1, min_lr=0.0001),

                EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
            ]
            history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot),
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)

            # Evaluate the performance of the trained model.
            # Here we load the Trained weights from the file saved in the hard
            # disk, which should be the same as the weights of the current model.
            model.load_weights(filepath)
            y_pred = model.predict(X_test).argmax(axis=-1)
            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, train] = accuracy_score(labels, y_pred)
            kappa[sub, train] = cohen_kappa_score(labels, y_pred)
            precision[sub, train], recall[sub, train], f1[sub, train], _ = precision_recall_fscore_support(labels,
                                                                                                           y_pred,
                                                                                                           average='macro')

            # Get the current 'OUT' time to calculate the 'run' training time
            out_run = time.time()
            # Print & write performance measures for each run
            info = 'Subject: {}   Train no. {}   Time: {:.1f} m   '.format(sub + 1, train + 1,
                                                                           ((out_run - in_run) / 60))
            info = info + 'Test_acc: {:.4f}   Test_kappa: {:.4f}  Test_precision: {:.4f}  Test_recall: {:.4f}  Test_f1: {:.4f}'.format(
                acc[sub, train], kappa[sub, train], precision[sub, train], recall[sub, train], f1[sub, train])
            print(info)
            log_write.write(info + '\n')
            # If current training run is better than previous runs, save the history.
            if (BestSubjAcc < acc[sub, train]):
                BestSubjAcc = acc[sub, train]

        # Store the path of the best model among several runs
        best_run = np.argmax(acc[sub, :])
        filepath = '/saved models/run-{}/subject-{}.weights.h5'.format(best_run + 1, sub + 1) + '\n'
        best_models.write(filepath)
        # Get the current 'OUT' time to calculate the subject training time
        out_sub = time.time()
        # Print & write the best subject performance among multiple runs
        info = '----------\n'
        info = info + 'Subject: {}   best_run: {}   Time: {:.1f} m   '.format(sub + 1, best_run + 1,
                                                                              ((out_sub - in_sub) / 60))
        info = info + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(acc[sub, best_run], np.average(acc[sub, :]),
                                                                          acc[sub, :].std())
        info = info + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(kappa[sub, best_run],
                                                                           np.average(kappa[sub, :]),
                                                                           kappa[sub, :].std())
        info = info + 'precision: {:.4f}   avg_precision: {:.4f} +- {:.4f}'.format(precision[sub, best_run],
                                                                                   np.average(precision[sub, :]),
                                                                                   precision[sub, :].std())
        info = info + 'recall: {:.4f}   avg_recall: {:.4f} +- {:.4f}'.format(recall[sub, best_run],
                                                                             np.average(recall[sub, :]),
                                                                             recall[sub, :].std())
        info = info + 'f1: {:.4f}   avg_f1: {:.4f} +- {:.4f}'.format(f1[sub, best_run],
                                                                     np.average(f1[sub, :]),
                                                                     f1[sub, :].std())
        info = info + '\n----------'
        print(info)
        log_write.write(info + '\n')

    # Get the current 'OUT' time to calculate the overall training time
    out_exp = time.time()
    info = '\nTime: {:.1f} h   '.format((out_exp - in_exp) / (60 * 60))
    print(info)
    log_write.write(info + '\n')

    # Close open files
    best_models.close()
    log_write.close()


##%% Evaluation
def test(model, dataset_conf, results_path, allRuns=False):
    # Open the  "Log" file to write the evaluation results
    log_write = open("results/log.txt", "a")
    # Open the file that stores the path of the best models among several random runs.
    best_models = open(results_path + "/best models.txt", "r")

    # Get dataset paramters
    dataset = dataset_conf.get('name')
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    classes_labels = dataset_conf.get('cl_labels')

    # Initialize variables
    acc_bestRun = np.zeros(n_sub)
    kappa_bestRun = np.zeros(n_sub)
    recall_bestRun = np.zeros(n_sub)
    precision_bestRun = np.zeros(n_sub)
    f1_bestRun = np.zeros(n_sub)
    cf_matrix = np.zeros([n_sub, n_classes, n_classes])

    for sub in range(n_sub):
        # Load data
        _, _, _, X_test, _, y_test_onehot = get_data(data_path, sub, dataset, LOSO=LOSO, isStandard=isStandard)

        # Load the best model out of multiple random runs (experiments).
        filepath = best_models.readline()
        model.load_weights(results_path + filepath[:-1], by_name=False)
        y_pred = model.predict(X_test).argmax(axis=-1)
        labels = y_test_onehot.argmax(axis=-1)
        acc_bestRun[sub] = accuracy_score(labels, y_pred)
        kappa_bestRun[sub] = cohen_kappa_score(labels, y_pred)
        precision_bestRun[sub, train], recall_bestRun[sub, train], f1_bestRun[
            sub, train], _ = precision_recall_fscore_support(labels, y_pred, average='macro')

        # Print & write performance measures for each subject
        info = 'Subject: {}   best_run: {:2}  '.format(sub + 1,
                                                       (filepath[filepath.find('run-') + 4:filepath.find('/sub')]))
        info = info + 'acc: {:.4f}   kappa: {:.4f}   precision: {:.4f}   recall: {:.4f}   f1: {:.4f} '.format(
            acc_bestRun[sub], kappa_bestRun[sub], precision_bestRun[sub], recall_bestRun[sub], f1_bestRun[sub])

        print(info)
        log_write.write('\n' + info)

    # Print & write the average performance measures for all subjects
    info = '\nAverage of {} subjects - best runs:\nAccuracy = {:.4f}   Kappa = {:.4f}   Precision = {:.4f}   Recall = {:.4f}    f1 = {:.4f}\n'.format(
        n_sub, np.average(acc_bestRun), np.average(kappa_bestRun), np.average(precision_bestRun),
        np.average(recall_bestRun), np.average(f1_bestRun))
    print(info)
    log_write.write(info)

    # Close open files
    log_write.close()


##%%
def getModel(model_name, dataset_conf):
    n_classes = dataset_conf.get('n_classes')
    n_channels = dataset_conf.get('n_channels')
    in_samples = dataset_conf.get('in_samples')

    # Select the model
    if (model_name == 'ATCNet'):
        model = ATCNet_(
            # Dataset parameters
            n_classes=n_classes,
            in_chans=n_channels,
            in_samples=in_samples,
            # Sliding window (SW) parameter
            n_windows=5,
            # Attention (AT) block parameter
            attention='mha',  # Options: None, 'mha','mhla', 'cbam', 'se'
            # Convolutional (CV) block parameters
            eegn_F1=32,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            # Temporal convolutional (TC) block parameters
            tcn_depth=2,
            tcn_kernelSize=4,
            tcn_filters=64,
            tcn_dropout=0.3,
            tcn_activation='elu'
        )
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model


##%%
def run():
    # Define dataset parameters
    dataset = 'BCI2a'

    if dataset == 'BCI2a':
        in_samples = 1125
        n_channels = 22
        n_sub = 9
        n_classes = 4
        classes_labels = ['Left hand', 'Right hand', 'Foot', 'Tongue']
        data_path = 'D:/02-learning-data/data/BCI Competition IV-2a/'
        # os.path.expanduser(')~') + '/BCI Competition IV-2a/
    else:
        raise Exception("'{}' dataset is not supported yet!".format(dataset))

    # Create a folder to store the results of the experiment
    results_path = os.getcwd() + "/results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)  # Create a new directory if it does not exist

    # Set dataset paramters
    dataset_conf = {'name': dataset, 'n_classes': n_classes, 'cl_labels': classes_labels,
                    'n_sub': n_sub, 'n_channels': n_channels, 'in_samples': in_samples,
                    'data_path': data_path, 'isStandard': True, 'LOSO': False}
    # Set training hyperparamters
    train_conf = {'batch_size': 64, 'epochs': 1000, 'patience': 300, 'lr': 0.001,
                  'LearnCurves': True, 'n_train': 10, 'model': 'ATCNet'}

    # Train the model
    train(dataset_conf, train_conf, results_path)

    # Evaluate the model based on the weights saved in the '/results' folder
    model = getModel(train_conf.get('model'), dataset_conf)
    test(model, dataset_conf, results_path)


##%%
if __name__ == "__main__":
    run()
