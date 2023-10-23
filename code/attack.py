import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from .preprocess import *

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband

logger = logging.getLogger("logger")

class CustomHyperModel(HyperModel):
    def __init__(self, mode, input_shape, num_classes):
        self.mode = mode
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build(self, hp):
        model = Sequential()

        model.add(Input(shape=self.input_shape))
        
        if self.mode == "rnn":
            for i in range(hp.Int('num_layers', 1, 3)):
                model.add(SimpleRNN(
                    units = hp.Int('units', min_value=64, max_value=256, step=16),
                    activation = hp.Choice('activation', values=['relu']),
                    return_sequences = True if i < hp.Int('num_layers', 1, 3) - 1 else False
                ))
        elif self.mode == "lstm":
            for i in range(hp.Int('num_layers', 1, 3)):
                model.add(LSTM(
                    units = hp.Int('units', min_value=64, max_value=256, step=16),
                    activation = hp.Choice('activation', values=['relu'], default='relu'),
                    return_sequences = True if i < hp.Int('num_layers', 1, 3) - 1 else False
                ))
        
        model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.3, step=0.1, default=0.3)))
        
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(
            optimizer = Adam(
                hp.Choice('learning_rate', values=[1e-3])
            ),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )

        return model

def check_single_class(y):
    if len(np.unique(y)) == 1:
        logger.info("Only 1 class in y. Skip.")
        logger.info(f"y: {y}")
        # return True
        return False
    
    return False

def dt_run(X, y, valid_X, valid_y):
    logger.info("Running Decision Tree...")

    X = np.array(X)
    y = np.array(y)

    params = {
        'max_depth': [5, 10, 15, 20, 25, 30],
        'min_samples_leaf': [2, 4, 8, 12, 16, 20],
        'min_samples_split': [2, 4, 8, 12, 16, 20],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    model = GridSearchCV(RandomForestClassifier(), params, cv=5, n_jobs=-1)

    # y의 클래스가 1개일 경우
    if not check_single_class(y):
        model.fit(X, y)

    return model

def rf_run(X, y, valid_X, valid_y):
    logger.info("Running Random Forest...")

    X = np.array(X)
    y = np.array(y)

    params = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [5, 10, 15, 20, 25, 30],
        'min_samples_leaf': [2, 4, 6, 8, 10],
        'min_samples_split': [2, 4, 6, 8, 10],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    model = GridSearchCV(RandomForestClassifier(), params, cv=5, n_jobs=-1)

    # y의 클래스가 1개일 경우
    if not check_single_class(y):
        model.fit(X, y)

    return model

def rnn_lstm_generate(X, y, valid_X, valid_y, mode):
    if X.shape[0] != y.shape[0]:
        logger.error("X, y shape mismatch.")
        exit(1)
    
    logger.debug(f"X shape: {X.shape}")
    logger.debug(f"y shape: {y.shape}")

    label_to_index = {label: i for i, label in enumerate(np.unique(y))}
    index_to_label = {i: label for label, i in label_to_index.items()}
    
    # y 변환
    unique_y = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_y)}
    y = np.array([label_map[label] for label in y])

    num_classes=len(unique_y)
    input_shape=(None, 4)

    y = to_categorical(y, num_classes=num_classes)

    # valid_y 변환
    valid_y = np.array([label_map[label] for label in valid_y])
    valid_y = to_categorical(valid_y, num_classes=num_classes)

    # hyperparameter tuning
    hypermodel = CustomHyperModel(mode, input_shape, num_classes)
    tuner = Hyperband(
        hypermodel,
        objective='val_accuracy',
        max_epochs=40,
        directory='hyperband',
        factor=3,
        project_name=f"{mode}_hyperband",
        overwrite=True
    )

    tuner.search_space_summary()

    tuner.search(X, y, epochs=40, validation_data=(valid_X, valid_y))

    best_model = tuner.get_best_models(num_models=1)[0]

    best_model.summary()

    best_model.fit(X, y, epochs=40, validation_data=(valid_X, valid_y))

    return best_model

def rnn_run(X, y, valid_X, valid_y):
    logger.info("Running RNN...")

    return rnn_lstm_generate(X, y, valid_X, valid_y, "rnn")

def lstm_run(X, y, valid_X, valid_y):
    logger.info("Running LSTM...")

    return rnn_lstm_generate(X, y, valid_X, valid_y, "lstm")

def attack_learn(flows, valid_flows, labels, algorithm):
    logger.info(f"Creating {algorithm} model...")

    algorithm_func = {
        "rf": rf_run, 
        "dt": dt_run, 
        "rnn": rnn_run,
        "lstm": lstm_run
    }

    if algorithm == "rnn" or algorithm == "lstm":
        X, y = extract_attack_features(flows, labels)
        valid_X, valid_y = extract_attack_features(valid_flows, labels)
    else:
        X, y = extract_attack_features_b(flows, labels)
        valid_X, valid_y = extract_attack_features_b(valid_flows, labels)

    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)

    valid_X = np.array(valid_X).astype(np.float32)
    valid_y = np.array(valid_y).astype(np.float32)

    logger.debug(f"X shape: {X.shape}")
    logger.debug(f"y shape: {y.shape}")
    logger.debug(f"number of benign: {len(y[y == 1])}")
    logger.debug(f"number of attack: {len(y[y == 0])}")
    
    model = algorithm_func[algorithm](X, y, valid_X, valid_y)

    logger.info(f"Created {algorithm} model.")

    # 생성 시간을 포함한 이름으로 모델 저장
    model_name = f"attack_{algorithm}_{time.strftime('%Y%m%d_%H%M%S')}"
    with open(f"./model/{model_name}.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Saved {algorithm} model as {model_name}.pkl.")

    return model

def attack_evaluate(test_flows, labels, algorithm, model):
    logger.info(f"Evaluating {algorithm} model...")

    if algorithm == "rnn" or algorithm == "lstm":
        X, y = extract_attack_features(test_flows, labels)
        X = np.array(X).astype(np.float32)
        y = np.array(y).astype(np.float32)

        label_to_index = {label: i for i, label in enumerate(np.unique(y))}
        index_to_label = {i: label for label, i in label_to_index.items()}

        unique_y = np.unique(y)
        label_map = {label: i for i, label in enumerate(unique_y)}
        y = np.array([label_map[label] for label in y])

        y = to_categorical(y, num_classes=len(unique_y))
        
        y_pred = model.predict(X)
        
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y, axis=1)

        logger.debug(f"y_pred: {y_pred[:10]}")
        logger.debug(f"y_true: {y_true[:10]}")
    else:
        X, y = extract_attack_features_b(test_flows, labels)
        y_pred = model.predict(X)
        y_true = y

    make_heatmap("./result/", y_true, y_pred, labels, algorithm)
    print_score(y_true, y_pred, algorithm)
 
def make_heatmap(path, y_true, y_pred, labels,  algorithm):
    # label index 생성
    label_to_index = {label: i for i, label in enumerate(np.unique(y_true))}
    index_to_label = {i: label for label, i in label_to_index.items()}

    # confusion matrix 생성
    cm = confusion_matrix(y_true, y_pred)

    # confusion matrix heatmap 생성
    plt.figure(figsize=(20, 20))

    labels = [index_to_label[i] for i in range(len(np.unique(y_true)))]

    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        annot_kws={"size": 15},
        xticklabels=labels, 
        yticklabels=labels
    )

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.xticks(np.arange(len(labels))+0.5, labels=labels, rotation=90)
    plt.yticks(np.arange(len(labels))+0.5, labels=labels, rotation=0)

    plt.title(f"{algorithm} model confusion matrix")

    plt.savefig(f"{path}attack_{algorithm}_{time.strftime('%Y%m%d_%H%M%S')}_heatmap.png")

def print_score(y_true, y_pred, algorithm):
    with open(f"./result/attack_{algorithm}_{time.strftime('%Y%m%d_%H%M%S')}_score.txt", 'w') as out:
        out.write(f"Accuracy: {accuracy_score(y_true, y_pred)}\n")
        out.write(f"Precision: {precision_score(y_true, y_pred, average='weighted')}\n")
        out.write(f"Recall: {recall_score(y_true, y_pred, average='weighted')}\n")
        out.write(f"F1: {f1_score(y_true, y_pred, average='weighted')}\n")

    logger.info(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    logger.info(f"Precision: {precision_score(y_true, y_pred, average='weighted')}")
    logger.info(f"Recall: {recall_score(y_true, y_pred, average='weighted')}")
    logger.info(f"F1: {f1_score(y_true, y_pred, average='weighted')}")