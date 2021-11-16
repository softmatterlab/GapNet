''' GapNet
Neural Network Training with Highly Incomplete Datasets - 29 July 2021
© Laura Natali, Yu-Wei Chang, Oveis Jamialahmadi,Stefano Romeo, Joana B. Pereira & Giovanni Volpe
http://www.softmatterlab.org
'''
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib as mpl
import scipy.stats as st
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import metrics, layers, losses
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
mpl.rcParams['figure.figsize'] = (12, 10)


METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    verbose=0,
    patience=30,
    mode='auto',
    restore_best_weights=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
EPOCHS = 200
BATCH_SIZE = 32

kern_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
bias_init = tf.keras.initializers.Zeros()

class generate_gapnet_model():
    """Generator of gapnet model."""
    
    def __init__(self, cluster_sizes: np.ndarray, n_feature : int = 40, n_classes : int = 2):
        """Define the GapNet model
        
        Parameters
        ---------- 
        cluster_sizes: (required) Number of features in each cluster (i.e. cluster_sizes[0] is the size of the first cluster)
        n_feature: positive integer default 40
        n_classes: positive integer default 2
        """
        
        self.Nfeat = n_feature
        self.Nclass = n_classes
        self.cluster_sizes = cluster_sizes
        self.architecture = {}
        self.history = {}
        self.best_epochs = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_aucs = []
        self.val_precisions = []
        self.val_sensitivities = []
        self.val_specificities = []
        self.val_y_preds = []
        self.val_y_labels = []
    
    def build_model(self, n_dense: int = 2, n_nodes: int = 0, 
                 dropout_rate: tf.keras.layers.Dropout = 0.5,
                 activation_function: tf.keras.activations = "relu", 
                 output_activation : tf.keras.activations = "softmax",
                 optim : tf.keras.optimizers = "adam",
                 learning_rate : float = 0.0,
                 loss_function : tf.keras.losses = 'categorical_crossentropy',
                 show_summary: bool = False):
        """Generate the neural network architecture.
        
        Parameters
        ----------
        n_dense: int, optional 
            Number of dense layers default = 2
        n_nodes: int, optional
            Number of nodes per layer default = 2*n_features
        dropout_rate: float, optional
            rate for the dropout layer, value between 0 and 1
        activation_function: function, optional
            one of keras built-in functions default relu
        output_activation: function, optional
            one of keras built-in functions default softmax
        optim: one of keras build-in optimizers default is adam
        learning_rate: float, optional
            between 0 and 1 default Adam learning rate = 0.001
        loss_function: function, optional
            one of keras build-in loss functions default is categorical crossentropy
        show_summary: bool, optional
            if True the keras model summary is printed, default = False
        """
        
        from tensorflow.python.keras.models import Sequential, Model

        self.architecture = {}
        self.history = {}
        
        # default number of nodes: twice the number of features
        if n_nodes == 0:
            n_nodes = np.multiply(self.cluster_sizes,2)
     
        
        # check that the dropout rate is in the correct range
        if dropout_rate> 1.0 or dropout_rate<0.0 :
            raise ValueError("The dropout rate must be a value between 0 and 1 ")
         
        # Assign correct learning rate if provided   
        if learning_rate > 0:
            optim.learning_rate.assign(learning_rate)

        # Convert the list into array if necessary
        if (type(self.cluster_sizes) == list):
            self.cluster_sizes = np.array(self.cluster_sizes)

        # Check that all lengths are positive
        if np.any( self.cluster_sizes < 0):
            raise ValueError("The number of features must always be positive")

        # Store the number of clusters
        n_clust = len(self.cluster_sizes)
        
        for i in range(n_clust):
            
            self.architecture['clust_'+str(i)] =  tf.keras.Sequential()
            print("Generating the "+str(i+1)+" neural network model ... ")  
            
            # Add the input layer
            self.architecture['clust_'+str(i)].add(layers.Input( shape = (self.cluster_sizes[i],) ))
        
            # Add n_dense consecutive dense layers followed by dropout layers
            for nl in range(n_dense):
                
                self.architecture['clust_'+str(i)].add(layers.Dense(n_nodes[i], activation = activation_function))
                self.architecture['clust_'+str(i)].add(layers.Dropout(dropout_rate))
                
            # Add the output layer with activation function "output_activation"
            self.architecture['clust_'+str(i)].add(layers.Dense(self.Nclass, activation=output_activation))
        
            self.architecture['clust_'+str(i)].build((None, self.cluster_sizes[i]))
        
            # Compile the model
            self.architecture['clust_'+str(i)].compile(optimizer = optim, loss = loss_function, metrics=METRICS)  
      
        # The list of input layers for each cluster    
        input_layer = [layers.Input( shape =(s,) ) for s in self.cluster_sizes]   
        
        print("Generating the final gapnet model ... ")    

        # Build the next layer on top of the previous one    
        prev_layer = input_layer
        for nl in range(n_dense):
            
            new_layer = [layers.Dense(n_nodes[i], activation = activation_function )(prev_layer[i]) for i in range(n_clust)]
            prev_layer = new_layer
            new_layer = [layers.Dropout(dropout_rate)(prev_layer[i]) for i in range(n_clust)]
            prev_layer = new_layer
        
        # Concatenate the list of dense layers
        conc_layer = layers.Concatenate(axis=1)(new_layer)
        
        # Build the output layer
        output_layer =  layers.Dense( self.Nclass, activation = output_activation )(conc_layer)
                       
        # Construct the Model    
        self.architecture['gapnet'] =  tf.keras.Model(inputs= input_layer, outputs=[output_layer])

        # Set all the dense layers as not trainable
        for nl, layer in enumerate(self.architecture['gapnet'].layers):
            if nl <= n_clust*2*n_dense:
                layer.trainable = False

        # Compile the gapnet architecture
        self.architecture['gapnet'].compile(optimizer = optim, loss = loss_function, metrics = METRICS)
  
        if show_summary:
            print(self.architecture['gapnet'].summary())    

        # store this values for the future
        self.Ndense = n_dense
        self.Nclust = n_clust            
            
            
    def train_first_stage(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray ):
        
        """ Train the first stage of the network on the clusters of features
        
        Parameters
        ----------
        X: np.ndarray of training inputs
        y: np.nedarray of training labels
        """  
        
        from numpy import isnan
        from tensorflow.keras.utils import to_categorical
        
        start_feat = 0
        
        # Train one cluster of features at a time
        for ns, s in enumerate(self.cluster_sizes):
            
            # select the features of interest
            X_temp = X[:,start_feat : start_feat + s]
            X_val_temp = X_val[:,start_feat : start_feat + s]

            # cut the missing values
            y_temp =  y[ ~isnan( X_temp ).any( axis=1 )] 
            y_val_temp =  y_val[ ~isnan( X_val_temp ).any( axis=1 )] 
            X_temp = X_temp[ ~isnan( X_temp ).any( axis=1 )] 
            X_val_tmp = X_val_temp[ ~isnan( X_val_temp ).any( axis=1 )] 
            
            self.history['clust_'+str(ns)] = self.architecture['clust_'+str(ns)].fit(X_temp, to_categorical(y_temp),  
                        #epochs=EPOCHS, verbose=0,callbacks=[early_stopping], 
                        epochs=EPOCHS, verbose=0, 
                        validation_data = ( X_val_tmp, to_categorical(y_val_temp)) )
        
            start_feat = start_feat + s
            #print("Training process of clust #{} is done.".format(ns+1))
    
        print("Training process of first stage is done.")

    def train_second_stage(self, X: np.ndarray, y:np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """ Train the concatenated network on the complete dataset
        
        
        Parameters
        ----------
        X: np.ndarray of training inputs
        y: np.nedarray of training labels
        """
        
        from numpy import isnan
        from tensorflow.keras.utils import to_categorical
        
        # retrain only on complete values
        y = y[ ~isnan( X ).any( axis=1 )]
        X = X[ ~isnan( X ).any( axis=1 )]
        y_val = y_val[ ~isnan( X_val ).any( axis=1 )]
        X_val = X_val[ ~isnan( X_val ).any( axis=1 )]

        # Convert the inputs into the correct shape
        X_list = []
        X_val_list = []
        
        start_point = 0
        
        for size in self.cluster_sizes:
            X_list.append( X[:, start_point:start_point + size] )
            X_val_list.append( X_val[:, start_point:start_point + size] )
            start_point = start_point + size

        # Arrays to select the matching layer from the correct network 
        cluster_arr = np.tile(np.arange(self.Nclust), 2*self.Ndense)
        layer_arr = np.sort( np.tile(np.arange(2*self.Ndense ), self.Nclust) )

        # Assign weights from first stage   
        for n, (nl, nc) in enumerate(zip( layer_arr, cluster_arr)) :

                self.architecture['gapnet'].layers[n + self.Nclust].set_weights( 
                        self.architecture['clust_'+str(nc)].layers[nl].get_weights() )   

        # Train the gapnet on complete data
        self.history['gapnet'] = self.architecture['gapnet'].fit( 
                        X_list, to_categorical(y), epochs = EPOCHS, verbose = 0,
                        #callbacks = [early_stopping], validation_data = (X_val_list, to_categorical(y_val)) )
                        validation_data = (X_val_list, to_categorical(y_val)) )

        best_epoch = np.argmin(self.history['gapnet'].history['val_loss'])+1
        train_accuracy = self.history['gapnet'].history['accuracy'][best_epoch-1]
        val_auc = self.history['gapnet'].history['val_auc'][best_epoch-1]
        val_y_pred = self.architecture['gapnet'].predict(X_val_list)
        threshold = 0.5
        
        val_y_pred_class = np.where(val_y_pred>threshold, 1, 0)
        m = tf.keras.metrics.TruePositives()
        m.update_state(val_y_pred_class[:,1], y_val)
        val_tp = m.result().numpy()
        m = tf.keras.metrics.TrueNegatives()
        m.update_state(val_y_pred_class[:,1], y_val)
        val_tn = m.result().numpy()
        m = tf.keras.metrics.FalsePositives()
        m.update_state(val_y_pred_class[:,1], y_val)
        val_fp = m.result().numpy()
        m = tf.keras.metrics.FalseNegatives()
        m.update_state(val_y_pred_class[:,1], y_val)
        val_fn = m.result().numpy()
        
        val_accuracy = (val_tp+val_tn)/(val_tp+val_tn+val_fp+val_fn)
        val_sensitivity = (val_tp)/(val_tp+val_fn)
        val_specificity = (val_tn)/(val_tn+val_fp)
        val_precision = (val_tp)/(val_tp+val_fp)
        val_auc = roc_auc_score(y_val, val_y_pred[:,1])

        self.best_epochs.append(best_epoch)
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)
        self.val_aucs.append(val_auc)
        self.val_precisions.append(val_precision)
        self.val_sensitivities.append(val_sensitivity)
        self.val_specificities.append(val_specificity)
        self.val_y_preds = np.append(self.val_y_preds,val_y_pred[:,1])
        self.val_y_labels = np.append(self.val_y_labels, y_val)
        print("Training process of second stage is done.")

        
    def reset_weight(self):
        """ Initialize the weights in the gapnet before each run.
        """
        
        for model_name in self.architecture:
            lengths = list(map(np.shape, self.architecture[model_name].get_weights()))
            lengths = np.array(lengths, dtype=object).reshape(int(len(lengths) / 2), 2)
 
            for n, layer in enumerate(self.architecture[model_name].layers):

                if layer.count_params() > 0:
                    new_weights = [
                        kern_init(shape=lengths[n, 0]),
                        bias_init(shape=lengths[n, 1]),
                    ]
                    layer.set_weights(new_weights)

                else:
                    lengths = np.vstack(([(0, 0), (0,)], lengths))

     
    
        
class generate_vanilla_model():
    """Generator of vanilla model."""
    
    def __init__(self, n_feature: int = 40, n_classes: int = 2):
        """Define the Vanilla model
        
        Parameters
        ----------
        n_feature: int, optional 
            positive integer default 40
        n_classes: int, optional
            positive integer default 2
        """        
        
        self.Nfeat = n_feature
        self.Nclass = n_classes
        self.architecture = None
        self.history = []
        self.best_epochs = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_aucs = []
        self.val_precisions = []
        self.val_sensitivities = []
        self.val_specificities = []
        self.val_y_preds = []
        self.val_y_labels = []

    def build_model(self, n_dense: int = 2, n_nodes: int = 0, 
                 dropout_rate: tf.keras.layers.Dropout = 0.5,
                 activation_function: tf.keras.activations = "relu", 
                 output_activation : tf.keras.activations = "softmax",
                 optim : tf.keras.optimizers = tf.keras.optimizers.Adam(),
                 learning_rate : float = 0.0,
                 loss_function : tf.keras.losses = 'categorical_crossentropy',
                 show_summary: bool = False):
        
        """Generate the neural network architecture.
        
        Parameters
        ----------
        n_dense: int, optional 
            Number of dense layers default = 2
        n_nodes: int, optional
            Number of nodes per layer default = 2*n_features
        dropout_rate: float, optional
            rate for the dropout layer, value between 0 and 1
        activation_function: function, optional
            one of keras built-in functions default relu
        output_activation: function, optional
            one of keras built-in functions default softmax
        optim: one of keras build-in optimizers default is adam
        learning_rate: float, optional
            between 0 and 1 default Adam learning rate = 0.001
        loss_function: function, optional
            one of keras build-in loss functions default is categorical crossentropy
        show_summary: bool, optional
            if True the keras model summary is printed, default = False
        """
        
        from tensorflow.python.keras.models import Sequential, Model

        # default number of nodes: twice the number of features
        if n_nodes == 0:
            n_nodes = 2*self.Nfeat
        
        # check that the dropout rate is in the correct range
        if dropout_rate> 1.0 or dropout_rate<0.0 :
            raise ValueError("The dropout rate must be a value between 0 and 1 ")
         
        # Assign correct learning rate if provided   
        if learning_rate > 0:
            optim.learning_rate.assign(learning_rate)

        self.architecture = {}
        self.history = {}
            
        from tensorflow import keras
        self.architecture = keras.Sequential()
        
        # Add the input layer
        self.architecture.add(tf.keras.layers.Input(shape=(self.Nfeat,)))
        
        # Add n_dense consecutive dense layers followed by dropout layers
        for nl in range(n_dense):
            self.architecture.add(tf.keras.layers.Dense(n_nodes, activation = activation_function))
            self.architecture.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Add the output layer with activation function "output_activation"
        self.architecture.add(tf.keras.layers.Dense(self.Nclass, activation=output_activation))
        
        # Compile the model
        self.architecture.compile(optimizer = optim, loss = loss_function, metrics=METRICS)
        if(show_summary):
            print(self.architecture.summary())

        
        
    def train_single_stage(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Training the Vanilla neural network on complete data.
        
        Parameters
        ----------        
        X_train: np.ndarray of training inputs
        y_train: np.nedarray of training labels
        X_test: np.ndarray of testing inputs
        y_test: np.nedarray of testing labels
        """
        self.build_model(show_summary=False)
        self.history = self.architecture.fit(X_train, to_categorical(y_train), epochs=EPOCHS, verbose=0,
        #    callbacks=[early_stopping], validation_data=(X_val, to_categorical(y_val)))
        validation_data=(X_val, to_categorical(y_val)))
        
        
        best_epoch = np.argmin(self.history.history['val_loss'])+1
        train_accuracy = self.history.history['accuracy'][best_epoch-1]
        val_auc = self.history.history['val_auc'][best_epoch-1]
        val_y_pred = self.architecture.predict(X_val)
        threshold = 0.5
        val_y_pred_class = np.where(val_y_pred>threshold, 1, 0)
        m = tf.keras.metrics.TruePositives()
        m.update_state(val_y_pred_class[:,1], y_val)
        val_tp = m.result().numpy()
        m = tf.keras.metrics.TrueNegatives()
        m.update_state(val_y_pred_class[:,1], y_val)
        val_tn = m.result().numpy()
        m = tf.keras.metrics.FalsePositives()
        m.update_state(val_y_pred_class[:,1], y_val)
        val_fp = m.result().numpy()
        m = tf.keras.metrics.FalseNegatives()
        m.update_state(val_y_pred_class[:,1], y_val)
        val_fn = m.result().numpy()

        val_accuracy = (val_tp+val_tn)/(val_tp+val_tn+val_fp+val_fn)
        val_sensitivity = (val_tp)/(val_tp+val_fn)
        val_specificity = (val_tn)/(val_tn+val_fp)
        val_precision = (val_tp)/(val_tp+val_fp)
        val_auc = roc_auc_score(y_val, val_y_pred[:,1])

        self.best_epochs.append(best_epoch)
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)
        self.val_aucs.append(val_auc)
        self.val_precisions.append(val_precision)
        self.val_sensitivities.append(val_sensitivity)
        self.val_specificities.append(val_specificity)
        self.val_y_preds = np.append(self.val_y_preds,val_y_pred[:,1])
        self.val_y_labels = np.append(self.val_y_labels, y_val)
        print("Training process of vanilla is done.")
        
def present_results(model):
    """Evauate the training results.    
        
    Parameters
    ----------   
    model: obejct of class generate_vanilla_model() or generate_gapnet_model()
    """
        
    print("Results :")
    print("best_epochs {}".format(model.best_epochs))
    print("train_accuracy {:.3f}+/-{:.3f} : {}".format(np.mean(model.train_accuracies), np.std(model.train_accuracies), np.round(model.train_accuracies, 3)))
    print("test_accuracy {:.3f}+/-{:.3f} : {}".format(np.mean(model.val_accuracies), np.std(model.val_accuracies), np.round(model.val_accuracies, 3)))
    print("test_auc {:.3f}+/-{:.3f} : {}".format(np.mean(model.val_aucs), np.std(model.val_aucs), np.round(model.val_aucs, 3)))
    print("test_sens {:.3f}+/-{:.3f} : {}".format(np.mean(model.val_sensitivities), np.std(model.val_sensitivities), np.round(model.val_sensitivities, 3)))
    print("test_spec {:.3f}+/-{:.3f} : {}".format(np.mean(model.val_specificities), np.std(model.val_specificities), np.round(model.val_specificities, 3)))
    print("test_prec {:.3f}+/-{:.3f} : {}".format(np.mean(model.val_precisions), np.std(model.val_precisions), np.round(model.val_precisions, 3)))

def preprocess_with_missing_data(X,Y):
    """Preprocess the input data and split it into subsets for training and test sets.    
        
    Parameters
    ----------   
    X: np.ndarray of overall dataset
    Y: np.nedarray of overall labels
    """
    
    from numpy import isnan
    from sklearn.preprocessing import StandardScaler
    
    Y_overlap = Y[ ~isnan( X ).any( axis=1 )]
    X_overlap = X[ ~isnan( X ).any( axis=1 )]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_overlap, Y_overlap, test_size=0.2, random_state=42)
    X_incomplete = X[ isnan( X ).any( axis=1 )]
    Y_incomplete = Y[ isnan( X ).any( axis=1 )]
    index = np.argwhere(isnan( X_incomplete ))
    rows, cols = zip(*index)
    X_incomplete[rows, cols] = 0
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_incomplete = scaler.transform(X_incomplete)
    X_test = scaler.transform(X_test)
    
    thres = 20
    X_incomplete = np.clip(X_incomplete, -thres, thres)
    X_train = np.clip(X_train, -thres, thres)
    X_test = np.clip(X_test, -thres, thres)
    
    X_incomplete[rows, cols] = np.nan
    X_train_overall = np.append(X_train, X_incomplete,axis=0)
    Y_train_overall = np.append(Y_train, Y_incomplete,axis=0)
    
    return X_train_overall, Y_train_overall, X_train, Y_train, X_test, Y_test

def separate_missing_data(X,Y):
    """Split the input data into subsets for complete and incomplete cases.    
        
    Parameters
    ----------   
    X: np.ndarray of overall dataset
    Y: np.nedarray of overall labels
    """
    
    from numpy import isnan
    from sklearn.preprocessing import StandardScaler
    
    Y_overlap = Y[ ~isnan( X ).any( axis=1 )]
    X_overlap = X[ ~isnan( X ).any( axis=1 )]
    
    X_incomplete = X[ isnan( X ).any( axis=1 )]
    Y_incomplete = Y[ isnan( X ).any( axis=1 )]
    
    return X_overlap, Y_overlap, X_incomplete, Y_incomplete

def preprocess_standardization(X_train, X_test):
    """Standarize the input data based on training set and apply it on testing set.    
        
    Parameters
    ----------   
    X_train: np.ndarray of training dataset
    X_test: np.ndarray of testing labels
    """
    
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test

def preprocess_standardization_with_missing_data(X_train, Y_train, X_test, X_incomplete, Y_incomplete):
    """Standarize the input data based on training dataset and apply it on testing dataset and incomplete dataset. 
    Also concatenate the incomplete dataset with training dataset after the standarization.
        
    Parameters
    ----------   
    X_train: np.ndarray of training dataset
    X_test: np.ndarray of testing labels
    """
    
    from numpy import isnan
    from sklearn.preprocessing import StandardScaler
    import copy
    from sklearn.impute import SimpleImputer
    
    X_incomplete_features = copy.deepcopy(X_incomplete)
    index = np.argwhere(isnan( X_incomplete_features ))
    rows, cols = zip(*index)
    X_incomplete_features[rows, cols] = 0
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_incomplete_features = scaler.transform(X_incomplete_features)
    X_test = scaler.transform(X_test)
    
    X_incomplete_features[rows, cols] = np.nan
    
    X_train_overall = np.append(X_train, X_incomplete_features, axis=0)
    Y_train_overall = np.append(Y_train, Y_incomplete, axis=0)
        
    return X_train, X_test, X_train_overall, Y_train_overall

def preprocess_standardization_with_imputed_data(X_train, Y_train, X_test, X_incomplete, Y_incomplete):
    """Standarize the input data based on training dataset and apply it on testing dataset and incomplete dataset. 
    Also concatenate the incomplete dataset with training dataset after the standarization.
        
    Parameters
    ----------   
    X_train: np.ndarray of training dataset
    X_test: np.ndarray of testing labels
    """
    
    from numpy import isnan
    from sklearn.preprocessing import StandardScaler
    import copy
    from sklearn.impute import SimpleImputer
    
    X_incomplete_features = copy.deepcopy(X_incomplete)
    
    X_train_with_imputation = np.append(X_train, X_incomplete_features, axis=0)
    Y_train_with_imputation = np.append(Y_train, Y_incomplete, axis=0)
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train_with_imputation = imp.fit_transform(X_train_with_imputation)
    
    scaler = StandardScaler()
    X_train_with_imputation = scaler.fit_transform(X_train_with_imputation)
    X_test_with_imputation = scaler.transform(X_test)
        
    return X_train_with_imputation, Y_train_with_imputation, X_test_with_imputation

def preprocess(X,y):
    """Preprocess the input data and split it into subsets for training and test sets.    
        
    Parameters
    ----------   
    X: np.ndarray of overall dataset
    y: np.ndarray of overall labels
    """
    
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # normalization
    thres = 20
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.clip(X_train, -thres, thres)
    X_test = np.clip(X_test, -thres, thres)
    
    return X_train, y_train, X_test, y_test

def plot_roc_avg(name, label, prediction, runs, **kwargs):
    plt.plot(np.linspace(0,1,11),np.linspace(0,1,11), linestyle='dashed',color='k', linewidth=1)
    from sklearn import metrics
    label= np.reshape(label,(runs,-1))
    prediction= np.reshape(prediction,(runs,-1))
    fp_all = []
    tp_all = []
    fp_base, tp_base, _ = metrics.roc_curve(label[0], prediction[0], drop_intermediate=False)
    for i in range(label.shape[0]):
        fp, tp, _ = metrics.roc_curve(label[i], prediction[i], drop_intermediate=False)
        from scipy.interpolate import UnivariateSpline
        old_indices = np.arange(0,len(fp))
        new_length = len(fp_base)
        new_indices = np.linspace(0,len(fp)-1,new_length)
        spl = UnivariateSpline(old_indices,fp,k=5,s=0)
        fp = spl(new_indices)
        old_indices = np.arange(0,len(tp))
        new_length = len(tp_base)
        new_indices = np.linspace(0,len(tp)-1,new_length)
        spl = UnivariateSpline(old_indices,tp,k=5,s=0)
        tp = spl(new_indices)
        fp_all.append(fp)
        tp_all.append(tp)
    plt.plot(np.mean(fp_all, axis=0), np.mean(tp_all, axis=0), label=name, linewidth=3, **kwargs)
    plt.fill_between(np.mean(fp_all, axis=0), np.mean(tp_all, axis=0)-np.std(tp_all, axis=0), np.mean(tp_all, axis=0)+np.std(tp_all, axis=0), alpha=0.1, **kwargs)
    plt.xlabel('FPR', fontsize=30)
    plt.ylabel('TPR', fontsize=30)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(False)
    ax = plt.gca()
    ax.set_aspect('equal')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)

def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color='black', label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
             color='black', linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.6,1])
        else:
            plt.ylim([0,1])

        plt.legend()

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))

def plot_hist(aucs, label_legend, **kwargs):
    plt.hist(aucs, label=label_legend, bins=[0.2, 0.25, 0.3, 0.35,0.4, 0.45, 0.5,0.55, 0.6, 0.65, 0.7,0.75, 0.8,0.85, 0.9,0.95,1.0], **kwargs)
    ax = plt.gca()
    ax.set_aspect('auto')
    plt.xlabel('AUC', fontsize=30)
    plt.ylabel('Count', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.ylim([0,50])
    leg = plt.legend(fontsize= '40')
    leg.get_frame().set_linewidth(0.0)
    leg.get_title().set_fontsize(40)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

def delong_test(pred1, pred2, label):
    X_A, Y_A = group_preds_by_label(pred1, label)
    X_B, Y_B = group_preds_by_label(pred2, label)
    V_A10, V_A01 = structural_components(X_A, Y_A)
    V_B10, V_B01 = structural_components(X_B, Y_B)
    auc_A = auc(X_A, Y_A)
    auc_B = auc(X_B, Y_B)
    # Compute entries of covariance matrix S (covar_AB = covar_BA)
    var_A = (get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)
         + get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
    var_B = (get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)
         + get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
    covar_AB = (get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)
            + get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))
    # Two tailed test
    z = z_score(var_A, var_B, covar_AB, auc_A, auc_B)
    p = st.norm.sf(abs(z))*2
    return z, p
    
def auc(X, Y):
    return 1/(len(X)*len(Y)) * sum([kernel(x, y) for x in X for y in Y])
def kernel(X, Y):
    return .5 if Y==X else int(Y < X)
def structural_components(X, Y):
    V10 = [1/len(Y) * sum([kernel(x, y) for y in Y]) for x in X]
    V01 = [1/len(X) * sum([kernel(x, y) for x in X]) for y in Y]
    return V10, V01
    
def get_S_entry(V_A, V_B, auc_A, auc_B):
    return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])
def z_score(var_A, var_B, covar_AB, auc_A, auc_B):
    return (auc_A - auc_B)/((var_A + var_B - 2*covar_AB)**(.5))
def group_preds_by_label(preds, actual):
    X = [p for (p, a) in zip(preds, actual) if a]
    Y = [p for (p, a) in zip(preds, actual) if not a]
    return X, Y
