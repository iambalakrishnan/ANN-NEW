import tensorflow as tf

def get_data(validation_datasize):
    
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    #create a validation dateset from the full training datasets
    #Scale the data between 0 to 1 by dividing it by 255. as its unsigned data between 0 to 255.
    X_valid, X_train = X_train_full[:validation_datasize] / 255.0, X_train_full[validation_datasize:] / 255.0
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]

    #Scale the test data as well
    X_test = X_test / 255.0
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)