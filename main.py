def main():
    train_data, test_data = load_data('mnist_train.csv', 'mnist_test.csv')
    
    train_np = convert_to_numpy(train_data)
    test_np = convert_to_numpy(test_data)
    
    X_train, y_train = split_features_labels(train_np)
    X_test, y_test = split_features_labels(test_np)
    
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
    
 
    input_size = 784  # 28x28 pixels
    hidden_size = 10  
    output_size = 10  
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    
    train_acc_history, test_acc_history = nn.train(
        X_train, y_train, X_test, y_test,
        learning_rate=0.1, epochs=1000
    )
    
    plt.figure(figsize=(10, 6))
    epochs_to_plot = range(0, len(train_acc_history) * 100, 100)
    plt.plot(epochs_to_plot, train_acc_history, label='Training Accuracy')
    plt.plot(epochs_to_plot, test_acc_history, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Performance')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
