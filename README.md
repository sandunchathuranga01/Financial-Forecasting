This project involves building and evaluating Multilayer Perceptrons (MLPs), a type of neural network, to predict USD/EUR exchange rates. The process starts by loading exchange rate data from an Excel file and generating lagged features to use past values as predictors. Features are normalized using min-max scaling to equalize their influence during model training. Four different MLP configurations are tested: the first model has one hidden layer with five neurons; the second has two hidden layers with five and three neurons; the third has three hidden layers with five, three, and two neurons; and the fourth model escalates complexity with four hidden layers of five, four, three, and two neurons. Each model is trained on a training dataset, and their structures are visualized to understand the network complexity. Predictions are made on a test dataset, and the models are evaluated using metrics such as RMSE, MAE, MAPE, and SMAPE to determine their accuracy in forecasting exchange rates. Additionally, plots comparing actual data with predictions are generated for each model to visually assess prediction accuracy. This comprehensive exercise not only demonstrates the application of neural networks to financial time-series forecasting but also explores how different network architectures can impact predictive performance.![3a505bc1-833b-4778-80d5-10d0a2da1c7b](https://github.com/sandunchathuranga01/Financial-Forecasting/assets/123801670/f5b00553-fcb1-4c79-b468-d93dba4a7577)