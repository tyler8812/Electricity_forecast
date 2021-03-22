# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p", "--predict", action="store_true", help="predict electricity")  
    group.add_argument("-t", "--training", action="store_true", help="training model")
     
    parser.add_argument('-ip', '--inputPredict', help='input predicting data file name')
    parser.add_argument('-im', '--inputModel', help='input model file name')
    parser.add_argument('-it', '--inputTraining', help='input training data file name')
    parser.add_argument('-o', '--output', help='output file name')
 
    args = parser.parse_args()
  
    if args.training and args.inputTraining and args.output:
        print("training model...\n")
        import math
        import os
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        from keras.models import Sequential
        from keras.layers import Dense, LSTM, Dropout
        import matplotlib.pyplot as plt
        import matplotlib
        import datetime

        print("Loading " + args.inputTraining + " file...\n")
        # get the data
        df_training = pd.read_csv(args.inputTraining, usecols=[0, 1])
        data = df_training.filter(['value'])
        # convert the dataframe to a numy array
        dataset = data.values

        training_data_len = math.ceil(len(dataset) * 0.9)
         # scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        n = 365
        # create training data
        # create the scaled training data
        train_data = scaled_data[0:training_data_len, :]
        # split the data into x_train and y_train data set
        x_train = []
        y_train = []
        for i in range(n, len(train_data)-7):
            x_train.append(train_data[i-n:i, 0])
            y_train.append(train_data[i: i+8, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0],n,1))


        # build LSTM model
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1],1)))
        model.add(Dropout(0.5))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(50))
        model.add(Dense(25))
        model.add(Dense(8))
        print("Model: \n")
        print(model.summary())
        print()
        # compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        print("Training... \n")
        # train model
        model.fit(x_train, y_train, batch_size=16, epochs=20)
        print("Saving model to " + args.output + " \n")
        if os.path.isfile(args.output): 
            os.remove(args.output)
        model.save(args.output)
        print("Finish\n")

    elif args.predict and args.inputPredict and args.inputModel:
        print("predicting...\n")
        import numpy as np
        import os
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        from keras.models import load_model
        import datetime

        n = 365
        print("Loading " + args.inputPredict + " file...\n")
        df_training = pd.read_csv(args.inputPredict, usecols=[0, 1])
        data = df_training.filter(['value'])
        # convert the dataframe to a numy array
        dataset = data.values
        # scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        print("Loading " + args.inputModel + " model...\n")
        model = load_model(args.inputModel)
        

        out_for_predict = df_training.copy()
        temp_time = out_for_predict['date'].iloc[-1]
        new_df = out_for_predict.filter(['value'])
        last_days = new_df[-n:].values
        last_days_scaled = scaler.transform(last_days)

        X_test = []
        X_test.append(last_days_scaled)
        X_test=np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred = model.predict(X_test)
        pred = scaler.inverse_transform(pred)
        df = pd.DataFrame()
        
        for i in range(1, 8):
            adding_time = (datetime.datetime.strptime(temp_time, "%Y/%m/%d") + datetime.timedelta(days=i+1)).strftime("%Y%m%d")
            temp = pd.DataFrame([[adding_time, int(pred[0][i])]], columns=(['date', 'operating_reserve(MW)']))
            # print(temp)
            df = df.append(temp)

        print("writing result to " + args.output + " file...\n")
        if os.path.isfile(args.output): 
            os.remove(args.output)
        df.to_csv(args.output, index=0)
        print("Finish\n")

        test = [3070, 3260, 3160, 3200, 2840, 3090, 3050]
        dif = 0
        # testing
        dif = np.sqrt(np.mean((test - df['operating_reserve(MW)']) ** 2))
        print(dif)

    else:
        print("Something goes wrong, please read README.md.")
