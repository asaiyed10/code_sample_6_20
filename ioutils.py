from sklearn.preprocessing import PolynomialFeatures

def _downloadData(csv_path):
    import urllib.request
    urllib.request.urlretrieve(
        'https://www.dropbox.com/scl/fi/pkfygi155be1lco0zvwba/banana_quality.csv?rlkey=y5f52n2j5k8gbgvbp7kgp6z52&dl=1',
        csv_path
    )

def readData():
    import pandas as pd
    import os

    csv_path = 'banana_quality.csv'

    if not os.path.exists(csv_path):
        print('Cached data not found. Downloading in progress...', end='')
        _downloadData(csv_path)
        if os.path.exists(csv_path):
            print('Success')
        else:
            print('Failed. Please try again later. If the problem persist, contact the instructor or TA.')

    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=1234).reset_index(drop=True)  # shuffle
        
    independent_vars = [var for var in df.columns if var != 'Quality']
    x = df[independent_vars].to_numpy()
    y = (df['Quality'] == 'Good').to_numpy()
    
    # train-test split
    N = len(y)
    N_train = int(N*0.7)

    x_train = x[:N_train,:]
    y_train = y[:N_train]
    x_test = x[N_train:,:]
    y_test = y[N_train:]

    return x_train, y_train, x_test, y_test

def readDataPoly(degree=2):
    import pandas as pd
    import os

    csv_path = 'banana_quality.csv'

    if not os.path.exists(csv_path):
        print('Cached data not found. Downloading in progress...', end='')
        _downloadData(csv_path)
        if os.path.exists(csv_path):
            print('Success')
        else:
            print('Failed. Please try again later. If the problem persist, contact the instructor or TA.')

    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=1234).reset_index(drop=True)  # shuffle
        
    independent_vars = [var for var in df.columns if var != 'Quality']
    x = df[independent_vars].to_numpy()
    y = (df['Quality'] == 'Good').to_numpy()

    poly = PolynomialFeatures(degree=degree, interaction_only=True)
    x = poly.fit_transform(x)
    
    # train-test split
    N = len(y)
    N_train = int(N*0.7)

    x_train = x[:N_train,:]
    y_train = y[:N_train]
    x_test = x[N_train:,:]
    y_test = y[N_train:]

    return x_train, y_train, x_test, y_test