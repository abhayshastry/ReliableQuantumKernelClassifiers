import numpy as np
#Copied from Hubregtsen qhack github. But modified to take 1 input parameter: num_samples
#We use test_train_split to split the data into test and train sets in the workscript

def symmetric_donuts(num_samples):
    """generate data in two circles, with flipped label regions
    Args:
        num_samples (int): Number of datapoints
        num_test (int): Number of test datapoints
    Returns:
        X (ndarray): datapoints
        y (ndarray): labels
    """
    # the radii are chosen so that data is balanced
    inv_sqrt2 = 1/np.sqrt(2)

    X = []
    X_test = []
    y = []
    y_test = []

    # Generate the training dataset
    x_donut = 1
    i = 0
    while (i<num_samples):
        x = np.random.uniform(-inv_sqrt2,inv_sqrt2, 2)
        r_squared = np.linalg.norm(x, 2)**2
        if r_squared < 0.5:
            i += 1
            X.append([x_donut+x[0],x[1]])
            if r_squared < .25:
                y.append(x_donut)
            else:
                y.append(-x_donut)
            # Move over to second donut
            if i==num_samples//2:
                x_donut = -1

    return np.array(X), np.array(y)

def checkerboard(num_samples, num_grid_col=4, num_grid_row=4):
    if num_samples%2:
        raise ValueError(f"This method wants to create a balanced dataset but received"
                f"odd num_train={num_train}.")
    max_samples = num_grid_row * num_grid_col * 40
    if num_samples>max_samples:
        raise ValueError(f"Due to intricate legacy reasons, the number of samples"
                f"may not exceed {max_samples}. Received {num_total}.")
    # creating negative (-1) and positive (+1) samples
    negatives = []
    positives = []
    for i in range(num_grid_col):
        for j in range(num_grid_row):
            data = (np.random.random((40,2))-0.5)#They generate 40 datapoints per grid unit
            data[:,0] = (data[:,0]+2*i+1)/(2*num_grid_col)
            data[:,1] = (data[:,1]+2*j+1)/(2*num_grid_row)
            if i%2==j%2:
                negatives.append(data)
            else:
                positives.append(data)
    negative = np.vstack(negatives)
    positive = np.vstack(positives)

    # shuffle the data
    np.random.shuffle(negative)
    np.random.shuffle(positive)
    #Store first num_samples points in X and and labels y
    X = np.vstack([negative[:num_samples], positive[:num_samples]])
    y = np.hstack([-np.ones((num_samples)), np.ones((num_samples))])
#    X_test = np.vstack([negative[num_train//2:num_total//2], positive[num_train//2:num_total//2]])
#    y_test = np.hstack([-np.ones((num_test//2)), np.ones((num_test//2))])

    return X, y
