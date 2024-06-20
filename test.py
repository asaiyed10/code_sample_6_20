from ioutils import *
from model import *

if __name__ == '__main__':
    print('Loading the banana quality dataset.')
    x_train, y_train, x_test, y_test = readDataPoly(degree=3)
    N_train = x_train.shape[0]
    N_test = x_test.shape[0]
    dim = x_test.shape[1]

    w, b = loadModel('model.pkl')
    y_pred = predict(w, b, x_test)

    thres = 0.5
    
    cls_pred = y_pred > thres

    print(cls_pred)
    print(y_test)
    print(cls_pred == y_test)
    print(np.sum(y_test)/N_test)

    TP = np.sum((cls_pred == y_test) & (y_test == 1))
    TN = np.sum((cls_pred == y_test) & (y_test == 0))
    FP = np.sum((cls_pred != y_test) & (y_test == 1))
    FN = np.sum((cls_pred != y_test) & (y_test == 0))
    
    print('Accuracy:', np.mean(cls_pred == y_test))
    print(f'True Positives: {TP} ({100*TP/N_test:.4f}%)')
    print(f'False Positives: {FP} ({100*FP/N_test:.4f}%)')
    print(f'True Negatives: {TN} ({100*TN/N_test:.4f}%)')
    print(f'False Negatives: {FN} ({100*FN/N_test:.4f}%)')
    print(f'F1 Score: {2*TP/(2*TP + FP + FN)}')


    # Visualization
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    pca.fit(x_train)

    grid_res = 100
    grid_range = 10

    u = pca.transform(x_test)
    U1, U2 = np.meshgrid(np.linspace(-grid_range,grid_range,grid_res), np.linspace(-grid_range,grid_range,grid_res))
    U = pca.inverse_transform(np.vstack((U1.reshape(-1,), U2.reshape(-1,))).T)
    V = predict(w, b, U)
    V = V.reshape(grid_res, grid_res)
    plt.contourf(U1, U2, V > thres, alpha=0.3, cmap='coolwarm_r')
    plt.scatter(u[:,0], u[:,1], c=y_test, cmap='coolwarm_r')
    cbar = plt.colorbar(ticks=[0,1])
    cbar.ax.set_yticklabels(['Bad', 'Good'])
    plt.show()

