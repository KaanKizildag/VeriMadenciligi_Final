from sklearn import svm

from Veri import veri_isleme

class SvmAlg():
    kararAlgoritmasi = svm.SVC()

    def __init__(self):
        isleme = veri_isleme.OnIsleme()
        x_train, x_test, y_train, y_test = isleme.seti_bol()
        self.kararAlgoritmasi.fit(x_train, y_train)

    def tahmin(self, x):
        return self.kararAlgoritmasi.predict(x)