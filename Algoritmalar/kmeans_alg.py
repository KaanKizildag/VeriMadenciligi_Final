from sklearn.cluster import KMeans
import Veri.veri_isleme

class KmeansAlg():
    # Biliyoruz ki veri setimizde 2 küme var
    kararAlgoritmasi = KMeans(n_clusters=2)

    def __init__(self):
        isleme = Veri.veri_isleme.OnIsleme()
        x_train, x_test, y_train, y_test = isleme.seti_bol()
        self.kararAlgoritmasi.fit(x_train, y_train)

    def tahmin(self, x):
        return self.kararAlgoritmasi.predict(x)

