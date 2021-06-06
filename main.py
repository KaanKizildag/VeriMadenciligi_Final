from Algoritmalar.kmeans_alg import KmeansAlg
from Algoritmalar.knn_alg import KnnAlg
from Algoritmalar.decision_tree_alg import DecisonTreeAlg
from Algoritmalar.naive_bayes_alg import NaiveBayes
from Algoritmalar.Ysa_alg import YsaAlg
from Algoritmalar.svm_alg import SvmAlg
from Degerlendirme.degerlendir import SoruCevaplayan
from Veri.veri_isleme import OnIsleme

kararAlgoritmasi = DecisonTreeAlg()

isleme = OnIsleme()
x_train, x_test, y_train, y_test = isleme.seti_bol()

tahminler = kararAlgoritmasi.tahmin(x = x_test)

sc = SoruCevaplayan(gercekDegerler = y_test, tahminler = tahminler)

print('confusion matris: \n', sc.confusionMatris())
print('tp', sc.get_tp())
print('tn', sc.get_tn())
print('fp', sc.get_fp())
print('fn', sc.get_fn())
print('dogruluk: ', sc.get_dogruluk())
print('recall: ', sc.get_recall())
print('precision: ', sc.get_precision())
