from sklearn.model_selection import train_test_split

from Veri import reader


class OnIsleme():

    def __init__(self):
        self.data = reader.Reader().getData()
        # Verinin başlıklarını yazıyorum.
        self.data.columns = [
                'id',
                'Clump Thickness',
                'Uniformity of Cell Size',
                'Uniformity of Cell Shape',
                'Marginal Adhesion',
                'Single Epithelial Cell Size',
                'Bare Nuclei',
                'Bland Chromatin',
                'Normal Nucleoli',
                'Mitoses',
                'Class'
            ]
        # Boş verilerin bulunduğu satırları kaldırıyorum.
        self.data.dropna(inplace=True, axis = 0)

        # Eğitimde Id kolonuna ihtiyacım olmayacak.
        self.data.drop(columns=['id'], inplace=True)

        # Sınıf isimlerini yönelgeye göre değiştiriyorum.
        self.data['Class'] = \
            self.data['Class'].replace({2 : 'benign', 4 : 'malignant'})

    def seti_bol(self):
        X = self.data.drop(columns=['Class']).values
        Y = self.data['Class'].values
        return train_test_split(X,Y, random_state=42)