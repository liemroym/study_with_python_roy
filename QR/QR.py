import pyqrcode
import pandas as pd

def createQRCode():
    df = pd.read_csv(r"D:\1. College Stuff\2. Coding Scripts\QR\data.csv", sep=';')

    for index, values in df.iterrows():
        nama = values["Nama Barang"]
        berat = values["Berat"]
        karat = values["Karat"]
        harga = values["Harga"]

        data = f'''
            Nama  = {nama}\n
            Berat = {berat}\n
            Karat = {karat}\n
            Harga = {harga}\n
        '''
        qr = pyqrcode.create(data)

        qr.png(f"{nama}.png")

createQRCode()