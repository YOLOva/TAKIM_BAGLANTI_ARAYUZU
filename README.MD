# Yarışmacılar için Örnek Sunucu Bağlantı Arayüzü
Bu repository Teknofest Ulaşımda Yapay Zeka yarışması değerlendirme sunucusuna(TUYZDS) örnek bağlantının kurulması için geliştirilmiştir.
Yarışmacılar örnek arayüzü ihtiyaçlarına göre güncelleyebilir yada farklı bir dil ile (C++, vs) benzerini implemente edebilirler.
 
 
 
 

# Kurulum
## Nvidia Toolkit Kurulumu
- Nvidia Ekran Kartı bulunan bir sistem de
Nvidia Cuda kurulmalı hangi versiyon Cuda kurulacağının kontrolü aşağıdaki linkte bulunan pythorch kurulumunda sunulan compute platformu olarak belirlenmiş cuda versiyonlarından hangisi daha sonra kurulacaksa o versiyon seçilebilir. Tercihen en ileri versiyon olabilir.
![image](https://user-images.githubusercontent.com/55704722/180610737-5aa45a1e-0c10-4986-9a9d-d19db973a7a3.png)

Versiyon seçildikten sonra örneğin 11.6 versiyonu için,
nvidia toolkit linkinde 11.6 versiyonunda bulunan 11.6.2 seçilebilir kurulabilir. Nvidia toolkit indirilirken kayıt olmanız gerekecektir kaydolun indirme linki ardından verilecektir. Nvidia toolkit kurulurken herhangi herhangi bir özel kuruluma ihtiyaç yoktur.
pythorch linki: https://pytorch.org/get-started/locally/
nvidia toolkit linki: https://developer.nvidia.com/cuda-toolkit-archive

Aşağıdaki kodu conda prompt üzerinde çalıştırmanız gerekmektedir. 
- Sistem için bir sanal ortam oluşturulur ve sanal ortamı aktifleştirmektedir. (Anaconda'nın kurulu olduğu varsayılmaktadır.) 
```shell
$ conda create -n teknofest_yarisma python=3.7
$ conda activate teknofest_yarisma
```
- Bundan sonraki kısımlarda bu sanal ortam kısmının kesinlikle aktif olması gerekmektedir!!! Active değilse yine yukarıdaki kodu çalıştırın.

## Pythorch kurulumu
Sistemde eğer daha önce pythorch kuruluysa kaldırmak için aşağıdaki kodu çalıştırın, bu durumda değilseniz dahi çalıştırın!!! (CPU versiyon Torch kaldırmak için gerekli)
```shell
$ pip uninstall torch torchvision torchaudio
```
pythorch linki: https://pytorch.org/get-started/locally/
- pythorch linkinde aşağıdaki resimde mavi ile işaretlenmiş alanda bir pip komutu bulunacaktır. Pythorch kurulumunu başlatmak için bu kodu Conda Prompt üzerinde çalıştırmanız gerekmektedir. 

![image](https://user-images.githubusercontent.com/55704722/180611172-315fbd5d-4f2f-4619-9588-cbe82922984a.png)

## Gerekli paketlerin yüklenmesi
- Gerekli paketler yüklenir.
```shell
$ pip install -r requirements.txt
```
 
## Sistemi çalıştırmak
- Sistemi çalıştırmak için aşağıdaki komut çalıştırılır.
````shell
python main.py
````
