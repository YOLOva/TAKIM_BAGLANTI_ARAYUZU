def overlap_area(box1, box2):
    """
    İki sınırlayıcı kutunun örtüşen alanını hesaplar.
    box1: Birinci sınırlayıcı kutu koordinatları ve boyutları (left, top, width, height)
    box2: İkinci sınırlayıcı kutu koordinatları ve boyutları (left, top, width, height)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    # sınırlayıcı kutuların sol ve sağ koordinatlarını hesapla
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    # sınırlayıcı kutuların üst ve alt koordinatlarını hesapla
    top = max(y1, y2)
    bottom = min(y1 + h1, y2 + h2)
    # iki kutunun örtüşme alanını hesapla
    overlap = max(0, right - left) * max(0, bottom - top)
    return overlap

def is_inside(box1, box2, threshold):
    """
    Birinci sınırlayıcı kutunun ikinci sınırlayıcı kutunun içinde olup olmadığını kontrol eder.
    box1: Birinci sınırlayıcı kutu koordinatları ve boyutları (left, top, width, height)
    box2: İkinci sınırlayıcı kutu koordinatları ve boyutları (left, top, width, height)
    """
    overlap = overlap_area(box1, box2)
    # Birinci kutunun alanını hesapla
    box1_area = box1[2] * box1[3]
    current_threshold=overlap / box1_area
    # Birinci kutunun, ikinci kutunun %75'ten fazlasında bulunup bulunmadığını kontrol et
    if current_threshold>0:
        print(current_threshold)
    if current_threshold > threshold:
        return True
    else:
        return False
        

""" box1 = (100, 100, 50, 50)   # Birinci sınırlayıcı kutu (left, top, width, height)
box2 = (90, 90, 60, 60)   # İkinci sınırlayıcı kutu (left, top, width, height)

if is_inside(box1, box2):
    print("Birinci kutu ikinci kutunun içinde.")
else:
    print("Birinci kutu ikinci kutunun içinde değil.") """