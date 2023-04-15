import statistics


class AracInsanFix:
    def fix(self, cocos):
        fix_ratio = 0.67
        fix_ratio_not_human = 0.8
        araclar = [coco for coco in cocos if coco["category_id"] == 0]
        insanlar = [coco for coco in cocos if coco["category_id"] == 1]
        diger = [coco for coco in cocos if coco["category_id"] not in [0, 1]]
        remove_class_ids = []
        not_human = []
        if len(araclar) > 0 and len(insanlar) > 0:
            arac_mw = statistics.median([x["bbox"][2] for x in araclar])
            arac_mh = statistics.median([x["bbox"][3] for x in araclar])
            not_human = [x["id"] for x in insanlar if x["bbox"][2]*fix_ratio_not_human >=
                         arac_mw or x["bbox"][3]*fix_ratio_not_human >= arac_mh]
        for insan in insanlar:
            id = insan["id"]
            i_w = insan["bbox"][2]
            i_h = insan["bbox"][3]
            insan_bbox = list(insan["bbox"])
            insan_bbox[2] = insan_bbox[0]+insan_bbox[2]
            insan_bbox[3] = insan_bbox[1]+insan_bbox[3]
            try:
                for arac in araclar:
                    a_w = arac["bbox"][2]
                    a_h = arac["bbox"][3]
                    if a_w == 0:
                        remove_class_ids.append(arac['id'])
                        continue
                    ratiox = min(i_w/a_w, a_w/i_w)
                    ratioy = min(i_h/a_h, a_h/i_h)
                    arac_bbox = list(arac["bbox"])
                    arac_bbox[2] = arac_bbox[0]+arac_bbox[2]
                    arac_bbox[3] = arac_bbox[1]+arac_bbox[3]
                    cx = (arac_bbox[2]+arac_bbox[0])/2
                    cy = (arac_bbox[3]+arac_bbox[1])/2
                    if (insan_bbox[0] <= cx <= insan_bbox[2]
                            and insan_bbox[1] <= cy <= insan_bbox[3]) and ratiox >= fix_ratio and ratioy >= fix_ratio and id not in remove_class_ids:
                        try:
                            remove_class_ids.append(id)
                        except:
                            pass
            except:
                pass

        new_insanlar = []
        for insan in insanlar:
            if insan["id"] not in remove_class_ids:
                new_insanlar.append(insan)
            """ if insan["id"] in not_human:
                insan["category_id"]=0
                insan["category_name"]=self.names[0]
                araclar.append(insan) """
        cocos = diger+araclar+new_insanlar
        return cocos