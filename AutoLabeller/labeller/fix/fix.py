from ...utils.param_classes.auto_labeller import AutoLabellerParams


class AllClassesFixs:
    params:AutoLabellerParams=None,
    frame=None
    def __init__(self, params, frame) -> None:
        self.params=params
        self.frame=frame
    def multiple_box_in_box_fix(self, cocos:list, same_class=True, same_size=True):
        if len(cocos)==0:
            return cocos
        remove_ids=[]
        for coco in cocos:
            if coco["id"] in remove_ids:
                continue
            for coco2 in cocos:
                if same_class and coco2["category_id"]!=coco["category_id"]:
                    continue
                if coco is not coco2 and self.coco_in_and_same_size_coco(coco2, coco, same_size):
                    if coco2["id"] not in remove_ids:
                        remove_ids.append(coco2["id"])
        return list(filter(lambda x: x["id"] not in remove_ids, cocos))
    
    def coco_in_and_same_size_coco(self, cocoin, coco, check_same_size=True):
        fix_ratio = self.params.fixs.same_size_class_box_in_box_fix_ratio
        if cocoin["bbox"][2] == 0 or cocoin["bbox"][3]==0 or coco["bbox"][2]==0 or coco["bbox"][3]==0:
            return False
        
        ratiox = cocoin["bbox"][2]*cocoin["bbox"][3]/coco["bbox"][2]*coco["bbox"][3]
        
        cx = cocoin["bbox"][2]/2+cocoin["bbox"][0]
        cy = cocoin["bbox"][3]/2+cocoin["bbox"][1]
        if (coco["bbox"][0] <= cx <= coco["bbox"][0]+coco["bbox"][2]
                            and coco["bbox"][1] <= cy <= coco["bbox"][1]+cocoin["bbox"][3]):
            ratio_pass=(not check_same_size or ratiox >= fix_ratio)
            if ratio_pass:
                return True
        return False
    
    def negative_value_fix(self, cocos):
        filtered_cocos = list(filter(lambda x: len(
            list(filter(lambda b: b < 0, x["bbox"]))) == 0, cocos))
        if len(cocos) > len(filtered_cocos):
            print("negative value detected")
        return filtered_cocos