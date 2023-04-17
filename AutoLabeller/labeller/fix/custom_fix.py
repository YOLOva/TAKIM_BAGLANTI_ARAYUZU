class CustomFix:
    def insan_motor_smaller_than_other_classes(cocos):
        def get_all_center_objects(cocos):
            return list(filter(lambda x: x["bbox"],cocos))
        return cocos
    def fix(cocos):
        return cocos