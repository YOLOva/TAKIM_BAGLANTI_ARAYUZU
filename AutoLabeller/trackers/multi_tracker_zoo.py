from ..trackers.strong_sort.utils.parser import get_config
from ..trackers.strong_sort.strong_sort import StrongSORT
from ..trackers.ocsort.ocsort import OCSort
#from trackers.bytetrack.byte_tracker import BYTETracker
from ..utils.params_saver import ParamsSaver


def create_tracker(tracker_type, appearance_descriptor_weights, device, half):
    if tracker_type == 'strongsort':
        params_saver=ParamsSaver()
        data = params_saver.get_data()
        strongsort = StrongSORT(
            appearance_descriptor_weights,
            device,
            half,
            max_dist=data["tracker"]["strongsort"]["max_dist"],
            max_iou_distance=data["tracker"]["strongsort"]["max_iou_distance"],
            max_age=data["tracker"]["strongsort"]["max_age"],
            max_unmatched_preds=data["tracker"]["strongsort"]["max_unmatched_preds"],
            n_init=data["tracker"]["strongsort"]["n_init"],
            nn_budget=data["tracker"]["strongsort"]["nn_budget"],
            mc_lambda=data["tracker"]["strongsort"]["mc_lambda"],
            ema_alpha=data["tracker"]["strongsort"]["ema_alpha"],

        )
        return strongsort
    elif tracker_type == 'ocsort':
        ocsort = OCSort(
            det_thresh=0.45,
            iou_threshold=0.2,
            use_byte=False 
        )
        return ocsort
    else:
        print('No such tracker')
        exit()