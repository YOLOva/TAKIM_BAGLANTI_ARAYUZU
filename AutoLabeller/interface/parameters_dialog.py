
import json
import os
from pathlib import Path
from tkinter import BooleanVar, Button, Checkbutton, DoubleVar, Entry, Frame, IntVar, Label, LabelFrame, StringVar, Variable, simpledialog
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Combobox, Notebook

from ..interface.formats import TXT_Format, YOLO_Model_Format
from ..utils.param_classes.auto_labeller import AutoLabellerParams
from ..utils.param_classes.model_params.model import ModelParams
from ..utils.params_saver import ParamsSaver


class ParemetersDialog(simpledialog.Dialog):
    def __init__(self, parent, title) -> None:
        self.paramsSaver = ParamsSaver()
        self.params = self.paramsSaver.getParams()
        super().__init__(parent, title)

    def body(self, master):
        self.vcmd = (master.register(self.validateFloatBetween),
                     '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W', 0, 1)
        self.vcmd2 = (master.register(self.validateType),
                      '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        tabControl = Notebook(master)
        tab1 = Frame(tabControl)
        tab2 = Frame(tabControl)
        tab3 = Frame(tabControl)
        tab4 = Frame(tabControl)
        tabControl.pack(expand=1, fill="both")

        tabControl.add(tab1, text='Genel')
        tabControl.add(tab2, text='Kaydetme')
        tabControl.add(tab3, text='Tracker')
        tabControl.add(tab4, text='Düzeltmeler')
        self.general_group(tab1)
        self.save_group(tab2)
        self.tracker_group(tab3)
        self.fix_groups(tab4)

        return tab1  # initial focus

    def add_model_part(self, model_tabControl: Notebook, modelParams: ModelParams):
        tabm = Frame(model_tabControl)
        model_tabControl.add(tabm, text=modelParams.name.get())
        tabControl = Notebook(tabm)
        tab1 = Frame(tabControl)
        tab2 = Frame(tabControl)
        tab3 = Frame(tabControl)
        tabControl.add(tab1, text='Genel')
        tabControl.add(tab2, text='SAHI')
        tabControl.add(tab3, text='Postprocess')
        
        def removeModelTab():
            if len(model_tabControl.children)>1:
                model_tabControl.forget(model_tabControl.select())
                tabm.destroy()
                self.params.removeModel(modelParams.id)
        
        self.model_general_group(tab1, modelParams,removeModelTab)
        self.sahi_group(tab2, modelParams)
        self.postprocess_group(tab3, modelParams)

        tabControl.pack(expand=1, fill="both")

    def postprocess_group(self, master, modelParams: ModelParams):
        post_process_list = ["GREEDYNMM", "NMM", "NMS", "LSNMS"]
        postprocess_match_metric_list = ["IOS", "IOU"]
        postprocess = modelParams.postprocess
        self.combobox(master, postprocess.postprocess_type,
                      post_process_list, "Post Process Type:", 0, 0)
        self.combobox(master, postprocess.match_metric,
                      postprocess_match_metric_list, "Post Process Match Metric:", 1, 0)
        self.entry(master, postprocess.match_threshold,
                   "Post Process Match Threshold:",  self.vcmd, 2, 0)
        Checkbutton(master, text='Post Process Class Agnotstic', variable=postprocess.class_agnostic,
                    onvalue=True, offvalue=False, anchor="w").grid(row=3, sticky="W")

    def fix_groups(self, master):
        fixs = self.params.fixs
        Checkbutton(master, text='Negatif bbox değerine sahip nesneleri temizle:',
                    variable=fixs.negative_bbox_values_fix, onvalue=True, offvalue=False, anchor="w").grid(row=0, sticky="W")
        frame = LabelFrame(master, text="İç içe aynı sınıf düzeltmesi")

        frame.grid(row=1, column=0, columnspan=3, sticky="NW")
        Checkbutton(frame, text='Aynı sınıfa ait iç içe cisimleri düzelt', variable=fixs.same_size_class_box_in_box_fix,
                    onvalue=True, offvalue=False, anchor="w").grid(row=0, column=0, sticky="W")
        self.entry(frame, fixs.same_size_class_box_in_box_fix_ratio,
                   "En ve Boy oranı:",  self.vcmd, 1, 0)

        Checkbutton(master, text='UYZ 2022 Modeli Düzeltmelerini Uygula', variable=fixs.enable_uyz2022_fix,
                    onvalue=True, offvalue=False, anchor="w").grid(row=2, column=0, sticky="W")

        frame = LabelFrame(master, text="UYZ2022 model düzeltmeleri")
        frame.grid(row=11, column=0, columnspan=3, sticky="NW")
        Checkbutton(frame, text='İniş Alanları Uygunluk Tespiti', variable=fixs.uyz2022.uaips_state_fix,
                    onvalue=True, offvalue=False, anchor="w").grid(row=0, column=0, sticky="W")
        Checkbutton(frame, text='Araç ile aynı boyutta araç üzerindeki insanları kaldır',
                    variable=fixs.uyz2022.person_same_size_in_car_fix, onvalue=True, offvalue=False, anchor="w").grid(row=1, column=0, sticky="W")

    def model_general_group(self, master, modelParams: ModelParams,removeModelTab):
        self.entry(master, modelParams.name, "Ad:",  self.vcmd, 0, 0)
        Checkbutton(master, text='Sahi', variable=modelParams.use_sahi,
                    onvalue=True, offvalue=False, anchor="w").grid(row=1, sticky="W")
        self.entry(master, modelParams.conf, "confidence:",  self.vcmd, 2, 0)
        self.file_selector_entry(
            master, modelParams.model, "model dizini:", YOLO_Model_Format, 3, 0)

        verbose_list = [0, 1, 2]
        self.combobox(master, modelParams.verbose,
                      verbose_list, "Verbose:", 4, 0)

        Button(master, text="Modeli Sil",
               command=lambda: removeModelTab()).grid(row=9)
        def move(left):
            pass
        Button(master, text="<",
               command=lambda: move(True)).grid(row=9, column=3)
        Button(master, text=">",
               command=lambda: move(False)).grid(row=9, column=4)

    def general_group(self, master):
        Checkbutton(master, text='Veri seti olarak kaydet', variable=self.params.enable_save,
                    onvalue=True, offvalue=False, anchor="w").grid(row=0, column=0, sticky="W")
        self.entry(master, self.params.imgsz, "imgsz:",  self.vcmd2, 1, 0)
        clist = ["cuda:0", "cpu"]
        self.combobox(master, self.params.device, clist, "device:", 2, 0)
        Checkbutton(master, text='Resmi Boyutlandır', variable=self.params.resize_img,
                    onvalue=True, offvalue=False, anchor="w").grid(row=3, sticky="W")
        frame = LabelFrame(master, text="Boyutlandırma")
        frame.grid(row=4, columnspan=4, sticky="W")
        self.entry(frame, self.params.resize_width,
                   "Boyut x:",  self.vcmd2, 0, 0)
        self.entry(frame, self.params.resize_height, "y:",  self.vcmd2, 0, 2)
        Checkbutton(master, text='Tracker', variable=self.params.use_tracker,
                    onvalue=True, offvalue=False, anchor="w").grid(row=5, sticky="W")

        self.file_selector_entry(
            master, self.params.classes_txt, "classes.txt dizini:", TXT_Format, 6, 0)
        Checkbutton(master, text='Label Mapping işlemi uygula', variable=self.params.enable_label_map,
                    onvalue=True, offvalue=False, anchor="w").grid(row=7, column=0, sticky="W")
        self.file_selector_entry(
            master, self.params.label_map_file, "label map dosyası dizini:", TXT_Format, 8, 0)
        Button(master, text="Varsayılanlara Döndür",
               command=lambda: self.restore_defaults()).grid(row=11, column=1)

        frame = LabelFrame(master, text="Modeller ve Ayarları")
        frame.grid(row=10, column=0, columnspan=3, sticky="W")

        model_tabControl = Notebook(frame)

        def add_model_tab():
            file_path = self.select_file(
                self.params.modelsParams[-1].model.get(), YOLO_Model_Format)
            if (file_path == ""):
                return
            self.add_model_part(
                model_tabControl, self.params.addNewModel(file_path))
        Button(master, text="Model Ekle",
               command=lambda: add_model_tab()).grid(row=9)

        for i, modelParams in enumerate(self.params.modelsParams):
            self.add_model_part(model_tabControl, modelParams)
        model_tabControl.pack(expand=1, fill="both")

    def save_group(self, master):
        Checkbutton(master, text='Video kaydederken frame azalt', variable=self.params.save.decrease_video_frame,
                    onvalue=True, offvalue=False, anchor="w").grid(row=0, sticky="W")
        self.entry(master, self.params.save.video_fps_to_save,
                   "Saniye başına kaydedilecek frame:",  self.vcmd2, 1, 0)
        Checkbutton(master, text='Kaydedilmeyecek Frame\'i atla', variable=self.params.save.pass_detection_not_saved_frames,
                    onvalue=True, offvalue=False, anchor="w").grid(row=2, sticky="W")

    def sahi_group(self, master, modelParams: ModelParams):
        i = 1
        sahi = modelParams.sahi

        self.slice_height_entry = self.entry(
            master, sahi.slice_height, "Slice Height:",  self.vcmd2, i+0, 0)
        self.slice_width_entry = self.entry(
            master, sahi.slice_width, "Slice Width:",  self.vcmd2, i+1, 0)

        def enable_slice_entrys(value):
            state = "normal" if value else "readonly"
            self.slice_height_entry.config(state=state)
            self.slice_width_entry.config(state=state)

        Checkbutton(master, text='Auto Slice Resolution', variable=sahi.auto_slice_resolution, onvalue=True, offvalue=False,
                    anchor="w", command=lambda: enable_slice_entrys(not sahi.auto_slice_resolution.get())).grid(row=i-1, sticky="W")
        self.entry(master, sahi.overlap_width_ratio,
                   "Overlap Width Ratio:",  self.vcmd, i+2, 0)
        self.entry(master, sahi.overlap_height_ratio,
                   "Overlap Height Ratio:",  self.vcmd, i+3, 0)
        enable_slice_entrys(not sahi.auto_slice_resolution.get())
        Checkbutton(master, text='perform_standard_pred', variable=sahi.perform_standard_pred,
                    onvalue=True, offvalue=False, anchor="w").grid(row=i+4, sticky="W")
        # self.enable_slice_entrys(not sahi.auto_slice_resolution.get())

    def tracker_group(self, master):
        self.entry(master, self.params.tracker.min_correlation,
                   "Min Correlation:",  self.vcmd, 0, 0)
        Checkbutton(master, text='Half:', variable=self.params.tracker.half,
                    onvalue=True, offvalue=False, anchor="w").grid(row=1, sticky="W")
        tracking_methods = ["strongsort", "ocsort", "bytetrack"]
        self.combobox(master, self.params.tracker.tracking_method,
                      tracking_methods, "Tracking Method:", 2, 0)
        self.file_selector_entry(master, self.params.tracker.reid_weights,
                                 "reid_weights dizini:", YOLO_Model_Format, 3, 0)
        frame = LabelFrame(master, text="StrongSort")
        frame.grid(row=4, columnspan=3)
        self.strong_sort_group(frame)

    def strong_sort_group(self, master):
        self.entry(master, self.params.tracker.strongsort.mc_lambda,
                   "mc lambda:",  self.vcmd, 0, 0)
        self.entry(master, self.params.tracker.strongsort.ema_alpha,
                   "ema alpha:",  self.vcmd, 1, 0)
        self.entry(master, self.params.tracker.strongsort.max_dist,
                   "max dist:",  self.vcmd, 2, 0)
        self.entry(master, self.params.tracker.strongsort.max_iou_distance,
                   "max iou distance:",  self.vcmd, 3, 0)
        self.entry(master, self.params.tracker.strongsort.max_unmatched_preds,
                   "max unmatched preds:",  self.vcmd2, 4, 0)
        self.entry(master, self.params.tracker.strongsort.max_age,
                   "max age:",  self.vcmd2, 5, 0)
        self.entry(master, self.params.tracker.strongsort.n_init,
                   "n init:",  self.vcmd2, 6, 0)
        self.entry(master, self.params.tracker.strongsort.nn_budget,
                   "nn budget:",  self.vcmd2, 7, 0)

    def entry(self, master, variable: Variable, label, vcmd, row, col):
        Label(master, text=label, anchor="w").grid(
            row=row, column=col, sticky="W")
        entry = Entry(master, textvariable=variable,
                      validate='key', validatecommand=vcmd)
        entry.grid(row=row, column=col+1, sticky="W")
        return entry

    def combobox(self, master, variable: Variable, list, label, row, col):
        Label(master, text=label, anchor="w").grid(
            row=row, column=col, sticky="W")
        Combobox(master, textvariable=variable, values=list,
                 state="readonly").grid(row=row, column=col+1)

    def select_file(self, init_file, file_types):
        """ ROOT_DIR = str(Path(os.path.dirname(
            os.path.abspath(__file__))).parent).replace("\\", "/")
        print(ROOT_DIR) """
        file_path = askopenfilename(filetypes=file_types, initialdir=Path(
            init_file).parent, initialfile=Path(init_file))

        """ def try_localize():
            if ROOT_DIR in file_path:
                return file_path.replace(ROOT_DIR+"/", "")
            return file_path
        file_path = try_localize()
        print(file_path) """
        return file_path

    def file_selector_entry(self, master, variable: StringVar, label, file_types, row, col):
        def select_file_to_entry():
            file_path = self.select_file(variable.get(), file_types)
            if file_path != "":
                variable.set(file_path)
        Label(master, text=label, anchor="w").grid(
            row=row, column=col, sticky="W")
        Entry(master, textvariable=variable, state="readonly").grid(
            row=row, column=col+1, sticky="W")
        Button(master, text="Seç", command=lambda: select_file_to_entry()).grid(
            row=row, column=col+2, sticky="W")

    # TODO Düzeltilecek
    def restore_defaults(self):
        self.params = self.paramsSaver.getDefaultParams()
        self.load_variables()

    def validateType(self, action, index, value_if_allowed,
                     prior_value, text, validation_type, trigger_type, widget_name, empty_return_val=True, valtype: type = int):
        if value_if_allowed:
            try:
                valtype(value_if_allowed)
                return True
            except ValueError:
                return False
        else:
            return empty_return_val

    def validateFloatBetween(self, action, index, value_if_allowed,
                             prior_value, text, validation_type, trigger_type, widget_name, start, end):
        if self.validateType(action, index, value_if_allowed,
                             prior_value, text, validation_type, trigger_type, widget_name, False, float):
            return float(start) <= float(value_if_allowed) <= float(end)
        return len(value_if_allowed) == 0

    def apply(self):
        self.paramsSaver.saveParams(self.params)
