
from threading import Timer
from tkinter import PhotoImage
import cv2
from PIL import ImageTk, Image
from numpy import asarray


class CanvasButton:

    def __init__(self, canvas, x, y, image_path, command):
        self.hovered=False
        self.active=True
        self.canvas=canvas
        # Convert window to canvas coords.
        x, y = canvas.canvasx(x), canvas.canvasy(y)
        hover_sub=15
        clicked_sub=30
        disabled_plus=0

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        (B, G, R,A) = cv2.split(img)
        img=cv2.merge([R, G,B,A])
        image = Image.fromarray(img)
        self.btn_image = ImageTk.PhotoImage(image=image)
        img=cv2.merge([R-hover_sub, G-hover_sub,B-hover_sub,A])
        image = Image.fromarray(img)
        self.btn_image_hovered = ImageTk.PhotoImage(image=image)
        img=cv2.merge([R-clicked_sub, G-clicked_sub,B-clicked_sub,A])
        image = Image.fromarray(img)
        self.pressed_btn_image = ImageTk.PhotoImage(image=image)
        img=cv2.merge([R+disabled_plus, G+disabled_plus,B+disabled_plus,A])
        image = Image.fromarray(img)
        self.disabled_btn_image = ImageTk.PhotoImage(image=image)

        self.canvas_btn_img_obj = canvas.create_image(
            x, y, anchor='nw', image=self.btn_image)

        canvas.tag_bind(self.canvas_btn_img_obj, "<Button-1>",
                        lambda event: self.on_click(command=command))
        canvas.tag_bind(self.canvas_btn_img_obj,'<Enter>', lambda event: self.set_hover(True))
        canvas.tag_bind(self.canvas_btn_img_obj,'<Leave>', lambda event: self.set_hover(False))
    def activate(self, value):
        self.active=value
        if not value:
            self.canvas.itemconfig(self.canvas_btn_img_obj,image=self.disabled_btn_image)
            return
        img=self.btn_image if not self.hovered else self.btn_image_hovered
        self.canvas.itemconfig(self.canvas_btn_img_obj,image=img)

    def on_click(self, command):
        if not self.active:
            return
        self.setSelected(True)
        command()
        r = Timer(0.5, self.setSelected, [False])
        r.start()

    def set_hover(self, value):
        if not self.active:
            return
        self.hovered=value
        btnimage = self.btn_image_hovered if value else self.btn_image
        self.canvas.itemconfig(self.canvas_btn_img_obj,image=btnimage)


    def setSelected(self, value):
        btnimage = self.pressed_btn_image if value else self.btn_image if not self.hovered else self.btn_image_hovered
        self.canvas.itemconfig(self.canvas_btn_img_obj,image=btnimage)
