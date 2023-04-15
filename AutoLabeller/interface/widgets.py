
from pathlib import Path
from threading import Timer
from tkinter import Canvas
import cv2
from PIL import ImageTk, Image, ImageFont, ImageDraw
import numpy as np
ASSETS_PATH = "./assets"
arial_font=r"arial-bold.ttf"
def relative_to_assets(path: str) -> Path:
    return Path(__file__).parent/ ASSETS_PATH / Path(path)



def putText(img, text, x=None, y=None, fontsize=24):
    color=(255, 255, 255, 255)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font_file=str(relative_to_assets(arial_font))
    font = ImageFont.truetype(font_file, fontsize)
    _, _, w, h = draw.textbbox((0, 0), text, font=font)
    textX = (img.shape[1] - w) / 2
    textY = (img.shape[0] - h) / 2
    org=(textX if x is None else x, textY if y is None else y)
    draw.text(org,  text, font = font, fill = color)
    img = np.array(img_pil)
    return img


class PauseButton:
    def __init__(self,  canvas, x, y, back_image=None, onValue="Duraklat", offValue="Devam", value=True) -> None:
        self.pause=value
        self.onButton= CanvasButton(canvas, x, y, back_image=back_image, text=onValue, command= lambda: self.onValue(True))
        self.offButton= CanvasButton(canvas, x, y,back_image=back_image, text=offValue, command= lambda: self.onValue(False))
        self.onValue(self.pause)
    def onValue(self, value):
        self.pause=value
        self.onButton.show(not self.pause)
        self.offButton.show( self.pause)
    
    def show(self, value):
        if value==False:
            self.onButton.show(value)
            self.offButton.show(value)
        else:
            self.onValue(self.pause)

class CanvasButton:

    def __init__(self, canvas, x, y, command,back_image=None, text=""):
       
        img_path =relative_to_assets("button_background.png" if back_image is None else back_image)
        
        self.hovered=False
        self.active=True
        self.canvas=canvas

        x, y = canvas.canvasx(x), canvas.canvasy(y)
        hover_sub=15
        clicked_sub=30
        disabled_plus=0

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img=putText(img, text)
        (B, G, R,A) = cv2.split(img)
        img=cv2.merge([R, G,B,A])
        image = Image.fromarray(img)
        self.btn_image = ImageTk.PhotoImage(image=image)

        img=cv2.merge([R-hover_sub, G-hover_sub,B-hover_sub,A])
        img=putText(img, text)
        image = Image.fromarray(img)
        putText(img, text)
        self.btn_image_hovered = ImageTk.PhotoImage(image=image)

        img=cv2.merge([R-clicked_sub, G-clicked_sub,B-clicked_sub,A])
        img=putText(img, text)
        image = Image.fromarray(img)
        putText(img, text)
        self.pressed_btn_image = ImageTk.PhotoImage(image=image)

        img=cv2.merge([R+disabled_plus, G+disabled_plus,B+disabled_plus,A])
        img=putText(img, text)
        image = Image.fromarray(img)
        putText(img, text)
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

    def show(self, value):
        if value:
            self.canvas.itemconfig(self.canvas_btn_img_obj, state='normal')
        else:
            self.canvas.itemconfig(self.canvas_btn_img_obj, state='hidden')
            
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

class CanvasCheckBox():
    def __init__(self, canvas:Canvas, x, y, initial_state, command=None, text=""):
        self.canvas=canvas
        self.state=initial_state
        x, y = canvas.canvasx(x), canvas.canvasy(y)

        img = cv2.imread(relative_to_assets("checkbox_checked.png"), cv2.IMREAD_UNCHANGED)
        img = putText(img, text, 29)
        (B, G, R,A) = cv2.split(img)
        img=cv2.merge([R, G,B,A])
        image = Image.fromarray(img)
        self.checked = ImageTk.PhotoImage(image=image)




        img = cv2.imread(relative_to_assets("checkbox.png"), cv2.IMREAD_UNCHANGED)
        img = putText(img, text, 29)
        (B, G, R,A) = cv2.split(img)
        img=cv2.merge([R, G,B,A])
        image = Image.fromarray(img)
        self.empty = ImageTk.PhotoImage(image=image)


        self.canvas_btn_img_obj = canvas.create_image(
            x, y, anchor='nw', image=self.checked if self.state else self.empty)
        canvas.tag_bind(self.canvas_btn_img_obj, "<Button-1>",
                        lambda event: self.on_click(command=command))
        self.canvas.grid_configure
    def show(self, value):
        if value:
            self.canvas.itemconfig(self.canvas_btn_img_obj, state='normal')
        else:
            self.canvas.itemconfig(self.canvas_btn_img_obj, state='hidden')
    def toggle(self):
        self.state= not self.state
        image=self.checked if self.state else self.empty
        self.canvas.itemconfig(self.canvas_btn_img_obj,image=image)

    def on_click(self, command):
        self.toggle()
        if command is not None:
            command()

