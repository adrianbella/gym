from PIL import Image
import psutil

class Display:
    @staticmethod
    def show_img(nump):
        Display.img = Image.fromarray(nump)
        Display.img.show()

    @staticmethod
    def close():
        for proc in psutil.process_iter():
            if proc.name() == "display":
                proc.kill()
